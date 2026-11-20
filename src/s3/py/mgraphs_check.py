"""Check and benchmark of the macro server's CUDA graphs (s3_server.py --mgraphs 1, macro_graphs.py), 2026-11-17.

Loads a macro ensemble file (anything macro_s3.Ensemble.load reads) and rows of the current layout (run chunks *.s3r
or directories of them; without any, random valid rows with random layout v3 inputs from layout3_check.py). Decision
rows come from the data; value rows (T_f table rows) are rows without a decision plus decision rows with their options
cleared. If the file has no trained layout v3 parameters (a v1 / v2 file such as macro_s049.pt) its v3 zero layers are
set to small random values (--random-v3 auto), so that the attention / cross-attention paths are not the identity.

  (m) memory: the eager server's activation cache at batches of 64 and 128 rows (padded to MBUCKETS as the server
      does) and the graphs' pool (all buckets of --buckets, both halves), each measured from an emptied cache
  (1) correctness: per bucket size, batches of mixed decision / value rows whose halves land in that bucket (and one
      above the largest bucket: eager fallback); graph outputs against the eager Ensemble.serve, max |difference| per
      output field (expected ~1e-6 from TF32 kernel choice; FAIL above --tol), and against the same static-shape
      forward run eagerly (same shapes and kernels: expected 0). If a field fails with TF32 on, the comparison is
      repeated with TF32 off to tell TF32 kernel choice from a graph error
  (2) timing: ms per serve call at batch sizes --sizes (default 8 / 18 / 32 / 64 rows, --dec-frac decision rows),
      the server's eager path (MBUCKETS padding, Ensemble.serve, .cpu()) against MacroServe with graphs
  (3) weight swaps through MacroServe.reload (the server's reload; "graphs built" = captures on CUDA): perturbed
      weights in place (no new capture, graph outputs equal the eager outputs of the new weights and moved), the
      original weights back (outputs as before), milestone blending switched on (graphs captured again, outputs equal
      eager), a file with fewer members (ensemble rebuilt, graphs captured again, outputs equal eager)
With --device cpu (or no CUDA) graphs cannot be captured: the static-shape forward runs eagerly on the static buffers
(a smoke test of the code path; timings are not meaningful).

GPU memory: --mem-cap-gb caps this process's allocator (default 1.4 GB; the CUDA context adds ~0.4 GB).

Usage: ./.venv/bin/python -I src/s3/py/mgraphs_check.py MACRO.pt [CHUNK.s3r | DIR ...] [--device cuda|cpu]
           [--buckets 8,16,32,64,128] [--sizes 8,18,32,64] [--iters 100] [--dec-frac 0.5] [--mdtype fp32|bf16]
Prints PASS / FAIL lines; exit code 1 if any check failed.
"""
import argparse, glob, os, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as ms  # noqa: E402
import macro_graphs as mg  # noqa: E402

MBUCKETS = [8, 32, 128, 512, 2048, 8192]          # s3_server.py
FIELDS = (("a", ms.O_A, ms.O_A + 1), ("c", ms.O_C, ms.O_C + ms.N_RAMPS), ("v", ms.O_V, ms.O_V + 1),
          ("std", ms.O_STD, ms.O_STD + 1), ("vpol", ms.O_VPOL, ms.O_VPOL + 1), ("opt", ms.O_OPT, ms.O_OPT + ms.K),
          ("slot", ms.O_SLOT, ms.O_SLOT + ms.MAX_DECK), ("ms_heads", ms.O_MS, ms.O_ALPHA), ("ms_consts", ms.O_ALPHA, ms.OW))
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def diffs(x, y):
    """Max |x - y| per output field (inf if a NaN / inf appears in one and not the other)."""
    d = {}
    for name, lo, hi in FIELDS:
        a, b = x[:, lo:hi].astype(np.float64), y[:, lo:hi].astype(np.float64)
        bad = np.isfinite(a) != np.isfinite(b)
        d[name] = float("inf") if bad.any() else float(np.nan_to_num(np.abs(a - b), nan=0.0).max(initial=0.0))
    return d


def merge(acc, d):
    for k, v in d.items():
        acc[k] = max(acc.get(k, 0.0), v)
    return acc


def fmt(d):
    return " ".join(f"{k} {v:.1e}" for k, v in d.items())


# ---------------------------------------------------------------- rows

def chunk_rows(paths, n):
    from train_macro_s3 import read_chunk
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.s3r"), recursive=True), key=os.path.getmtime)[::-1] if os.path.isdir(p) else [p]
    I, F, got = [], [], 0
    for f in files:
        try:
            _, ints, floats, _ = read_chunk(f)
        except Exception as e:
            print(f"skip {f}: {e}", flush=True)
            continue
        I.append(ints.astype(np.int32)); F.append(floats); got += len(ints)
        if got >= n:
            break
    if not I:
        return None
    return np.concatenate(I)[:n], np.concatenate(F)[:n], len(I)


def synthetic_rows(ck, n, rng):
    import layout3_check as lc
    I, F = lc.random_rows(n, ck["vocab"], ck["special"], rng)
    I, F = lc.with_v3(I, F, rng, len(ck["vocab"]["encounter"]))
    return I.astype(np.int32), F.astype(np.float32)


def as_value_rows(I, F):
    """Rows with their decision removed (options cleared, deck-slot decision types set to 0): T_f-like value rows."""
    I, F = I.copy(), F.copy()
    I[:, ms.OFF["opt_kind"][0]:ms.OFF["opt_upg"][1]] = 0
    F[:, 16:ms.FW_V1] = 0; F[:, ms.F_OPT2:ms.F_SCAL2] = 0
    d = ms.OFF["dtype"][0]
    I[np.isin(I[:, d], ms.DT_SLOT), d] = 0
    return I, F


class Pools:
    def __init__(self, I, F):
        dec = ms.decision_rows(I)
        self.D = (I[dec], F[dec])
        VI, VF = as_value_rows(I[dec], F[dec])
        self.V = (np.concatenate([I[~dec], VI]), np.concatenate([F[~dec], VF]))
        assert len(self.D[0]) and not ms.decision_rows(self.V[0]).any()

    def batch(self, nd, nv, rng):
        """nd decision rows and nv value rows in a random order."""
        a = rng.choice(len(self.D[0]), nd, replace=len(self.D[0]) < nd)
        b = rng.choice(len(self.V[0]), nv, replace=len(self.V[0]) < nv)
        I = np.concatenate([self.D[0][a], self.V[0][b]]); F = np.concatenate([self.D[1][a], self.V[1][b]])
        p = rng.permutation(len(I))
        return I[p], F[p]

    def mixed(self, n, frac, rng):
        nd = int(rng.binomial(n, frac))
        return self.batch(nd, n - nd, rng)


# ---------------------------------------------------------------- serving paths

class ServerEager:
    """s3_server.py's macro batch without --mgraphs: rows padded to MBUCKETS, Ensemble.serve, rows back to the host."""

    def __init__(self, dev):
        self.dev = dev
        self.I = np.zeros((MBUCKETS[-1], ms.IW), np.int32); self.F = np.zeros((MBUCKETS[-1], ms.FW), np.float32)

    def __call__(self, ens, ints, floats):
        n = len(ints)
        Bp = next(b for b in MBUCKETS if b >= n)
        self.I[:n] = ints; self.F[:n] = floats
        self.I[n:Bp] = 0; self.F[n:Bp] = 0; self.I[n:Bp, mg.CTX_ROOM] = 1
        return ens.serve(self.I[:Bp], self.F[:Bp], self.dev)[:n].cpu().numpy()


def eager(ens, ints, floats, dev):
    return ens.serve(ints, floats, dev).cpu().numpy()


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def randomize_v3_zero(ens, seed, std=0.05):
    g = torch.Generator().manual_seed(seed)
    for n in ens.nets:
        for k, p in n.named_parameters():
            if ms.is_v3_zero(k):
                p.copy_((torch.randn(p.shape, generator=g) * std).to(p.device))


def save_ck(path, ens, members=None, **kw):
    sds = members if members is not None else [{k: v.detach().cpu().clone() for k, v in n.state_dict().items()} for n in ens.nets]
    ck = {"members": sds, "vocab": ens.vocab, "special": ens.special, "version": ens.version, "milestones": ens.milestones,
          "alpha": ens.alpha, "boss1_share": ens.boss1_share, "layout": 3}
    ck.update(kw)
    torch.save(ck, path)
    return sds


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("macro"); ap.add_argument("chunks", nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--buckets", default="", help="graph buckets (default 8,16,32,64,128 on CUDA, 8,16,32 on CPU)")
    ap.add_argument("--sizes", default="8,18,32,64", help="timing: rows per serve call")
    ap.add_argument("--iters", type=int, default=0, help="timing: calls per size and path (default 100 CUDA, 3 CPU)")
    ap.add_argument("--dec-frac", type=float, default=0.5, help="share of decision rows in a timing batch")
    ap.add_argument("--rows", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-v3", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--mdtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--tol", type=float, default=0.0, help="max |graph - eager| per field (default 1e-4; bf16: 0.25)")
    ap.add_argument("--no-tf32", action="store_true", help="TF32 matmuls off (the server has them on)")
    ap.add_argument("--mem-cap-gb", type=float, default=1.4, help="CUDA: cap of this process's allocator (GB)")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--tmp", default=tempfile.gettempdir())
    ap.add_argument("--skip", default="", help="comma list of steps to skip: m,1,2,3")
    a = ap.parse_args()
    t_start = time.time()
    dev = a.device
    cuda = dev == "cuda"
    skip = set(a.skip.split(",")) - {""}
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if cuda:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gb * 2 ** 30 / tot))
        torch.backends.cuda.matmul.allow_tf32 = not a.no_tf32                       # as s3_server.py
    else:
        torch.set_num_threads(a.threads)
    buckets = [int(x) for x in (a.buckets or ("8,16,32,64,128" if cuda else "8,16,32")).split(",")]
    iters = a.iters or (100 if cuda else 3)
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.mdtype]
    # TF32 kernel choice depends on the batch shape: on the server's v3 file graph vs eager differs by up to 1.5e-2
    # (slot logits) with TF32 on and 2e-5 with it off (window 1, 2026-11-16). With TF32 on, the strict check is
    # the TF32-off rerun in (1) (FAIL above 1e-4) and the static-shape comparison (same kernels, <= 1e-6).
    tf32 = cuda and not a.no_tf32
    tol = a.tol or (0.25 if dt != torch.float32 else 5e-2 if tf32 else 1e-4)

    # ---- ensemble and rows
    ck = torch.load(a.macro, map_location="cpu", weights_only=False)
    has_v3 = any(ms.is_v3(k) for k in ck["members"][0])
    ens = ms.Ensemble.from_ck(ck, dev, a.macro)
    if a.random_v3 == "1" or (a.random_v3 == "auto" and not has_v3):
        randomize_v3_zero(ens, a.seed)
        v3note = "v3 zero layers set random (std 0.05)"
    else:
        v3note = "v3 parameters as in the file" if has_v3 else "v3 zero layers at zero (identity paths)"
    got = chunk_rows(a.chunks, a.rows) if a.chunks else None
    if got is not None:
        I, F, nfiles = got
        src = f"{len(I)} rows from {nfiles} chunk file(s)"
    else:
        I, F = synthetic_rows(ck, min(a.rows, 1024), rng)
        src = f"{len(I)} synthetic rows (layout3_check.random_rows + with_v3)"
    del ck
    P = Pools(I, F)
    print(f"{a.macro}: {len(ens.nets)} members, version {ens.version}, milestones {ens.milestones}; {v3note}; {src}: "
          f"{len(P.D[0])} decision rows, {len(P.V[0])} value rows; device {dev}, TF32 {cuda and not a.no_tf32}, "
          f"dtype {a.mdtype}, buckets {buckets}, tol {tol:g}", flush=True)
    if not cuda:
        print("(CPU: no graph capture; the static-shape forward runs eagerly on the static buffers)", flush=True)
    se = ServerEager(dev)
    ms_graph = mg.MacroServe(ens, dev, graphs=True, dtype=dt, buckets=buckets)
    ms_static = mg.MacroServe(ens, dev, graphs=True, dtype=dt, buckets=buckets, capture=False) if cuda else None

    # ---- (m) memory
    if cuda and "m" not in skip:
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_reserved()
        params = sum(p.numel() * p.element_size() for n in ens.nets for p in n.parameters())
        torch.cuda.reset_peak_memory_stats()
        for n in (64, 128):
            se(ens, *P.mixed(n, a.dec_frac, rng))
        e_peak = torch.cuda.max_memory_reserved() - base
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_reserved()
        for b in buckets:                                  # every graph: both halves at every bucket
            ms_graph.serve(*P.batch(b, b, rng))
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        g_mem = torch.cuda.memory_reserved() - base
        print(f"(m) parameters {params / 2**20:.0f} MB; eager server activation cache at 64 / 128-row batches "
              f"{e_peak / 2**20:.0f} MB; graph pool + static buffers for {len(ms_graph.graphs)} graphs (buckets up to "
              f"{buckets[-1]}) {g_mem / 2**20:.0f} MB ({ms_graph.captures} captures)", flush=True)

    # ---- (1) correctness
    if "1" not in skip:
        sets = []
        prev = 0
        for b in buckets:
            sets += [(int(rng.integers(prev + 1, b + 1)), int(rng.integers(prev + 1, b + 1))), (b, 0), (0, prev + 1)]
            prev = b
        sets.append((buckets[-1] + 2, 3))                  # decision half above the largest bucket: eager
        batches = [P.batch(nd, nv, rng) for nd, nv in sets]

        def compare(label):
            dg, ds, bad = {}, {}, 0
            for Ib, Fb in batches:
                ref = eager(ens, Ib, Fb, dev)
                out = ms_graph.serve(Ib, Fb)
                bad += int(not np.isfinite(out[:, :ms.O_MS]).all())
                merge(dg, diffs(out, ref))
                if ms_static is not None:
                    merge(ds, diffs(out, ms_static.serve(Ib, Fb)))
            return dg, ds, bad
        c0 = ms_graph.captures
        dg, ds, bad = compare("tf32")
        print(f"(1) {len(batches)} batches (decision / value rows: {sets}); captures {ms_graph.captures - c0} new, "
              f"{ms_graph.captures} total; eager rows {ms_graph.eager_rows}", flush=True)
        if cuda:
            check("(1) graphs captured (no capture failure, no fallback to eager)",
                  ms_graph.use_graphs and ms_graph.captures > 0, f"{ms_graph.captures} captures")
        check("(1) no NaN / inf in the graph outputs", bad == 0)
        worst = max(dg.values())
        check(f"(1) graph vs eager Ensemble.serve, max |diff| per field <= {tol:g}", worst <= tol, fmt(dg))
        if ms_static is not None:
            check("(1) graph vs the same static-shape forward run eagerly (same kernels), max |diff| <= 1e-6",
                  max(ds.values()) <= 1e-6, fmt(ds))
        if tf32 and dt == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = False
            ms_graph.reset(); ms_static.reset()
            dg2, _, _ = compare("fp32")
            check("(1) the same batches with TF32 off (graphs captured again), graph vs eager <= 1e-4",
                  max(dg2.values()) <= 1e-4, fmt(dg2))
            torch.backends.cuda.matmul.allow_tf32 = True
            ms_graph.reset(); ms_static.reset()
        if dt != torch.float32:
            ref32 = {}
            for Ib, Fb in batches[:6]:
                merge(ref32, diffs(ms_graph.serve(Ib, Fb), eager(ens, Ib, Fb, dev)))
            print(f"(1) bf16 cost: graph (bf16) vs fp32 eager {fmt(ref32)}", flush=True)

    # ---- (2) timing
    if "2" not in skip:
        print(f"(2) ms per serve call, {iters} calls per size, {a.dec_frac:.0%} decision rows "
              f"({'not meaningful on CPU' if not cuda else 'CUDA'}):", flush=True)
        for n in [int(x) for x in a.sizes.split(",")]:
            bs = [P.mixed(n, a.dec_frac, rng) for _ in range(iters)]
            res = {}
            for name, fn in (("eager", lambda Ib, Fb: se(ens, Ib, Fb)), ("graphs", ms_graph.serve)):
                for Ib, Fb in bs[:min(10, iters)]:         # warm-up (and captures)
                    fn(Ib, Fb)
                sync(dev); t = time.perf_counter()
                for Ib, Fb in bs:
                    fn(Ib, Fb)
                sync(dev)
                res[name] = 1e3 * (time.perf_counter() - t) / iters
            print(f"    {n:4d} rows: eager {res['eager']:7.2f} ms, graphs {res['graphs']:7.2f} ms "
                  f"(x{res['eager'] / max(res['graphs'], 1e-9):.1f})", flush=True)
        print(f"    captures so far {ms_graph.captures}, eager rows {ms_graph.eager_rows}", flush=True)

    # ---- (3) weight swaps
    if "3" not in skip:
        tmp = tempfile.mkdtemp(prefix="mgraphs_", dir=a.tmp)
        f0, f1, f2, f3 = (os.path.join(tmp, f"{x}.pt") for x in ("orig", "perturbed", "milestones", "fewer"))
        try:
            S = [P.mixed(n, 0.5, rng) for n in (8, 18, 40, 64)]
            run = lambda: [ms_graph.serve(Ib, Fb).copy() for Ib, Fb in S]
            ref = lambda: [eager(ms_graph.ens, Ib, Fb, dev) for Ib, Fb in S]
            Y0 = run(); c0 = ms_graph.builds
            sd0 = save_ck(f0, ens)
            g = torch.Generator().manual_seed(a.seed + 1)
            sd1 = [{k: (v + 0.05 * torch.randn(v.shape, generator=g) * (v.abs().mean() + 1e-3)) if v.is_floating_point() else v
                    for k, v in sd.items()} for sd in sd0]
            save_ck(f1, ens, sd1, version=ens.version + 1)
            rebuilt = ms_graph.reload(f1)
            Y1, E1 = run(), ref()
            d1 = {}
            for y, e in zip(Y1, E1):
                merge(d1, diffs(y, e))
            moved = max(float(np.abs(y[:, :ms.O_MS] - y0[:, :ms.O_MS]).max()) for y, y0 in zip(Y1, Y0))
            check("(3) perturbed weights load in place (no rebuild, no new capture)",
                  not rebuilt and ms_graph.builds == c0 and ms_graph.ens is ens, f"graphs built {c0} -> {ms_graph.builds}")
            check(f"(3) after the swap the graph outputs equal eager with the new weights (<= {tol:g})", max(d1.values()) <= tol, fmt(d1))
            check("(3) ... and moved with the weights (max |change| > 1e-3)", moved > 1e-3, f"{moved:.3g}")
            ms_graph.reload(f0)
            Y0b = run()
            back = max(float(np.abs(y - y0).max()) for y, y0 in zip(Y0b, Y0))
            check("(3) original weights back: graph outputs as before the swap (<= 1e-6)", back <= 1e-6, f"{back:.1e}")
            c1 = ms_graph.builds
            sd2 = [dict(sd) for sd in sd0]                 # milestone heads still at their zero start: made random
            for sd in sd2:
                for k in ("ms_win.weight", "ms_mono.weight"):
                    if k in sd and not sd[k].abs().sum():
                        sd[k] = torch.randn(sd[k].shape, generator=g) * 0.05
            save_ck(f2, ens, sd2, milestones=True, alpha=0.3, boss1_share=0.4)
            rebuilt = ms_graph.reload(f2)
            Y2, E2 = run(), ref()
            d2 = {}
            for y, e in zip(Y2, E2):
                merge(d2, diffs(y, e))
            check("(3) milestone blending on (alpha 0.3, boss-1 share 0.4): graphs captured again, outputs equal eager",
                  not rebuilt and ms_graph.builds > c1 and max(d2.values()) <= tol and all((y[:, ms.O_HASMS] == 1).all() for y in Y2),
                  f"graphs built {c1} -> {ms_graph.builds}; {fmt(d2)}")
            c2 = ms_graph.builds
            k = max(1, len(ens.nets) - 2)
            save_ck(f3, ens, sd0[:k], milestones=False, alpha=0.0, boss1_share=0.0)
            rebuilt = ms_graph.reload(f3)
            Y3, E3 = run(), ref()
            d3 = {}
            for y, e in zip(Y3, E3):
                merge(d3, diffs(y, e))
            check(f"(3) a file of {k} members: ensemble rebuilt, graphs captured again, outputs equal eager",
                  rebuilt and len(ms_graph.ens.nets) == k and ms_graph.builds > c2 and max(d3.values()) <= tol,
                  f"graphs built {c2} -> {ms_graph.builds}, rebuilds {ms_graph.rebuilds}; {fmt(d3)}")
        finally:
            for f in (f0, f1, f2, f3):
                if os.path.exists(f):
                    os.remove(f)
            os.rmdir(tmp)

    if cuda:
        print(f"GPU memory of this process: peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MB, "
              f"reserved now {torch.cuda.memory_reserved() / 2**20:.0f} MB (plus the CUDA context)", flush=True)
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL'} ({time.time() - t_start:.0f} s); {ms_graph.stats()}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
