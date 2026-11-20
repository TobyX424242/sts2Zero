"""Macro inference on CPU vs GPU (2026-11-14): time macro_s3.Ensemble.serve on real rows (a run chunk *.s3r) in batches
like the live macro server's (~30 rows), with T CPU threads, and compare the outputs with the same batches on CUDA.
Layout v3 (2026-11-16): --compare-v2 also times the frozen layout v2 network (macro_s3_v2.py) on the same rows (its
widths), --batch takes a list (e.g. 64,256), --no-cuda skips the GPU, and without a chunk the rows are random valid rows
with random layout v3 inputs (layout3_check.random_rows / with_v3).
usage: ./.venv/bin/python -I src/s3/py/macro_cpu_bench.py MACRO.pt [CHUNK.s3r] [--threads 1,2,4] [--batch 32]
           [--iters 300] [--compare-v2] [--no-cuda]
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as ms  # noqa: E402
from train_macro_s3 import read_chunk  # noqa: E402


def rows(a):
    if a.chunk:
        _, ints, floats, _ = read_chunk(a.chunk)
        return ints, floats
    import layout3_check as lc
    ck = torch.load(a.macro, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(0)
    I, Fl = lc.random_rows(2048, ck["vocab"], ck["special"], rng)
    return lc.with_v3(I, Fl, rng, len(ck["vocab"]["encounter"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("macro"); ap.add_argument("chunk", nargs="?", default="")
    ap.add_argument("--threads", default="1,2,4"); ap.add_argument("--batch", default="32")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--compare-v2", action="store_true"); ap.add_argument("--no-cuda", action="store_true")
    a = ap.parse_args()
    ints, floats = rows(a)
    dec = (ints[:, ms.OFF["opt_kind"][0]] != 0)
    print(f"{len(ints)} rows, {dec.mean():.0%} decision rows; {a.iters} iterations per batch size", flush=True)
    mods = [("v3", ms, ms.IW, ms.FW)]
    if a.compare_v2:
        import macro_s3_v2 as v2m
        mods.append(("v2", v2m, ms.IW_V2, ms.FW_V2))
    for bs in [int(x) for x in a.batch.split(",")]:
        rng = np.random.default_rng(0)
        batches = [rng.choice(len(ints), bs, replace=len(ints) < bs) for _ in range(a.iters)]
        for name, mod, iw, fw in mods:
            ref = None
            if torch.cuda.is_available() and not a.no_cuda:
                torch.backends.cuda.matmul.allow_tf32 = True     # as s3_server.py
                g = mod.Ensemble.load(a.macro, "cuda")
                with torch.inference_mode():
                    for b in batches[:20]:
                        g.serve(ints[b, :iw], floats[b, :fw], "cuda")
                    torch.cuda.synchronize(); t = time.perf_counter()
                    ref = [g.serve(ints[b, :iw], floats[b, :fw], "cuda").cpu().numpy() for b in batches]
                    dt = time.perf_counter() - t
                print(f"{name} batch {bs} cuda      : {1e3 * dt / a.iters:7.2f} ms/batch = {a.iters * bs / dt:7.0f} rows/s", flush=True)
                del g
            for th in [int(x) for x in a.threads.split(",")]:
                torch.set_num_threads(th)
                c = mod.Ensemble.load(a.macro, "cpu")
                with torch.inference_mode():
                    for b in batches[:10]:
                        c.serve(ints[b, :iw], floats[b, :fw], "cpu")
                    t = time.perf_counter()
                    out = [c.serve(ints[b, :iw], floats[b, :fw], "cpu").numpy() for b in batches]
                    dt = time.perf_counter() - t
                diff = max(float(np.abs(o - r).max()) for o, r in zip(out, ref)) if ref is not None else float("nan")
                print(f"{name} batch {bs} cpu {th:2d} thr: {1e3 * dt / a.iters:7.2f} ms/batch = {a.iters * bs / dt:7.0f} rows/s"
                      + (f"; max |cpu - cuda| {diff:.2e}" if ref is not None else ""), flush=True)


if __name__ == "__main__":
    main()
