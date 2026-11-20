"""Checks of the macro row layout v2 (2026-11-16; macro_s3.py header). CPU, seconds, no training run.

A macro file (an ensemble file such as macro_s049.pt, or an S1 model.pt) is loaded into the frozen pre-change network
(macro_s3_v1.py) and into the current one (macro_s3.py). Rows: random valid rows of every decision type, plus rows of run
chunks (*.s3r of either layout) when given. The layout v2 inputs are the two boss ids, the 8 extra state scalars and
the 8 extra features per option; the layout v2 parameters are macro_s3.V2_PARAMS.

  (0) the file loads; a file without layout v2 parameters leaves them at exactly zero
  (1) v2 inputs all zero: every output of the current network equals the frozen network's to 1e-5 (per member: value a,
      c, logit, option and slot logits, milestone heads; and the served rows); v1-width rows are accepted and padded
  (2) v2 inputs random: the outputs do not move (the parameters are zero); a hot reload of the file into an ensemble
      that holds other v2 parameters gives the same function again
  (3) v2 parameters random (copies): each input group moves the outputs it is wired to, in every member (boss, second
      boss, scalars: the value and the policy; option features: the logits of map options and nothing else), and rows
      whose v2 inputs are zero still equal the frozen network (old rows stay neutral whatever is learned)
  (4) the value is monotone in HP: over an HP sweep the value logit and the milestone logits never fall (float32
      noise up to 1e-4 allowed) and (a, c) do not depend on HP, with the file's weights and with random v2 parameters
  (5) the trainer reads chunks of both layouts into one window (old rows zero-padded)
  (6) a carry file of layout v1 (rows and Adam states of the old network) loads into the trainer: rows padded,
      optimizers restored for the old parameters, an Adam step runs, gradients reach every v2 parameter group

Usage: ./.venv/bin/python -I src/s3/py/layout2_check.py MACRO.pt [CHUNK.s3r | CHUNK_DIR ...] [--rows 512]
Prints PASS / FAIL lines; exit code 1 if any check failed.
"""
import argparse, copy, glob, os, shutil, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as ms  # noqa: E402
import macro_s3_v1 as v1  # noqa: E402
import train_macro_s3 as tr  # noqa: E402

FAILS = []
S = ms.S
O = ms.OFF


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def is_v2(name):
    return name.split(".")[0] in ms.V2_PARAMS


def v2_params(net):
    return [(k, p) for k, p in net.named_parameters() if is_v2(k)]


def load_both(path):
    """The file in the frozen network (v2 parameters, if the file has any, left out) and in the current one, through
    the loader the server and the trainer use."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ensemble = "members" in ck
    sds = ck["members"] if ensemble else [ck["state_dict"]]
    vocab, special = ck["vocab"], ck["special"]
    rest = [i for i, s in enumerate(special) if s.startswith("REST:")]
    old, has_v2 = [], False
    for sd in sds:
        has_v2 |= any(is_v2(k) for k in sd)
        n = v1.S3MacroNet(vocab, len(special), rest)
        n.load_s1_or_s3({k: v for k, v in sd.items() if not is_v2(k)})
        old.append(n.eval())
    e1 = v1.Ensemble(old, vocab, special, int(ck.get("version", 1)))
    e1.milestones, e1.alpha, e1.boss1_share = bool(ck.get("milestones", False)), float(ck.get("alpha", 0.0)), float(ck.get("boss1_share", 0.0))
    if ensemble:
        e2 = ms.Ensemble.load(path, "cpu")
    else:
        net, _ = ms.load_checkpoint(path, "cpu")
        e2 = ms.Ensemble([net.eval()], vocab, special)
    return ck, e1, e2, has_v2, ensemble


def random_rows(n, vocab, special, rng):
    """Valid rows of the current widths with the layout v2 inputs at zero: a state, and by turns no decision, a card
    reward, a map choice, an event, a shop, a deck-slot decision, a rest site."""
    I = np.zeros((n, ms.IW), np.int64); Fl = np.zeros((n, ms.FW), np.float32)
    rest_ids = [i for i, s in enumerate(special) if s.startswith("REST:")] or [1, 2]

    def ids(kind, size=None):                             # random ids of a vocabulary (1 = unknown, a valid index)
        return rng.integers(1, max(2, len(vocab[kind])), size)

    def put(r, key, vals):
        I[r, O[key][0]:O[key][0] + len(vals)] = vals

    def opt(r, j, kind, oid, upg=0):
        I[r, O["opt_kind"][0] + j] = kind; I[r, O["opt_id"][0] + j] = oid; I[r, O["opt_upg"][0] + j] = upg

    for r in range(n):
        nd, nr, npot = int(rng.integers(8, 45)), int(rng.integers(1, 20)), int(rng.integers(0, 4))
        ench = rng.random(nd) < 0.1
        put(r, "d_ids", ids("card", nd)); put(r, "d_upg", rng.integers(0, 2, nd))
        put(r, "d_ench", np.where(ench, ids("ench", nd), 0)); put(r, "d_eamt", np.where(ench, rng.integers(1, 6, nd), 0))
        put(r, "d_age", rng.integers(0, 40, nd))
        put(r, "r_ids", ids("relic", nr)); put(r, "r_age", rng.integers(0, 40, nr))
        put(r, "p_ids", ids("potion", npot))
        act, asc, room = int(rng.integers(0, 3)), int(rng.integers(0, 11)), int(rng.integers(2, 10))
        put(r, "ctx", [2, asc, ms.ERA_V0111, act, room, int(ids("encounter"))])
        max_hp = int(rng.integers(40, 100)); hp = int(rng.integers(1, max_hp + 1)); in_act = int(rng.integers(1, 17))
        for k, x in (("floor", (act * 17 + in_act) / 50), ("act", act / 3), ("hp", hp / 100), ("max_hp", max_hp / 100), ("hp_frac", hp / max_hp),
                     ("gold", rng.integers(0, 500) / 500), ("deck_size", nd / 40), ("n_upgraded", I[r, O["d_upg"][0]:O["d_upg"][1]].sum() / 20),
                     ("n_enchanted", ench.sum() / 10), ("n_relics", nr / 20), ("n_potions", npot / 5),
                     ("is_combat", float(room in (3, 4, 5))), ("floor_in_act", in_act / 20)):
            Fl[r, S[k]] = x
        kind = r % 7
        if kind == 1:                                     # card reward: three cards and SKIP
            for j in range(3):
                opt(r, j, 1, int(ids("card")), int(rng.integers(0, 2)))
            opt(r, 3, 4, 1); I[r, O["dtype"][0]] = 0
        elif kind == 2:                                   # map: 2-4 points with the 12 path features
            k = int(rng.integers(2, 5))
            for j in range(k):
                opt(r, j, 5, int(rng.integers(1, 9)))
                Fl[r, 16 + j * ms.MF:16 + (j + 1) * ms.MF] = rng.random(ms.MF)
            I[r, O["dtype"][0]] = 7
        elif kind == 3:                                   # event options
            for j in range(int(rng.integers(2, 4))):
                opt(r, j, 6, int(rng.integers(0, ms.N_EVENT_BUCKETS)))
            I[r, O["dtype"][0]] = 8
        elif kind == 4:                                   # shop: card, relic, potion, REMOVE, LEAVE
            opt(r, 0, 1, int(ids("card"))); opt(r, 1, 2, int(ids("relic")))
            opt(r, 2, 3, int(ids("potion"))); opt(r, 3, 4, 2); opt(r, 4, 7, 0); I[r, O["dtype"][0]] = 6
        elif kind == 5:                                   # deck-slot decision (smith / removal target)
            I[r, O["dtype"][0]] = int(rng.choice(ms.DT_SLOT))
        elif kind == 6:                                   # rest site
            for j, sid in enumerate(rest_ids[:4]):
                opt(r, j, 4, sid)
            I[r, O["dtype"][0]] = 3
    return I, Fl


def chunk_rows(paths, n, rng):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.s3r")))[:4] if os.path.isdir(p) else [p]
    ints, floats = [], []
    for f in files:
        _, i, fl, _ = tr.read_chunk(f)                    # either layout; v1 rows come back padded
        ints.append(i.astype(np.int64)); floats.append(fl)
    if not ints:
        return np.zeros((0, ms.IW), np.int64), np.zeros((0, ms.FW), np.float32), 0
    ints, floats = np.concatenate(ints), np.concatenate(floats)
    k = np.sort(rng.choice(len(ints), min(n, len(ints)), replace=False))
    return ints[k], floats[k], len(files)


def with_v2(I, Fl, rng, n_enc, groups=("boss", "boss2", "scal2", "opt_feat2")):
    """A copy of the rows with the named layout v2 input groups random and the other groups zero."""
    I, Fl = I.copy(), Fl.copy()
    I[:, O["boss"][0]:O["boss"][1]] = 0; Fl[:, ms.F_OPT2:ms.FW] = 0
    if "boss" in groups:
        I[:, O["boss"][0]] = rng.integers(1, max(2, n_enc), len(I))
    if "boss2" in groups:
        I[:, O["boss"][0] + 1] = rng.integers(1, max(2, n_enc), len(I))
    if "scal2" in groups:
        Fl[:, ms.F_SCAL2:ms.FW] = rng.random((len(I), ms.NS2))
    if "opt_feat2" in groups:
        Fl[:, ms.F_OPT2:ms.F_SCAL2] = rng.random((len(I), ms.K * ms.MF2))
    return I, Fl


@torch.no_grad()
def outputs(ens, mod, I, Fl):
    """Everything the networks compute for the rows: per member the value pieces, the policy logits and the milestone
    heads; and the served rows, as the file says and with the milestone blend switched on."""
    b = mod.rows_to_batch(I, Fl, "cpu")
    out = {}
    for j, n in enumerate(ens.nets):
        a, c, v, opt, slot = n.forward_rows(b, policy=True)
        ma, mc, ml = n.forward_ms(b)
        out[j] = {"a": a, "c": c, "v": v, "opt": opt, "slot": slot, "ms_a": ma, "ms_c": mc, "ms_logit": ml}
    out["serve"] = {"rows": ens.serve(I, Fl, "cpu")}
    keep = ens.milestones, ens.alpha, ens.boss1_share
    ens.milestones, ens.alpha, ens.boss1_share = True, 0.3, 0.5
    out["serve_ms"] = {"rows": ens.serve(I, Fl, "cpu")}
    ens.milestones, ens.alpha, ens.boss1_share = keep
    return out


def maxdiff(x, y, keys=None):
    """Largest absolute difference, and where (member or 'serve', output name)."""
    worst, where = 0.0, ""
    for j in x:
        for k in x[j]:
            if keys is not None and k not in keys:
                continue
            d = float((x[j][k].double() - y[j][k].double()).abs().max()) if x[j][k].numel() else 0.0
            if not d <= worst:                            # also catches NaN
                worst, where = d, f"{j}.{k}"
    return worst, where


def randomize_v2(ens, seed, std=0.5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n in ens.nets:
            for k, p in v2_params(n):
                p.copy_(torch.randn(p.shape, generator=g) * std)
                if k.startswith("boss"):
                    p[0].zero_()                          # index 0 (no boss) is the padding row: never trained
    return ens


@torch.no_grad()
def hp_monotone(ens, I, Fl, rows):
    """HP sweep 0..max HP of the given rows: the largest fall of the value logit and of a milestone logit between
    neighbouring HP values, and the largest spread of a and c over a sweep, over all members."""
    ii, ff, spans = [], [], []
    for r in rows:
        mx = int(round(float(Fl[r, S["max_hp"]]) * 100))
        if mx < 2:
            continue
        hs = np.arange(0, mx + 1)
        f = np.repeat(Fl[r][None], len(hs), 0); f[:, S["hp"]] = hs / 100.0; f[:, S["hp_frac"]] = hs / mx
        ii.append(np.repeat(I[r][None], len(hs), 0)); ff.append(f); spans.append(len(hs))
    b = ms.rows_to_batch(np.concatenate(ii), np.concatenate(ff), "cpu")
    fall = fall_ms = spread = 0.0; c_min = float("inf")
    for n in ens.nets:
        a, c, v, _, _ = n.forward_rows(b, policy=False)
        ma, mc, ml = n.forward_ms(b)
        c_min = min(c_min, float(c.min()), float(mc.min()))
        o = 0
        for L in spans:
            s = slice(o, o + L); o += L
            fall = max(fall, float((v[s][:-1] - v[s][1:]).max()))
            fall_ms = max(fall_ms, float((ml[s][:-1] - ml[s][1:]).max()))
            for x in (a[s], c[s], ma[s], mc[s]):
                spread = max(spread, float((x.max(0).values - x.min(0).values).max()))
    return fall, fall_ms, spread, c_min, len(spans), int(sum(spans))


def write_chunk(path, I, Fl, n_runs, iw, fw, rng):
    """A run chunk as RunChunkWriter (src/s3/S3/Records.cs) writes it, with rows iw / fw wide."""
    N = len(I)
    cut = np.linspace(0, N, n_runs + 1).astype(int)
    runs = np.zeros((n_runs, tr.RM), np.int32); meta = np.zeros((N, tr.RW), np.float32)
    for r in range(n_runs):
        a, e = int(cut[r]), int(cut[r + 1])
        act = int(rng.integers(0, 3)); won = int(rng.random() < 0.3)
        runs[r] = [r + 1, 7, int(rng.integers(0, 11)), won, 17 * act + 5, act, 0 if not won else 30, 70, 0, a, e - a, 10, 9, 2, 1, 1]
        meta[a:e, tr.M_RUN] = r; meta[a:e, tr.M_FLOOR] = np.arange(e - a) // 3 + 1; meta[a:e, tr.M_ACT] = act
        dec = (I[a:e, O["opt_kind"][0]] != 0) | np.isin(I[a:e, O["dtype"][0]], ms.DT_SLOT)
        meta[a:e, tr.M_KIND] = dec; meta[a:e, tr.M_DTYPE] = np.where(dec, I[a:e, O["dtype"][0]], -1)
        meta[a:e, tr.M_NOPT] = (I[a:e, O["opt_kind"][0]:O["opt_kind"][1]] != 0).sum(1)
    with open(path, "wb") as f:
        f.write(b"S3RUNS01"); f.write(np.array([1, n_runs, N, iw, fw, tr.RM, tr.RW], "<i4").tobytes()); f.write(bytes(64 - 8 - 7 * 4))
        f.write(runs.astype("<i4").tobytes()); f.write(I[:, :iw].astype("<i2").tobytes()); f.write(Fl[:, :fw].astype("<f4").tobytes())
        f.write(meta.astype("<f4").tobytes())


def train_loss(net, b):
    """A loss that touches the value head, the policy heads and the milestone heads (for the optimizer checks)."""
    _, _, v, opt, slot = net.forward_rows(b, policy=True)
    return v.mean() + opt.clamp(min=-50).mean() + slot.mean() + net.forward_ms(b)[2].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("macro", help="macro ensemble file (or an S1 model.pt)")
    ap.add_argument("chunks", nargs="*", help="run chunks (*.s3r of either layout) or directories of them: rows added to the random ones")
    ap.add_argument("--rows", type=int, default=512, help="rows sampled from the chunks")
    ap.add_argument("--random-rows", type=int, default=210)
    ap.add_argument("--hp-rows", type=int, default=48, help="rows of the HP sweep")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tmp", default=os.path.join(HERE, "..", "out"), help="where the temporary chunk / carry files go")
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"layout v2: ints {ms.IW} (v1 {ms.IW_V1}; boss ids at {O['boss'][0]}, {O['boss'][0] + 1}), floats {ms.FW} (v1 {ms.FW_V1}; "
          f"option features from {ms.F_OPT2}: {ms.K} x {ms.MF2}, scalars from {ms.F_SCAL2}: {ms.NS2}), outputs {ms.OW}; parameters {ms.V2_PARAMS}")
    assert (v1.IW, v1.FW, v1.OW) == (ms.IW_V1, ms.FW_V1, ms.OW), "the frozen module is not the v1 layout"

    # ---- (0) loading
    ck, e1, e2, has_v2, is_ens = load_both(a.macro)
    zero = all(bool((p == 0).all()) for n in e2.nets for _, p in v2_params(n))
    names = sorted({k for n in e2.nets for k, _ in v2_params(n)})
    n_new = sum(p.numel() for _, p in v2_params(e2.nets[0]))
    print(f"file {a.macro}: {len(e2.nets)} member(s), version {e2.version}, milestones {e2.milestones}, "
          f"{'has' if has_v2 else 'no'} layout v2 parameters; new parameters per member: {names} ({n_new:,} values)")
    check("(0) every layout v2 parameter exists in every member", names == sorted(k + ".weight" for k in ms.V2_PARAMS), str(names))
    order = [k for k, _ in e2.nets[0].named_parameters()]
    check("(0) the layout v2 parameters are the last ones (optimizer states of older carry files stay aligned)",
          all(is_v2(k) for k in order[-len(names):]) and not any(is_v2(k) for k in order[:-len(names)]))
    if not has_v2:
        check("(0) loaded from a file without them, the layout v2 parameters are exactly zero", zero)
    else:
        print(f"note: the file has layout v2 parameters ({'all zero' if zero else 'trained'}); checks (2) use a copy with them at zero")

    # ---- rows
    n_enc = len(ck["vocab"]["encounter"])
    RI, RF = random_rows(a.random_rows, ck["vocab"], ck["special"], rng)
    CI, CF, n_files = chunk_rows(a.chunks, a.rows, rng)
    I = np.concatenate([RI, CI]); Fl = np.concatenate([RF, CF])
    I0, F0 = with_v2(I, Fl, rng, n_enc, groups=())            # every v2 input zero
    Ir, Fr = with_v2(I, Fl, rng, n_enc)                       # every v2 input random
    is_map = I[:, O["opt_kind"][0]:O["opt_kind"][1]] == 5
    print(f"rows: {len(RI)} random + {len(CI)} from {n_files} chunk file(s); {int(is_map.any(1).sum())} map decisions, "
          f"{int(((I[:, O['opt_kind'][0]] == 0) & ~np.isin(I[:, O['dtype'][0]], ms.DT_SLOT)).sum())} rows without a decision")

    # ---- (1) function preserved
    ref = outputs(e1, v1, I0[:, :ms.IW_V1], F0[:, :ms.FW_V1])
    new0 = outputs(e2, ms, I0, F0)
    if has_v2 and not zero:                                   # the reference function of such a file is the file itself
        print("note: (1) compares a file with trained layout v2 parameters against the frozen network on rows with zero v2 inputs")
    d, w = maxdiff(new0, ref)
    check("(1) v2 inputs zero: outputs equal the frozen v1 network", d < 1e-5, f"max |difference| {d:.2e}" + (f" at {w}" if d > 0 else ""))
    d, w = maxdiff(outputs(e2, ms, I0[:, :ms.IW_V1], F0[:, :ms.FW_V1]), new0)
    check("(1) v1-width rows are accepted and padded with zeros", d <= 1e-7, f"max |difference| {d:.2e}")
    pi, pf = ms.pad_rows(I0[:, :ms.IW_V1].astype(np.int16), F0[:, :ms.FW_V1])
    check("(1) pad_rows: widths, dtypes, old columns kept, new columns zero",
          pi.shape == (len(I0), ms.IW) and pf.shape == (len(I0), ms.FW) and pi.dtype == np.int16 and pf.dtype == np.float32
          and np.array_equal(pi[:, :ms.IW_V1], I0[:, :ms.IW_V1]) and np.array_equal(pf[:, :ms.FW_V1], F0[:, :ms.FW_V1])
          and not pi[:, ms.IW_V1:].any() and not pf[:, ms.FW_V1:].any())

    # ---- (2) zero parameters: the new inputs do nothing
    ez = e2
    if not zero:
        ez = copy.deepcopy(e2)
        with torch.no_grad():
            for n in ez.nets:
                for _, p in v2_params(n):
                    p.zero_()
    zr = outputs(ez, ms, Ir, Fr)
    d, w = maxdiff(zr, outputs(ez, ms, I0, F0) if ez is not e2 else new0)
    check("(2) v2 parameters zero, v2 inputs random: outputs unchanged", d < 1e-6, f"max |difference| {d:.2e}" + (f" at {w}" if d > 0 else ""))
    d, w = maxdiff(zr, ref)
    check("(2) ... and equal to the frozen v1 network", d < 1e-5, f"max |difference| {d:.2e}" + (f" at {w}" if d > 0 else ""))
    e3 = randomize_v2(copy.deepcopy(e2), a.seed + 1)
    if is_ens:
        eh = copy.deepcopy(e3)
        eh.load_members_from(a.macro)                         # the server's hot reload
        d, w = maxdiff(outputs(eh, ms, Ir, Fr), outputs(e2, ms, Ir, Fr))
        check("(2) hot reload of the file over other v2 parameters gives the file's function", d < 1e-6, f"max |difference| {d:.2e}")

    # ---- (3) wired in: with random parameters every input group moves what it should
    r0 = outputs(e3, ms, I0, F0)
    d, w = maxdiff(r0, ref)
    check("(3) v2 parameters random, v2 inputs zero: still the frozen v1 network (old rows stay neutral)", d < 1e-5,
          f"max |difference| {d:.2e}" + (f" at {w}" if d > 0 else ""))
    members = list(range(len(e3.nets)))
    for grp in ("boss", "boss2", "scal2"):
        rg = outputs(e3, ms, *with_v2(I, Fl, rng, n_enc, groups=(grp,)))
        da = [float((rg[j]["a"] - r0[j]["a"]).abs().max()) for j in members]
        dm = [float((rg[j]["ms_logit"] - r0[j]["ms_logit"]).abs().max()) for j in members]
        dp = float((rg[0]["opt"] - r0[0]["opt"]).abs().max()); ds = float((rg[0]["slot"] - r0[0]["slot"]).abs().max())
        dsv = float((rg["serve"]["rows"] - r0["serve"]["rows"]).abs().max())
        # the milestone heads are listed only: in a file that never trained them they do not read the state at all
        check(f"(3) {grp} moves the value of every member, the policy and the served rows",
              min(da) > 1e-4 and dp > 1e-4 and ds > 1e-4 and dsv > 1e-4,
              f"max |change| of a per member {[round(x, 4) for x in da]}, option logits {dp:.4f}, slot logits {ds:.4f}, "
              f"milestone logits per member {[round(x, 4) for x in dm]}")
    rg = outputs(e3, ms, *with_v2(I, Fl, rng, n_enc, groups=("opt_feat2",)))
    mk = torch.from_numpy(is_map)
    d_map = min(float((rg[j]["opt"] - r0[j]["opt"]).abs()[mk].max()) for j in members)
    share = float(((rg[0]["opt"] - r0[0]["opt"]).abs()[mk] > 1e-6).float().mean())
    d_other = max(float((rg[j]["opt"] - r0[j]["opt"]).abs()[~mk].max()) for j in members)
    d_rest, w = maxdiff(rg, r0, keys=("a", "c", "v", "slot", "ms_a", "ms_c", "ms_logit"))
    check("(3) opt_feat2 moves the logits of map options", d_map > 1e-4, f"max |change| {d_map:.4f}; {100 * share:.0f}% of map options moved")
    check("(3) opt_feat2 moves nothing else (other options, values, slot logits, milestone heads)", d_other < 1e-6 and d_rest < 1e-6,
          f"other options {d_other:.2e}, the rest {d_rest:.2e}")

    # ---- (4) monotone in HP
    pick = rng.choice(len(I), min(a.hp_rows, len(I)), replace=False)
    for name, ens in (("the file's weights", e2), ("random v2 parameters", e3)):
        fall, fall_ms, spread, c_min, n_rows, n_pts = hp_monotone(ens, Ir, Fr, pick)
        check(f"(4) value monotone in HP, {name} (v2 inputs random)", fall <= 1e-4 and fall_ms <= 1e-4 and spread <= 1e-4 and c_min >= 0,
              f"{n_rows} rows x every HP = {n_pts} points x {len(ens.nets)} members: largest fall of the value logit {max(fall, 0):.1e}, "
              f"of a milestone logit {max(fall_ms, 0):.1e}; spread of (a, c) over HP {spread:.1e}; smallest ramp weight {c_min:.1e}")

    # ---- (5), (6) the trainer's files
    os.makedirs(a.tmp, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="layout2_check_", dir=a.tmp)
    try:
        n_half = len(RI) // 2
        p1, p2 = os.path.join(tmp, "old_000000.s3r"), os.path.join(tmp, "new_000000.s3r")
        Iv, Fv = with_v2(RI, RF, rng, n_enc)
        write_chunk(p1, RI[:n_half], RF[:n_half], 6, ms.IW_V1, ms.FW_V1, rng)        # as an old actor wrote it
        write_chunk(p2, Iv[n_half:], Fv[n_half:], 6, ms.IW, ms.FW, rng)              # as a new actor writes it
        st = tr.Store(10**6)
        got = st.scan(None, files=[p1, p2])
        wi, wf = tr.cat(st.runs, "ints"), tr.cat(st.runs, "floats")
        ok = (len(got) == 12 and wi.shape == (len(RI), ms.IW) and wf.shape == (len(RI), ms.FW)
              and np.array_equal(wi[:n_half, :ms.IW_V1], RI[:n_half, :ms.IW_V1]) and not wi[:n_half, ms.IW_V1:].any()
              and np.array_equal(wf[:n_half, :ms.FW_V1], RF[:n_half, :ms.FW_V1]) and not wf[:n_half, ms.FW_V1:].any()
              and np.array_equal(wi[n_half:], Iv[n_half:]) and np.array_equal(wf[n_half:], Fv[n_half:]))
        check("(5) chunks of layout v1 and v2 read into one window (v1 rows zero-padded, v2 rows as written)", ok,
              f"{len(got)} runs, rows {wi.shape} ints, {wf.shape} floats")

        # a layout v1 carry file: v1 rows, Adam states of the frozen network after one step
        nets1 = copy.deepcopy(e1.nets)
        opts1 = [torch.optim.Adam(n.parameters(), lr=1e-5) for n in nets1]
        b1 = v1.rows_to_batch(RI[:70, :ms.IW_V1], RF[:70, :ms.FW_V1], "cpu")
        for n, op in zip(nets1, opts1):
            op.zero_grad(); train_loss(n, b1).backward(); op.step()
        old_runs = st.runs[:6]
        carry = {"magic": "S3MCARRY1", "version": 1, "total_runs": len(old_runs),
                 "n_rows": torch.tensor([len(r["meta"]) for r in old_runs], dtype=torch.int64),
                 "n_orig": torch.tensor([len(r["meta"]) for r in old_runs], dtype=torch.int64),
                 **{k: torch.tensor([float(r[k]) for r in old_runs], dtype=torch.float64) for k in tr.RUN_KEYS},
                 "ints": torch.from_numpy(np.ascontiguousarray(tr.cat(old_runs, "ints")[:, :ms.IW_V1]).astype(np.int16)),
                 "floats": torch.from_numpy(np.ascontiguousarray(tr.cat(old_runs, "floats")[:, :ms.FW_V1])),
                 "meta": torch.from_numpy(np.ascontiguousarray(tr.cat(old_runs, "meta"))),
                 "opts": [o.state_dict() for o in opts1], "shapes": [[list(p.shape) for p in n.parameters()] for n in nets1]}
        pc = os.path.join(tmp, "carry_v1.pt")
        torch.save(carry, pc)
        nets2 = copy.deepcopy(e2.nets)
        opts2 = [torch.optim.Adam(n.parameters(), lr=1e-4) for n in nets2]
        st2 = tr.Store(10**6)
        tr.load_carry(pc, st2, opts2, nets2, 1e-4, "cpu")     # prints which optimizers were restored
        ci = tr.cat(st2.runs, "ints") if st2.runs else np.zeros((0, 0))
        check("(6) a layout v1 carry file loads: rows zero-padded", len(st2.runs) == len(old_runs) and ci.shape[1:] == (ms.IW,)
              and tr.cat(st2.runs, "floats").shape[1:] == (ms.FW,) and not ci[:, ms.IW_V1:].any(), f"{len(st2.runs)} runs, {len(ci)} rows")
        n_state = [len(op.state_dict()["state"]) for op in opts2]
        want = [len(o.state_dict()["state"]) for o in opts1]
        check("(6) ... its Adam states are restored for the old parameters", n_state == want and min(n_state) > 0,
              f"parameters with state per member {n_state} (carry: {want}; network: {len(list(nets2[0].parameters()))} parameters)")
        b2 = ms.rows_to_batch(Iv[:70], Fv[:70], "cpu")
        grads = {}
        for n, op in zip(nets2, opts2):
            op.zero_grad(); train_loss(n, b2).backward()
            for k, p in v2_params(n):
                grads.setdefault(k, []).append(0.0 if p.grad is None else float(p.grad.abs().max()))
            op.step()
        finite = all(bool(torch.isfinite(p).all()) for n in nets2 for p in n.parameters())
        moved = all(float(p.abs().max()) > 0 for n in nets2 for _, p in v2_params(n))
        check("(6) ... an Adam step runs with the partly restored states", finite)
        check("(6) gradients reach every layout v2 parameter of every member, and the step moves them from zero",
              all(min(g) > 0 for g in grads.values()) and (moved or not zero),
              ", ".join(f"{k} {min(g):.1e}" for k, g in sorted(grads.items())) + " (smallest max |gradient| over members)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED: ' + '; '.join(FAILS)} ({time.time() - t0:.1f} s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
