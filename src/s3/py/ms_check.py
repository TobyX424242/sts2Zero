"""Checks of the milestone heads (2026-11-14), CPU, no training: on stored rows of a chunk directory and a macro file
with random milestone heads,
  (1) the served value (column 27) equals the formula the C# actor uses for its tables (MacroLayout.MilestoneValue,
      re-implemented here line by line) at the row's own HP;
  (2) the blend is non-decreasing in HP for every row and lies between the banked progress and 1;
  (3) the blend's limits: every head 1 -> 1, every head 0 -> the banked progress;
  (4) the trainer's labels: the share of rows per stage and head that train it, and the labels' means.
usage: python -I src/s3/py/ms_check.py <chunk dir> <macro.pt>"""
import glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import macro_s3 as ms
import train_macro_s3 as t

torch.manual_seed(0)
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.s3r")))[:20]
st = t.Store(10**9); runs = st.scan(None, files=files)
ints, floats = t.cat(runs, "ints"), t.cat(runs, "floats")
print(f"runs {len(runs)}, rows {len(ints)}")
ens = ms.Ensemble.load(sys.argv[2], "cpu")
for n in ens.nets:
    torch.nn.init.normal_(n.ms_win.weight, std=0.3); torch.nn.init.normal_(n.ms_win.bias, std=1.0)
    torch.nn.init.normal_(n.ms_mono.weight, std=0.1)
ens.milestones, ens.alpha, ens.boss1_share = True, 0.33, 0.5
sel = np.flatnonzero((ints[:, ms.OFF["opt_kind"][0]] == 0) & ~np.isin(ints[:, ms.OFF["dtype"][0]], ms.DT_SLOT))[:3000]   # value rows
out = ens.serve(ints[sel], floats[sel], "cpu").numpy()
hp = np.round(floats[sel, ms.S["hp"]] * 100).astype(int); mx = np.round(floats[sel, ms.S["max_hp"]] * 100).astype(int)
c0 = ms.OFF["ctx"][0]
stage = ms.stage_of(ints[sel, c0 + 3].astype(int), ints[sel, c0 + 4].astype(int)); dbl = ints[sel, c0 + 1] >= 10


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit_at(o, h, m, at):        # MacroLayout.LogitAt
    z = o[at]
    for k in range(16):
        z += o[at + 1 + k] * np.clip((h - 10.0 * k) / 10.0, 0, 1)
    fr = np.clip(h / max(1, m), 0, 1)
    for k in range(10):
        z += o[at + 17 + k] * np.clip((fr - 0.1 * k) / 0.1, 0, 1)
    return z


def ms_value(o, h, m, s, d):      # MacroLayout.MilestoneValue
    al, w, run, r = o[ms.O_ALPHA], o[ms.O_W1], 1.0, []
    for j in range(4):
        q = 1.0 if (j == 3 and not d) else sig(logit_at(o, h, m, ms.O_MS + j * 27))
        run = 1.0 if s > j else run * q
        r.append(run)
    return al * r[3] + (1 - al) * (r[0] + r[1] + w * r[2] + (1 - w) * r[3]) / 3


err = max(abs(np.log(np.clip(ms_value(out[i], hp[i], mx[i], stage[i], dbl[i]), 1e-6, 1 - 1e-6) / (1 - np.clip(ms_value(out[i], hp[i], mx[i], stage[i], dbl[i]), 1e-6, 1 - 1e-6))) - out[i, ms.O_V])
          for i in range(len(sel)))
print(f"(1) served logit vs the actor's formula at own HP, {len(sel)} rows: max |difference| {err:.2e}")
viol, lo_viol = 0, 0
for i in range(len(sel)):
    vs = [ms_value(out[i], h, mx[i], stage[i], dbl[i]) for h in range(1, mx[i] + 1)]
    viol += int(np.any(np.diff(vs) < -1e-9))
    lo_viol += int(min(vs) < ms.banked(np.array([stage[i]]), 0.33, 0.5)[0] - 1e-9)
print(f"(2) rows with V decreasing in HP: {viol}; rows with V below the banked progress: {lo_viol}")
q1 = torch.ones(5, 4); q0 = torch.zeros(5, 4); sg = torch.arange(5).clamp(max=3); d = torch.tensor([True] * 5)
print("(3) all heads 1:", ms.blend(q1, sg, d, 0.33, 0.5).numpy().round(4), " all heads 0:", ms.blend(q0, sg, d, 0.33, 0.5).numpy().round(4),
      " banked:", ms.banked(sg.numpy(), 0.33, 0.5).round(4))
T = [t.ms_targets(r) for r in runs]
L, M = np.concatenate([x[0] for x in T]), np.concatenate([x[1] for x in T]) > 0
S = t.stage_rows(ints); A = np.concatenate([np.full(len(r["meta"]), r["asc"]) for r in runs])
print("(4) rows per stage (A10 / below):", [int(((S == k) & (A >= 10)).sum()) for k in range(4)], [int(((S == k) & (A < 10)).sum()) for k in range(4)])
for m, hn in enumerate(("act1", "act2", "boss1", "boss2")):
    for name, g in (("A10", A >= 10), ("A<10", A < 10)):
        k = M[:, m] & g
        print(f"    head {hn:5s} {name:4s}: rows {int(k.sum()):6d}  label mean {L[k, m].mean() if k.any() else float('nan'):.3f}")
b1 = np.array([r["b1"] for r in runs]); won = np.array([r["won"] for r in runs]); asc = np.array([r["asc"] for r in runs])
print(f"    runs: A10 won {won[asc >= 10].mean():.3f}, beat act-3 boss 1 {b1[asc >= 10].mean():.3f}; below A10 b1 == won: {bool((b1[asc < 10] == won[asc < 10]).all())}")
