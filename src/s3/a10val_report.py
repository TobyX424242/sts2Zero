"""A10 run-value analysis on the output of a10val_extract.py (2026-11-14). Statistics only: no network is loaded.

Floor rows carry no recorded value (the actor does not evaluate them), so everything about the value uses decision
rows: the value the published ensemble gave the state at play time, before the trainer saw the run.

Sections: (1) the blended target per ascension group x act: how much of the loss any predictor could gain,
(2) the recorded value against the 9-cell table and how well it ranks the outcomes that matter,
(3) act 3: the value against the banked progress and against the real chance of winning, by HP,
(4) state after the first act-3 boss (A10), (5) HP and potions through acts 2-3 at A9 vs A10.

Usage: python -I src/s3/a10val_report.py <extract.npz> [alpha]
"""
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=False)
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.175
RC = {n: i for i, n in enumerate(d["run_cols"])}
C = {n: i for i, n in enumerate(d["row_cols"])}
runs, rows = d["runs"], d["rows"]
ok = runs[:, RC["perturbed"]] == 0
ri = rows[:, C["run"]].astype(np.int64)
asc_r, won_r, act_r, bw_r = runs[:, RC["asc"]], runs[:, RC["won"]], runs[:, RC["act"]], runs[:, RC["bosses_won"]]
prog_r = np.where(won_r == 1, 1.0, np.minimum(act_r, 2) / 3.0)
z_r = ALPHA * won_r + (1 - ALPHA) * prog_r
kind = rows[:, C["kind"]]
fl = (kind == 0) & ok[ri]
dec = (kind == 1) & ok[ri]
asc, won, z = asc_r[ri], won_r[ri], z_r[ri]
act_end, bw = act_r[ri], bw_r[ri]
act, v, hpf, room = rows[:, C["act"]].astype(int), rows[:, C["value"]].astype(np.float64), rows[:, C["hp_frac"]], rows[:, C["room"]].astype(int)
floor = rows[:, C["floor"]].astype(int)
npot = rows[:, C["n_potions"]]
GROUPS = (("A5-8", (asc >= 5) & (asc <= 8)), ("A9", asc == 9), ("A10", asc == 10))
BANKED = (1 - ALPHA) * 2 / 3
HPB = ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01))


def bce(z, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(z * np.log(p) + (1 - z) * np.log(1 - p))


def auc(s, y):
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 20 or n0 < 20:
        return float("nan")
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)       # midranks for ties
    rs = np.zeros(len(cnt)); np.add.at(rs, inv, r); r = (rs / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_within(s, y, key):
    """AUC averaged over groups of `key` (weighted by positive x negative pairs): ranking inside one floor only."""
    num = den = 0.0
    for k in np.unique(key):
        m = key == k
        a = auc(s[m], y[m]); w = float(y[m].sum()) * float((1 - y[m]).sum())
        if a == a:
            num += a * w; den += w
    return num / den if den else float("nan")


print(f"runs {len(runs)} (unperturbed {int(ok.sum())}), floor rows {int(fl.sum())}, decision rows {int(dec.sum())}, alpha {ALPHA}, banked in act 3 {BANKED:.3f}")
n10 = ok & (asc_r == 10)
print(f"A10 runs {int(n10.sum())}: won {won_r[n10].mean():.4f}; ended in act 1/2/3 (lost) {[round(float(((act_r == a) & (won_r == 0))[n10].mean()), 3) for a in (0, 1, 2)]}; "
      f"labels {[round(ALPHA * w + (1 - ALPHA) * p, 3) for w, p in ((0, 0), (0, 1 / 3), (0, 2 / 3), (1, 1))]}")

print("\n== 1. Blended target per cell, floor rows. 'entropy': loss of a predictor that knows every run's outcome (soft labels have a\n"
      "   floor above 0). 'table': the cell mean. 'room' = table - entropy: the most ANY predictor can gain over the table in this cell.")
print(f"{'cell':12s} {'rows':>8s} {'mean z':>7s} {'sd z':>6s} {'entropy':>8s} {'table':>7s} {'room':>6s}")
for name, g in GROUPS:
    for a in (0, 1, 2):
        m = fl & g & (act == a)
        if m.sum() < 100:
            continue
        zz = z[m]
        ent = bce(zz, zz).mean(); tab = bce(zz, np.full_like(zz, zz.mean())).mean()
        print(f"{name + ' act' + str(a + 1):12s} {int(m.sum()):8d} {zz.mean():7.3f} {zz.std():6.3f} {ent:8.4f} {tab:7.4f} {tab - ent:6.3f}")

print("\n== 2a. Recorded value on decision rows against the cell table. 'R2': share of the cell's label variance explained by V.\n"
      "    'P(win) spread': sd of V / (1 - alpha-weight) is not shown; see section 3 for act 3.")
print(f"{'cell':12s} {'rows':>8s} {'mean z':>7s} {'mean V':>7s} {'sd z':>6s} {'sd V':>6s} {'table':>7s} {'V':>7s} {'gain':>8s} {'room':>6s} {'R2':>7s} {'corr':>6s} {'logit sd':>8s}")
for name, g in GROUPS:
    for a in (0, 1, 2):
        m = dec & g & (act == a)
        if m.sum() < 100:
            continue
        zz, vv = z[m], v[m]
        ent = bce(zz, zz).mean(); tab = bce(zz, np.full_like(zz, zz.mean())).mean(); bv = bce(zz, vv).mean()
        print(f"{name + ' act' + str(a + 1):12s} {int(m.sum()):8d} {zz.mean():7.3f} {vv.mean():7.3f} {zz.std():6.3f} {vv.std():6.3f} {tab:7.4f} {bv:7.4f} {tab - bv:+8.4f} {tab - ent:6.3f} "
              f"{1 - ((zz - vv) ** 2).mean() / zz.var():7.3f} {np.corrcoef(zz, vv)[0, 1]:6.3f} {rows[m, C['vstd']].mean():8.3f}")

print("\n== 2b. Ranking by the recorded value on decision rows (AUC; 'in-floor' compares rows of the same floor only; 0.5 = no information),\n"
      "    next to ranking by HP share alone.")
c1 = (act_end >= 1) | (won == 1); c2 = (act_end >= 2) | (won == 1)
b1 = np.where(asc == 10, bw >= 3, won == 1)            # beat the first act-3 boss (A10: 3 bosses won; below: the run is won)
print(f"{'rows':18s} {'target':20s} {'n':>8s} {'base':>6s} {'AUC V':>6s} {'in-floor':>8s} {'AUC HP':>7s} {'in-floor':>8s}")
for name, g in GROUPS:
    for a, targets in ((0, (("clear act 1", c1), ("clear act 2", c2), ("win", won == 1))),
                       (1, (("clear act 2", c2), ("beat act-3 boss 1", b1), ("win", won == 1))),
                       (2, (("beat act-3 boss 1", b1), ("win", won == 1)))):
        m = dec & g & (act == a) & ~((room == 5) & (act == 2))
        for tn, y in targets:
            if name != "A10" and tn == "beat act-3 boss 1":
                continue
            yy = y[m].astype(float)
            print(f"{name + ' act ' + str(a + 1):18s} {tn:20s} {int(m.sum()):8d} {yy.mean():6.3f} {auc(v[m], yy):6.3f} {auc_within(v[m], yy, floor[m]):8.3f} "
                  f"{auc(hpf[m], yy):7.3f} {auc_within(hpf[m], yy, floor[m]):8.3f}")

print(f"\n== 3. Act 3 before the boss, decision rows: every run here has already banked {BANKED:.3f}; the label is banked + {1 - BANKED:.3f} x won.\n"
      "   'implied P(win)' = (V - banked) / (1 - banked): what the value says about winning.")
for name, g in (("A8", asc == 8), ("A9", asc == 9), ("A10", asc == 10)):
    m = dec & g & (act == 2) & (room != 5)
    if m.sum() < 100:
        continue
    print(f"  {name}: rows {int(m.sum())}, won {won[m].mean():.4f}, mean V {v[m].mean():.4f} (sd {v[m].std():.4f}), V below banked in {100 * (v[m] < BANKED).mean():.1f}% of rows, "
          f"V within +-0.02 of banked in {100 * (np.abs(v[m] - BANKED) < 0.02).mean():.1f}%")
    for lo, hi in HPB:
        k = m & (hpf >= lo) & (hpf < hi)
        if k.sum() < 30:
            continue
        print(f"    HP [{lo:.1f},{hi:.1f}): n={int(k.sum()):6d}  won {won[k].mean():.4f}  beat boss 1 {b1[k].mean():.4f}  mean V {v[k].mean():.4f}  implied P(win) {(v[k].mean() - BANKED) / (1 - BANKED):+.4f}")
    for q in (0, 1, 2, 3):
        k = m & (npot == q) if q < 3 else m & (npot >= 3)
        if k.sum() < 30:
            continue
        print(f"    potions {q}{'+' if q == 3 else ' '}: n={int(k.sum()):6d}  won {won[k].mean():.4f}  beat boss 1 {b1[k].mean():.4f}  mean V {v[k].mean():.4f}  implied P(win) {(v[k].mean() - BANKED) / (1 - BANKED):+.4f}")

print("\n== 4. A10, state after the first act-3 boss (floor row of that boss; no value recorded there): real chance of beating boss 2")
m = fl & (asc == 10) & (act == 2) & (room == 5)
if m.sum() > 50:
    yy = (won[m] == 1).astype(float)
    print(f"  rows {int(m.sum())}, won {yy.mean():.3f}; AUC of HP share alone {auc(hpf[m], yy):.3f}; mean HP share {hpf[m].mean():.3f}; holding no potion {100 * (npot[m] == 0).mean():.0f}%")
    print("  by HP share:   " + "   ".join(f"[{lo:.1f},{hi:.1f}) n={int(k.sum())} won {yy[k].mean():.3f}" for lo, hi in HPB for k in [(hpf[m] >= lo) & (hpf[m] < hi)] if k.sum() > 10))
    print("  by potions:    " + "   ".join(f"{int(q)}: n={int((npot[m] == q).sum())} won {yy[npot[m] == q].mean():.3f}" for q in np.unique(npot[m]) if (npot[m] == q).sum() > 10))
    dm = dec & (asc == 10) & (act == 2) & (room == 5)
    if dm.sum() > 50:
        print(f"  decision rows on that floor: {int(dm.sum())}; mean V {v[dm].mean():.4f}, implied P(win) {(v[dm].mean() - BANKED) / (1 - BANKED):+.4f}, real {won[dm].mean():.4f}")
        for lo, hi in HPB:
            k = dm & (hpf >= lo) & (hpf < hi)
            if k.sum() > 10:
                print(f"    HP [{lo:.1f},{hi:.1f}): n={int(k.sum()):5d}  won {won[k].mean():.3f}  mean V {v[k].mean():.4f}  implied P(win) {(v[k].mean() - BANKED) / (1 - BANKED):+.4f}")

print("\n== 5. Acts 2 and 3 at A9 vs A10 (same content until the first act-3 boss): floor rows by floors into the act (survivors only)")
fia = np.round(rows[:, C["floor_in_act"]]).astype(int)
for a in (1, 2):
    print(f"  act {a + 1}: floor in act: A9 rows / HP share / potions | A10 rows / HP share / potions")
    for k in range(1, 16):
        out = []
        for g in (asc == 9, asc == 10):
            m = fl & g & (act == a) & (fia == k) & (room != 5)
            out.append(f"{int(m.sum()):6d} {hpf[m].mean() if m.any() else float('nan'):.3f} {npot[m].mean() if m.any() else float('nan'):.2f}")
        print(f"    {k:2d}: " + " | ".join(out))
