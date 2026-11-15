"""A10 analysis, act-3 bosses from fight logs (2026-11-14). Statistics only.
(1) boss 1 won fights at A10: the table's score at the fight's end against the real chance of then winning boss 2;
(2) win rates by boss identity. Usage: python -I src/s3/a10val_boss.py <fights dir> <runs dir> [alpha]"""
import collections, glob, json, os, sys

fdir, rdir = sys.argv[1:3]
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.175
BANKED = (1 - ALPHA) * 2 / 3
runs = {}
for p in glob.glob(os.path.join(rdir, "*.jsonl")):
    for l in open(p):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if not d.get("error") and not d.get("perturbed"):
            runs[d["seed"]] = d
by_run = collections.defaultdict(list)
for p in glob.glob(os.path.join(fdir, "*.jsonl")):
    for l in open(p):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("practice") or d.get("error") or d["run_seed"] not in runs or d["room_type"] != "boss":
            continue
        by_run[d["run_seed"]].append(d)
m = lambda xs: sum(xs) / max(1, len(xs))
B1, B2, A2 = [], [], collections.defaultdict(list)
for seed, fs in by_run.items():
    r = runs[seed]
    fs.sort(key=lambda d: int(d["id"].rsplit(":", 1)[1]))
    for i, d in enumerate(fs):
        d["run_won"] = r["won"]; d["asc"] = r["ascension"]
        if i == 1:
            A2[(r["ascension"], d["encounter"])].append(d)
        if r["ascension"] == 10 and i == 2:
            d["next"] = fs[3] if len(fs) > 3 else None
            B1.append(d)
        if r["ascension"] == 10 and i == 3:
            d["prev"] = fs[2]; B2.append(d)
W = [d for d in B1 if d["result"] == "won"]
print(f"A10 act-3 boss 1: fights {len(B1)}, won {len(W) / len(B1):.3f}; mean T_max {m([d['t_max'] for d in B1]):.3f} -> implied P(win boss 2) at the table's best entry "
      f"{(m([d['t_max'] for d in B1]) - BANKED) / (1 - BANKED):+.3f}; banked {BANKED:.3f}")
print("  won boss-1 fights by HP share at the end: table score at the end -> implied P(win boss 2) = (score - banked) / (1 - banked) vs real")
for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
    S = [d for d in W if lo <= d["end_hp"] / d["max_hp"] < hi]
    if len(S) < 20:
        continue
    sc = m([d["score"] for d in S])
    print(f"    HP [{lo:.1f},{hi:.1f}): n={len(S):5d}  share {len(S) / len(W):.2f}  score {sc:.4f}  implied {(sc - BANKED) / (1 - BANKED):+.3f}  real {m([d['run_won'] for d in S]):.3f}  "
          f"potions kept {m([d['potions_start'] - d['potions_used'] for d in S]):.2f}")
for q in (0, 1, 2):
    S = [d for d in W if (d["potions_start"] - d["potions_used"] == q if q < 2 else d["potions_start"] - d["potions_used"] >= 2)]
    if len(S) >= 20:
        sc = m([d["score"] for d in S])
        print(f"    potions kept {q}{'+' if q == 2 else ' '}: n={len(S):5d}  score {sc:.4f}  implied {(sc - BANKED) / (1 - BANKED):+.3f}  real {m([d['run_won'] for d in S]):.3f}")
print("\nA10 act-3 boss 1 by boss: fights, won | boss 2 by boss: fights, won")
for name in sorted({d["encounter"] for d in B1}):
    S1 = [d for d in B1 if d["encounter"] == name]; S2 = [d for d in B2 if d["encounter"] == name]
    print(f"  {name.replace('ENCOUNTER.', ''):24s} boss 1: n={len(S1):5d} won {m([d['result'] == 'won' for d in S1]):.3f} HP start {m([d['start_hp'] / d['max_hp'] for d in S1]):.3f} | "
          f"boss 2: n={len(S2):5d} won {m([d['result'] == 'won' for d in S2]):.3f} HP start {m([d['start_hp'] / d['max_hp'] for d in S2]):.3f}")
print("  pairs (boss 1 -> boss 2): boss-2 fights, won")
pc = collections.defaultdict(list)
for d in B2:
    pc[(d["prev"]["encounter"], d["encounter"])].append(d["result"] == "won")
for k, v in sorted(pc.items()):
    print(f"    {k[0].replace('ENCOUNTER.', ''):20s} -> {k[1].replace('ENCOUNTER.', ''):20s} n={len(v):4d} won {m(v):.3f}")
print("\nact-2 boss by boss and ascension: fights, won")
for name in sorted({k[1] for k in A2}):
    print(f"  {name.replace('ENCOUNTER.', ''):24s} " + " | ".join(f"A{a}: n={len(A2[(a, name)]):5d} won {m([d['result'] == 'won' for d in A2[(a, name)]]):.3f}" for a in (8, 9, 10)))
