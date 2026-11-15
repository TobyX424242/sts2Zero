"""A10 analysis, fight logs (2026-11-14): fights by ascension, act and room type from a segment's fights/*.jsonl joined to
its runs/*.jsonl. Statistics only. Usage: python -I src/s3/a10val_fights.py <fights dir> <runs dir> [ascensions]"""
import collections, glob, json, os, sys

fdir, rdir = sys.argv[1:3]
ASCS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [8, 9, 10]
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
        if d.get("practice") or d.get("error") or d["run_seed"] not in runs:
            continue
        by_run[d["run_seed"]].append(d)
F = []
for seed, fs in by_run.items():
    r = runs[seed]
    fs.sort(key=lambda d: int(d["id"].rsplit(":", 1)[1]))
    bw = 0
    for d in fs:
        d["asc"] = r["ascension"]; d["act"] = min(bw, 2); d["boss_no"] = bw + 1 if d["room_type"] == "boss" else 0
        d["run_won"] = r["won"]
        F.append(d)
        if d["room_type"] == "boss" and d["result"] == "won":
            bw += 1
print(f"runs {len(runs)}, runs with fights {len(by_run)}, fights {len(F)}")
m = lambda xs: sum(xs) / max(1, len(xs))
for act in (0, 1, 2):
    print(f"\n== act {act + 1}")
    print(f"{'asc':>4s} {'room':8s} {'fights':>7s} {'per run in act':>14s} {'won':>6s} {'HP start':>8s} {'HP lost (won)':>13s} {'HP end (won)':>12s} {'pot start':>9s} {'pot used':>8s} {'T_max':>6s} {'rel(won)':>8s}")
    for room in ("monster", "elite", "boss"):
        for asc in ASCS:
            S = [d for d in F if d["asc"] == asc and d["act"] == act and d["room_type"] == room]
            if room == "boss" and act == 2 and asc == 10:
                groups = (("boss 1", [d for d in S if d["boss_no"] == 3]), ("boss 2", [d for d in S if d["boss_no"] == 4]))
            else:
                groups = ((room, S),)
            nr = len({d["run_seed"] for d in F if d["asc"] == asc and d["act"] == act})
            for name, S in groups:
                if len(S) < 30:
                    continue
                W = [d for d in S if d["result"] == "won"]
                print(f"{asc:4d} {name:8s} {len(S):7d} {len(S) / max(1, nr):14.2f} {len(W) / len(S):6.3f} {m([d['start_hp'] / d['max_hp'] for d in S]):8.3f} "
                      f"{m([(d['start_hp'] - d['end_hp']) / d['max_hp'] for d in W]):13.3f} {m([d['end_hp'] / d['max_hp'] for d in W]):12.3f} "
                      f"{m([d['potions_start'] for d in S]):9.2f} {m([d['potions_used'] for d in S]):8.2f} {m([d['t_max'] for d in S]):6.3f} {m([d['rel'] for d in W]):8.3f}")
# same encounter, A9 vs A10, act 3 hallway and elite fights
print("\n== act 3, same encounter at A9 vs A10 (fights >= 60 at A9): won, HP lost share in won fights, potions used")
enc = collections.defaultdict(lambda: collections.defaultdict(list))
for d in F:
    if d["act"] == 2 and d["room_type"] in ("monster", "elite") and d["asc"] in (9, 10):
        enc[(d["room_type"], d["encounter"])][d["asc"]].append(d)
tot = {9: [0, 0, 0.0, 0.0, 0], 10: [0, 0, 0.0, 0.0, 0]}
for k, v in sorted(enc.items()):
    if len(v[9]) < 60:
        continue
    line = f"  {k[0]:7s} {k[1].replace('ENCOUNTER.', ''):30s}"
    for a in (9, 10):
        S = v[a]; W = [d for d in S if d["result"] == "won"]
        line += f" | A{a} n={len(S):5d} won {len(W) / len(S):.3f} lost {m([(d['start_hp'] - d['end_hp']) / d['max_hp'] for d in W]):.3f} pots {m([d['potions_used'] for d in S]):.2f} start {m([d['start_hp'] / d['max_hp'] for d in S]):.3f}"
    print(line)
