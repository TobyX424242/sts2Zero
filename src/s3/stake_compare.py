"""Paired comparison of two eval_s3.sh arms on act 3 (2026-11-14): per seed played by both, the stages of the last act
and potion / HP use in its fights, with bootstrap intervals of the paired difference.
usage: python -I src/s3/stake_compare.py OUTDIR ARM_A ARM_B"""
import collections, json, os, random, sys

O, A, B = sys.argv[1:4]


def load(arm):
    runs = {}
    for l in open(os.path.join(O, f"{arm}.jsonl")):
        d = json.loads(l)
        if not d.get("error"):
            runs[d["seed"]] = d
    fights = collections.defaultdict(list)
    for l in open(os.path.join(O, f"{arm}_fights.jsonl")):
        d = json.loads(l)
        if d["run_seed"] in runs and not d.get("practice"):
            fights[d["run_seed"]].append(d)
    out = {}
    for s, r in runs.items():
        fs = sorted(fights[s], key=lambda f: int(f["id"].rsplit(":", 1)[1]))
        bw, x = 0, collections.defaultdict(float)
        for f in fs:
            if bw >= 2 and f["room_type"] in ("monster", "elite"):
                k = f["room_type"]
                x[f"{k}_n"] += 1; x[f"{k}_pot"] += f["potions_used"]; x[f"{k}_lost"] += (f["start_hp"] - f["end_hp"]) / f["max_hp"]
                x[f"{k}_dead"] += f["result"] != "won"
            if f["room_type"] == "boss":
                if bw == 2:
                    x["b1_fought"] = 1; x["b1_won"] = float(f["result"] == "won"); x["b1_hp"] = f["start_hp"] / f["max_hp"]; x["b1_pot"] = f["potions_start"]
                if bw == 3:
                    x["b2_hp"] = f["start_hp"] / f["max_hp"]; x["b2_pot"] = f["potions_start"]
                if f["result"] == "won":
                    bw += 1
        x["act3"] = float(r["act"] >= 2 or r["won"]); x["won"] = float(r["won"]); x["bosses_won"] = r["bosses_won"]
        out[s] = x
    return out


a, b = load(A), load(B)
seeds = sorted(set(a) & set(b))
print(f"seeds played by both: {len(seeds)}")


def paired(name, f, cond=lambda x: True):
    S = [s for s in seeds if cond(a[s]) and cond(b[s])]
    va, vb = [f(a[s]) for s in S], [f(b[s]) for s in S]
    d = [y - x for x, y in zip(va, vb)]
    rnd = random.Random(0); bs = []
    for _ in range(2000):
        k = [rnd.randrange(len(d)) for _ in d]; bs.append(sum(d[i] for i in k) / len(d))
    bs.sort()
    print(f"  {name:48s} n={len(S):4d}  {A} {sum(va) / len(S):.4f}  {B} {sum(vb) / len(S):.4f}  diff {sum(d) / len(d):+.4f} ({bs[50]:+.4f} to {bs[1949]:+.4f})")


paired("runs won", lambda x: x["won"])
paired("bosses won per run", lambda x: x["bosses_won"])
paired("reached act 3", lambda x: x["act3"])
both3 = lambda x: x["act3"] == 1
paired("act 3: reached boss 1 (runs in act 3 in both)", lambda x: x.get("b1_fought", 0.0), both3)
paired("act 3: hallway potions per run", lambda x: x["monster_pot"], both3)
paired("act 3: elite potions per run", lambda x: x["elite_pot"], both3)
paired("act 3: hallway HP lost per run (share of max)", lambda x: x["monster_lost"], both3)
paired("act 3: elite HP lost per run", lambda x: x["elite_lost"], both3)
fb = lambda x: x.get("b1_fought", 0) == 1
paired("boss 1 (fought in both): HP share at start", lambda x: x["b1_hp"], fb)
paired("boss 1: potions at start", lambda x: x["b1_pot"], fb)
paired("boss 1: won", lambda x: x["b1_won"], fb)
paired("boss 1: run won", lambda x: x["won"], fb)
b2 = lambda x: "b2_hp" in x
paired("boss 2 (reached in both): HP share at start", lambda x: x["b2_hp"], b2)
paired("boss 2: potions at start", lambda x: x["b2_pot"], b2)
paired("boss 2: run won", lambda x: x["won"], b2)
