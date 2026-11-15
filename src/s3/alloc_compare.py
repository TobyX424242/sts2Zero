#!/usr/bin/env python3
"""Compares the arms of research/s3/alloc_test.sh: run results, fights by room (won, HP lost, potions used, search spent),
HP at boss starts, search cost per run, and each arm against the first arm on the same seeds (paired bootstrap).
usage: alloc_compare.py OUTDIR [arm ...]   (default: every <arm>.jsonl in OUTDIR, the first listed is the reference)"""
import json, os, sys, glob, random, collections

O = sys.argv[1]
arms = sys.argv[2:] or sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{O}/*.jsonl") if not p.endswith("_fights.jsonl"))


def load(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try: out.append(json.loads(line))
            except ValueError: pass
    return out


def act_of(floor):
    return 1 if floor <= 17 else 2 if floor <= 33 else 3


R, F = {}, {}
for a in arms:
    R[a] = {r["seed"]: r for r in load(f"{O}/{a}.jsonl") if not r.get("error")}
    F[a] = load(f"{O}/{a}_fights.jsonl")
common = set.intersection(*(set(R[a]) for a in arms)) if arms else set()
print(f"arms {arms}; seeds played by every arm without error: {len(common)}")
for a in arms:
    print(f"  {a}: {len(R[a])} runs, errors {sum(1 for r in load(f'{O}/{a}.jsonl') if r.get('error'))}")


def run_stats(a, seeds):
    rs = [R[a][s] for s in seeds]; n = len(rs)
    acts = [3 if r["won"] else min(r["act"], 2) for r in rs]
    return {"n": n, "won": sum(r["won"] for r in rs) / n, "act1": sum(x >= 1 for x in acts) / n, "act2": sum(x >= 2 for x in acts) / n,
            "floor": sum(r["floors"] for r in rs) / n, "sims": sum(r["simulations"] for r in rs) / n,
            "bosses_won": sum(r["bosses_won"] for r in rs) / n, "potions": sum(r["potions_used"] for r in rs) / n}


print("\n== runs (seeds common to all arms)")
for a in arms:
    s = run_stats(a, common)
    print(f"  {a:6s} won {100*s['won']:5.1f}%  act 1 {100*s['act1']:5.1f}%  act 2 {100*s['act2']:5.1f}%  floor {s['floor']:5.1f}  bosses won {s['bosses_won']:.2f}"
          f"  potions used {s['potions']:.1f}  simulations per run {s['sims']:,.0f} ({100*s['sims']/run_stats(arms[0], common)['sims']:.0f}% of {arms[0]})")


def fights_by(a, seeds):
    g = collections.defaultdict(list)
    for x in F[a]:
        if x.get("run_seed") in seeds and not x.get("error"):
            g[(x["room_type"], act_of(x["floor"]))].append(x)
    return g


print("\n== fights by room and act: n, won, HP lost per fight (share of max HP, fights won), potions used, simulations per decision, HP share at start")
G = {a: fights_by(a, common) for a in arms}
keys = sorted({k for a in arms for k in G[a]}, key=lambda k: (k[1], ["monster", "elite", "boss"].index(k[0]) if k[0] in ("monster", "elite", "boss") else 9))
for k in keys:
    print(f"  {k[0]:7s} act {k[1]}")
    for a in arms:
        xs = G[a].get(k, [])
        if not xs: continue
        won = [x for x in xs if x["result"] == "won"]
        lost = sum((x["start_hp"] - x["end_hp"]) / max(1, x["max_hp"]) for x in won) / max(1, len(won))
        hp0 = sum(x["start_hp"] / max(1, x["max_hp"]) for x in xs) / len(xs)
        spd = sum(x["simulations"] for x in xs) / max(1, sum(x["decisions"] for x in xs))
        print(f"    {a:6s} n {len(xs):5d}  won {100*len(won)/len(xs):5.1f}%  HP lost {100*lost:5.1f}%  potions {sum(x['potions_used'] for x in xs)/len(xs):.2f}"
              f"  sims/decision {spd:5.1f}  HP at start {100*hp0:5.1f}%")


def act3_bosses(a, seeds):
    """Per seed: the run's act-3 boss fights in order (boss fights at floor >= 40; at A10 the act ends with two)."""
    g = collections.defaultdict(list)
    for x in F[a]:
        if x.get("run_seed") in seeds and not x.get("error") and not x.get("practice") and x["room_type"] == "boss" and x["floor"] >= 40:
            g[x["run_seed"]].append(x)
    return {s: sorted(v, key=lambda x: x["fight"]) for s, v in g.items()}


def hall(a, seeds):
    """Per seed: hallway fights (potions used, HP lost share)."""
    g = collections.defaultdict(list)
    for x in F[a]:
        if x.get("run_seed") in seeds and not x.get("error") and not x.get("practice") and x["room_type"] == "monster":
            g[x["run_seed"]].append(x)
    return g


print("\n== act 3 bosses (double boss at A10): boss 1 reached / won, boss 2 reached / won, run wins after a boss-1 win; HP share and potions at the starts")
B3 = {a: act3_bosses(a, common) for a in arms}
H = {a: hall(a, common) for a in arms}
for a in arms:
    b = B3[a]
    b1 = [v[0] for v in b.values()]; b1w = [x for x in b1 if x["result"] == "won"]
    b2 = [v[1] for v in b.values() if len(v) >= 2]; b2w = [x for x in b2 if x["result"] == "won"]
    won_after = sum(R[a][s]["won"] for s, v in b.items() if v[0]["result"] == "won")
    sh = lambda xs: sum(x["start_hp"] / max(1, x["max_hp"]) for x in xs) / max(1, len(xs))
    po = lambda xs: sum(x["potions_start"] for x in xs) / max(1, len(xs))
    pu = lambda xs: sum(x["potions_used"] for x in xs) / max(1, len(xs))
    lost = lambda xs: sum((x["start_hp"] - x["end_hp"]) / max(1, x["max_hp"]) for x in xs) / max(1, len(xs))
    print(f"  {a:6s} boss 1: n {len(b1):4d} won {100*len(b1w)/max(1,len(b1)):5.1f}%  HP {100*sh(b1):5.1f}%  potions {po(b1):.2f} used {pu(b1):.2f}  HP lost when won {100*lost(b1w):5.1f}%"
          f" | boss 2: n {len(b2):4d} won {100*len(b2w)/max(1,len(b2)):5.1f}%  HP {100*sh(b2):5.1f}%  potions {po(b2):.2f} used {pu(b2):.2f}"
          f" | run won after boss-1 win {won_after}/{len(b1w)} = {100*won_after/max(1,len(b1w)):.1f}%")
for a in arms:
    xs = [x for v in H[a].values() for x in v]; won = [x for x in xs if x["result"] == "won"]
    print(f"  {a:6s} hallway fights per run {len(xs)/len(common):.2f}: potions used per fight {sum(x['potions_used'] for x in xs)/max(1,len(xs)):.3f}"
          f" (per run {sum(x['potions_used'] for x in xs)/len(common):.2f}), HP lost per won fight {100*sum((x['start_hp']-x['end_hp'])/max(1,x['max_hp']) for x in won)/max(1,len(won)):.1f}% of max,"
          f" won {100*len(won)/max(1,len(xs)):.2f}%")


def boot(a, b, seeds, f, n=2000):
    seeds = sorted(seeds); rng = random.Random(1)
    d = [f(R[b][s]) - f(R[a][s]) for s in seeds]
    m = sum(d) / len(d)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n)]


print(f"\n== paired differences against {arms[0]} on the same seeds (mean, 95% bootstrap CI)")
for b in arms[1:]:
    parts = []
    for name, f in (("won", lambda r: float(r["won"])), ("acts cleared", lambda r: 3.0 if r["won"] else float(min(r["act"], 2))),
                    ("floor", lambda r: float(r["floors"])), ("bosses won", lambda r: float(r["bosses_won"]))):
        m, lo, hi = boot(arms[0], b, common, f)
        parts.append(f"{name} {m:+.3f} ({lo:+.3f} to {hi:+.3f})")
    print(f"  {b}: " + "; ".join(parts))
    # act 3 and hallway per seed (0 when the run never got there): boss 2 reached, act-3 bosses won, hallway potions used
    parts = []
    for name, f in (("boss 2 reached", lambda arm, s: float(len(B3[arm].get(s, [])) >= 2)),
                    ("act-3 bosses won", lambda arm, s: float(sum(x["result"] == "won" for x in B3[arm].get(s, [])))),
                    ("hallway potions used", lambda arm, s: float(sum(x["potions_used"] for x in H[arm].get(s, [])))),
                    ("potions at act-3 boss 1", lambda arm, s: float(B3[arm][s][0]["potions_start"]) if s in B3[arm] else 0.0)):
        ss = sorted(common); rng = random.Random(1)
        d = [f(b, s) - f(arms[0], s) for s in ss]
        m = sum(d) / len(d); bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
        parts.append(f"{name} {m:+.3f} ({bs[50]:+.3f} to {bs[1949]:+.3f})")
    print(f"  {' ' * len(b)}  " + "; ".join(parts))
    # seeds where both arms reached act-3 boss 1: potions held and HP share there (conditional, so selection differs little)
    both = [s for s in common if s in B3[arms[0]] and s in B3[b]]
    if both:
        dp = sum(B3[b][s][0]["potions_start"] - B3[arms[0]][s][0]["potions_start"] for s in both) / len(both)
        dh = sum(B3[b][s][0]["start_hp"] / max(1, B3[b][s][0]["max_hp"]) - B3[arms[0]][s][0]["start_hp"] / max(1, B3[arms[0]][s][0]["max_hp"]) for s in both) / len(both)
        print(f"  {' ' * len(b)}  on the {len(both)} seeds where both reached act-3 boss 1: potions held there {dp:+.2f}, HP share {dh:+.3f}")
