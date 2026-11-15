"""What a held potion is worth in T_f terms, from the actor's T_f dump (tf_dump=FILE), under three scorings of the same
dumped states: base (the macro network's table, death 0), F1 (death_value=progress: death = (1 - alpha) acts cleared / 3,
capped at T_f(1 HP, no potions)) and F1+F2 (also the double-boss model W2 for the first of two act-3 bosses).

Per fight kind (hallway fight of act 1/2/3; act-3 boss: first of two, final), at the fight's start HP with its whole belt:
  potion   mean marginal T_f of one held potion, T(h, all) - T(h, all without it)
  stakes   T(h, all) - death value (what the fight's start position is worth above dying)
  ratio    potion / stakes
  hp_eq    the potion in HP: h - min{h' : T(h', all) >= T(h, all without it)} (unchanged by F1)
Tables get free disposal as in TfScorer.

usage: python -I research/s3/tf_dump_report.py ALPHA BOSS2.json DUMP.jsonl [DUMP.jsonl ...]
"""
import json, math, sys
from collections import defaultdict


def free_disposal(subsets, tables):
    t = [list(x) for x in tables]
    for k, sk in enumerate(subsets):
        for j, sj in enumerate(subsets):
            if j != k and len(sj) < len(sk) and set(sj) <= set(sk):
                for h in range(1, min(len(t[k]), len(t[j]))):
                    t[k][h] = max(t[k][h], t[j][h])
    return t


def main():
    alpha, w2 = float(sys.argv[1]), json.load(open(sys.argv[2]))
    sig = lambda x: 1 / (1 + math.exp(-x))
    W2 = lambda h, n: sig(w2["b0"] + w2["b_hp"] * h + w2.get("b_hp2", 0) * h * h + w2["b_pot"] * min(n, w2.get("pot_cap", 3)))
    acc = defaultdict(lambda: defaultdict(list))
    for path in sys.argv[3:]:
        for line in open(path):
            x = json.loads(line)
            S, M, hp, act = x["subsets"], x["max_hp"], max(1, min(x["hp"], x["max_hp"])), x["act"]
            if not S or len(S[-1]) == 0:
                continue                                    # no potion held: nothing to price
            kind = x["kind"] if x["kind"] != "boss3" else ("boss3_final" if x["final_boss"] else "boss3_first" if x["first_of_two_bosses"] else "boss3_other")
            dform = (1 - alpha) * min(act, 2) / 3
            raw = x["raw"]
            variants = {"base": (raw, 0.0)}
            d1 = dform if x["final_boss"] or dform == 0 else min(dform, raw[0][1])
            variants["F1"] = (raw, d1)
            if x["first_of_two_bosses"]:
                variants["F1+F2"] = ([[dform] + [dform + (1 - dform) * W2(h / M, len(s)) for h in range(1, M + 1)] for s in S], dform)
            else:
                variants["F1+F2"] = (raw, d1)
            full = len(S) - 1
            for name, (tabs, d) in variants.items():
                t = free_disposal(S, tabs)
                top = t[full][hp]
                pots, hpeq = [], []
                for p in S[full]:
                    rest = sorted(set(S[full]) - {p})
                    k = next(i for i, s in enumerate(S) if sorted(s) == rest)
                    pots.append(top - t[k][hp])
                    target = t[k][hp]
                    hmin = next((h2 for h2 in range(1, hp + 1) if t[full][h2] >= target - 1e-12), hp)
                    hpeq.append(hp - hmin if top - target > 1e-9 else 0)   # a potion worth nothing is worth 0 HP
                a = acc[kind][name]
                a.append((sum(pots) / len(pots), top - d, sum(hpeq) / len(hpeq), len(S[full])))
    print(f"alpha {alpha}; W2 {w2['b0']:+.3f} {w2['b_hp']:+.3f}*h {w2['b_pot']:+.3f}*P")
    print(f"{'kind':14s} {'n':>4s} {'potions':>7s} | " + " | ".join(f"{v:^33s}" for v in ("base", "F1", "F1+F2")))
    print(f"{'':14s} {'':>4s} {'held':>7s} | " + " | ".join(f"{'potion':>8s} {'stakes':>7s} {'ratio':>6s} {'hp_eq':>6s}" for _ in range(3)))
    order = ["monster1", "monster2", "monster3", "boss3_first", "boss3_final", "boss3_other"]
    for kind in [k for k in order if k in acc] + [k for k in acc if k not in order]:
        rows = acc[kind]
        n = len(rows["base"])
        held = sum(r[3] for r in rows["base"]) / n
        cells = []
        for v in ("base", "F1", "F1+F2"):
            r = rows[v]
            pm = sum(x[0] for x in r) / n; st = sum(x[1] for x in r) / n
            ratio = sum(x[0] / x[1] for x in r if x[1] > 1e-9) / max(1, sum(1 for x in r if x[1] > 1e-9))
            cells.append(f"{pm:8.4f} {st:7.4f} {ratio:6.3f} {sum(x[2] for x in r) / n:6.1f}")
        print(f"{kind:14s} {n:4d} {held:7.2f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
