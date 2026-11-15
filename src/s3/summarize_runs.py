"""Summary of S3 run logs (one file per arm or actor; *_fights.jsonl files are read as fight logs of the same arm).

Per file: runs, win rate with a Wilson 95% interval, floors reached, act reached, deaths by encounter, and the risk
indicators tracked against human play (PROPOSAL.md §7): elites fought, rest vs smith. With a fight log: fights by room
and the act-3 boss line (boss 1 -> boss 2 conversion, HP share and potions at the second boss's start). With two or more arms on the same
seeds: paired win differences (McNemar exact test).

Usage: python -I src/s3/summarize_runs.py RUNS.jsonl [RUNS2.jsonl ...]
"""
import collections, json, math, os, sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def read_jsonl(p):
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except ValueError:      # a line cut off by a crash
            pass
    return out


def mean(x):
    return sum(x) / max(1, len(x))


def main():
    arms = {}
    for p in sys.argv[1:]:
        if p.endswith("_fights.jsonl") or not os.path.exists(p):
            continue
        R = read_jsonl(p)
        R = [r for r in R if not r.get("perturbed")]
        name = os.path.basename(p)[:-6]
        arms[name] = R
        ok = [r for r in R if not r.get("error")]
        n, k = len(ok), sum(r["won"] for r in ok)
        lo, hi = wilson(k, n)
        print(f"== {name}: {len(R)} runs ({len(R) - n} errors), won {k} = {100 * k / max(1, n):.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}), "
              f"mean floor {mean([r['floors'] for r in ok]):.1f}, act 1 cleared {100 * mean([r['act'] >= 1 or r['won'] for r in ok]):.0f}%, "
              f"act 2 cleared {100 * mean([r['act'] >= 2 or r['won'] for r in ok]):.0f}%")
        print(f"   per run: fights {mean([r['fights'] for r in ok]):.1f} (won {100 * sum(r['fights_won'] for r in ok) / max(1, sum(r['fights'] for r in ok)):.1f}%), "
              f"elites {mean([r['elites'] for r in ok]):.2f}, bosses won {sum(r['bosses_won'] for r in ok)}/{sum(r['bosses'] for r in ok)}, "
              f"rests {mean([r['rests'] for r in ok]):.2f}, smiths {mean([r['smiths'] for r in ok]):.2f} "
              f"(rest share {100 * sum(r['rests'] for r in ok) / max(1, sum(r['rests'] + r['smiths'] for r in ok)):.0f}%), shops {mean([r['shops'] for r in ok]):.2f}, "
              f"cards taken {mean([r['cards_taken'] for r in ok]):.1f} / skipped {mean([r['cards_skipped'] for r in ok]):.1f}, deck {mean([r['deck'] for r in ok]):.1f}, "
              f"potions used {mean([r['potions_used'] for r in ok]):.1f}, simulations {mean([r['simulations'] for r in ok]):.0f}, seconds {mean([r['seconds'] for r in ok]):.0f}")
        dead = collections.Counter(r["killed_by"] for r in ok if not r["won"])
        print("   deaths: " + ", ".join(f"{e.replace('ENCOUNTER.', '')} {c}" for e, c in dead.most_common(8)))
        fp = p[:-6] + "_fights.jsonl"
        if os.path.exists(fp):
            F = read_jsonl(fp)
            by = collections.defaultdict(list)
            for f in F:
                by[f["room_type"]].append(f)
            print("   fights: " + "; ".join(f"{t} {len(v)} won {100 * mean([f['result'] == 'won' for f in v]):.1f}% rel T_f {mean([f['rel'] for f in v]):.3f} "
                                              f"HP lost {mean([f['start_hp'] - f['end_hp'] for f in v]):.1f} potions {mean([f['potions_used'] for f in v]):.2f}"
                                              for t, v in sorted(by.items())))
            # act 3 (two bosses back to back at A10): boss-1 wins that reach and win boss 2, HP and potions at boss-2 start
            seeds = {r["seed"]: r for r in ok}
            b3 = collections.defaultdict(list)
            for f in F:
                if f["room_type"] == "boss" and f["floor"] >= 40 and not f.get("practice") and not f.get("error") and f.get("run_seed") in seeds:
                    b3[f["run_seed"]].append(f)
            b1w = [s for s, v in b3.items() if min(v, key=lambda x: x["fight"])["result"] == "won"]
            b2 = [sorted(b3[s], key=lambda x: x["fight"])[1] for s in b1w if len(b3[s]) >= 2]
            if b3:
                print(f"   act 3: boss 1 reached {len(b3)}, won {len(b1w)}; boss 2 reached {len(b2)}, won {sum(f['result'] == 'won' for f in b2)}; "
                      f"boss-1 win -> run win {sum(seeds[s]['won'] for s in b1w)}/{len(b1w)}; boss-2 start: HP share "
                      f"{mean([f['start_hp'] / max(1, f['max_hp']) for f in b2]):.3f}, potions {mean([f['potions_start'] for f in b2]):.2f}")
    names = list(arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A = {r["seed"]: r["won"] for r in arms[names[i]] if not r.get("error")}
            B = {r["seed"]: r["won"] for r in arms[names[j]] if not r.get("error")}
            both = sorted(set(A) & set(B))
            if len(both) < 10:
                continue
            b = sum(A[s] and not B[s] for s in both); c = sum(B[s] and not A[s] for s in both)
            print(f"paired {names[i]} vs {names[j]} on {len(both)} seeds: {names[i]} only {b}, {names[j]} only {c}, "
                  f"difference {100 * (b - c) / len(both):+.1f} points, McNemar p = {mcnemar(b, c):.3f}")


if __name__ == "__main__":
    main()
