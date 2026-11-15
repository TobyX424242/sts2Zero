"""Report of an ascension sweep (eval_asc_sweep.sh): win rate per ascension and where the runs die.

A death is placed by the run's lost fight in the fight log (room type, act from `act`, encounter); a lost run without a
lost fight is counted as "other" (event or non-combat death). Act 3 bosses at A10 are split into boss 1 and boss 2.

Usage: python -I research/s3/asc_sweep_report.py OUTDIR   (OUTDIR/aNN/tf64.jsonl and tf64_fights.jsonl)
"""
import collections, glob, json, math, os, sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rd(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    return out


def main(O):
    rows, killers, where_all = [], collections.Counter(), {}
    for d in sorted(glob.glob(os.path.join(O, "a[0-9][0-9]"))):
        a = int(os.path.basename(d)[1:])
        R = [r for r in rd(os.path.join(d, "tf64.jsonl")) if not r.get("perturbed")]
        ok = [r for r in R if not r.get("error")]
        F = [f for f in rd(os.path.join(d, "tf64_fights.jsonl")) if not f.get("practice")]
        byrun = collections.defaultdict(list)
        for f in F:
            byrun[f["run_seed"]].append(f)
        n, k = len(ok), sum(r["won"] for r in ok)
        lo, hi = wilson(k, n)
        where, kc = collections.Counter(), collections.Counter()
        for r in ok:
            if r["won"]:
                continue
            fl = sorted(byrun.get(r["seed"], []), key=lambda f: f["fight"])
            lost = [f for f in fl if f["result"] != "won"]
            act = min(r["act"], 2) + 1
            if not lost:
                where[(act, "other")] += 1; kc["(no lost fight: event/other)"] += 1
                continue
            f = lost[-1]
            rt = {"monster": "hallway"}.get(f["room_type"], f["room_type"])
            if rt == "boss" and act == 3:
                nb = sum(1 for g in fl if g["room_type"] == "boss" and g["floor"] >= 40)
                rt = "boss 1" if nb <= 1 else "boss 2"
            where[(act, rt)] += 1
            e = f["encounter"].replace("ENCOUNTER.", "")
            kc[e] += 1; killers[e] += 1
        where_all[a] = (n, where)
        rows.append((a, len(R), n, k, lo, hi, sum(r["floors"] for r in ok) / max(1, n),
                     sum(r["act"] >= 1 or r["won"] for r in ok) / max(1, n),
                     sum(r["act"] >= 2 or r["won"] for r in ok) / max(1, n), kc,
                     sum(r["seconds"] for r in ok) / max(1, n)))
    print("Win rate by ascension (greedy macro, search 64 on every move, seeds EVAL000000+)")
    print(f"{'asc':>4} {'runs':>5} {'err':>4} {'won':>4} {'win%':>6} {'95% CI':>12} {'floor':>6} {'act1 clr':>8} {'act2 clr':>8}")
    for a, nr, n, k, lo, hi, fl, a1, a2, kc, s in rows:
        print(f"A{a:<3} {n:>5} {nr - n:>4} {k:>4} {100 * k / max(1, n):>5.1f}% {100 * lo:>5.1f}-{100 * hi:<5.1f} {fl:>6.1f} {100 * a1:>7.0f}% {100 * a2:>7.0f}%")
    cols = [(1, "hallway"), (1, "elite"), (1, "boss"), (2, "hallway"), (2, "elite"), (2, "boss"),
            (3, "hallway"), (3, "elite"), (3, "boss 1"), (3, "boss 2"), (3, "boss")]
    present = [c for c in cols if any(w[c] for _, w in where_all.values())]
    other = any(w[(x, "other")] for _, w in where_all.values() for x in (1, 2, 3))
    print("\nWhere runs die (% of all runs at that ascension)")
    hdr = " ".join(f"{'A%d %s' % c:>11}" for c in present) + (f" {'other':>7}" if other else "")
    print(f"{'asc':>4} " + hdr)
    for a, (n, w) in where_all.items():
        line = " ".join(f"{100 * w[c] / max(1, n):>10.1f}%" for c in present)
        if other:
            line += f" {100 * sum(w[(x, 'other')] for x in (1, 2, 3)) / max(1, n):>6.1f}%"
        print(f"A{a:<3} " + line)
    print("\nTop killers per ascension")
    for a, *_, kc, s in rows:
        print(f"A{a:<3} " + ", ".join(f"{e} {c}" for e, c in kc.most_common(6)))
    print("\nTop killers, all ascensions: " + ", ".join(f"{e} {c}" for e, c in killers.most_common(12)))


if __name__ == "__main__":
    main(sys.argv[1])
