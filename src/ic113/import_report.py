"""Aggregates `S0Tests importfirst` result lines (ic113 battle-start import test, src/ic113/import_starts.sh).

usage: python -I src/ic113/import_report.py [--classes CLASSES.json] RESULTS.jsonl [...]   (one group per file name
stem: everything before the last "_NN" shard suffix; with --classes (src/ic113/run_classes.py) each group is also
split by the class of the start's run: clean / modded_id / game_mode_X / modifiers / abandoned / unknown)
Prints per group: starts, share at a playable first decision, failure causes by count (with the most frequent details),
fidelity flags of the imported starts (deck / relics / potions / max HP differ from the log, with the ids most often
missing or extra), starting-enemy match classes, and the success rate by build, act and room type.
"""
import collections, json, math, os, re, sys


def ci(k, n):
    if n == 0: return (0.0, 0.0)
    p = k / n; h = 1.96 * math.sqrt(p * (1 - p) / n)
    return 100 * max(0.0, p - h), 100 * min(1.0, p + h)


def main():
    args = sys.argv[1:]; classes = None
    if args[:1] == ["--classes"]:
        classes = json.load(open(args[1])); args = args[2:]
    groups = collections.defaultdict(list)
    for p in args:
        stem = re.sub(r"_\d+$", "", os.path.basename(p).rsplit(".", 1)[0])
        for l in open(p):
            try: r = json.loads(l)
            except json.JSONDecodeError: print(f"skipped a partial line in {p} (file still being written)"); continue
            groups[stem].append(r)
            if classes is not None:
                groups[f"{stem}/{classes.get(r['id'].rsplit(':', 1)[0], 'unknown')}"].append(r)
    for g, rows in sorted(groups.items()):
        n = len(rows); ok = sum(bool(r.get("ok")) for r in rows); lo, hi = ci(ok, n)
        print(f"== {g}: {n} starts, playable first decision {ok} = {100 * ok / max(n, 1):.2f}% (95% CI {lo:.2f}-{hi:.2f})")
        causes = collections.Counter(r["cause"] for r in rows if not r.get("ok"))
        for c, k in causes.most_common():
            det = collections.Counter(r.get("detail", "") for r in rows if r.get("cause") == c)
            print(f"   fail x{k}: {c}")
            for d, m in det.most_common(5):
                if d: print(f"        x{m}: {d[:240]}")
        imp = [r for r in rows if "checks" in r]
        flags = collections.Counter(f for r in imp for f in r["checks"])
        print(f"   imported {len(imp)}; fidelity flags: " + ", ".join(f"{f} {k} ({100 * k / max(len(imp), 1):.2f}%)" for f, k in flags.most_common()))
        for what in ("deck", "relics", "potions"):
            for side in ("missing", "extra"):
                c = collections.Counter(x for r in imp for x in r.get(f"{what}_{side}", []))
                if c: print(f"     {what} {side} in the simulator: " + ", ".join(f"{x} {k}" for x, k in c.most_common(10)))
        print("   starting enemies vs log: " + ", ".join(f"{m} {k}" for m, k in collections.Counter(r.get("match", "-") for r in imp).most_common()))
        for key in ("build", "act", "room"):
            c = collections.defaultdict(lambda: [0, 0])
            for r in rows: c[r[key]][0] += 1; c[r[key]][1] += bool(r.get("ok"))
            print(f"   by {key}: " + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))))


if __name__ == "__main__":
    main()
