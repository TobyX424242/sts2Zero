#!/usr/bin/env python3
"""ic113 potion valuation phase 2: the paired final-boss probe (PLAN step 8; src/ic113/CHAIN.md). Written, NOT run yet.

T(k) = win points of one potion k at the final boss: every sampled final-boss entry is played once with its belt emptied
("none") and once with exactly the one potion k (the S2 fight harness, src/ic113/potion_probe.sh), and
T(k) = 100 x (win rate with k - win rate with none) over the same entries (paired by entry). As experiment 21 of v0.111
(src/exp/swap_potions.py, kind_report.py), but on fresh final-boss entries of the current agent, NOT on the
boss-practice copies of self-play (mutating those would bias the combat network's training fights).

  potion_probe.py prep STARTS_GLOB OUTDIR [--budget 2000] [--asc 10] [--kinds FILE.json|all] [--seed 0]
      STARTS_GLOB: starts_dump files of S3 actors (S3 play starts_dump=FILE; live boss fights only, no perturbed runs);
      the final-boss lines (final_boss true) of ascension --asc are sampled, S = budget / (kinds + 1) entries; writes
      OUTDIR/probe.jsonl (ids "<kind|none>|<entry id>") and OUTDIR/probe_tables.jsonl (final boss: table 1 above 0 HP).
      --kinds: a potion_mix Vbar file (its act-3 kinds) or a JSON list; default all kinds seen in the entries' belts.
  potion_probe.py report RESULTS.jsonl OUT_T.json [--min-pairs 20]
      RESULTS.jsonl: the S2 fight log of the probe; writes T(k) (win points) for potion_mix.py vbar --table, and prints
      every kind with its standard error (paired).
"""
import argparse, glob, json, math, os, random, sys
from collections import defaultdict


def cmd_prep(a):
    lines = []
    for p in sorted(glob.glob(a.starts)):
        for l in open(p):
            if not l.strip():
                continue
            d = json.loads(l)
            if d.get("final_boss") and (a.asc < 0 or int(d.get("ascension", -1)) == a.asc):
                lines.append(d)
    if not lines:
        raise SystemExit(f"no final-boss entries of ascension {a.asc} in {a.starts}")
    if a.kinds == "all":
        kinds = sorted({k for d in lines for k in d.get("potions", [])})
    elif a.kinds.endswith(".json"):
        j = json.load(open(a.kinds))
        kinds = sorted(j["acts"]["3"]) if isinstance(j, dict) else sorted(j)
    else:
        kinds = [k for k in a.kinds.split(",") if k]
    rng = random.Random(a.seed)
    n = a.entries if a.entries > 0 else max(1, a.budget // (len(kinds) + 1))
    sample = rng.sample(lines, min(n, len(lines)))
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "probe.jsonl"), "w") as fo, open(os.path.join(a.outdir, "probe_tables.jsonl"), "w") as ft:
        for s0 in sample:
            for k in ["none"] + kinds:
                s = dict(s0); s.pop("table", None)
                s["id"] = f"{k}|{s0['id']}"; s["base_id"] = s0["id"]
                s["potions"] = [] if k == "none" else [k]; s["potion_slots"] = max(int(s0.get("potion_slots", 1)), 1)
                fo.write(json.dumps(s) + "\n")
                subsets = [[]] + ([[0]] if s["potions"] else [])
                ft.write(json.dumps({"id": s["id"], "max_hp": s["max_hp"], "belt": s["potions"], "subsets": subsets,
                                     "tables": [[0.0] + [1.0] * s["max_hp"] for _ in subsets], "final_boss": True}) + "\n")
    print(f"final-boss entries {len(lines)} (ascension {a.asc}), sampled {len(sample)}, kinds {len(kinds)}, "
          f"fights {len(sample) * (len(kinds) + 1)} -> {a.outdir}/probe.jsonl")


def cmd_report(a):
    won = defaultdict(dict)   # entry -> kind -> won
    for l in open(a.results):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("error"):
            continue
        kind, base = d["id"].split("|", 1)
        won[base][kind] = d.get("result") == "won"
    none = {b: v["none"] for b, v in won.items() if "none" in v}
    kinds = sorted({k for v in won.values() for k in v if k != "none"})
    T, rows = {}, []
    for k in kinds:
        diffs = [int(v[k]) - int(none[b]) for b, v in won.items() if k in v and b in none]
        if len(diffs) < a.min_pairs:
            continue
        m = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / max(1, len(diffs) - 1))
        T[k] = round(100 * m, 3)
        rows.append((100 * m, 100 * sd / math.sqrt(len(diffs)), len(diffs), k))
    base = 100 * sum(none.values()) / max(1, len(none))
    print(f"entries {len(none)}, win rate with an empty belt {base:.1f}%; kinds {len(T)} (>= {a.min_pairs} pairs)")
    for m, se, n, k in sorted(rows, reverse=True):
        print(f"  {k:40s} {m:+6.1f} (se {se:.1f}, n {n})")
    json.dump(T, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("starts"); p.add_argument("outdir"); p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--entries", type=int, default=0, help="explicit number of final-boss entries to sample (0 = compute from --budget)")
    p.add_argument("--asc", type=int, default=10); p.add_argument("--kinds", default="all"); p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("results"); r.add_argument("out"); r.add_argument("--min-pairs", type=int, default=20)
    a = ap.parse_args()
    {"prep": cmd_prep, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
