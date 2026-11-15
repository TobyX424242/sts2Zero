"""Independent checks of a sample directory: re-parse every run in runs.jsonl.gz and compare with
index.csv; check filters, balance and split leakage.  Usage: python -I verify_sample.py <sample_dir>"""
import collections, csv, gzip, os, re, statistics, sys

import orjson

d = sys.argv[1]
idx = list(csv.DictReader(open(os.path.join(d, "index.csv"))))
OFFICIAL = {"IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"}
MOD_RE = re.compile(rb'"(?:CARD|RELIC|POTION|CHARACTER|ENCOUNTER|MONSTER|EVENT|ENCHANTMENT|POWER|ORB|AFFLICTION|ACT)\.[A-Za-z0-9_]+-')
problems = collections.Counter()
n = 0
with gzip.open(os.path.join(d, "runs.jsonl.gz"), "rb") as f:
    for i, line in enumerate(f):
        n += 1
        r = orjson.loads(line)
        m = idx[i]
        p = r["players"]
        ch = p[0]["character"].replace("CHARACTER.", "") if p else ""
        mph = r.get("map_point_history") or []
        outcome = "win" if r.get("win") else f"died_act{min(max(len(mph), 1), 3)}"
        if r.get("run_hash") != m["run_hash"]: problems["run_hash_mismatch"] += 1
        if ch != m["character"]: problems["character_mismatch"] += 1
        if str(r.get("ascension")) != m["ascension"]: problems["ascension_mismatch"] += 1
        if outcome != m["outcome"]: problems["outcome_mismatch"] += 1
        if r.get("build_id") != m["build_id"]: problems["build_mismatch"] += 1
        if len(p) != 1: problems["multiplayer"] += 1
        if r.get("game_mode") != "standard": problems["not_standard"] += 1
        if r.get("modifiers"): problems["modifiers"] += 1
        if r.get("was_abandoned"): problems["abandoned"] += 1
        if ch not in OFFICIAL: problems["unofficial_character"] += 1
        if MOD_RE.search(line): problems["modded_content"] += 1
        if not any(mph): problems["no_floors"] += 1
print(f"records in runs.jsonl.gz: {n:,}; rows in index.csv: {len(idx):,}")
print("record-level problems:", dict(problems) or "none")
print("unique run_hash:", len({m['run_hash'] for m in idx}) == len(idx))
cells = collections.Counter((m["character"], m["ascension"], m["outcome"]) for m in idx)
print(f"cells: {len(cells)}; per-cell count min={min(cells.values())} max={max(cells.values())}")
ce = collections.Counter((m["character"], m["ascension"], m["outcome"], m["era"]) for m in idx)
vals = list(ce.values())
print(f"cell x era strata: {len(ce)}; count min={min(vals)} median={statistics.median(vals)} max={max(vals)}")
seed_splits = collections.defaultdict(set)
for m in idx:
    seed_splits[m["seed"]].add(m["split"])
print("seeds appearing in >1 split:", sum(len(s) > 1 for s in seed_splits.values()))
print("seeds shared by >1 sampled run:", sum(v > 1 for v in collections.Counter(m['seed'] for m in idx).values()))
for col in ("character", "ascension", "outcome", "era", "split"):
    c = collections.Counter(m[col] for m in idx)
    print(f"{col:10s}", dict(sorted(c.items(), key=lambda kv: (len(kv[0]), kv[0]))))
fl = [int(m["n_floors"]) for m in idx]
print(f"floors per run: mean={statistics.mean(fl):.1f} median={statistics.median(fl)} total floor records={sum(fl):,}")
print(f"file size: {os.path.getsize(os.path.join(d, 'runs.jsonl.gz'))/1e9:.2f} GB gz")
