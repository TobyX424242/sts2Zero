"""Human reference on the held-out evaluation starts: the logged human's fight end (HP after floor g, potions of the
starting belt still held after floor g, death if the run ended on floor g without a win) scored with the same T_f
tables the agent is scored by (free disposal applied, relative to the table maximum). Caveats: HP after the floor
includes post-fight healing (relics such as Burning Blood); a belt potion drunk and re-gained as a reward counts as
held; the human fought the real fight, the agent a simulated one with its own draws (same in expectation).
Usage: ./.venv/bin/python -I src/s2/human_reference.py <dir with starts.jsonl tables.jsonl and agent arm jsonl> <runs.jsonl.gz> [arms...]
"""
import gzip, json, math, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "..", "value_v1"))
import orjson
from build_floor_dataset import reconstruct

d, runs_path, arms = sys.argv[1], sys.argv[2], sys.argv[3:] or ["net", "s64", "ex64"]
starts = {s["id"]: s for s in map(json.loads, open(f"{d}/starts.jsonl"))}
tables = {t["id"]: t for t in map(json.loads, open(f"{d}/tables.jsonl"))}
for t in tables.values():  # free disposal, as TfScorer
    for k, sk in enumerate(t["subsets"]):
        for j, sj in enumerate(t["subsets"]):
            if len(sj) < len(sk) and set(sj) <= set(sk):
                t["tables"][k] = [max(a, b) if h > 0 else a for h, (a, b) in enumerate(zip(t["tables"][k], t["tables"][j]))]
want = {i.split(":")[0] for i in starts}
runs = {}
with gzip.open(runs_path, "rb") as f:
    for line in f:
        if b"v0.111.0" in line and b"DEFECT" in line:
            r = orjson.loads(line)
            if r.get("run_hash") in want: runs[r["run_hash"]] = r
human, skipped = {}, 0
for sid, s in starts.items():
    h, g = sid.split(":"); g = int(g); r = runs.get(h); t = tables.get(sid)
    if r is None or t is None: skipped += 1; continue
    floors, snaps, _, _ = reconstruct(r)
    gi = g - 1
    if gi >= len(floors): skipped += 1; continue
    died = gi == len(floors) - 1 and not r.get("win")
    hp = 0 if died else int(((floors[gi][1].get("player_stats") or [{}])[0]).get("current_hp") or 0)
    after = Counter(p for p in snaps[gi][2] if p)
    held = []
    for i, p in enumerate(s["potions"]):
        if after[p] > 0: after[p] -= 1; held.append(i)
    k = next(k for k, sub in enumerate(t["subsets"]) if sorted(sub) == held)
    tab = t["tables"][k]; tmax = max(max(x) for x in t["tables"])
    rel = 0.0 if hp <= 0 else tab[min(hp, len(tab) - 1)] / max(1e-3, tmax)
    human[sid] = {"rel": rel, "won": hp > 0, "pot_used": len(s["potions"]) - len(held), "pot_start": len(s["potions"])}
json.dump(human, open(f"{d}/human.json", "w"))
n = len(human)
print(f"human fights scored {n} (skipped {skipped}): won {100*sum(x['won'] for x in human.values())/n:.1f}%, "
      f"mean rel T_f {sum(x['rel'] for x in human.values())/n:.4f}, potions used {100*sum(x['pot_used'] for x in human.values())/max(1,sum(x['pot_start'] for x in human.values())):.1f}%")
for arm in arms:
    A = {x["id"]: x for x in map(json.loads, open(f"{d}/{arm}.jsonl")) if x["result"] != "error"}
    ids = sorted(set(A) & set(human)); m_ = len(ids)
    dif = [A[i]["rel"] - human[i]["rel"] for i in ids]; m = sum(dif) / m_; sd = math.sqrt(sum((x - m) ** 2 for x in dif) / (m_ - 1))
    print(f"{arm:5s} vs human on {m_} starts: agent won {100*sum(A[i]['result']=='won' for i in ids)/m_:.1f}% vs {100*sum(human[i]['won'] for i in ids)/m_:.1f}%, "
          f"rel {sum(A[i]['rel'] for i in ids)/m_:.4f} vs {sum(human[i]['rel'] for i in ids)/m_:.4f}, diff {m:+.4f} (95% CI {m-1.96*sd/math.sqrt(m_):+.4f} to {m+1.96*sd/math.sqrt(m_):+.4f})")
