"""Self-play learning curve by training time: maps each fight's model version to the trainer time at which that version
was saved (train_log.jsonl), then averages fights per 10-minute bucket. Usage: python3 curve_by_time.py RUN [RUN...]"""
import glob, json, sys, collections
for run in sys.argv[1:]:
    tl = [json.loads(l) for l in open(f"src/s2/runs/{run}/train_log.jsonl")]
    vt = {r["version"]: r["t"] for r in tl}
    b = collections.defaultdict(list)
    for f in glob.glob(f"src/s2/runs/{run}/fights/*.jsonl"):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r["result"] == "error": continue
            t = 0.0 if r["model_version"] <= 1 else vt.get(r["model_version"])
            if t is None: continue
            b[int(t // 600)].append(r)
    print(f"== {run}: {len(tl)} saves, last version {tl[-1]['version'] if tl else 0} at {tl[-1]['t'] if tl else 0:.0f} s")
    for k in sorted(b):
        x = b[k]; n = len(x)
        print(f"   trained {10*k:3d}-{10*k+10:3d} min: fights {n:6d}, won {100*sum(r['result']=='won' for r in x)/n:5.1f}%, "
              f"rel T_f {sum(r['rel'] for r in x)/n:.4f}, potions used {100*sum(r['potions_used'] for r in x)/max(1,sum(r['potions_start'] for r in x)):4.1f}%")
