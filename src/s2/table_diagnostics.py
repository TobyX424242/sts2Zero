"""Diagnostics of the macro -> combat signal on held-out battle starts (laptop, light):
(a) free disposal: for every table pair P and P + one potion, how often T(P + p) < T(P) (holding a potion valued below
    not holding it), at all HP values; (b) per-member prices: HP price T(full HP) - T(half HP) and potion price
    T(all potions) - T(none) at full HP, for each of the 5 ensemble members, and how much the members disagree.
Usage: ./.venv/bin/python -I src/s2/table_diagnostics.py <tables.jsonl> <starts.jsonl> <runs.jsonl.gz> <n> <member_dir>...
"""
import gzip, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import numpy as np, orjson
import make_tables as mt

tables_p, starts_p, runs_p, n, dirs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
# (a) from the ensemble tables
viol = tot = 0; worst = []; starts_with = 0
for line in open(tables_p):
    t = json.loads(line)
    if t["final_boss"]:
        continue
    idx = {tuple(s): k for k, s in enumerate(t["subsets"])}
    any_v = False
    for s, k in idx.items():
        for p in range(len(t["belt"])):
            if p in s: continue
            k2 = idx[tuple(sorted(s + (p,)))]
            a, b = np.array(t["tables"][k][1:]), np.array(t["tables"][k2][1:])
            d = b - a
            tot += len(d); v = d < -0.005; viol += v.sum()
            if v.any(): any_v = True; worst.append(d.min())
    starts_with += any_v
print(f"(a) free disposal: {viol}/{tot} table entries ({100*viol/max(1,tot):.2f}%) value an extra potion more than 0.5 point below "
      f"not having it; {starts_with} battle starts affected; worst {100*min(worst) if worst else 0:.1f} points")
# (b) per-member prices on n held-out starts
ms, vocab = mt.m1.load_members(dirs); mt.bfd.init_worker(vocab)
starts = [json.loads(l) for l, _ in zip(open(starts_p), range(10**9))]
starts = [s for s in starts if len(s["potions"]) >= 1 and not (s["room_type"] == "boss" and s["act_index"] == 2)][:n]
want = {s["id"].split(":")[0] for s in starts}; lines = {}
with gzip.open(runs_p, "rb") as f:
    for line in f:
        if b"v0.111.0" in line and b"DEFECT" in line:
            r = orjson.loads(line)
            if r.get("run_hash") in want: lines[r["run_hash"]] = (line, r)
hp_p, pot_p, lvl = [], [], []
for s in starts:
    h = s["id"].split(":")[0]
    if h not in lines: continue
    line, r = lines[h]; p = r["players"][0]
    _, arr, *_ = mt.bfd.build_run((line, {"sample_id": 0, "character": "DEFECT", "ascension": r["ascension"], "era": mt.ERA}))
    got = mt.rows_for(s, arr) if arr else None
    if got is None: continue
    st, subsets, mhp, belt = got
    z = mt.m1.logits(ms, st); T = 1 / (1 + np.exp(-z))           # [members, K * mhp]
    K = len(subsets); T = T.reshape(len(ms), K, mhp)
    full, half = mhp - 1, mhp // 2 - 1
    hp_p.append(T[:, -1, full] - T[:, -1, half]); pot_p.append(T[:, -1, full] - T[:, 0, full]); lvl.append(T[:, -1, full])
hp_p, pot_p, lvl = np.array(hp_p), np.array(pot_p), np.array(lvl)   # [starts, members]
def rep(name, x):
    m = x.mean(1); sd = x.std(1)
    sign = ((x > 0).all(1) | (x < 0).all(1)).mean()
    print(f"    {name}: mean {100*m.mean():.2f} pts; member std {100*sd.mean():.2f} pts (median ratio std/|mean| {np.median(sd/np.maximum(1e-4,np.abs(m))):.2f}); "
          f"all 5 members agree on the sign in {100*sign:.0f}% of starts")
print(f"(b) {len(hp_p)} held-out starts with potions (not the final boss), 5 members:")
rep("price of HP, full vs half HP", hp_p); rep("price of the whole belt at full HP", pot_p); rep("level T(full HP, all potions)", lvl)
