"""T_f tables (fight-value tables) for M1 battle starts, from the S1 macro ensemble (PROPOSAL.md §4.2).

For each battle start (id = "<run_hash>:<floor g>") the state before the fight is rebuilt exactly as in the S1 dataset
(value_v1/build_floor_dataset.py build_run, with the models' own vocabulary, era of v0.111.0), as the example of floor g-1.
The post-fight state s_g(h, P) takes floor, act, room type and encounter of fight g, gold + the median gold reward of that
room type, HP = h and potion belt = P (every subset of the belt held at fight start, in the battle-start file's order).
T(0) = 0 (death); T = 1 for every h >= 1 if the fight is the final boss (act index 2, boss room); otherwise
T(h) = ensemble value = sigmoid(mean of the five members' logits) (no temperature: the raw ensemble was best calibrated).

Output: one JSON line per battle start: {"id", "max_hp", "belt", "subsets", "tables", "final_boss", "spread_pts"}.

Usage: python -I make_tf_tables.py <runs.jsonl.gz> <battle_starts.jsonl> <out.jsonl> <member_dir> [<member_dir> ...]
"""
import gzip, itertools, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "value_v1")); sys.path.insert(0, os.path.join(ROOT, "value_v2"))
import numpy as np
import orjson
import torch

import build_floor_dataset as bfd
import train_macro_v2 as tm
import train_value_v2 as v2

S = v2.S
GOLD = {"monster": 13.0, "elite": 32.0, "boss": 75.0}         # medians measured on validation fights (s1_final.json)
ROOM_IDX = {"monster": 3, "elite": 4, "boss": 5}
ERA_V0111 = 4                                                 # v0.111.0 is in era 4 (v0.108.0 - v0.111.0) of floors_v2
FIELDS = ("d_ids", "d_upg", "d_ench", "d_eamt", "d_age", "r_ids", "r_age", "p_ids", "ctx", "scal")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_members(dirs):
    ms, vocab = [], None
    for d in dirs:
        ck = torch.load(os.path.join(d, "model.pt"), map_location=dev, weights_only=False)
        special = ck["special"]; rest = [i for i, s in enumerate(special) if s.startswith("REST:")]
        m = tm.MacroNet(ck["vocab"], len(special), rest, arch=ck["config"]["arch"]).to(dev)
        m.load_state_dict(ck["state_dict"]); m.eval(); ms.append(m); vocab = ck["vocab"]
    return ms, vocab


@torch.no_grad()
def logits(ms, arr, bs=8192):
    n = len(arr["scal"]); out = np.zeros((len(ms), n), np.float32)
    for i in range(0, n, bs):
        b = {}
        for k in ("d_ids", "r_ids", "p_ids", "d_upg", "d_ench"):
            b[k] = torch.from_numpy(arr[k][i:i + bs].astype(np.int32)).to(dev)
        for k in ("d_eamt", "d_age", "r_age"):
            b[k] = torch.from_numpy(arr[k][i:i + bs]).to(dev)
        b["ctx"] = torch.from_numpy(arr["ctx"][i:i + bs].astype(np.int64)).to(dev)
        b["scal"] = torch.from_numpy(arr["scal"][i:i + bs].astype(np.float32)).to(dev)
        for j, m in enumerate(ms):
            with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
                lw, _ = m(b)
            out[j, i:i + bs] = lw.float().cpu().numpy()
    return out


def main():
    runs_path, starts_path, out_path, dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    t0 = time.time()
    ms, vocab = load_members(dirs)
    bfd.init_worker(vocab)
    starts = [json.loads(l) for l in open(starts_path)]
    want = {s["id"].split(":")[0] for s in starts}
    lines = {}
    with gzip.open(runs_path, "rb") as f:
        for line in f:
            if b'"run_hash"' in line:
                r = orjson.loads(line)
                if r.get("run_hash") in want:
                    lines[r["run_hash"]] = (line, r)
    print(f"{len(ms)} members on {dev}; {len(starts)} battle starts, {len(lines)} runs found ({time.time()-t0:.0f}s)", flush=True)
    examples = {}
    for h_, (line, r) in lines.items():
        p = r["players"][0]
        meta = {"sample_id": 0, "character": p["character"].replace("CHARACTER.", ""), "ascension": r["ascension"], "era": ERA_V0111}
        _, arr, *_ = bfd.build_run((line, meta))
        if arr:
            examples[h_] = arr
    out, skipped, t_rows = [], 0, 0
    for s in starts:
        run_hash, g = s["id"].split(":"); g = int(g)
        arr = examples.get(run_hash)
        if arr is None or s["room_type"] not in ROOM_IDX:
            skipped += 1; continue
        fl = arr["floor"]
        pre = np.flatnonzero(fl == g - 1)
        if not len(pre):
            skipped += 1; continue
        i = int(pre[0]); post = np.flatnonzero(fl == g)
        base = {k: arr[k][i:i + 1].copy() for k in FIELDS}
        mhp = int(s["max_hp"])
        belt = list(s["potions"]); btok = [bfd.tok("potion", x) for x in belt]
        subsets = [list(c) for r_ in range(len(belt) + 1) for c in itertools.combinations(range(len(belt)), r_)]
        rows = []
        for sub in subsets:
            st = {k: np.repeat(v, mhp, 0) for k, v in base.items()}
            st["ctx"][:, 3] = s["act_index"]; st["ctx"][:, 4] = ROOM_IDX[s["room_type"]]; st["ctx"][:, 5] = bfd.tok("encounter", s["encounter_id"])
            st["scal"][:, S["floor"]] = g / 50; st["scal"][:, S["act"]] = s["act_index"] / 3; st["scal"][:, S["is_combat"]] = 1.0
            st["scal"][:, S["floor_in_act"]] = arr["scal"][int(post[0]), S["floor_in_act"]] if len(post) else base["scal"][0, S["floor_in_act"]] + 1 / 20
            st["scal"][:, S["gold"]] = (s["gold"] + GOLD[s["room_type"]]) / 500
            st["scal"][:, S["max_hp"]] = mhp / 100
            h = np.arange(1, mhp + 1); st["scal"][:, S["hp"]] = h / 100; st["scal"][:, S["hp_frac"]] = h / mhp
            st["p_ids"][:] = 0
            for j, bi in enumerate(sub):
                st["p_ids"][:, j] = btok[bi]
            st["scal"][:, S["n_potions"]] = len(sub) / 5
            rows.append(st)
        big = {k: np.concatenate([r_[k] for r_ in rows]) for k in FIELDS}
        z = logits(ms, big); t_rows += len(z[0])
        pm = 1 / (1 + np.exp(-z))
        T = 1 / (1 + np.exp(-z.mean(0)))
        final = s["room_type"] == "boss" and s["act_index"] == 2
        tables = []
        for k in range(len(subsets)):
            t = T[k * mhp:(k + 1) * mhp]
            tables.append([0.0] + ([1.0] * mhp if final else [round(float(x), 5) for x in t]))
        out.append({"id": s["id"], "max_hp": mhp, "belt": belt, "subsets": subsets, "tables": tables, "final_boss": final,
                    "spread_pts": round(100 * float(pm.std(0).mean()), 2)})
    with open(out_path, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    # checks
    mono = all(all(t[i + 1] >= t[i] - 1e-9 for i in range(1, len(t) - 1)) for o in out for t in o["tables"])
    full = [o["tables"][-1] for o in out if not o["final_boss"]]       # full belt, non-final fights
    print(json.dumps({"written": len(out), "skipped": skipped, "states_evaluated": t_rows, "final_boss_fights": sum(o["final_boss"] for o in out),
                      "monotone_in_hp": mono, "mean_T_at_full_hp": round(float(np.mean([t[-1] for t in full])), 4),
                      "mean_T_at_half_hp": round(float(np.mean([t[len(t) // 2] for t in full])), 4),
                      "mean_member_spread_pts": round(float(np.mean([o["spread_pts"] for o in out])), 2),
                      "elapsed_s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
