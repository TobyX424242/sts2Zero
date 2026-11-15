"""T_f tables for the S2 battle-start pool, batched (same construction as src/s0/make_tf_tables.py, whose
functions are reused: post-fight state s_g(h, P) for every HP h and held-potion subset P, valued by the S1 macro
ensemble; T(0) = 0; T = 1 at the final boss).

Differences from the M1 script: rows of many battle starts are valued in one GPU call (--rows-per-call), the run
records are found in one pass over the dump, and the output is written incrementally.

Usage: ./.venv/bin/python -I src/s2/make_tables.py <runs.jsonl.gz> <starts.jsonl> <out.jsonl> <member_dir>...
Env TABLE_BS: rows per GPU call (default 8192; lower it when shards share a GPU).
Env SHARD=i/N: only battle starts with line index % N == i (run N shards in parallel and concatenate).
"""
import gzip, itertools, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "s0"))
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import make_tf_tables as m1  # noqa: E402  (sets up the S1 imports)

bfd, S, FIELDS, GOLD, ROOM_IDX, ERA = m1.bfd, m1.S, m1.FIELDS, m1.GOLD, m1.ROOM_IDX, m1.ERA_V0111
ROWS_PER_CALL = 400_000


def rows_for(s, arr):
    """Rows (dict of arrays) of one battle start, or None if its pre-fight floor is missing."""
    g = int(s["id"].split(":")[1])
    fl = arr["floor"]
    pre = np.flatnonzero(fl == g - 1)
    if not len(pre) or s["room_type"] not in ROOM_IDX:
        return None
    i = int(pre[0]); post = np.flatnonzero(fl == g)
    base = {k: arr[k][i:i + 1] for k in FIELDS}
    mhp = int(s["max_hp"])
    belt = list(s["potions"]); btok = [bfd.tok("potion", x) for x in belt]
    subsets = [list(c) for r_ in range(len(belt) + 1) for c in itertools.combinations(range(len(belt)), r_)]
    K = len(subsets)
    st = {k: np.repeat(v, mhp * K, 0) for k, v in base.items()}
    st["ctx"][:, 3] = s["act_index"]; st["ctx"][:, 4] = ROOM_IDX[s["room_type"]]; st["ctx"][:, 5] = bfd.tok("encounter", s["encounter_id"])
    sc = st["scal"]
    sc[:, S["floor"]] = g / 50; sc[:, S["act"]] = s["act_index"] / 3; sc[:, S["is_combat"]] = 1.0
    sc[:, S["floor_in_act"]] = arr["scal"][int(post[0]), S["floor_in_act"]] if len(post) else base["scal"][0, S["floor_in_act"]] + 1 / 20
    sc[:, S["gold"]] = (s["gold"] + GOLD[s["room_type"]]) / 500
    sc[:, S["max_hp"]] = mhp / 100
    h = np.tile(np.arange(1, mhp + 1), K); sc[:, S["hp"]] = h / 100; sc[:, S["hp_frac"]] = h / mhp
    st["p_ids"][:] = 0
    for k, sub in enumerate(subsets):
        for j, bi in enumerate(sub):
            st["p_ids"][k * mhp:(k + 1) * mhp, j] = btok[bi]
        sc[k * mhp:(k + 1) * mhp, S["n_potions"]] = len(sub) / 5
    return st, subsets, mhp, belt


def main():
    runs_path, starts_path, out_path, dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    t0 = time.time()
    ms, vocab = m1.load_members(dirs)
    bfd.init_worker(vocab)
    si, sn = map(int, os.environ.get("SHARD", "0/1").split("/"))
    starts = [json.loads(l) for i, l in enumerate(open(starts_path)) if i % sn == si]
    by_run = {}
    for s in starts:
        by_run.setdefault(s["id"].split(":")[0], []).append(s)
    lines = {}
    with gzip.open(runs_path, "rb") as f:
        for line in f:
            if b"v0.111.0" not in line or b"DEFECT" not in line:
                continue
            r = orjson.loads(line)
            if r.get("run_hash") in by_run and r["run_hash"] not in lines:
                lines[r["run_hash"]] = (line, r)
    print(f"{len(ms)} members on {m1.dev}; {len(starts)} battle starts in {len(by_run)} runs, {len(lines)} runs found "
          f"({time.time() - t0:.0f}s)", flush=True)
    out = open(out_path, "w")
    pend, pend_rows = [], 0
    stats = {"written": 0, "skipped": 0, "rows": 0, "final_boss": 0, "nonmonotone": 0}

    def flush():
        nonlocal pend, pend_rows
        if not pend:
            return
        big = {k: np.concatenate([p[1][k] for p in pend]) for k in FIELDS}
        z = m1.logits(ms, big, bs=int(os.environ.get("TABLE_BS", "8192")))
        T = 1 / (1 + np.exp(-z.mean(0)))
        spread = (1 / (1 + np.exp(-z))).std(0)
        o = 0
        for s, st, subsets, mhp, belt in pend:
            n = len(subsets) * mhp
            t, sp = T[o:o + n], spread[o:o + n]; o += n
            final = s["room_type"] == "boss" and s["act_index"] == 2
            tables = []
            for k in range(len(subsets)):
                tk = t[k * mhp:(k + 1) * mhp]
                if not final and np.any(np.diff(tk) < -1e-6):
                    stats["nonmonotone"] += 1
                tables.append([0.0] + ([1.0] * mhp if final else [round(float(x), 5) for x in tk]))
            stats["final_boss"] += final
            out.write(json.dumps({"id": s["id"], "max_hp": mhp, "belt": belt, "subsets": subsets, "tables": tables,
                                  "final_boss": final, "spread_pts": round(100 * float(sp.mean()), 2)}) + "\n")
            stats["written"] += 1
        stats["rows"] += o
        pend, pend_rows = [], 0
        print(f"[tables] {stats['written']} written, {stats['rows']} rows, {time.time() - t0:.0f}s", flush=True)

    for h_, ss in by_run.items():
        if h_ not in lines:
            stats["skipped"] += len(ss); continue
        line, r = lines[h_]
        p = r["players"][0]
        meta = {"sample_id": 0, "character": p["character"].replace("CHARACTER.", ""), "ascension": r["ascension"], "era": ERA}
        _, arr, *_ = bfd.build_run((line, meta))
        if not arr:
            stats["skipped"] += len(ss); continue
        for s in ss:
            got = rows_for(s, arr)
            if got is None:
                stats["skipped"] += 1; continue
            st, subsets, mhp, belt = got
            pend.append((s, st, subsets, mhp, belt)); pend_rows += len(subsets) * mhp
            if pend_rows >= ROWS_PER_CALL:
                flush()
    flush()
    out.close()
    stats["elapsed_s"] = round(time.time() - t0)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
