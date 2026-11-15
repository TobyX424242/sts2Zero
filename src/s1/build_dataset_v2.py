"""S1 v2 dataset: per-floor examples from ALL clean runs at natural frequencies (PROPOSAL.md §6.3, item 5).

Differences from value_v1/build_floor_dataset.py (whose per-run rebuild, leak guards and sanity filter are reused
unchanged):
  - no balanced sample: every clean, unique run of the scan is used, read straight from the dump (streamed);
  - run selection, dedupe and era assignment follow pipeline/sample_runs.py; the split hashes the same group key
    (player token, else seed, else run hash), so runs keep the split they had in the 100k sample;
  - extra arrays: build index per example (builds.json), so the target cell (Defect, A10, v0.111.0) can be selected;
  - weight = 1 for every example (natural frequencies);
  - vocabulary threshold is --min-runs training runs (default 100; mod IDs are mapped to <unk>).

Usage: python -I build_dataset_v2.py <runs-latest.jsonl.gz> <meta_all.csv.gz> <out_dir> [--workers 24] [--min-runs 100]
       [--max-runs N]   (keep only the first N selected runs, for smoke tests)
"""
import argparse, collections, csv, gzip, hashlib, json, multiprocessing as mp, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "value_v1"))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import numpy as np

import build_floor_dataset as bfd          # reconstruct / build_run / count_ids / init_worker
from sample_runs import iter_raw_lines, make_eras, version_key


def select_runs(meta_path, drop_builds, n_eras):
    excl = collections.Counter()
    seen_hash, seen_key, rows, dups = set(), set(), [], 0
    with gzip.open(meta_path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["exclude"]:
                excl[r["exclude"]] += 1
                continue
            if r["build_id"] in drop_builds:
                excl["excluded_build"] += 1
                continue
            key = (r["seed"], r["character"], r["ascension"], r["start_time"])
            if r["run_hash"] in seen_hash or key in seen_key:
                dups += 1
                continue
            seen_hash.add(r["run_hash"]); seen_key.add(key)
            rows.append({"idx": int(r["idx"]), "run_hash": r["run_hash"], "build_id": r["build_id"],
                         "character": r["character"].replace("CHARACTER.", ""), "ascension": int(r["ascension"]),
                         "win": int(r["win"]), "player_token": r["player_token"], "seed": r["seed"]})
    excl["duplicate"] = dups
    build_counts = collections.Counter(r["build_id"] for r in rows)
    era_of, eras = make_eras(build_counts, n_eras)
    builds = sorted(build_counts, key=version_key)
    b_index = {b: i for i, b in enumerate(builds)}
    for r in rows:
        r["era"] = era_of[r["build_id"]]
        r["build"] = b_index[r["build_id"]]
        g = r["player_token"] or (f"seed:{r['seed']}" if r["seed"] else f"run:{r['run_hash']}")
        h = int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100
        r["split"] = "train" if h < 80 else ("val" if h < 90 else "test")
    rows.sort(key=lambda r: r["idx"])
    for i, r in enumerate(rows):
        r["sample_id"] = i
    return rows, excl, builds, eras


def stream_selected(src, by_idx):
    """Yield (line, meta) for selected dump lines, in dump order."""
    for i, line in enumerate(iter_raw_lines(src)):
        m = by_idx.get(i)
        if m is not None:
            yield line, m


def _count(args):
    line, m = args
    return m, bfd.count_ids(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("meta"); ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-runs", type=int, default=100)
    ap.add_argument("--eras", type=int, default=5)
    ap.add_argument("--exclude-builds", default="NON-RELEASE-VERSION")
    ap.add_argument("--max-runs", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    rows, excl, builds, eras = select_runs(a.meta, {b for b in a.exclude_builds.split(",") if b}, a.eras)
    if a.max_runs:
        rows = rows[:a.max_runs]
    by_idx = {r["idx"]: r for r in rows}
    print(f"selected {len(rows):,} clean unique runs ({time.time()-t0:.0f}s); exclusions {dict(excl)}", flush=True)
    with open(os.path.join(a.out_dir, "index.csv"), "w", newline="") as f:
        cols = ["sample_id", "idx", "run_hash", "seed", "build_id", "build", "era", "character", "ascension", "win", "split"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    # ---- pass 1: vocabulary from training runs only
    counts = collections.defaultdict(collections.Counter)
    with mp.Pool(a.workers) as pool:
        n = 0
        for m, ids in pool.imap(_count, (x for x in stream_selected(a.src, by_idx) if x[1]["split"] == "train"), chunksize=256):
            for k, v in ids.items():
                counts[k].update(v)
            n += 1
            if n % 200_000 == 0:
                print(f"  vocab pass: {n:,} train runs ({time.time()-t0:.0f}s)", flush=True)
    vocab = {k: ["<pad>", "<unk>"] + sorted(i for i, c in counts[k].items() if c >= a.min_runs)
             for k in ("card", "ench", "relic", "encounter", "potion")}
    vocab_stats = {k: {"kept": len(vocab[k]) - 2, "seen": len(counts[k])} for k in vocab}
    assert len(vocab["ench"]) < 256 and len(vocab["card"]) < 65535
    print(f"vocab {vocab_stats} ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass 2: examples
    parts = collections.defaultdict(list)
    run_of, build_of, n_used = [], [], 0
    miss_tot, miss_runs, insane, dropped_future = collections.Counter(), 0, collections.Counter(), 0
    start_decks = collections.defaultdict(collections.Counter)
    with mp.Pool(a.workers, initializer=bfd.init_worker, initargs=(vocab,)) as pool:
        for k, (sid, arr, miss, sdeck, dropped, F, bad) in enumerate(
                pool.imap(bfd.build_run, stream_selected(a.src, by_idx), chunksize=64)):
            if (k + 1) % 200_000 == 0:
                print(f"  example pass: {k+1:,} runs ({time.time()-t0:.0f}s)", flush=True)
            if bad:
                insane[bad] += 1
                continue
            miss_tot.update(miss); miss_runs += bool(miss); dropped_future += dropped
            m = rows[sid]
            start_decks[(m["character"], m["ascension"])][tuple(sdeck)] += 1
            if not arr:
                continue
            n_used += 1
            ne = len(arr["win"])
            for key, v in arr.items():
                parts[key].append(v)
            run_of.append(np.full(ne, sid, np.int32))
            build_of.append(np.full(ne, m["build"], np.int16))
    print(f"built {n_used:,} runs; concatenating ({time.time()-t0:.0f}s)", flush=True)
    split_code = {"train": 0, "val": 1, "test": 2}
    run = np.concatenate(run_of); del run_of
    run_split = np.array([split_code[r["split"]] for r in rows], np.int8)
    np.save(os.path.join(a.out_dir, "run.npy"), run)
    np.save(os.path.join(a.out_dir, "split.npy"), run_split[run])
    np.save(os.path.join(a.out_dir, "weight.npy"), np.ones(len(run), np.float32))
    np.save(os.path.join(a.out_dir, "build.npy"), np.concatenate(build_of)); del build_of
    n_examples, by_split = len(run), {s: int((run_split[run] == i).sum()) for s, i in split_code.items()}
    del run
    for key in list(parts):
        np.save(os.path.join(a.out_dir, f"{key}.npy"), np.concatenate(parts.pop(key)))
    mode_hits = sum(c.most_common(1)[0][1] for c in start_decks.values())
    total = sum(sum(c.values()) for c in start_decks.values())
    stats = {"runs_selected": len(rows), "runs_with_examples": n_used, "examples": n_examples,
             "examples_by_split": by_split, "exclusions": dict(excl), "vocab": vocab_stats, "min_runs": a.min_runs,
             "insane_runs_dropped": dict(insane), "undo_misses": dict(miss_tot), "runs_with_any_miss": miss_runs,
             "future_cards_dropped": dropped_future, "start_deck_matches_mode": round(mode_hits / max(1, total), 4),
             "eras": eras, "scalars": bfd.SCALARS, "room_types": bfd.ROOM_TYPES, "characters": bfd.CHARS,
             "ctx_fields": ["character", "ascension", "era", "act", "room_type", "encounter"],
             "elapsed_s": round(time.time() - t0)}
    json.dump(vocab, open(os.path.join(a.out_dir, "vocab.json"), "w"))
    json.dump(builds, open(os.path.join(a.out_dir, "builds.json"), "w"))
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1, default=str)
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
