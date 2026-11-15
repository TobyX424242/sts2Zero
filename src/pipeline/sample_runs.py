"""Pass 2: draw an evenly stratified sample of clean runs and extract their raw JSON lines.

Strata ("cases"):
  primary cell   = character (5) x ascension (0-10) x outcome (win / died_act1 / died_act2 / died_act3)
  secondary      = game-version era inside each cell (chronological build groups of similar size)
Quotas are spread evenly over cells, then evenly over eras inside each cell; when a stratum has
fewer runs than its quota, all of it is taken and the remainder is redistributed (water-filling).
Inside a stratum runs are drawn uniformly at random, with a cap on runs per player_token so a few
heavy uploaders cannot dominate. Each run carries an inverse-inclusion weight (population/sample of
its cell x era), so training can re-weight back to the population distribution if wanted.
Splits (train/val/test 80/10/10) are assigned by hashing player_token, or the seed when the run is
anonymous (all runs in the 2026-11-09 dump are), so runs sharing a seed never straddle splits.
Builds listed in --exclude-builds (default NON-RELEASE-VERSION) are dropped before sampling.

Usage:
  python -I sample_runs.py <runs.jsonl.gz> <meta_all.csv.gz> <out_dir>
         [--n 100000] [--eras 5] [--player-cap 30] [--seed 20261110]
"""
import argparse, collections, csv, gzip, hashlib, json, os, random, re, subprocess, time

import orjson

OUTCOMES = ["win", "died_act1", "died_act2", "died_act3"]
BLOCK = 32 << 20


def version_key(build):
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)(.*)$", build or "")
    if not m:
        return (9999, 0, 0, 1, build or "")
    major, minor, patch, rest = int(m[1]), int(m[2]), int(m[3]), m[4]
    return (major, minor, patch, 0 if rest else 1, rest)  # "-rc" sorts before the release


def waterfill(avail, total):
    """Spread `total` as evenly as possible over keys, never exceeding avail[k]."""
    alloc = {k: 0 for k in avail}
    active = sorted(k for k, v in avail.items() if v > 0)
    remaining = total
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        nxt = []
        for k in active:
            if remaining == 0:
                nxt.append(k)
                continue
            take = min(share, avail[k] - alloc[k], remaining)
            alloc[k] += take
            remaining -= take
            if alloc[k] < avail[k]:
                nxt.append(k)
        active = nxt
    return alloc


def make_eras(build_counts, k):
    """Split the chronologically sorted builds into k contiguous eras whose sizes are as equal as
    possible (minimum sum of squared deviations from total/k), never splitting a build."""
    builds = sorted(build_counts, key=version_key)
    n, total = len(builds), sum(build_counts.values())
    k = min(k, n)
    pre = [0]
    for b in builds:
        pre.append(pre[-1] + build_counts[b])
    target = total / k
    INF = float("inf")
    cost = [[INF] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    cost[0][0] = 0.0
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            for m in range(j - 1, i):
                c = cost[j - 1][m] + (pre[i] - pre[m] - target) ** 2
                if c < cost[j][i]:
                    cost[j][i], back[j][i] = c, m
    cuts, i = [], n
    for j in range(k, 0, -1):
        m = back[j][i]
        cuts.append((m, i))
        i = m
    eras = [builds[m:i] for m, i in reversed(cuts)]
    return {b: ei for ei, era in enumerate(eras) for b in era}, eras


def iter_raw_lines(path):
    proc = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE)
    rem = b""
    while True:
        chunk = proc.stdout.read(BLOCK)
        if not chunk:
            break
        buf = rem + chunk
        cut = buf.rfind(b"\n")
        if cut < 0:
            rem = buf
            continue
        for line in buf[:cut].split(b"\n"):
            if line.strip():
                yield line
        rem = buf[cut + 1:]
    if rem.strip():
        yield rem
    if proc.wait() != 0:
        raise RuntimeError("pigz failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("meta"); ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--eras", type=int, default=5)
    ap.add_argument("--player-cap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--exclude-builds", default="NON-RELEASE-VERSION",
                    help="comma-separated build_ids to drop (unknown content version)")
    a = ap.parse_args()
    drop_builds = {b for b in a.exclude_builds.split(",") if b}
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    t0 = time.time()

    # ---- load clean rows, count exclusions, dedupe
    excl = collections.Counter()
    seen_hash, seen_key, rows, dups = set(), set(), [], 0
    with gzip.open(a.meta, "rt", newline="") as f:
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
            rows.append({
                "idx": int(r["idx"]), "run_hash": r["run_hash"], "build_id": r["build_id"],
                "character": r["character"].replace("CHARACTER.", ""), "ascension": int(r["ascension"]),
                "outcome": r["outcome"], "win": int(r["win"]), "player_token": r["player_token"],
                "seed": r["seed"],
                "n_floors": int(r["n_floors"]), "n_acts": int(r["n_acts"]),
                "last_floor_type": r["last_floor_type"], "killed_by_encounter": r["killed_by_encounter"],
            })
    excl["duplicate"] = dups
    print(f"clean unique runs: {len(rows):,} (loaded in {time.time()-t0:.0f}s)", flush=True)

    # ---- eras
    build_counts = collections.Counter(r["build_id"] for r in rows)
    era_of, eras = make_eras(build_counts, a.eras)
    for r in rows:
        r["era"] = era_of[r["build_id"]]

    # ---- strata populations
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        cells[(r["character"], r["ascension"], r["outcome"])][r["era"]].append(r)
    cell_avail = {c: sum(len(v) for v in eras_.values()) for c, eras_ in cells.items()}
    cell_alloc = waterfill(cell_avail, a.n)

    # ---- select
    per_player = collections.Counter()
    chosen, strata_rows = [], []
    for c in sorted(cells, key=lambda c: (cell_avail[c], c)):  # scarce cells first
        era_avail = {e: len(v) for e, v in cells[c].items()}
        era_alloc = waterfill(era_avail, cell_alloc[c])
        for e in sorted(era_alloc):
            q, cand = era_alloc[e], list(cells[c][e])
            if q == 0:
                strata_rows.append((c, e, len(cand), 0))
                continue
            rng.shuffle(cand)
            pick, skipped = [], []
            for r in cand:
                if len(pick) == q:
                    break
                t = r["player_token"]
                if t and per_player[t] >= a.player_cap:
                    skipped.append(r)
                    continue
                pick.append(r)
                if t:
                    per_player[t] += 1
            for r in skipped[: q - len(pick)]:  # relax the cap only if the stratum would run short
                pick.append(r)
                per_player[r["player_token"]] += 1
            w = len(cand) / len(pick)
            for r in pick:
                r["weight"] = w
            chosen.extend(pick)
            strata_rows.append((c, e, len(cand), len(pick)))

    sw = sum(r["weight"] for r in chosen)
    for r in chosen:
        r["weight_norm"] = r["weight"] * len(chosen) / sw
        # group by player when known, else by seed (shared seeds must not straddle splits)
        g = r["player_token"] or (f"seed:{r['seed']}" if r["seed"] else f"run:{r['run_hash']}")
        h = int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100
        r["split"] = "train" if h < 80 else ("val" if h < 90 else "test")
    chosen.sort(key=lambda r: r["idx"])
    for i, r in enumerate(chosen):
        r["sample_id"] = i
    print(f"selected {len(chosen):,} runs ({time.time()-t0:.0f}s)", flush=True)

    # ---- index
    cols = ["sample_id", "idx", "run_hash", "seed", "build_id", "era", "character", "ascension", "outcome", "win",
            "n_acts", "n_floors", "last_floor_type", "killed_by_encounter", "player_token", "split",
            "weight", "weight_norm"]
    with open(os.path.join(a.out_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(chosen)
    with open(os.path.join(a.out_dir, "strata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["character", "ascension", "outcome", "era", "population", "sampled"])
        for (ch, asc, oc), e, pop, s in sorted(strata_rows):
            w.writerow([ch, asc, oc, e, pop, s])

    # ---- extract raw lines (second stream over the dump), checking alignment by run_hash
    want = {r["idx"]: r["run_hash"] for r in chosen}
    out = open(os.path.join(a.out_dir, "runs.jsonl.gz"), "wb")
    zp = subprocess.Popen(["pigz", "-c", "-6"], stdin=subprocess.PIPE, stdout=out)
    written, mismatch = 0, 0
    for i, line in enumerate(iter_raw_lines(a.src)):
        if i in want:
            if orjson.loads(line).get("run_hash") != want[i]:
                mismatch += 1
            zp.stdin.write(line + b"\n")
            written += 1
            if written % 20000 == 0:
                print(f"  extracted {written:,} ({time.time()-t0:.0f}s)", flush=True)
    zp.stdin.close(); zp.wait(); out.close()
    if mismatch or written != len(chosen):
        raise SystemExit(f"ALIGNMENT ERROR: written={written} expected={len(chosen)} mismatches={mismatch}")

    # ---- summary
    def tab(key):
        pop = collections.Counter(key(r) for r in rows)
        smp = collections.Counter(key(r) for r in chosen)
        return {str(k): {"population": pop[k], "sample": smp.get(k, 0)} for k in sorted(pop)}
    under = [c for c, pop in cell_avail.items() if cell_alloc[c] == pop]  # cells taken in full
    summary = {
        "source": os.path.basename(a.src), "n_requested": a.n, "n_selected": len(chosen),
        "seed": a.seed, "player_cap": a.player_cap, "exclusions": dict(excl),
        "clean_unique_population": len(rows),
        "eras": [{"era": i, "builds": f"{e[0]} .. {e[-1]}", "n_builds": len(e),
                  "population": sum(build_counts[b] for b in e)} for i, e in enumerate(eras)],
        "n_cells": len(cells), "cells_taken_in_full": len(under),
        "by_character": tab(lambda r: r["character"]),
        "by_ascension": tab(lambda r: r["ascension"]),
        "by_outcome": tab(lambda r: r["outcome"]),
        "by_era": tab(lambda r: r["era"]),
        "splits": dict(collections.Counter(r["split"] for r in chosen)),
        "players": {"distinct_tokens_in_sample": len({r["player_token"] for r in chosen if r["player_token"]}),
                    "anonymous_runs_in_sample": sum(not r["player_token"] for r in chosen),
                    "max_runs_one_player": max(per_player.values()) if per_player else 0,
                    "distinct_tokens_in_population": len({r["player_token"] for r in rows if r["player_token"]}),
                    "anonymous_runs_in_population": sum(not r["player_token"] for r in rows)},
        "weight_norm_range": [min(r["weight_norm"] for r in chosen), max(r["weight_norm"] for r in chosen)],
        "elapsed_s": round(time.time() - t0),
    }
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("n_selected", "clean_unique_population", "exclusions", "eras", "splits", "players")}, indent=1))


if __name__ == "__main__":
    main()
