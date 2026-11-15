"""Pass 1: stream a Spire Codex runs export (.jsonl.gz) and write one metadata row per line,
including a single exclusion reason (empty = clean single-player official run).

Usage:
  python -I scan_runs.py <runs.jsonl.gz> <out_meta.csv.gz> [--workers N]

Line numbering: `idx` counts non-empty lines from 0. sample_runs.py relies on the same rule.
"""
import argparse, csv, gzip, multiprocessing as mp, re, subprocess, sys, threading, time

import orjson

OFFICIAL = {"CHARACTER.IRONCLAD", "CHARACTER.SILENT", "CHARACTER.DEFECT",
            "CHARACTER.NECROBINDER", "CHARACTER.REGENT"}
# Modded content uses a namespaced id with a hyphen, e.g. CARD.ANCIENTSAWAKENED-COSMIC_STRIKE
MOD_RE = re.compile(rb'"(?:CARD|RELIC|POTION|CHARACTER|ENCOUNTER|MONSTER|EVENT|ENCHANTMENT|POWER|ORB|AFFLICTION|ACT)\.[A-Za-z0-9_]+-')
FIELDS = ["idx", "exclude", "run_hash", "build_id", "schema_version", "game_mode", "n_players",
          "has_modifiers", "abandoned", "win", "ascension", "character", "player_token", "seed",
          "start_time", "run_time", "n_acts", "n_floors", "last_floor_type", "outcome",
          "killed_by_encounter", "modded", "line_bytes"]
BLOCK = 32 << 20  # 32 MiB of decompressed text per task


def iter_lines(block):
    for line in block.split(b"\n"):
        if line.strip():
            yield line


def parse_line(line):
    row = dict.fromkeys(FIELDS, "")
    row["line_bytes"] = len(line)
    row["modded"] = int(MOD_RE.search(line) is not None)
    try:
        r = orjson.loads(line)
    except orjson.JSONDecodeError:
        row["exclude"] = "parse_error"
        return row
    players = r.get("players") or []
    p0 = players[0] if players else {}
    mph = r.get("map_point_history") or []
    n_floors = sum(len(a) for a in mph if isinstance(a, list))
    last_type = ""
    for act in reversed(mph):
        if act:
            last_type = act[-1].get("map_point_type", "")
            break
    win = bool(r.get("win"))
    n_acts = len(mph)
    row.update(
        run_hash=r.get("run_hash", ""), build_id=r.get("build_id", ""),
        schema_version=r.get("schema_version", ""), game_mode=r.get("game_mode", ""),
        n_players=len(players), has_modifiers=int(bool(r.get("modifiers"))),
        abandoned=int(bool(r.get("was_abandoned"))), win=int(win),
        ascension=r.get("ascension", ""), character=p0.get("character", ""),
        player_token=r.get("player_token") or "", seed=r.get("seed", ""),
        start_time=r.get("start_time", ""), run_time=r.get("run_time", ""),
        n_acts=n_acts, n_floors=n_floors, last_floor_type=last_type,
        outcome="win" if win else f"died_act{min(max(n_acts, 1), 3)}",
        killed_by_encounter=r.get("killed_by_encounter", ""),
    )
    asc = r.get("ascension")
    if r.get("game_mode") != "standard":
        row["exclude"] = "not_standard_mode"
    elif row["has_modifiers"]:
        row["exclude"] = "has_modifiers"
    elif len(players) != 1:
        row["exclude"] = "multiplayer"
    elif row["character"] not in OFFICIAL:
        row["exclude"] = "unofficial_character"
    elif row["modded"]:
        row["exclude"] = "modded_content"
    elif row["abandoned"]:
        row["exclude"] = "abandoned"
    elif not isinstance(asc, int) or not 0 <= asc <= 10:
        row["exclude"] = "ascension_out_of_range"
    elif n_floors == 0:
        row["exclude"] = "no_floors"
    elif win and n_acts < 3:
        row["exclude"] = "inconsistent_win"
    return row


def process_block(block):
    return [parse_line(line) for line in iter_lines(block)]


def read_blocks(path, sem):
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
        sem.acquire()
        yield buf[: cut + 1]
        rem = buf[cut + 1:]
    if rem.strip():
        sem.acquire()
        yield rem
    if proc.wait() != 0:
        raise RuntimeError(f"pigz exited with {proc.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    a = ap.parse_args()
    sem = threading.Semaphore(a.workers * 2)  # bound in-flight blocks (Pool.imap has no backpressure)
    t0, idx, nbytes, counts = time.time(), 0, 0, {}
    with mp.Pool(a.workers) as pool, gzip.open(a.out, "wt", newline="", compresslevel=6) as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for bi, rows in enumerate(pool.imap(process_block, read_blocks(a.src, sem))):
            sem.release()
            for row in rows:
                row["idx"] = idx
                idx += 1
                nbytes += row["line_bytes"]
                counts[row["exclude"] or "clean"] = counts.get(row["exclude"] or "clean", 0) + 1
                w.writerow(row)
            if bi % 20 == 0:
                el = time.time() - t0
                print(f"[{el:7.0f}s] lines={idx:,} raw={nbytes/1e9:.1f} GB ({nbytes/1e6/max(el,1e-9):.0f} MB/s)", flush=True)
    print(f"done: {idx:,} lines, {nbytes/1e9:.2f} GB raw, {time.time()-t0:.0f}s")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:>10,}")


if __name__ == "__main__":
    main()
