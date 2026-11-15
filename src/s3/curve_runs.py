"""Learning curve of S3 self-play from the run and fight logs of one or more run directories (segments), in play order.

Runs are grouped into --buckets groups of equal size by the order in which they finished (segments in name order);
perturbed and errored runs are left out. Per group and ascension level (target level and "others"): runs, win rate,
acts cleared, floors, elites fought, rest share, and fights won by room type.

Then one line per directory for the target level and the level below it: the stages of a run (act 1 cleared, act 2
cleared of those, last-act runs that reach the boss, first boss won, second boss won at a double-boss level), where
last-act runs die, and from the fight logs read: potions used per last-act hallway fight and the HP share at the start
of last-act elite fights. Added 2026-11-14: the win rate alone hid that at A10 the share of act-3 runs reaching the boss
fell from 73% to 55% over four segments while both boss win rates rose.

Usage: python3 research/s3/curve_runs.py RUN_DIR [RUN_DIR ...] [--buckets 8] [--asc 10] [--fight-files 3]
"""
import argparse, glob, json, os


def read(p):
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def mean(x):
    return sum(x) / max(1, len(x))


DOUBLE_BOSS_ASC = 10    # AscensionLevel.DoubleBoss: two bosses at the end of the last act


def kind(r):
    kb = r.get("killed_by") or ""
    return "boss" if kb.endswith("_BOSS") else "elite" if kb.endswith("_ELITE") else "hallway" if kb.startswith("ENCOUNTER.") else "other"


def stages(name, runs, fights, asc):
    """One line: stage shares of the unperturbed runs of one directory at one ascension level."""
    R = [r for r in runs if r["ascension"] == asc]
    if len(R) < 50:
        return None
    pct = lambda a, b: f"{100 * len(a) / max(1, len(b)):.1f}%"
    c1 = [r for r in R if r["act"] >= 1 or r["won"]]
    c2 = [r for r in R if r["act"] >= 2 or r["won"]]
    fb = [r for r in c2 if r.get("bosses", 0) >= 3]
    b1 = [r for r in fb if r.get("bosses_won", 0) >= 3]
    won = [r for r in R if r["won"]]
    double = asc >= DOUBLE_BOSS_ASC
    lost3 = [r for r in c2 if not r["won"]]
    d = {k: [r for r in lost3 if kind(r) == k] for k in ("hallway", "elite", "other", "boss")}
    dboss2 = [r for r in d["boss"] if double and r.get("bosses_won", 0) >= 3]
    line = (f"   {name} A{asc}: runs {len(R)}  act 1 cleared {pct(c1, R)}  act 2 cleared of those {pct(c2, c1)}  act-3 runs reaching the boss {pct(fb, c2)}  "
            f"boss 1 won {pct(b1, fb)}" + (f"  boss 2 won {pct(won, b1)}" if double else "") + f"  runs won {100 * len(won) / len(R):.2f}%  "
            f"act-3 runs lost at: hallway {pct(d['hallway'], c2)} elite {pct(d['elite'], c2)} other {pct(d['other'], c2)} "
            f"boss 1 {100 * (len(d['boss']) - len(dboss2)) / max(1, len(c2)):.1f}%" + (f" boss 2 {pct(dboss2, c2)}" if double else ""))
    hall, elite = [], []
    for r in R:
        fs = sorted((f for f in fights.get(r["seed"], []) if not f.get("practice") and not f.get("error")),
                    key=lambda f: int(str(f.get("id", "0:0")).rsplit(":", 1)[-1]))
        bw = 0
        for f in fs:
            if bw >= 2 and f["room_type"] == "monster":
                hall.append(f.get("potions_used", 0))
            if bw >= 2 and f["room_type"] == "elite":
                elite.append(f["start_hp"] / max(1, f["max_hp"]))
            if f["room_type"] == "boss" and f["result"] == "won":
                bw += 1
    if len(hall) >= 100:
        line += f"  | act-3 hallway potions per fight {mean(hall):.2f}" + (f", HP share at act-3 elite start {mean(elite):.2f}" if len(elite) >= 50 else "")
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--buckets", type=int, default=8); ap.add_argument("--asc", type=int, default=-1)
    ap.add_argument("--fight-files", type=int, default=3, help="fight logs read per directory (one per actor; they are large: "
                    "fight statistics come from these actors' runs only; 0 = all)")
    a = ap.parse_args()
    runs, fights, by_dir = [], {}, {}
    for d in sorted(a.dirs):
        seg = []
        for f in glob.glob(os.path.join(d, "runs", "*.jsonl")):
            seg += read(f)
        seg.sort(key=lambda r: (r.get("macro_version", 0), r.get("model_version", 0)))   # finish order is not logged: order by versions
        runs += seg
        by_dir[os.path.basename(os.path.normpath(d))] = [r for r in seg if not r.get("error") and not r.get("perturbed")]
        ff = sorted(glob.glob(os.path.join(d, "fights", "*.jsonl")))
        for f in (ff[:a.fight_files] if a.fight_files > 0 else ff):
            for x in read(f):
                fights.setdefault(x.get("run_seed"), []).append(x)
    runs = [r for r in runs if not r.get("error") and not r.get("perturbed")]
    if not runs:
        print("no runs"); return
    asc = a.asc if a.asc >= 0 else max(r["ascension"] for r in runs)
    B = max(1, min(a.buckets, len(runs) // 50))
    print(f"{len(runs)} runs (errors and perturbed runs excluded), {B} groups in training order; target ascension {asc}")
    for name, sel in ((f"A{asc}", lambda r: r["ascension"] == asc), ("lower levels", lambda r: r["ascension"] != asc)):
        if not any(sel(r) for r in runs):
            continue
        print(f"-- {name}")
        for b in range(B):
            g = [r for r in runs[b * len(runs) // B:(b + 1) * len(runs) // B] if sel(r)]
            if not g:
                continue
            fs = [f for r in g for f in fights.get(r["seed"], [])]   # runs of the actors whose fight logs were read
            by = {t: [f for f in fs if f["room_type"] == t] for t in ("monster", "elite", "boss")}
            acts = mean([3.0 if r["won"] else min(r["act"], 2) for r in g])
            print(f"   {b + 1:2d}: runs {len(g):6d}  won {100 * mean([r['won'] for r in g]):5.1f}%  acts cleared {acts:.3f}  act1 {100 * mean([r['act'] >= 1 or r['won'] for r in g]):4.0f}%  "
                  f"act2 {100 * mean([r['act'] >= 2 or r['won'] for r in g]):4.0f}%  floor {mean([r['floors'] for r in g]):5.1f}  elites {mean([r['elites'] for r in g]):.2f}  "
                  f"rest share {100 * sum(r['rests'] for r in g) / max(1, sum(r['rests'] + r['smiths'] for r in g)):3.0f}%  deck {mean([r['deck'] for r in g]):4.1f}  "
                  f"fights won: hallway {100 * mean([f['result'] == 'won' for f in by['monster']]):.1f}% elite {100 * mean([f['result'] == 'won' for f in by['elite']]):.1f}% "
                  f"boss {100 * mean([f['result'] == 'won' for f in by['boss']]):.1f}%  versions {g[0].get('model_version')}/{g[0].get('macro_version')}-{g[-1].get('model_version')}/{g[-1].get('macro_version')}")
    print("-- stages per directory (fight numbers from the fight logs read)")
    for lvl in (asc, asc - 1):
        for name, seg in by_dir.items():
            line = stages(name, seg, fights, lvl)
            if line:
                print(line)


if __name__ == "__main__":
    main()
