"""Human fight outcomes for the larger-combat-network test (src/s3/shadow_fix.sh): the A10 training battle starts
of strong human runs, each with the logged outcome of that fight.

Selection (owner, 2026-11-14: "good human data that in high level and cleared high act/win"): Defect v0.111.0 A10
single-player runs of the S2 training split (src/s2/export_starts.py) that won or reached act 3.
Outcome of the fight on floor g (as src/s2/human_reference.py): death if the run ended on floor g without a win,
otherwise the HP after floor g (includes post-fight healing).
Caveat: selecting runs by how they ended means every act 1-2 fight in the selection was survived.

usage: ./.venv/bin/python -I src/s3/py/human_rows_prep.py OUT_STARTS [--starts F] [--runs F] [--min-act 2] [--wins-only]
Writes OUT_STARTS (the start lines with "h_surv" and "h_end_hp" added) and prints counts.
"""
import argparse, gzip, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "src", "value_v1"))
import orjson  # noqa: E402
from build_floor_dataset import reconstruct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--starts", default=os.path.join(ROOT, "src/s2/data/starts_v0.111.0_defect_a10_train.jsonl"))
    ap.add_argument("--runs", default=os.path.join(ROOT, "src/data/spire_codex/runs-latest.jsonl.gz"))
    ap.add_argument("--min-act", type=int, default=2, help="keep runs that won or reached this act index (2 = act 3)")
    ap.add_argument("--wins-only", action="store_true")
    a = ap.parse_args()
    want = {}
    for line in open(a.starts, "rb"):
        h, g = orjson.loads(line)["id"].split(":")
        want.setdefault(h, set()).add(int(g))
    print(f"{sum(map(len, want.values()))} starts from {len(want)} runs in {a.starts}", flush=True)
    outcome, stat = {}, Counter()
    with gzip.open(a.runs, "rb") as f:
        for line in f:
            if b"v0.111.0" not in line or b"DEFECT" not in line:
                continue
            r = orjson.loads(line)
            h = r.get("run_hash") or r.get("seed")
            if h not in want or h in outcome:
                continue
            floors, _, _, _ = reconstruct(r)
            won = bool(r.get("win"))
            top = max((fl[0] for fl in floors), default=0)
            stat["runs"] += 1; stat["won"] += won; stat[f"top_act_{top}"] += 1
            if not (won or (not a.wins_only and top >= a.min_act)):
                outcome[h] = None
                continue
            stat["kept_runs"] += 1
            o = {}
            for g in want[h]:
                gi = g - 1
                if gi >= len(floors):
                    continue
                died = gi == len(floors) - 1 and not won
                hp = 0 if died else int(((floors[gi][1].get("player_stats") or [{}])[0]).get("current_hp") or 0)
                o[g] = (0 if died or hp <= 0 else 1, max(0, hp))
            outcome[h] = o
    n, surv = 0, 0
    by_room = Counter()
    with open(a.out, "w") as w:
        for line in open(a.starts, "rb"):
            s = orjson.loads(line)
            h, g = s["id"].split(":")
            o = (outcome.get(h) or {}).get(int(g))
            if o is None:
                continue
            s["h_surv"], s["h_end_hp"] = o
            w.write(json.dumps(s) + "\n")
            n += 1; surv += o[0]; by_room[s.get("room_type")] += 1
    print(json.dumps({"runs_seen": stat["runs"], "won": stat["won"], "kept_runs": stat["kept_runs"],
                      "top_act": {k: v for k, v in stat.items() if k.startswith("top_act")}, "starts": n,
                      "survived": surv, "by_room": dict(by_room)}), flush=True)


if __name__ == "__main__":
    main()
