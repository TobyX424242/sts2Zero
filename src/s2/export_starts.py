"""S2 battle-start pool: every combat floor of every single-player run of one character (--character, default DEFECT) and
game build(s) (--build, comma-separated, default v0.111.0; ic113: e.g. --character IRONCLAD --build v0.111.0,v0.113.0,v0.113.1)
at the given ascension in the
full Spire Codex dump (src/s0/export_battle_starts.py battle_starts(), same JSON format), split by run:
runs whose hash falls in the held-out 5% (sha1(run_hash) mod 20 == 0) go to the evaluation file (gate G2), the rest
to the training file.

Usage: ./.venv/bin/python -I src/s2/export_starts.py [--runs PATH] [--ascension 10] [--character DEFECT] [--build v0.111.0]
       [--out-dir src/s2/data] [--standard]
--standard (ic113, 2026-11-18): only runs of the standard game (game_mode "standard", no modifiers, no modded id: an id
with a "-" namespace, as in src/ic113/fetch_human.sh); abandoned runs are kept (their fights were played). Without it
the pool also holds modded, custom, daily, Dread, blitz and modifier runs (src/ic113/DAY0.md: about 8% of the
v0.111 Ironclad A10 starts and 33% of the v0.113 A8-A10 ones, Dread 29%). Output files get the suffix "_std".
--macro-ctx (ic113 warm-up, 2026-11-18): each start gets "macro_ctx", the run-state inputs of the macro network's layout
v2 / v3 that the battle start itself lacks, from the same run record (src/ic113/WARMUP.md; read by `S3 tables`):
floor in act, elites / rest sites / shops entered in the act (current floor included, by the floor's first room), the
act's boss(es) when the run reached them, final / first-of-two act-3 boss, the last 8 completed fights (encounter, room
1 hallway / 2 elite / 3 boss, HP at start, HP at end before post-fight heals, max HP at start, potions used, turns,
floor, act), and the floor each deck card / relic was added (card and relic ages).
"""
import argparse, gzip, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "src", "s0"))
import orjson  # noqa: E402
from export_battle_starts import battle_starts  # noqa: E402
from build_floor_dataset import reconstruct  # noqa: E402  (path set up by export_battle_starts)

MOD_RE = re.compile(rb'"(?:CARD|RELIC|POTION|CHARACTER|ENCOUNTER|MONSTER|EVENT|ENCHANTMENT|POWER|ORB|AFFLICTION|ACT)\.[A-Za-z0-9_]+-')


def _ps(fl):
    return (fl.get("player_stats") or [{}])[0]


def macro_ctx(r, starts):
    """Adds "macro_ctx" to every start of run r (see --macro-ctx)."""
    floors, snaps, _, _ = reconstruct(r)
    act_start, bosses = {}, {}
    for gi, (a, fl) in enumerate(floors):
        act_start.setdefault(a, gi + 1)
        rooms = fl.get("rooms") or []
        if rooms and rooms[0].get("room_type") == "boss":
            bosses.setdefault(a, []).append(rooms[0].get("model_id", ""))
    room_code = {"monster": 1, "elite": 2, "boss": 3}
    for s in starts:
        g = s["floor"]; gi = g - 1; a = s["act_index"]
        el = re = sh = 0
        for gj in range(act_start.get(a, g) - 1, gi + 1):
            rooms = floors[gj][1].get("rooms") or []
            rt = rooms[0].get("room_type") if rooms else None
            el += rt == "elite"; re += rt == "rest_site"; sh += rt == "shop"
        fights = []
        for gj in range(1, gi):
            rooms = floors[gj][1].get("rooms") or []
            if not rooms or rooms[0].get("room_type") not in room_code:
                continue
            prev, cur = _ps(floors[gj - 1][1]), _ps(floors[gj][1])
            end = int(cur.get("current_hp") or 0) - int(cur.get("hp_healed") or 0)
            fights.append([rooms[0].get("model_id", ""), room_code[rooms[0]["room_type"]], int(prev.get("current_hp") or 0),
                           max(0, end), int(prev.get("max_hp") or 0), len(cur.get("potion_used") or []),
                           int(rooms[0].get("turns_taken") or 0), gj + 1, floors[gj][0]])
        bl = bosses.get(a, [])
        prev_rooms = floors[gi - 1][1].get("rooms") or [] if gi >= 1 else []
        prev_boss = bool(prev_rooms) and prev_rooms[0].get("room_type") == "boss" and floors[gi - 1][0] == a
        is_boss3 = s["room_type"] == "boss" and a == 2
        deck, relics, _ = snaps[gi - 1]
        s["macro_ctx"] = {
            "floor_in_act": g - act_start.get(a, g) + 1, "elites": el, "rests": re, "shops": sh,
            "boss": bl[0] if bl else "", "boss2": bl[1] if a == 2 and len(bl) > 1 else "",
            "final": is_boss3 and (prev_boss or s["ascension"] < 10), "first_of_two": is_boss3 and not prev_boss and s["ascension"] >= 10,
            "fights": fights[-8:], "card_floors": [int(c[4]) for c in deck], "relic_floors": [int(x[1]) for x in relics if x[0]],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "src/data/spire_codex/runs-latest.jsonl.gz"))
    ap.add_argument("--build", default="v0.111.0")
    ap.add_argument("--character", default="DEFECT")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--standard", action="store_true", help="standard, unmodded runs only (abandoned kept)")
    ap.add_argument("--macro-ctx", action="store_true", help="add the macro network's run-state context to every start")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    want = "CHARACTER." + a.character
    builds = [b for b in a.build.split(",") if b]
    tag = f"{'+'.join(builds)}_{a.character.lower()}_a{a.ascension}" + ("_std" if a.standard else "")
    n_runs = {"train": 0, "eval": 0}; n_out = {"train": 0, "eval": 0}; seen = set()
    outs = {k: open(os.path.join(a.out_dir, f"starts_{tag}_{k}.jsonl"), "w") for k in n_out}
    with gzip.open(a.runs, "rb") as f:
        for line in f:
            if not any(b.encode() in line for b in builds) or want.encode() not in line:
                continue
            if a.standard and MOD_RE.search(line):
                continue
            r = orjson.loads(line)
            if (r.get("build_id") not in builds or len(r.get("players") or []) != 1 or r["players"][0].get("character") != want
                    or int(r.get("ascension") or 0) != a.ascension):
                continue
            if a.standard and (r.get("game_mode", "standard") != "standard" or r.get("modifiers")):
                continue
            h = r.get("run_hash") or r.get("seed")
            if h in seen:
                continue
            seen.add(h)
            starts = battle_starts(r)
            if not starts:
                continue
            if a.macro_ctx:
                macro_ctx(r, starts)
            split = "eval" if int(hashlib.sha1(str(h).encode()).hexdigest(), 16) % 20 == 0 else "train"
            n_runs[split] += 1
            for s in starts:
                outs[split].write(json.dumps(s) + "\n")
                n_out[split] += 1
    for w in outs.values():
        w.close()
    print(json.dumps({"runs": n_runs, "battle_starts": n_out, "tag": tag}))


if __name__ == "__main__":
    main()
