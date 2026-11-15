"""Export fight starts ("battle starts") from Spire Codex run records as JSON lines for the C# loader
(src/s0/S0.Sim/BattleStart.cs).

The deck / relic / potion state before fight g is the state after floor g-1, rebuilt with reconstruct() from
src/value_v1/build_floor_dataset.py (imported read-only, not modified). HP, max HP and gold come from the
player_stats of floor g-1; the encounter id is rooms[0].model_id of floor g; act index is the act of floor g.

Usage: ./.venv/bin/python -I src/s0/export_battle_starts.py [--runs PATH] [--out PATH]
           [--build v0.111.0] [--character DEFECT] [--max 300] [--per-run 3] [--seed 0]
"""
import argparse, gzip, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True  # do not write .pyc files into src/value_v1
sys.path.insert(0, os.path.join(ROOT, "src", "value_v1"))
import orjson  # noqa: E402
from build_floor_dataset import reconstruct, insane  # noqa: E402

COMBAT = {"monster", "elite", "boss"}


def battle_starts(r):
    floors, snaps, miss, _ = reconstruct(r)
    if insane(floors, snaps):
        return []
    out = []
    for gi in range(1, len(floors)):  # fight on floor g = gi + 1, state after floor g - 1 = snaps[gi - 1]
        act_index, fl = floors[gi]
        rooms = fl.get("rooms") or []
        if not rooms or rooms[0].get("room_type") not in COMBAT:
            continue
        prev = (floors[gi - 1][1].get("player_stats") or [{}])[0]
        hp, mhp = int(prev.get("current_hp") or 0), int(prev.get("max_hp") or 0)
        if hp <= 0 or mhp <= 0:
            continue
        deck, relics, potions = snaps[gi - 1]
        out.append({
            "id": f"{r.get('run_hash', r.get('seed'))}:{gi + 1}",
            "build_id": r.get("build_id"),
            "character": r["players"][0]["character"],
            "ascension": int(r.get("ascension") or 0),
            "act_ids": list(r.get("acts") or []),
            "act_index": act_index,
            "floor": gi + 1,
            "room_type": rooms[0]["room_type"],
            "encounter_id": rooms[0].get("model_id", ""),
            "monster_ids": list(rooms[0].get("monster_ids") or []),
            "hp": hp, "max_hp": mhp, "gold": int(prev.get("current_gold") or 0),
            "deck": [{"id": c[0], "upgrade": c[1], "enchantment": c[2] or None, "enchantment_amount": c[3]} for c in deck],
            "relics": [x[0] for x in relics if x[0]],
            "potions": [p for p in potions if p],
            "potion_slots": 0,  # loader: max(simulator default for the ascension, number of potions)
            "won_fight": gi + 1 < len(floors) or bool(r.get("win")),  # the player survived this floor
            "reconstruct_misses": dict(miss),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "data/spire_codex/poc_100k/runs.jsonl.gz"))
    ap.add_argument("--out", default=os.path.join(HERE, "data/battle_starts_v0.111.0_defect.jsonl"))
    ap.add_argument("--build", default="v0.111.0")
    ap.add_argument("--character", default="DEFECT")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--per-run", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    want = "CHARACTER." + a.character
    n_runs = n_out = 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with gzip.open(a.runs, "rb") as f, open(a.out, "w") as w:
        for line in f:
            r = orjson.loads(line)
            if r.get("build_id") != a.build or r["players"][0].get("character") != want or len(r["players"]) != 1:
                continue
            starts = battle_starts(r)
            if not starts:
                continue
            n_runs += 1
            for s in rnd.sample(starts, min(a.per_run, len(starts))):
                w.write(json.dumps(s) + "\n")
                n_out += 1
                if n_out >= a.max:
                    break
            if n_out >= a.max:
                break
    print(f"wrote {n_out} battle starts from {n_runs} runs to {a.out}")


if __name__ == "__main__":
    main()
