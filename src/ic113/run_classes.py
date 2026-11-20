"""Classifies human runs by the filter of the human win-rate tables (src/ic113/fetch_human.sh): "clean" = single
player, game_mode standard, not abandoned, no modifiers, no modded id (an id with a "-" namespace, e.g.
CARD.THEHEROEXPANSION-TRY_AGAIN); otherwise the first failing reason. src/s2/export_starts.py does not apply this
filter, so its battle starts include modded and non-standard runs; src/ic113/import_report.py --classes splits the
import test by these classes.

usage: python3 -I src/ic113/run_classes.py OUT.json RUNS.jsonl.gz [...]   (map run_hash -> class, Ironclad runs only)
"""
import gzip, json, re, sys

MOD_RE = re.compile(rb'"(?:CARD|RELIC|POTION|CHARACTER|ENCOUNTER|MONSTER|EVENT|ENCHANTMENT|POWER|ORB|AFFLICTION|ACT)\.[A-Za-z0-9_]+-')


def main():
    out = {}
    for fn in sys.argv[2:]:
        for line in gzip.open(fn, "rb"):
            if b"CHARACTER.IRONCLAD" not in line:
                continue
            try: r = json.loads(line)
            except Exception: continue
            pl = r.get("players") or []
            if len(pl) != 1 or pl[0].get("character") != "CHARACTER.IRONCLAD":
                continue
            h = str(r.get("run_hash", r.get("seed")))   # as src/s0/export_battle_starts.py builds the start id
            if MOD_RE.search(line): c = "modded_id"
            elif r.get("game_mode", "standard") != "standard": c = "game_mode_" + str(r.get("game_mode"))
            elif r.get("modifiers"): c = "modifiers"
            elif r.get("was_abandoned"): c = "abandoned"
            else: c = "clean"
            out.setdefault(h, c)
    json.dump(out, open(sys.argv[1], "w"))
    import collections
    print(len(out), dict(collections.Counter(out.values())))


if __name__ == "__main__":
    main()
