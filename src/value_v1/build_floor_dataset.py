"""Turn sampled Spire Codex runs into per-floor value-training examples.

For every run the deck, relics and potions at the end of each floor are rebuilt by starting from the
final state stored in the record and undoing each floor's changes, last floor first. One example per
floor g = 1 .. F-1 (the last floor is terminal) with current_hp > 0:
  input  : character, ascension, version era, act, room type, encounter of floor g, scalar status,
           deck cards (id, upgrade level, enchantment, enchantment amount, age), relics (id, age), potions
  labels : win (run eventually won), floors_remaining = F - g
Leakage guards: no act-boss identity (only visible in the record if the run reached the boss), no
end-of-run fields, and any card/relic whose floor_added_to_deck > g is dropped from snapshot g.

Usage: python -I build_floor_dataset.py <sample_dir> <out_dir> [--workers N] [--min-runs 20]
"""
import argparse, collections, csv, gzip, json, multiprocessing as mp, os, time

import numpy as np
import orjson

MAX_DECK, MAX_REL, MAX_POT = 64, 40, 5
ROOM_TYPES = ["<pad>", "<unk>", "ancient", "monster", "elite", "boss", "unknown", "shop", "rest_site", "treasure"]
CHARS = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]
SCALARS = ["floor", "act", "hp", "max_hp", "hp_frac", "gold", "dmg_taken", "hp_healed", "deck_size",
           "n_upgraded", "n_enchanted", "n_relics", "n_potions", "turns_taken", "is_combat", "floor_in_act"]


def norm_card(c):
    e = c.get("enchantment") or {}
    return [c.get("id", ""), int(c.get("current_upgrade_level") or 0), e.get("id", ""),
            int(e.get("amount") or 0), int(c.get("floor_added_to_deck") or 0)]


def find(deck, pred, prefer=None):
    cand = [i for i, c in enumerate(deck) if pred(c)]
    if not cand:
        return -1
    if prefer:
        best = [i for i in cand if prefer(deck[i])]
        if best:
            return best[-1]
    return cand[-1]


def reconstruct(r):
    """Return per-floor snapshots (list index g-1) and miss counters."""
    p = r["players"][0]
    floors = [(a, fl) for a, act in enumerate(r.get("map_point_history") or []) for fl in act]
    deck = [norm_card(c) for c in p.get("deck", [])]
    relics = [[x.get("id", ""), int(x.get("floor_added_to_deck") or 0)] for x in p.get("relics", [])]
    potions = []  # filled by the forward pass below
    miss = collections.Counter()
    snaps = [None] * len(floors)
    for gi in range(len(floors) - 1, -1, -1):
        g = gi + 1
        snaps[gi] = ([list(c) for c in deck], [list(x) for x in relics], list(potions))
        ps = (floors[gi][1].get("player_stats") or [{}])[0]
        # undo in reverse of the assumed application order: removal -> gain -> transform -> upgrade -> enchant
        for e in ps.get("cards_enchanted", []):
            cid, ench = (e.get("card") or {}).get("id", ""), e.get("enchantment", "")
            i = find(deck, lambda c: c[0] == cid and c[2] == ench)
            if i < 0: miss["enchant"] += 1
            else: deck[i][2], deck[i][3] = "", 0
        for cid in ps.get("downgraded_cards", []):
            i = find(deck, lambda c: c[0] == cid)
            if i < 0: miss["downgrade"] += 1
            else: deck[i][1] += 1
        for cid in ps.get("upgraded_cards", []):
            i = find(deck, lambda c: c[0] == cid and c[1] > 0)
            if i < 0: miss["upgrade"] += 1
            else: deck[i][1] -= 1
        for t in ps.get("cards_transformed", []):
            fc, oc = norm_card(t.get("final_card") or {}), norm_card(t.get("original_card") or {})
            i = find(deck, lambda c: c[0] == fc[0], prefer=lambda c: c[4] == fc[4])
            if i < 0: miss["transform"] += 1
            else: deck[i] = oc
        for c in ps.get("cards_gained", []):
            cid = c.get("id", "")
            i = find(deck, lambda x: x[0] == cid, prefer=lambda x: x[4] == g)
            if i < 0: miss["gain"] += 1
            else: deck.pop(i)
        for c in ps.get("cards_removed", []):
            deck.append(norm_card(c))
        # relics: those added on this floor were not there before it; removed ones were
        relics = [x for x in relics if x[1] != g]
        for rid in ps.get("relics_removed", []):
            relics.append([rid, -1])
    start_deck = sorted(c[0] for c in deck)
    # Potions are rebuilt FORWARD from an empty belt: rebuilding them backward from the final belt leaks
    # potions obtained through unlogged routes into earlier floors (seen: 5 potions after Neow). Forward
    # can only under-count, never use future information.
    belt = []
    for gi, (_, fl) in enumerate(floors):
        ps = (fl.get("player_stats") or [{}])[0]
        for pid in list(ps.get("potion_used", [])) + list(ps.get("potion_discarded", [])):
            if pid in belt: belt.remove(pid)
            else: miss["potion_use_unlogged_source"] += 1
        belt += [x.get("choice", "") for x in ps.get("potion_choices", []) if x.get("was_picked")]
        belt += list(ps.get("bought_potions", []))
        belt = belt[-MAX_POT:]
        d, rl, _ = snaps[gi]
        snaps[gi] = (d, rl, list(belt))
    return floors, snaps, miss, start_deck


def count_ids(line):
    r = orjson.loads(line)
    p = r["players"][0]
    ids = collections.defaultdict(set)
    for c in p.get("deck", []):
        ids["card"].add(c.get("id", "")); ids["ench"].add((c.get("enchantment") or {}).get("id", ""))
    for x in p.get("relics", []): ids["relic"].add(x.get("id", ""))
    for act in r.get("map_point_history") or []:
        for fl in act:
            for rm in fl.get("rooms", []) or []: ids["encounter"].add(rm.get("model_id", ""))
            for ps in fl.get("player_stats", []):
                for c in ps.get("cards_gained", []) + ps.get("cards_removed", []):
                    ids["card"].add(c.get("id", ""))
                for t in ps.get("cards_transformed", []):
                    ids["card"].add((t.get("original_card") or {}).get("id", ""))
                for x in ps.get("potion_choices", []): ids["potion"].add(x.get("choice", ""))
                for pid in ps.get("potion_used", []) + ps.get("bought_potions", []): ids["potion"].add(pid)
                for rid in ps.get("relics_removed", []): ids["relic"].add(rid)
                for x in ps.get("relic_choices", []): ids["relic"].add(x.get("choice", ""))
    return {k: [i for i in v if i] for k, v in ids.items()}


VOC = None


def init_worker(vocab):
    global VOC
    VOC = {k: {s: i for i, s in enumerate(v)} for k, v in vocab.items()}


def tok(kind, s):
    return VOC[kind].get(s, 1) if s else 0  # 0 = pad / none, 1 = unk


SANITY = {"max_floors": 55, "max_max_hp": 250, "max_gold": 5000, "max_deck": 120, "max_turns": 60}


def insane(floors, snaps):
    """Physically impossible runs (console commands / cheats): drop the whole run."""
    if len(floors) > SANITY["max_floors"]:
        return "too_many_floors"
    for gi, (_, fl) in enumerate(floors):
        ps = (fl.get("player_stats") or [{}])[0]
        hp, mhp = int(ps.get("current_hp") or 0), int(ps.get("max_hp") or 0)
        if mhp > SANITY["max_max_hp"]: return "max_hp"
        if hp > mhp: return "hp_above_max"
        if int(ps.get("current_gold") or 0) > SANITY["max_gold"]: return "gold"
        if len(snaps[gi][0]) > SANITY["max_deck"]: return "deck_size"
        for rm in fl.get("rooms") or []:
            if int(rm.get("turns_taken") or 0) > SANITY["max_turns"]: return "turns"
    return ""


def build_run(args):
    line, meta = args
    r = orjson.loads(line)
    floors, snaps, miss, start_deck = reconstruct(r)
    F = len(floors)
    bad = insane(floors, snaps)
    if bad:
        return meta["sample_id"], {}, dict(miss), start_deck, 0, F, bad
    win = int(bool(r.get("win")))
    ch = CHARS.index(meta["character"])
    out = collections.defaultdict(list)
    dropped_future = 0
    floor_in_act = collections.Counter()
    for gi in range(F):
        a, fl = floors[gi]
        floor_in_act[a] += 1
        g = gi + 1
        if gi == F - 1:
            break  # terminal floor
        ps = (fl.get("player_stats") or [{}])[0]
        hp, mhp = int(ps.get("current_hp") or 0), int(ps.get("max_hp") or 0)
        if hp <= 0 or mhp <= 0:
            continue
        deck, relics, potions = snaps[gi]
        nd = len(deck)
        deck = [c for c in deck if c[4] <= g]
        relics = [x for x in relics if x[1] <= g]
        dropped_future += nd - len(deck)
        rooms = fl.get("rooms") or [{}]
        rt = fl.get("map_point_type", "")
        d_ids = np.zeros(MAX_DECK, np.uint16); d_upg = np.zeros(MAX_DECK, np.uint8)
        d_ench = np.zeros(MAX_DECK, np.uint8); d_eamt = np.zeros(MAX_DECK, np.uint8); d_age = np.zeros(MAX_DECK, np.uint8)
        for j, c in enumerate(deck[:MAX_DECK]):
            d_ids[j] = tok("card", c[0]); d_upg[j] = max(0, min(c[1], 3)); d_ench[j] = tok("ench", c[2])
            d_eamt[j] = max(0, min(c[3], 20)); d_age[j] = min(max(g - c[4], 0), 63)
        r_ids = np.zeros(MAX_REL, np.uint16); r_age = np.zeros(MAX_REL, np.uint8)
        for j, x in enumerate(relics[:MAX_REL]):
            r_ids[j] = tok("relic", x[0]); r_age[j] = min(max(g - x[1], 0), 63) if x[1] >= 0 else 63
        p_ids = np.zeros(MAX_POT, np.uint16)
        for j, pid in enumerate(potions[:MAX_POT]):
            p_ids[j] = tok("potion", pid)
        ctx = np.array([ch, int(meta["ascension"]), int(meta["era"]), a,
                        ROOM_TYPES.index(rt) if rt in ROOM_TYPES else 1,
                        tok("encounter", rooms[0].get("model_id", "") if rooms else "")], np.int16)
        is_combat = rt in ("monster", "elite", "boss")
        sc = np.array([g / 50, a / 3, hp / 100, mhp / 100, hp / mhp, int(ps.get("current_gold") or 0) / 500,
                       int(ps.get("damage_taken") or 0) / 50, int(ps.get("hp_healed") or 0) / 50, len(deck) / 40,
                       sum(c[1] > 0 for c in deck) / 20, sum(bool(c[2]) for c in deck) / 10, len(relics) / 20,
                       len(potions) / 5, (int(rooms[0].get("turns_taken") or 0) if rooms and is_combat else 0) / 10,
                       float(is_combat), floor_in_act[a] / 20], np.float32)
        for k, v in (("d_ids", d_ids), ("d_upg", d_upg), ("d_ench", d_ench), ("d_eamt", d_eamt), ("d_age", d_age),
                     ("r_ids", r_ids), ("r_age", r_age), ("p_ids", p_ids), ("ctx", ctx), ("scal", sc)):
            out[k].append(v)
        out["win"].append(win); out["floors_left"].append(F - g); out["floor"].append(g)
        out["deck_trunc"].append(int(len(deck) > MAX_DECK))
    arrays = {k: np.stack(v) if k not in ("win", "floors_left", "floor", "deck_trunc") else np.array(v, np.int16)
              for k, v in out.items()} if out else {}
    return meta["sample_id"], arrays, dict(miss), start_deck, dropped_future, F, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir"); ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-runs", type=int, default=20, help="ids seen in fewer train runs map to <unk>")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    idx = list(csv.DictReader(open(os.path.join(a.sample_dir, "index.csv"))))
    lines = gzip.open(os.path.join(a.sample_dir, "runs.jsonl.gz"), "rb").read().splitlines()
    assert len(lines) == len(idx)
    # vocab from the train split only
    with mp.Pool(a.workers) as pool:
        counts = collections.defaultdict(collections.Counter)
        for m, ids in zip(idx, pool.imap(count_ids, lines, chunksize=256)):
            if m["split"] == "train":
                for k, v in ids.items(): counts[k].update(v)
    vocab = {k: ["<pad>", "<unk>"] + sorted(i for i, n in c.items() if n >= a.min_runs) for k, c in counts.items()}
    vocab_stats = {k: {"kept": len(vocab[k]) - 2, "seen": len(c), "unk_ids": len(c) - len(vocab[k]) + 2} for k, c in counts.items()}
    assert len(vocab["ench"]) < 256
    print("vocab:", vocab_stats, f"({time.time()-t0:.0f}s)", flush=True)
    # examples
    parts = collections.defaultdict(list)
    miss_tot, miss_runs, start_decks = collections.Counter(), 0, collections.defaultdict(collections.Counter)
    run_of, n_dropped_future, n_runs_used, insane_runs = [], 0, 0, collections.Counter()
    with mp.Pool(a.workers, initializer=init_worker, initargs=(vocab,)) as pool:
        for m, (sid, arr, miss, sdeck, dropped, F, bad) in zip(idx, pool.imap(build_run, zip(lines, idx), chunksize=64)):
            if bad:
                insane_runs[bad] += 1
                continue
            miss_tot.update(miss); miss_runs += bool(miss); n_dropped_future += dropped
            start_decks[(m["character"], int(m["ascension"]))][tuple(sdeck)] += 1
            if not arr:
                continue
            n_runs_used += 1
            n = len(arr["win"])
            for k, v in arr.items(): parts[k].append(v)
            run_of.append(np.full(n, int(sid), np.int32))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["run"] = np.concatenate(run_of)
    split_of = {int(m["sample_id"]): m["split"] for m in idx}
    wnorm = {int(m["sample_id"]): float(m["weight_norm"]) for m in idx}
    era = {int(m["sample_id"]): int(m["era"]) for m in idx}
    data["split"] = np.array([{"train": 0, "val": 1, "test": 2}[split_of[s]] for s in data["run"]], np.int8)
    data["weight"] = np.array([wnorm[s] for s in data["run"]], np.float32)
    for k, v in data.items():
        np.save(os.path.join(a.out_dir, f"{k}.npy"), v)
    # start-deck consistency: share of runs whose rebuilt floor-0 deck equals the modal deck for its char x asc
    mode_hits = sum(c.most_common(1)[0][1] for c in start_decks.values())
    total_runs = sum(sum(c.values()) for c in start_decks.values())
    stats = {
        "runs": len(idx), "runs_with_examples": n_runs_used, "examples": int(len(data["win"])),
        "examples_by_split": {s: int((data["split"] == i).sum()) for i, s in enumerate(["train", "val", "test"])},
        "vocab": vocab_stats, "insane_runs_dropped": dict(insane_runs), "sanity_thresholds": SANITY,
        "undo_misses": dict(miss_tot), "runs_with_any_miss": miss_runs,
        "future_cards_dropped": n_dropped_future, "deck_truncated_examples": int(data["deck_trunc"].sum()),
        "start_deck_matches_mode": round(mode_hits / total_runs, 4),
        "max_deck": MAX_DECK, "max_relics": MAX_REL, "max_potions": MAX_POT,
        "scalars": SCALARS, "room_types": ROOM_TYPES, "characters": CHARS,
        "ctx_fields": ["character", "ascension", "era", "act", "room_type", "encounter"],
        "elapsed_s": round(time.time() - t0),
    }
    json.dump(vocab, open(os.path.join(a.out_dir, "vocab.json"), "w"))
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
