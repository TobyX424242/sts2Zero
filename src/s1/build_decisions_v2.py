"""S1 item 10: decision dataset for the macro policy head (PROPOSAL.md §6.3).

Each decision points to the floor example of its floor in the floors_v2 dataset (state AFTER the floor) plus a short
list of edits that undo the choice, so the policy sees the state BEFORE it chose (picked card removed, removed card put
back, gold spent restored, smithed card un-upgraded, HP healed at a rest site removed, bought relics/potions removed).

Decision types (dtype):
  0 CARD_REWARD   choose one: offered cards + SKIP             (non-shop floors, 2-5 offered, at most one picked)
  1 RELIC_PICK    choose one: offered relics                   (non-shop, non-treasure, >=2 offered, one picked)
  2 ANCIENT       choose one: ancient options (as relic ids)   (ancient floors)
  3 REST          choose one: rest-site option (global vocabulary; offered set not logged)
  4 SMITH_TARGET  choose one deck card to upgrade              (rest site SMITH with one upgraded card)
  5 REMOVE_TARGET choose one deck card to remove               (shop or event removal of exactly one card)
  6 SHOP          per item buy / not buy: shop cards, relics, potions, and the REMOVE service (binary labels)
Events and map paths are not in the logs as choices with alternatives, so they are left to S3.

Output (in <floors_dir>/decisions/): ex.npy (floor example index), dtype.npy, n_opt.npy, opt_kind/opt_id/opt_upg/opt_label
[N, K] (kind 0 pad, 1 card, 2 relic, 3 potion, 4 special), target_slot.npy (deck slot for types 4-5, else -1),
edits.npy [N, E, 6] (op, a, b, c, d, e), plus special_vocab.json, stats.json.

Usage: python -I build_decisions_v2.py <runs-latest.jsonl.gz> <floors_dir> [--workers 28] [--max-runs N]
"""
import argparse, collections, csv, json, multiprocessing as mp, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "value_v1"))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import numpy as np
import orjson

from sample_runs import iter_raw_lines

K, E = 16, 12
# edit ops
DEL_CARD, ADD_CARD, UPG, DEL_RELIC, DEL_POT, SET_GOLD, SET_HP = 1, 2, 3, 4, 5, 6, 7
SPECIAL = ["<pad>", "SKIP", "REMOVE"]            # rest-site options are appended as "REST:<name>"
REWARD_ROOMS = {"monster", "elite", "boss", "unknown", "ancient"}

G = {}   # per-worker globals


def init_worker(floors_dir, vocab):
    G["a"] = {k: np.load(os.path.join(floors_dir, f"{k}.npy"), mmap_mode="r")
              for k in ("d_ids", "d_upg", "d_age", "r_ids", "r_age", "p_ids", "scal")}
    G["voc"] = {k: {s: i for i, s in enumerate(v)} for k, v in vocab.items()}


def tok(kind, s):
    return G["voc"][kind].get(s, 1) if s else 0


def free_slot(d):
    z = np.flatnonzero(d == 0)
    return int(z[0]) if len(z) else -1


def run_decisions(args):
    line, sid, start, floors = args
    r = orjson.loads(line)
    a = G["a"]
    ex_of = {(sid, f): start + i for i, f in enumerate(floors)}    # this run's floor examples
    out, special_seen = [], collections.Counter()
    g = 0
    for act in r.get("map_point_history") or []:
        for fl in act:
            g += 1
            ex = ex_of.get((sid, g))
            if ex is None:
                continue
            ps = (fl.get("player_stats") or [{}])[0]
            rt = fl.get("map_point_type", "")
            d_ids, d_upg, d_age = a["d_ids"][ex], a["d_upg"][ex], a["d_age"][ex]
            r_ids, r_age, p_ids = a["r_ids"][ex], a["r_age"][ex], a["p_ids"][ex]
            used_d, used_r, used_p = set(), set(), set()

            def find_new_card(cid):          # slot of a card added on this floor (age 0)
                t = tok("card", cid)
                for j in np.flatnonzero((d_ids == t) & (d_age == 0))[::-1]:
                    if int(j) not in used_d:
                        used_d.add(int(j)); return int(j)
                return -1

            def find_new_relic(rid):
                t = tok("relic", rid)
                for j in np.flatnonzero((r_ids == t) & (r_age == 0))[::-1]:
                    if int(j) not in used_r:
                        used_r.add(int(j)); return int(j)
                return -1

            def find_potion(pid):
                t = tok("potion", pid)
                for j in np.flatnonzero(p_ids == t)[::-1]:
                    if int(j) not in used_p:
                        used_p.add(int(j)); return int(j)
                return -1

            def emit(dtype, opts, labels, edits, target=-1):
                if len(opts) > K or len(edits) > E:
                    return
                out.append((ex, dtype, opts, labels, edits, target))

            # ---- 0 CARD_REWARD
            cc = ps.get("card_choices") or []
            if rt in REWARD_ROOMS and 2 <= len(cc) <= 5:
                picks = [k for k, x in enumerate(cc) if x.get("was_picked")]
                if len(picks) <= 1:
                    opts = [(1, tok("card", (x.get("card") or {}).get("id", "")),
                             min(int((x.get("card") or {}).get("current_upgrade_level") or 0), 3)) for x in cc] + [(4, 1, 0)]
                    edits = []
                    if picks:
                        slot = find_new_card((cc[picks[0]].get("card") or {}).get("id", ""))
                        if slot < 0:
                            opts = None
                        else:
                            edits.append((DEL_CARD, slot, 0, 0, 0, 0))
                    if opts is not None and min(o[1] for o in opts[:-1]) >= 1:
                        lab = [0.0] * len(opts); lab[picks[0] if picks else len(opts) - 1] = 1.0
                        emit(0, opts, lab, edits)
            # ---- 1 RELIC_PICK
            rc = ps.get("relic_choices") or []
            if rt not in ("shop", "treasure") and len(rc) >= 2:
                picks = [k for k, x in enumerate(rc) if x.get("was_picked")]
                if len(picks) == 1:
                    slot = find_new_relic(rc[picks[0]].get("choice", ""))
                    if slot >= 0:
                        opts = [(2, tok("relic", x.get("choice", "")), 0) for x in rc]
                        lab = [0.0] * len(opts); lab[picks[0]] = 1.0
                        emit(1, opts, lab, [(DEL_RELIC, slot, 0, 0, 0, 0)])
            # ---- 2 ANCIENT
            ac = ps.get("ancient_choice") or []
            if rt == "ancient" and len(ac) >= 2:
                ch = [k for k, x in enumerate(ac) if x.get("was_chosen")]
                if len(ch) == 1:
                    edits = []
                    slot = find_new_relic("RELIC." + ac[ch[0]].get("TextKey", ""))
                    if slot >= 0:
                        edits.append((DEL_RELIC, slot, 0, 0, 0, 0))
                    for c in ps.get("cards_gained") or []:
                        s2 = find_new_card(c.get("id", ""))
                        if s2 >= 0:
                            edits.append((DEL_CARD, s2, 0, 0, 0, 0))
                    opts = [(2, tok("relic", "RELIC." + x.get("TextKey", "")), 0) for x in ac]
                    lab = [0.0] * len(opts); lab[ch[0]] = 1.0
                    emit(2, opts, lab, edits)
            # ---- 3 REST and 4 SMITH_TARGET
            rs = ps.get("rest_site_choices") or []
            if rt == "rest_site" and len(rs) == 1:
                name = "REST:" + str(rs[0]); special_seen[name] += 1
                hp, healed = int(ps.get("current_hp") or 0), int(ps.get("hp_healed") or 0)
                edits = [(SET_HP, max(1, hp - healed), 0, 0, 0, 0)] if healed else []
                target = -1
                up = ps.get("upgraded_cards") or []
                if rs[0] == "SMITH" and len(up) == 1:
                    t = tok("card", up[0])
                    cand = [int(j) for j in np.flatnonzero((d_ids == t) & (d_upg > 0)) if int(j) not in used_d]
                    if cand:
                        target = cand[-1]; used_d.add(target)
                        edits.append((UPG, target, -1, 0, 0, 0))
                emit(3, [(4, name, 0)], [1.0], edits)          # special id resolved to an index after the pass
                if target >= 0:
                    emit(4, [], [], edits, target)
            # ---- 5 REMOVE_TARGET and 6 SHOP
            rem = ps.get("cards_removed") or []
            rem_edit, rem_slot = None, -1
            if len(rem) == 1 and rt in ("shop", "unknown"):
                c = rem[0]; s2 = free_slot(d_ids)
                if s2 >= 0:
                    e = c.get("enchantment") or {}
                    rem_edit = (ADD_CARD, tok("card", c.get("id", "")), min(int(c.get("current_upgrade_level") or 0), 3),
                                tok("ench", e.get("id", "")), min(int(e.get("amount") or 0), 20),
                                min(max(g - int(c.get("floor_added_to_deck") or 0), 0), 63))
                    rem_slot = s2
            if rt == "shop":
                gained = [c.get("id", "") for c in ps.get("cards_gained") or []]
                edits, opts, lab = [], [], []
                gcount = collections.Counter(gained)
                for x in cc:                                         # shop card inventory
                    cid = (x.get("card") or {}).get("id", "")
                    bought = gcount[cid] > 0
                    if bought:
                        gcount[cid] -= 1
                    opts.append((1, tok("card", cid), min(int((x.get("card") or {}).get("current_upgrade_level") or 0), 3)))
                    lab.append(float(bought))
                for cid in gained:
                    s2 = find_new_card(cid)
                    if s2 >= 0:
                        edits.append((DEL_CARD, s2, 0, 0, 0, 0))
                for x in rc:
                    opts.append((2, tok("relic", x.get("choice", "")), 0)); lab.append(float(bool(x.get("was_picked"))))
                    if x.get("was_picked"):
                        s2 = find_new_relic(x.get("choice", ""))
                        if s2 >= 0:
                            edits.append((DEL_RELIC, s2, 0, 0, 0, 0))
                bought_p = collections.Counter(ps.get("bought_potions") or [])
                for x in ps.get("potion_choices") or []:
                    pid = x.get("choice", "")
                    b = bought_p[pid] > 0
                    if b:
                        bought_p[pid] -= 1
                        s2 = find_potion(pid)
                        if s2 >= 0:
                            edits.append((DEL_POT, s2, 0, 0, 0, 0))
                    opts.append((3, tok("potion", pid), 0)); lab.append(float(b))
                opts.append((4, 2, 0)); lab.append(float(bool(rem)))   # REMOVE service
                if rem_edit is not None:
                    edits.append(rem_edit)
                spent = int(ps.get("gold_spent") or 0)
                if spent:
                    edits.append((SET_GOLD, int(ps.get("current_gold") or 0) + spent, 0, 0, 0, 0))
                if opts and min(o[1] for o in opts) >= 1:
                    emit(6, opts, lab, edits)
            if rem_edit is not None:
                ed = [rem_edit]
                spent = int(ps.get("gold_spent") or 0)
                if rt == "shop" and spent:
                    ed.append((SET_GOLD, int(ps.get("current_gold") or 0) + spent, 0, 0, 0, 0))
                emit(5, [], [], ed, rem_slot)
    return out, special_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("floors_dir")
    ap.add_argument("--workers", type=int, default=28); ap.add_argument("--max-runs", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    F = a.floors_dir; O = os.path.join(F, "decisions"); os.makedirs(O, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(F, "index.csv"))))
    if a.max_runs:
        rows = rows[:a.max_runs]
    by_idx = {int(r["idx"]): int(r["sample_id"]) for r in rows}
    run = np.load(os.path.join(F, "run.npy")); floor = np.load(os.path.join(F, "floor.npy"))
    starts = np.flatnonzero(np.r_[True, run[1:] != run[:-1]])            # examples are stored run by run
    lens = np.diff(np.r_[starts, len(run)])
    span = {int(run[s]): (int(s), int(l)) for s, l in zip(starts, lens)}
    vocab = json.load(open(os.path.join(F, "vocab.json")))
    print(f"{len(rows):,} runs, {len(span):,} runs with floor examples ({time.time()-t0:.0f}s)", flush=True)

    def gen():
        for i, line in enumerate(iter_raw_lines(a.src)):
            s = by_idx.get(i)
            if s is not None and s in span:
                st, ln = span[s]
                yield line, s, st, floor[st:st + ln].tolist()
    dec, special = [], collections.Counter()
    with mp.Pool(a.workers, initializer=init_worker, initargs=(F, vocab)) as pool:
        for k, (d, sp) in enumerate(pool.imap(run_decisions, gen(), chunksize=64)):
            dec.extend(d); special.update(sp)
            if (k + 1) % 200_000 == 0:
                print(f"  {k+1:,} runs, {len(dec):,} decisions ({time.time()-t0:.0f}s)", flush=True)
    spec = SPECIAL + sorted(special)
    spec_idx = {s: i for i, s in enumerate(spec)}
    N = len(dec)
    ex = np.zeros(N, np.int64); dt = np.zeros(N, np.int8); nopt = np.zeros(N, np.int8); tgt = np.full(N, -1, np.int16)
    ok, oid, oup, olab = (np.zeros((N, K), np.int8), np.zeros((N, K), np.int16), np.zeros((N, K), np.int8), np.zeros((N, K), np.float32))
    ed = np.zeros((N, E, 6), np.int16)
    for i, (e_, d_, opts, labs, edits, t_) in enumerate(dec):
        ex[i], dt[i], nopt[i], tgt[i] = e_, d_, len(opts), t_
        for j, (kind, idv, up) in enumerate(opts):
            ok[i, j] = kind; oid[i, j] = spec_idx[idv] if isinstance(idv, str) else idv; oup[i, j] = up
        olab[i, :len(labs)] = labs
        for j, e in enumerate(edits):
            ed[i, j] = e
    for name, arr in (("ex", ex), ("dtype", dt), ("n_opt", nopt), ("target_slot", tgt), ("opt_kind", ok), ("opt_id", oid),
                      ("opt_upg", oup), ("opt_label", olab), ("edits", ed)):
        np.save(os.path.join(O, f"{name}.npy"), arr)
    names = ["CARD_REWARD", "RELIC_PICK", "ANCIENT", "REST", "SMITH_TARGET", "REMOVE_TARGET", "SHOP"]
    split = np.load(os.path.join(F, "split.npy"))[ex] if N else np.zeros(0)
    stats = {"runs": len(rows), "decisions": N, "by_type": {names[t]: int((dt == t).sum()) for t in range(7)},
             "by_split": {s: int((split == i).sum()) for i, s in enumerate(["train", "val", "test"])},
             "card_reward_skip_share": round(float(olab[dt == 0][np.arange((dt == 0).sum()), nopt[dt == 0] - 1].mean()), 4) if (dt == 0).any() else None,
             "shop_items_bought_share": round(float(olab[dt == 6].sum() / max(1, (ok[dt == 6] > 0).sum())), 4) if (dt == 6).any() else None,
             "rest_options": {s: int(special[s]) for s in sorted(special, key=lambda s: -special[s])},
             "special_vocab": spec, "elapsed_s": round(time.time() - t0)}
    json.dump(spec, open(os.path.join(O, "special_vocab.json"), "w"))
    json.dump(stats, open(os.path.join(O, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
