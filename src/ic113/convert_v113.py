"""ic113 (2026-11-18): move the v0.111 networks and vocabularies to the v0.113.1 simulator, BY ID.

The vocabularies are sorted id lists, so new content (the Fool's cards, tarot cards, new relics, potions, powers,
monsters, events) shifts the indices of existing ids. This script

  vocab   builds src/s3/data/macro_vocab_v113.json: the S1 macro vocabulary (src/s3/data/macro_vocab.json)
          plus every id of the v0.113 simulator it lacks (src/s3/data/sim_ids_v113.json, `S3 vocab`), each
          category sorted again (<pad>, <unk> first); "special" unchanged. Ids the simulator no longer has are kept (human
          data). The combat vocabulary src/s0/data/combat_vocab_v113.json comes from `S3 vocab` (ObsLayout.BuildVocab).
  props   pins the relic / potion probe-key columns of src/s3/data/content_props_v113.json (`S3 props`) to the
          columns of the v0.111 file (content_props.json), so the trained property layers read the same features
          ("item_keys"; macro_s3.load_props).
  combat  converts a combat checkpoint (src/s2/py/model.py format: state_dict / train_state_dict, config, version):
          the one id table `ids` [n_ids + 1, d] (categories in combat_obs.VOCAB_ORDER) is rebuilt for the new vocabulary
          row by row by id; the `prim` / `sec` offset buffers are recomputed. New card ids, and the cards of
          --reset-pools (default IRONCLAD: in the v0.111 vocabulary but never trained, since only Defect was played),
          start at the mean of the trained card embeddings of the same card type (Attack / Skill / Power / Status /
          Curse / ...; trained = cards of --trained-pools, default DEFECT, COLORLESS, STATUS, CURSE; card types from
          src/s3/data/card_info_v113.json, `S3 cardinfo`); a type without trained cards gets the mean of all trained
          cards. New ids of the other categories (powers, monsters, relics, potions) start at the mean of their category's
          old rows (pad / unk excluded). Optimizer states (carry files) are not converted: a chain starts with a fresh
          optimizer from the converted weights (train_combat.py --init).
  macro   converts a macro ensemble (macro_s3.Ensemble: members, vocab, special, ...): the card / ench / relic / potion /
          encounter embeddings and the layout-v2 boss embeddings (encounter id space) are rebuilt by id; new ids start
          as the existing convention for ids outside the vocabulary: the <unk> row (S1 mapped every id seen in fewer
          than 20 runs to <unk>), so a new id starts as the network saw it before, and its content properties (layout v3,
          zero-initialised layers that are trained) tell it apart. The era embedding grows to 7 rows, row 6 (v0.113) =
          row 4 (v0.111). The file's own vocabulary is the old one. Old macro chunks / carry files hold v0.111 indices:
          they are not converted and must not be mixed into a v0.113 chain.
  check   (CPU) combat: the old network on stored observations (v0.111 indices) vs the converted network on the same
          observations remapped by id: identical outputs when no reset / new id occurs; macro: rows of the v0.113 encoder
          (S3 encode rows.jsonl) mapped back to v0.111 indices, old file at era 4 vs converted file at era 6: identical
          when every id exists in both vocabularies.

Usage (from the repository root):
  python -I src/ic113/convert_v113.py vocab
  python -I src/ic113/convert_v113.py props
  python -I src/ic113/convert_v113.py combat IN.pt OUT.pt [--reset-pools IRONCLAD] [--trained-pools DEFECT,COLORLESS,STATUS,CURSE]
  python -I src/ic113/convert_v113.py macro IN.pt OUT.pt
  python -I src/ic113/convert_v113.py check-combat OLD.pt NEW.pt src/s0/data/obs_v3_1000
  python -I src/ic113/convert_v113.py check-macro OLD.pt NEW.pt src/s3/out/encode_x/rows.jsonl
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)                                   # src/
sys.path.insert(0, os.path.join(R, "s0", "py")); sys.path.insert(0, os.path.join(R, "s2", "py")); sys.path.insert(0, os.path.join(R, "s3", "py"))
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

P = {
    "cvocab_old": os.path.join(R, "s0", "data", "combat_vocab_v1.json"),
    "cvocab_new": os.path.join(R, "s0", "data", "combat_vocab_v113.json"),
    "mvocab_old": os.path.join(R, "s3", "data", "macro_vocab.json"),
    "mvocab_new": os.path.join(R, "s3", "data", "macro_vocab_v113.json"),
    "sim_ids": os.path.join(R, "s3", "data", "sim_ids_v113.json"),
    "card_info": os.path.join(R, "s3", "data", "card_info_v113.json"),
    "props_old": os.path.join(R, "s3", "data", "content_props.json"),
    "props_new": os.path.join(R, "s3", "data", "content_props_v113.json"),
}
MCATS = ("card", "ench", "relic", "potion", "encounter")


def sorted_vocab(ids):
    return ["<pad>", "<unk>"] + sorted(set(i for i in ids if i not in ("<pad>", "<unk>")))


def cmd_vocab(a):
    old = json.load(open(P["mvocab_old"])); sim = json.load(open(P["sim_ids"]))
    new = {k: sorted_vocab(old[k] + sim[k]) for k in MCATS}
    new["special"] = old["special"]
    for k in MCATS:
        added = sorted(set(new[k]) - set(old[k]))
        print(f"{k}: {len(old[k])} -> {len(new[k])} (+{len(added)}) e.g. {added[:6]}")
    json.dump(new, open(P["mvocab_new"], "w"))
    print("wrote", P["mvocab_new"])


def cmd_props(a):
    import macro_s3 as ms
    from collections import Counter
    old = json.load(open(P["props_old"])); new = json.load(open(P["props_new"]))

    def top_keys(d, kind):
        c = Counter(k for v in d[kind].values() for k in ((v.get("probe") or {}).get("values") or {}))
        return [k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:ms.N_ITEM_KEYS]]
    keys = {"relic": top_keys(old, "relic"), "potion": top_keys(old, "potion")}
    fresh = {"relic": top_keys(new, "relic"), "potion": top_keys(new, "potion")}
    for k in keys:
        print(f"{k}: v0.111 columns kept; the v0.113 file's own top {ms.N_ITEM_KEYS} keys differ in "
              f"{len(set(keys[k]) ^ set(fresh[k]))} keys, order differs: {keys[k] != fresh[k]}")
    new["item_keys"] = keys
    json.dump(new, open(P["props_new"], "w"))
    print("pinned item_keys in", P["props_new"])


# ---------------------------------------------------------------- combat

def combat_rows(old_v, new_v, ids_w, card_info, reset_pools, trained_pools, log=True):
    import combat_obs as co
    off_o, _ = co.global_offsets(old_v); off_n, n_new = co.global_offsets(new_v)
    d = ids_w.shape[1]
    out = torch.zeros(n_new + 1, d, dtype=ids_w.dtype)
    out[n_new] = ids_w[-1]                                         # "no category" row
    info = card_info["card"]

    def ctype(cid):
        return (info.get(cid) or {}).get("type", "?")
    trained = [cid for cid in old_v["card"][2:] if set((info.get(cid) or {}).get("pools", [])) & set(trained_pools)]
    by_type = {}
    for cid in trained:
        by_type.setdefault(ctype(cid), []).append(ids_w[off_o["card"] + old_v["card"].index(cid)])
    type_mean = {t: torch.stack(v).mean(0) for t, v in by_type.items()}
    all_mean = torch.stack([ids_w[off_o["card"] + old_v["card"].index(c)] for c in trained]).mean(0)
    stats = {}
    for c in co.VOCAB_ORDER:
        oi = {s: i for i, s in enumerate(old_v[c])}
        cat_mean = ids_w[off_o[c] + 2: off_o[c] + len(old_v[c])].mean(0) if len(old_v[c]) > 2 else ids_w[off_o[c] + 1]
        st = stats.setdefault(c, {"copied": 0, "new": 0, "reset": 0})
        for j, s in enumerate(new_v[c]):
            reset = c == "card" and s in oi and set((info.get(s) or {}).get("pools", [])) & set(reset_pools)
            if s in oi and not reset:
                out[off_n[c] + j] = ids_w[off_o[c] + oi[s]]; st["copied"] += 1
            elif c == "card":
                out[off_n[c] + j] = type_mean.get(ctype(s), all_mean); st["reset" if reset else "new"] += 1
            else:
                out[off_n[c] + j] = cat_mean; st["new"] += 1
    if log:
        print("combat ids:", ", ".join(f"{c} {v}" for c, v in stats.items()))
        print("card-type means from trained cards:", {t: len(v) for t, v in by_type.items()}, f"(trained pools {trained_pools})")
    return out


def convert_combat_sd(sd, old_v, new_v, card_info, reset_pools, trained_pools, log=True):
    import combat_obs as co
    sd = dict(sd)
    sd["ids.weight"] = combat_rows(old_v, new_v, sd["ids.weight"], card_info, reset_pools, trained_pools, log)
    prim, sec = co.kind_offsets(new_v)
    sd["prim"] = torch.from_numpy(prim).to(sd["prim"].dtype); sd["sec"] = torch.from_numpy(sec).to(sd["sec"].dtype)
    return sd


def cmd_combat(a):
    import combat_obs as co
    old_v, new_v = co.load_vocab(P["cvocab_old"]), co.load_vocab(P["cvocab_new"])
    ci = json.load(open(P["card_info"]))
    ck = torch.load(a.inp, map_location="cpu", weights_only=False)
    assert ck["state_dict"]["ids.weight"].shape[0] == co.global_offsets(old_v)[1] + 1, "the input does not use the v0.111 vocabulary"
    rp, tp = a.reset_pools.split(",") if a.reset_pools else [], a.trained_pools.split(",")
    out = dict(ck)
    out["state_dict"] = convert_combat_sd(ck["state_dict"], old_v, new_v, ci, rp, tp)
    if "train_state_dict" in ck:
        out["train_state_dict"] = convert_combat_sd(ck["train_state_dict"], old_v, new_v, ci, rp, tp, log=False)
    out["ic113"] = {"from": os.path.basename(a.inp), "vocab": "combat_vocab_v113.json", "reset_pools": rp, "trained_pools": tp}
    torch.save(out, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    print("wrote", a.out)


def cmd_check_combat(a):
    import combat_obs as co
    from model import load_net
    old_v, new_v = co.load_vocab(P["cvocab_old"]), co.load_vocab(P["cvocab_new"])
    ck_o = torch.load(a.old, map_location="cpu", weights_only=False); ck_n = torch.load(a.new, map_location="cpu", weights_only=False)
    net_o, net_n = load_net(old_v, ck_o).eval(), load_net(new_v, ck_n).eval()
    obs, act = co.load_arrays(a.data)
    n = min(len(obs), a.n)
    obs, act = obs[:n].copy(), act[:n].copy()
    reset = set(ck_n.get("ic113", {}).get("reset_pools", []))
    info = json.load(open(P["card_info"]))["card"]
    # remap ids by name: token columns 1 (primary) and 2 (secondary) per kind; action selection cards
    maps = {c: np.array([new_v[c].index(s) if s in new_v[c] else 1 for s in old_v[c]], np.int64) for c in co.VOCAB_ORDER}
    obs_n = obs.copy(); act_n = act.copy(); touched = np.zeros(n, bool)
    for k, (c1, c2) in co.CATEGORIES.items():
        m = obs[..., 0] == k
        for col, c in ((1, c1), (2, c2)):
            if not c:
                continue
            obs_n[..., col][m] = maps[c][obs[..., col][m]]
            if c == "card":
                bad = np.array([bool(set((info.get(s) or {}).get("pools", [])) & reset) for s in old_v["card"]])
                touched |= (m & bad[obs[..., col]]).any(1)
    for j in range(co.MAX_SEL if hasattr(co, "MAX_SEL") else 2):
        col = 4 + 2 * j
        used = act[..., col] > 0
        act_n[..., col][used] = maps["card"][act[..., col][used]]
    with torch.no_grad():
        po, vo = net_o(torch.from_numpy(obs).long(), torch.from_numpy(act).long())
        pn, vn = net_n(torch.from_numpy(obs_n).long(), torch.from_numpy(act_n).long())
    keep = torch.from_numpy(~touched)
    fin = torch.isfinite(po) & torch.isfinite(pn)
    same_inf = (po == pn) | ~(torch.isfinite(po) | torch.isfinite(pn))
    dp = torch.where(fin, (po - pn).abs(), (~same_inf).float())[keep].max().item(); dv = (vo - vn).abs()[keep].max().item()
    print(f"check-combat: {n} states ({int(touched.sum())} hold a reset card and are skipped): max |policy logit diff| {dp:.3g}, "
          f"max |value output diff| {dv:.3g} -> {'PASS' if dp == 0 and dv == 0 else 'FAIL'}")
    return 0 if dp == 0 and dv == 0 else 1


# ---------------------------------------------------------------- macro

def macro_member(sd, old_v, new_v):
    import macro_s3 as ms
    sd = dict(sd)
    tables = {"card": ["card.weight"], "ench": ["ench.weight"], "relic": ["relic.weight"], "potion": ["potion.weight"],
              "encounter": ["enc.weight", "boss_emb.weight", "boss2_emb.weight"]}
    for c, names in tables.items():
        oi = {s: i for i, s in enumerate(old_v[c])}
        idx = torch.tensor([oi.get(s, 1) for s in new_v[c]], dtype=torch.long)    # new id -> its old row (<unk> = 1 if new)
        for nm in names:
            if nm not in sd:
                continue
            w = sd[nm]
            assert w.shape[0] == len(old_v[c]), f"{nm}: {w.shape[0]} rows, vocabulary {len(old_v[c])}"
            sd[nm] = w[idx].clone()
            sd[nm][0] = w[0]                                                     # <pad> stays
    if "era.weight" in sd and sd["era.weight"].shape[0] < ms.N_ERA:
        w = sd["era.weight"]; e = torch.zeros(ms.N_ERA, w.shape[1], dtype=w.dtype)
        e[:w.shape[0]] = w; e[ms.ERA_V0113] = w[ms.ERA_V0111]; sd["era.weight"] = e
    return sd


def cmd_macro(a):
    ck = torch.load(a.inp, map_location="cpu", weights_only=False)
    new_v = json.load(open(P["mvocab_new"]))
    old_v = ck["vocab"]
    for c in MCATS:
        missing = set(old_v[c]) - set(new_v[c])
        assert not missing, f"{c}: ids of the file missing in the new vocabulary {sorted(missing)[:5]}"
        print(f"macro {c}: {len(old_v[c])} -> {len(new_v[c])} ids ({len(new_v[c]) - len(old_v[c])} new, start = <unk> row)")
    assert list(ck["special"]) == list(new_v["special"]), "special vocabulary differs"
    out = dict(ck)
    out["members"] = [macro_member(sd, old_v, new_v) for sd in ck["members"]]
    out["vocab"] = {c: list(new_v[c]) for c in MCATS}
    out["ic113"] = {"from": os.path.basename(a.inp), "vocab": "macro_vocab_v113.json", "era_v0113": "copy of era 4"}
    torch.save(out, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    print("wrote", a.out, f"({len(out['members'])} members)")


def cmd_check_macro(a):
    import macro_s3 as ms
    e_o, e_n = ms.Ensemble.load(a.old, "cpu"), ms.Ensemble.load(a.new, "cpu")
    old_v, new_v = e_o.vocab, e_n.vocab
    back = {c: {i: (old_v[c].index(s) if s in old_v[c] else -1) for i, s in enumerate(new_v[c])} for c in MCATS}
    ints, floats = [], []
    for line in open(a.rows):
        r = json.loads(line)
        ints.append(np.array(r["ints"], np.int64)); floats.append(np.array(r["floats"], np.float32))
        if len(ints) >= a.n:
            break
    I_n = np.stack(ints); F = np.stack(floats); I_o = I_n.copy(); ok = np.ones(len(I_n), bool)
    o = ms.OFF

    def remap(cols, c, rows=slice(None)):
        nonlocal ok
        for col in cols:
            v = I_n[rows, col]
            m = np.array([back[c].get(int(x), -1) if x > 1 else int(x) for x in v])
            ok[rows] &= m >= 0
            I_o[rows, col] = np.maximum(m, 0)
    remap(range(*o["d_ids"]), "card"); remap(range(*o["d_ench"]), "ench"); remap(range(*o["r_ids"]), "relic")
    remap(range(*o["p_ids"]), "potion"); remap([o["ctx"][0] + 5], "encounter"); remap(range(*o["boss"]), "encounter")
    remap(range(*o["f_enc"]), "encounter")
    for j in range(ms.K):
        kc, ic = o["opt_kind"][0] + j, o["opt_id"][0] + j
        for kind, c in ((1, "card"), (2, "relic"), (3, "potion")):
            rows = np.where(I_n[:, kc] == kind)[0]
            if len(rows):
                remap([ic], c, rows)
    I_o[:, o["ctx"][0] + 2] = ms.ERA_V0111; I_n[:, o["ctx"][0] + 2] = ms.ERA_V0113
    with torch.no_grad():
        out_o = e_o.serve(I_o[ok], F[ok], "cpu"); out_n = e_n.serve(I_n[ok], F[ok], "cpu")
    fin = torch.isfinite(out_o) & torch.isfinite(out_n)
    d = torch.where(fin, (out_o - out_n).abs(), (out_o != out_n).float()).max().item()
    print(f"check-macro: {len(I_n)} rows, {int(ok.sum())} with ids in both vocabularies: max |output diff| {d:.3g} -> {'PASS' if d == 0 else 'FAIL'}")
    return 0 if d == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("vocab"); sp.add_parser("props")
    c = sp.add_parser("combat"); c.add_argument("inp"); c.add_argument("out")
    c.add_argument("--reset-pools", default="IRONCLAD"); c.add_argument("--trained-pools", default="DEFECT,COLORLESS,STATUS,CURSE")
    m = sp.add_parser("macro"); m.add_argument("inp"); m.add_argument("out")
    cc = sp.add_parser("check-combat"); cc.add_argument("old"); cc.add_argument("new"); cc.add_argument("data"); cc.add_argument("--n", type=int, default=1000)
    cm = sp.add_parser("check-macro"); cm.add_argument("old"); cm.add_argument("new"); cm.add_argument("rows"); cm.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    return {"vocab": cmd_vocab, "props": cmd_props, "combat": cmd_combat, "macro": cmd_macro,
            "check-combat": cmd_check_combat, "check-macro": cmd_check_macro}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
