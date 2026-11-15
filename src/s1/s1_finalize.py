"""S1 wrap-up: ensemble, calibration, dominance checks, the T_f (fight-value table) builder and the G1 report.

  1. Ensemble: the light macro networks given on the command line; ensemble logit = mean of member logits.
  2. Calibration: one temperature per act, fitted on validation floors (p = sigmoid(z / T_act)). Monotone in HP is kept.
  3. Dominance probes on 50k validation states: +10 HP, +5 max HP with +5 HP, +100 gold, +1 potion, -1 potion (should
     raise / lower V) and + one of four curses (should lower V). "Wrong" = moves the wrong way by more than 0.5 point.
  4. T_f builder (PROPOSAL.md §4.2): from the state before a fight (floor g-1 example) and the fight (floor g room,
     encounter, act), build the post-fight state s_g(h, P) for every HP h and every subset P of the belt; T(0) = 0;
     T = 1 if the fight is the final boss. Tests: death, final boss, batched = direct evaluation, monotone in HP,
     potion dominance across subsets, and closeness to V of the actual logged post-fight state.
  5. G1 report against the targets in PROPOSAL.md §6.3.

Usage: python -I s1_finalize.py <floors_dir> <out.json> <member_dir> [<member_dir> ...]
"""
import itertools, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "value_v1"))
import numpy as np
import torch

import train_value as tv
import train_value_v2 as v2
import train_macro_v2 as tm

S = v2.S
FIELDS = ("d_ids", "d_upg", "d_ench", "d_eamt", "d_age", "r_ids", "r_age", "p_ids", "ctx", "scal")
ROOM = {"monster": 3, "elite": 4, "boss": 5}
dev = "cuda"


def load_members(dirs, dd):
    ms = []
    for d in dirs:
        ck = torch.load(os.path.join(d, "model.pt"), map_location=dev, weights_only=False)
        m = tm.MacroNet(ck["vocab"], len(dd.special), dd.rest_ids, arch=ck["config"]["arch"]).to(dev)
        m.load_state_dict(ck["state_dict"]); m.eval(); ms.append(m)
    return ms


def take(A, idx):
    return {k: A[k][idx].copy() for k in FIELDS}


def to_batch(arr):
    b = {}
    for k in ("d_ids", "r_ids", "p_ids", "d_upg", "d_ench"):
        b[k] = torch.from_numpy(arr[k].astype(np.int32)).to(dev)
    for k in ("d_eamt", "d_age", "r_age"):
        b[k] = torch.from_numpy(arr[k]).to(dev)
    b["ctx"] = torch.from_numpy(arr["ctx"].astype(np.int64)).to(dev)
    b["scal"] = torch.from_numpy(arr["scal"].astype(np.float32)).to(dev)
    return b


@torch.no_grad()
def member_logits(ms, arr, bs=8192):
    """[n_members, n] value logits for full-width state arrays."""
    n = len(arr["scal"]); out = np.zeros((len(ms), n), np.float32)
    for i in range(0, n, bs):
        b = to_batch({k: v[i:i + bs] for k, v in arr.items()})
        for j, m in enumerate(ms):
            with torch.autocast("cuda", torch.bfloat16):
                lw, _ = m(b)
            out[j, i:i + bs] = lw.float().cpu().numpy()
    return out


def calibrated(z_members, act, temps):
    z = z_members.mean(0)
    return 1 / (1 + np.exp(-z / temps[np.clip(act, 0, 2)]))


def fit_temps(z, y, act):
    temps = np.ones(3)
    grid = np.arange(0.5, 2.0, 0.005)
    for a in range(3):
        m = act == a
        best = min(grid, key=lambda t: tv.metrics(1 / (1 + np.exp(-z[m] / t)), y[m])["logloss"])
        temps[a] = best
    return temps


def eval_block(p, y, fl):
    num = den = 0.0
    for f in np.unique(fl):
        m = fl == f
        if m.sum() >= 500 and 0 < y[m].mean() < 1:
            num += v2.auc(p[m], y[m]) * m.sum(); den += m.sum()
    bands = ((1, 5), (6, 10), (11, 16), (17, 24), (25, 33), (34, 42), (43, 60))
    ece = 0.0; worst_band = 0.0
    for lo, hi in bands:
        m = (fl >= lo) & (fl <= hi)
        if m.any():
            ece += tv.metrics(p[m], y[m])["ece"] * m.sum()
            worst_band = max(worst_band, abs(float(p[m].mean() - y[m].mean())))
    return {**tv.metrics(p, y), "auc_within_floor": round(num / max(den, 1), 4), "ece_within_floor_bands": round(ece / len(p), 4),
            "worst_floor_band_mean_gap_pts": round(100 * worst_band, 2)}


def main():
    F, out_path, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    t0 = time.time()
    data = v2.DataV2(F); A = data.a
    dd = tm.DecisionData(F, data)
    ms = load_members(dirs, dd)
    print(f"{len(ms)} members loaded ({time.time()-t0:.0f}s)", flush=True)
    voc = json.load(open(os.path.join(F, "vocab.json")))
    builds = json.load(open(os.path.join(F, "builds.json"))); tb = builds.index("v0.111.0")
    rng = np.random.default_rng(0)
    res = {"members": dirs}

    # ---- 1-2: ensemble and calibration
    val = np.flatnonzero(A["split"] == 1); val = np.sort(rng.choice(val, min(1_000_000, len(val)), replace=False))
    test = np.flatnonzero(A["split"] == 2)
    zv = member_logits(ms, take(A, val)); zt = member_logits(ms, take(A, test))
    yv, yt = A["win"][val].astype(np.float64), A["win"][test].astype(np.float64)
    act_v, act_t = A["ctx"][val, 3], A["ctx"][test, 3]
    temps = fit_temps(zv.mean(0), yv, np.clip(act_v, 0, 2))
    res["temperatures_by_act"] = [round(float(t), 3) for t in temps]
    fl_t = A["floor"][test]; cell = (A["ctx"][test, 0] == 2) & (A["ctx"][test, 1] == 10) & (A["build"][test] == tb)
    p_raw = 1 / (1 + np.exp(-zt.mean(0))); p_cal = calibrated(zt, act_t, temps)
    res["test"] = {"members": [eval_block(1 / (1 + np.exp(-zt[j])), yt, fl_t) for j in range(len(ms))],
                   "ensemble_raw": eval_block(p_raw, yt, fl_t), "ensemble_calibrated": eval_block(p_cal, yt, fl_t),
                   "target_cell_calibrated": eval_block(p_cal[cell], yt[cell], fl_t[cell]),
                   "member_spread_pts": {"mean": round(100 * float((1 / (1 + np.exp(-zt))).std(0).mean()), 2),
                                         "p90": round(100 * float(np.quantile((1 / (1 + np.exp(-zt))).std(0), 0.9)), 2)}}
    print("test", json.dumps(res["test"]["ensemble_calibrated"]), flush=True)

    # ---- 3: dominance probes (calibrated ensemble)
    idx = np.sort(rng.choice(val, 50_000, replace=False)); base = take(A, idx)
    act = np.clip(base["ctx"][:, 3], 0, 2)
    p0 = calibrated(member_logits(ms, base), act, temps)
    held = A["p_ids"][val][A["p_ids"][val] > 1]
    probes = {}

    def probe(name, edit, want_up, mask=None):
        arr = {k: v.copy() for k, v in base.items()}; m = edit(arr) if mask is None else mask
        d = calibrated(member_logits(ms, arr), act, temps) - p0
        sel = np.ones(len(d), bool) if m is None else m
        d = d[sel]
        wrong = (d < -0.005) if want_up else (d > 0.005)
        probes[name] = {"n": int(sel.sum()), "wrong_over_0.5pt": round(float(wrong.mean()), 4),
                        "right_sign": round(float(((d > 0) if want_up else (d < 0)).mean()), 4),
                        "mean_pts_by_act": [round(100 * float(d[act[sel] == a].mean()), 2) for a in range(3)]}

    def hp_plus(arr):
        hp, mhp = arr["scal"][:, S["hp"]] * 100, arr["scal"][:, S["max_hp"]] * 100
        room = hp < mhp - 9.5
        h2 = np.minimum(hp + 10, mhp); arr["scal"][:, S["hp"]] = h2 / 100; arr["scal"][:, S["hp_frac"]] = h2 / np.maximum(mhp, 1)
        return room

    def maxhp(arr):
        arr["scal"][:, S["max_hp"]] += 0.05; arr["scal"][:, S["hp"]] += 0.05
        arr["scal"][:, S["hp_frac"]] = arr["scal"][:, S["hp"]] / arr["scal"][:, S["max_hp"]]
        return None

    def gold(arr):
        arr["scal"][:, S["gold"]] += 100 / 500; return None

    def pot_add(arr):
        n = (arr["p_ids"] > 0).sum(1); ok = n < 3
        new = rng.choice(held, len(n))
        for i in np.flatnonzero(ok):
            arr["p_ids"][i, n[i]] = new[i]
        arr["scal"][ok, S["n_potions"]] += 0.2
        return ok

    def pot_del(arr):
        n = (arr["p_ids"] > 0).sum(1); ok = n > 0
        for i in np.flatnonzero(ok):
            arr["p_ids"][i, n[i] - 1] = 0
        arr["scal"][ok, S["n_potions"]] -= 0.2
        return ok

    def curse(cid):
        def f(arr):
            n = (arr["d_ids"] > 0).sum(1); ok = n < arr["d_ids"].shape[1]
            for i in np.flatnonzero(ok):
                j = n[i]; arr["d_ids"][i, j] = cid
                for k in ("d_upg", "d_ench", "d_eamt", "d_age"):
                    arr[k][i, j] = 0
            arr["scal"][ok, S["deck_size"]] += 1 / 40
            return ok
        return f
    probe("hp_plus10", hp_plus, True); probe("maxhp_plus5_hp_plus5", maxhp, True); probe("gold_plus100", gold, True)
    probe("potion_plus1", pot_add, True); probe("potion_minus1", pot_del, False)
    for c in ("CARD.CLUMSY", "CARD.INJURY", "CARD.REGRET", "CARD.DOUBT"):
        if c in voc["card"]:
            probe("curse_" + c.split(".")[1].lower(), curse(voc["card"].index(c)), False)
    res["dominance"] = probes
    print("dominance", json.dumps({k: v["wrong_over_0.5pt"] for k, v in probes.items()}), flush=True)

    # ---- 4: T_f builder tests on validation fights
    run, flr, ctx, scal = A["run"], A["floor"], A["ctx"], A["scal"]
    vv = np.flatnonzero(A["split"] == 1)
    pair = (run[vv][1:] == run[vv][:-1]) & (flr[vv][1:] == flr[vv][:-1] + 1) & np.isin(ctx[vv][1:, 4], list(ROOM.values()))
    pre_i, post_i = vv[:-1][pair], vv[1:][pair]
    pick = np.sort(rng.choice(len(pre_i), 2000, replace=False)); pre_i, post_i = pre_i[pick], post_i[pick]
    # expected gold reward per room type, from consecutive validation floors
    gold_gain = {r: float(np.median((scal[post_i, S["gold"]] - scal[pre_i, S["gold"]])[ctx[post_i, 4] == r] * 500)) for r in ROOM.values()}
    final_boss = (ctx[post_i, 4] == ROOM["boss"]) & (ctx[post_i, 3] == 2)
    # note: the dataset excludes each run's last floor, so a won final-boss fight has no logged post-fight state;
    # the final-boss rule (T = 1) is therefore checked only if such pairs occur, and is otherwise enforced by construction
    tf = {"fights": int(len(pre_i)), "expected_gold_by_room": {k: gold_gain[v] for k, v in ROOM.items()}, "final_boss_fights": int(final_boss.sum())}
    states, meta = [], []
    for k, (a_, b_) in enumerate(zip(pre_i, post_i)):
        pre = take(A, np.array([a_]))
        belt = [x for x in pre["p_ids"][0] if x > 0]
        mhp = int(round(scal[a_, S["max_hp"]] * 100))
        for sub in itertools.chain.from_iterable(itertools.combinations(range(len(belt)), r) for r in range(len(belt) + 1)):
            s = {kk: np.repeat(v, mhp, 0) for kk, v in pre.items()}
            s["ctx"][:, 3], s["ctx"][:, 4], s["ctx"][:, 5] = ctx[b_, 3], ctx[b_, 4], ctx[b_, 5]
            for col in ("floor", "act", "floor_in_act", "is_combat"):
                s["scal"][:, S[col]] = scal[b_, S[col]]
            s["scal"][:, S["gold"]] = scal[a_, S["gold"]] + gold_gain[int(ctx[b_, 4])] / 500
            h = np.arange(1, mhp + 1)
            s["scal"][:, S["hp"]] = h / 100; s["scal"][:, S["hp_frac"]] = h / mhp
            s["p_ids"][:] = 0
            for j, bi in enumerate(sub):
                s["p_ids"][:, j] = belt[bi]
            s["scal"][:, S["n_potions"]] = len(sub) / 5
            states.append(s); meta.append((k, sub, mhp))
    big = {kk: np.concatenate([s[kk] for s in states]) for kk in FIELDS}
    zb = member_logits(ms, big)
    act_b = np.clip(big["ctx"][:, 3], 0, 2)
    Tb = calibrated(zb, act_b, temps)
    # direct (unbatched, one state at a time) evaluation of a random sample of entries: must match the batched table
    samp = rng.choice(len(Tb), 200, replace=False)
    direct = np.array([calibrated(member_logits(ms, {kk: big[kk][[i]] for kk in FIELDS}, bs=1), act_b[[i]], temps)[0] for i in samp])
    tf["batched_vs_direct_max_abs_pts"] = round(100 * float(np.abs(direct - Tb[samp]).max()), 4)
    # tables, monotonicity, potion dominance, final boss, closeness to the actual post-fight state
    pos = 0; dec_steps = steps = 0; pot_wrong = pot_n = 0; gaps = []
    tables = {}
    for (k, sub, mhp) in meta:
        T = Tb[pos:pos + mhp]; pos += mhp
        if final_boss[k]:
            T = np.ones_like(T)
        tables[(k, sub)] = np.r_[0.0, T]
        d = np.diff(T); dec_steps += int((d < -1e-6).sum()); steps += len(d)
    for (k, sub), T in tables.items():
        for r in range(len(sub)):
            smaller = sub[:r] + sub[r + 1:]
            if (k, smaller) in tables:
                pot_n += len(T) - 1
                pot_wrong += int((T[1:] < tables[(k, smaller)][1:] - 0.005).sum())
    post = take(A, post_i)
    v_post = calibrated(member_logits(ms, post), np.clip(post["ctx"][:, 3], 0, 2), temps)
    pre_belts = [[x for x in A["p_ids"][a_] if x > 0] for a_ in pre_i]
    for k, b_ in enumerate(post_i):
        h = int(round(scal[b_, S["hp"]] * 100)); kept = tuple(i for i, x in enumerate(pre_belts[k]) if x in set(A["p_ids"][b_]))
        if (k, kept) in tables and not final_boss[k] and 1 <= h < len(tables[(k, kept)]):
            gaps.append(tables[(k, kept)][h] - v_post[k])
    gaps = np.array(gaps)
    tf.update({"death_value": 0.0, "final_boss_value_is_1": bool(all(tables[(k, s)][1:].min() == 1 for (k, s) in tables if final_boss[k])),
               "hp_steps_decreasing": int(dec_steps), "hp_steps_total": int(steps),
               "potion_subset_wrong_over_0.5pt": round(pot_wrong / max(1, pot_n), 4),
               "tf_minus_actual_post_state_pts": {"n": int(len(gaps)), "mean": round(100 * float(gaps.mean()), 2),
                                                  "median_abs": round(100 * float(np.median(np.abs(gaps))), 2),
                                                  "p90_abs": round(100 * float(np.quantile(np.abs(gaps), 0.9)), 2)}})
    res["tf_builder"] = tf
    print("T_f", json.dumps(tf), flush=True)

    # ---- 5: policy (per member, from training results) and the G1 table
    pol = [json.load(open(os.path.join(d, "results.json"))) for d in dirs]
    res["policy_test_top1_mean"] = {k: round(float(np.mean([p["test_policy"][k]["top1"] for p in pol])), 4)
                                    for k in pol[0]["test_policy"] if isinstance(pol[0]["test_policy"][k], dict)}
    res["card_reward_pickrate_baseline"] = pol[0]["test_card_reward_pickrate_baseline"]["top1"]
    e, c = res["test"]["ensemble_calibrated"], res["test"]["target_cell_calibrated"]
    worst_dom = max(v["wrong_over_0.5pt"] for k, v in probes.items() if k != "hp_plus10")
    res["G1"] = {
        "a_calibration": {"ece_bands_all": e["ece_within_floor_bands"], "ece_bands_cell": c["ece_within_floor_bands"],
                          "worst_band_gap_pts_all": e["worst_floor_band_mean_gap_pts"], "worst_band_gap_pts_cell": c["worst_floor_band_mean_gap_pts"],
                          "pass": e["ece_within_floor_bands"] <= 0.02 and c["ece_within_floor_bands"] <= 0.02},
        "b_dominance": {"hp_wrong": probes["hp_plus10"]["wrong_over_0.5pt"], "worst_other": worst_dom,
                        "pass": probes["hp_plus10"]["wrong_over_0.5pt"] == 0 and worst_dom <= 0.01},
        "c_accuracy": {"test_logloss": e["logloss"], "auc_within_floor": e["auc_within_floor"],
                       "baselines_100k": {"noattn": [0.4800, 0.7864], "transformer": [0.4841, 0.7830]},
                       "pass": e["logloss"] < 0.4800 and e["auc_within_floor"] > 0.7864},
        "d_tf_builder": {"pass": tf["hp_steps_decreasing"] == 0 and tf["final_boss_value_is_1"] and tf["batched_vs_direct_max_abs_pts"] < 0.5},
        "e_policy_vs_pickrate": {"card_reward_top1": res["policy_test_top1_mean"].get("CARD_REWARD"), "baseline": res["card_reward_pickrate_baseline"]}}
    res["elapsed_s"] = round(time.time() - t0)
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print("G1", json.dumps(res["G1"], indent=1, default=float))


if __name__ == "__main__":
    main()
