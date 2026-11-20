#!/usr/bin/env python3
"""T2 held-out evaluator for offline macro replay arms (LAPTOP_GPU_PROPOSAL.md, REPR_TEST_PLAN.md).

Measures per checkpoint (arm) on strictly held-out future runs (+ train split for train-test gap):
  - Held-out value log loss (prob_avg and logit_avg), gain over 9-cell table, train-test gap
  - Residual AUC within (floor, HP fifth) for next milestone and for win
  - T1 calibration cells including A10_after_boss1 by HP fifth (actual vs predicted, Q5-Q1 spread, win LL)
  - Member spread on one-card dV (sd across members 0..4 and 1..4 on sampled CARD_REWARD decisions)
  - Agreement of raw Member-0 card argmax with look_beta=20 lookahead argmax (both own ensemble and frozen ref)
  - Policy training diagnostics parsed from macro_train_log.jsonl (pre-PPO lag KL, clip, policy_steps, etc.)
"""
import argparse, json, math, os, sys, time
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "s3", "py"))

import numpy as np
import torch
import macro_s3 as ms
import train_macro_s3 as tm

ROOM_BOSS = 5
N_ACTS = 3
HP_EDGES = (0.2, 0.4, 0.6, 0.8)
ASC_GROUPS = (("A0-8", lambda a: a <= 8), ("A9", lambda a: a == 9), ("A10", lambda a: a >= 10))


def bce(z, v):
    v = np.clip(v, 1e-6, 1.0 - 1e-6)
    return -(z * np.log(v) + (1.0 - z) * np.log(1.0 - v))


def wilson_ci(k, n, z=1.959963984540054):
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2.0 * n)) / den
    rad = (z / den) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return [round(max(0.0, ctr - rad), 4), round(min(1.0, ctr + rad), 4)]


def auc_grouped(score, y, group_key):
    """Weighted pairwise AUC within exact groups (e.g. (floor, hp_bin) or floor)."""
    num = den = 0.0
    for g in np.unique(group_key):
        m = group_key == g
        yy = y[m]
        n1 = float(yy.sum())
        n0 = float(len(yy) - n1)
        if n1 == 0 or n0 == 0:
            continue
        r = np.empty(len(yy))
        r[np.argsort(score[m], kind="stable")] = np.arange(1, len(yy) + 1)
        num += r[yy].sum() - n1 * (n1 + 1.0) / 2.0
        den += n1 * n0
    return round(num / den, 4) if den >= 400 else None


def load_chunks_list(list_path, max_chunks=0):
    files = [l.strip() for l in open(list_path) if l.strip()]
    if max_chunks > 0:
        files = files[:max_chunks]
    runs_all, ints_all, floats_all, meta_all = [], [], [], []
    row_off = 0
    for f in files:
        runs, ints, floats, meta = tm.read_chunk(f)
        for r in runs:
            a, n = int(r[9]), int(r[10])
            rec = {
                "a": row_off + a, "n": n, "asc": int(r[2]), "won": float(r[3]), "floors": int(r[4]),
                "act": int(r[5]), "perturbed": bool(r[8]),
                "progress": 1.0 if r[3] else min(int(r[5]), N_ACTS - 1) / N_ACTS,
            }
            runs_all.append(rec)
        ints_all.append(ints); floats_all.append(floats); meta_all.append(meta)
        row_off += len(meta)
    ints = np.concatenate(ints_all); floats = np.concatenate(floats_all); meta = np.concatenate(meta_all)
    n_rows = np.array([r["n"] for r in runs_all], dtype=np.int64)
    return {
        "runs": runs_all, "n_rows": n_rows, "ints": ints, "floats": floats, "meta": meta,
        "asc": np.repeat([r["asc"] for r in runs_all], n_rows),
        "won": np.repeat([r["won"] for r in runs_all], n_rows),
        "act_end": np.repeat([r["act"] for r in runs_all], n_rows),
        "perturbed": np.repeat([r["perturbed"] for r in runs_all], n_rows),
        "progress": np.repeat([r["progress"] for r in runs_all], n_rows),
        "run_id": np.repeat(np.arange(len(runs_all)), n_rows),
    }


@torch.no_grad()
def eval_member_probs(ens, ints, floats, dev, alpha=0.70, boss1_share=0.0, bs=2048):
    """Returns per-member probabilities [5, N] (using milestone blend when ens.milestones is True),
    plus ensemble prob_avg [N] and serve_values logit_avg [N]."""
    n_mem = len(ens.nets)
    N = len(ints)
    pmem = np.empty((n_mem, N), dtype=np.float32)
    v_serve = np.empty(N, dtype=np.float32)
    for i in range(0, N, bs):
        b = ms.rows_to_batch(ints[i:i + bs], floats[i:i + bs], dev)
        if ens.milestones:
            st = ms.stage_of(b["ctx"][:, 3], b["ctx"][:, 4])
            dbl = b["ctx"][:, 1] >= ms.DOUBLE_BOSS_ASC
            for j, n in enumerate(ens.nets):
                lg = n.forward_ms(b)[2]
                pmem[j, i:i + bs] = ms.blend(torch.sigmoid(lg), st, dbl, ens.alpha, ens.boss1_share).float().cpu().numpy()
            o = ens.serve_values(b)
            v_serve[i:i + bs] = torch.sigmoid(o[:, ms.O_V]).float().cpu().numpy()
        else:
            L = []
            for j, n in enumerate(ens.nets):
                lg = n.forward_rows(b, policy=False)[2]
                L.append(lg)
                pmem[j, i:i + bs] = torch.sigmoid(lg).float().cpu().numpy()
            v_serve[i:i + bs] = torch.sigmoid(torch.stack(L).mean(0)).float().cpu().numpy()
    v_prob = pmem.mean(0)
    return pmem, v_prob, v_serve


def value_metrics(data, pmem, v_prob, v_serve, alpha=0.70):
    fl = (data["meta"][:, tm.M_KIND] == 0)
    ints, floats, meta = data["ints"][fl], data["floats"][fl], data["meta"][fl]
    asc = data["asc"][fl]; won = data["won"][fl] > 0.5; progress = data["progress"][fl]
    run_id = data["run_id"][fl]
    z = alpha * won.astype(np.float64) + (1.0 - alpha) * progress.astype(np.float64)
    vp = v_prob[fl].astype(np.float64); vl = v_serve[fl].astype(np.float64)
    pm = pmem[:, fl].astype(np.float64)

    act = meta[:, tm.M_ACT].astype(np.int64)
    floor = meta[:, tm.M_FLOOR].astype(np.int64)
    hp_bin = np.digitize(floats[:, ms.S["hp_frac"]], HP_EDGES)
    fh_key = floor * 8 + hp_bin   # exact (floor, HP fifth) cell for residual AUC

    after_b1 = (act == N_ACTS - 1) & (ints[:, ms.OFF["ctx"][0] + 4] == ROOM_BOSS)
    b1_runs = set(np.unique(run_id[after_b1]).tolist())
    b1_row = np.isin(run_id, list(b1_runs)) if b1_runs else np.zeros(len(run_id), dtype=bool)
    cleared = np.where(won, N_ACTS, np.minimum(data["act_end"][fl], N_ACTS - 1))
    nxt = np.where(act == 0, cleared >= 1, np.where(act == 1, cleared >= 2, won | b1_row))

    grp = np.where(asc >= 10, 2, np.where(asc == 9, 1, 0)) * 3 + np.clip(act, 0, 2)
    tab = np.zeros_like(z)
    for g in np.unique(grp):
        tab[grp == g] = z[grp == g].mean()

    ll_tab = float(bce(z, tab).mean())
    ll_p = float(bce(z, vp).mean())
    ll_l = float(bce(z, vl).mean())
    ent = float(bce(z, z).mean())
    a10 = asc >= 10
    ll_a10_tab = float(bce(z[a10], tab[a10]).mean())
    ll_a10_p = float(bce(z[a10], vp[a10]).mean())
    ll_a10_l = float(bce(z[a10], vl[a10]).mean())

    out = {
        "rows": int(fl.sum()),
        "ll_table": round(ll_tab, 5),
        "ll_prob": round(ll_p, 5),
        "ll_logit": round(ll_l, 5),
        "gain_table": round(ll_tab - ll_p, 5),
        "share": round((ll_tab - ll_p) / max(ll_tab - ent, 1e-9), 4),
        "ll_a10_table": round(ll_a10_tab, 5),
        "ll_a10_prob": round(ll_a10_p, 5),
        "ll_a10_logit": round(ll_a10_l, 5),
        "res_auc_win_all": auc_grouped(vp, won, asc * 1000 + fh_key),
        "res_auc_win_a10": auc_grouped(vp[a10], won[a10], fh_key[a10]),
        "res_auc_next_a10": auc_grouped(vp[a10], nxt[a10], fh_key[a10]),
        "cells": {},
    }
    for a in range(N_ACTS):
        m = a10 & (act == a) & (~after_b1)
        if m.sum() < 50:
            continue
        ent_c = float(bce(z[m], z[m]).mean())
        llt = float(bce(z[m], np.full(m.sum(), z[m].mean())).mean())
        llp = float(bce(z[m], vp[m]).mean())
        out["cells"][f"A10_act{a + 1}"] = {
            "rows": int(m.sum()), "z": round(float(z[m].mean()), 4), "v_p": round(float(vp[m].mean()), 4),
            "v_l": round(float(vl[m].mean()), 4), "m0": round(float(pm[0, m].mean()), 4),
            "m14": round(float(pm[1:, m].mean()), 4),
            "ll_p": round(llp, 4), "share": round((llt - llp) / max(llt - ent_c, 1e-9), 4),
            "res_auc_win": auc_grouped(vp[m], won[m], fh_key[m]),
            "res_auc_next": auc_grouped(vp[m], nxt[m], fh_key[m]),
            "below_banked": round(float((vp[m] < (1.0 - alpha) * a / N_ACTS).mean()), 4) if a > 0 else 0.0,
        }

    mb1 = a10 & after_b1
    if mb1.sum() >= 20:
        banked = (1.0 - alpha) * (N_ACTS - 1) / N_ACTS
        pw_p = (vp[mb1] - banked) / (1.0 - banked)
        pw_l = (vl[mb1] - banked) / (1.0 - banked)
        pw_m0 = (pm[0, mb1] - banked) / (1.0 - banked)
        pw_m14 = (pm[1:, mb1].mean(0) - banked) / (1.0 - banked)
        w = won[mb1].astype(np.float64)
        hh = hp_bin[mb1]
        hp_list = []
        for k in range(5):
            hk = hh == k
            nk = int(hk.sum())
            if nk == 0:
                hp_list.append({"bin": k, "n": 0}); continue
            wk = int(w[hk].sum())
            hp_list.append({
                "bin": k, "n": nk, "actual": round(wk / nk, 4), "wilson": wilson_ci(wk, nk),
                "p_prob": round(float(pw_p[hk].mean()), 4), "p_logit": round(float(pw_l[hk].mean()), 4),
                "gap_prob": round(float(pw_p[hk].mean() - wk / nk), 4),
                "gap_logit": round(float(pw_l[hk].mean() - wk / nk), 4),
                "m0": round(float(pw_m0[hk].mean()), 4), "m14": round(float(pw_m14[hk].mean()), 4),
            })
        out["A10_after_boss1"] = {
            "n": int(mb1.sum()), "actual": round(float(w.mean()), 4),
            "p_prob": round(float(pw_p.mean()), 4), "p_logit": round(float(pw_l.mean()), 4),
            "auc_prob": auc_grouped(pw_p, won[mb1], np.zeros(len(w), dtype=np.int64)),
            "ll_win_const": round(float(bce(w, np.full_like(w, w.mean())).mean()), 4),
            "ll_win_prob": round(float(bce(w, np.clip(pw_p, 1e-4, 1 - 1e-4)).mean()), 4),
            "ll_win_logit": round(float(bce(w, np.clip(pw_l, 1e-4, 1 - 1e-4)).mean()), 4),
            "q5_q1_actual": round(hp_list[4]["actual"] - hp_list[0]["actual"], 4) if hp_list[0]["n"] and hp_list[4]["n"] else None,
            "q5_q1_prob": round(hp_list[4]["p_prob"] - hp_list[0]["p_prob"], 4) if hp_list[0]["n"] and hp_list[4]["n"] else None,
            "q5_q1_logit": round(hp_list[4]["p_logit"] - hp_list[0]["p_logit"], 4) if hp_list[0]["n"] and hp_list[4]["n"] else None,
            "hp": hp_list,
        }
    return out


@torch.no_grad()
def card_dv_and_agreement(ens, ref_ens, data, dev, n_cards=1500, look_beta=20.0, seed=0):
    """On a fixed sample of held-out CARD_REWARD decisions:
    1) one-card dV member disagreement (sd across all 5 members and across members 1..4) in logit and prob space;
    2) agreement of raw Member-0 card argmax with look_beta=20 lookahead argmax (both own ensemble and frozen ref)."""
    rng = np.random.default_rng(seed)
    ok_off = ms.OFF["opt_kind"][0]; oi_off = ms.OFF["opt_id"][0]; ou_off = ms.OFF["opt_upg"][0]
    d_off = ms.OFF["d_ids"][0]; du_off = ms.OFF["d_upg"][0]; de_off = ms.OFF["d_ench"][0]
    dea_off = ms.OFF["d_eamt"][0]; da_off = ms.OFF["d_age"][0]; dt_off = ms.OFF["dtype"][0]

    ints, floats, meta = data["ints"], data["floats"], data["meta"]
    dec = meta[:, tm.M_KIND] == 1
    dt = meta[:, tm.M_DTYPE].astype(np.int64)
    dids = ints[:, d_off:d_off + ms.MAX_DECK]
    deck_n = (dids > 0).sum(1)
    opt_kind = ints[:, ok_off:ok_off + ms.K]
    elig = np.flatnonzero(dec & (dt == 0) & (deck_n < ms.MAX_DECK) & (opt_kind == 1).any(1))
    if len(elig) > n_cards:
        elig = np.sort(rng.choice(elig, n_cards, replace=False))

    # Build base + post rows
    ri_list, rf_list, owners, slots = [], [], [], []
    for k, idx in enumerate(elig):
        bi = ints[idx].copy(); bf = floats[idx].copy()
        bi[dt_off] = -1; bi[ok_off:ok_off + ms.K] = 0; bi[oi_off:oi_off + ms.K] = 0; bi[ou_off:ou_off + ms.K] = 0
        ri_list.append(bi); rf_list.append(bf); owners.append(k); slots.append(-1)
        j = int(deck_n[idx])
        for s in range(ms.K):
            if opt_kind[idx, s] != 1:
                continue
            ri = bi.copy(); rf = bf.copy()
            upg = int(np.clip(ints[idx, ou_off + s], 0, 3))
            ri[d_off + j] = ints[idx, oi_off + s]; ri[du_off + j] = upg
            ri[de_off + j] = 0; ri[dea_off + j] = 0; ri[da_off + j] = 0
            rf[ms.S["deck_size"]] = np.float32((j + 1) / 40.0)
            if upg > 0:
                rf[ms.S["n_upgraded"]] = np.float32(rf[ms.S["n_upgraded"]] + 1.0 / 20.0)
            ri_list.append(ri); rf_list.append(rf); owners.append(k); slots.append(s)

    RI = np.asarray(ri_list, dtype=ints.dtype); RF = np.asarray(rf_list, dtype=floats.dtype)

    def get_dz(e):
        pmem, _, v_serve = eval_member_probs(e, RI, RF, dev, bs=2048)
        lg_mem = np.log(np.clip(pmem, 1e-6, 1 - 1e-6) / np.clip(1 - pmem, 1e-6, 1 - 1e-6))
        lg_srv = np.log(np.clip(v_serve, 1e-6, 1 - 1e-6) / np.clip(1 - v_serve, 1e-6, 1 - 1e-6))
        z0_m = np.zeros((5, len(elig)), dtype=np.float64)
        p0_m = np.zeros((5, len(elig)), dtype=np.float64)
        z0_s = np.zeros(len(elig), dtype=np.float64)
        dz_s = np.zeros((len(elig), ms.K), dtype=np.float64)
        sd_lg_all, sd_lg_14, sd_p_all, sd_p_14, abs_dz = [], [], [], [], []
        for idx_r, (own, sl) in enumerate(zip(owners, slots)):
            if sl < 0:
                z0_m[:, own] = lg_mem[:, idx_r]; p0_m[:, own] = pmem[:, idx_r]; z0_s[own] = lg_srv[idx_r]
            else:
                d_lg = lg_mem[:, idx_r] - z0_m[:, own]
                d_p = pmem[:, idx_r] - p0_m[:, own]
                dz_s[own, sl] = lg_srv[idx_r] - z0_s[own]
                sd_lg_all.append(float(d_lg.std())); sd_lg_14.append(float(d_lg[1:].std()))
                sd_p_all.append(float(d_p.std())); sd_p_14.append(float(d_p[1:].std()))
                abs_dz.append(abs(float(dz_s[own, sl])))
        return dz_s, {
            "dv_logit_sd_all5": round(float(np.mean(sd_lg_all)), 5),
            "dv_logit_sd_m14": round(float(np.mean(sd_lg_14)), 5),
            "dv_prob_sd_all5": round(float(np.mean(sd_p_all)), 5),
            "dv_prob_sd_m14": round(float(np.mean(sd_p_14)), 5),
            "mean_abs_dz": round(float(np.mean(abs_dz)), 5),
        }

    dz_own, dv_stats = get_dz(ens)
    dz_ref, _ = get_dz(ref_ens)

    # Raw policy logits of ens on elig
    b = ms.rows_to_batch(ints[elig], floats[elig], dev)
    p2c = ens.p2_context(b) if ens.p2 else None
    _, _, _, opt_lg, _ = ens.nets[0].forward_rows(b, policy=True, p2_ctx=p2c)
    opt_lg = opt_lg.float().cpu().numpy()
    legal = opt_kind[elig] != 0
    raw_arg = np.where(legal, opt_lg, -1e9).argmax(1)
    look_own_arg = np.where(legal, opt_lg + look_beta * dz_own, -1e9).argmax(1)
    look_ref_arg = np.where(legal, opt_lg + look_beta * dz_ref, -1e9).argmax(1)
    # Also ref policy raw + ref lookahead (fixed target across arms)
    _, _, _, ref_opt_lg, _ = ref_ens.nets[0].forward_rows(b, policy=True)
    ref_opt_lg = ref_opt_lg.float().cpu().numpy()
    fixed_ref_target = np.where(legal, ref_opt_lg + look_beta * dz_ref, -1e9).argmax(1)

    return {
        "n_cards": int(len(elig)),
        **dv_stats,
        "agree_raw_vs_own_look": round(float((raw_arg == look_own_arg).mean()), 4),
        "agree_raw_vs_ref_look": round(float((raw_arg == look_ref_arg).mean()), 4),
        "agree_raw_vs_fixed_ref_target": round(float((raw_arg == fixed_ref_target).mean()), 4),
    }


def parse_train_log(path):
    if not path or not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return {}
    mean_k = lambda k: round(float(np.mean([r[k] for r in rows if k in r])), 5) if any(k in r for r in rows) else None
    max_k = lambda k: round(float(np.max([r[k] for r in rows if k in r])), 5) if any(k in r for r in rows) else None
    return {
        "iters": len(rows),
        "policy_steps_mean": mean_k("policy_steps"),
        "policy_steps_per_iter": [r.get("policy_steps") for r in rows],
        "early_stops": sum(1 for r in rows if r.get("early_stop")),
        "clip_mean": mean_k("clip"),
        "clip_noncard_mean": mean_k("clip_noncard"),
        "behaviour_lag_kl_mean": mean_k("behaviour_lag_kl"),
        "behaviour_lag_kl_noncard_mean": mean_k("behaviour_lag_kl_noncard"),
        "ent_last": rows[-1].get("ent"),
        "distill_mean": mean_k("distill"),
        "look_agree_before_iters": [r.get("look_agree_before") for r in rows if "look_agree_before" in r],
        "p2_lag_kl_live_max": max_k("p2_lag_kl_live"),
        "p2_lag_kl_ema_max": max_k("p2_lag_kl_ema"),
        "ms_on_iters": [r.get("ms_on") for r in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="frozen initial checkpoint (s005/macro.pt)")
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--heldout-list", required=True)
    ap.add_argument("--cross-list", required=True, help="strictly cross-segment heldout list (s007 seq 4)")
    ap.add_argument("--arms", nargs="+", required=True, help="name=MACRO.pt[:LOG.jsonl]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.70)
    a = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[t2_eval] loading data on {dev}", flush=True)
    d_tr = load_chunks_list(a.train_list, max_chunks=60)   # 1,920 train runs sample for train-test gap
    d_ho = load_chunks_list(a.heldout_list)
    d_cr = load_chunks_list(a.cross_list)
    print(f"[t2_eval] train_sample={len(d_tr['runs'])} runs, heldout={len(d_ho['runs'])} runs, cross_seg={len(d_cr['runs'])} runs", flush=True)

    ref_ens = ms.Ensemble.load(a.ref, dev)
    results = {"ref": a.ref, "arms": {}}
    for spec in a.arms:
        name, rest = spec.split("=", 1)
        mpath, _, lpath = rest.partition(":")
        ta = time.time()
        print(f"[t2_eval] evaluating arm {name}: {mpath}", flush=True)
        ens = ms.Ensemble.load(mpath, dev)
        pm_tr, vp_tr, vl_tr = eval_member_probs(ens, d_tr["ints"], d_tr["floats"], dev, a.alpha)
        pm_ho, vp_ho, vl_ho = eval_member_probs(ens, d_ho["ints"], d_ho["floats"], dev, a.alpha)
        pm_cr, vp_cr, vl_cr = eval_member_probs(ens, d_cr["ints"], d_cr["floats"], dev, a.alpha)
        m_tr = value_metrics(d_tr, pm_tr, vp_tr, vl_tr, a.alpha)
        m_ho = value_metrics(d_ho, pm_ho, vp_ho, vl_ho, a.alpha)
        m_cr = value_metrics(d_cr, pm_cr, vp_cr, vl_cr, a.alpha)
        dv = card_dv_and_agreement(ens, ref_ens, d_ho, dev)
        tr_log = parse_train_log(lpath)
        results["arms"][name] = {
            "macro": mpath, "milestones": ens.milestones, "p2": ens.p2, "version": ens.version,
            "train_ll_prob": m_tr["ll_prob"], "train_a10_ll_prob": m_tr["ll_a10_prob"],
            "train_test_gap": round(m_ho["ll_prob"] - m_tr["ll_prob"], 5),
            "train_test_gap_a10": round(m_ho["ll_a10_prob"] - m_tr["ll_a10_prob"], 5),
            "heldout": m_ho, "cross_seg": m_cr, "card_dv_agree": dv, "train_log": tr_log,
        }
        print(f"[t2_eval] {name} done in {time.time() - ta:.1f}s: ho_ll={m_ho['ll_prob']:.5f} (gap {m_ho['ll_prob']-m_tr['ll_prob']:+.5f}), "
              f"res_auc_a10={m_ho['res_auc_win_a10']}, dv_sd={dv['dv_logit_sd_all5']}, agree={dv['agree_raw_vs_fixed_ref_target']}", flush=True)

    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[t2_eval] wrote {a.out} in {results['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
