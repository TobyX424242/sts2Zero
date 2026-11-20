#!/usr/bin/env python3
"""T1: Independent value calibration audit for s3ic (LAPTOP_GPU_PROPOSAL.md).

Evaluates s3ic_s008 macro.pt on:
  1. s3ic_s008 carry value window (50,000 runs, ~830k rows; in-window, huge N for tight CIs)
     - both all runs and non-perturbed runs separately
  2. strictly held-out runs from s3ic_s011 carry (runs ingested after s3ic_s008 finished)
  3. independent aggregation of fresh-run pre-training metrics from s3ic_s006..s008 macro_train_log.jsonl

Checks both prob-avg (train_macro_s3.py --prob-avg) and logit-avg (Ensemble.serve_values, used for T_f tables),
as well as Member 0 vs Members 1-4. Outputs JSON + human-readable summary.
"""
import argparse, glob, json, math, os, sys, time
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "s3", "py"))

import numpy as np
import torch
import macro_s3 as ms

M_KIND, M_FLOOR, M_ACT = 1, 2, 14
ROOM_BOSS = 5
N_ACTS = 3
HP_EDGES = (0.2, 0.4, 0.6, 0.8)
ASC_GROUPS = (("A0-8", lambda a: a <= 8), ("A9", lambda a: a == 9), ("A10", lambda a: a >= 10))


def wilson_ci(k, n, z=1.959963984540054):
    if n <= 0:
        return (None, None)
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2.0 * n)) / den
    rad = (z / den) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, ctr - rad), min(1.0, ctr + rad))


def cluster_mean_ci(vals, run_ids, z=1.959963984540054):
    """Mean and 95% CI with run-clustered SE (observations within a run are correlated)."""
    n = len(vals)
    if n == 0:
        return None, None, None, 0
    mu = float(vals.mean())
    uniq, inv = np.unique(run_ids, return_inverse=True)
    n_runs = len(uniq)
    if n_runs <= 1:
        return mu, mu, mu, n_runs
    # Delta method for ratio sum_g S_g / sum_g m_g
    m_g = np.bincount(inv).astype(np.float64)
    s_g = np.bincount(inv, weights=vals.astype(np.float64))
    resid_g = s_g - mu * m_g
    se = math.sqrt((n_runs / (n_runs - 1.0)) * float((resid_g ** 2).sum())) / float(m_g.sum())
    return mu, mu - z * se, mu + z * se, n_runs


def bce(z, v):
    v = np.clip(v, 1e-6, 1.0 - 1e-6)
    return -(z * np.log(v) + (1.0 - z) * np.log(1.0 - v))


def auc_in_floor(s, y, floor):
    num = den = 0.0
    for f in np.unique(floor):
        m = floor == f
        yy = y[m]
        n1 = float(yy.sum())
        n0 = float(len(yy) - n1)
        if n1 == 0 or n0 == 0:
            continue
        r = np.empty(len(yy))
        r[np.argsort(s[m], kind="stable")] = np.arange(1, len(yy) + 1)
        num += r[yy].sum() - n1 * (n1 + 1.0) / 2.0
        den += n1 * n0
    return round(num / den, 4) if den >= 400 else None


@torch.no_grad()
def predict_members(ens, ints, floats, dev, bs=4096):
    """Returns member logits [5, N] float32 on CPU."""
    n_mem = len(ens.nets)
    out = [np.empty(len(ints), dtype=np.float32) for _ in range(n_mem)]
    for i in range(0, len(ints), bs):
        b = ms.rows_to_batch(ints[i:i + bs], floats[i:i + bs], dev)
        for j, net in enumerate(ens.nets):
            out[j][i:i + bs] = net.forward_rows(b, policy=False)[2].float().cpu().numpy()
    return np.stack(out, axis=0)


def load_carry_arrays(path, slice_last_runs=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck.get("magic") == "S3MCARRY1", f"unexpected magic in {path}"
    n_rows = ck["n_rows"].numpy().astype(np.int64)
    total_runs_seen = int(ck["total_runs"])
    n_runs_in_carry = len(n_rows)
    start_run = 0 if slice_last_runs is None else max(0, n_runs_in_carry - slice_last_runs)
    row_off = int(n_rows[:start_run].sum())
    n_rows_s = n_rows[start_run:]
    ints = ck["ints"].numpy()[row_off:]
    floats = ck["floats"].numpy()[row_off:]
    meta = ck["meta"].numpy()[row_off:]
    ints, floats = ms.pad_rows(ints, floats)
    out = {
        "total_runs_seen": total_runs_seen,
        "n_runs_in_carry": n_runs_in_carry,
        "n_runs_selected": len(n_rows_s),
        "n_rows": n_rows_s,
        "ints": ints,
        "floats": floats,
        "meta": meta,
        "won": ck["won"].numpy()[start_run:],
        "perturbed": ck["perturbed"].numpy()[start_run:] > 0.5,
        "asc": ck["asc"].numpy()[start_run:].astype(np.int64),
        "act": ck["act"].numpy()[start_run:].astype(np.int64),
        "progress": ck["progress"].numpy()[start_run:],
        "version": int(ck.get("version", -1)),
    }
    return out


def analyze_split(name, data, logits_mem, alpha=0.70, exclude_perturbed=False):
    n_rows = data["n_rows"]
    rep = lambda x: np.repeat(np.asarray(x), n_rows)
    pert_row = rep(data["perturbed"])
    fl = (data["meta"][:, M_KIND] == 0)
    if exclude_perturbed:
        fl = fl & (~pert_row)

    ints = data["ints"][fl]
    floats = data["floats"][fl]
    meta = data["meta"][fl]
    lg = logits_mem[:, fl].astype(np.float64)

    asc = rep(data["asc"])[fl]
    won = (rep(data["won"])[fl] > 0.5)
    progress = rep(data["progress"])[fl]
    z = alpha * won.astype(np.float64) + (1.0 - alpha) * progress.astype(np.float64)
    run_id = rep(np.arange(len(n_rows)))[fl]
    cleared = rep(np.where(data["won"] > 0.5, N_ACTS, np.minimum(data["act"], N_ACTS - 1)))[fl]

    act = meta[:, M_ACT].astype(np.int64)
    floor = meta[:, M_FLOOR].astype(np.int64)
    hp_frac = floats[:, ms.S["hp_frac"]]
    hp_bin = np.digitize(hp_frac, HP_EDGES)

    after_b1 = (act == N_ACTS - 1) & (ints[:, ms.OFF["ctx"][0] + 4] == ROOM_BOSS)
    # runs that reached after_boss1 (within this split)
    b1_runs = set(np.unique(run_id[after_b1]).tolist())
    b1_row = np.isin(run_id, list(b1_runs)) if b1_runs else np.zeros(len(run_id), dtype=bool)
    nxt = np.where(act == 0, cleared >= 1, np.where(act == 1, cleared >= 2, won | b1_row))

    p_mem = 1.0 / (1.0 + np.exp(-np.clip(lg, -30.0, 30.0)))
    v_prob = p_mem.mean(axis=0)                                        # --prob-avg (train_macro_s3)
    v_logit = 1.0 / (1.0 + np.exp(-np.clip(lg.mean(axis=0), -30, 30))) # serve_values (T_f tables)
    v_m0 = p_mem[0]
    v_m14 = p_mem[1:].mean(axis=0)
    logit_sd = lg.std(axis=0)

    # Overall & 9-cell table baselines
    grp = np.where(asc >= 10, 2, np.where(asc == 9, 1, 0)) * 3 + np.clip(act, 0, 2)
    tab = np.zeros_like(z)
    for g in np.unique(grp):
        tab[grp == g] = z[grp == g].mean()

    def summary_ll(mask):
        zz, vp, vl, tt = z[mask], v_prob[mask], v_logit[mask], tab[mask]
        ll_c = float(bce(zz, np.full_like(zz, zz.mean())).mean())
        ll_t = float(bce(zz, tt).mean())
        ll_p = float(bce(zz, vp).mean())
        ll_l = float(bce(zz, vl).mean())
        ent = float(bce(zz, zz).mean())
        return {
            "rows": int(mask.sum()),
            "runs": int(len(np.unique(run_id[mask]))),
            "z_mean": round(float(zz.mean()), 4),
            "v_prob_mean": round(float(vp.mean()), 4),
            "v_logit_mean": round(float(vl.mean()), 4),
            "ll_const": round(ll_c, 4),
            "ll_table": round(ll_t, 4),
            "ll_prob_avg": round(ll_p, 4),
            "ll_logit_avg": round(ll_l, 4),
            "gain_over_table_prob": round(ll_t - ll_p, 4),
            "gain_over_table_logit": round(ll_t - ll_l, 4),
            "room": round(ll_t - ent, 4),
            "share_prob": round((ll_t - ll_p) / max(ll_t - ent, 1e-9), 4),
            "r2_prob": round(float(1.0 - np.var(zz - vp) / max(np.var(zz), 1e-9)), 4),
        }

    res = {
        "split": name,
        "exclude_perturbed": exclude_perturbed,
        "alpha": alpha,
        "overall": summary_ll(np.ones(len(z), dtype=bool)),
        "a10_all": summary_ll(asc >= 10),
        "a0_9_all": summary_ll(asc < 10),
        "cells": {},
        "after_boss1": {},
    }

    for gname, gfn in ASC_GROUPS:
        gmask = np.vectorize(gfn, otypes=[bool])(asc)
        for a in range(N_ACTS):
            m = gmask & (act == a) & (~after_b1)
            if m.sum() < 50:
                continue
            zz, vp, vl, m0, m14, hh, rids = z[m], v_prob[m], v_logit[m], v_m0[m], v_m14[m], hp_bin[m], run_id[m]
            ent = float(bce(zz, zz).mean())
            ll_t = float(bce(zz, np.full_like(zz, zz.mean())).mean())
            ll_p = float(bce(zz, vp).mean())
            ll_l = float(bce(zz, vl).mean())
            z_mu, z_lo, z_hi, nr = cluster_mean_ci(zz, rids)
            diff_mu, diff_lo, diff_hi, _ = cluster_mean_ci(vp - zz, rids)
            hp_rows = []
            for k in range(len(HP_EDGES) + 1):
                hk = hh == k
                if not hk.any():
                    hp_rows.append({"bin": k, "rows": 0})
                    continue
                zm, zlo, zhi, hnr = cluster_mean_ci(zz[hk], rids[hk])
                dm, dlo, dhi, _ = cluster_mean_ci(vp[hk] - zz[hk], rids[hk])
                hp_rows.append({
                    "bin": k,
                    "rows": int(hk.sum()),
                    "runs": hnr,
                    "z": round(zm, 4), "z_ci": [round(zlo, 4), round(zhi, 4)],
                    "v_prob": round(float(vp[hk].mean()), 4),
                    "v_logit": round(float(vl[hk].mean()), 4),
                    "gap_prob": round(dm, 4), "gap_ci": [round(dlo, 4), round(dhi, 4)],
                })
            cell = {
                "rows": int(m.sum()), "runs": nr,
                "z": round(z_mu, 4), "z_ci": [round(z_lo, 4), round(z_hi, 4)],
                "v_prob": round(float(vp.mean()), 4),
                "v_logit": round(float(vl.mean()), 4),
                "v_m0": round(float(m0.mean()), 4),
                "v_m14": round(float(m14.mean()), 4),
                "logit_sd": round(float(logit_sd[m].mean()), 3),
                "gap_prob": round(diff_mu, 4), "gap_ci": [round(diff_lo, 4), round(diff_hi, 4)],
                "ll_table": round(ll_t, 4), "ll_prob": round(ll_p, 4), "ll_logit": round(ll_l, 4),
                "room": round(ll_t - ent, 4), "gain_prob": round(ll_t - ll_p, 4),
                "share_prob": round((ll_t - ll_p) / max(ll_t - ent, 1e-9), 4),
                "auc_next": auc_in_floor(vp, nxt[m], floor[m]),
                "auc_win": auc_in_floor(vp, won[m], floor[m]),
                "hp": hp_rows,
            }
            if a > 0:
                banked = (1.0 - alpha) * a / N_ACTS
                cell["below_banked_prob"] = round(float((vp < banked).mean()), 4)
                cell["below_banked_logit"] = round(float((vl < banked).mean()), 4)
            res["cells"][f"{gname}_act{a + 1}"] = cell

        # after_boss1 (relevant at A10 double boss)
        m = gmask & after_b1
        if m.sum() >= 20:
            banked = (1.0 - alpha) * (N_ACTS - 1) / N_ACTS
            p_win_prob = (v_prob[m] - banked) / (1.0 - banked)
            p_win_logit = (v_logit[m] - banked) / (1.0 - banked)
            p_win_m0 = (v_m0[m] - banked) / (1.0 - banked)
            p_win_m14 = (v_m14[m] - banked) / (1.0 - banked)
            w = won[m].astype(np.float64)
            hh = hp_bin[m]
            rids = run_id[m]
            w_lo, w_hi = wilson_ci(int(w.sum()), len(w))
            hp_rows = []
            for k in range(len(HP_EDGES) + 1):
                hk = hh == k
                nk = int(hk.sum())
                if nk == 0:
                    hp_rows.append({"bin": k, "rows": 0})
                    continue
                wk = int(w[hk].sum())
                wlo, whi = wilson_ci(wk, nk)
                # Gap CI using normal/cluster SE on (p_pred - w)
                gp_m, gp_lo, gp_hi, _ = cluster_mean_ci(p_win_prob[hk] - w[hk], rids[hk])
                gl_m, gl_lo, gl_hi, _ = cluster_mean_ci(p_win_logit[hk] - w[hk], rids[hk])
                hp_rows.append({
                    "bin": k, "rows": nk, "wins": wk,
                    "actual": round(wk / nk, 4), "actual_wilson": [round(wlo, 4), round(whi, 4)],
                    "p_win_prob": round(float(p_win_prob[hk].mean()), 4),
                    "p_win_logit": round(float(p_win_logit[hk].mean()), 4),
                    "p_win_m0": round(float(p_win_m0[hk].mean()), 4),
                    "p_win_m14": round(float(p_win_m14[hk].mean()), 4),
                    "gap_prob": round(gp_m, 4), "gap_prob_ci": [round(gp_lo, 4), round(gp_hi, 4)],
                    "gap_logit": round(gl_m, 4), "gap_logit_ci": [round(gl_lo, 4), round(gl_hi, 4)],
                    "below_banked_prob": round(float((v_prob[m][hk] < banked).mean()), 4),
                    "below_banked_logit": round(float((v_logit[m][hk] < banked).mean()), 4),
                })
            # Also compute implied win log loss on after_boss1
            ll_w_const = float(bce(w, np.full_like(w, w.mean())).mean())
            ll_w_prob = float(bce(w, np.clip(p_win_prob, 1e-4, 1 - 1e-4)).mean())
            ll_w_logit = float(bce(w, np.clip(p_win_logit, 1e-4, 1 - 1e-4)).mean())
            res["after_boss1"][gname] = {
                "rows": int(m.sum()), "runs": int(len(np.unique(rids))), "wins": int(w.sum()),
                "actual": round(float(w.mean()), 4), "actual_wilson": [round(w_lo, 4), round(w_hi, 4)],
                "p_win_prob": round(float(p_win_prob.mean()), 4),
                "p_win_logit": round(float(p_win_logit.mean()), 4),
                "p_win_m0": round(float(p_win_m0.mean()), 4),
                "p_win_m14": round(float(p_win_m14.mean()), 4),
                "auc_prob": auc_in_floor(p_win_prob, won[m], np.zeros(len(w), dtype=np.int64)),
                "auc_logit": auc_in_floor(p_win_logit, won[m], np.zeros(len(w), dtype=np.int64)),
                "ll_win_const": round(ll_w_const, 4),
                "ll_win_prob": round(ll_w_prob, 4),
                "ll_win_logit": round(ll_w_logit, 4),
                "hp": hp_rows,
            }
    return res


def pool_fresh_logs(seg_glob):
    """Aggregate pre-training fresh-run value_cells across all iterations of s006-s008 macro_train_log.jsonl."""
    files = sorted(glob.glob(seg_glob))
    b1_n = np.zeros(5, dtype=np.int64)
    b1_w = np.zeros(5, dtype=np.float64)
    b1_p = np.zeros(5, dtype=np.float64)
    a1_n = np.zeros(5, dtype=np.int64)
    a1_z = np.zeros(5, dtype=np.float64)
    a1_v = np.zeros(5, dtype=np.float64)
    per_seg = {}
    for f in files:
        seg = f.split("/")[-3]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        sn = np.zeros(5, dtype=np.int64); sw = np.zeros(5); sp = np.zeros(5)
        for r in rows:
            vc = r.get("value_cells", {})
            b1 = vc.get("A10_after_boss1")
            if b1:
                for k, item in enumerate(b1["hp"]):
                    if item[0] > 0:
                        sn[k] += item[0]; sw[k] += item[0] * item[1]; sp[k] += item[0] * item[2]
                        b1_n[k] += item[0]; b1_w[k] += item[0] * item[1]; b1_p[k] += item[0] * item[2]
            a1 = vc.get("A10_act1")
            if a1:
                for k, item in enumerate(a1["hp"]):
                    if item[0] > 0:
                        a1_n[k] += item[0]; a1_z[k] += item[0] * item[1]; a1_v[k] += item[0] * item[2]
        per_seg[seg] = {"iters": len(rows), "b1_rows": int(sn.sum())}
    b1_out = []
    for k in range(5):
        nk = int(b1_n[k])
        wk = int(round(b1_w[k]))
        wlo, whi = wilson_ci(wk, nk)
        pm = b1_p[k] / nk if nk else None
        am = wk / nk if nk else None
        b1_out.append({
            "bin": k, "rows": nk, "wins": wk,
            "actual": round(am, 4) if am is not None else None,
            "actual_wilson": [round(wlo, 4), round(whi, 4)] if wlo is not None else None,
            "p_win": round(pm, 4) if pm is not None else None,
            "gap": round(pm - am, 4) if am is not None else None,
        })
    a1_out = []
    for k in range(5):
        nk = int(a1_n[k])
        a1_out.append({
            "bin": k, "rows": nk,
            "z": round(a1_z[k] / nk, 4) if nk else None,
            "v": round(a1_v[k] / nk, 4) if nk else None,
            "gap": round((a1_v[k] - a1_z[k]) / nk, 4) if nk else None,
        })
    return {"per_seg": per_seg, "A10_after_boss1": b1_out, "A10_act1": a1_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macro", required=True)
    ap.add_argument("--carry-s008", required=True)
    ap.add_argument("--carry-s011", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--bs", type=int, default=4096)
    a = ap.parse_args()

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[t1] loading macro {a.macro} on {dev}", flush=True)
    ens = ms.Ensemble.load(a.macro, dev)
    print(f"[t1] macro version={ens.version}, milestones={ens.milestones}, members={len(ens.nets)}", flush=True)

    out = {"macro": a.macro, "macro_version": ens.version, "alpha": a.alpha, "device": dev}

    print(f"[t1] loading carry s008: {a.carry_s008}", flush=True)
    d8 = load_carry_arrays(a.carry_s008)
    print(f"[t1] s008 carry: total_runs_seen={d8['total_runs_seen']}, n_runs={d8['n_runs_selected']}, rows={len(d8['meta'])}, version={d8['version']}", flush=True)
    t1 = time.time()
    lg8 = predict_members(ens, d8["ints"], d8["floats"], dev, a.bs)
    print(f"[t1] evaluated s008 carry ({len(d8['meta'])} rows x 5 members) in {time.time() - t1:.1f}s", flush=True)

    out["s008_carry_all"] = analyze_split("s008_carry_all", d8, lg8, a.alpha, exclude_perturbed=False)
    out["s008_carry_clean"] = analyze_split("s008_carry_clean", d8, lg8, a.alpha, exclude_perturbed=True)
    s008_total = d8["total_runs_seen"]
    del d8, lg8

    if a.carry_s011 and os.path.exists(a.carry_s011):
        # First peek total_runs to find strictly post-s008 runs
        ck11 = torch.load(a.carry_s011, map_location="cpu", weights_only=False)
        s011_total = int(ck11["total_runs"])
        new_runs = s011_total - s008_total
        del ck11
        print(f"[t1] s011 carry: total_runs_seen={s011_total}, strictly post-s008 runs={new_runs}", flush=True)
        if new_runs > 0:
            d11 = load_carry_arrays(a.carry_s011, slice_last_runs=new_runs)
            t2 = time.time()
            lg11 = predict_members(ens, d11["ints"], d11["floats"], dev, a.bs)
            print(f"[t1] evaluated s011 post-s008 ({d11['n_runs_selected']} runs, {len(d11['meta'])} rows) in {time.time() - t2:.1f}s", flush=True)
            out["s011_heldout_all"] = analyze_split("s011_heldout_all", d11, lg11, a.alpha, exclude_perturbed=False)
            out["s011_heldout_clean"] = analyze_split("s011_heldout_clean", d11, lg11, a.alpha, exclude_perturbed=True)
            del d11, lg11

    out["fresh_logs_s006_s008"] = pool_fresh_logs(os.path.join(REPO, "src/server_backup/s3ic/s3ic_s00[678]/meta/macro_train_log.jsonl"))
    out["elapsed_s"] = round(time.time() - t0, 1)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[t1] wrote {a.out} in {out['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
