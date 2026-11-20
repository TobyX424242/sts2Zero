#!/usr/bin/env python3
"""T3: Act-3 boss search budget prep and paired report (LAPTOP_GPU_PROPOSAL.md).

Subcommands:
  prep   STARTS_GLOB OUTDIR   Filter Act-3 boss starts (act_index==2, room_type==boss) from S3 starts_dump files,
                              split out BattleStartSpec -> starts.jsonl, embedded T_f table -> tables.jsonl, and
                              metadata -> meta.json.
  report META.json BASE_JSONL ARM_JSONL...   Compute paired statistics (win rate, Wilson 95% CI, paired diff + 95% CI,
                              exact two-sided McNemar p-value, end HP, T_f score/rel, potions used, seconds/fight)
                              overall, by Boss 1 vs Boss 2 (final boss), and per encounter.
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict
import numpy as np


def wilson_ci(k, n, z=1.959963984540054):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2.0 * n)) / den
    rad = (z / den) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, ctr - rad), min(1.0, ctr + rad))


def mcnemar_exact_p(b, c):
    """Exact two-sided binomial test on discordant pairs (b = arm-only wins, c = base-only wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # sum_{i=0..k} C(n, i) * 0.5^n * 2
    s = 0.0
    log2n = n * math.log(2.0)
    for i in range(k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - log2n)
    return min(1.0, 2.0 * s)


def paired_mean_ci(diffs, z=1.959963984540054):
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mu = float(arr.mean())
    if n == 1:
        return (mu, mu, mu)
    se = float(arr.std(ddof=1)) / math.sqrt(n)
    return (mu, mu - z * se, mu + z * se)


def cmd_prep(a):
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(glob.glob(a.starts))
    if not files:
        raise SystemExit(f"no files matching {a.starts}")
    starts, tables, meta = [], [], {}
    seen = set()
    by_slot = defaultdict(int)
    by_enc = defaultdict(int)
    for p in files:
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            if int(d.get("act_index", -1)) != 2 or d.get("room_type") != "boss":
                continue
            sid = d["id"]
            if sid in seen:
                continue
            seen.add(sid)
            tbl = d.pop("table")
            starts.append(d)
            tables.append(tbl)
            slot = "boss2" if d.get("final_boss") else ("boss1" if d.get("first_of_two_bosses") else "other")
            enc = d.get("encounter_id", "")
            by_slot[slot] += 1
            by_enc[enc] += 1
            meta[sid] = {
                "slot": slot, "encounter": enc, "final_boss": bool(d.get("final_boss")),
                "first_of_two": bool(d.get("first_of_two_bosses")),
                "hp": int(d["hp"]), "max_hp": int(d["max_hp"]), "potions": len(d.get("potions", [])),
                "run_seed": d.get("run_seed", ""),
            }
    with open(os.path.join(a.outdir, "starts.jsonl"), "w") as fs, open(os.path.join(a.outdir, "tables.jsonl"), "w") as ft:
        for s, t in zip(starts, tables):
            fs.write(json.dumps(s) + "\n")
            ft.write(json.dumps(t) + "\n")
    with open(os.path.join(a.outdir, "meta.json"), "w") as fm:
        json.dump({"count": len(starts), "by_slot": dict(by_slot), "by_enc": dict(by_enc), "items": meta}, fm, indent=2)
    print(f"[t3 prep] {len(starts)} Act-3 boss starts from {len(files)} files: slots={dict(by_slot)}, encounters={dict(by_enc)}")


def load_arm(path):
    """Map (id, fight_idx) -> fight record (exact chance-seed pairing even with concurrent reps > 1)."""
    out = {}
    errors = capped = 0
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        sid = d["id"]
        fidx = int(d["fight"])
        if d.get("result") == "error":
            errors += 1
        elif d.get("result") == "capped":
            capped += 1
        out[(sid, fidx)] = d
    return out, errors, capped


def summarize_group(keys, base_d, arm_d):
    n = len(keys)
    if n == 0:
        return None
    bw = np.array([1.0 if base_d[k]["result"] == "won" else 0.0 for k in keys])
    aw = np.array([1.0 if arm_d[k]["result"] == "won" else 0.0 for k in keys])
    b_only = int(((bw == 1) & (aw == 0)).sum())
    a_only = int(((bw == 0) & (aw == 1)).sum())
    dw_mu, dw_lo, dw_hi = paired_mean_ci(aw - bw)
    p_mc = mcnemar_exact_p(a_only, b_only)

    bhp = np.array([base_d[k]["end_hp"] if base_d[k]["result"] == "won" else 0.0 for k in keys])
    ahp = np.array([arm_d[k]["end_hp"] if arm_d[k]["result"] == "won" else 0.0 for k in keys])
    dhp_mu, dhp_lo, dhp_hi = paired_mean_ci(ahp - bhp)

    bscore = np.array([float(base_d[k]["score"]) for k in keys])
    ascore = np.array([float(arm_d[k]["score"]) for k in keys])
    dsc_mu, dsc_lo, dsc_hi = paired_mean_ci(ascore - bscore)

    brel = np.array([float(base_d[k]["rel"]) for k in keys])
    arel = np.array([float(arm_d[k]["rel"]) for k in keys])
    drel_mu, drel_lo, drel_hi = paired_mean_ci(arel - brel)

    bpot = np.array([float(base_d[k]["potions_used"]) for k in keys])
    apot = np.array([float(arm_d[k]["potions_used"]) for k in keys])
    bsec = np.array([float(base_d[k]["seconds"]) for k in keys])
    asec = np.array([float(arm_d[k]["seconds"]) for k in keys])
    bsims = np.array([float(base_d[k]["simulations"]) for k in keys])
    asims = np.array([float(arm_d[k]["simulations"]) for k in keys])

    # Both won end HP diff
    both_w = (bw == 1) & (aw == 1)
    both_hp_d = paired_mean_ci(ahp[both_w] - bhp[both_w]) if both_w.any() else (0.0, 0.0, 0.0)

    b_ci = wilson_ci(int(bw.sum()), n)
    a_ci = wilson_ci(int(aw.sum()), n)
    return {
        "n": n,
        "base_win": round(float(bw.mean()), 4), "base_ci": [round(b_ci[0], 4), round(b_ci[1], 4)],
        "arm_win": round(float(aw.mean()), 4), "arm_ci": [round(a_ci[0], 4), round(a_ci[1], 4)],
        "diff_win": round(dw_mu, 4), "diff_win_ci": [round(dw_lo, 4), round(dw_hi, 4)],
        "arm_only": a_only, "base_only": b_only, "mcnemar_p": round(p_mc, 5),
        "base_hp": round(float(bhp.mean()), 2), "arm_hp": round(float(ahp.mean()), 2),
        "diff_hp": round(dhp_mu, 2), "diff_hp_ci": [round(dhp_lo, 2), round(dhp_hi, 2)],
        "both_won_n": int(both_w.sum()),
        "both_won_diff_hp": round(both_hp_d[0], 2), "both_won_diff_hp_ci": [round(both_hp_d[1], 2), round(both_hp_d[2], 2)],
        "base_score": round(float(bscore.mean()), 4), "arm_score": round(float(ascore.mean()), 4),
        "diff_score": round(dsc_mu, 4), "diff_score_ci": [round(dsc_lo, 4), round(dsc_hi, 4)],
        "base_rel": round(float(brel.mean()), 4), "arm_rel": round(float(arel.mean()), 4),
        "diff_rel": round(drel_mu, 4), "diff_rel_ci": [round(drel_lo, 4), round(drel_hi, 4)],
        "base_pot": round(float(bpot.mean()), 3), "arm_pot": round(float(apot.mean()), 3),
        "base_sec": round(float(bsec.mean()), 3), "arm_sec": round(float(asec.mean()), 3),
        "base_sims": round(float(bsims.mean()), 1), "arm_sims": round(float(asims.mean()), 1),
    }


def cmd_report(a):
    meta = json.load(open(a.meta))["items"]
    arms = {}
    arm_order = []
    for spec in a.arms:
        name, path = spec.split("=", 1)
        d, errs, caps = load_arm(path)
        arms[name] = {"data": d, "errors": errs, "capped": caps, "path": path}
        arm_order.append(name)
    base_name = arm_order[0]
    base_d = arms[base_name]["data"]
    common = sorted(k for k in base_d if all(k in arms[nm]["data"] for nm in arm_order) and k[0] in meta)
    print(f"[t3 report] arms={arm_order}, paired fights={len(common)}")
    for nm in arm_order:
        print(f"  {nm}: total={len(arms[nm]['data'])}, errors={arms[nm]['errors']}, capped={arms[nm]['capped']}")

    groups = {
        "all_act3_boss": common,
        "boss1": [k for k in common if meta[k[0]]["slot"] == "boss1"],
        "boss2_final": [k for k in common if meta[k[0]]["slot"] == "boss2"],
    }
    encs = sorted({meta[k[0]]["encounter"] for k in common})
    for e in encs:
        groups[f"enc:{e}"] = [k for k in common if meta[k[0]]["encounter"] == e]
        groups[f"boss2:{e}"] = [k for k in common if meta[k[0]]["slot"] == "boss2" and meta[k[0]]["encounter"] == e]
        groups[f"boss1:{e}"] = [k for k in common if meta[k[0]]["slot"] == "boss1" and meta[k[0]]["encounter"] == e]

    out = {"arms": {nm: {"errors": arms[nm]["errors"], "capped": arms[nm]["capped"]} for nm in arm_order}, "comparisons": {}}
    for nm in arm_order[1:]:
        comp = {}
        print(f"\n=== Paired: {nm} vs {base_name} ===")
        for gname, ks in groups.items():
            if not ks:
                continue
            s = summarize_group(ks, base_d, arms[nm]["data"])
            comp[gname] = s
            print(f"  {gname:32s} n={s['n']:4d} | win {100*s['base_win']:5.1f}% -> {100*s['arm_win']:5.1f}% "
                  f"({100*s['diff_win']:+5.1f} pp [{100*s['diff_win_ci'][0]:+5.1f},{100*s['diff_win_ci'][1]:+5.1f}], +{s['arm_only']}/-{s['base_only']}, p={s['mcnemar_p']:.4f}) | "
                  f"HP {s['base_hp']:5.1f}->{s['arm_hp']:5.1f} ({s['diff_hp']:+5.2f}) | T_f {s['base_score']:.4f}->{s['arm_score']:.4f} ({s['diff_score']:+.4f}) | "
                  f"sec/fight {s['base_sec']:.2f}->{s['arm_sec']:.2f}")
        out["comparisons"][f"{nm}_vs_{base_name}"] = comp

    if len(arm_order) >= 3:
        # Also compare arm 2 vs arm 1 (e.g. s256 vs s128)
        n1, n2 = arm_order[1], arm_order[2]
        comp = {}
        print(f"\n=== Paired: {n2} vs {n1} ===")
        for gname in ("all_act3_boss", "boss1", "boss2_final"):
            ks = groups[gname]
            if not ks:
                continue
            s = summarize_group(ks, arms[n1]["data"], arms[n2]["data"])
            comp[gname] = s
            print(f"  {gname:32s} n={s['n']:4d} | win {100*s['base_win']:5.1f}% -> {100*s['arm_win']:5.1f}% "
                  f"({100*s['diff_win']:+5.1f} pp [{100*s['diff_win_ci'][0]:+5.1f},{100*s['diff_win_ci'][1]:+5.1f}], +{s['arm_only']}/-{s['base_only']}, p={s['mcnemar_p']:.4f}) | "
                  f"T_f {s['base_score']:.4f}->{s['arm_score']:.4f} ({s['diff_score']:+.4f})")
        out["comparisons"][f"{n2}_vs_{n1}"] = comp

    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[t3 report] wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("starts"); p.add_argument("outdir")
    r = sub.add_parser("report")
    r.add_argument("meta"); r.add_argument("--arms", nargs="+", required=True); r.add_argument("--out", default="")
    a = ap.parse_args()
    {"prep": cmd_prep, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
