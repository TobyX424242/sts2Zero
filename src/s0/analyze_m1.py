"""M1 analysis: T_f vs HP-only battle-end scoring on the same battle starts (paired by fight id).

For each simulation budget: survival (with McNemar's exact test on discordant fights), mean T_f value at the end of
the fight (the quantity the T_f arm optimizes, scored for both arms with the same table), end HP, potions used,
turns, and breakdowns by room type. Also search speed.

Usage: python -I analyze_m1.py <m1_out_dir>   (reads <dir>/{tf,hp}{200,1000}_s*.jsonl)
"""
import glob, json, math, os, sys

import numpy as np


def load(d, tag):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, f"{tag}_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line); rows[r["id"]] = r
    return rows


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def summarize(tf, hp):
    ids = sorted(set(tf) & set(hp))
    ok = [i for i in ids if tf[i]["error"] is None and hp[i]["error"] is None]
    A = {k: np.array([tf[i][k] for i in ok], float) for k in ("end_hp", "max_hp", "start_hp", "potions_used", "potions_start", "turns", "score_tf", "score_hp", "sims_per_s")}
    B = {k: np.array([hp[i][k] for i in ok], float) for k in A}
    wa = np.array([tf[i]["result"] == "won" for i in ok]); wb = np.array([hp[i]["result"] == "won" for i in ok])
    rooms = np.array([tf[i]["room_type"] for i in ok])
    b, c = int((wa & ~wb).sum()), int((~wa & wb).sum())
    out = {"fights_paired": len(ok), "errors": {"tf": sum(tf[i]["error"] is not None for i in ids), "hp": sum(hp[i]["error"] is not None for i in ids)},
           "won": {"tf": round(float(wa.mean()), 4), "hp": round(float(wb.mean()), 4), "tf_only": b, "hp_only": c, "mcnemar_p": round(mcnemar_p(b, c), 4)},
           "mean_T_f_at_end": {"tf": round(float(A["score_tf"].mean()), 4), "hp": round(float(B["score_tf"].mean()), 4),
                               "diff_tf_minus_hp": round(float((A["score_tf"] - B["score_tf"]).mean()), 4), "diff_95ci": boot_ci(A["score_tf"] - B["score_tf"])},
           "mean_end_hp_won_both": {"tf": round(float(A["end_hp"][wa & wb].mean()), 2), "hp": round(float(B["end_hp"][wa & wb].mean()), 2)},
           "potions_used_share": {"tf": round(float(A["potions_used"].sum() / max(1, A["potions_start"].sum())), 4),
                                  "hp": round(float(B["potions_used"].sum() / max(1, B["potions_start"].sum())), 4)},
           "turns_mean": {"tf": round(float(A["turns"].mean()), 2), "hp": round(float(B["turns"].mean()), 2)},
           "sims_per_s_per_process_median": {"tf": round(float(np.median(A["sims_per_s"])), 1), "hp": round(float(np.median(B["sims_per_s"])), 1)},
           "by_room": {}}
    for r in ("monster", "elite", "boss"):
        m = rooms == r
        if m.any():
            out["by_room"][r] = {"n": int(m.sum()), "won_tf": round(float(wa[m].mean()), 4), "won_hp": round(float(wb[m].mean()), 4),
                                 "T_f_diff": round(float((A["score_tf"][m] - B["score_tf"][m]).mean()), 4),
                                 "potions_used_tf": int(A["potions_used"][m].sum()), "potions_used_hp": int(B["potions_used"][m].sum())}
    return out


def main():
    d = sys.argv[1]; res = {}
    for sims in (200, 1000):
        tf, hp = load(d, f"tf{sims}"), load(d, f"hp{sims}")
        if tf and hp:
            res[f"sims_{sims}"] = summarize(tf, hp)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(d, "m1_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
