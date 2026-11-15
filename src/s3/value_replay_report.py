"""Compares the arms of research/s3/value_replay.sh: fresh-run value loss (measured before each update) against the
constant and the 9-cell table, training loss, explained variance and member spread, over the last third of the replay.
Then, per arm: the value minus the outcome on fresh runs by ascension group and act (mean of the members' probabilities;
the row mix of the value window shows here), and from the trainer's `value_cells` the share of the room above the cell
mean the value takes, the in-floor AUC for the next milestone and for the run win, and the share of act-3 values below
the banked progress.
usage: python -I research/s3/value_replay_report.py OUT ARM [ARM ...] [--from FRACTION]   (default: last third)
"""
import json, os, sys

import numpy as np


def main():
    args = sys.argv[1:]
    frac = 2 / 3
    if "--from" in args:
        i = args.index("--from"); frac = float(args[i + 1]); del args[i:i + 2]
    out, arms = args[0], args[1:]
    keys = ["value_logloss", "value_logloss_table", "value_logloss_const", "value_loss", "value_explained",
            "value_logloss_a10", "value_logloss_table_a10", "value_logloss_a0_9", "value_logloss_table_a0_9"]
    print(f"iterations from {frac:.2f} of each replay; fresh = loss on the next runs before training on them")
    print(f"{'arm':10s} {'iters':>5s} {'fresh':>7s} {'table':>7s} {'const':>7s} {'fresh-tab':>9s} {'train':>7s} {'gap':>7s} "
          f"{'expl':>6s} {'A10 f-t':>8s} {'A0-9 f-t':>8s} {'sd A0-8':>7s} {'sd A10':>7s}  fresh by quarter")
    for arm in arms:
        p = os.path.join(out, arm, "log.jsonl")
        if not os.path.exists(p):
            print(f"{arm:10s} no log"); continue
        L = [json.loads(l) for l in open(p) if l.strip()]
        if not L:
            print(f"{arm:10s} empty"); continue
        R = L[int(len(L) * frac):] or L
        m = {k: float(np.mean([r[k] for r in R if k in r])) if any(k in r for r in R) else float("nan") for k in keys}
        sd = {}
        for g in ("A0-8", "A10"):
            v = [r["calibration"][g]["logit_sd"] for r in R if g in r.get("calibration", {})]
            sd[g] = float(np.mean(v)) if v else float("nan")
        q = [float(np.mean([r["value_logloss"] for r in L[i * len(L) // 4:(i + 1) * len(L) // 4]] or [np.nan])) for i in range(4)]
        print(f"{arm:10s} {len(L):5d} {m['value_logloss']:7.4f} {m['value_logloss_table']:7.4f} {m['value_logloss_const']:7.4f} "
              f"{m['value_logloss'] - m['value_logloss_table']:+9.4f} {m['value_loss']:7.4f} {m['value_logloss'] - m['value_loss']:+7.4f} "
              f"{m['value_explained']:6.3f} {m['value_logloss_a10'] - m['value_logloss_table_a10']:+8.4f} "
              f"{m['value_logloss_a0_9'] - m['value_logloss_table_a0_9']:+8.4f} {sd['A0-8']:7.3f} {sd['A10']:7.3f}  "
              + " ".join(f"{x:.4f}" for x in q))
    CELLS = [f"{g}_act{a}" for g in ("A0-8", "A9", "A10") for a in (1, 2, 3)]
    logs = {}
    for arm in arms:
        p = os.path.join(out, arm, "log.jsonl")
        if os.path.exists(p):
            L = [json.loads(l) for l in open(p) if l.strip()]
            if L:
                logs[arm] = L[int(len(L) * frac):] or L

    def wmean(R, top, cell, key, wkey="rows"):
        xs = [(r[top][cell][key], r[top][cell][wkey]) for r in R if cell in r.get(top, {}) and r[top][cell].get(key) is not None]
        return sum(x * w for x, w in xs) / sum(w for _, w in xs) if xs else float("nan")

    print("\nvalue minus outcome on fresh runs (floor rows), by ascension group and act")
    print(f"{'arm':10s} " + " ".join(f"{c:>10s}" for c in CELLS))
    for arm, R in logs.items():
        print(f"{arm:10s} " + " ".join(f"{wmean(R, 'calibration', c, 'ens_p') - wmean(R, 'calibration', c, 'outcome'):+10.4f}" for c in CELLS))
    if any("value_cells" in r for R in logs.values() for r in R):
        for title, fn in (("share of the room above the cell mean taken by the value (gain / room)",
                           lambda R, c: wmean(R, "value_cells", c, "gain") / wmean(R, "value_cells", c, "room")),
                          ("in-floor AUC for the cell's next milestone", lambda R, c: wmean(R, "value_cells", c, "auc_next")),
                          ("in-floor AUC for the run win", lambda R, c: wmean(R, "value_cells", c, "auc_win")),
                          ("share of values below the banked progress (acts 2-3)", lambda R, c: wmean(R, "value_cells", c, "below_banked"))):
            print(f"\n{title}")
            print(f"{'arm':10s} " + " ".join(f"{c:>10s}" for c in CELLS))
            for arm, R in logs.items():
                print(f"{arm:10s} " + " ".join(f"{fn(R, c):10.3f}" for c in CELLS))
        print("\nA10 after the first act-3 boss: rows per iteration, runs won, implied P(win) from the value, its AUC; then [won, implied] per HP fifth")
        for arm, R in logs.items():
            c = "A10_after_boss1"
            hp = []
            for k in range(5):
                xs = [(r["value_cells"][c]["hp"][k]) for r in R if c in r.get("value_cells", {}) and r["value_cells"][c]["hp"][k][0] > 0]
                n = sum(x[0] for x in xs)
                hp.append(f"{sum(x[1] * x[0] for x in xs) / n:.2f}/{sum(x[2] * x[0] for x in xs) / n:+.2f}" if n else "-")
            print(f"{arm:10s} rows {wmean(R, 'value_cells', c, 'rows', 'rows'):6.0f}  won {wmean(R, 'value_cells', c, 'won'):.3f}  implied {wmean(R, 'value_cells', c, 'p_win'):+.3f}  "
                  f"AUC {wmean(R, 'value_cells', c, 'auc_win'):.3f}  by HP fifth: " + "  ".join(hp))


if __name__ == "__main__":
    main()
