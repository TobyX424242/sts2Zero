"""Table of the S1 scaling-curve results. Usage: python -I summarize_scaling.py <runs/s1_scaling>"""
import json, os, sys

R = sys.argv[1]
hdr = ["run", "train runs", "epochs", "best ep", "test ll", "auc", "wf auc", "ece bands", "cell ll", "cell auc", "cell ece", "hp pts/10 by act", "minutes"]
print(" | ".join(hdr))
for a in ("noattn", "transformer"):
    for n in ("25000", "50000", "100000", "300000", "0"):
        f = os.path.join(R, f"{a}_{n}", "results.json")
        if not os.path.exists(f):
            print(f"{a}_{n} | missing"); continue
        r = json.load(open(f)); t = r["test"]; h = r["history"]; c = t["target_cell"]
        best = min(h, key=lambda x: x["val"]["logloss"])["epoch"]
        print(" | ".join(str(x) for x in [f"{a}_{n}", r["train_runs"], len(h), best, t["all"]["logloss"], t["all"]["auc"],
              t["auc_within_floor"], t["ece_within_floor_bands"], c["logloss"], c["auc"], c["ece"],
              r["hp_probe_val"]["mean_pts_by_act"], round(r["elapsed_s"] / 60, 1)]))
for run in sys.argv[2:]:
    r = json.load(open(os.path.join(R, run, "results.json")))
    print(run, [(e["epoch"], e["train_loss"], e["val"]["logloss"], e["val"]["ece"]) for e in r["history"]])
