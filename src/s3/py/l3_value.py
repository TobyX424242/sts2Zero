"""Layout v3, offline value test (2026-11-16): value-only training of the macro ensemble on a fixed set of self-play runs
(src/exp/l3_gen.sh: frozen segment 49 in the layout v3 build, rows with fights and map nodes), from the same start
file for two arms with identical settings, data order and seeds:
  v3   every parameter trainable (the layout v3 paths learn);
  ctl  the same class with the layout v3 zero layers frozen at zero (the layout v2 function, trained the same way).
Only the value heads matter here: member 0 (the policy member) trains its value heads alone (encoder detached, as in
train_macro_s3.py), members 1-4 train fully, so the policy logits of member 0 stay as in the start file.

Runs are split by run seed: a share --holdout of the runs is held out (never trained on); old carry rows (--carry, any
layout, zero-padded: no fights, no map) can be added to the training set only. Target: the trainer's outcome,
alpha x won + (1 - alpha) x acts cleared / 3, alpha = A10 win rate of the training runs / 0.2 (at most 1) unless --alpha.
Report (held-out runs, before training and every --eval-every steps): value log loss on all rows and on floor rows,
within-floor AUC for the run win, calibration per act (mean value vs mean outcome) and ECE, act-3 rows alone, per member.
Each arm's ensemble is saved (OUT/macro_<arm>.pt) for the play test (src/exp/eval_l3.sh); the report is
OUT/report.json and OUT/report.txt.

usage: ./.venv/bin/python -I src/s3/py/l3_value.py --mdata DIR[,DIR] --init macro_s049.pt --out DIR
           [--carry carry.pt] [--carry-runs 4096] [--steps 4000] [--batch 1024] [--lr 1e-4] [--arms v3,ctl] [--device cuda]
"""
import argparse, glob, json, math, os, sys, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import macro_s3 as ms  # noqa: E402
import train_macro_s3 as tr  # noqa: E402


def load_runs(dirs):
    """Every run of the chunks in the given directories: dict of row arrays and per-run fields."""
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.s3r"))):
            meta_r, ints, floats, meta = tr.read_chunk(p)
            for r in meta_r:
                a, n = int(r[9]), int(r[10])
                seed = (int(r[0]) & 0xFFFFFFFF) | ((int(r[1]) & 0xFFFFFFFF) << 32)
                runs.append({"ints": ints[a:a + n], "floats": floats[a:a + n], "meta": meta[a:a + n], "won": float(r[3]),
                             "asc": int(r[2]), "act": int(r[5]), "perturbed": bool(r[8]), "seed": seed, "src": "new"})
    return runs


def load_carry_runs(path, last):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_rows = ck["n_rows"].numpy()
    ints, floats = ms.pad_rows(ck["ints"].numpy(), ck["floats"].numpy())
    meta = ck["meta"].numpy()
    runs, o = [], 0
    for i, n in enumerate(n_rows):
        n = int(n)
        runs.append({"ints": ints[o:o + n], "floats": floats[o:o + n], "meta": meta[o:o + n], "won": float(ck["won"][i]),
                     "asc": int(ck["asc"][i]), "act": int(ck["act"][i]), "perturbed": bool(ck["perturbed"][i]), "seed": -1, "src": "carry"})
        o += n
    return runs[-last:] if last > 0 else runs


def outcome(r, alpha):
    return alpha * r["won"] + (1 - alpha) * (1.0 if r["won"] else min(r["act"], 2) / 3)


def flat(runs, alpha):
    cat = lambda k: np.concatenate([r[k] for r in runs])
    z = np.concatenate([np.full(len(r["meta"]), outcome(r, alpha), np.float32) for r in runs])
    won = np.concatenate([np.full(len(r["meta"]), r["won"] > 0) for r in runs])
    asc = np.concatenate([np.full(len(r["meta"]), r["asc"]) for r in runs])
    return cat("ints"), cat("floats"), cat("meta"), z, won, asc


@torch.no_grad()
def member_logits(nets, ints, floats, dev, bs=4096):
    out = []
    for i in range(0, len(ints), bs):
        b = ms.rows_to_batch(ints[i:i + bs], floats[i:i + bs], dev)
        out.append(torch.stack([n.forward_rows(b, policy=False)[2] for n in nets]).cpu())
    return torch.cat(out, 1).numpy().astype(np.float64)          # [members, rows]


def bce(z, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(z * np.log(p) + (1 - z) * np.log(1 - p)).mean())


def report(nets, H, dev):
    ints, floats, meta, z, won, asc = H
    L = member_logits(nets, ints, floats, dev)
    p = 1 / (1 + np.exp(-L.mean(0)))                              # the ensemble value of the T_f tables
    fl = meta[:, tr.M_KIND] == 0
    act = meta[:, tr.M_ACT].astype(int); floor = meta[:, tr.M_FLOOR].astype(int)
    out = {"rows": int(len(z)), "floor_rows": int(fl.sum()), "logloss_all": round(bce(z, p), 5), "logloss_floor": round(bce(z[fl], p[fl]), 5),
           "auc_win_within_floor": tr._auc_in_floor(p[fl], won[fl], floor[fl]),
           "members_logloss_floor": [round(bce(z[fl], 1 / (1 + np.exp(-L[j, fl]))), 5) for j in range(len(L))]}
    bins = np.clip((p[fl] * 10).astype(int), 0, 9)
    out["ece_floor"] = round(float(sum(abs(p[fl][bins == k].mean() - z[fl][bins == k].mean()) * (bins == k).sum()
                                       for k in range(10) if (bins == k).any()) / max(1, fl.sum())), 5)
    out["by_act"] = {}
    for k in range(3):
        m = fl & (act == k)
        if m.sum() >= 50:
            out["by_act"][f"act{k + 1}"] = {"rows": int(m.sum()), "outcome": round(float(z[m].mean()), 4), "value": round(float(p[m].mean()), 4),
                                            "logloss": round(bce(z[m], p[m]), 5), "auc_win_within_floor": tr._auc_in_floor(p[m], won[m], floor[m])}
    m = fl & (asc >= 10)
    if m.sum() >= 50:
        out["a10_floor"] = {"rows": int(m.sum()), "logloss": round(bce(z[m], p[m]), 5), "auc_win_within_floor": tr._auc_in_floor(p[m], won[m], floor[m])}
    return out


def line(tag, r):
    a3 = r["by_act"].get("act3", {})
    return (f"{tag}: logloss all {r['logloss_all']:.5f} floor {r['logloss_floor']:.5f} | AUC(win) in-floor {r['auc_win_within_floor']} | "
            f"ECE {r['ece_floor']:.4f} | act 3: logloss {a3.get('logloss')} AUC {a3.get('auc_win_within_floor')} "
            f"(value {a3.get('value')} vs outcome {a3.get('outcome')}) | members {r['members_logloss_floor']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdata", required=True, help="chunk directories (comma-separated) of l3_gen.sh")
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--carry", default="", help="old carry file: its runs join the training set (rows zero-padded)")
    ap.add_argument("--carry-runs", type=int, default=4096, help="the carry's last N runs (0 = all)")
    ap.add_argument("--holdout", type=float, default=0.1); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=-1.0); ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--arms", default="v3,ctl"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "report.txt"), "a")

    def log(s):
        s = f"[{time.strftime('%H:%M:%S')}] {s}"; print(s, flush=True); logf.write(s + "\n"); logf.flush()
    t0 = time.time()
    runs = load_runs([d for d in a.mdata.split(",") if d])
    hold = lambda r: (zlib.crc32(str(r["seed"]).encode()) % 1000) < a.holdout * 1000
    test = [r for r in runs if hold(r)]; train = [r for r in runs if not hold(r)]
    carry = load_carry_runs(a.carry, a.carry_runs) if a.carry else []
    tgt = [r for r in train if r["asc"] >= 10 and not r["perturbed"]] or train
    wr = float(np.mean([r["won"] for r in tgt]))
    alpha = a.alpha if a.alpha >= 0 else min(1.0, wr / 0.2)
    Tr = flat(train + carry, alpha); H = flat(test, alpha)
    n_map = float((Tr[1][:, ms.F_MAP] > 0).mean()); n_fight = float((Tr[0][:, ms.OFF["f_room"][0]] > 0).mean())
    log(f"config {vars(a)}")
    log(f"runs: {len(runs)} new ({len(train)} train, {len(test)} held out), {len(carry)} carry runs (train only); rows train {len(Tr[3]):,}, "
        f"held out {len(H[3]):,}; A10 win rate (train) {wr:.3f} -> alpha {alpha:.3f}; train rows with map nodes {n_map:.1%}, with fights {n_fight:.1%}; "
        f"loaded in {time.time() - t0:.0f} s")
    res = {"config": vars(a), "alpha": alpha, "runs_train": len(train), "runs_test": len(test), "runs_carry": len(carry), "arms": {}}
    dev = a.device
    for arm in [x for x in a.arms.split(",") if x]:
        torch.manual_seed(a.seed)
        ens = ms.Ensemble.load(a.init, dev)
        nets = ens.nets
        if arm == "ctl":
            for n in nets:
                with torch.no_grad():
                    for p in n.v3_zero_params():
                        p.zero_(); p.requires_grad_(False)
        elif arm != "v3":
            raise SystemExit(f"unknown arm {arm} (v3 | ctl)")
        for n in nets:
            n.eval()
        opts = [torch.optim.Adam([p for p in n.parameters() if p.requires_grad], lr=a.lr) for n in nets]
        hist = [{"step": 0, **report(nets, H, dev)}]
        log(line(f"{arm} step 0", hist[-1]))
        rngs = [np.random.default_rng(a.seed * 100 + j) for j in range(len(nets))]   # the same draws in every arm
        te = time.time(); tl = 0.0
        for step in range(1, a.steps + 1):
            for j, (n, op) in enumerate(zip(nets, opts)):
                idx = rngs[j].integers(0, len(Tr[3]), a.batch)
                b = ms.rows_to_batch(Tr[0][idx], Tr[1][idx], dev)
                logit = n.forward_rows(b, policy=False, detach=(j == 0))[2]
                loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(Tr[3][idx]).to(dev))
                op.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0); op.step()
                tl += loss.item()
            if step % a.eval_every == 0 or step == a.steps:
                hist.append({"step": step, "train_loss": round(tl / (a.eval_every * len(nets)), 5), **report(nets, H, dev)})
                tl = 0.0
                log(line(f"{arm} step {step} (train {hist[-1]['train_loss']}, {time.time() - te:.0f} s)", hist[-1]))
        ens.version += 1
        ens.save(os.path.join(a.out, f"macro_{arm}.pt"))
        res["arms"][arm] = hist
        json.dump(res, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    if "v3" in res["arms"] and "ctl" in res["arms"]:
        v, c = res["arms"]["v3"][-1], res["arms"]["ctl"][-1]
        log(f"v3 - ctl (held out, last step): logloss floor {v['logloss_floor'] - c['logloss_floor']:+.5f}, AUC(win) in-floor "
            f"{(v['auc_win_within_floor'] or 0) - (c['auc_win_within_floor'] or 0):+.4f}, act 3 logloss "
            f"{v['by_act'].get('act3', {}).get('logloss', 0) - c['by_act'].get('act3', {}).get('logloss', 0):+.5f}")
    log(f"done in {time.time() - t0:.0f} s; files {a.out}/macro_<arm>.pt, report.json")


if __name__ == "__main__":
    main()
