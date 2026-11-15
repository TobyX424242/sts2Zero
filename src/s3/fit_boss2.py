"""Fit W2, the chance of winning the second act-3 boss at ascension 10 (the double boss), from S3 self-play logs.

W2(h, P) = sigmoid(b0 + b_hp * h + b_pot * min(P, pot_cap) [+ b_hp2 * h^2]) with h = HP / max HP and P = potions held at
the start of the second boss fight. The actor uses it for the first act-3 boss at A10 (option boss2_model=FILE, see
src/s3/S3/RunAgent.cs): T_f(h, P) = d + (1 - d) W2(h, P), d = the death value of death_value=progress.

Reads one or more segment log directories (each with runs/*.jsonl and fights/*.jsonl, e.g.
the stored run metadata of segments 13-16; runs with errors or perturbed fight starts and practice fights are skipped.
The second boss of a run is its last boss fight of act 3 when the run had two act-3 boss fights (the first one won).

usage: python -I src/s3/fit_boss2.py [--out FILE.json] [--test DIR] [--hp2] DIR [DIR ...]
  --test DIR  also report the fit's log loss on this directory (held out) against simpler models
Prints the act-3 double-boss statistics (conversion, HP and potions at the second boss, win rate by HP band) as well.
"""
import argparse, glob, json, math, sys
import numpy as np


def load(d):
    runs = {}
    for f in glob.glob(d + "/runs/*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("error") or r.get("perturbed"):
                continue
            runs[r["seed"]] = r
    fights = {}
    for f in glob.glob(d + "/fights/*.jsonl"):
        for l in open(f):
            try:
                x = json.loads(l)
            except Exception:
                continue
            if x.get("practice") or x.get("error") or x.get("run_seed") not in runs:
                continue
            fights.setdefault(x["run_seed"], []).append(x)
    return runs, fights


def act3_bosses(runs, fights, asc=10):
    """Per A10 run that reached an act-3 boss: (run, [boss fights of act 3 in order]). Act 3 = floor >= 40."""
    out = []
    for s, fs in fights.items():
        r = runs[s]
        if r["ascension"] != asc:
            continue
        b = sorted((x for x in fs if x["room_type"] == "boss" and x["floor"] >= 40), key=lambda x: x["fight"])
        if b:
            out.append((r, b))
    return out


def features(b2, hp2):
    h = b2["start_hp"] / max(1, b2["max_hp"])
    p = b2["potions_start"]
    return [1.0, h, float(p)] + ([h * h] if hp2 else [])


def fit_logistic(X, y, l2=1e-3, iters=50):
    X = np.asarray(X, float); y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def pairs(dirs, hp2):
    X, y, rows = [], [], []
    stats = {"reached_b1": 0, "won_b1": 0, "b2": 0, "won_b2": 0}
    for d in dirs:
        runs, fights = load(d)
        for r, b in act3_bosses(runs, fights):
            stats["reached_b1"] += 1
            if b[0]["result"] == "won":
                stats["won_b1"] += 1
            if len(b) >= 2:
                b2 = b[1]
                stats["b2"] += 1; stats["won_b2"] += b2["result"] == "won"
                X.append(features(b2, hp2)); y.append(1.0 if b2["result"] == "won" else 0.0)
                rows.append((b[0], b2))
    return np.array(X), np.array(y), rows, stats


def report(X, y, rows, stats, w=None, title=""):
    print(f"--- {title}")
    print(f"A10 runs at act-3 boss 1: {stats['reached_b1']}, boss 1 won {stats['won_b1']}, boss-2 fights {stats['b2']} "
          f"(won {stats['won_b2']} = {stats['won_b2'] / max(1, stats['b2']):.3f})")
    if not len(y):
        return
    h, p = X[:, 1], X[:, 2]
    heal = np.array([b2["start_hp"] - b1["end_hp"] for b1, b2 in rows])
    print(f"boss-2 start: HP share mean {h.mean():.3f} median {np.median(h):.3f}, below 0.6: {(h < 0.6).mean():.3f}; "
          f"potions mean {p.mean():.2f}; HP gained between the bosses median {np.median(heal):.0f} mean {heal.mean():.1f}")
    b1p = np.array([b1["potions_start"] for b1, _ in rows]); b1u = np.array([b1["potions_used"] for b1, _ in rows])
    b1hp = np.array([(b1["start_hp"] - b1["end_hp"]) / max(1, b1["max_hp"]) for b1, _ in rows])
    print(f"boss 1 (won, followed by boss 2): HP lost {b1hp.mean():.3f} of max, potions at start {b1p.mean():.2f}, used {b1u.mean():.2f}")
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = (h >= lo) & (h < hi)
        if m.any():
            fitted = "" if w is None else f", fitted {np.mean(1 / (1 + np.exp(-X[m] @ w))):.3f}"
            print(f"  HP share {lo:.1f}-{min(hi, 1):.1f}: n {m.sum():4d}, boss-2 won {y[m].mean():.3f}{fitted}")
    for k in range(0, int(p.max()) + 1):
        m = p == k
        if m.any():
            fitted = "" if w is None else f", fitted {np.mean(1 / (1 + np.exp(-X[m] @ w))):.3f}"
            print(f"  potions {k}: n {m.sum():4d}, boss-2 won {y[m].mean():.3f}{fitted}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", default=""); ap.add_argument("--test", default="")
    ap.add_argument("--hp2", action="store_true", help="add an HP-share squared term")
    ap.add_argument("--pot-cap", type=int, default=3, help="potion counts above this are treated as this (sparse data)")
    a = ap.parse_args()
    X, y, rows, stats = pairs(a.dirs, a.hp2)
    if len(y) < 30:
        sys.exit(f"only {len(y)} boss-2 fights: not enough to fit")
    Xc = X.copy(); Xc[:, 2] = np.minimum(Xc[:, 2], a.pot_cap)
    w = fit_logistic(Xc, y)
    names = ["b0", "b_hp", "b_pot"] + (["b_hp2"] if a.hp2 else [])
    report(Xc, y, rows, stats, w, "training data: " + " ".join(a.dirs))
    print("fit: " + ", ".join(f"{n} {v:+.4f}" for n, v in zip(names, w)) + f"; log loss {logloss(1 / (1 + np.exp(-Xc @ w)), y):.4f} "
          f"(constant {logloss(np.full(len(y), y.mean()), y):.4f}), n {len(y)}")
    print("W2 grid (rows HP share 0.1..1.0, columns potions 0..3): ")
    for hh in np.arange(0.1, 1.01, 0.1):
        print(f"  {hh:.1f}: " + " ".join(f"{1 / (1 + math.exp(-(w[0] + w[1] * hh + w[2] * k + (w[3] * hh * hh if a.hp2 else 0)))):.3f}" for k in range(4)))
    if a.test:
        Xt, yt, rt, st = pairs([a.test], a.hp2)
        Xt[:, 2] = np.minimum(Xt[:, 2], a.pot_cap)
        report(Xt, yt, rt, st, w, "held out: " + a.test)
        if len(yt):
            w_hp = fit_logistic(Xc[:, :2], y)
            print(f"held-out log loss: fit {logloss(1 / (1 + np.exp(-Xt @ w)), yt):.4f}, HP only {logloss(1 / (1 + np.exp(-Xt[:, :2] @ w_hp)), yt):.4f}, "
                  f"constant (train mean) {logloss(np.full(len(yt), y.mean()), yt):.4f}, n {len(yt)}")
    if a.out:
        js = {"b0": w[0], "b_hp": w[1], "b_pot": w[2], "b_hp2": w[3] if a.hp2 else 0.0, "pot_cap": a.pot_cap,
              "n": int(len(y)), "won_rate": float(y.mean()), "sources": a.dirs}
        with open(a.out, "w") as f:
            json.dump(js, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
