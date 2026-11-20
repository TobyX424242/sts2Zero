"""ic113 combat warm-up reports (src/ic113/WARMUP.md). Statistics only.

  python3 -I src/ic113/warmup_report.py harness DIR STARTS ARM_A ARM_B
      fight harness, arms paired by start (DIR/<arm>.jsonl; warmup_eval.sh writes OUTDIR/<name>.jsonl, so ARM may be "h1/h1"): per room type and act the starts both
      arms played, win rates, paired difference B - A with a 95% CI (normal approximation on the per-start difference),
      McNemar counts (A only / B only), mean end HP (death = 0) and the paired end-HP difference; capped fights (round 50).
  python3 -I src/ic113/warmup_report.py runs RUNS_A RUNS_B [FIGHTS_A FIGHTS_B]
      full runs paired by seed (eval_ic113.sh outputs): wins, paired difference with a 95% CI, McNemar, mean floor, act 1 /
      act 2 cleared, errors by cause; with fight logs: elite and boss fight win rates.
  python3 -I src/ic113/warmup_report.py train RUN_DIR [HOURS_PER_BIN]
      self-play fight logs (RUN_DIR/fights/a*.jsonl): per time bin fights, win rate by room type, capped fights
      (stalemates) by encounter, mutated starts and their win rate.
"""
import collections, glob, json, math, os, sys


def load(path):
    out = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        out[d["id"]] = d
    return out


def ci(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m, 1.96 * math.sqrt(v / n)


def harness(d, starts, a, b):
    meta = {}
    for line in open(starts):
        s = json.loads(line)
        meta[s["id"]] = (s["room_type"], s["act_index"])
    A, B = load(os.path.join(d, a + ".jsonl")), load(os.path.join(d, b + ".jsonl"))
    ids = sorted(set(A) & set(B) & set(meta))
    print(f"harness {d}: {a} {len(A)} fights, {b} {len(B)}, paired {len(ids)}; errors {a} "
          f"{sum(A[i]['result'] == 'error' for i in A)}, {b} {sum(B[i]['result'] == 'error' for i in B)}; capped (round 50) "
          f"{a} {sum(A[i]['result'] == 'capped' for i in A)}, {b} {sum(B[i]['result'] == 'capped' for i in B)}")
    groups = collections.defaultdict(list)
    for i in ids:
        rt, act = meta[i]
        groups[(rt, act)].append(i); groups[(rt, "all")].append(i); groups[("all", "all")].append(i)
    print(f"{'room':7s} {'act':>3s} {'n':>6s} {'win ' + a:>12s} {'win ' + b:>12s} {'diff (95% CI)':>20s} {'only A':>6s} {'only B':>6s} "
          f"{'HP ' + a:>9s} {'HP ' + b:>9s} {'HP diff (CI)':>16s}")
    order = [(r, x) for r in ("elite", "boss", "monster", "all") for x in (0, 1, 2, "all")]
    for k in order:
        if k not in groups:
            continue
        g = groups[k]
        wa = [A[i]["result"] == "won" for i in g]; wb = [B[i]["result"] == "won" for i in g]
        m, h = ci([int(y) - int(x) for x, y in zip(wa, wb)])
        ha = [A[i]["end_hp"] if A[i]["result"] == "won" else 0 for i in g]
        hb = [B[i]["end_hp"] if B[i]["result"] == "won" else 0 for i in g]
        mh, hh = ci([y - x for x, y in zip(ha, hb)])
        print(f"{k[0]:7s} {str(k[1] if k[1] == 'all' else k[1] + 1):>3s} {len(g):6d} {100 * sum(wa) / len(g):11.1f}% {100 * sum(wb) / len(g):11.1f}% "
              f"{100 * m:+7.1f} ({100 * (m - h):+.1f}, {100 * (m + h):+.1f}) {sum(x and not y for x, y in zip(wa, wb)):6d} "
              f"{sum(y and not x for x, y in zip(wa, wb)):6d} {sum(ha) / len(g):9.1f} {sum(hb) / len(g):9.1f} {mh:+7.1f} (+-{hh:.1f})")


def runs(pa, pb, fa=None, fb=None):
    def rl(p):
        out = {}
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            out[d["seed"]] = d
        return out
    A, B = rl(pa), rl(pb)
    seeds = sorted(set(A) & set(B))
    print(f"runs: A {pa} ({len(A)}), B {pb} ({len(B)}), paired seeds {len(seeds)}")
    for name, R in (("A", A), ("B", B)):
        ok = [R[s] for s in seeds if not R[s].get("error")]
        err = collections.Counter((R[s].get("error") or "")[:60] for s in seeds if R[s].get("error"))
        w = sum(r["won"] for r in ok)
        p = w / max(1, len(ok)); h = 1.96 * math.sqrt(p * (1 - p) / max(1, len(ok)))
        print(f"  {name}: won {w}/{len(ok)} = {100 * p:.1f}% (+-{100 * h:.1f}), mean floor {sum(r['floors'] for r in ok) / max(1, len(ok)):.1f}, "
              f"act 1 cleared {100 * sum(r['act'] >= 1 or r['won'] for r in ok) / max(1, len(ok)):.0f}%, act 2 cleared "
              f"{100 * sum(r['act'] >= 2 or r['won'] for r in ok) / max(1, len(ok)):.0f}%, errors {sum(err.values())} {dict(err)}")
    both = [s for s in seeds if not A[s].get("error") and not B[s].get("error")]
    m, h = ci([int(B[s]["won"]) - int(A[s]["won"]) for s in both])
    fl, fh = ci([B[s]["floors"] - A[s]["floors"] for s in both])
    print(f"  paired (both without error, {len(both)}): win B - A {100 * m:+.1f} ({100 * (m - h):+.1f}, {100 * (m + h):+.1f}); A only "
          f"{sum(A[s]['won'] and not B[s]['won'] for s in both)}, B only {sum(B[s]['won'] and not A[s]['won'] for s in both)}; "
          f"floor B - A {fl:+.1f} (+-{fh:.1f})")
    for name, fp in (("A", fa), ("B", fb)):
        if not fp:
            continue
        c = collections.defaultdict(lambda: [0, 0])
        caps = collections.Counter()
        for line in open(fp):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("practice"):
                continue
            c[d["room_type"]][0] += 1; c[d["room_type"]][1] += d["result"] == "won"
            if d["result"] == "capped":
                caps[d.get("encounter", "")] += 1
        print(f"  {name} fights: " + ", ".join(f"{k} {v[1]}/{v[0]} = {100 * v[1] / max(1, v[0]):.1f}%" for k, v in sorted(c.items()))
              + f"; capped {dict(caps)}")


def train(rd, hours=1.0):
    fights = []
    for p in sorted(glob.glob(os.path.join(rd, "fights", "a*.jsonl"))):
        for line in open(p):
            try:
                fights.append(json.loads(line))
            except Exception:
                pass
    if not fights:
        print("no fights"); return
    # bins of equal model-version range (the trainer version grows with time)
    fights = [f for f in fights if f.get("model_version", 0) > 0]   # 0 = no network evaluation in the fight
    vs = sorted(f["model_version"] for f in fights)
    lo, hi = vs[0], vs[-1]
    nb = max(1, int(round(hours)))
    edges = [lo + (hi - lo) * k / nb for k in range(nb + 1)]
    print(f"self-play fights {len(fights)}, model versions {lo}-{hi} in {nb} bins")
    for k in range(nb):
        g = [f for f in fights if edges[k] <= f["model_version"] < edges[k + 1] or (k == nb - 1 and f["model_version"] == hi)]
        if not g:
            continue
        by = collections.defaultdict(lambda: [0, 0])
        for f in g:
            by[f["room_type"]][0] += 1; by[f["room_type"]][1] += f["result"] == "won"
        mut = [f for f in g if f.get("mutated")]
        caps = collections.Counter(f["encounter"] for f in g if f["result"] == "capped")
        print(f"  v{int(edges[k])}-{int(edges[k + 1])}: {len(g)} fights; " + ", ".join(
            f"{r} {100 * v[1] / v[0]:.1f}% of {v[0]}" for r, v in sorted(by.items())) +
            f"; mutated {len(mut)} won {100 * sum(f['result'] == 'won' for f in mut) / max(1, len(mut)):.1f}%; capped {sum(caps.values())} "
            f"{dict(caps.most_common(4))}; errors {sum(f['result'] == 'error' for f in g)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "harness":
        harness(*a[1:5])
    elif a[0] == "runs":
        runs(*a[1:])
    elif a[0] == "train":
        train(a[1], float(a[2]) if len(a) > 2 else 6)
