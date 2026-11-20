#!/usr/bin/env python3
"""ic113 chain s3ic: the ascension curriculum (PLAN step 7) and the per-segment evaluation report (src/ic113/CHAIN.md).
Standard library only.

  chain_stats.py curriculum --runs DIR --stage-file FILE --log FILE [--t1-a10 0.05] [--t1-low 0.50] [--t2-a10 0.12]
                 [--min-runs 300] [--segment NAME]
      Self-play statistics of a finished segment (DIR/*.jsonl run records; runs with perturbed starts left out; runs that
      ended in an error, e.g. a stalemate, count as not won) and the stage rule: stage 2 when A10 win >= t1-a10 or A0-A4
      win >= t1-low; stage 3 when A10 win >= t2-a10; a rate counts only over at least --min-runs runs. The stage is the
      highest one whose condition holds, never lower than the current one (so a second call on the same segment changes
      nothing). Writes the stage file, appends one line to the log, prints the line.
  chain_stats.py stage-args N
      Prints "ASC_MIX ASC_WEIGHTS" of stage N (ASC_WEIGHTS "-" = uniform over A0-A9).
  chain_stats.py eval DIR
      Report of a chain evaluation (DIR/*.jsonl runs + *_fights.jsonl): win rate, act funnel, HP share entering each boss,
      stalemates and other errors by encounter, potion_mix decisions.
"""
import argparse, glob, json, math, os, re, sys, time
from collections import Counter, defaultdict

# stage -> (asc_mix = share of self-play runs at A10, weights of A0..A9 or None = uniform); PLAN step 7
STAGES = {
    1: (0.20, None),
    2: (0.30, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    3: (0.2716, [0, 0, 0, 0, 0, 0, 0, 0, 3.181, 4.057]),   # Defect's final mix: A8 32% / A9 41% / A10 27% of all runs
    4: (0.55, [0, 0, 0, 0, 0, 0, 0, 0, 1, 3.5]),           # Stage 4 (A10 focus): A8 10% / A9 35% / A10 55% of all runs
}
STALE = re.compile(r"stalemate: (\S+) still")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def read_runs(paths):
    out = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass     # a line cut off by a stopped actor
    return out


def stalemates(runs):
    c = Counter()
    for r in runs:
        m = STALE.search(r.get("error") or "")
        if m:
            c[m.group(1).replace("ENCOUNTER.", "")] += 1
    return c


def selfplay_stats(runs):
    g = {"a10": [0, 0], "a0_4": [0, 0], "a5_9": [0, 0], "all": [0, 0]}
    errors = 0
    for r in runs:
        if r.get("perturbed"):
            continue
        a = int(r.get("ascension", 10)); won = bool(r.get("won"))
        errors += bool(r.get("error"))
        for k in ("all", "a10" if a == 10 else "a0_4" if a <= 4 else "a5_9"):
            g[k][0] += 1; g[k][1] += won
    return g, errors


def cmd_curriculum(a):
    runs = read_runs(sorted(glob.glob(os.path.join(a.runs, "*.jsonl"))))
    g, errors = selfplay_stats(runs)
    cur = 1
    if os.path.exists(a.stage_file):
        cur = int(open(a.stage_file).read().split()[0])
    rate = lambda k: g[k][1] / g[k][0] if g[k][0] >= a.min_runs else None
    r10, rlow = rate("a10"), rate("a0_4")
    want, why = 1, "stage 1"
    if (r10 is not None and r10 >= a.t1_a10) or (rlow is not None and rlow >= a.t1_low):
        want, why = 2, (f"A10 {100 * r10:.1f}% >= {100 * a.t1_a10:g}%" if r10 is not None and r10 >= a.t1_a10
                        else f"A0-A4 {100 * rlow:.1f}% >= {100 * a.t1_low:g}%")
    if r10 is not None and r10 >= a.t2_a10:
        want, why = 3, f"A10 {100 * r10:.1f}% >= {100 * a.t2_a10:g}%"
    if a.t3_a10 > 0 and r10 is not None and r10 >= a.t3_a10:
        want, why = 4, f"A10 {100 * r10:.1f}% >= {100 * a.t3_a10:g}%"
    new = max(cur, want)
    fmt = lambda k: f"{g[k][1]}/{g[k][0]} = {100 * g[k][1] / max(1, g[k][0]):.1f}%"
    st = stalemates(runs)
    line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}] {a.segment or a.runs}: self-play (no perturbed starts; errors count "
            f"as lost) A10 {fmt('a10')}, A0-A4 {fmt('a0_4')}, A5-A9 {fmt('a5_9')}, all {fmt('all')}; errors {errors}, stalemates "
            f"{sum(st.values())} ({', '.join(f'{k} {v}' for k, v in st.most_common(4)) or '-'}); "
            + (f"STAGE {cur} -> {new} ({why})" if new != cur else f"stage stays {cur}"
               + (f" (rule says {want}: never down)" if want < cur else "")))
    os.makedirs(os.path.dirname(os.path.abspath(a.stage_file)), exist_ok=True)
    with open(a.stage_file + ".tmp", "w") as f:
        mix, w = STAGES[new]
        f.write(f"{new} asc_mix {mix} asc_weights {','.join(map(str, w)) if w else 'uniform A0-A9'}\n")
    os.replace(a.stage_file + ".tmp", a.stage_file)
    with open(a.log, "a") as f:
        f.write(line + "\n")
    print(line)


def cmd_stage_args(a):
    mix, w = STAGES[a.stage]
    print(mix, ",".join(f"{x:g}" for x in w) if w else "-")


def cmd_eval(a):
    runs = read_runs([p for p in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))) if "_fights" not in os.path.basename(p)])
    fights = read_runs(sorted(glob.glob(os.path.join(a.dir, "*_fights.jsonl"))))
    n = len(runs); won = sum(bool(r.get("won")) for r in runs)
    lo, hi = wilson(won, n)
    errs = [r for r in runs if r.get("error")]
    st = stalemates(runs)
    other = Counter((r.get("error") or "")[:80] for r in errs if not STALE.search(r.get("error") or ""))
    # act funnel from the boss fights of each run (in floor order): boss 1 = act 1, 2 = act 2, 3 / 4 = act 3 (A10: two)
    boss = defaultdict(list)
    for f in fights:
        if f.get("room_type") == "boss" and not f.get("practice"):
            boss[f.get("run_seed")].append(f)
    reach = Counter(); win = Counter(); hp = defaultdict(list); pots = defaultdict(list)
    for seed, fs in boss.items():
        for i, f in enumerate(sorted(fs, key=lambda x: x.get("floor", 0))[:4]):
            reach[i] += 1; win[i] += f.get("result") == "won"
            if f.get("max_hp"):
                hp[i].append(f["start_hp"] / f["max_hp"])
            pots[i].append(f.get("potions_start", 0))
    names = ["act-1 boss", "act-2 boss", "act-3 boss 1", "act-3 boss 2"]
    print(f"runs {n}, won {won} = {100 * won / max(1, n):.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}), errors {len(errs)}, "
          f"mean floor {sum(r.get('floors', 0) for r in runs) / max(1, n):.1f}")
    print("funnel (runs entering / winning each boss fight): " + "; ".join(
        f"{names[i]} {reach[i]} ({100 * reach[i] / max(1, n):.0f}%) won {win[i]} ({100 * win[i] / max(1, reach[i]):.0f}%)" for i in range(4)))
    print("HP share entering: " + "; ".join(
        f"{names[i]} {sum(hp[i]) / max(1, len(hp[i])):.3f} (potions {sum(pots[i]) / max(1, len(pots[i])):.2f})" for i in range(4) if hp[i]))
    print(f"stalemates {sum(st.values())}: " + (", ".join(f"{k} {v}" for k, v in st.most_common()) or "-")
          + (f"; other errors: " + ", ".join(f"{k} ({v})" for k, v in other.most_common(4)) if other else ""))
    deaths = Counter((r.get("killed_by") or "").replace("ENCOUNTER.", "") for r in runs if not r.get("won") and r.get("killed_by"))
    print("deaths: " + ", ".join(f"{k} {v}" for k, v in deaths.most_common(8)))
    look = Counter()
    for r in runs:
        for k, v in (r.get("look") or {}).items():
            if k.startswith("potion_mix"):
                look[k] += v
    print("potion_mix per run: " + (", ".join(f"{k} {v / max(1, n):.2f}" for k, v in sorted(look.items())) or "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curriculum")
    c.add_argument("--runs", required=True); c.add_argument("--stage-file", required=True); c.add_argument("--log", required=True)
    c.add_argument("--t1-a10", type=float, default=0.05); c.add_argument("--t1-low", type=float, default=0.50)
    c.add_argument("--t2-a10", type=float, default=0.12); c.add_argument("--t3-a10", type=float, default=0.0)
    c.add_argument("--min-runs", type=int, default=300)
    c.add_argument("--segment", default="")
    s = sub.add_parser("stage-args"); s.add_argument("stage", type=int, choices=sorted(STAGES))
    e = sub.add_parser("eval"); e.add_argument("dir")
    a = ap.parse_args()
    {"curriculum": cmd_curriculum, "stage-args": cmd_stage_args, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
