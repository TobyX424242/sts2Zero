#!/usr/bin/env python3
"""Where S3 self-play runs are lost: deaths by act and room, boss fights by encounter, HP at boss starts, macro habits,
and where the search budget goes. Reads the run and fight logs of the given segment directories.
usage: diag_runs.py SEGMENT_DIR... [--asc 10]
Runs with errors and runs with perturbed fight starts are excluded; fights are joined to their run by seed, so fights
of runs still in flight are skipped."""
import glob, sys, collections
try:
    import orjson
    loads = orjson.loads
except ImportError:
    import json
    loads = json.loads

args = sys.argv[1:]
asc = 10
if "--asc" in args:
    i = args.index("--asc"); asc = int(args[i + 1]); del args[i:i + 2]
segs = args


def kind(enc):
    e = enc or ""
    return "boss" if e.endswith("_BOSS") else "elite" if e.endswith("_ELITE") else "hallway" if e else "none"


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    - "


runs = {}          # seed -> (segment index, group, record)
per_seg = []
for si, seg in enumerate(segs):
    rows = []
    for f in sorted(glob.glob(f"{seg}/runs/*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                try: r = loads(line)
                except Exception: continue
                if r.get("error") or r.get("perturbed"): continue
                g = "target" if r["ascension"] == asc else "lower"
                runs[r["seed"]] = (si, g, r)
                rows.append((g, r))
    per_seg.append(rows)

print(f"== runs by segment (errors and perturbed runs excluded); target = A{asc}, lower = the levels below")
for g in ("target", "lower"):
    print(f"-- {g}")
    for si, seg in enumerate(segs):
        rs = [r for gg, r in per_seg[si] if gg == g]
        n = len(rs)
        if not n: continue
        acts = collections.Counter(3 if r["won"] else min(r["act"], 2) for r in rs)
        dk = collections.Counter((min(r["act"], 2) + 1, kind(r.get("killed_by"))) for r in rs if not r["won"])
        m = lambda k: sum(r[k] for r in rs) / n
        print(f"  {seg.split('/')[-1]}: runs {n}  won {pct(sum(r['won'] for r in rs), n)}  died in act 1/2/3: {pct(acts[0], n)} {pct(acts[1], n)} {pct(acts[2], n)}"
              f"  | deaths: a1 hall {pct(dk[(1, 'hallway')], n)} elite {pct(dk[(1, 'elite')], n)} boss {pct(dk[(1, 'boss')], n)}"
              f"; a2 hall {pct(dk[(2, 'hallway')], n)} elite {pct(dk[(2, 'elite')], n)} boss {pct(dk[(2, 'boss')], n)}"
              f"; a3 hall {pct(dk[(3, 'hallway')], n)} elite {pct(dk[(3, 'elite')], n)} boss {pct(dk[(3, 'boss')], n)}"
              f"; no fight {pct(sum(v for (a, k), v in dk.items() if k == 'none'), n)}")
        print(f"      per run: floors {m('floors'):.1f} elites {m('elites'):.2f} rests {m('rests'):.2f} smiths {m('smiths'):.2f} shops {m('shops'):.2f} events {m('events'):.2f}"
              f" cards taken {m('cards_taken'):.1f} skipped {m('cards_skipped'):.1f} deck {m('deck'):.1f} relics {m('relics'):.1f} potions used {m('potions_used'):.1f}"
              f" sims {m('simulations'):.0f} seconds {m('seconds'):.0f}")

# fights
F = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0])   # n, won, start hp frac, potions start, turns, floor, sims, hp lost (won), root q
B = collections.defaultdict(lambda: [0, 0])                                 # boss fights by (group, act, hp bucket)
P = collections.defaultdict(lambda: [0, 0])                                 # boss fights by (group, act, potions at start)
T = collections.defaultdict(lambda: [0, 0, 0.0])                            # by (group, room kind): fights, sims, seconds
half = len(segs) // 2
for si, seg in enumerate(segs):
    part = "early" if si < half else "late"
    for f in sorted(glob.glob(f"{seg}/fights/*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                try: x = loads(line)
                except Exception: continue
                rr = runs.get(x.get("run_seed"))
                if rr is None or x.get("error"): continue
                g = rr[1]; k = x["room_type"]
                won = x["result"] == "won"
                t = T[(g, k)]; t[0] += 1; t[1] += x["simulations"]; t[2] += x["seconds"]
                if k != "boss" and k != "elite": continue
                a = F[(g, part, k, x["encounter"])]
                hpf = x["start_hp"] / max(1, x["max_hp"])
                a[0] += 1; a[1] += won; a[2] += hpf; a[3] += x["potions_start"]; a[4] += x["turns"]; a[5] += x["floor"]; a[6] += x["simulations"]
                if won: a[7] += (x["start_hp"] - x["end_hp"]) / max(1, x["max_hp"])
                a[8] += x.get("mean_root_q") or 0.0
                if k == "boss":
                    act = 1 if x["floor"] <= 20 else 2 if x["floor"] <= 38 else 3
                    b = B[(g, act, min(4, int(hpf * 5)))]; b[0] += 1; b[1] += won
                    p = P[(g, act, min(3, x["potions_start"]))]; p[0] += 1; p[1] += won

for k in ("boss", "elite"):
    print(f"== {k} fights by encounter: early = first {half} segment(s), late = the rest (n, won, HP share at start, potions at start, turns, mean floor, sims per fight, HP share lost when won, mean root value)")
    for g in ("target", "lower"):
        print(f"-- {g}")
        encs = sorted({e for (gg, p, kk, e) in F if gg == g and kk == k}, key=lambda e: F[(g, "late", k, e)][5] / max(1, F[(g, "late", k, e)][0]))
        for e in encs:
            s = f"  {e.replace('ENCOUNTER.', ''):28s}"
            for part in ("early", "late"):
                a = F[(g, part, k, e)]; n = a[0]
                s += (f" | {part} n {n:5d} won {pct(a[1], n)} hp {a[2] / n:.2f} pot {a[3] / n:.2f} turns {a[4] / n:4.1f} floor {a[5] / n:4.1f} sims {a[6] / n:5.0f}"
                      f" lost {a[7] / max(1, a[1]):.2f} q {a[8] / n:.2f}") if n else f" | {part} n     0"
            print(s)

print("== boss fights won by HP share at the start (fifths) and by potions held at the start, all segments")
for g in ("target", "lower"):
    for act in (1, 2, 3):
        s = f"  {g:6s} act {act}: HP " + "  ".join(f"{20 * b}-{20 * b + 20}%: {pct(B[(g, act, b)][1], B[(g, act, b)][0])} (n {B[(g, act, b)][0]})" for b in range(5))
        s += "  || potions " + "  ".join(f"{p}{'+' if p == 3 else ''}: {pct(P[(g, act, p)][1], P[(g, act, p)][0])} (n {P[(g, act, p)][0]})" for p in range(4))
        print(s)

print("== where the search budget goes (fights of finished runs): share of fights, of simulations, of fight seconds")
for g in ("target", "lower"):
    tf = sum(v[0] for (gg, k), v in T.items() if gg == g); ts = sum(v[1] for (gg, k), v in T.items() if gg == g); tt = sum(v[2] for (gg, k), v in T.items() if gg == g)
    print(f"  {g}: " + "; ".join(f"{k} fights {pct(v[0], tf)} sims {pct(v[1], ts)} seconds {pct(v[2], tt)} ({v[1] / max(1, v[0]):.0f} sims/fight)" for (gg, k), v in sorted(T.items()) if gg == g))
