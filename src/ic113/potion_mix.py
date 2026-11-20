#!/usr/bin/env python3
"""ic113 mixed potion valuation, the Python side (PLAN step 8; src/ic113/CHAIN.md; actor: S3 play potion_mix=FILE,
src/s3/S3/PotionMix.cs).

dV(s, k) = V_run(s with the belt replaced by the single potion k) - V_run(s with an empty belt), V_run = sigmoid of the
ensemble's value output for a row without a decision (Ensemble.serve_values, the function the macro server answers the
actor's rows with). The belt change is MacroRow.SetPotions: potion ids, potions / 5, free slots / 3 (from the row's slot
count); everything else of the row is kept. Vbar_act(k) = mean of dV(s, k) over sampled self-play states s of act `act`
(value rows = states after a floor, of runs without perturbed starts, potion slots > 0).

  potion_mix.py vbar  --carry CARRY_MACRO.pt --macro MACRO.pt --out FILE [--rows 500] [--recent-runs 20000] [--asc all]
                      [--device cpu] [--threads 16] [--seed 0] [--table T.json [--w 0,0.35,0.70]]
      Vbar per act and kind from the newest runs of a macro trainer carry (the value window it saves at a segment's end);
      --table adds the phase-2 block (T(k) in win points from potion_probe.sh, c = sd_k(T) / sd_k(Vbar_3) over the kinds
      in both, weights w for acts 1 / 2 / 3).
  potion_mix.py empty --macro MACRO.pt --out FILE [--why TEXT]
      A file without means (the chain's first segment: no self-play states yet): the actor scores every kind by its state
      term alone (V_mix = dV).
  potion_mix.py check --dump LOOK_DUMP.jsonl [...] --macro MACRO.pt [--tol 2e-4]
      The actor's dV of its potion_mix decisions (look_dump records of type potion_mix, macro file fixed: evaluation)
      against this file's row change and network: the C# and Python belt changes must give the same values.
  potion_mix.py show FILE
"""
import argparse, glob, hashlib, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "s3", "py"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as ms  # noqa: E402

S_POT = 12                      # scal: potions / 5 (MacroLayout.SPotions)
X_FREE, X_SLOTS = 3, 4          # scal2: free slots / 3, slots / 5 (ML.XPotFree, ML.XPotSlots)
P0 = ms.OFF["p_ids"][0]
CTX = ms.OFF["ctx"][0]


def with_belt(I0, F0, belts):
    """Copies of rows (I0[i], F0[i]) with belt belts[i] (list of potion tokens): MacroRow.SetPotions."""
    I, Fl = I0.copy(), F0.copy()
    I[:, P0:P0 + ms.MAX_POT] = 0
    for r, b in enumerate(belts):
        b = b[-ms.MAX_POT:]
        I[r, P0:P0 + len(b)] = b
        slots = int(round(float(Fl[r, ms.F_SCAL2 + X_SLOTS]) * 5))
        Fl[r, S_POT] = np.float32(min(len(b), ms.MAX_POT) / 5.0)
        Fl[r, ms.F_SCAL2 + X_FREE] = np.float32(max(0, slots - len(b)) / 3.0)
    return I, Fl


@torch.no_grad()
def values(ens, I, Fl, dev, bs=1024):
    """V_run (probability) of value rows: sigmoid of the ensemble's value output, as MacroEval.Value."""
    out = []
    for s in range(0, len(I), bs):
        o = ens.serve_values(ms.rows_to_batch(I[s:s + bs], Fl[s:s + bs], dev))
        out.append(torch.sigmoid(o[:, ms.O_V].double()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def kinds_of(ens):
    return [(i, p) for i, p in enumerate(ens.vocab["potion"]) if p.startswith("POTION.")]


def load_states(carry, rows, recent, asc, rng):
    ck = torch.load(carry, map_location="cpu", weights_only=False)
    n_rows = ck["n_rows"].numpy().astype(np.int64)
    ints, floats = ms.pad_rows(ck["ints"].numpy(), ck["floats"].numpy())
    run_asc, pert = ck["asc"].numpy(), ck["perturbed"].numpy()
    total = len(n_rows)
    first_run = max(0, total - recent) if recent > 0 else 0
    starts = np.concatenate([[0], np.cumsum(n_rows)[:-1]]) if total else np.zeros(0, np.int64)
    keep = np.zeros(len(ints), bool)
    for i in range(first_run, total):
        if bool(pert[i]) or (asc != "all" and int(run_asc[i]) not in asc):
            continue
        keep[starts[i]:starts[i] + n_rows[i]] = True
    keep &= ~ms.decision_rows(ints)
    keep &= floats[:, ms.F_SCAL2 + X_SLOTS] > 0
    act = ints[:, CTX + 3]
    out, avail = {}, {}
    for a in (0, 1, 2):
        idx = np.flatnonzero(keep & (act == a))
        avail[a + 1] = int(len(idx))
        if len(idx) > rows:
            idx = np.sort(rng.choice(idx, rows, replace=False))
        out[a + 1] = idx
    info = {"carry_runs": int(total), "runs_used": int(total - first_run), "rows_available": avail}
    return ints, floats, out, info


def cmd_vbar(a):
    t0 = time.time()
    torch.set_num_threads(a.threads)
    dev = a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    asc = "all" if a.asc == "all" else {int(x) for x in a.asc.split(",")}
    ints, floats, ctx, info = load_states(a.carry, a.rows, a.recent_runs, asc, rng)
    print(f"[vmix] carry {a.carry}: {info['carry_runs']} runs, newest {info['runs_used']} used; value rows available per act "
          f"{info['rows_available']}; sampled " + ", ".join(f"act {k} {len(v)}" for k, v in ctx.items()) +
          f" ({time.time() - t0:.0f} s)", flush=True)
    ens = ms.Ensemble.load(a.macro, dev)
    if a.value_members:
        ens.set_value_members(a.value_members)
    kinds = kinds_of(ens)
    res = {"version": 1, "units": "vrun", "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "macro": os.path.abspath(a.macro), "macro_md5": md5(a.macro), "carry": os.path.abspath(a.carry),
           "source": f"carry {os.path.basename(os.path.dirname(os.path.abspath(a.carry)))}, newest {info['runs_used']} runs, asc {a.asc}",
           "rows": a.rows, "seed": a.seed, "info": info, "acts": {}, "n": {}, "sd": {}, "v_empty": {}}
    if a.value_members:
        res["value_members"] = list(ens.value_members)
    for act, idx in ctx.items():
        if len(idx) < a.min_rows:
            print(f"[vmix] act {act}: {len(idx)} states < --min-rows {a.min_rows}: no means for this act (state term alone)", flush=True)
            continue
        t1 = time.time()
        I0, F0 = ints[idx], floats[idx]
        base = values(ens, *with_belt(I0, F0, [[]] * len(idx)), dev)
        # one big batch per chunk of kinds: rows repeated per kind
        D = np.zeros((len(kinds), len(idx)))
        for ki, (tok, _) in enumerate(kinds):
            D[ki] = values(ens, *with_belt(I0, F0, [[tok]] * len(idx)), dev) - base
        mean, sd = D.mean(1), D.std(1)
        res["acts"][str(act)] = {p: round(float(m), 6) for (_, p), m in zip(kinds, mean)}
        res["sd"][str(act)] = {p: round(float(s), 6) for (_, p), s in zip(kinds, sd)}
        res["n"][str(act)] = int(len(idx)); res["v_empty"][str(act)] = round(float(base.mean()), 6)
        order = np.argsort(-mean)
        short = lambda p: p.split(".", 1)[-1]
        print(f"[vmix] act {act}: {len(idx)} states, V(empty belt) {100 * base.mean():.1f}%, Vbar over {len(kinds)} kinds: "
              f"mean {100 * mean.mean():+.2f} sd {100 * mean.std():.2f} points, per-state sd of a kind {100 * sd.mean():.2f}; "
              f"top: " + ", ".join(f"{short(kinds[i][1])} {100 * mean[i]:+.2f}" for i in order[:6]) +
              "; bottom: " + ", ".join(f"{short(kinds[i][1])} {100 * mean[i]:+.2f}" for i in order[-3:]) +
              f" ({time.time() - t1:.0f} s, {len(idx) * (len(kinds) + 1)} rows)", flush=True)
    if a.table:
        T = json.load(open(a.table))
        w = [float(x) for x in a.w.split(",")]
        v3 = res["acts"].get("3", {})
        common = [k for k in T if k in v3]
        if len(common) < 3:
            raise SystemExit(f"[vmix] --table: only {len(common)} kinds in both the table and Vbar_3: c is undefined")
        c = float(np.std([T[k] for k in common]) / max(1e-9, np.std([v3[k] for k in common])))
        res["phase2"] = {"table": {k: float(v) for k, v in T.items()}, "table_file": os.path.abspath(a.table), "c": c,
                         "w": {"1": w[0], "2": w[1], "3": w[2]}, "kinds_common": len(common)}
        print(f"[vmix] phase 2 block: table {len(T)} kinds ({len(common)} also in Vbar_3), c = sd(T) / sd(Vbar_3) = {c:.2f}, w {w}", flush=True)
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
    print(f"[vmix] wrote {a.out}: means for acts {sorted(res['acts'])} ({time.time() - t0:.0f} s, device {dev})", flush=True)


def cmd_empty(a):
    res = {"version": 1, "units": "vrun", "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "macro": os.path.abspath(a.macro), "source": a.why, "acts": {}, "n": {}}
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
    print(f"[vmix] wrote {a.out}: no means ({a.why}): every kind scored by its state term alone", flush=True)


def cmd_check(a):
    dev = "cpu"
    ens = ms.Ensemble.load(a.macro, dev)
    if a.value_members:
        ens.set_value_members(a.value_members)
    tok = {p: i for i, p in enumerate(ens.vocab["potion"])}
    recs = []
    for pat in a.dump:
        for p in sorted(glob.glob(pat)):
            for line in open(p):
                d = json.loads(line)
                if d.get("type") == "potion_mix":
                    recs.append(d)
    if not recs:
        raise SystemExit("[vmix] check: no potion_mix records")
    worst, n, swaps, nomean = 0.0, 0, 0, 0
    for d in recs:
        I0 = np.array([d["base"]["i"]], dtype=np.int64); F0 = np.array([d["base"]["f"]], dtype=np.float32)
        belts = [[]] + [[tok.get(k, 1)] for k in d["kinds"]]
        v = values(ens, *with_belt(np.repeat(I0, len(belts), 0), np.repeat(F0, len(belts), 0), belts), dev)
        dv = v[1:] - v[0]
        worst = max(worst, float(np.abs(dv - np.array(d["dv"])).max()), abs(float(v[0]) - d["v_empty"]))
        n += 1; swaps += d["choice"] >= 0; nomean += sum(x is None for x in d["vbar"])
        # the decision itself from the record's numbers
        sc = dict(zip(d["kinds"], d["score"]))
        held = d["belt"]; w = min(range(len(held)), key=lambda j: (sc[held[j]], j))
        want = w if sc[d["offered"]] > sc[held[w]] + d["margin"] else -1
        if want != d["choice"]:
            raise SystemExit(f"[vmix] check FAILED: decision {want} expected, record has {d['choice']} ({d['seed']} floor {d['floor']})")
    ok = worst <= a.tol
    print(f"[vmix] check {'PASSED' if ok else 'FAILED'}: {n} potion_mix decisions ({swaps} swaps, {nomean} kinds without Vbar); "
          f"largest |dV(actor) - dV(python)| or |V(empty)| difference {worst:.2e} (tolerance {a.tol:g}); decisions re-derived from the scores", flush=True)
    sys.exit(0 if ok else 1)


def cmd_show(a):
    d = json.load(open(a.file))
    print(f"{a.file}: source {d.get('source')}, macro {d.get('macro')}, made {d.get('made')}")
    for act in sorted(d.get("acts", {})):
        m = d["acts"][act]; o = sorted(m, key=lambda k: -m[k])
        print(f"  act {act} (n {d['n'].get(act)}): " + ", ".join(f"{k.split('.', 1)[-1]} {100 * m[k]:+.2f}" for k in o))
    if "phase2" in d:
        print(f"  phase2: c {d['phase2']['c']:.3f}, w {d['phase2']['w']}, table {len(d['phase2']['table'])} kinds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vbar")
    v.add_argument("--carry", required=True); v.add_argument("--macro", required=True); v.add_argument("--out", required=True)
    v.add_argument("--rows", type=int, default=500); v.add_argument("--min-rows", type=int, default=20)
    v.add_argument("--recent-runs", type=int, default=20000, help="newest runs of the carry to sample from (0 = all)")
    v.add_argument("--asc", default="all", help="ascensions of the runs sampled ('all' or e.g. '10' or '8,9,10')")
    v.add_argument("--device", default="cpu"); v.add_argument("--threads", type=int, default=16); v.add_argument("--seed", type=int, default=0)
    v.add_argument("--table", default=""); v.add_argument("--w", default="0,0.35,0.70")
    v.add_argument("--value-members", default="", help="comma-separated ensemble member indices for V_run (default '' = all)")
    e = sub.add_parser("empty")
    e.add_argument("--macro", required=True); e.add_argument("--out", required=True); e.add_argument("--why", default="no self-play states yet")
    c = sub.add_parser("check")
    c.add_argument("--dump", nargs="+", required=True); c.add_argument("--macro", required=True); c.add_argument("--tol", type=float, default=2e-4)
    c.add_argument("--value-members", default="", help="comma-separated ensemble member indices (must match server)")
    s = sub.add_parser("show"); s.add_argument("file")
    a = ap.parse_args()
    {"vbar": cmd_vbar, "empty": cmd_empty, "check": cmd_check, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
