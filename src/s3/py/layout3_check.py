"""Checks of the macro row layout v3 (2026-11-16; macro_s3.py header). CPU only, no training run.

A macro file (an ensemble file such as macro_s049.pt) is loaded into the frozen layout v2 network (macro_s3_v2.py) and
into the current one (macro_s3.py). Rows: random valid rows of every decision type (map decisions with map nodes whose
option-reach bits point at the row's map options), plus rows of run chunks (*.s3r of any layout) when given.

  (0) the file loads; the v3 parameters exist in every member and come after all older ones; from a file without them
      the v3 zero layers are exactly zero
  (a) function preserved: v3 inputs random (fights, map nodes, shop prices, removal price): every output of every member
      (value a, c, logit; option and slot logits of every decision type; milestone heads) and the served rows (value
      rows: the ensemble a, c, mean logit and spread the T_f tables use; with and without milestone blending) equal the
      frozen v2 network's: the maximum |difference| is printed (0.0 = bit-identical); v1 / v2 width rows are padded
  (b) every new path is wired: with one zero layer made random at a time, the outputs move (self-attention, cross-
      attention, fight pool, map pool, card / relic / potion properties, deck counts into z and into q, shop prices), and
      the fight and map inputs move the outputs through each path that reads them; the map options' subtree bonus
      changes the map option logits when the option-reach bits change
  (c) the value is monotone in HP with every v3 parameter random (zero layers included), and (a, c) do not depend on HP
  (d) chunks of v1, v2 and v3 widths read into one window (older rows zero-padded); carry files of v1 and v2 (rows and
      Adam states of the frozen networks) load into the v3 trainer: rows padded, Adam states restored for the old
      parameters, an Adam step runs, gradients reach the v3 zero layers; --freeze-v3 keeps them at zero
  (e) with --encode DIR (the output of `dotnet S3.dll encode out=DIR`): the map nodes and fights unpacked from the rows
      equal what the C# encoder meant (rows3.jsonl), with structural checks (one current node, visited path, option bits)
  (f) with --combat FILE: s3_server.py (CPU, --serve macro) answers bridges of the v3, v2 and v1 widths with the rows
      Ensemble.serve computes

Usage: ./.venv/bin/python -I src/s3/py/layout3_check.py MACRO.pt [CHUNK.s3r | DIR ...] [--encode DIR] [--combat FILE]
Prints PASS / FAIL lines; exit code 1 if any check failed.
"""
import argparse, copy, glob, json, mmap, os, shutil, struct, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as ms  # noqa: E402
import macro_s3_v2 as v2m  # noqa: E402
import macro_s3_v1 as v1m  # noqa: E402
import train_macro_s3 as tr  # noqa: E402

FAILS = []
S = ms.S
O = ms.OFF


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def v3_names(net):
    return [k for k, _ in net.named_parameters() if ms.is_v3(k)]


def load_all(path):
    """The file in the frozen v2 network (v3 parameters left out) and in the current one (server / trainer loader)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vocab, special = ck["vocab"], ck["special"]
    rest = [i for i, s in enumerate(special) if s.startswith("REST:")]
    old = []
    for sd in ck["members"]:
        n = v2m.S3MacroNet(vocab, len(special), rest)
        n.load_s1_or_s3({k: v for k, v in sd.items() if not ms.is_v3(k)})
        old.append(n.eval())
    e2 = v2m.Ensemble(old, vocab, special, int(ck.get("version", 1)))
    e2.milestones, e2.alpha, e2.boss1_share = bool(ck.get("milestones", False)), float(ck.get("alpha", 0.0)), float(ck.get("boss1_share", 0.0))
    e3 = ms.Ensemble.load(path, "cpu")
    return ck, e2, e3


def pack_map(typ, row, col, vis, cur, reach, mask):
    return (typ & 15) | (row << ms.MAP_ROW_SHIFT) | (col << ms.MAP_COL_SHIFT) | (vis << ms.MAP_VIS_BIT) | (cur << ms.MAP_CUR_BIT) \
        | (reach << ms.MAP_REACH_BIT) | (mask << ms.MAP_MASK_SHIFT)


def random_rows(n, vocab, special, rng):
    """Valid rows of the current widths with every layout v3 input zero: a state, and by turns no decision, a card
    reward, a map choice, an event, a shop (opt_feat2 zero), a deck-slot decision, a rest site."""
    I = np.zeros((n, ms.IW), np.int64); Fl = np.zeros((n, ms.FW), np.float32)
    rest_ids = [i for i, s in enumerate(special) if s.startswith("REST:")] or [1, 2]

    def ids(kind, size=None):
        return rng.integers(1, max(2, len(vocab[kind])), size)

    def put(r, key, vals):
        I[r, O[key][0]:O[key][0] + len(vals)] = vals

    def opt(r, j, kind, oid, upg=0):
        I[r, O["opt_kind"][0] + j] = kind; I[r, O["opt_id"][0] + j] = oid; I[r, O["opt_upg"][0] + j] = upg

    for r in range(n):
        nd, nr, npot = int(rng.integers(8, 45)), int(rng.integers(1, 20)), int(rng.integers(0, 4))
        ench = rng.random(nd) < 0.1
        put(r, "d_ids", ids("card", nd)); put(r, "d_upg", rng.integers(0, 2, nd))
        put(r, "d_ench", np.where(ench, ids("ench", nd), 0)); put(r, "d_eamt", np.where(ench, rng.integers(1, 6, nd), 0))
        put(r, "d_age", rng.integers(0, 40, nd))
        put(r, "r_ids", ids("relic", nr)); put(r, "r_age", rng.integers(0, 40, nr))
        put(r, "p_ids", ids("potion", npot))
        act, asc, room = int(rng.integers(0, 3)), int(rng.integers(0, 11)), int(rng.integers(2, 10))
        put(r, "ctx", [2, asc, ms.ERA_V0111, act, room, int(ids("encounter"))])
        put(r, "boss", [int(ids("encounter")), int(ids("encounter")) if act == 2 else 0])
        max_hp = int(rng.integers(40, 100)); hp = int(rng.integers(1, max_hp + 1)); in_act = int(rng.integers(1, 17))
        for k, x in (("floor", (act * 17 + in_act) / 50), ("act", act / 3), ("hp", hp / 100), ("max_hp", max_hp / 100), ("hp_frac", hp / max_hp),
                     ("gold", rng.integers(0, 500) / 500), ("deck_size", nd / 40), ("n_upgraded", I[r, O["d_upg"][0]:O["d_upg"][1]].sum() / 20),
                     ("n_enchanted", ench.sum() / 10), ("n_relics", nr / 20), ("n_potions", npot / 5),
                     ("is_combat", float(room in (3, 4, 5))), ("floor_in_act", in_act / 20)):
            Fl[r, S[k]] = x
        Fl[r, ms.F_SCAL2:ms.F_SCAL2 + 6] = rng.random(6)
        kind = r % 7
        if kind == 1:
            for j in range(3):
                opt(r, j, 1, int(ids("card")), int(rng.integers(0, 2)))
            opt(r, 3, 4, 1); I[r, O["dtype"][0]] = 0
        elif kind == 2:
            k = int(rng.integers(2, 5))
            for j in range(k):
                opt(r, j, 5, int(rng.integers(1, 9)))
                Fl[r, 16 + j * ms.MF:16 + (j + 1) * ms.MF] = rng.random(ms.MF)
                Fl[r, ms.F_OPT2 + j * ms.MF2:ms.F_OPT2 + (j + 1) * ms.MF2] = rng.random(ms.MF2)
            I[r, O["dtype"][0]] = 7
        elif kind == 3:
            for j in range(int(rng.integers(2, 4))):
                opt(r, j, 6, int(rng.integers(0, ms.N_EVENT_BUCKETS)))
            I[r, O["dtype"][0]] = 8
        elif kind == 4:
            opt(r, 0, 1, int(ids("card"))); opt(r, 1, 2, int(ids("relic")))
            opt(r, 2, 3, int(ids("potion"))); opt(r, 3, 4, 2); opt(r, 4, 7, 0); I[r, O["dtype"][0]] = 6
        elif kind == 5:
            I[r, O["dtype"][0]] = int(rng.choice(ms.DT_SLOT))
        elif kind == 6:
            for j, sid in enumerate(rest_ids[:4]):
                opt(r, j, 4, sid)
            I[r, O["dtype"][0]] = 3
    return I, Fl


def with_v3(I, Fl, rng, n_enc, groups=("fights", "map", "shop", "rprice")):
    """A copy of the rows with the named layout v3 input groups random and the others zero. Map nodes: 20-64 nodes in
    (row, column) order, one current node, visited nodes below it; on map decision rows the option-reach bits of the
    nodes ahead name the row's map option slots."""
    I, Fl = I.copy(), Fl.copy()
    I[:, ms.IW_V2:] = 0; Fl[:, ms.FW_V2:] = 0; Fl[:, ms.F_SCAL2 + ms.X_REMOVE_PRICE] = 0
    shop = I[:, O["dtype"][0]] == 6
    for r in np.flatnonzero(shop):                    # shop options: opt_feat2 is v3 input there (prices)
        Fl[r, ms.F_OPT2:ms.F_SCAL2] = 0
    n = len(I)
    if "fights" in groups:
        nf = rng.integers(0, ms.NF + 1, n)
        for r in range(n):
            k = int(nf[r])
            I[r, O["f_enc"][0]:O["f_enc"][0] + k] = rng.integers(1, max(2, n_enc), k)
            I[r, O["f_room"][0]:O["f_room"][0] + k] = rng.integers(1, 4, k)
            Fl[r, ms.F_FIGHT:ms.F_FIGHT + k * ms.FFW] = rng.random(k * ms.FFW)
    if "map" in groups:
        for r in range(n):
            k = int(rng.integers(20, ms.NMAP + 1))
            rows = np.sort(rng.integers(0, 18, k)); cols = rng.integers(0, 7, k)
            cur = int(rng.integers(0, k)); nopt = int((I[r, O["opt_kind"][0]:O["opt_kind"][1]] == 5).sum())
            vals = []
            for i in range(k):
                ahead = rows[i] > rows[cur]
                mask = int(rng.integers(1, 1 << max(1, nopt))) if ahead and nopt else (int(rng.integers(0, 8)) if ahead else 0)
                vals.append(pack_map(int(rng.integers(1, 10)), int(rows[i]), int(cols[i]), int(rows[i] < rows[cur] and rng.random() < 0.4) | int(i == cur),
                                     int(i == cur), int(ahead), mask))
            Fl[r, ms.F_MAP:ms.F_MAP + k] = np.asarray(vals, np.float32)
    if "shop" in groups:
        for r in np.flatnonzero(shop):
            Fl[r, ms.F_OPT2:ms.F_SCAL2] = rng.random(ms.K * ms.MF2)
    if "rprice" in groups:
        Fl[:, ms.F_SCAL2 + ms.X_REMOVE_PRICE] = rng.random(n)
    return I, Fl


@torch.no_grad()
def outputs(ens, mod, I, Fl, full=True):
    """Per member: value a, c, logit, option and slot logits, milestone heads; the served rows; with full also the served
    rows with the milestone blend switched on."""
    b = mod.rows_to_batch(I, Fl, "cpu")
    out = {}
    for j, n in enumerate(ens.nets):
        a, c, v, opt, slot = n.forward_rows(b, policy=True)
        ma, mc, ml = n.forward_ms(b)
        out[j] = {"a": a, "c": c, "v": v, "opt": opt, "slot": slot, "ms_a": ma, "ms_c": mc, "ms_logit": ml}
    out["serve"] = {"rows": ens.serve(I, Fl, "cpu")}
    if not full:
        return out
    keep = ens.milestones, ens.alpha, ens.boss1_share
    ens.milestones, ens.alpha, ens.boss1_share = True, 0.3, 0.5
    out["serve_ms"] = {"rows": ens.serve(I, Fl, "cpu")}
    ens.milestones, ens.alpha, ens.boss1_share = keep
    return out


def maxdiff(x, y, keys=None):
    worst, where = 0.0, ""
    for j in x:
        for k in x[j]:
            if keys is not None and k not in keys:
                continue
            d = float((x[j][k].double() - y[j][k].double()).abs().max()) if x[j][k].numel() else 0.0
            if not d <= worst:
                worst, where = d, f"{j}.{k}"
    return worst, where


def set_random(ens, which, seed, std=0.3):
    """Copy of the ensemble with the v3 parameters whose name contains one of `which` random (None: every v3 parameter)."""
    e = copy.deepcopy(ens)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n in e.nets:
            for k, p in n.named_parameters():
                if ms.is_v3(k) and (which is None or any(w in k.split(".") for w in which)):
                    p.copy_(torch.randn(p.shape, generator=g) * std)
                    if k.startswith("map_type."):
                        p[0].zero_()
    return e


@torch.no_grad()
def hp_monotone(ens, I, Fl, rows):
    ii, ff, spans = [], [], []
    for r in rows:
        mx = int(round(float(Fl[r, S["max_hp"]]) * 100))
        if mx < 2:
            continue
        hs = np.arange(0, mx + 1)
        f = np.repeat(Fl[r][None], len(hs), 0); f[:, S["hp"]] = hs / 100.0; f[:, S["hp_frac"]] = hs / mx
        ii.append(np.repeat(I[r][None], len(hs), 0)); ff.append(f); spans.append(len(hs))
    b = ms.rows_to_batch(np.concatenate(ii), np.concatenate(ff), "cpu")
    fall = fall_ms = spread = 0.0; c_min = float("inf")
    for n in ens.nets:
        a, c, v, _, _ = n.forward_rows(b, policy=False)
        ma, mc, ml = n.forward_ms(b)
        c_min = min(c_min, float(c.min()), float(mc.min()))
        o = 0
        for L in spans:
            s = slice(o, o + L); o += L
            fall = max(fall, float((v[s][:-1] - v[s][1:]).max()))
            fall_ms = max(fall_ms, float((ml[s][:-1] - ml[s][1:]).max()))
            for x in (a[s], c[s], ma[s], mc[s]):
                spread = max(spread, float((x.max(0).values - x.min(0).values).max()))
    return fall, fall_ms, spread, c_min, len(spans), int(sum(spans))


def write_chunk(path, I, Fl, n_runs, iw, fw, rng):
    N = len(I)
    cut = np.linspace(0, N, n_runs + 1).astype(int)
    runs = np.zeros((n_runs, tr.RM), np.int32); meta = np.zeros((N, tr.RW), np.float32)
    for r in range(n_runs):
        a, e = int(cut[r]), int(cut[r + 1])
        act = int(rng.integers(0, 3)); won = int(rng.random() < 0.3)
        runs[r] = [r + 1, 7, int(rng.integers(0, 11)), won, 17 * act + 5, act, 0 if not won else 30, 70, 0, a, e - a, 10, 9, 2, 1, 1]
        meta[a:e, tr.M_RUN] = r; meta[a:e, tr.M_FLOOR] = np.arange(e - a) // 3 + 1; meta[a:e, tr.M_ACT] = act
        dec = (I[a:e, O["opt_kind"][0]] != 0) | np.isin(I[a:e, O["dtype"][0]], ms.DT_SLOT)
        meta[a:e, tr.M_KIND] = dec; meta[a:e, tr.M_DTYPE] = np.where(dec, I[a:e, O["dtype"][0]], -1)
        meta[a:e, tr.M_NOPT] = (I[a:e, O["opt_kind"][0]:O["opt_kind"][1]] != 0).sum(1)
    with open(path, "wb") as f:
        f.write(b"S3RUNS01"); f.write(np.array([1, n_runs, N, iw, fw, tr.RM, tr.RW], "<i4").tobytes()); f.write(bytes(64 - 8 - 7 * 4))
        f.write(runs.astype("<i4").tobytes()); f.write(I[:, :iw].astype("<i2").tobytes()); f.write(Fl[:, :fw].astype("<f4").tobytes())
        f.write(meta.astype("<f4").tobytes())


def train_loss(net, b):
    _, _, v, opt, slot = net.forward_rows(b, policy=True)
    return v.mean() + opt.clamp(min=-50).mean() + slot.mean() + net.forward_ms(b)[2].mean()


def chunk_rows(paths, n, rng):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.s3r")))[:4] if os.path.isdir(p) else [p]
    ints, floats = [], []
    for f in files:
        _, i, fl, _ = tr.read_chunk(f)
        ints.append(i.astype(np.int64)); floats.append(fl)
    if not ints:
        return np.zeros((0, ms.IW), np.int64), np.zeros((0, ms.FW), np.float32), 0
    ints, floats = np.concatenate(ints), np.concatenate(floats)
    k = np.sort(rng.choice(len(ints), min(n, len(ints)), replace=False))
    return ints[k], floats[k], len(files)


def make_carry(path, runs, nets, opts, iw, fw):
    carry = {"magic": "S3MCARRY1", "version": 1, "total_runs": len(runs),
             "n_rows": torch.tensor([len(r["meta"]) for r in runs], dtype=torch.int64),
             "n_orig": torch.tensor([len(r["meta"]) for r in runs], dtype=torch.int64),
             **{k: torch.tensor([float(r[k]) for r in runs], dtype=torch.float64) for k in tr.RUN_KEYS},
             "ints": torch.from_numpy(np.ascontiguousarray(tr.cat(runs, "ints")[:, :iw]).astype(np.int16)),
             "floats": torch.from_numpy(np.ascontiguousarray(tr.cat(runs, "floats")[:, :fw])),
             "meta": torch.from_numpy(np.ascontiguousarray(tr.cat(runs, "meta"))),
             "opts": [o.state_dict() for o in opts], "shapes": [[list(p.shape) for p in n.parameters()] for n in nets]}
    torch.save(carry, path)


# ---------------------------------------------------------------- (e) C# round trip

def encode_check(d):
    path = os.path.join(d, "rows3.jsonl")
    rows = [json.loads(l) for l in open(path)]
    bad = {"map": 0, "fights": 0, "struct": 0}
    n_map = n_fight = n_dec = n_mapdec = 0
    examples = []
    for r in rows:
        I = np.asarray(r["ints"], np.int64)[None]; Fl = np.asarray(r["floats"], np.float32)[None]
        if I.shape[1] != ms.IW or Fl.shape[1] != ms.FW:
            bad["struct"] += 1; continue
        b = ms.rows_to_batch(I, Fl, "cpu")
        u = ms.unpack_map(b["map"][0])
        k = int((u["type"] > 0).sum())
        got = [[int(u[x][i]) for x in ("type", "row", "col", "visited", "current", "reach", "mask")] for i in range(k)]
        want = r["map"]
        if got != want or any(int(v) != 0 for v in b["map"][0, k:]):
            bad["map"] += 1
            if len(examples) < 3:
                examples.append(("map", r["run"], r["floor"], got[:3], want[:3]))
        n_map += k
        # fights: tokens vs the C# notes (most recent first)
        fr = b["f_room"][0].numpy(); fe = b["f_enc"][0].numpy(); fn = b["f_num"][0].numpy()
        nf = int((fr > 0).sum())
        ok = nf == len(r["fights"])
        for i, f in enumerate(r["fights"][:ms.NF]):
            enc_tok, _, room, sh, eh, mx, pot, turns, floor, act = f
            mx = max(1, mx)
            want_num = [np.clip((sh - eh) / mx, -1, 1), np.clip(sh / mx, 0, 1), pot / 3, min(turns, 100) / 20,
                        np.clip(r["floor"] - floor, 0, 60) / 20, float(act == r["act"])]
            ok &= fe[i] == enc_tok and fr[i] == room and np.allclose(fn[i], want_num, atol=1e-6)
        n_fight += nf
        if not ok:
            bad["fights"] += 1
            if len(examples) < 6:
                examples.append(("fights", r["run"], r["floor"], fe.tolist(), fr.tolist(), r["fights"][:2]))
        # structure: at most one current node; on map decisions every option slot's own node is in its subtree
        if k and int((u["current"] > 0).sum()) > 1:
            bad["struct"] += 1
        if r["kind"] == 1:
            n_dec += 1
        if r["kind"] == 1 and r["dtype"] == 7:
            n_mapdec += 1
            nopt = r["nopt"]
            for j in range(min(nopt, ms.MAP_SLOTS)):
                if not ((u["mask"][:k] >> j) & 1).any():
                    bad["struct"] += 1
                    break
    check("(e) C# -> Python: map nodes unpacked from every row equal the encoder's nodes", bad["map"] == 0,
          f"{len(rows)} rows ({n_dec} decisions, {n_mapdec} map decisions), {n_map} nodes; {bad['map']} rows differ" +
          (f"; e.g. {examples[0]}" if bad["map"] else ""))
    check("(e) C# -> Python: fight tokens equal the encoder's fights (ids, room kinds, the 6 numbers)", bad["fights"] == 0,
          f"{n_fight} fight tokens; {bad['fights']} rows differ" + (f"; e.g. {[x for x in examples if x[0] == 'fights'][:1]}" if bad["fights"] else ""))
    check("(e) row structure: widths, one current node at most, every map option slot reaches its own subtree", bad["struct"] == 0,
          f"{bad['struct']} rows with a problem")


# ---------------------------------------------------------------- (f) server

def server_check(macro, combat, ens, I, Fl, tmp):
    """s3_server.py --serve macro on CPU, one actor per layout run: the bridge rows of that layout's widths, answered."""
    MAGIC = 0x314F5243414D3353
    for lay in (3, 2, 1):
        iw, fw = ms.WIDTHS[lay]
        pre = f"/dev/shm/l3chk_{os.getpid()}_{lay}_"
        log = open(os.path.join(tmp, f"server{lay}.txt"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
        p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "s3_server.py"), "--serve", "macro", "--prefix", pre + "c",
                              "--mprefix", pre + "m", "--actors", "1", "--mcapacity", "64", "--weights", combat, "--macro", macro,
                              "--device", "cpu", "--mlayout", str(lay), "--idle-exit", "60"], stdout=log, stderr=subprocess.STDOUT, env=env)
        path = pre + "m0"; ok = False; detail = ""
        try:
            t0 = time.time()
            while time.time() - t0 < 120:
                if os.path.exists(path) and os.path.getsize(path) > 256:
                    with open(path, "r+b") as f:
                        mm = mmap.mmap(f.fileno(), 0)
                        if struct.unpack_from("<Q", mm, 0)[0] == MAGIC:
                            break
                        mm.close()
                time.sleep(0.2)
            B, iw_h, fw_h, ow_h = struct.unpack_from("<iiii", mm, 8)
            n = min(B, 48)
            ints = np.frombuffer(mm, "<i4", B * iw_h, 256).reshape(B, iw_h)
            flo = np.frombuffer(mm, "<f4", B * fw_h, 256 + B * iw_h * 4).reshape(B, fw_h)
            out = np.frombuffer(mm, "<f4", B * ow_h, 256 + B * iw_h * 4 + B * fw_h * 4).reshape(B, ow_h)
            ints[:n] = I[:n, :iw]; flo[:n] = Fl[:n, :fw]
            struct.pack_into("<i", mm, 72, n); struct.pack_into("<q", mm, 64, 1)
            t0 = time.time()
            while struct.unpack_from("<q", mm, 128)[0] != 1 and time.time() - t0 < 60:
                time.sleep(0.01)
            got = out[:n].copy()
            want = ens.serve(I[:n, :iw], Fl[:n, :fw], "cpu").numpy()
            d = float(np.abs(got - want).max())
            ok = (iw_h, fw_h) == (iw, fw) and d < 1e-5
            detail = f"header widths {iw_h} / {fw_h}, {n} rows, max |server - Ensemble.serve| {d:.1e}"
            struct.pack_into("<i", mm, 192, 1)
            del ints, flo, out
            mm.close()
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
        finally:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        check(f"(f) s3_server.py --mlayout {lay} serves {iw} / {fw} rows", ok, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("macro"); ap.add_argument("chunks", nargs="*")
    ap.add_argument("--rows", type=int, default=256); ap.add_argument("--random-rows", type=int, default=140)
    ap.add_argument("--hp-rows", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--encode", default="")
    ap.add_argument("--combat", default="", help="combat checkpoint: run the server check (f)")
    ap.add_argument("--tmp", default=os.path.join(HERE, "..", "out"))
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"layout v3: ints {ms.IW} (v2 {ms.IW_V2}; fights from {O['f_enc'][0]}), floats {ms.FW} (v2 {ms.FW_V2}; fight numbers from "
          f"{ms.F_FIGHT}, map nodes from {ms.F_MAP}), outputs {ms.OW}")
    assert (v2m.IW, v2m.FW, v2m.OW) == (ms.IW_V2, ms.FW_V2, ms.OW), "the frozen module is not the v2 layout"

    # ---- (0)
    ck, e2, e3 = load_all(a.macro)
    has_v3 = any(ms.is_v3(k) for k in ck["members"][0])
    names = v3_names(e3.nets[0]); order = [k for k, _ in e3.nets[0].named_parameters()]
    n_new = sum(p.numel() for k, p in e3.nets[0].named_parameters() if ms.is_v3(k))
    n_all = sum(p.numel() for p in e3.nets[0].parameters())
    print(f"file {a.macro}: {len(e3.nets)} members, version {e3.version}, {'has' if has_v3 else 'no'} v3 parameters; "
          f"v3 parameters per member {n_new:,} of {n_all:,}; content properties {e3.nets[0].props_digest or 'none'} "
          f"(card {tuple(e3.nets[0].card_props.shape)}, relic {tuple(e3.nets[0].relic_props.shape)}, potion {tuple(e3.nets[0].pot_props.shape)})")
    check("(0) every v3 module exists in every member", all(sorted({k.split('.')[0] for k in v3_names(n)}) == sorted(ms.V3_TOP) for n in e3.nets))
    check("(0) the v3 parameters come after all older ones (optimizer states of older carry files stay aligned)",
          all(ms.is_v3(k) for k in order[-len(names):]) and not any(ms.is_v3(k) for k in order[:-len(names)]))
    zero = all(bool((p == 0).all()) for n in e3.nets for p in n.v3_zero_params())
    if not has_v3:
        check("(0) loaded from a file without them, the v3 zero layers are exactly zero", zero)
    check("(0) content property tables loaded (non-zero)", float(e3.nets[0].card_props.abs().sum()) > 0)

    # ---- rows
    n_enc = len(ck["vocab"]["encounter"])
    RI, RF = random_rows(a.random_rows, ck["vocab"], ck["special"], rng)
    CI, CF, n_files = chunk_rows(a.chunks, a.rows, rng)
    I = np.concatenate([RI, CI]); Fl = np.concatenate([RF, CF])
    I0, F0 = with_v3(I, Fl, rng, n_enc, groups=())
    Ir, Fr = with_v3(I, Fl, rng, n_enc)
    is_map = I[:, O["opt_kind"][0]:O["opt_kind"][1]] == 5
    print(f"rows: {len(RI)} random + {len(CI)} from {n_files} chunk file(s); {int(is_map.any(1).sum())} map decisions")

    # ---- (a)
    ref = outputs(e2, v2m, I0[:, :ms.IW_V2], F0[:, :ms.FW_V2])
    if has_v3 and not zero:
        print("note: the file has trained v3 parameters: (a) uses a copy with the zero layers at zero")
        ez = copy.deepcopy(e3)
        with torch.no_grad():
            for n in ez.nets:
                for p in n.v3_zero_params():
                    p.zero_()
    else:
        ez = e3
    new_r = outputs(ez, ms, Ir, Fr)
    d, w = maxdiff(new_r, ref)
    check("(a) v3 inputs random: every output equals the frozen v2 network (values, T_f serve rows, policy logits of every "
          "decision type, milestone heads)", d < 1e-6, f"max |difference| {d:.2e}" + (f" at {w}" if d > 0 else " (bit-identical)"))
    for dt in sorted(set(I[:, O['dtype'][0]].tolist())):
        m = torch.from_numpy(I[:, O["dtype"][0]] == dt)
        dd = max(float((new_r[j]["opt"][m] - ref[j]["opt"][m]).abs().max()) for j in range(len(e3.nets)))
        print(f"    decision type {ms.DT[dt] if dt < len(ms.DT) else dt}: {int(m.sum())} rows, max |option logit difference| {dd:.1e}")
    d1, _ = maxdiff(outputs(ez, ms, I0[:, :ms.IW_V2], F0[:, :ms.FW_V2]), outputs(ez, ms, I0, F0))
    d0, _ = maxdiff(outputs(ez, ms, I0[:, :ms.IW_V1], F0[:, :ms.FW_V1]), outputs(ez, ms, *with_v3(*v2_zero(I0, F0), rng, n_enc, groups=())))
    check("(a) v2-width and v1-width rows are accepted and padded with zeros", d1 == 0 and d0 == 0, f"max |difference| v2 {d1:.1e}, v1 {d0:.1e}")
    pi, pf = ms.pad_rows(I0[:, :ms.IW_V2].astype(np.int16), F0[:, :ms.FW_V2])
    check("(a) pad_rows: widths, dtypes, old columns kept, new columns zero", pi.shape == (len(I0), ms.IW) and pf.shape == (len(I0), ms.FW)
          and pi.dtype == np.int16 and np.array_equal(pi[:, :ms.IW_V2], I0[:, :ms.IW_V2]) and not pi[:, ms.IW_V2:].any() and not pf[:, ms.FW_V2:].any())
    eh = set_random(e3, None, a.seed + 7)
    eh.load_members_from(a.macro)
    dh, _ = maxdiff(outputs(eh, ms, Ir, Fr), outputs(e3, ms, Ir, Fr))
    check("(a) hot reload of the file over other v3 parameters gives the file's function", dh < 1e-6, f"max |difference| {dh:.1e}")

    # ---- (b)
    base = outputs(ez, ms, Ir, Fr, False)
    Ir2, Fr2 = with_v3(I, Fl, rng, n_enc)                        # other random v3 inputs
    mk = torch.from_numpy(is_map)
    dec = torch.from_numpy(I[:, O["opt_kind"][0]] != 0)

    def moved(e, x, y, key, members=None):
        return min(float((x[j][key] - y[j][key]).abs().max()) for j in (members or range(len(e.nets))))
    for label, which, key in (("self-attention (attn_out, attn_ff2)", ("attn_out", "attn_ff2"), "a"),
                              ("fight pool", ("fight_pool",), "a"), ("map pool", ("map_pool",), "a"),
                              ("card properties", ("card_prop",), "a"), ("relic properties", ("relic_prop",), "a"),
                              ("potion properties", ("pot_prop",), "a"), ("deck counts into z", ("count_z",), "a")):
        e = set_random(ez, which, a.seed + 11)
        o = outputs(e, ms, Ir, Fr, False)
        check(f"(b) {label}: random zero layer moves the value of every member and the policy", moved(e, o, base, key) > 1e-4
              and float((o[0]["opt"][dec] - base[0]["opt"][dec]).abs().max()) > 1e-4,
              f"min over members of max |change of a| {moved(e, o, base, key):.3f}")
    for label, which in (("deck counts into the query (count_q)", ("count_q",)), ("option cross-attention (x_out)", ("x_out",))):
        e = set_random(ez, which, a.seed + 13)
        o = outputs(e, ms, Ir, Fr, False)
        dv = maxdiff(o, base, keys=("a", "c", "v", "ms_logit"))[0]
        dp = min(float((o[j]["opt"][dec] - base[j]["opt"][dec]).abs().max()) for j in range(len(e.nets)))
        check(f"(b) {label}: moves the option logits and not the value", dp > 1e-4 and dv == 0, f"option logits {dp:.3f}, value {dv:.1e}")
    e = set_random(ez, ("shop_feat",), a.seed + 17)
    o = outputs(e, ms, Ir, Fr, False)
    shop = torch.from_numpy(I[:, O["dtype"][0]] == 6)
    ds = float((o[0]["opt"][shop] - base[0]["opt"][shop]).abs().max()); dn = float((o[0]["opt"][~shop] - base[0]["opt"][~shop]).abs().max())
    check("(b) shop prices (shop_feat): move the logits of shop options only", ds > 1e-4 and dn == 0, f"shop {ds:.3f}, others {dn:.1e}")
    # the inputs reach the outputs through each path
    for label, which, grp, key in (("fight inputs through the fight pool", ("fight_pool",), "fights", "a"),
                                   ("fight inputs through self-attention", ("attn_out", "attn_ff2"), "fights", "a"),
                                   ("fight inputs through the cross-attention", ("x_out",), "fights", "opt"),
                                   ("map inputs through the map pool", ("map_pool",), "map", "a"),
                                   ("map inputs through self-attention", ("attn_out", "attn_ff2"), "map", "a"),
                                   ("map inputs through the cross-attention", ("x_out",), "map", "opt")):
        e = set_random(ez, which, a.seed + 19)
        xa = outputs(e, ms, *with_v3(I, Fl, np.random.default_rng(5), n_enc, groups=(grp,)), False)
        xb = outputs(e, ms, *with_v3(I, Fl, np.random.default_rng(6), n_enc, groups=(grp,)), False)
        dd = min(float((xa[j][key] - xb[j][key]).abs().max()) for j in range(len(e.nets)))
        check(f"(b) {label}: other inputs, other outputs", dd > 1e-4, f"min over members of max |change of {key}| {dd:.3f}")
    # subtree bonus: change only the option-reach bits of the map nodes
    e = set_random(ez, ("x_out",), a.seed + 23)
    Ia, Fa = with_v3(I, Fl, np.random.default_rng(8), n_enc, groups=("map",))
    Fb = Fa.copy()
    mb = Fb[:, ms.F_MAP:ms.FW].astype(np.int64)
    msk = mb >> ms.MAP_MASK_SHIFT
    # flip bit 0 only where another bit is set: the tokens' "reachable from some option" flag stays, only the bonus moves
    mb = np.where((mb > 0) & ((msk & ~1) != 0), (mb & ((1 << ms.MAP_MASK_SHIFT) - 1)) | ((msk ^ 1) << ms.MAP_MASK_SHIFT), mb)
    Fb[:, ms.F_MAP:ms.FW] = mb.astype(np.float32)
    xa, xb = outputs(e, ms, Ia, Fa, False), outputs(e, ms, Ia, Fb, False)
    real = torch.from_numpy(I[:, O["opt_kind"][0]:O["opt_kind"][1]] != 0)
    dm = float((xa[0]["opt"][mk] - xb[0]["opt"][mk]).abs().max()); dno = float((xa[0]["opt"][real & ~mk] - xb[0]["opt"][real & ~mk]).abs().max())
    dv = maxdiff(xa, xb, keys=("a", "c", "v", "slot"))[0]
    check("(b) map option subtree bonus: flipping option-reach bit 0 moves the map option logits and nothing else", dm > 1e-4 and dno == 0 and dv == 0,
          f"map options {dm:.3f}, other options {dno:.1e}, values and slot logits {dv:.1e}")

    # ---- (c)
    pick = rng.choice(len(I), min(a.hp_rows, len(I)), replace=False)
    er = set_random(e3, None, a.seed + 29)
    for name, ens in (("the file's weights", e3), ("every v3 parameter random", er)):
        fall, fall_ms, spread, c_min, n_rows, n_pts = hp_monotone(ens, Ir, Fr, pick)
        check(f"(c) value monotone in HP, {name} (v3 inputs random)", fall <= 1e-4 and fall_ms <= 1e-4 and spread <= 1e-4 and c_min >= 0,
              f"{n_rows} rows x every HP = {n_pts} points x {len(ens.nets)} members: largest fall of the value logit {max(fall, 0):.1e}, "
              f"of a milestone logit {max(fall_ms, 0):.1e}; spread of (a, c) over HP {spread:.1e}; smallest ramp weight {c_min:.1e}")

    # ---- (d)
    os.makedirs(a.tmp, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="layout3_check_", dir=a.tmp)
    try:
        n3 = len(RI) // 3
        Iv, Fv = with_v3(RI, RF, rng, n_enc)
        p1, p2, p3 = (os.path.join(tmp, f"{x}_000000.s3r") for x in ("v1", "v2", "v3"))
        write_chunk(p1, *v2_zero(RI[:n3], RF[:n3]), 6, ms.IW_V1, ms.FW_V1, rng)
        write_chunk(p2, RI[n3:2 * n3], RF[n3:2 * n3], 6, ms.IW_V2, ms.FW_V2, rng)
        write_chunk(p3, Iv[2 * n3:], Fv[2 * n3:], 6, ms.IW, ms.FW, rng)
        st = tr.Store(10**6)
        got = st.scan(None, files=[p1, p2, p3])
        wi, wf = tr.cat(st.runs, "ints"), tr.cat(st.runs, "floats")
        ok = (len(got) == 18 and wi.shape == (len(RI), ms.IW) and not wi[:2 * n3, ms.IW_V2:].any() and not wf[:2 * n3, ms.FW_V2:].any()
              and not wi[:n3, ms.IW_V1:].any() and np.array_equal(wi[n3:2 * n3, :ms.IW_V2], RI[n3:2 * n3, :ms.IW_V2])
              and np.array_equal(wi[2 * n3:], Iv[2 * n3:]) and np.array_equal(wf[2 * n3:], Fv[2 * n3:]))
        check("(d) chunks of layouts v1, v2 and v3 read into one window (older rows zero-padded, v3 rows as written)", ok,
              f"{len(got)} runs, rows {wi.shape} ints, {wf.shape} floats")
        old_runs = st.runs[:6]
        for lay, mod, ens_old in ((1, v1m, None), (2, v2m, e2)):
            if ens_old is None:
                nets_o = []
                for sd in ck["members"]:
                    n = v1m.S3MacroNet(ck["vocab"], len(ck["special"]), [i for i, s in enumerate(ck["special"]) if s.startswith("REST:")])
                    n.load_s1_or_s3({k: v for k, v in sd.items() if not ms.is_v3(k) and k.split(".")[0] not in ms.V2_PARAMS})
                    nets_o.append(n.eval())
            else:
                nets_o = copy.deepcopy(ens_old.nets)
            iw, fw = ms.WIDTHS[lay]
            opts_o = [torch.optim.Adam(n.parameters(), lr=1e-5) for n in nets_o]
            bo = mod.rows_to_batch(RI[:70, :iw], RF[:70, :fw], "cpu")
            for n, op in zip(nets_o, opts_o):
                op.zero_grad(); train_loss(n, bo).backward(); op.step()
            pc = os.path.join(tmp, f"carry_v{lay}.pt")
            make_carry(pc, old_runs, nets_o, opts_o, iw, fw)
            nets3 = copy.deepcopy(e3.nets)
            opts3 = [torch.optim.Adam(n.parameters(), lr=1e-4) for n in nets3]
            st2 = tr.Store(10**6)
            tr.load_carry(pc, st2, opts3, nets3, 1e-4, "cpu")
            ci = tr.cat(st2.runs, "ints") if st2.runs else np.zeros((0, 0))
            check(f"(d) a layout v{lay} carry file loads: rows zero-padded", len(st2.runs) == len(old_runs) and ci.shape[1:] == (ms.IW,)
                  and not ci[:, iw:].any(), f"{len(st2.runs)} runs, {len(ci)} rows")
            n_state = [len(op.state_dict()["state"]) for op in opts3]; want = [len(o.state_dict()["state"]) for o in opts_o]
            check(f"(d) ... its Adam states are restored for the old parameters", n_state == want and min(n_state) > 0,
                  f"parameters with state per member {n_state} (carry {want}; network {len(list(nets3[0].parameters()))})")
            b3 = ms.rows_to_batch(Iv[:70], Fv[:70], "cpu")
            grads = {}
            for n, op in zip(nets3, opts3):
                op.zero_grad(); train_loss(n, b3).backward()
                for k, p in n.named_parameters():
                    if ms.is_v3_zero(k):
                        grads.setdefault(k, []).append(0.0 if p.grad is None else float(p.grad.abs().max()))
                op.step()
            finite = all(bool(torch.isfinite(p).all()) for n in nets3 for p in n.parameters())
            zero_left = [k for k, g in grads.items() if min(g) == 0]
            check(f"(d) ... an Adam step runs; gradients reach every v3 zero layer of every member", finite and not zero_left,
                  f"{len(grads)} zero-layer tensors" + (f"; without gradient: {zero_left}" if zero_left else ""))
        # --freeze-v3 (control arm): zero layers stay zero through a step
        nets_f = copy.deepcopy(e3.nets)
        for n in nets_f:
            for p in n.v3_zero_params():
                p.requires_grad_(False)
        opts_f = [torch.optim.Adam(n.parameters(), lr=1e-3) for n in nets_f]
        for n, op in zip(nets_f, opts_f):
            op.zero_grad(); train_loss(n, ms.rows_to_batch(Iv[:70], Fv[:70], "cpu")).backward(); op.step()
        check("(d) --freeze-v3: the v3 zero layers stay exactly zero through an Adam step",
              all(bool((p == 0).all()) for n in nets_f for p in n.v3_zero_params()))
        if a.encode:
            encode_check(a.encode)
        if a.combat:
            server_check(a.macro, a.combat, e3, Ir, Fr, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED: ' + '; '.join(FAILS)} ({time.time() - t0:.1f} s)")
    sys.exit(1 if FAILS else 0)


def v2_zero(I, Fl):
    """Copy of the rows with the layout v2 inputs (boss ids, opt_feat2, scal2) zero: what a v1 row holds."""
    I, Fl = I.copy(), Fl.copy()
    I[:, ms.IW_V1:] = 0; Fl[:, ms.FW_V1:] = 0
    return I, Fl


if __name__ == "__main__":
    main()
