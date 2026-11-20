"""S3 macro trainer (PROPOSAL.md §6.5): PPO for the macro policy and V_run on the agent's own run outcomes.

Reads the run chunks written by the C# actors (src/s3/S3/Records.cs RunChunkWriter, *.s3r) and publishes the macro
ensemble file the inference server reloads (macro_s3.Ensemble: member 0 is the policy member).

Per iteration (every --runs-per-iter new runs):
  policy (member 0): PPO with the clipped ratio against the recorded behaviour log-probabilities; advantages by GAE
    over the run's macro decisions (reward = 1 at a won run's end, else 0; gamma 1); entropy bonus; KL penalty toward
    the behaviour-cloning policy (the frozen S1 member 0) on the decision types S1 covered, its weight falling linearly
    to zero when --kl-runs runs have been ingested (the plan: the first quarter of the budget);
  value (all members; the policy member only its value heads, so that the value loss cannot move the policy):
    binary cross-entropy of V(row) against the run outcome, on the states after every floor and on
    the decision states, sampled from the last --window-runs runs. The target is the outcome itself (lambda = 1): the
    tables T_f need a calibrated value for this agent, and bootstrapping from the human-fitted start would keep its
    bias. Members draw independent samples, so their spread stays a measure of disagreement.
  outcome = alpha x won + (1 - alpha) x (acts cleared / 3). A new agent wins no runs (0 of 1,024 at ascension 0 in the
    first test), so a pure win target sends V_run to zero everywhere and the tables and advantages carry no signal;
    acts cleared is the same goal at a horizon the agent reaches. alpha rises with the win rate of the window and is 1
    (the plan's pure P(win)) from --alpha-full on.
  Runs with perturbed fight starts are used for the value only when --no-perturbed-policy is given (default: both).

Carry-over between trainer processes (S3 segments; off by default): --save-carry P writes the value window (the last
--window-runs runs) and the five Adam states to P on SIGTERM/SIGINT and at the end (temp file + rename); --load-carry P
puts those runs into the value window and restores the optimizers whose parameter shapes match. PPO is not affected:
carried runs are never used for the policy (only runs ingested by this process are), and the run counter is unchanged.
--carry-every S (2026-11-19, interruptible servers; 0 = off): also write the --save-carry file every S seconds of wall
time, between iterations (same code path and file; fsync'ed before the rename), so a segment killed without a signal
keeps a carry at most S seconds + one iteration old. A SIGTERM during a periodic save abandons it (the temp file) and
the save at exit writes the file again.

Value-fit options (2026-11-13, IMPROVEMENT_PLAN.md change A; all off by default = the old behaviour):
  --value-rows-per-run K  the value window keeps K random rows of each run (PPO still uses every decision of the new
                 runs): ~95 rows of one run share one label, so whole runs in the window let the members memorise runs
  --value-steps N  value steps per member per iteration (default 0: --value-reuse x new rows / batch)
  --value-dropout  members 1-4 take their value steps in training mode (dropout; era dropout off)
  --value-wd W     decoupled weight decay W x lr per value step on weight matrices (member 0: value heads only)
  --prob-avg       advantages and logged ensemble value from the mean of member probabilities, not of logits
  --policy-warmup-s S  runs ingested in the first S seconds of this process train the value only (a segment's first
                 runs are its shortest: centring their advantages within that batch biases the policy)
Row mix of the value window (2026-11-14; both off by default). K rows of every run over-weight short runs among the rows
of the early acts (a run that dies in act 1 has ~33 rows, a won run ~120), which pulled the act-1 value 0.04-0.07 below
the outcomes of fresh runs. Two ways to give every recorded row the same chance again:
  --value-row-share S  the window keeps the share S of each run's rows (random rows, stochastic rounding) instead of K
  --value-row-weights  with --value-rows-per-run K: value batches draw window rows in proportion to
                 rows of the run / rows kept
Milestone heads (2026-11-14; off by default; macro_s3.py header): --milestones trains the four milestone heads of every
member next to the old value head (labels from the run: act 1 cleared, act 2 cleared, first act-3 boss beaten, won;
each head on the rows of runs that reached the stage before it) and, after --ms-warmup-iters iterations (or at once
when the --init file already has trained heads), takes the value for the advantages, the logs and the served tables
from their blend; the file then carries alpha and the boss-1 share for the server and the actor. --boss1-share W
(default 0 = D14 as it is): at the double-boss ascension, beating the first act-3 boss is worth W of the last act in the
outcome (target of the old head, the advantages and the blend alike).
The per-iteration log line also carries `value_cells` (log only): per ascension group x act on the fresh runs' floor
rows, how much of the loss any predictor could gain over the cell mean and the share the value takes, in-floor AUC for
the next milestone and for the run win, the value by HP fifth and the share of values below the banked progress.
Offline replay (--replay LIST): read the *.s3r files named in LIST (one path per line, in that order) instead of
watching --mdata; files are not moved or deleted; the trainer exits when the list is used up. With --value-only this
replays a stored run through the value trainer (the logged fresh-run loss is measured before each update).

Row layout v2 (2026-11-16, macro_s3.py header): chunks and carry files written with the v1 row widths (ints 460,
floats 208) are still read; their rows are zero-padded to the current widths (macro_s3.pad_rows), so the window can
hold rows of both kinds. A carry file's optimizer states are restored for the parameters it has (the layout v2
parameters are registered last and start with fresh Adam state); the carry file is always written in the current widths.

Row layout v3 (2026-11-16, macro_s3.py header): chunks and carry files of layouts v1 (460 / 208), v2 (462 / 344) and v3
(478 / 472) are read; older rows are zero-padded (no fights, no map nodes). A carry file's optimizer states are restored
for the parameters it has (the v3 parameters are registered after the v2 ones and start with fresh Adam state).
--freeze-v3 keeps the v3 zero layers at zero (they get no updates): the control arm of src/exp/l3_value.sh.

Usage: python -I src/s3/py/train_macro_s3.py --mdata DIR --macro OUT.pt [--init src/s3/data/macro_s1.pt]
"""
import argparse, glob, json, math, os, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import macro_s3 as ms  # noqa: E402

RM, RW = 16, 16   # run-meta and row-meta widths
# row meta columns
M_RUN, M_KIND, M_FLOOR, M_DTYPE, M_CHOSEN, M_LOGP, M_VALUE, M_VSTD, M_VER, M_NOPT, M_MASK, M_ACT, M_VPOL = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15
BC_TYPES = (0, 1, 2, 3, 4, 5, 6)
N_ACTS = 3


BOSS1_SHARE = [0.0]   # --boss1-share (module level so that outcome() keeps its signature)


def outcome(r, alpha):
    """The run outcome the value and the policy are trained on: alpha x won + (1 - alpha) x share of acts cleared
    (with --boss1-share W, a lost run that beat the first act-3 boss at the double-boss level counts W more of an act)."""
    extra = BOSS1_SHARE[0] * r.get("b1", 0.0) / N_ACTS if not r["won"] else 0.0
    return alpha * r["won"] + (1 - alpha) * (r["progress"] + extra)


def stage_rows(ints):
    """Stage of each row (0..3, macro_s3.stage_of) from the int rows' context."""
    c0 = ms.OFF["ctx"][0]
    return ms.stage_of(ints[:, c0 + 3].astype(np.int64), ints[:, c0 + 4].astype(np.int64))


def run_b1(rec):
    """1 if the run beat the first boss of act 3: at the double-boss level a row in the state after that boss exists
    (the actor records the floor after every boss but the last), below it the first act-3 boss is the last one."""
    if rec["won"]:
        return 1.0
    if rec["asc"] < ms.DOUBLE_BOSS_ASC:
        return 0.0
    return float((stage_rows(rec["ints"]) >= 3).any())


def ms_targets(rec):
    """Milestone labels and masks [rows, N_MS] of a run's (kept) rows (cached on the run: its rows do not change after
    the window trimmed them)."""
    c = rec.get("_mst")
    if c is not None and len(c[0]) == len(rec["meta"]):
        return c
    rec["_mst"] = _ms_targets(rec)
    return rec["_mst"]


def _ms_targets(rec):
    st = stage_rows(rec["ints"])
    reached = np.array([1.0 if (rec["act"] >= 1 or rec["won"]) else 0.0, 1.0 if (rec["act"] >= 2 or rec["won"]) else 0.0,
                        rec.get("b1", 0.0), float(rec["won"])], np.float32)
    m = np.arange(ms.N_MS)[None, :]
    prev = np.concatenate([[1.0], reached[:-1]])[None, :].repeat(len(st), 0)
    mask = (m >= st[:, None]) & ((m == st[:, None]) | (prev > 0))
    if rec["asc"] < ms.DOUBLE_BOSS_ASC:
        mask[:, 3] = False
    return np.broadcast_to(reached[None, :], mask.shape).astype(np.float32), mask.astype(np.float32)


def read_chunk(path):
    with open(path, "rb") as f:
        b = f.read()
    assert b[:8] == b"S3RUNS01", path
    ver, R, N, iw, fw, rm, rw = (int(x) for x in np.frombuffer(b, "<i4", 7, 8))
    # the header's row widths: the current layout, or layout v1 / v2 (chunks of actors built before layout v3)
    assert (ver, rm, rw) == (1, RM, RW) and (iw, fw) in ms.WIDTHS.values(), (path, ver, iw, fw, rm, rw)
    off = 64
    runs = np.frombuffer(b, "<i4", R * RM, off).reshape(R, RM).copy(); off += runs.nbytes
    ints = np.frombuffer(b, "<i2", N * iw, off).reshape(N, iw).copy(); off += ints.nbytes
    floats = np.frombuffer(b, "<f4", N * fw, off).reshape(N, fw).copy(); off += floats.nbytes
    meta = np.frombuffer(b, "<f4", N * RW, off).reshape(N, RW).copy()
    ints, floats = ms.pad_rows(ints, floats)              # v1 / v2 rows: zeros in the newer columns
    return runs, ints, floats, meta


class Store:
    """The last --window-runs runs: flat row arrays plus, per run, its row span, outcome and flags."""

    def __init__(self, window_runs):
        self.W = window_runs
        self.runs = []      # dicts: ints, floats, meta, won, perturbed, floors, asc
        self.seen = set()
        self.total_runs = 0

    def scan(self, d, archive="", files=None):
        new = []
        replay = files is not None
        if files is None:
            files = sorted((p for p in glob.glob(os.path.join(d, "*.s3r")) if p not in self.seen), key=os.path.getmtime)
        for p in files:
            try:
                runs, ints, floats, meta = read_chunk(p)
            except Exception as e:
                print(f"[mtrain] skip {p}: {e}", flush=True)
                continue
            self.seen.add(p)
            for r in runs:
                a, n = int(r[9]), int(r[10])
                rec = {"ints": ints[a:a + n], "floats": floats[a:a + n], "meta": meta[a:a + n], "won": float(r[3]),
                       "perturbed": bool(r[8]), "floors": int(r[4]), "asc": int(r[2]), "act": int(r[5]),
                       "progress": 1.0 if r[3] else min(int(r[5]), N_ACTS - 1) / N_ACTS, "t_in": time.time(), "n_orig": n}
                rec["b1"] = run_b1(rec)                   # before any row is dropped from the window
                self.runs.append(rec); new.append(rec)
            if replay:
                pass
            elif archive:
                os.replace(p, os.path.join(archive, os.path.basename(p)))
            else:
                os.remove(p)
        self.total_runs += len(new)
        if len(self.runs) > self.W:
            self.runs = self.runs[-self.W:]
        return new


RUN_KEYS = ("won", "perturbed", "floors", "asc", "act", "progress", "b1")


ROWS_PER_FLOOR = (-2.59, 2.499)   # rows of a run ~ a + b x floors (segment 28: rmse 4.9 rows), for carry files without n_orig


def trim_rows(rec, k, rng, share=0.0):
    """Value-window rows of a run. share > 0: keep that share of the run's recorded rows (stochastic rounding, so every
    row of every run has the same chance and the window has the rows' natural mix). Else keep k random rows (k <= 0 or
    a short run: unchanged)."""
    n = len(rec["meta"])
    if share > 0:
        x = share * rec.get("n_orig", n)
        k = int(x) + int(rng.random() < x - int(x))
    elif k <= 0:
        return
    if n <= k:
        return
    i = np.sort(rng.choice(n, k, replace=False))
    for key in ("ints", "floats", "meta"):
        rec[key] = np.ascontiguousarray(rec[key][i])


def decay_params(net, value_only):
    """Weight matrices (ndim >= 2, not embeddings) that get weight decay; value_only: the value heads alone."""
    out = []
    for name, mod in net.named_modules():
        if value_only and not (name.startswith("win_head") or name.startswith("mono")):
            continue
        if isinstance(mod, torch.nn.Embedding):
            continue
        for pn, p in mod.named_parameters(recurse=False):
            if p.ndim >= 2:
                out.append(p)
    return out


def save_carry(path, store, opts, nets, version, why="end"):
    """Value window (rows as three flat arrays + per-run fields) and optimizer states, via path.tmp and a rename."""
    t = time.time()
    try:
        runs = store.runs
        ck = {"magic": "S3MCARRY1", "version": version, "total_runs": store.total_runs,
              "n_rows": torch.tensor([len(r["meta"]) for r in runs], dtype=torch.int64),
              "n_orig": torch.tensor([int(r.get("n_orig", len(r["meta"]))) for r in runs], dtype=torch.int64),
              **{k: torch.tensor([float(r[k]) for r in runs], dtype=torch.float64) for k in RUN_KEYS},
              **{k: torch.from_numpy(np.ascontiguousarray(cat(runs, k)) if runs else np.zeros((0, w), dt))
                 for k, w, dt in (("ints", ms.IW, np.int16), ("floats", ms.FW, np.float32), ("meta", RW, np.float32))},
              "opts": [o.state_dict() for o in opts], "shapes": [[list(p.shape) for p in n.parameters()] for n in nets]}
        with open(path + ".tmp", "wb") as f:
            torch.save(ck, f); f.flush(); os.fsync(f.fileno())   # on disk before the rename
        os.replace(path + ".tmp", path)
        try:
            fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY); os.fsync(fd); os.close(fd)
        except OSError:
            pass
        print(f"[mtrain] carry saved ({why}) to {path}: {len(runs)} runs, {int(ck['n_rows'].sum())} rows, "
              f"{os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t:.1f} s", flush=True)
    except Exception as e:
        print(f"[mtrain] carry NOT saved ({type(e).__name__}: {e})", flush=True)


def load_carry(path, store, opts, nets, lr, dev):
    if not os.path.exists(path):
        print(f"[mtrain] no carry file {path}: fresh start", flush=True)
        return
    t = time.time()
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck.get("magic") == "S3MCARRY1", "bad magic"
        n_rows = ck["n_rows"].numpy()
        ints, floats, meta = (ck[k].numpy() for k in ("ints", "floats", "meta"))
        w_file = (ints.shape[1:], floats.shape[1:])
        if w_file in (((ms.IW_V1,), (ms.FW_V1,)), ((ms.IW_V2,), (ms.FW_V2,))):   # an older carry: zeros in the new columns
            ints, floats = ms.pad_rows(ints, floats)
        assert ints.shape[1:] == (ms.IW,) and floats.shape[1:] == (ms.FW,) and meta.shape[1:] == (RW,), f"row layout differs {w_file}"
        assert len(ints) == len(floats) == len(meta) == int(n_rows.sum()), "row counts differ"
        runs, o = [], 0
        for i, n in enumerate(n_rows):
            n = int(n)
            rec = {"ints": ints[o:o + n], "floats": floats[o:o + n], "meta": meta[o:o + n]}
            for k in RUN_KEYS:
                if k not in ck:                           # files written before the key existed (b1: from the rows below)
                    continue
                v = float(ck[k][i])
                rec[k] = bool(v) if k == "perturbed" else (int(v) if k in ("floors", "asc", "act") else v)
            # rows the run had before trimming: stored from 2026-11-14 on, else estimated from its floors
            rec["n_orig"] = int(ck["n_orig"][i]) if "n_orig" in ck else max(n, int(round(ROWS_PER_FLOOR[0] + ROWS_PER_FLOOR[1] * rec["floors"])))
            if "b1" not in ck:                            # kept rows only: may miss a beaten boss whose row was dropped
                rec["b1"] = run_b1(rec)
            runs.append(rec); o += n
    except Exception as e:
        print(f"[mtrain] carry file {path} rejected ({type(e).__name__}: {e}): fresh start", flush=True)
        return
    store.runs = (runs + store.runs)[-store.W:]
    msg = f"[mtrain] carry loaded from {path}: {len(runs)} runs ({len(store.runs)} kept in the value window), {len(ints)} rows, " \
          f"{os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t:.1f} s; optimizers"
    for j, (n, op) in enumerate(zip(nets, opts)):
        try:
            shapes = [list(p.shape) for p in n.parameters()]
            if j >= len(ck["opts"]):
                raise ValueError("no state")
            old, k = ck["opts"][j], len(ck["shapes"][j])
            if ck["shapes"][j] == shapes:
                op.load_state_dict(old)                # moves the state to the parameters' device
                msg += f" {j}:restored"
            elif k < len(shapes) and ck["shapes"][j] == shapes[:k] and len(old["param_groups"]) == 1:
                # a carry written before parameters were appended to the network (2026-11-15: the milestone heads
                # ms_win / ms_mono are registered last): restore the state of the first k parameters; Adam starts the
                # appended ones fresh (lazily, as for any parameter without state)
                op.load_state_dict({"state": old["state"],
                                    "param_groups": [dict(old["param_groups"][0], params=list(range(len(shapes))))]})
                msg += f" {j}:restored({k}/{len(shapes)})"
            else:
                raise ValueError("shapes differ")
            for g in op.param_groups:
                g["lr"] = lr
        except Exception as e:
            msg += f" {j}:fresh({type(e).__name__})"
    print(msg, flush=True)


def cat(runs, key):
    return np.concatenate([r[key] for r in runs]) if runs else np.zeros((0,))


def policy_logp(net, b, meta, dev, p2_ctx=None):
    """Log-probabilities of every candidate under member `net` for decision rows: returns (logp of the chosen answer,
    entropy, full log-prob matrix over the 16 options + 64 slots with invalid entries at -inf)."""
    _, _, v, opt, slot = net.forward_rows(b, policy=True, p2_ctx=p2_ctx)
    dt = b["dtype"]
    is_slot = (dt == 4) | (dt == 5)
    bits = torch.arange(16, device=dev)
    mask = torch.cat([((meta[:, M_MASK + k].long()[:, None] >> bits) & 1) for k in range(4)], 1).bool()   # [B, 64]
    slot = slot.masked_fill(~mask, -1e9)
    opt_lp = torch.log_softmax(opt, -1); slot_lp = torch.log_softmax(slot, -1)
    lp = torch.cat([torch.where(is_slot[:, None], torch.full_like(opt_lp, -1e9), opt_lp),
                    torch.where(is_slot[:, None], slot_lp, torch.full_like(slot_lp, -1e9))], 1)          # [B, 80]
    chosen = meta[:, M_CHOSEN].long() + torch.where(is_slot, ms.K, 0)
    lp_c = lp.gather(1, chosen[:, None]).squeeze(1)
    p = lp.exp()
    ent = -(p * lp.clamp(min=-50)).sum(-1)
    return lp_c, ent, lp, v


@torch.no_grad()
def card_lookahead_targets(ens, di, dfl, lp_start, look_beta, dev, bs=2048):
    """Reconstruct one-step card-reward lookahead targets (RunAgent.cs lines 942-966) from recorded CARD_REWARD rows.
    Returns (q_look [D, K] float32 on dev, mask [D] bool on dev, agree_before float)."""
    D = len(di)
    ok_off = ms.OFF["opt_kind"][0]; oi_off = ms.OFF["opt_id"][0]; ou_off = ms.OFF["opt_upg"][0]
    d_off = ms.OFF["d_ids"][0]; du_off = ms.OFF["d_upg"][0]; de_off = ms.OFF["d_ench"][0]
    dea_off = ms.OFF["d_eamt"][0]; da_off = ms.OFF["d_age"][0]; dt_off = ms.OFF["dtype"][0]
    dtype = di[:, dt_off]
    dids = di[:, d_off:d_off + ms.MAX_DECK]
    deck_n = (dids > 0).sum(1)
    opt_kind = di[:, ok_off:ok_off + ms.K]
    has_card = (opt_kind == 1).any(1)
    eligible = np.flatnonzero((dtype == 0) & (deck_n < ms.MAX_DECK) & has_card)

    q_all = torch.zeros((D, ms.K), dtype=torch.float32, device=dev)
    mask_all = torch.zeros(D, dtype=torch.bool, device=dev)
    if len(eligible) == 0:
        return q_all, mask_all, 0.0

    # Build base + candidate post-states in compact flat arrays
    row_i_list, row_f_list, owners, slots = [], [], [], []
    for idx in eligible:
        base_i = di[idx].copy(); base_f = dfl[idx].copy()
        # clear decision fields so serve_values sees the exact StateRow() post-state
        base_i[dt_off] = -1
        base_i[ok_off:ok_off + ms.K] = 0
        base_i[oi_off:oi_off + ms.K] = 0
        base_i[ou_off:ou_off + ms.K] = 0
        row_i_list.append(base_i); row_f_list.append(base_f); owners.append(idx); slots.append(-1)
        j = int(deck_n[idx])
        for s in range(ms.K):
            if opt_kind[idx, s] != 1:
                continue
            ri = base_i.copy(); rf = base_f.copy()
            upg = int(np.clip(di[idx, ou_off + s], 0, 3))
            ri[d_off + j] = di[idx, oi_off + s]
            ri[du_off + j] = upg
            ri[de_off + j] = 0; ri[dea_off + j] = 0; ri[da_off + j] = 0
            rf[ms.S["deck_size"]] = np.float32((j + 1) / 40.0)
            if upg > 0:
                rf[ms.S["n_upgraded"]] = np.float32(rf[ms.S["n_upgraded"]] + 1.0 / 20.0)
            row_i_list.append(ri); row_f_list.append(rf); owners.append(idx); slots.append(s)

    RI = np.asarray(row_i_list, dtype=di.dtype)
    RF = np.asarray(row_f_list, dtype=dfl.dtype)
    vals = np.empty(len(RI), dtype=np.float32)
    for s in range(0, len(RI), bs):
        b = ms.rows_to_batch(RI[s:s + bs], RF[s:s + bs], dev)
        vals[s:s + bs] = ens.serve_values(b)[:, ms.O_V].float().cpu().numpy()

    dz = np.zeros((len(eligible), ms.K), dtype=np.float32)
    pos = {int(idx): k for k, idx in enumerate(eligible)}
    z0 = np.zeros(len(eligible), dtype=np.float32)
    for v, own, sl in zip(vals, owners, slots):
        k = pos[int(own)]
        if sl < 0:
            z0[k] = v
        else:
            dz[k, sl] = v - z0[k]

    el_t = torch.from_numpy(eligible).to(dev)
    opt_lp0 = lp_start[el_t, :ms.K]                                # [E, K]
    legal = torch.from_numpy(opt_kind[eligible] != 0).to(dev)
    dz_t = torch.from_numpy(dz).to(dev)
    look_logits = (opt_lp0 + look_beta * dz_t).masked_fill(~legal, -1e9)
    q_look = torch.softmax(look_logits, -1)
    agree = float((opt_lp0.argmax(-1) == look_logits.argmax(-1)).float().mean().item())
    q_all[el_t] = q_look
    mask_all[el_t] = True
    return q_all, mask_all, agree


ASC_GROUPS = (("A0-8", lambda a: a <= 8), ("A9", lambda a: a == 9), ("A10", lambda a: a >= 10))


def calibration(new, meta, z, v, v_mem):
    """Log-only: value calibration of the published ensemble on the new runs' floor rows, before training, per
    ascension group and per act within each group: rows, mean outcome, mean ensemble value (sigmoid of the members' mean
    logit: what T_f tables use), mean of the members' probabilities ("ens_p"; the difference to "ens" is the gap from
    averaging logits of disagreeing members), mean value of each member (member 0 = the policy member, whose value heads
    read its policy encoder detached) and the mean over rows of the members' logit standard deviation ("logit_sd")."""
    asc = np.concatenate([np.full(len(r["meta"]), r["asc"]) for r in new]) if new else np.zeros(0)
    fl = meta[:, M_KIND] == 0
    act = meta[:, M_ACT].astype(int)
    lg = np.log(np.clip(v_mem, 1e-7, 1 - 1e-7) / np.clip(1 - v_mem, 1e-7, 1))   # member logits (from float32 probabilities)
    out = {}
    for name, f in ASC_GROUPS:
        g = fl & np.vectorize(f, otypes=[bool])(asc) if len(asc) else fl
        for a in (None, 0, 1, 2):
            m = g if a is None else g & (act == a)
            if m.sum() == 0:
                continue
            out[name if a is None else f"{name}_act{a + 1}"] = {
                "rows": int(m.sum()), "outcome": round(float(z[m].mean()), 4), "ens": round(float(v[m].mean()), 4),
                "ens_p": round(float(v_mem[:, m].mean()), 4), "logit_sd": round(float(lg[:, m].std(0).mean()), 3),
                "members": [round(float(x), 4) for x in v_mem[:, m].mean(1)]}
    return out


def _bce(z, v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return -(z * np.log(v) + (1 - z) * np.log(1 - v))


def baselines(new, meta, z, v, fl):
    """Log-only yardsticks for the fresh-run value loss on floor rows: a constant (the batch's mean outcome), the
    9-cell table (ascension group x act, cell means of this batch: in-sample, so slightly optimistic) and the share
    of outcome variance explained by V. A value worse than the table is not using the state."""
    if not fl.any():
        return {}
    asc = np.concatenate([np.full(len(r["meta"]), r["asc"]) for r in new])[fl]
    act = meta[fl, M_ACT].astype(int)
    zf, vf = z[fl], v[fl]
    grp = np.where(asc >= 10, 2, np.where(asc == 9, 1, 0)) * 3 + np.clip(act, 0, 2)
    tab = np.zeros_like(zf)
    for g in np.unique(grp):
        tab[grp == g] = zf[grp == g].mean()
    out = {"value_logloss_const": round(float(_bce(zf, np.full_like(zf, zf.mean())).mean()), 4),
           "value_logloss_table": round(float(_bce(zf, tab).mean()), 4),
           "value_explained": round(float(1 - np.var(zf - vf) / max(np.var(zf), 1e-9)), 4)}
    for name, m in (("a10", asc >= 10), ("a0_9", asc < 10)):
        if m.any():
            out[f"value_logloss_{name}"] = round(float(_bce(zf[m], vf[m]).mean()), 4)
            out[f"value_logloss_table_{name}"] = round(float(_bce(zf[m], tab[m]).mean()), 4)
    return out


ROOM_BOSS = 5                      # ML.RoomIndex(MapPointType.Boss) in the row's ctx
HP_EDGES = (0.2, 0.4, 0.6, 0.8)    # HP-share fifths of the `hp` lists in value_cells


def _auc_in_floor(s, y, floor):
    """AUC of the score s for the 0/1 target y over pairs of rows of the same floor (None with too few pairs)."""
    num = den = 0.0
    for f in np.unique(floor):
        m = floor == f
        yy = y[m]; n1 = float(yy.sum()); n0 = float(len(yy) - n1)
        if n1 == 0 or n0 == 0:
            continue
        r = np.empty(len(yy)); r[np.argsort(s[m], kind="stable")] = np.arange(1, len(yy) + 1)
        num += r[yy].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
    return round(num / den, 4) if den >= 400 else None


def cells(new, ints, floats, meta, z, v, alpha):
    """Log-only: the published value on the fresh runs' floor rows, per ascension group x act, in units that mean the
    same at every ascension (the loss against the 9-cell table does not: where most runs of a cell end alike, the
    blended label leaves little to explain). Per cell:
      rows, z / v (mean outcome / value), room (table loss - the loss of a predictor that knows every run's outcome:
      the most any predictor can gain over the cell mean), gain (table loss - the value's loss), share (gain / room),
      auc_next / auc_win (rows of the same floor only: ranking for the cell's next milestone - act cleared, or the first
      boss of the last act - and for the run win), hp ([rows, mean outcome, mean value] per HP-share fifth),
      below_banked (acts 2-3: share of rows whose value is under the progress the run has already secured).
    <group>_after_boss1: the state after the first of two last-act bosses (rows, runs won, implied P(win) =
    (v - banked) / (1 - banked), its AUC, and [rows, won, implied P(win)] per HP fifth)."""
    if not new:
        return {}
    n_rows = [len(r["meta"]) for r in new]
    rep = lambda x: np.repeat(np.asarray(x), n_rows)
    fl = meta[:, M_KIND] == 0
    act, floor = meta[:, M_ACT].astype(int), meta[:, M_FLOOR].astype(int)
    hp = np.digitize(floats[:, ms.S["hp_frac"]], HP_EDGES)
    asc, won, run_id = rep([r["asc"] for r in new]), rep([r["won"] > 0 for r in new]), rep(np.arange(len(new)))
    cleared = rep([N_ACTS if r["won"] else min(r["act"], N_ACTS - 1) for r in new])
    after_b1 = fl & (act == N_ACTS - 1) & (ints[:, ms.OFF["ctx"][0] + 4] == ROOM_BOSS)
    b1_run = np.zeros(len(new), bool); b1_run[np.unique(run_id[after_b1])] = True
    nxt = np.where(act == 0, cleared >= 1, np.where(act == 1, cleared >= 2, won | b1_run[run_id]))
    z, v = z.astype(np.float64), v.astype(np.float64)
    f4 = lambda x: round(float(x), 4)
    out = {}
    for name, f in ASC_GROUPS:
        g = fl & np.vectorize(f, otypes=[bool])(asc)
        for a in range(N_ACTS):
            m = g & (act == a) & ~after_b1
            if m.sum() < 200:
                continue
            zz, vv, hh = z[m], v[m], hp[m]
            ent, tab, loss = _bce(zz, zz).mean(), _bce(zz, np.full_like(zz, zz.mean())).mean(), _bce(zz, vv).mean()
            c = {"rows": int(m.sum()), "z": f4(zz.mean()), "v": f4(vv.mean()), "room": f4(tab - ent), "gain": f4(tab - loss),
                 "share": round(float((tab - loss) / max(tab - ent, 1e-6)), 3),
                 "auc_next": _auc_in_floor(vv, nxt[m], floor[m]), "auc_win": _auc_in_floor(vv, won[m], floor[m]),
                 "hp": [[int((hh == k).sum()), f4(zz[hh == k].mean()), f4(vv[hh == k].mean())] if (hh == k).any() else [0, None, None]
                        for k in range(len(HP_EDGES) + 1)]}
            if a > 0:
                c["below_banked"] = f4((vv < (1 - alpha) * a / N_ACTS).mean())
            out[f"{name}_act{a + 1}"] = c
        m = g & after_b1
        if m.sum() >= 50:
            banked = (1 - alpha) * (N_ACTS - 1) / N_ACTS
            p, w, hh = (v[m] - banked) / (1 - banked), won[m], hp[m]
            out[f"{name}_after_boss1"] = {
                "rows": int(m.sum()), "won": f4(w.mean()), "p_win": f4(p.mean()), "auc_win": _auc_in_floor(p, w, np.zeros(len(p), int)),
                "hp": [[int((hh == k).sum()), f4(w[hh == k].mean()), f4(p[hh == k].mean())] if (hh == k).any() else [0, None, None]
                       for k in range(len(HP_EDGES) + 1)]}
    return out


def gae(values, won, lam):
    """Advantages and lambda-returns for one run's decision values (gamma 1, reward `won` at the end)."""
    n = len(values)
    adv = np.zeros(n, np.float32)
    nxt, run = won, 0.0
    for t in range(n - 1, -1, -1):
        delta = nxt - values[t]
        run = delta + lam * run
        adv[t] = run
        nxt = values[t]
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdata", required=True); ap.add_argument("--macro", required=True)
    ap.add_argument("--init", default=os.path.join(HERE, "..", "data", "macro_s1.pt"))
    ap.add_argument("--bc", default=os.path.join(HERE, "..", "data", "macro_s1.pt"),
                    help="behaviour-cloning reference of the KL penalty (stays the S1 policy when --init is a later segment's file)")
    ap.add_argument("--runs-offset", type=int, default=0, help="runs already trained on in earlier segments (KL schedule, logs)")
    ap.add_argument("--target-asc", type=int, default=10, help="ascension whose win rate sets alpha and is logged separately")
    ap.add_argument("--ckpt-dir", default=""); ap.add_argument("--archive-dir", default="")
    ap.add_argument("--runs-per-iter", type=int, default=256)
    ap.add_argument("--window-runs", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=8192, help="chunk size for no-grad value / calibration / lp_start forwards (lower reduces peak GPU VRAM)")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--kl0", type=float, default=0.5, help="initial weight of the KL penalty toward the S1 policy")
    ap.add_argument("--kl-runs", type=int, default=75_000, help="runs after which the KL weight is zero (a quarter of the budget)")
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--value-reuse", type=float, default=4.0, help="value samples per new row per iteration")
    ap.add_argument("--value-only", action="store_true", help="train V_run only; the policy stays as initialised")
    ap.add_argument("--alpha", type=float, default=-1.0, help="weight of the win in the outcome (the rest is the share of acts "
                    "cleared); -1 = automatic: window win rate / --alpha-full, at most 1")
    ap.add_argument("--alpha-full", type=float, default=0.2, help="window win rate at which the outcome is the win alone")
    ap.add_argument("--adv-floor", type=float, default=0.02, help="advantages are divided by max(their std, this)")
    ap.add_argument("--no-perturbed-policy", action="store_true")
    ap.add_argument("--keep-every", type=int, default=10); ap.add_argument("--max-iters", type=int, default=10**9)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log", default="")
    ap.add_argument("--save-carry", default="", help="on SIGTERM/SIGINT and at the end write the value window and the "
                    "optimizer states to this file (temp file + rename)")
    ap.add_argument("--carry-every", type=float, default=0.0, help="also write the --save-carry file every this many "
                    "seconds of wall time, between iterations (0 = only on SIGTERM/SIGINT and at the end)")
    ap.add_argument("--load-carry", default="", help="start with the value window and optimizer states of this carry file "
                    "if it exists and loads (else a log line and a fresh start)")
    ap.add_argument("--value-rows-per-run", type=int, default=0, help="rows of each run kept in the value window (0 = all)")
    ap.add_argument("--value-steps", type=int, default=0, help="value steps per member per iteration (0 = --value-reuse rule)")
    ap.add_argument("--value-dropout", action="store_true", help="members 1-4 take value steps in training mode (dropout)")
    ap.add_argument("--value-wd", type=float, default=0.0, help="decoupled weight decay of the value steps")
    ap.add_argument("--prob-avg", action="store_true", help="ensemble value = mean of member probabilities (advantages, logs)")
    ap.add_argument("--policy-warmup-s", type=float, default=0.0, help="runs ingested in the first S seconds: value only")
    ap.add_argument("--replay", default="", help="offline: file listing *.s3r chunks to read in order (see the docstring)")
    ap.add_argument("--value-row-share", type=float, default=0.0, help="share of each run's rows kept in the value window "
                    "(takes the place of --value-rows-per-run; 0 = off)")
    ap.add_argument("--value-row-weights", action="store_true", help="value batches draw window rows in proportion to "
                    "rows of the run / rows kept (with --value-rows-per-run)")
    ap.add_argument("--milestones", action="store_true", help="train the milestone heads; their blend becomes the value")
    ap.add_argument("--ms-warmup-iters", type=int, default=3, help="iterations of milestone training before the blend "
                    "is used (0, or an --init file with trained heads: at once)")
    ap.add_argument("--boss1-share", type=float, default=0.0, help="share of the last act earned by beating the first "
                    "act-3 boss at the double-boss level (0 = D14 as it is)")
    ap.add_argument("--freeze-v3", action="store_true", help="keep the layout v3 zero layers at zero (no updates; control arm)")
    ap.add_argument("--p2", action="store_true", help="opt-in P2: attach zero-init residual adapter on member 0 reading "
                    "stopgrad(EMA(mean members 1-4)) representations (REPR_TEST_PLAN.md)")
    ap.add_argument("--p2-ema", type=float, default=0.95, help="EMA decay of members 1-4 for P2 (0 = live members 1-4)")
    ap.add_argument("--look-distill", type=float, default=0.0, help="opt-in M3: distillation weight toward one-step "
                    "lookahead distribution on CARD_REWARD rows (0 = off)")
    ap.add_argument("--look-distill-beta", type=float, default=20.0, help="lookahead beta for --look-distill")
    ap.add_argument("--look-distill-replace", action="store_true", help="exclude lookahead CARD_REWARD rows from the "
                    "clipped PPO surrogate so PPO runs purely on non-lookahead decisions")
    a = ap.parse_args()
    BOSS1_SHARE[0] = a.boss1_share
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ens = ms.Ensemble.load(a.init, dev)
    if a.p2:
        ens.enable_p2(use_ema=(a.p2_ema > 0))
    bc = ms.Ensemble.load(a.bc, dev).nets[0]              # frozen behaviour-cloning reference
    for p in bc.parameters():
        p.requires_grad_(False)
    nets = ens.nets
    for n in nets:
        n.eval()                                          # no dropout: PPO ratios need a deterministic policy
        if a.value_dropout:
            n.era_dropout = 0.0                           # S3 rows are one game version: no "unknown era" token
    if a.freeze_v3:
        for n in nets:
            with torch.no_grad():
                for p in n.v3_zero_params():
                    p.zero_(); p.requires_grad_(False)
    opts = [torch.optim.Adam(n.parameters(), lr=a.lr) for n in nets]
    ens.save(a.macro)
    log = open(a.log or os.path.join(os.path.dirname(os.path.abspath(a.macro)), "macro_train_log.jsonl"), "a")
    for d in (a.ckpt_dir, a.archive_dir):
        if d:
            os.makedirs(d, exist_ok=True)
    print(f"[mtrain] {len(nets)} members on {dev}; wrote version {ens.version} to {a.macro}", flush=True)
    store = Store(a.window_runs)
    store.total_runs = a.runs_offset
    if a.load_carry:
        load_carry(a.load_carry, store, opts, nets, a.lr, dev)
        for r in store.runs:
            trim_rows(r, a.value_rows_per_run, rng, a.value_row_share)
    pending, it, t0 = [], 0, time.time()

    class Stop(BaseException):   # not an Exception: a periodic carry save's error handler must not swallow it
        pass

    def on_signal(sig, frame):   # an iteration in progress is abandoned (not published); its runs are in the window
        raise Stop
    if a.save_carry:
        signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)
    try:
        loop(a, store, nets, opts, ens, bc, log, rng, dev, pending, it, t0)
    except Stop:
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)   # save undisturbed
    if a.save_carry:
        save_carry(a.save_carry, store, opts, nets, ens.version)


def loop(a, store, nets, opts, ens, bc, log, rng, dev, pending, it, t0):

    def batch_of(ints, floats, idx):
        return ms.rows_to_batch(ints[idx], floats[idx], dev)

    @torch.no_grad()
    def ens_value(ints, floats, members=False, alpha=0.0):
        """Ensemble value (sigmoid of the members' mean logit) per row; with members=True also every member's own
        value [n_members, rows] (log-only diagnostics; the ensemble value is computed exactly as without it).
        Once the milestone heads are in use (msst["on"]), each member's value is the blend of its heads."""
        out, mem = [], []
        eb = a.eval_batch
        for i in range(0, len(ints), eb):
            b = ms.rows_to_batch(ints[i:i + eb], floats[i:i + eb], dev)
            if msst["on"]:
                st = ms.stage_of(b["ctx"][:, 3], b["ctx"][:, 4]); dbl = b["ctx"][:, 1] >= ms.DOUBLE_BOSS_ASC
                L = torch.stack([torch.logit(ms.blend(torch.sigmoid(n.forward_ms(b)[2]), st, dbl, alpha, a.boss1_share).clamp(1e-6, 1 - 1e-6))
                                 for n in nets])
            else:
                L = torch.stack([n.forward_rows(b, policy=False)[2] for n in nets])
            out.append(torch.logit(torch.sigmoid(L).mean(0).clamp(1e-6, 1 - 1e-6)) if a.prob_avg else L.mean(0))
            if members:
                mem.append(torch.sigmoid(L).cpu())
        v = torch.sigmoid(torch.cat(out)).cpu().numpy() if out else np.zeros(0, np.float32)
        if members:
            return v, (torch.cat(mem, 1).numpy() if mem else np.zeros((len(nets), 0), np.float32))
        return v

    @torch.no_grad()
    def ms_calibration(new, ni, nf):
        """Log-only: per ascension group and head, on the fresh runs' rows before training: rows that train the head,
        mean label, mean of the members' probabilities (heads untrained before the first --milestones iteration)."""
        if not new:
            return {}
        lab = np.concatenate([ms_targets(r)[0] for r in new]); msk = np.concatenate([ms_targets(r)[1] for r in new]) > 0
        asc = np.concatenate([np.full(len(r["meta"]), r["asc"]) for r in new])
        P = []
        eb = a.eval_batch
        for i in range(0, len(ni), eb):
            b = ms.rows_to_batch(ni[i:i + eb], nf[i:i + eb], dev)
            P.append(torch.stack([torch.sigmoid(n.forward_ms(b)[2]) for n in nets]).mean(0).cpu().numpy())
        P = np.concatenate(P)
        out = {}
        for name, f in ASC_GROUPS:
            g = np.vectorize(f, otypes=[bool])(asc)
            for m, hn in enumerate(("act1", "act2", "boss1", "boss2")):
                k = g & msk[:, m]
                if k.sum() >= 50:
                    out[f"{name}_{hn}"] = [int(k.sum()), round(float(lab[k, m].mean()), 4), round(float(P[k, m].mean()), 4),
                                           round(float(_bce(lab[k, m], P[k, m]).mean()), 4)]
        return out

    msst = {"on": a.milestones and (ens.milestones or a.ms_warmup_iters <= 0), "iters": 0}
    if not a.milestones:
        ens.milestones = False                            # heads not trained here: never serve their blend
    rp_files = [l.strip() for l in open(a.replay) if l.strip()] if a.replay else None
    t_start = time.time()
    dparams = [decay_params(n, j == 0) for j, n in enumerate(nets)] if a.value_wd > 0 else None
    last_carry, carried_at = time.time(), None
    while it < a.max_iters:
        if a.save_carry and a.carry_every > 0 and time.time() - last_carry >= a.carry_every:
            if (store.total_runs, ens.version) != carried_at:   # nothing new since the last periodic save: skip it
                save_carry(a.save_carry, store, opts, nets, ens.version, why="periodic")
                carried_at = (store.total_runs, ens.version)
            last_carry = time.time()
        if rp_files is not None:
            while len(pending) < a.runs_per_iter and rp_files:
                pending += store.scan(None, files=[rp_files.pop(0)])
            if len(pending) < a.runs_per_iter:
                print(f"[mtrain] replay list used up after {it} iterations", flush=True)
                break
        else:
            pending += store.scan(a.mdata, a.archive_dir)
        if len(pending) < a.runs_per_iter:
            time.sleep(2.0)
            continue
        new, pending = pending, []
        it += 1
        tgt = [r for r in store.runs if r["asc"] == a.target_asc and not r["perturbed"]] or store.runs
        wr = float(np.mean([r["won"] for r in tgt]))     # win rate at the target ascension sets alpha
        alpha = a.alpha if a.alpha >= 0 else min(1.0, wr / a.alpha_full)
        info = {"t": round(time.time() - t0, 1), "iter": it, "runs": store.total_runs, "new_runs": len(new),
                "win_rate": round(float(np.mean([r["won"] for r in new])), 4), "mean_floors": round(float(np.mean([r["floors"] for r in new])), 2),
                "ascension": round(float(np.mean([r["asc"] for r in new])), 2),
                "acts_cleared_2plus": round(float(np.mean([r["act"] >= 2 for r in new])), 4),
                "act1_cleared": round(float(np.mean([r["act"] >= 1 for r in new])), 4)}
        by_asc = {}
        for r in new:
            if not r["perturbed"]:
                by_asc.setdefault(r["asc"], []).append(r)
        info["by_ascension"] = {str(k): {"runs": len(v), "won": round(float(np.mean([r["won"] for r in v])), 4),
                                         "acts_cleared": round(float(np.mean([3.0 if r["won"] else min(r["act"], 2) for r in v])), 3),
                                         "floors": round(float(np.mean([r["floors"] for r in v])), 1)} for k, v in sorted(by_asc.items())}
        info["target_window_win_rate"] = round(wr, 4)
        # ---- value drift on the new runs, before training (calibration of the published V_run on the agent's own states)
        ni, nf, nm = cat(new, "ints"), cat(new, "floats"), cat(new, "meta")
        z_new = np.concatenate([np.full(len(r["meta"]), outcome(r, alpha), np.float32) for r in new])
        info["alpha"] = round(alpha, 4); info["mean_outcome"] = round(float(np.mean([outcome(r, alpha) for r in new])), 4)
        v_new, v_mem = ens_value(ni, nf, members=True, alpha=alpha)
        info["ms_on"] = msst["on"]
        if a.milestones:
            info["ms_calibration"] = ms_calibration(new, ni, nf)
        fl = nm[:, M_KIND] == 0
        info["value_mean_floor_rows"] = round(float(v_new[fl].mean()), 4); info["outcome_mean_floor_rows"] = round(float(z_new[fl].mean()), 4)
        info["calibration"] = calibration(new, nm, z_new, v_new, v_mem)
        info["value_logloss"] = round(float(-(z_new[fl] * np.log(v_new[fl].clip(1e-6)) + (1 - z_new[fl]) * np.log((1 - v_new[fl]).clip(1e-6))).mean()), 4)
        info.update(baselines(new, nm, z_new, v_new, fl))
        info["value_cells"] = cells(new, ni, nf, nm, z_new, v_new, alpha)
        # ---- policy: PPO on the new runs' decisions
        di = []
        if not a.value_only:
            pol_runs = {id(r) for r in new if not (a.no_perturbed_policy and r["perturbed"])
                        and r["t_in"] - t_start >= a.policy_warmup_s}
            info["policy_runs"] = len(pol_runs)
            di, dfl, dm, adv = [], [], [], []
            o = 0
            for r in new:
                n = len(r["meta"]); dec = r["meta"][:, M_KIND] == 1
                if id(r) in pol_runs and dec.any():
                    adv.append(gae(v_new[o:o + n][dec], outcome(r, alpha), a.lam))
                    di.append(r["ints"][dec]); dfl.append(r["floats"][dec]); dm.append(r["meta"][dec])
                o += n
        if di:
            di, dfl, dm, adv = np.concatenate(di), np.concatenate(dfl), np.concatenate(dm), np.concatenate(adv)
            adv_n = (adv - adv.mean()) / max(float(adv.std()), a.adv_floor)   # no signal must not be scaled up to noise
            beta = a.kl0 * max(0.0, 1.0 - store.total_runs / max(1, a.kl_runs))
            D = len(dm); stats = {"pol": 0.0, "ent": 0.0, "kl_bc": 0.0, "kl_step": 0.0, "clip": 0.0}; steps = 0
            stop = False
            # the trust region is measured from the policy at the start of this iteration; the recorded behaviour
            # log-probabilities (from weights up to a few versions old) are used for the importance ratios only
            with torch.no_grad():
                parts = []
                eb = a.eval_batch
                for s in range(0, D, eb):
                    bs_b = batch_of(di, dfl, np.arange(s, min(D, s + eb)))
                    p2c = ens.p2_context(bs_b) if ens.p2 else None
                    parts.append(policy_logp(nets[0], bs_b, torch.from_numpy(dm[s:s + eb]).to(dev), dev, p2_ctx=p2c))
                lp_start = torch.cat([x[2] for x in parts])            # [D, 80] log-probabilities at the iteration start
                lp_c0 = torch.cat([x[0] for x in parts])
                old0 = torch.from_numpy(dm[:, M_LOGP]).to(dev)
                info["behaviour_lag_kl"] = round(float((old0 - lp_c0).mean()), 5)
                non_card_m = torch.from_numpy(dm[:, M_DTYPE].astype(np.int64) != 0).to(dev)
                if non_card_m.any():
                    info["behaviour_lag_kl_noncard"] = round(float((old0[non_card_m] - lp_c0[non_card_m]).mean()), 5)
            q_look_all = look_mask_all = None
            if a.look_distill > 0:
                ens.milestones = msst["on"]; ens.alpha = float(alpha); ens.boss1_share = float(a.boss1_share)
                q_look_all, look_mask_all, agree0 = card_lookahead_targets(ens, di, dfl, lp_start, a.look_distill_beta, dev, bs=a.eval_batch)
                info["look_agree_before"] = round(agree0, 4)
                stats["distill"] = 0.0; stats["clip_noncard"] = 0.0
            for ep in range(a.epochs):
                perm = rng.permutation(D)
                for s in range(0, D, a.batch):
                    idx = perm[s:s + a.batch]
                    b = batch_of(di, dfl, idx)
                    m = torch.from_numpy(dm[idx]).to(dev)
                    A = torch.from_numpy(adv_n[idx]).to(dev)
                    p2c = ens.p2_context(b) if ens.p2 else None
                    lp_c, ent, lp, _ = policy_logp(nets[0], b, m, dev, p2_ctx=p2c)
                    old = m[:, M_LOGP]
                    ratio = (lp_c - old).exp()
                    ppo_elem = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A)
                    if a.look_distill > 0:
                        idx_t = torch.from_numpy(idx).to(dev)
                        lm = look_mask_all[idx_t]
                        ql = q_look_all[idx_t]
                        # KL(q_look || pi_theta) on eligible CARD_REWARD rows
                        opt_lp = lp[:, :ms.K].clamp(min=-50)
                        ql_ent = -(ql * torch.log(ql.clamp(min=1e-12))).sum(-1)
                        kl_d = -(ql * opt_lp).sum(-1) - ql_ent
                        distill_loss = kl_d[lm].mean() if lm.any() else lp.sum() * 0
                        if a.look_distill_replace:
                            pol = ppo_elem[~lm].mean() if (~lm).any() else lp.sum() * 0
                            ent_term = ent[~lm].mean() if (~lm).any() else lp.sum() * 0
                        else:
                            pol = ppo_elem.mean()
                            ent_term = ent.mean()
                    else:
                        pol = ppo_elem.mean()
                        ent_term = ent.mean()
                        distill_loss = None
                    with torch.no_grad():
                        _, _, lp_bc, _ = policy_logp(bc, b, m, dev)
                    is_bc = torch.isin(b["dtype"], torch.tensor(BC_TYPES, device=dev))
                    kl_bc = ((lp.exp() * (lp.clamp(min=-50) - lp_bc.clamp(min=-50))).sum(-1) * is_bc).sum() / is_bc.sum().clamp(min=1)
                    loss = pol - a.ent * ent_term + beta * kl_bc
                    if distill_loss is not None:
                        loss = loss + a.look_distill * distill_loss
                    opts[0].zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(nets[0].parameters(), 1.0); opts[0].step()
                    with torch.no_grad():
                        ls = lp_start[torch.from_numpy(idx).to(dev)]
                        kl_step = (ls.exp() * (ls.clamp(min=-50) - lp.clamp(min=-50))).sum(-1).mean().item()   # KL(start || now)
                        stats["pol"] += pol.item(); stats["ent"] += ent.mean().item(); stats["kl_bc"] += kl_bc.item()
                        stats["kl_step"] += kl_step; stats["clip"] += ((ratio - 1).abs() > a.clip).float().mean().item(); steps += 1
                        if distill_loss is not None:
                            stats["distill"] += distill_loss.item()
                            nc = ~lm
                            stats["clip_noncard"] += (((ratio[nc] - 1).abs() > a.clip).float().mean().item() if nc.any() else 0.0)
                    if kl_step > 1.5 * a.target_kl:
                        stop = True; break
                if stop:
                    break
            info.update({"decisions": int(D), "policy_steps": steps, "kl_weight": round(beta, 4),
                         **{k: round(v / max(1, steps), 5) for k, v in stats.items()},
                         "adv_std": round(float(adv.std()), 4), "early_stop": stop})
            dt = dm[:, M_DTYPE].astype(int)
            info["decisions_by_type"] = {ms.DT[t]: int((dt == t).sum()) for t in np.unique(dt)}
        # ---- value: all members, on the window
        for r in new:                                     # the window keeps a part of each run's rows (see the docstring)
            trim_rows(r, a.value_rows_per_run, rng, a.value_row_share)
        win = store.runs
        wi, wf = cat(win, "ints"), cat(win, "floats")
        wz = np.concatenate([np.full(len(r["meta"]), outcome(r, alpha), np.float32) for r in win])
        # --value-row-weights: a kept row stands for (rows of its run / rows kept) recorded rows
        cdf = np.cumsum(np.concatenate([np.full(len(r["meta"]), r.get("n_orig", len(r["meta"])) / max(1, len(r["meta"])), np.float64)
                                        for r in win])) if a.value_row_weights else None
        if a.milestones:
            T = [ms_targets(r) for r in win]
            wl, wm = np.concatenate([x[0] for x in T]), np.concatenate([x[1] for x in T])
        vl_ms = 0.0
        n_steps = a.value_steps if a.value_steps > 0 else max(1, int(math.ceil(a.value_reuse * len(ni) / a.batch)))
        vl = 0.0
        for j, (n, op) in enumerate(zip(nets, opts)):
            if a.value_dropout and j > 0:
                n.train()
            for _ in range(n_steps):
                idx = rng.integers(0, len(wz), a.batch) if cdf is None else \
                    np.minimum(np.searchsorted(cdf, rng.random(a.batch) * cdf[-1], side="right"), len(wz) - 1)
                b = batch_of(wi, wf, idx)
                z = torch.from_numpy(wz[idx]).to(dev)
                # the policy member (0) trains only its value heads here: a value gradient into the encoder it shares with
                # the policy moved the policy outside PPO's trust region (behaviour-lag KL 0.41 after four updates in the
                # first server run). Members 1-4 have no policy in use and train fully.
                if a.milestones:
                    zz, phi, _ = n.encode(b)
                    if j == 0:
                        zz = zz.detach()
                    logit = n.win_head(zz).squeeze(-1).float() + (F.softplus(n.mono(zz).float()) * phi).sum(-1)
                    lm = n.ms_from(zz, phi)[2]
                    ml = torch.from_numpy(wl[idx]).to(dev); mm = torch.from_numpy(wm[idx]).to(dev)
                    loss_ms = (F.binary_cross_entropy_with_logits(lm, ml, reduction="none") * mm).sum() / mm.sum().clamp(min=1)
                    vl_ms += loss_ms.item()
                else:
                    logit = n.forward_rows(b, policy=False, detach=(j == 0))[2]
                loss = F.binary_cross_entropy_with_logits(logit, z)
                if a.milestones:
                    loss = loss + loss_ms
                op.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0); op.step()
                if dparams is not None:
                    with torch.no_grad():
                        torch._foreach_mul_(dparams[j], 1.0 - op.param_groups[0]["lr"] * a.value_wd)
                vl += loss.item() - (loss_ms.item() if a.milestones else 0.0)
            n.eval()
        info["value_loss"] = round(vl / (n_steps * len(nets)), 5); info["value_steps"] = n_steps; info["window_runs"] = len(win)
        info["window_rows"] = int(len(wz))
        if ens.p2:
            # Measure pre-PPO lag KL caused by members 1-4 moving during the value steps (both live and EMA)
            if len(di) > 0:
                sub = np.arange(min(len(di), a.eval_batch))
                b_chk = batch_of(di, dfl, sub); m_chk = torch.from_numpy(dm[sub]).to(dev)
                with torch.no_grad():
                    lp_pre = policy_logp(nets[0], b_chk, m_chk, dev, p2_ctx=ens.p2_context(b_chk))[2]
                    lp_live = policy_logp(nets[0], b_chk, m_chk, dev, p2_ctx=ens.p2_context(b_chk, src_nets=nets[1:]))[2]
                    info["p2_lag_kl_live"] = round(float((lp_pre.exp() * (lp_pre.clamp(min=-50) - lp_live.clamp(min=-50))).sum(-1).mean()), 6)
            if ens.p2_ema_nets is not None:
                with torch.no_grad():
                    for ne, nl in zip(ens.p2_ema_nets, nets[1:]):
                        for pe, pl in zip(ne.parameters(), nl.parameters()):
                            pe.lerp_(pl, 1.0 - a.p2_ema)
                if len(di) > 0:
                    with torch.no_grad():
                        lp_ema = policy_logp(nets[0], b_chk, m_chk, dev, p2_ctx=ens.p2_context(b_chk))[2]
                        info["p2_lag_kl_ema"] = round(float((lp_pre.exp() * (lp_pre.clamp(min=-50) - lp_ema.clamp(min=-50))).sum(-1).mean()), 6)
        if a.milestones:
            info["ms_loss"] = round(vl_ms / (n_steps * len(nets)), 5)
            msst["iters"] += 1
            if not msst["on"] and msst["iters"] >= a.ms_warmup_iters:
                msst["on"] = True
            ens.milestones = msst["on"]; ens.alpha = float(alpha); ens.boss1_share = float(a.boss1_share)
        info["window_win_rate"] = round(float(np.mean([r["won"] for r in win])), 4)
        if not all(torch.isfinite(p).all() for n in nets for p in n.parameters()):
            # never publish broken weights: go back to the last published file and fresh optimizers
            print("[mtrain] non-finite weights after this iteration: restored the last published ensemble", flush=True)
            ens.load_members_from(a.macro)
            opts[:] = [torch.optim.Adam(n.parameters(), lr=a.lr) for n in nets]
            info["restored"] = True
        ens.version += 1
        ens.save(a.macro)
        if a.ckpt_dir and (ens.version % a.keep_every == 0):
            ens.save(os.path.join(a.ckpt_dir, f"macro_v{ens.version:05d}.pt"))
        info["version"] = ens.version
        log.write(json.dumps(info) + "\n"); log.flush()
        print("[mtrain] " + json.dumps(info), flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
