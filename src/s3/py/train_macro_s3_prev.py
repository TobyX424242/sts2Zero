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

Value-fit options (2026-11-13, IMPROVEMENT_PLAN.md change A; all off by default = the old behaviour):
  --value-rows-per-run K  the value window keeps K random rows of each run (PPO still uses every decision of the new
                 runs): ~95 rows of one run share one label, so whole runs in the window let the members memorise runs
  --value-steps N  value steps per member per iteration (default 0: --value-reuse x new rows / batch)
  --value-dropout  members 1-4 take their value steps in training mode (dropout; era dropout off)
  --value-wd W     decoupled weight decay W x lr per value step on weight matrices (member 0: value heads only)
  --prob-avg       advantages and logged ensemble value from the mean of member probabilities, not of logits
  --policy-warmup-s S  runs ingested in the first S seconds of this process train the value only (a segment's first
                 runs are its shortest: centring their advantages within that batch biases the policy)
Offline replay (--replay LIST): read the *.s3r files named in LIST (one path per line, in that order) instead of
watching --mdata; files are not moved or deleted; the trainer exits when the list is used up. With --value-only this
replays a stored run through the value trainer (the logged fresh-run loss is measured before each update).

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


def outcome(r, alpha):
    """The run outcome the value and the policy are trained on: alpha x won + (1 - alpha) x share of acts cleared."""
    return alpha * r["won"] + (1 - alpha) * r["progress"]


def read_chunk(path):
    with open(path, "rb") as f:
        b = f.read()
    assert b[:8] == b"S3RUNS01", path
    ver, R, N, iw, fw, rm, rw = np.frombuffer(b, "<i4", 7, 8)
    assert (ver, iw, fw, rm, rw) == (1, ms.IW, ms.FW, RM, RW), (path, ver, iw, fw, rm, rw)
    off = 64
    runs = np.frombuffer(b, "<i4", R * RM, off).reshape(R, RM).copy(); off += runs.nbytes
    ints = np.frombuffer(b, "<i2", N * ms.IW, off).reshape(N, ms.IW).copy(); off += ints.nbytes
    floats = np.frombuffer(b, "<f4", N * ms.FW, off).reshape(N, ms.FW).copy(); off += floats.nbytes
    meta = np.frombuffer(b, "<f4", N * RW, off).reshape(N, RW).copy()
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
                       "progress": 1.0 if r[3] else min(int(r[5]), N_ACTS - 1) / N_ACTS, "t_in": time.time()}
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


RUN_KEYS = ("won", "perturbed", "floors", "asc", "act", "progress")


def trim_rows(rec, k, rng):
    """Keep k random rows of a run for the value window (k <= 0 or a short run: unchanged)."""
    n = len(rec["meta"])
    if k <= 0 or n <= k:
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


def save_carry(path, store, opts, nets, version):
    """Value window (rows as three flat arrays + per-run fields) and optimizer states, via path.tmp and a rename."""
    t = time.time()
    try:
        runs = store.runs
        ck = {"magic": "S3MCARRY1", "version": version, "total_runs": store.total_runs,
              "n_rows": torch.tensor([len(r["meta"]) for r in runs], dtype=torch.int64),
              **{k: torch.tensor([float(r[k]) for r in runs], dtype=torch.float64) for k in RUN_KEYS},
              **{k: torch.from_numpy(np.ascontiguousarray(cat(runs, k)) if runs else np.zeros((0, w), dt))
                 for k, w, dt in (("ints", ms.IW, np.int16), ("floats", ms.FW, np.float32), ("meta", RW, np.float32))},
              "opts": [o.state_dict() for o in opts], "shapes": [[list(p.shape) for p in n.parameters()] for n in nets]}
        torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)
        print(f"[mtrain] carry saved to {path}: {len(runs)} runs, {int(ck['n_rows'].sum())} rows, "
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
        assert ints.shape[1:] == (ms.IW,) and floats.shape[1:] == (ms.FW,) and meta.shape[1:] == (RW,), "row layout differs"
        assert len(ints) == len(floats) == len(meta) == int(n_rows.sum()), "row counts differ"
        runs, o = [], 0
        for i, n in enumerate(n_rows):
            n = int(n)
            rec = {"ints": ints[o:o + n], "floats": floats[o:o + n], "meta": meta[o:o + n]}
            for k in RUN_KEYS:
                v = float(ck[k][i])
                rec[k] = bool(v) if k == "perturbed" else (int(v) if k in ("floors", "asc", "act") else v)
            runs.append(rec); o += n
    except Exception as e:
        print(f"[mtrain] carry file {path} rejected ({type(e).__name__}: {e}): fresh start", flush=True)
        return
    store.runs = (runs + store.runs)[-store.W:]
    msg = f"[mtrain] carry loaded from {path}: {len(runs)} runs ({len(store.runs)} kept in the value window), {len(ints)} rows, " \
          f"{os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t:.1f} s; optimizers"
    for j, (n, op) in enumerate(zip(nets, opts)):
        try:
            if j >= len(ck["opts"]) or ck["shapes"][j] != [list(p.shape) for p in n.parameters()]:
                raise ValueError("shapes differ")
            op.load_state_dict(ck["opts"][j])          # moves the state to the parameters' device
            for g in op.param_groups:
                g["lr"] = lr
            msg += f" {j}:restored"
        except Exception as e:
            msg += f" {j}:fresh({type(e).__name__})"
    print(msg, flush=True)


def cat(runs, key):
    return np.concatenate([r[key] for r in runs]) if runs else np.zeros((0,))


def policy_logp(net, b, meta, dev):
    """Log-probabilities of every candidate under member `net` for decision rows: returns (logp of the chosen answer,
    entropy, full log-prob matrix over the 16 options + 64 slots with invalid entries at -inf)."""
    _, _, v, opt, slot = net.forward_rows(b, policy=True)
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
    ap.add_argument("--load-carry", default="", help="start with the value window and optimizer states of this carry file "
                    "if it exists and loads (else a log line and a fresh start)")
    ap.add_argument("--value-rows-per-run", type=int, default=0, help="rows of each run kept in the value window (0 = all)")
    ap.add_argument("--value-steps", type=int, default=0, help="value steps per member per iteration (0 = --value-reuse rule)")
    ap.add_argument("--value-dropout", action="store_true", help="members 1-4 take value steps in training mode (dropout)")
    ap.add_argument("--value-wd", type=float, default=0.0, help="decoupled weight decay of the value steps")
    ap.add_argument("--prob-avg", action="store_true", help="ensemble value = mean of member probabilities (advantages, logs)")
    ap.add_argument("--policy-warmup-s", type=float, default=0.0, help="runs ingested in the first S seconds: value only")
    ap.add_argument("--replay", default="", help="offline: file listing *.s3r chunks to read in order (see the docstring)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ens = ms.Ensemble.load(a.init, dev)
    bc = ms.Ensemble.load(a.bc, dev).nets[0]              # frozen behaviour-cloning reference
    for p in bc.parameters():
        p.requires_grad_(False)
    nets = ens.nets
    for n in nets:
        n.eval()                                          # no dropout: PPO ratios need a deterministic policy
        if a.value_dropout:
            n.era_dropout = 0.0                           # S3 rows are one game version: no "unknown era" token
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
            trim_rows(r, a.value_rows_per_run, rng)
    pending, it, t0 = [], 0, time.time()

    class Stop(Exception):
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
    def ens_value(ints, floats, members=False):
        """Ensemble value (sigmoid of the members' mean logit) per row; with members=True also every member's own
        value [n_members, rows] (log-only diagnostics; the ensemble value is computed exactly as without it)."""
        out, mem = [], []
        for i in range(0, len(ints), 8192):
            b = ms.rows_to_batch(ints[i:i + 8192], floats[i:i + 8192], dev)
            L = torch.stack([n.forward_rows(b, policy=False)[2] for n in nets])
            out.append(torch.logit(torch.sigmoid(L).mean(0).clamp(1e-6, 1 - 1e-6)) if a.prob_avg else L.mean(0))
            if members:
                mem.append(torch.sigmoid(L).cpu())
        v = torch.sigmoid(torch.cat(out)).cpu().numpy() if out else np.zeros(0, np.float32)
        if members:
            return v, (torch.cat(mem, 1).numpy() if mem else np.zeros((len(nets), 0), np.float32))
        return v

    rp_files = [l.strip() for l in open(a.replay) if l.strip()] if a.replay else None
    t_start = time.time()
    dparams = [decay_params(n, j == 0) for j, n in enumerate(nets)] if a.value_wd > 0 else None
    while it < a.max_iters:
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
        v_new, v_mem = ens_value(ni, nf, members=True)
        fl = nm[:, M_KIND] == 0
        info["value_mean_floor_rows"] = round(float(v_new[fl].mean()), 4); info["outcome_mean_floor_rows"] = round(float(z_new[fl].mean()), 4)
        info["calibration"] = calibration(new, nm, z_new, v_new, v_mem)
        info["value_logloss"] = round(float(-(z_new[fl] * np.log(v_new[fl].clip(1e-6)) + (1 - z_new[fl]) * np.log((1 - v_new[fl]).clip(1e-6))).mean()), 4)
        info.update(baselines(new, nm, z_new, v_new, fl))
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
                parts = [policy_logp(nets[0], batch_of(di, dfl, np.arange(s, min(D, s + 8192))), torch.from_numpy(dm[s:s + 8192]).to(dev), dev)
                         for s in range(0, D, 8192)]
                lp_start = torch.cat([x[2] for x in parts])            # [D, 80] log-probabilities at the iteration start
                info["behaviour_lag_kl"] = round(float((torch.from_numpy(dm[:, M_LOGP]).to(dev) - torch.cat([x[0] for x in parts])).mean()), 5)
            for ep in range(a.epochs):
                perm = rng.permutation(D)
                for s in range(0, D, a.batch):
                    idx = perm[s:s + a.batch]
                    b = batch_of(di, dfl, idx)
                    m = torch.from_numpy(dm[idx]).to(dev)
                    A = torch.from_numpy(adv_n[idx]).to(dev)
                    lp_c, ent, lp, _ = policy_logp(nets[0], b, m, dev)
                    old = m[:, M_LOGP]
                    ratio = (lp_c - old).exp()
                    pol = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                    with torch.no_grad():
                        _, _, lp_bc, _ = policy_logp(bc, b, m, dev)
                    is_bc = torch.isin(b["dtype"], torch.tensor(BC_TYPES, device=dev))
                    kl_bc = ((lp.exp() * (lp.clamp(min=-50) - lp_bc.clamp(min=-50))).sum(-1) * is_bc).sum() / is_bc.sum().clamp(min=1)
                    loss = pol - a.ent * ent.mean() + beta * kl_bc
                    opts[0].zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(nets[0].parameters(), 1.0); opts[0].step()
                    with torch.no_grad():
                        ls = lp_start[torch.from_numpy(idx).to(dev)]
                        kl_step = (ls.exp() * (ls.clamp(min=-50) - lp.clamp(min=-50))).sum(-1).mean().item()   # KL(start || now)
                        stats["pol"] += pol.item(); stats["ent"] += ent.mean().item(); stats["kl_bc"] += kl_bc.item()
                        stats["kl_step"] += kl_step; stats["clip"] += ((ratio - 1).abs() > a.clip).float().mean().item(); steps += 1
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
        for r in new:                                     # the window keeps --value-rows-per-run rows of each run
            trim_rows(r, a.value_rows_per_run, rng)
        win = store.runs
        wi, wf = cat(win, "ints"), cat(win, "floats")
        wz = np.concatenate([np.full(len(r["meta"]), outcome(r, alpha), np.float32) for r in win])
        n_steps = a.value_steps if a.value_steps > 0 else max(1, int(math.ceil(a.value_reuse * len(ni) / a.batch)))
        vl = 0.0
        for j, (n, op) in enumerate(zip(nets, opts)):
            if a.value_dropout and j > 0:
                n.train()
            for _ in range(n_steps):
                idx = rng.integers(0, len(wz), a.batch)
                b = batch_of(wi, wf, idx)
                z = torch.from_numpy(wz[idx]).to(dev)
                # the policy member (0) trains only its value heads here: a value gradient into the encoder it shares with
                # the policy moved the policy outside PPO's trust region (behaviour-lag KL 0.41 after four updates in the
                # first server run). Members 1-4 have no policy in use and train fully.
                logit = n.forward_rows(b, policy=False, detach=(j == 0))[2]
                loss = F.binary_cross_entropy_with_logits(logit, z)
                op.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0); op.step()
                if dparams is not None:
                    with torch.no_grad():
                        torch._foreach_mul_(dparams[j], 1.0 - op.param_groups[0]["lr"] * a.value_wd)
                vl += loss.item()
            n.eval()
        info["value_loss"] = round(vl / (n_steps * len(nets)), 5); info["value_steps"] = n_steps; info["window_runs"] = len(win)
        info["window_rows"] = int(len(wz))
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


if __name__ == "__main__":
    main()
