"""S2 combat-network trainer: learns from the self-play chunks written by the C# actors (research/s2/S2/Fights.cs
ChunkWriter) and publishes weights for the inference server.

Loop: scan --data for new *.s2d chunks into a replay window of the last --window decisions; train while
steps * batch < reuse * decisions ingested (each decision is used about --reuse times); every --save-every steps write
the checkpoint to --weights (temp file + rename; the server reloads it) with the version bumped, and keep a copy every
--keep-every versions in --ckpt-dir. Losses: policy cross-entropy against the search's improved policy (fully searched
moves only), plus the value head:
  outcome head (default): cross-entropy of the end-HP bin distribution (death = bin 0; surviving fights get an HL-Gauss
    target around the exact end HP, sigma 0.75 bin), and per held potion slot the binary cross-entropy of "still held at
    the end" (surviving fights, slots held at the decision only);
  scalar head: P(survive) binary cross-entropy, relative T_f squared error (surviving fights only).

Options from KataGo (Wu 2019), all off by default (then the trainer behaves exactly as before):
  --surprise W   policy-surprise weighting: a decision's sampling weight is (1 - W) + W * s / mean(s), s = KL(policy
                 target || network policy) measured the last time it was trained on, mean(s) over the measured decisions
                 of the replay buffer; never-trained decisions and decisions without a policy target (fast-search moves)
                 have weight 1. Drawn as: --surprise-pool x batch uniform candidates, batch resampled from them in
                 proportion to weight (systematic resampling; no importance correction, as KataGo).
  --ema D        publish an exponential moving average of the weights (decay D per step) under "state_dict" (what the
                 servers and evaluations load) and the raw training weights under "train_state_dict"; --init resumes
                 the raw weights from "train_state_dict" when present and the average from "state_dict".
  --window-min C growing replay window: train on the newest min(--window, f(N)) decisions, N = decisions ingested by
                 this trainer, f(N) = N for N <= C, else C * (1 + beta * ((N / C)^alpha - 1) / alpha) (KataGo's formula;
                 alpha = --window-exp, beta = --window-beta).

Carry-over between trainer processes (S3 segments; off by default):
  --save-carry P on SIGTERM/SIGINT and at the end, write the newest --carry-keep replay entries (with their surprise
                 priorities), the optimizer state and the counters to P (temp file + rename: P is complete or absent;
                 a size + trailer check rejects anything else).
  --load-carry P at the start, if P is complete: the entries fill the replay buffer (they count toward --min-data but not
                 toward the reuse budget, see --carry-reuse) and the optimizer state is restored when the parameter
                 shapes match. Missing or broken file: a log line and a fresh start as without the option.

Shadow trainers and offline comparisons (2026-11-13, IMPROVEMENT_PLAN.md change C; off by default):
  --tee-dir D1[,D2..] every chunk this trainer reads is hard-linked into each Di first (before it is archived or
                 deleted), so other trainers (e.g. a larger network) learn from the same self-play stream.
  --skip-every K chunks whose sequence number (<tag>_<seq>.s2d) is a multiple of K are teed but not trained on: a
                 held-out set that is the same for every trainer of the stream (research/s2/py/combat_heldout.py).

Network configuration overrides (2026-11-15; none by default):
  --config K=V   (repeatable) set a config key of the network, e.g. --config ext=1: the layout-v4/v5 network
                 (combat_obs.CombatNet ext_feat; research/s2/py/model.py). With --init the checkpoint's weights are loaded
                 into the overridden network: a v3 checkpoint + ext=1 = an ext network whose outputs equal the
                 checkpoint's at step 0 (ext_feat's last layer starts at zero). Saved checkpoints carry the config.
                 Shadow on the live stream (run_s3.sh SHADOW_ARGS="--config ext=1"): the shadow trainer starts from the
                 previous segment's shadow.pt, so for its first segment copy the live weights.pt to <previous segment>/
                 shadow.pt (no shadow.pt = random weights); later segments continue the ext shadow (--config ext=1 is
                 then a no-op). Do not put --init into SHADOW_ARGS: it would come after run_s3.sh's own --init and
                 restart the shadow from that file in every segment.

Usage: python -I research/s2/py/train_combat.py --data DIR --weights PATH [--ckpt-dir DIR] [--window 500000]
"""
import argparse, copy, glob, io, json, math, os, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model import load_net, load_weights, co, HP_BINS, POTION_SLOTS  # noqa: E402

T, TW, A, AW = co.MAX_TOKENS, co.TOKEN_WIDTH, co.MAX_ACTIONS, co.ACTION_WIDTH
MW = 12


def read_chunk(path):
    with open(path, "rb") as f:
        b = f.read()
    assert b[:8] == b"S2DATA01", path
    ver, n, t, tw, a, aw, mw = np.frombuffer(b, "<i4", 7, 8)
    assert (t, tw, a, aw, mw) == (T, TW, A, AW, MW) and ver in (2, 3), (path, ver, t, tw, a, aw, mw)
    off = 64
    obs = np.frombuffer(b, "<i2", n * T * TW, off).reshape(n, T, TW); off += obs.nbytes
    act = np.frombuffer(b, "<i2", n * A * AW, off).reshape(n, A, AW); off += act.nbytes
    pol = np.frombuffer(b, "<f4", n * A, off).reshape(n, A); off += pol.nbytes
    meta = np.frombuffer(b, "<f4", n * MW, off).reshape(n, MW)
    return obs, act, pol, meta


def chunk_seq(path):
    """Sequence number of a chunk file <tag>_<seq>.s2d (-1 if the name has another form)."""
    try:
        return int(os.path.basename(path).rsplit("_", 1)[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def kata_window(n, cap, c, alpha, beta):
    """KataGo's growing window: all n decisions up to c, then c * (1 + beta * ((n / c)^alpha - 1) / alpha), at most cap."""
    if n <= c:
        return min(cap, n)
    return min(cap, int(c * (1 + beta * ((n / c) ** alpha - 1) / alpha)))


class Replay:
    def __init__(self, W, surprise=False):
        self.W = W
        self.obs = np.zeros((W, T, TW), np.int16)
        self.act = np.zeros((W, A, AW), np.int16)
        self.pol = np.zeros((W, A), np.float32)
        self.meta = np.zeros((W, MW), np.float32)
        self.ptr = 0
        self.size = 0
        self.ingested = 0
        self.seen = set()
        self.win = None                    # effective window (newest decisions) for the growing window; None = all
        self.surp = np.full(W, -1.0, np.float32) if surprise else None   # last measured KL; -1 = none
        self.surp_mean, self.surp_frac = 0.0, 0.0
        self.preloaded = 0                 # entries from --load-carry (not counted in ingested)

    def arrays(self):
        out = [("obs", self.obs), ("act", self.act), ("pol", self.pol), ("meta", self.meta)]
        return out + ([("surp", self.surp)] if self.surp is not None else [])

    def newest(self, n):
        """Ring slices holding the newest n entries, oldest first."""
        n = min(n, self.size)
        s = (self.ptr - n) % self.W
        return [(s, s + n)] if s + n <= self.W else [(s, self.W), (0, self.ptr)]

    def save_carry(self, path, keep, extra):
        """Newest `keep` entries + `extra` (bytes: optimizer state, counters) to path via path.tmp and a rename.
        Layout: 8-byte magic, 8-byte header length, JSON header (padded to 4096 bytes), raw arrays, extra, 16-byte trailer."""
        n = min(keep, self.size)
        sl = self.newest(n)
        arrs = self.arrays()
        total = 16 + 4096 + sum(n * a[0].nbytes for _, a in arrs) + len(extra) + 16
        hdr = {"n": n, "arrays": [[k, a.dtype.str, list(a.shape[1:])] for k, a in arrs], "extra": len(extra), "total": total}
        hb = json.dumps(hdr).encode().ljust(4096)
        assert len(hb) == 4096
        tmp = path + ".tmp"
        def put(f, b):
            # a raw write may store fewer bytes than asked (Linux: at most ~2.1 GB per call): loop until all is written
            mv = memoryview(b)
            if mv.nbytes == 0:                # e.g. --carry-keep 0 (optimizer state only): nothing to write
                return
            mv = mv.cast("B")
            while len(mv):
                k = f.write(mv)
                if not k:
                    raise OSError("write returned 0")
                mv = mv[k:]
        with open(tmp, "wb", buffering=0) as f:
            put(f, b"S2CARRY1" + len(hb).to_bytes(8, "little") + hb)
            for _, a in arrs:
                for s, e in sl:
                    put(f, np.ascontiguousarray(a[s:e]))
            put(f, extra)
            put(f, b"S2CARRYEND000000")
        if os.path.getsize(tmp) != total:
            raise OSError(f"carry file size {os.path.getsize(tmp)} != {total}")
        os.replace(tmp, path)
        return n, total

    def load_carry(self, path):
        """Fills the buffer from a carry file; returns (entries, bytes, extra bytes). Raises on any inconsistency."""
        size = os.path.getsize(path)
        with open(path, "rb", buffering=0) as f:
            head = f.read(16)
            assert head[:8] == b"S2CARRY1", "bad magic"
            hl = int.from_bytes(head[8:16], "little")
            assert 0 < hl < 1 << 20, "bad header length"
            hdr = json.loads(f.read(hl))
            assert hdr["total"] == size, f"size {size} != {hdr['total']} (truncated)"
            f.seek(size - 16)
            assert f.read(16) == b"S2CARRYEND000000", "bad trailer"
            f.seek(16 + hl)
            n = hdr["n"]
            mine = dict(self.arrays())
            for k, dt, shp in hdr["arrays"]:
                assert k not in mine or (mine[k].dtype.str == dt and list(mine[k].shape[1:]) == shp), f"array {k} differs"
            m = min(n, self.W)                                  # a smaller buffer keeps the newest entries
            for k, dt, shp in hdr["arrays"]:
                row = np.dtype(dt).itemsize * int(np.prod(shp))
                if k not in mine:                               # e.g. surprise priorities, this trainer without --surprise
                    f.seek(n * row, 1)
                    continue
                f.seek((n - m) * row, 1)
                buf = memoryview(mine[k][:m].reshape(-1).view(np.uint8))
                got = 0
                while got < len(buf):
                    r = f.readinto(buf[got:])
                    assert r, "short read"
                    got += r
            extra = f.read(hdr["extra"])
            assert len(extra) == hdr["extra"], "short extra"
        if self.surp is not None and not any(k == "surp" for k, _, _ in hdr["arrays"]):
            self.surp[:m] = -1.0
        self.ptr, self.size, self.preloaded = m % self.W, m, m
        if self.surp is not None:
            self.refresh_surprise()
        return m, size, extra

    def add(self, obs, act, pol, meta):
        n = len(obs)
        for s in range(0, n, self.W):
            o, a, p, m = obs[s:s + self.W], act[s:s + self.W], pol[s:s + self.W], meta[s:s + self.W]
            k = len(o)
            i = np.arange(self.ptr, self.ptr + k) % self.W
            self.obs[i], self.act[i], self.pol[i], self.meta[i] = o, a, p, m
            if self.surp is not None:
                self.surp[i] = -1.0
            self.ptr = (self.ptr + k) % self.W
            self.size = min(self.W, self.size + k)
        self.ingested += n

    def scan(self, d, delete=True, archive="", tee=(), skip_every=0):
        new = sorted((p for p in glob.glob(os.path.join(d, "*.s2d")) if p not in self.seen), key=os.path.getmtime)
        for p in new:
            try:
                chunk = read_chunk(p)
                for t in tee:                 # shadow trainers / held-out collection get the same chunk
                    try:
                        os.link(p, os.path.join(t, os.path.basename(p)))
                    except FileExistsError:
                        pass
                if not (skip_every > 0 and chunk_seq(p) % skip_every == 0):
                    self.add(*chunk)
                self.seen.add(p)
                side = p + ".obs"   # layout-v5 sidecar written next to the chunk (obs=..., intent_damage=..., relic_counters=...)
                if archive:  # kept in the run archive (KEEP=1); the data is paid for, do not silently discard it
                    os.replace(p, os.path.join(archive, os.path.basename(p)))
                    if os.path.exists(side):
                        os.replace(side, os.path.join(archive, os.path.basename(side)))
                elif delete:  # the replay window holds the data now; chunks fill the disk otherwise (~9 GB per hour, 2026-11-11)
                    os.remove(p)
                    if os.path.exists(side):
                        os.remove(side)
            except Exception as e:  # partially visible file: retry next scan
                print(f"[train] skip {p}: {e}", flush=True)
        if new and self.surp is not None:
            self.refresh_surprise()
        return len(new)

    def refresh_surprise(self):
        """Mean of the measured surprises in the buffer (the normaliser of the weights) and the measured share."""
        s = self.surp[:self.size]
        m = s >= 0
        c = int(m.sum())
        self.surp_mean = float(s[m].mean()) if c else 0.0
        self.surp_frac = c / max(1, self.size)

    def draw(self, k, rng):
        if self.win is None:
            return rng.integers(0, self.size, k)
        n = max(1, min(self.size, self.win))
        return (self.ptr - 1 - rng.integers(0, n, k)) % self.W   # the newest n entries of the ring

    def sample(self, B, rng, surprise=0.0, pool=8):
        """Returns (obs, act, pol, meta, indices, info); info has the surprise-sampling statistics (empty when off)."""
        info = {}
        if surprise <= 0:
            i = self.draw(B, rng)
        else:
            cand = self.draw(pool * B, rng)
            s = self.surp[cand]
            meas = s >= 0
            if self.surp_mean > 0:
                w = np.where(meas, (1 - surprise) + surprise * s / self.surp_mean, 1.0)
            else:
                w = np.ones(len(cand))
            # systematic resampling: candidate j is taken floor or ceil of B * w_j / sum(w) times (the candidates are
            # in random order, so the batch is too); repeats only for w_j > sum(w) / B, i.e. above pool x the mean
            c = np.cumsum(w)
            pick = np.minimum(np.searchsorted(c, (rng.random() + np.arange(B)) * (c[-1] / B), side="right"), len(cand) - 1)
            i = cand[pick]
            sp, mp = s[pick], meas[pick]
            info = {"surp_sampled": float(sp[mp].mean()) if mp.any() else 0.0,      # stored surprise, this batch
                    "surp_uniform": float(s[meas].mean()) if meas.any() else 0.0,   # same, uniform candidates
                    "surp_ess": float(w.sum() ** 2 / (len(w) * (w * w).sum())),     # Kish ESS / pool size
                    "surp_dup": 1.0 - len(np.unique(i)) / B}                       # share of repeated rows
        return self.obs[i], self.act[i], self.pol[i], self.meta[i], i, info


def hl_gauss(frac, survived, sigma=0.75):
    """End-HP target distribution over the 41 bins: death -> bin 0; alive -> Gaussian mass over bins 1..40 around
    position 40 * frac + 0.5 (bin k covers ((k-1)/40, k/40] of max HP, so its centre k - 0.5 maps to 40 * frac)."""
    k = torch.arange(1, HP_BINS, device=frac.device, dtype=torch.float32)
    x = (40 * frac).clamp(1e-3, 40)[:, None] + 0.5                     # bin centres sit at k = 40 * hp_frac + 0.5
    cdf = lambda z: 0.5 * (1 + torch.erf(z / (sigma * math.sqrt(2))))
    mass = cdf(k + 0.5 - x) - cdf(k - 0.5 - x)
    mass = mass / mass.sum(1, keepdim=True).clamp(min=1e-9)
    tgt = torch.zeros(len(frac), HP_BINS, device=frac.device)
    tgt[:, 1:] = mass * survived[:, None]
    tgt[:, 0] = 1 - survived
    return tgt


def value_losses(head, v, m, rel_weight):
    """Returns (value loss, logged metrics)."""
    surv = m[:, 1]
    if head == "scalar":
        s, r = v[:, 0], v[:, 1]
        v_loss = F.binary_cross_entropy_with_logits(s, surv)
        alive = surv > 0
        rel_loss = F.mse_loss(r[alive], m[alive, 2]) if alive.any() else r.sum() * 0
        return v_loss + rel_weight * rel_loss, {"survive": v_loss.item(), "rel": rel_loss.item()}
    hp_logits, keep_logits = v[:, :HP_BINS], v[:, HP_BINS:HP_BINS + POTION_SLOTS]
    tgt = hl_gauss(m[:, 11], surv)
    logp = torch.log_softmax(hp_logits, -1)
    hp_loss = -(tgt * logp).sum(-1).mean()
    held = m[:, 9].long(); end = m[:, 10].long()
    bits = 2 ** torch.arange(POTION_SLOTS, device=v.device)
    held_b = (held[:, None] & bits) > 0
    end_b = ((end[:, None] & bits) > 0).float()
    mask = held_b & (surv[:, None] > 0)
    keep_loss = (F.binary_cross_entropy_with_logits(keep_logits[mask], end_b[mask]) if mask.any() else keep_logits.sum() * 0)
    with torch.no_grad():
        p_death = logp[:, 0].exp().clamp(1e-6, 1 - 1e-6)
        surv_bce = -(surv * torch.log(1 - p_death) + (1 - surv) * torch.log(p_death)).mean().item()
        centres = (torch.arange(HP_BINS, device=v.device, dtype=torch.float32) - 0.5).clamp(min=0) / 40
        hp_mae = ((logp.exp() * centres).sum(-1) - m[:, 11]).abs().mean().item()
        kept_rate = end_b[mask].mean().item() if mask.any() else 0.0
    return hp_loss + keep_loss, {"hp": hp_loss.item(), "keep": keep_loss.item(), "survive": surv_bce, "hp_mae": hp_mae, "kept_rate": kept_rate}


def losses(net, obs, act, pol, meta, dev, head="outcome", rel_weight=2.0, per_sample=False):
    """Returns (policy loss, value loss, logged metrics[, per-decision KL(target || net) when per_sample])."""
    o = torch.from_numpy(obs).to(dev, non_blocking=True).long()
    a = torch.from_numpy(act).to(dev, non_blocking=True).long()
    p = torch.from_numpy(pol).to(dev, non_blocking=True)
    m = torch.from_numpy(meta).to(dev, non_blocking=True)
    logits, v = net(o, a)
    logits = logits.float(); v = v.float()
    has = m[:, 0] > 0
    logp = torch.log_softmax(logits, -1)
    ce = -torch.where(p > 0, p * logp, torch.zeros_like(logp)).sum(-1)
    ent = -torch.where(p > 0, p * torch.log(p.clamp(min=1e-12)), torch.zeros_like(p)).sum(-1)
    pol_loss = (ce[has].mean() if has.any() else logits.sum() * 0)
    kl = ((ce - ent)[has].mean() if has.any() else torch.zeros((), device=dev))
    v_loss, vinfo = value_losses(head, v, m, rel_weight)
    with torch.no_grad():
        acc = ((logits.argmax(-1) == p.argmax(-1)).float()[has].mean() if has.any() else torch.zeros((), device=dev))
    info = {"kl": kl.item(), "pol_acc": acc.item(), "target_entropy": ent[has].mean().item() if has.any() else 0.0, **vinfo}
    if per_sample:
        return pol_loss, v_loss, info, (ce - ent).detach()
    return pol_loss, v_loss, info


def save_carry(a, rb, opt, net, version, steps_total, credit):
    t = time.time()
    try:
        bio = io.BytesIO()
        torch.save({"opt": opt.state_dict(), "shapes": [list(p.shape) for p in net.parameters()], "version": version,
                    "steps_total": steps_total, "credit": float(credit), "reuse": a.reuse}, bio)
        n, nb = rb.save_carry(a.save_carry, a.carry_keep, bio.getvalue())
        print(f"[train] carry saved to {a.save_carry}: {n} entries, {nb / 1e6:.0f} MB, {time.time() - t:.1f} s; "
              f"optimizer state, {steps_total} steps in total, unused budget {credit:.0f} samples", flush=True)
    except Exception as e:
        print(f"[train] carry NOT saved ({type(e).__name__}: {e})", flush=True)


def load_carry(a, rb, net, opt, dev):
    """Returns (start budget in samples, steps of the earlier trainers). Anything wrong: fresh start, buffer empty."""
    if not os.path.exists(a.load_carry):
        print(f"[train] no carry file {a.load_carry}: fresh start", flush=True)
        return 0, 0
    t = time.time()
    try:
        n, nb, extra = rb.load_carry(a.load_carry)
    except Exception as e:
        rb.ptr = rb.size = rb.preloaded = 0
        print(f"[train] carry file {a.load_carry} rejected ({type(e).__name__}: {e}): fresh start", flush=True)
        return 0, 0
    msg = f"[train] carry loaded from {a.load_carry}: {n} entries, {nb / 1e6:.0f} MB, {time.time() - t:.1f} s"
    steps_before, unused = 0, 0.0
    try:
        st = torch.load(io.BytesIO(extra), map_location=dev, weights_only=False)
        steps_before, unused = int(st.get("steps_total", 0)), max(0.0, float(st.get("credit", 0.0)))
        if st["shapes"] != [list(p.shape) for p in net.parameters()]:
            raise ValueError("parameter shapes differ")
        opt.load_state_dict(st["opt"])
        for g in opt.param_groups:             # hyperparameters come from this command line, not from the file
            g["lr"], g["weight_decay"] = a.lr, a.wd
        msg += f"; optimizer state restored ({steps_before} steps before)"
    except Exception as e:
        msg += f"; optimizer state NOT restored ({type(e).__name__}: {e}), fresh optimizer"
    # budget: what the previous trainer had not used yet (at most --reuse uses per loaded entry) + --carry-reuse uses
    credit = min(unused, a.reuse * n) + a.carry_reuse * n
    print(msg + f"; start budget {credit:.0f} samples ({credit / a.batch:.0f} steps before new data is needed)", flush=True)
    return credit, steps_before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--vocab", default=os.path.join(HERE, "..", "..", "s0", "data", "combat_vocab_v1.json"))
    ap.add_argument("--window", type=int, default=500_000, help="replay buffer size in decisions (the largest window)")
    ap.add_argument("--window-min", type=int, default=0,
                    help="growing window (KataGo): train on the newest min(--window, f(N)) decisions, N = decisions "
                    "ingested by this trainer, f(N) = N up to this value C, then C*(1+beta*((N/C)^alpha-1)/alpha); "
                    "0 = off (the whole buffer). KataGo: C = 250000, alpha 0.75, beta 0.4")
    ap.add_argument("--window-exp", type=float, default=0.75, help="alpha of the growing window")
    ap.add_argument("--window-beta", type=float, default=0.4, help="beta of the growing window")
    ap.add_argument("--surprise", type=float, default=0.0,
                    help="policy-surprise weighting W (KataGo uses 0.5): sampling weight (1-W) + W*s/mean(s), s = "
                    "KL(policy target || network) measured when the decision was last trained on; decisions not yet "
                    "trained on or without a policy target weigh 1; 0 = uniform sampling")
    ap.add_argument("--surprise-pool", type=int, default=8,
                    help="surprise weighting draws this many x batch uniform candidates and resamples the batch from them")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decay per step of the published weight average (e.g. 0.999); the file then holds the average "
                    "under state_dict and the raw weights under train_state_dict; 0 = publish the raw weights")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reuse", type=float, default=4.0)
    ap.add_argument("--min-data", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", default="const", choices=["const", "cosine"],
                    help="cosine: from --lr to --lr-min over --total-steps, then held at --lr-min")
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--total-steps", type=int, default=26000, help="cosine length; the 1-hour 30-actor run made ~26k steps")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--rel-weight", type=float, default=2.0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--keep-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head", default="outcome", choices=["outcome", "scalar"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="")
    ap.add_argument("--archive-dir", default="", help="move ingested chunks here (for backup) instead of deleting them")
    ap.add_argument("--init", default="", help="start from this checkpoint (S3: the S2 network) instead of random weights; "
                    "its size and head replace --d/--layers/--head and the version numbering continues; the raw weights "
                    "come from its train_state_dict when present (a file written with --ema), else from state_dict")
    ap.add_argument("--save-carry", default="", help="on SIGTERM/SIGINT and at the end write the newest --carry-keep replay "
                    "entries, the optimizer state and the counters to this file (temp file + rename)")
    ap.add_argument("--load-carry", default="", help="start with the replay entries and optimizer state of this carry file "
                    "if it exists and is complete (else a log line and a fresh start)")
    ap.add_argument("--carry-keep", type=int, default=1_000_000, help="entries written by --save-carry (4.1 kB each)")
    ap.add_argument("--carry-reuse", type=float, default=0.5,
                    help="training budget granted at the start for the loaded entries, in uses per entry (training can "
                    "start before new data arrives); on top of the budget the previous trainer had not used, which is "
                    "capped at --reuse uses per loaded entry. New data then buys --reuse uses per decision as usual")
    ap.add_argument("--tee-dir", default="", help="comma-separated dirs: hard-link every chunk read into each (shadow trainers)")
    ap.add_argument("--skip-every", type=int, default=0, help="chunks with sequence number %% K == 0 are not trained on (held out)")
    ap.add_argument("--obs", type=int, default=0,
                    help="observation layout the actors used: 4 = pre-2026-11-15 (default), 5 = relic counters in the masked "
                         "v5 slots, 6 = also the corrected intent damage (an existing slot). Recorded in the checkpoint "
                         "config so evaluation can match; it does not change the network by itself. Pass --config ext=1 "
                         "explicitly for a v3 checkpoint that must start with an ext head (the shadow restart).")
    ap.add_argument("--config", action="append", default=[], metavar="K=V",
                    help="network config override (repeatable), e.g. ext=1: the layout-v4 network; applied on top of "
                    "--init's config (its weights are loaded into the overridden network) or of --d/--layers/--head")
    a = ap.parse_args()
    over = {}
    for kv in a.config:
        k, _, v = kv.partition("=")
        assert k and _, f"--config expects K=V, got {kv!r}"
        for conv in (int, float):
            try:
                v = conv(v); break
            except ValueError:
                pass
        over[k] = v
    if a.obs >= 5:   # 4 is the default: no key, exactly the old checkpoint config
        over["obs"] = int(a.obs)
    tee = [t for t in a.tee_dir.split(",") if t]
    for t in tee:
        os.makedirs(t, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    vocab = co.load_vocab(a.vocab)
    version = 1
    ck0 = None
    if a.init:
        ck0 = torch.load(a.init, map_location=dev, weights_only=False)
        net = load_net(vocab, ck0, over).to(dev)
        if "train_state_dict" in ck0:      # written with --ema: state_dict is the average, resume the raw weights
            load_weights(net, ck0["train_state_dict"])
        a.head = net.config["head"]; version = int(ck0.get("version", 0)) + 1
    else:
        net = load_net(vocab, None, {"d": a.d, "layers": a.layers, "head": a.head, **over}).to(dev)
    if over:
        print(f"[train] config overrides {over}: network config {net.config}", flush=True)
    ema, ema_n = None, 0
    if a.ema > 0:
        ema = copy.deepcopy(net)           # average starts at the published weights of --init (the average, if any)
        if ck0 is not None:
            load_weights(ema, ck0["state_dict"] if "state_dict" in ck0 else ck0)   # ext_feat: net's fresh copy
        ema.requires_grad_(False)
        ema_p, net_p = list(ema.parameters()), list(net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    log = open(a.log or os.path.join(os.path.dirname(os.path.abspath(a.weights)), "train_log.jsonl"), "a")
    if a.ckpt_dir:
        os.makedirs(a.ckpt_dir, exist_ok=True)

    def save():
        if ema is None:
            ck = {"state_dict": net.state_dict(), "config": net.config, "version": version}
        else:
            for be, b in zip(ema.buffers(), net.buffers()):
                be.copy_(b)
            ck = {"state_dict": ema.state_dict(), "train_state_dict": net.state_dict(), "config": net.config, "version": version}
        tmp = a.weights + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, a.weights)
        if a.ckpt_dir and (version % a.keep_every == 0 or version == 1):
            torch.save(ck, os.path.join(a.ckpt_dir, f"v{version:05d}.pt"))
    save()  # the first version the actors start with (random weights, or --init)
    print(f"[train] {sum(p.numel() for p in net.parameters()):,} parameters on {dev}; wrote version {version} to {a.weights}", flush=True)
    rb = Replay(a.window, surprise=a.surprise > 0)
    credit, steps_before = 0, 0        # training budget (samples) on top of reuse x ingested; steps of earlier trainers
    if a.load_carry:
        credit, steps_before = load_carry(a, rb, net, opt, dev)
    rng = np.random.default_rng(a.seed)
    step, t0, acc = 0, time.time(), {}
    last_scan = 0.0
    stop = {"flag": False, "sleeping": False}

    class Stop(Exception):
        pass

    def on_signal(sig, frame):         # the loop ends after the current step; an idle wait is cut short
        stop["flag"] = True
        if stop["sleeping"]:
            raise Stop
    if a.save_carry:
        signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)
    try:
        while step < a.max_steps and not stop["flag"]:
            if time.time() - last_scan > 5 or rb.size == 0:
                rb.scan(a.data, archive=a.archive_dir, tee=tee, skip_every=a.skip_every); last_scan = time.time()
                if a.window_min > 0:
                    rb.win = kata_window(rb.ingested + rb.preloaded, a.window, a.window_min, a.window_exp, a.window_beta)
            if rb.ingested + rb.preloaded < a.min_data or step * a.batch >= a.reuse * rb.ingested + credit:
                if a.save_carry:
                    stop["sleeping"] = True
                    try:
                        if not stop["flag"]:
                            time.sleep(1.0)
                    finally:
                        stop["sleeping"] = False
                else:
                    time.sleep(1.0)
                continue
            net.train()
            obs, act, pol, meta, idx, sinfo = rb.sample(a.batch, rng, a.surprise, a.surprise_pool)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                if a.surprise > 0:
                    pl, vl, info, kl_i = losses(net, obs, act, pol, meta, dev, a.head, a.rel_weight, per_sample=True)
                else:
                    pl, vl, info = losses(net, obs, act, pol, meta, dev, a.head, a.rel_weight)
            loss = pl + vl
            if not torch.isfinite(loss):          # one bad batch must not poison the weights the actors use
                print("[train] non-finite loss: batch skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            if a.surprise > 0:                    # surprise of the batch before this step = its new sampling priority
                has = meta[:, 0] > 0
                k = kl_i.float().cpu().numpy()
                ok = has & np.isfinite(k)
                rb.surp[idx[ok]] = np.maximum(k[ok], 0.0)
                sinfo["pol_share"] = float(has.mean())
                sinfo["surp_mean"] = rb.surp_mean; sinfo["surp_measured"] = rb.surp_frac
            if a.lr_schedule == "cosine":
                frac = min(1.0, step / max(1, a.total_steps))
                for g in opt.param_groups:
                    g["lr"] = a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * frac))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if ema is not None:
                # from random weights the average warms up (decay (1+n)/(10+n) while smaller); from --init it does not
                d = a.ema if a.init else min(a.ema, (1 + ema_n) / (10 + ema_n))
                with torch.no_grad():
                    torch._foreach_lerp_(ema_p, net_p, 1.0 - d)
                ema_n += 1
            if a.surprise > 0 and step % 100 == 0:
                rb.refresh_surprise()
            for k, v in {"policy": pl.item(), "value": vl.item(), "grad_norm": gn.item(), **info, **sinfo}.items():
                acc[k] = acc.get(k, 0.0) + v
            if step % a.save_every == 0:
                version += 1
                save()
                rec = {"t": round(time.time() - t0, 1), "step": step, "version": version, "ingested": rb.ingested, "window": rb.size,
                       "lr": opt.param_groups[0]["lr"],
                       **{k: round(v / a.save_every, 5) for k, v in acc.items()}}
                if rb.win is not None:
                    rec["window_eff"] = min(rb.size, rb.win)
                if rb.preloaded:
                    rec["carried"] = rb.preloaded
                if ema is not None:
                    with torch.no_grad():
                        num = sum(((pe - pn).float() ** 2).sum() for pe, pn in zip(ema_p, net_p))
                        den = sum((pn.float() ** 2).sum() for pn in net_p)
                    rec["ema_rel_dist"] = round(math.sqrt(num.item() / max(den.item(), 1e-12)), 6)
                acc = {}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print("[train] " + json.dumps(rec), flush=True)
    except Stop:
        pass
    if a.save_carry:
        save_carry(a, rb, opt, net, version, steps_before + step, a.reuse * rb.ingested + credit - step * a.batch)


if __name__ == "__main__":
    main()
