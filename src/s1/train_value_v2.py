"""S1 v2 value network (PROPOSAL.md §6.3, items 1, 2, 4 and 5).

Changes from value_v1/train_value.py (whose data class, metrics and optimizer recipe are reused):
  1. floors sampled uniformly from all training floors (samples per epoch = --spr x training runs);
  2. state-only inputs: damage taken, HP healed and turns taken are zeroed;
  4. HP monotone by construction: HP and HP fraction are removed from the encoder input and enter only through
       logit V = a(z) + sum_k softplus(b_k(z)) * phi_k(HP, max HP),   phi_k fixed increasing ramps,
     so raising HP with everything else fixed can never lower V;
  5. trained on natural frequencies (dataset from build_dataset_v2.py); --max-train-runs for the scaling curve.
  Architecture: --arch transformer (trial 1) or noattn (per-token MLP + mean pooling; won at 100k runs, V4).
  Speed (r2, from the training-speed study in perf/): entity tokens are left-packed into one list and each batch is
  sorted by length and encoded in --pack chunks (exact: no positional encodings); batches are sliced to their real
  lengths in narrow dtypes in a thread pool; no per-step GPU syncs; prediction sorts by length once and unsorts;
  optional --compile of the encoder. The tested pre-speedup version is kept as train_value_v2_r1.py.

Usage: python -I train_value_v2.py <floors_dir> <out_dir> [--arch noattn] [--max-train-runs N] [--epochs 15] ...
"""
import argparse, collections, itertools, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "value_v1"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_value as tv

S = {n: i for i, n in enumerate(["floor", "act", "hp", "max_hp", "hp_frac", "gold", "dmg_taken", "hp_healed", "deck_size",
                                 "n_upgraded", "n_enchanted", "n_relics", "n_potions", "turns_taken", "is_combat", "floor_in_act"])}
ZERO_ALWAYS = [S["dmg_taken"], S["hp_healed"], S["turns_taken"]]       # item 2
ZERO_ENCODER = [S["hp"], S["hp_frac"]]                                   # item 4: HP only via the monotone head
HP_KNOTS = torch.arange(0, 160, 10.0)                                    # absolute-HP ramps, width 10
FRAC_KNOTS = torch.arange(0, 1.0, 0.1)                                   # HP-fraction ramps, width 0.1
N_RAMPS = len(HP_KNOTS) + len(FRAC_KNOTS)


def hp_features(scal):
    hp = scal[:, S["hp"]].float() * 100
    frac = scal[:, S["hp_frac"]].float().clamp(0, 1)
    a = ((hp[:, None] - HP_KNOTS.to(hp.device)) / 10).clamp(0, 1)
    b = ((frac[:, None] - FRAC_KNOTS.to(hp.device)) / 0.1).clamp(0, 1)
    return torch.cat([a, b], 1)


DROPS = ("relics", "potions", "card_meta", "era", "encounter", "room_act", "scalars")


def apply_drop(b, drop):
    """Input ablations: remove one input group for every example (training and evaluation alike)."""
    if not drop:
        return b
    b = dict(b)
    if "relics" in drop:
        b["r_ids"] = torch.zeros_like(b["r_ids"]); b["r_age"] = torch.zeros_like(b["r_age"])
    if "potions" in drop:
        b["p_ids"] = torch.zeros_like(b["p_ids"])
    if "card_meta" in drop:     # keep card identities, drop upgrade level, enchantment and age
        for k in ("d_upg", "d_ench", "d_eamt", "d_age"):
            b[k] = torch.zeros_like(b[k])
    ctx = b["ctx"].clone()
    if "era" in drop:
        ctx[:, 2] = 5
    if "encounter" in drop:
        ctx[:, 5] = 1
    if "room_act" in drop:
        ctx[:, 3] = 0; ctx[:, 4] = 1
    b["ctx"] = ctx
    if "scalars" in drop:       # all scalars except HP, HP fraction and max HP (those feed the monotone head / max HP)
        scal = b["scal"].clone(); keep = [S["hp"], S["hp_frac"], S["max_hp"]]
        mask = torch.zeros(scal.shape[1], device=scal.device); mask[keep] = 1
        b["scal"] = scal * mask
    return b


class NoAttention(nn.Module):
    def __init__(self, d, ff, blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)) for _ in range(blocks))

    def forward(self, x, src_key_padding_mask=None):
        for b in self.blocks:
            x = x + b(x)
        return x


class ValueNetV2(tv.ValueNet):
    def __init__(self, vocab, d=128, layers=4, heads=4, ff=512, dropout=0.1, era_dropout=0.3, arch="transformer", monotone=True, drop=()):
        super().__init__(vocab, d, layers, heads, ff, dropout, era_dropout)
        self.arch, self.monotone, self.drop = arch, monotone, tuple(drop)
        if arch == "noattn":
            self.encoder = NoAttention(d, ff)
            self.trunk = nn.Sequential(nn.Linear(2 * d, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 256), nn.GELU(),
                                       nn.Linear(256, d), nn.GELU())
        if monotone:
            self.mono = nn.Linear(d, N_RAMPS)
            nn.init.zeros_(self.mono.weight); nn.init.constant_(self.mono.bias, -4.0)   # softplus(-4) = 0.018 per ramp: starts near zero
        # non-persistent buffers: no CPU->GPU copies or synchronising index writes per step; state_dict keys unchanged
        self.register_buffer("hp_knots", HP_KNOTS.clone(), persistent=False)
        self.register_buffer("frac_knots", FRAC_KNOTS.clone(), persistent=False)
        m_all = torch.ones(16); m_all[ZERO_ALWAYS] = 0
        m_enc = m_all.clone(); m_enc[ZERO_ENCODER] = 0
        self.register_buffer("mask_always", m_all, persistent=False)
        self.register_buffer("mask_encoder", m_enc, persistent=False)
        self._enc_fast = None      # optional compiled encoder, set by enable_compile() on the training copy only

    def enable_compile(self):
        object.__setattr__(self, "_enc_fast", torch.compile(self.encoder, dynamic=True))

    def _hp_features(self, scal):
        hp = scal[:, S["hp"]].float() * 100
        frac = scal[:, S["hp_frac"]].float().clamp(0, 1)
        a = ((hp[:, None] - self.hp_knots) / 10).clamp(0, 1)
        b = ((frac[:, None] - self.frac_knots) / 0.1).clamp(0, 1)
        return torch.cat([a, b], 1)

    def forward(self, b, hide_era=False):
        b = apply_drop(b, self.drop)
        scal = b["scal"] * self.mask_always
        phi = self._hp_features(scal) if self.monotone else None
        if self.monotone:
            scal = scal * self.mask_encoder
        b2 = dict(b); b2["scal"] = scal
        z = self._trunk(b2, hide_era)
        logit = self.win_head(z).squeeze(-1).float()
        if self.monotone:
            logit = logit + (F.softplus(self.mono(z).float()) * phi).sum(-1)
        return logit, self.floor_head(z)

    def _trunk(self, b, hide_era):
        # the encoder part of tv.ValueNet.forward, returning the pooled state vector z
        B = b["ctx"].shape[0]
        ctx = b["ctx"]
        era = ctx[:, 2].clone()
        if hide_era:
            era[:] = 5
        elif self.training and self.era_dropout > 0:
            era = torch.where(torch.rand(B, device=era.device) < self.era_dropout, torch.full_like(era, 5), era)
        T = self.tok_type.weight
        ctx_tok = torch.stack([self.char(ctx[:, 0]), self.asc(ctx[:, 1]), self.era(era), self.act(ctx[:, 3].clamp(0, 3)),
                               self.room(ctx[:, 4]), self.enc(ctx[:, 5])], 1) + T[1]
        scal_tok = (self.scal(b["scal"].clamp(-10, 10)) + T[2]).unsqueeze(1)
        card_tok = (self.card(b["d_ids"]) + self.upg(b["d_upg"]) + self.ench(b["d_ench"])
                    + self.card_num(torch.stack([b["d_eamt"] / 20.0, b["d_age"] / 63.0], -1).float()) + T[3])
        relic_tok = self.relic(b["r_ids"]) + self.relic_num((b["r_age"] / 63.0).unsqueeze(-1).float()) + T[4]
        pot_tok = self.potion(b["p_ids"]) + T[5]
        x = torch.cat([self.cls.expand(B, -1, -1) + T[0], ctx_tok, scal_tok, card_tok, relic_tok, pot_tok], 1)
        pad = torch.cat([torch.zeros(B, 8, dtype=torch.bool, device=x.device), b["d_ids"] == 0, b["r_ids"] == 0, b["p_ids"] == 0], 1)
        enc = self._enc_fast if self._enc_fast is not None else self.encoder
        chunks = b.get("_chunks")
        if not chunks:
            h = enc(x, src_key_padding_mask=pad)
            valid = (~pad).unsqueeze(-1).to(h.dtype)
            pooled = torch.cat([h[:, 0], (h * valid).sum(1) / valid.sum(1)], -1)
            return self.trunk(self.norm(pooled))
        # packed: move the valid tokens to the front of each row (stable, so CLS and context stay first), then encode
        # chunks of rows of similar length, each cut to its own length. Exact, because no token carries a position.
        Lmax = max(L for _, _, L in chunks)
        if x.shape[1] < Lmax:
            x = F.pad(x, (0, 0, 0, Lmax - x.shape[1])); pad = F.pad(pad, (0, Lmax - pad.shape[1]), value=True)
        order = torch.argsort(pad.to(torch.int8), dim=1, stable=True)[:, :Lmax]
        x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        pad = torch.gather(pad, 1, order)
        pooled = []
        for r0, r1, L in chunks:
            xc, pc = x[r0:r1, :L], pad[r0:r1, :L]
            h = enc(xc, src_key_padding_mask=pc)
            valid = (~pc).unsqueeze(-1).to(h.dtype)
            pooled.append(torch.cat([h[:, 0], (h * valid).sum(1) / valid.sum(1)], -1))
        return self.trunk(self.norm(torch.cat(pooled, 0)))


class DataV2(tv.Data):
    def __init__(self, d, max_train_runs=0, seed=0):
        super().__init__(d)
        self.a["build"] = np.load(os.path.join(d, "build.npy"))
        self.builds = json.load(open(os.path.join(d, "builds.json")))
        tr_runs = np.flatnonzero(self.run_split == 0)
        if max_train_runs and max_train_runs < len(tr_runs):
            keep = np.random.default_rng(seed).choice(tr_runs, max_train_runs, replace=False)
            self.train_runs = np.sort(keep)
        else:
            self.train_runs = tr_runs
        a = self.a   # entity counts per example, for slicing batches to their real lengths and sorting by length
        self.nd = (a["d_ids"] > 0).sum(1).astype(np.int16)
        self.nr = (a["r_ids"] > 0).sum(1).astype(np.int16)
        self.np_ = (a["p_ids"] > 0).sum(1).astype(np.int16)
        self.tot = self.nd + self.nr + self.np_
        st, ln = self.run_start[self.train_runs], self.run_len[self.train_runs]
        self.train_idx = np.concatenate([np.arange(s, s + l) for s, l in zip(st, ln)]) if len(st) else np.zeros(0, np.int64)

    def epoch_indices(self, split, k, rng):   # item 1: uniform over training floors, k samples per training run
        n = min(len(self.train_runs) * k, len(self.train_idx))
        return rng.choice(self.train_idx, n, replace=False)


ENT = (("d_ids", ("d_upg", "d_ench", "d_eamt", "d_age")), ("r_ids", ("r_age",)), ("p_ids", ()))
INT_KEYS = ("d_ids", "d_upg", "d_ench", "r_ids", "p_ids")
PACK = 4        # set from --pack in main()


def chunk_meta(tot_sorted, k):
    B = len(tot_sorted); out = []
    for c in range(k):
        r0, r1 = c * B // k, (c + 1) * B // k
        if r1 > r0:
            L = 8 + max(int(tot_sorted[r0:r1].max()), 1)
            out.append((r0, r1, int(math.ceil(L / 8) * 8)))   # multiple of 8: fewer shapes for torch.compile
    return out


def batches(data, idx, bs, device, workers=6, ahead=12, pack=None):
    """Batches assembled in a thread pool, in order: rows sorted by entity count (if packing), arrays sliced to the
    batch's real lengths in their stored narrow dtypes, pinned, copied asynchronously and widened on the GPU."""
    pack = PACK if pack is None else pack
    a = data.a

    def make(i):
        ids = idx[i:i + bs]
        meta = {}
        if pack:
            ids = ids[np.argsort(data.tot[ids], kind="stable")]
            meta["_chunks"] = chunk_meta(data.tot[ids], pack)
        n = {"d_ids": max(int(data.nd[ids].max()), 1), "r_ids": max(int(data.nr[ids].max()), 1), "p_ids": max(int(data.np_[ids].max()), 1)}
        out = {}
        for key, others in ENT:
            for k in (key,) + others:
                v = a[k][ids, :n[key]]
                out[k] = torch.from_numpy(v.view(np.int16) if v.dtype == np.uint16 else v).pin_memory()
        out["ctx"] = torch.from_numpy(a["ctx"][ids]).pin_memory()
        out["scal"] = torch.from_numpy(a["scal"][ids]).pin_memory()
        out["win"] = torch.from_numpy(a["win"][ids].astype(np.float32)).pin_memory()
        out["floors_left"] = torch.from_numpy(a["floors_left"][ids].astype(np.float32)).pin_memory()
        return out, meta
    starts = iter(range(0, len(idx), bs))
    with ThreadPoolExecutor(workers) as ex:
        dq = collections.deque(ex.submit(make, i) for i in itertools.islice(starts, ahead))
        while dq:
            b, meta = dq.popleft().result()
            nxt = next(starts, None)
            if nxt is not None:
                dq.append(ex.submit(make, nxt))
            g = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            for k in INT_KEYS:
                g[k] = g[k].int()
            g["ctx"] = g["ctx"].long()
            g.update(meta)
            yield g


@torch.no_grad()
def predict(model, data, idx, device, bs=16384, scal_edit=None):
    """Predictions in the order of idx. Examples are sorted by entity count once, so batches are short and dense."""
    model.eval(); out = []
    idx = np.asarray(idx)
    if not len(idx):
        return np.zeros(0)
    order = np.argsort(data.tot[idx], kind="stable") if PACK else np.arange(len(idx))
    for b in batches(data, idx[order], bs, device):
        if scal_edit is not None:
            b["scal"] = scal_edit(b["scal"])
        with torch.autocast("cuda", torch.bfloat16):
            lw, _ = model(b)
        out.append(torch.sigmoid(lw.float()))
    p = torch.cat(out).cpu().numpy()
    res = np.empty_like(p); res[order] = p
    return res


def auc(p, y):
    return tv._auc_simple(*tv._sorted(p, y, np.ones(len(p)))) if len(p) and 0 < y.mean() < 1 else float("nan")


def report(model, data, split, device, target_build):
    a = data.a
    idx = np.flatnonzero(a["split"] == split)
    p = predict(model, data, idx, device)
    y = a["win"][idx].astype(np.float64); fl = a["floor"][idx]; ctx = a["ctx"][idx]; bld = a["build"][idx]
    num = den = 0.0
    for f in np.unique(fl):
        m = fl == f
        if m.sum() >= 500 and 0 < y[m].mean() < 1:
            num += auc(p[m], y[m]) * m.sum(); den += m.sum()
    bands = ((1, 5), (6, 10), (11, 16), (17, 24), (25, 33), (34, 42), (43, 60))
    ece_band = sum(tv.metrics(p[(fl >= lo) & (fl <= hi)], y[(fl >= lo) & (fl <= hi)])["ece"] * ((fl >= lo) & (fl <= hi)).sum()
                   for lo, hi in bands if ((fl >= lo) & (fl <= hi)).any()) / len(idx)
    cell = (ctx[:, 0] == 2) & (ctx[:, 1] == 10) & (bld == target_build)
    out = {"examples": int(len(idx)), "all": tv.metrics(p, y), "auc_within_floor": round(num / max(den, 1), 4),
           "ece_within_floor_bands": round(float(ece_band), 4),
           "auc_by_act": [round(auc(p[ctx[:, 3] == k], y[ctx[:, 3] == k]), 4) for k in range(3)],
           "calibration_by_floor": [[f"{lo}-{hi}", round(float(p[(fl >= lo) & (fl <= hi)].mean()), 4),
                                     round(float(y[(fl >= lo) & (fl <= hi)].mean()), 4)] for lo, hi in bands],
           "target_cell": {"examples": int(cell.sum()), **(tv.metrics(p[cell], y[cell]) if cell.any() else {}),
                           "auc_by_act": [round(auc(p[cell & (ctx[:, 3] == k)], y[cell & (ctx[:, 3] == k)]), 4) for k in range(3)]}}
    return out


def hp_probe(model, data, device, n=50000):
    idx = np.random.default_rng(0).choice(np.flatnonzero(data.a["split"] == 1), n, replace=False)
    base = predict(model, data, idx, device)

    def edit(s):
        s = s.clone(); hp, mhp = s[:, S["hp"]] * 100, s[:, S["max_hp"]] * 100
        hp2 = torch.minimum(hp + 10, mhp); s[:, S["hp"]], s[:, S["hp_frac"]] = hp2 / 100, hp2 / mhp.clamp(min=1)
        return s
    d = predict(model, data, idx, device, scal_edit=edit) - base
    room = data.a["scal"][idx, S["hp"]] < data.a["scal"][idx, S["max_hp"]] - 0.095
    act = data.a["ctx"][idx, 3].clip(0, 2)
    return {"share_increasing": round(float((d[room] > 0).mean()), 4), "share_decreasing": round(float((d[room] < 0).mean()), 4),
            "share_decreasing_over_0.5pt": round(float((d[room] < -0.005).mean()), 4),
            "mean_pts_by_act": [round(100 * float(d[room & (act == k)].mean()), 2) for k in range(3)]}


def train(model, data, args, device, log):
    rng = np.random.default_rng(args.seed)
    decay, no_decay = [], []
    for mod in model.modules():
        for pn, p_ in mod.named_parameters(recurse=False):
            (decay if p_.ndim >= 2 and not isinstance(mod, nn.Embedding) and pn != "cls" else no_decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    steps_per_epoch = math.ceil(min(len(data.train_runs) * args.spr, len(data.train_idx)) / args.bs)
    total = steps_per_epoch * args.epochs

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, total - args.warmup))))
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))
    ema.n_averaged = ema.n_averaged.to(device)     # otherwise a CPU tensor is copied to the GPU every step
    if args.compile:
        model.enable_compile()                     # after the EMA copy, so the EMA model keeps the eager encoder
    val_idx = np.flatnonzero(data.a["split"] == 1)
    if args.val_floors and args.val_floors < len(val_idx):
        val_idx = np.sort(np.random.default_rng(1).choice(val_idx, args.val_floors, replace=False))
    yv = data.a["win"][val_idx].astype(np.float64)
    best, best_state, bad, step, hist = 1e9, None, 0, 0, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tl = torch.zeros((), device=device); tn = 0
        for b in batches(data, data.epoch_indices(0, args.spr, rng), args.bs, device):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", torch.bfloat16):
                lw, lf = model(b)
                loss = F.binary_cross_entropy_with_logits(lw.float(), b["win"]) + \
                       args.aux * -(tv.hl_gauss(b["floors_left"]) * F.log_softmax(lf.float(), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update_parameters(model)
            n_b = len(b["win"]); tl += loss.detach() * n_b; tn += n_b; step += 1
        torch.cuda.synchronize(); t_train = time.time() - t0
        tl = float(tl) / tn
        pv = predict(ema.module, data, val_idx, device)
        v = tv.metrics(pv, yv); t_val = time.time() - t0 - t_train
        rec = {"epoch": ep + 1, "train_loss": round(tl, 4), "val": v, "sec": round(time.time() - t0, 1),
               "train_sec": round(t_train, 1), "val_sec": round(t_val, 1), "train_samples_per_s": round(tn / t_train)}
        hist.append(rec)
        log(f"ep {ep+1:2d} train {tl:.4f} | val logloss {v['logloss']:.4f} auc {v['auc']:.4f} ece {v['ece']:.4f} | "
            f"train {t_train:.1f}s ({rec['train_samples_per_s']}/s), val {t_val:.1f}s")
        if v["logloss"] < best - 1e-4:
            best, bad = v["logloss"], 0
            best_state = {k: t.detach().clone() for k, t in ema.module.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {ep+1}")
                break
    ema.module.load_state_dict(best_state)
    return ema.module, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("floors_dir"); ap.add_argument("out_dir")
    ap.add_argument("--arch", default="transformer", choices=["transformer", "noattn"])
    ap.add_argument("--no-monotone", action="store_true")
    ap.add_argument("--max-train-runs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--spr", type=int, default=8)
    ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--aux", type=float, default=0.25); ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--era-dropout", type=float, default=0.3)
    ap.add_argument("--val-floors", type=int, default=1_000_000, help="validation floors used for early stopping")
    ap.add_argument("--target-build", default="v0.111.0")
    ap.add_argument("--drop", default="", help=f"comma-separated input groups to remove: {','.join(DROPS)}")
    ap.add_argument("--pack", type=int, default=4, help="row chunks per batch for packed encoding; 0 = off (r1 behaviour)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the encoder of the training model")
    args = ap.parse_args()
    global PACK
    PACK = args.pack
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"
    logf = open(os.path.join(args.out_dir, "train.log"), "a")

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    t0 = time.time()
    data = DataV2(args.floors_dir, args.max_train_runs, args.seed)
    tb = data.builds.index(args.target_build) if args.target_build in data.builds else -1
    log(f"config {vars(args)}")
    log(f"data loaded in {time.time()-t0:.0f}s: {len(data.a['win']):,} floors; train runs used {len(data.train_runs):,} "
        f"({len(data.train_idx):,} floors); val/test runs {int((data.run_split == 1).sum()):,}/{int((data.run_split == 2).sum()):,}")
    model = ValueNetV2(data.vocab, args.d, args.layers, args.heads, args.ff, args.dropout, args.era_dropout,
                       args.arch, not args.no_monotone, [g for g in args.drop.split(",") if g]).to(device)
    assert all(g in DROPS for g in model.drop), f"unknown --drop group; choose from {DROPS}"
    log(f"{args.arch} params {sum(p.numel() for p in model.parameters())/1e6:.2f}M, monotone HP head: {not args.no_monotone}, dropped inputs: {model.drop}")
    model, hist = train(model, data, args, device, log)
    res = {"config": vars(args), "history": hist, "train_runs": int(len(data.train_runs)),
           "test": report(model, data, 2, device, tb), "hp_probe_val": hp_probe(model, data, device),
           "elapsed_s": round(time.time() - t0)}
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "vocab": data.vocab}, os.path.join(args.out_dir, "model.pt"))
    json.dump(res, open(os.path.join(args.out_dir, "results.json"), "w"), indent=1)
    t = res["test"]
    log(f"TEST logloss {t['all']['logloss']} auc {t['all']['auc']} within-floor {t['auc_within_floor']} "
        f"ece-bands {t['ece_within_floor_bands']} | target cell {t['target_cell']} | hp probe {res['hp_probe_val']}")


if __name__ == "__main__":
    main()
