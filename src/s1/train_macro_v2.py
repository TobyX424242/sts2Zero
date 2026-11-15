"""S1 macro network: value (P(win), HP-monotone) + macro policy on one shared encoder (PROPOSAL.md §6.3 item 10).

Value examples: the floors_v2 floor examples, as in train_value_v2.py. Policy examples: decisions/ from
build_decisions_v2.py; each is a floor example plus edits that restore the state before the choice.
Policy scoring:
  - option decisions (card reward + skip, relic pick, ancient, rest-site option): softmax over the options;
    score = MLP([q, e, q*e]), q = state vector + decision-type embedding, e = option embedding
    (card / relic / potion embedding shared with the encoder, or a learned special-option embedding);
  - shop: the same score per item, trained as buy / not buy (binary);
  - which deck card to upgrade or remove: score = MLP([q, h_j, q*h_j]) over the encoder outputs h_j of the deck card
    tokens; identical cards count as the same answer (loss on the summed probability of all copies).
No-attention model: pooled vector = mean of the real tokens (the CLS token is dropped: without attention it is a constant).
Loss = value loss + --pol-weight x policy loss; one policy batch per value batch.

Usage: python -I train_macro_v2.py <floors_dir> <out_dir> [--arch noattn] [--max-train-runs N] [--epochs 15] ...
"""
import argparse, collections, itertools, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "value_v1"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_value as tv
import train_value_v2 as v2
from build_decisions_v2 import DEL_CARD, ADD_CARD, UPG, DEL_RELIC, DEL_POT, SET_GOLD, SET_HP

S = v2.S
DT_NAMES = ["CARD_REWARD", "RELIC_PICK", "ANCIENT", "REST", "SMITH_TARGET", "REMOVE_TARGET", "SHOP"]
SOFTMAX_TYPES, SLOT_TYPES, BINARY_TYPES = (0, 1, 2, 3), (4, 5), (6,)


class MacroNet(v2.ValueNetV2):
    def __init__(self, vocab, n_special, rest_ids, d=128, layers=4, heads=4, ff=512, dropout=0.1, era_dropout=0.3, arch="noattn"):
        super().__init__(vocab, d, layers, heads, ff, dropout, era_dropout, arch, monotone=True)
        self.use_cls = arch != "noattn"
        if not self.use_cls:     # mean pooling only
            self.norm = nn.LayerNorm(d)
            self.trunk = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 256), nn.GELU(),
                                       nn.Linear(256, d), nn.GELU())
        self.special = nn.Embedding(n_special, d)
        self.dt_emb = nn.Embedding(len(DT_NAMES), d)
        self.opt_mlp = nn.Sequential(nn.Linear(3 * d, 256), nn.GELU(), nn.Linear(256, 1))
        self.slot_mlp = nn.Sequential(nn.Linear(3 * d, 256), nn.GELU(), nn.Linear(256, 1))
        self.register_buffer("rest_ids", torch.tensor(rest_ids, dtype=torch.long), persistent=False)

    def encode(self, b):
        """State vector z, plus per-token encoder outputs h and the padding mask (unpacked layout)."""
        b = v2.apply_drop(b, self.drop)
        scal = b["scal"] * self.mask_always
        phi = self._hp_features(scal)
        scal_enc = scal * self.mask_encoder
        B = b["ctx"].shape[0]; ctx = b["ctx"]
        era = ctx[:, 2]
        if self.training and self.era_dropout > 0:
            era = torch.where(torch.rand(B, device=era.device) < self.era_dropout, torch.full_like(era, 5), era)
        T = self.tok_type.weight
        ctx_tok = torch.stack([self.char(ctx[:, 0]), self.asc(ctx[:, 1]), self.era(era), self.act(ctx[:, 3].clamp(0, 3)),
                               self.room(ctx[:, 4]), self.enc(ctx[:, 5])], 1) + T[1]
        scal_tok = (self.scal(scal_enc.clamp(-10, 10)) + T[2]).unsqueeze(1)
        card_tok = (self.card(b["d_ids"]) + self.upg(b["d_upg"]) + self.ench(b["d_ench"])
                    + self.card_num(torch.stack([b["d_eamt"] / 20.0, b["d_age"] / 63.0], -1).float()) + T[3])
        relic_tok = self.relic(b["r_ids"]) + self.relic_num((b["r_age"] / 63.0).unsqueeze(-1).float()) + T[4]
        pot_tok = self.potion(b["p_ids"]) + T[5]
        x = torch.cat([self.cls.expand(B, -1, -1) + T[0], ctx_tok, scal_tok, card_tok, relic_tok, pot_tok], 1)
        pad = torch.cat([torch.zeros(B, 8, dtype=torch.bool, device=x.device), b["d_ids"] == 0, b["r_ids"] == 0, b["p_ids"] == 0], 1)
        if not self.use_cls:
            pad[:, 0] = True                      # CLS excluded from pooling
        h = self.encoder(x, src_key_padding_mask=pad) if self.use_cls else self.encoder(x)
        valid = (~pad).unsqueeze(-1).to(h.dtype)
        mean = (h * valid).sum(1) / valid.sum(1)
        pooled = torch.cat([h[:, 0], mean], -1) if self.use_cls else mean
        return self.trunk(self.norm(pooled)), phi, h

    def value_from(self, z, phi):
        logit = self.win_head(z).squeeze(-1).float() + (F.softplus(self.mono(z).float()) * phi).sum(-1)
        return logit, self.floor_head(z)

    def forward(self, b, hide_era=False):           # value interface used by train_value_v2.predict / report
        z, phi, _ = self.encode(b)
        return self.value_from(z, phi)

    def opt_emb(self, kind, oid, upg):
        e = torch.zeros(*kind.shape, self.card.embedding_dim, device=kind.device, dtype=self.card.weight.dtype)
        for k, fn in ((1, lambda i: self.card(i)), (2, lambda i: self.relic(i)), (3, lambda i: self.potion(i)), (4, lambda i: self.special(i))):
            m = kind == k
            if m.any():
                e[m] = fn(oid[m]).to(e.dtype)
        m = kind == 1
        if m.any():
            e[m] = e[m] + self.upg(upg[m]).to(e.dtype)
        return e

    def policy_logits(self, b):
        """Logits for every candidate of every decision: options [B, K] and deck slots [B, n_d]."""
        z, phi, h = self.encode(b)
        q = z + self.dt_emb(b["dtype"])
        kind, oid, upg = b["opt_kind"], b["opt_id"], b["opt_upg"]
        e = self.opt_emb(kind, oid, upg)
        qe = q.unsqueeze(1).expand_as(e).to(e.dtype)
        opt_logit = self.opt_mlp(torch.cat([qe, e, qe * e], -1)).squeeze(-1).float()
        nd = b["d_ids"].shape[1]
        hs = h[:, 8:8 + nd]
        qs = q.unsqueeze(1).expand_as(hs).to(hs.dtype)
        slot_logit = self.slot_mlp(torch.cat([qs, hs, qs * hs], -1)).squeeze(-1).float()
        # rest-site decisions: the offered set is not logged, so every rest option of the global vocabulary is a candidate
        er = self.special(self.rest_ids).unsqueeze(0).expand(len(q), -1, -1)
        qr = q.unsqueeze(1).expand_as(er).to(er.dtype)
        rest_logit = self.opt_mlp(torch.cat([qr, er, qr * er], -1)).squeeze(-1).float()
        return opt_logit, slot_logit, rest_logit, kind, oid


class DecisionData:
    def __init__(self, floors_dir, data, max_train_runs_set=None):
        D = os.path.join(floors_dir, "decisions")
        self.d = {k: np.load(os.path.join(D, f"{k}.npy")) for k in
                  ("ex", "dtype", "n_opt", "target_slot", "opt_kind", "opt_id", "opt_upg", "opt_label", "edits")}
        self.special = json.load(open(os.path.join(D, "special_vocab.json")))
        self.rest_ids = [i for i, s in enumerate(self.special) if s.startswith("REST:")]
        split = data.a["split"][self.d["ex"]]
        run = data.a["run"][self.d["ex"]]
        self.split = split
        tr = split == 0
        if max_train_runs_set is not None:
            tr &= np.isin(run, max_train_runs_set)
        self.train_idx = np.flatnonzero(tr)
        self.data = data

    def make(self, ids):
        """Pre-choice states (floor example + edits) as full-width numpy arrays, then sliced to real lengths."""
        A, d = self.data.a, self.d
        ex = d["ex"][ids]
        out = {k: A[k][ex].copy() for k in ("d_ids", "d_upg", "d_ench", "d_eamt", "d_age", "r_ids", "r_age", "p_ids", "scal")}
        out["ctx"] = A["ctx"][ex]
        ed = d["edits"][ids]
        sc = out["scal"]
        for i in range(len(ids)):
            for op, a1, a2, a3, a4, a5 in ed[i]:
                if op == 0:
                    break
                if op == DEL_CARD:
                    sc[i, S["deck_size"]] -= 1 / 40; sc[i, S["n_upgraded"]] -= float(out["d_upg"][i, a1] > 0) / 20
                    for k in ("d_ids", "d_upg", "d_ench", "d_eamt", "d_age"):
                        out[k][i, a1] = 0
                elif op == ADD_CARD:
                    z = np.flatnonzero(out["d_ids"][i] == 0)
                    if len(z):
                        j = z[0]
                        out["d_ids"][i, j], out["d_upg"][i, j], out["d_ench"][i, j], out["d_eamt"][i, j], out["d_age"][i, j] = a1, a2, a3, a4, a5
                        sc[i, S["deck_size"]] += 1 / 40; sc[i, S["n_upgraded"]] += float(a2 > 0) / 20
                elif op == UPG:
                    before = out["d_upg"][i, a1] > 0
                    out["d_upg"][i, a1] = max(0, int(out["d_upg"][i, a1]) + a2)
                    sc[i, S["n_upgraded"]] += (float(out["d_upg"][i, a1] > 0) - float(before)) / 20
                elif op == DEL_RELIC:
                    out["r_ids"][i, a1] = 0; out["r_age"][i, a1] = 0; sc[i, S["n_relics"]] -= 1 / 20
                elif op == DEL_POT:
                    out["p_ids"][i, a1] = 0; sc[i, S["n_potions"]] -= 1 / 5
                elif op == SET_GOLD:
                    sc[i, S["gold"]] = a1 / 500
                elif op == SET_HP:
                    sc[i, S["hp"]] = a1 / 100; sc[i, S["hp_frac"]] = a1 / max(1.0, sc[i, S["max_hp"]] * 100)
        nd = max(int((out["d_ids"] > 0).sum(1).max()), int(d["target_slot"][ids].max()) + 1, 1)
        nr = max(int((out["r_ids"] > 0).sum(1).max()), 1); npo = max(int((out["p_ids"] > 0).sum(1).max()), 1)
        # deck slots are kept in place (slot targets refer to them), so slice to the highest used slot, not the count
        nd = max(nd, int(np.max(np.flatnonzero(out["d_ids"].any(0)), initial=0)) + 1)
        nr = max(nr, int(np.max(np.flatnonzero(out["r_ids"].any(0)), initial=0)) + 1)
        npo = max(npo, int(np.max(np.flatnonzero(out["p_ids"].any(0)), initial=0)) + 1)
        t = {}
        for k in ("d_ids", "d_upg", "d_ench", "d_eamt", "d_age"):
            v = out[k][:, :nd]; t[k] = torch.from_numpy(v.view(np.int16) if v.dtype == np.uint16 else v)
        for k in ("r_ids", "r_age"):
            v = out[k][:, :nr]; t[k] = torch.from_numpy(v.view(np.int16) if v.dtype == np.uint16 else v)
        t["p_ids"] = torch.from_numpy(out["p_ids"][:, :npo].view(np.int16))
        t["ctx"] = torch.from_numpy(out["ctx"]); t["scal"] = torch.from_numpy(out["scal"])
        for k in ("dtype", "n_opt", "target_slot", "opt_kind", "opt_id", "opt_upg", "opt_label"):
            t[k] = torch.from_numpy(d[k][ids])
        return {k: v.pin_memory() for k, v in t.items()}

    def batches(self, idx, bs, device, workers=6, ahead=12):
        starts = iter(range(0, len(idx), bs))
        with ThreadPoolExecutor(workers) as ex:
            dq = collections.deque(ex.submit(self.make, idx[i:i + bs]) for i in itertools.islice(starts, ahead))
            while dq:
                b = dq.popleft().result()
                nxt = next(starts, None)
                if nxt is not None:
                    dq.append(ex.submit(self.make, idx[nxt:nxt + bs]))
                g = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                for k in ("d_ids", "d_upg", "d_ench", "r_ids", "p_ids", "opt_id", "opt_upg"):
                    g[k] = g[k].int() if k not in ("opt_id", "opt_upg") else g[k].long()
                g["ctx"] = g["ctx"].long(); g["dtype"] = g["dtype"].long(); g["opt_kind"] = g["opt_kind"].long()
                g["target_slot"] = g["target_slot"].long(); g["n_opt"] = g["n_opt"].long()
                yield g


def policy_loss(model, b, n_rest):
    """Mean loss over the batch and per-example correctness (top-1) for the choose-one types."""
    opt_logit, slot_logit, rest_logit, kind, oid = model.policy_logits(b)
    dt = b["dtype"]; B = dt.shape[0]
    loss = torch.zeros(B, device=dt.device); correct = torch.zeros(B, device=dt.device)
    # choose-one over the offered options
    m = (dt == 0) | (dt == 1) | (dt == 2)
    if m.any():
        lg = opt_logit[m].masked_fill(kind[m] == 0, -1e9)
        tgt = b["opt_label"][m].argmax(1)
        loss[m] = F.cross_entropy(lg, tgt, reduction="none")
        correct[m] = (lg.argmax(1) == tgt).float()
    # rest site: the stored option id is the chosen one; candidates are model.rest_ids
    m = dt == 3
    if m.any():
        tgt = (model.rest_ids.unsqueeze(0) == b["opt_id"][m][:, :1]).float().argmax(1)
        loss[m] = F.cross_entropy(rest_logit[m], tgt, reduction="none")
        correct[m] = (rest_logit[m].argmax(1) == tgt).float()
    # which deck card (identical copies are the same answer)
    m = (dt == 4) | (dt == 5)
    if m.any():
        ids, upg, ench = b["d_ids"][m], b["d_upg"][m], b["d_ench"][m]
        ts = b["target_slot"][m].clamp(min=0)
        r = torch.arange(len(ts), device=ts.device)
        same = (ids == ids[r, ts].unsqueeze(1)) & (upg == upg[r, ts].unsqueeze(1)) & (ench == ench[r, ts].unsqueeze(1))
        valid = ids > 0
        valid = valid & torch.where((dt[m] == 4).unsqueeze(1), upg == 0, torch.ones_like(valid))   # smith: un-upgraded cards
        valid[r, ts] = True; same &= valid
        lg = slot_logit[m].masked_fill(~valid, -1e9)
        lse = torch.logsumexp(lg, 1)
        loss[m] = lse - torch.logsumexp(lg.masked_fill(~same, -1e9), 1)
        correct[m] = same[r, lg.argmax(1)].float()
    # shop: buy / not buy per item
    m = dt == 6
    if m.any():
        valid = b["opt_kind"][m] > 0
        bce = F.binary_cross_entropy_with_logits(opt_logit[m], b["opt_label"][m], reduction="none")
        loss[m] = (bce * valid).sum(1) / valid.sum(1).clamp(min=1)
        pred = (opt_logit[m] > 0).float()
        correct[m] = (((pred == b["opt_label"][m]) | ~valid).all(1)).float()
    return loss, correct


@torch.no_grad()
def policy_eval(model, dd, split, device, n_rest, max_n=400_000, seed=0):
    idx = np.flatnonzero(dd.split == split)
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
    model.eval()
    L = collections.defaultdict(float); C = collections.defaultdict(float); N = collections.Counter()
    for b in dd.batches(idx, 4096, device):
        with torch.autocast("cuda", torch.bfloat16):
            loss, corr = policy_loss(model, b, n_rest)
        dt = b["dtype"].cpu().numpy(); loss = loss.float().cpu().numpy(); corr = corr.cpu().numpy()
        for t in np.unique(dt):
            m = dt == t
            L[int(t)] += float(loss[m].sum()); C[int(t)] += float(corr[m].sum()); N[int(t)] += int(m.sum())
    out = {DT_NAMES[t]: {"n": N[t], "loss": round(L[t] / N[t], 4), "top1": round(C[t] / N[t], 4)} for t in sorted(N)}
    out["all_loss"] = round(sum(L.values()) / max(1, sum(N.values())), 4)
    return out


def pickrate_baseline(dd, split):
    """Card rewards: pick the option with the highest training pick rate (skip = global skip rate). Top-1 accuracy."""
    d = dd.d; K = d["opt_id"].shape[1]; ar = np.arange(K)

    def rows(mask):
        i = np.flatnonzero(mask & (d["dtype"] == 0))
        n = d["n_opt"][i].astype(np.int64)
        return d["opt_id"][i].astype(np.int64), d["opt_label"][i] > 0, n
    ids, lab, n = rows(dd.split == 0)
    card = ar[None, :] < (n - 1)[:, None]
    V = int(d["opt_id"].max()) + 1
    offers = np.bincount(ids[card], minlength=V); picks = np.bincount(ids[card & lab], minlength=V)
    rate = np.where(offers > 0, picks / np.maximum(offers, 1), 0.0)
    skip_rate = float(lab[np.arange(len(n)), n - 1].mean())
    ids, lab, n = rows(dd.split == split)
    card = ar[None, :] < (n - 1)[:, None]
    sc = np.where(card, rate[ids], -1.0)
    sc[np.arange(len(n)), n - 1] = skip_rate
    hit = lab[np.arange(len(n)), sc.argmax(1)]
    return {"n": int(len(n)), "top1": round(float(hit.mean()), 4), "train_skip_rate": round(skip_rate, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("floors_dir"); ap.add_argument("out_dir")
    ap.add_argument("--arch", default="noattn", choices=["transformer", "noattn"])
    ap.add_argument("--max-train-runs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--spr", type=int, default=8)
    ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--aux", type=float, default=0.25); ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pol-weight", type=float, default=1.0)
    ap.add_argument("--val-floors", type=int, default=1_000_000); ap.add_argument("--val-decisions", type=int, default=200_000)
    ap.add_argument("--target-build", default="v0.111.0")
    ap.add_argument("--pack", type=int, default=4)
    args = ap.parse_args()
    v2.PACK = args.pack
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); device = "cuda"
    logf = open(os.path.join(args.out_dir, "train.log"), "a")

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    t0 = time.time()
    data = v2.DataV2(args.floors_dir, args.max_train_runs, args.seed)
    dd = DecisionData(args.floors_dir, data, data.train_runs if args.max_train_runs else None)
    tb = data.builds.index(args.target_build) if args.target_build in data.builds else -1
    log(f"config {vars(args)}")
    log(f"floors {len(data.a['win']):,}; train runs {len(data.train_runs):,}; decisions {len(dd.d['ex']):,} "
        f"(train {len(dd.train_idx):,}); loaded in {time.time()-t0:.0f}s")
    model = MacroNet(data.vocab, len(dd.special), dd.rest_ids, arch=args.arch).to(device)
    log(f"MacroNet {args.arch}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    decay, no_decay = [], []
    for mod in model.modules():
        for pn, p_ in mod.named_parameters(recurse=False):
            (decay if p_.ndim >= 2 and not isinstance(mod, nn.Embedding) and pn != "cls" else no_decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    n_val = min(len(data.train_runs) * args.spr, len(data.train_idx))
    steps_per_epoch = math.ceil(n_val / args.bs); total = steps_per_epoch * args.epochs

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, total - args.warmup))))
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))
    ema.n_averaged = ema.n_averaged.to(device)
    val_idx = np.flatnonzero(data.a["split"] == 1)
    if args.val_floors and args.val_floors < len(val_idx):
        val_idx = np.sort(np.random.default_rng(1).choice(val_idx, args.val_floors, replace=False))
    yv = data.a["win"][val_idx].astype(np.float64)
    rng = np.random.default_rng(args.seed)
    best, best_state, bad, step, hist = 1e9, None, 0, 0, []
    for ep in range(args.epochs):
        model.train(); te = time.time()
        tlv = torch.zeros((), device=device); tlp = torch.zeros((), device=device); n = 0
        vidx = data.epoch_indices(0, args.spr, rng)
        pidx = rng.choice(dd.train_idx, min(len(vidx), len(dd.train_idx)), replace=False)
        for bv, bp in zip(v2.batches(data, vidx, args.bs, device), dd.batches(pidx, args.bs, device)):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", torch.bfloat16):
                lw, lf = model(bv)
                lv = F.binary_cross_entropy_with_logits(lw.float(), bv["win"]) + \
                     args.aux * -(tv.hl_gauss(bv["floors_left"]) * F.log_softmax(lf.float(), -1)).sum(-1).mean()
                lp, _ = policy_loss(model, bp, len(dd.rest_ids))
                lp = lp.mean()
                loss = lv + args.pol_weight * lp
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update_parameters(model)
            tlv += lv.detach(); tlp += lp.detach(); n += 1; step += 1
        torch.cuda.synchronize(); t_train = time.time() - te
        pv = v2.predict(ema.module, data, val_idx, device)
        v = tv.metrics(pv, yv)
        pe = policy_eval(ema.module, dd, 1, device, len(dd.rest_ids), args.val_decisions)
        crit = v["logloss"] + args.pol_weight * pe["all_loss"]
        rec = {"epoch": ep + 1, "train_value_loss": round(float(tlv) / n, 4), "train_policy_loss": round(float(tlp) / n, 4),
               "val": v, "val_policy": pe, "criterion": round(crit, 4), "train_sec": round(t_train, 1), "sec": round(time.time() - te, 1)}
        hist.append(rec)
        log(f"ep {ep+1:2d} value {rec['train_value_loss']:.4f} policy {rec['train_policy_loss']:.4f} | val logloss {v['logloss']:.4f} "
            f"auc {v['auc']:.4f} | val policy {pe['all_loss']:.4f} " +
            " ".join(f"{k[:6]}:{x['top1']:.3f}" for k, x in pe.items() if isinstance(x, dict)) + f" | {rec['sec']}s")
        if crit < best - 1e-4:
            best, bad = crit, 0
            best_state = {k: t.detach().clone() for k, t in ema.module.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {ep+1}"); break
    model = ema.module; model.load_state_dict(best_state)
    res = {"config": vars(args), "history": hist, "train_runs": int(len(data.train_runs)),
           "test": v2.report(model, data, 2, device, tb), "test_policy": policy_eval(model, dd, 2, device, len(dd.rest_ids), 10**9),
           "test_card_reward_pickrate_baseline": pickrate_baseline(dd, 2),
           "hp_probe_val": v2.hp_probe(model, data, device), "elapsed_s": round(time.time() - t0)}
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "vocab": data.vocab, "special": dd.special},
               os.path.join(args.out_dir, "model.pt"))
    json.dump(res, open(os.path.join(args.out_dir, "results.json"), "w"), indent=1)
    t = res["test"]
    log(f"TEST value logloss {t['all']['logloss']} auc {t['all']['auc']} within-floor {t['auc_within_floor']} "
        f"ece-bands {t['ece_within_floor_bands']} | cell {t['target_cell'].get('logloss')} / {t['target_cell'].get('auc')}")
    log(f"TEST policy {json.dumps(res['test_policy'])} | pick-rate baseline (card reward) {res['test_card_reward_pickrate_baseline']}")


if __name__ == "__main__":
    main()
