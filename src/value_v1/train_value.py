"""Trial 1: run-value transformer V(state after floor) -> P(win), trained on Spire Codex per-floor examples.

Design notes (see RESEARCH.md section 13):
- entity-set transformer (Deep Sets / Set Transformer / AlphaStar-style): deck cards, relics and potions
  are unordered tokens with no positional encoding; context tokens for character, ascension, version era,
  act, room type and encounter; one token for scalar status; CLS + mean pooling.
- K floors per run per epoch (AlphaGo: successive positions of one game share a label and overfit).
- auxiliary head: floors remaining, HL-Gauss classification (Farebrother et al. 2024), weight 0.25.
- version-era dropout 0.3 (the era token is replaced by <unk>), EMA weights for evaluation.
- unweighted loss on the balanced sample; metrics both balanced and population-weighted (weight_norm).

Usage: python -I train_value.py <floors_dir> <out_dir> [--epochs 20] [--k 8] ...
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FIELDS = ["d_ids", "d_upg", "d_ench", "d_eamt", "d_age", "r_ids", "r_age", "p_ids", "ctx", "scal"]
N_BINS = 61


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, d):
        self.a = {k: np.load(os.path.join(d, f"{k}.npy")) for k in FIELDS + ["win", "floors_left", "floor", "run", "split", "weight"]}
        self.vocab = json.load(open(os.path.join(d, "vocab.json")))
        run = self.a["run"]
        starts = np.flatnonzero(np.r_[True, run[1:] != run[:-1]])
        self.run_start, self.run_len = starts, np.diff(np.r_[starts, len(run)])
        self.run_split = self.a["split"][starts]

    def batch(self, idx, device):
        out = {}
        for k in FIELDS:
            out[k] = torch.from_numpy(self.a[k][idx].astype(np.float32 if k in ("scal",) else np.int64)).to(device, non_blocking=True)
        out["win"] = torch.from_numpy(self.a["win"][idx].astype(np.float32)).to(device)
        out["floors_left"] = torch.from_numpy(self.a["floors_left"][idx].astype(np.float32)).to(device)
        return out

    def epoch_indices(self, split, k, rng):
        sel = np.flatnonzero(self.run_split == split)
        st, ln = self.run_start[sel], self.run_len[sel]
        pick = st[:, None] + (rng.random((len(sel), k)) * ln[:, None]).astype(np.int64)
        idx = pick.ravel()
        rng.shuffle(idx)
        return idx


def trim(b):
    """Entities are left-packed; drop all-padding columns to save compute."""
    for ids, others in (("d_ids", ("d_upg", "d_ench", "d_eamt", "d_age")), ("r_ids", ("r_age",)), ("p_ids", ())):
        n = int((b[ids] > 0).sum(1).max().item()) or 1
        b[ids] = b[ids][:, :n]
        for o in others:
            b[o] = b[o][:, :n]
    return b


# ----------------------------------------------------------------------------- model
class ValueNet(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, ff=512, dropout=0.1, era_dropout=0.3):
        super().__init__()
        V = {k: len(v) for k, v in vocab.items()}
        self.era_dropout = era_dropout
        self.card = nn.Embedding(V["card"], d, padding_idx=0)
        self.upg = nn.Embedding(4, d)
        self.ench = nn.Embedding(V["ench"], d, padding_idx=0)
        self.card_num = nn.Linear(2, d)            # enchant amount, age
        self.relic = nn.Embedding(V["relic"], d, padding_idx=0)
        self.relic_num = nn.Linear(1, d)
        self.potion = nn.Embedding(V["potion"], d, padding_idx=0)
        self.char = nn.Embedding(5, d)
        self.asc = nn.Embedding(11, d)
        self.era = nn.Embedding(6, d)              # 0-4 eras, 5 = unknown
        self.act = nn.Embedding(4, d)
        self.room = nn.Embedding(10, d)
        self.enc = nn.Embedding(V["encounter"], d)
        self.scal = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.tok_type = nn.Embedding(6, d)             # cls, ctx, scal, card, relic, potion
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(2 * d)
        self.trunk = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.win_head = nn.Linear(d, 1)
        self.floor_head = nn.Linear(d, N_BINS)

    def forward(self, b, hide_era=False):
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
        pad = torch.cat([torch.zeros(B, 8, dtype=torch.bool, device=x.device),
                         b["d_ids"] == 0, b["r_ids"] == 0, b["p_ids"] == 0], 1)
        h = self.encoder(x, src_key_padding_mask=pad)
        valid = (~pad).unsqueeze(-1).float()
        pooled = torch.cat([h[:, 0], (h * valid).sum(1) / valid.sum(1)], -1)
        z = self.trunk(self.norm(pooled))
        return self.win_head(z).squeeze(-1), self.floor_head(z)


class ContextMLP(nn.Module):
    """Baseline without deck/relic/potion tokens: one-hot context + scalars."""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5 + 11 + 6 + 4 + 10 + 16, hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1 + N_BINS))

    def forward(self, b, hide_era=False):
        c = b["ctx"]
        era = torch.full_like(c[:, 2], 5) if hide_era else c[:, 2]
        x = torch.cat([F.one_hot(c[:, 0], 5), F.one_hot(c[:, 1], 11), F.one_hot(era, 6), F.one_hot(c[:, 3].clamp(0, 3), 4),
                       F.one_hot(c[:, 4], 10), b["scal"].clamp(-10, 10)], -1).float()
        o = self.net(x)
        return o[:, 0], o[:, 1:]


def hl_gauss(y, sigma=0.75):
    edges = torch.arange(N_BINS + 1, device=y.device, dtype=torch.float32) - 0.5
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - y.clamp(0, N_BINS - 1)[:, None]) / (sigma * math.sqrt(2))))
    p = cdf[:, 1:] - cdf[:, :-1]
    return p / p.sum(1, keepdim=True)


# ----------------------------------------------------------------------------- metrics
def _auc_simple(s, y, w):
    # weighted Mann-Whitney with ties counted half: sum over positives of (weight of negatives below + half of tied)
    neg_cum = np.cumsum(w * (1 - y))
    uniq, first, counts = np.unique(s, return_index=True, return_counts=True)
    last = first + counts - 1
    neg_below = np.r_[0, neg_cum[:-1]][first]                 # negatives strictly below the tie group
    neg_tied = neg_cum[last] - neg_below
    grp = np.repeat(np.arange(len(uniq)), counts)
    credit = neg_below[grp] + 0.5 * neg_tied[grp]
    wp, wn = (w * y).sum(), (w * (1 - y)).sum()
    return float((w * y * credit).sum() / (wp * wn))


def metrics(p, y, w=None, bins=15):
    w = np.ones_like(p) if w is None else w
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())
    brier = float((w * (p - y) ** 2).sum() / w.sum())
    b = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs((w[b == i] * (p[b == i] - y[b == i])).sum()) for i in range(bins)) / w.sum()
    return {"logloss": round(ll, 4), "auc": round(_auc_simple(*_sorted(p, y, w)), 4), "brier": round(brier, 4), "ece": round(float(ece), 4)}


def _sorted(p, y, w):
    o = np.argsort(p, kind="mergesort")
    return p[o], y[o], w[o]


@torch.no_grad()
def predict(model, data, idx, device, hide_era=False, bs=8192, scal_edit=None):
    model.eval()
    ps, fls = [], []
    for i in range(0, len(idx), bs):
        b = trim(data.batch(idx[i:i + bs], device))
        if scal_edit is not None:
            b["scal"] = scal_edit(b["scal"])
        with torch.autocast("cuda", torch.bfloat16):
            lw, lf = model(b, hide_era=hide_era)
        ps.append(torch.sigmoid(lw.float()).cpu().numpy())
        fls.append((torch.softmax(lf.float(), -1) * torch.arange(N_BINS, device=device)).sum(-1).cpu().numpy())
    return np.concatenate(ps), np.concatenate(fls)


def evaluate(model, data, split, device, hide_era=False):
    idx = np.flatnonzero(data.a["split"] == split)
    p, fl = predict(model, data, idx, device, hide_era)
    y = data.a["win"][idx].astype(np.float64); w = data.a["weight"][idx].astype(np.float64)
    act = data.a["ctx"][idx, 3]
    res = {"balanced": metrics(p, y), "population": metrics(p, y, w),
           "by_act_auc": {f"act{a+1}": round(_auc_simple(*_sorted(p[act == a], y[act == a], np.ones((act == a).sum()))), 4) for a in range(3)},
           "floors_left_mae": round(float(np.abs(fl - data.a["floors_left"][idx]).mean()), 2)}
    return res


def hp_probe(model, data, split, device, n=50000, plus=10):
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(data.a["split"] == split), n, replace=False)
    base, _ = predict(model, data, idx, device)

    def edit(s):
        s = s.clone()
        hp, mhp = s[:, 2] * 100, s[:, 3] * 100
        hp2 = torch.minimum(hp + plus, mhp)
        s[:, 2], s[:, 4] = hp2 / 100, hp2 / mhp.clamp(min=1)
        return s
    more, _ = predict(model, data, idx, device, scal_edit=edit)
    act = data.a["ctx"][idx, 3]
    room = data.a["scal"][idx, 2] * 100 < data.a["scal"][idx, 3] * 100  # had room to heal
    d = more - base
    return {"share_increasing": round(float((d[room] > 0).mean()), 4), "share_decreasing": round(float((d[room] < 0).mean()), 4),
            "mean_delta_by_act": {f"act{a+1}": round(float(d[room & (act == a)].mean()), 4) for a in range(3)}}


# ----------------------------------------------------------------------------- training
def train(model, data, args, device, name, log):
    rng = np.random.default_rng(args.seed)
    decay, no_decay = [], []
    for mod in model.modules():
        for pn, p_ in mod.named_parameters(recurse=False):
            (decay if p_.ndim >= 2 and not isinstance(mod, nn.Embedding) and pn != "cls" else no_decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil((data.run_split == 0).sum() * args.k / args.bs)
    total = steps_per_epoch * args.epochs

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))
    best, best_state, bad, step, hist = 1e9, None, 0, 0, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tl, tn = 0.0, 0
        idx = data.epoch_indices(0, args.k, rng)
        for i in range(0, len(idx), args.bs):
            b = trim(data.batch(idx[i:i + args.bs], device))
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", torch.bfloat16):
                lw, lf = model(b)
                loss = F.binary_cross_entropy_with_logits(lw.float(), b["win"]) + \
                       args.aux * -(hl_gauss(b["floors_left"]) * F.log_softmax(lf.float(), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update_parameters(model)
            tl += loss.item() * len(b["win"]); tn += len(b["win"]); step += 1
        v = evaluate(ema.module, data, 1, device)
        rec = {"model": name, "epoch": ep + 1, "train_loss": round(tl / tn, 4), "val": v, "lr": round(lr_at(step), 6), "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        log(f"[{name}] ep {ep+1:2d} train {tl/tn:.4f} | val logloss {v['balanced']['logloss']:.4f} auc {v['balanced']['auc']:.4f} "
            f"| pop-weighted auc {v['population']['auc']:.4f} | act AUC {v['by_act_auc']} | floorsMAE {v['floors_left_mae']} | {rec['sec']}s")
        if v["balanced"]["logloss"] < best - 1e-4:
            best, bad = v["balanced"]["logloss"], 0
            best_state = {k: t.detach().clone() for k, t in ema.module.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"[{name}] early stop at epoch {ep+1}")
                break
    ema.module.load_state_dict(best_state)
    return ema.module, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("floors_dir"); ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--aux", type=float, default=0.25); ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--patience", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--era-dropout", type=float, default=0.3)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"
    logf = open(os.path.join(args.out_dir, "train.log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    data = Data(args.floors_dir)
    log(f"config: {vars(args)}")
    log(f"examples: {len(data.a['win']):,}; runs train/val/test: {[(int((data.run_split == s).sum())) for s in range(3)]}")
    results = {"config": vars(args)}
    models = {}
    if not args.skip_baseline:
        base = ContextMLP().to(device)
        models["context_mlp"], results["context_mlp_history"] = train(base, data, args, device, "context_mlp", log)
    vnet = ValueNet(data.vocab, args.d, args.layers, args.heads, args.ff, args.dropout, args.era_dropout).to(device)
    log(f"ValueNet params: {sum(p.numel() for p in vnet.parameters())/1e6:.2f}M "
        f"(non-embedding {sum(p.numel() for n, p in vnet.named_parameters() if 'weight' in n and p.ndim == 2 and not any(e in n for e in ('card.', 'relic.', 'potion.', 'enc.', 'ench.', 'char.', 'asc.', 'era.', 'act.', 'room.', 'upg.', 'tok_type.')))/1e6:.2f}M)")
    models["value_transformer"], results["value_transformer_history"] = train(vnet, data, args, device, "value_transformer", log)
    # constant baseline
    tr = data.a["split"] == 0
    rate = float(data.a["win"][tr].mean())
    final = {}
    for split_name, s in (("val", 1), ("test", 2)):
        idx = np.flatnonzero(data.a["split"] == s)
        y = data.a["win"][idx].astype(np.float64); w = data.a["weight"][idx].astype(np.float64)
        final[split_name] = {"constant": {"balanced": metrics(np.full(len(y), rate), y), "population": metrics(np.full(len(y), rate), y, w)}}
        for name, m in models.items():
            final[split_name][name] = evaluate(m, data, s, device)
            final[split_name][name + "_era_hidden"] = evaluate(m, data, s, device, hide_era=True)
    results["final"] = final
    results["hp_probe_val"] = hp_probe(models["value_transformer"], data, 1, device)
    torch.save({"state_dict": models["value_transformer"].state_dict(), "config": vars(args), "vocab": data.vocab},
               os.path.join(args.out_dir, "value_transformer.pt"))
    json.dump(results, open(os.path.join(args.out_dir, "results.json"), "w"), indent=1)
    log("FINAL " + json.dumps({k: {m: (v[m]["balanced"] if "balanced" in v[m] else v[m]) for m in v} for k, v in final.items()}))
    log("HP probe " + json.dumps(results["hp_probe_val"]))


if __name__ == "__main__":
    main()
