"""Offline test of fixes for the larger combat network's value overfitting (owner, 2026-11-14; src/s3/shadow_fix.sh).

One arm = one trainer that replays a fixed stream of self-play chunks in order (as the live trainer meets them: replay
window, surprise sampling, EMA, the same reuse), optionally mixed with human fight rows, and measures the published (EMA)
weights on held-out chunks of a later segment every --eval-every steps (the losses of src/s2/py/combat_heldout.py).

Fixes it can test:
  --vw W            weight of the self-play value loss (policy loss weight stays 1)
  --fight-weight    value loss per row weighted by 1 / (decisions in its fight), renormalised to mean 1 in the batch:
                    every fight counts once for the value, however long (AlphaGo's value network, Silver et al. 2016,
                    overfit on whole games and was trained on one position per game)
  --dropout P       dropout in the transformer layers; --vdrop P: dropout on the pooled state before the value head
  --wd X            AdamW weight decay
  --human F.s2d     human rows (S2 play mode=human), --human-frac share of each batch, --human-vw their value-loss weight

usage: ./.venv/bin/python -I src/s3/py/shadow_fix.py --out DIR --stream LIST --heldout LIST [options]
  LIST: text file of chunk paths (.s2d or .s2d.zst), in stream order
"""
import argparse, copy, json, math, os, subprocess, sys, threading, time, queue

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "src", "s2", "py"))
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import train_combat as tc  # noqa: E402
from model import load_net, co, HP_BINS, POTION_SLOTS  # noqa: E402


def parse_chunk(b, path=""):
    assert b[:8] == b"S2DATA01", path
    ver, n, t, tw, a, aw, mw = np.frombuffer(b, "<i4", 7, 8)
    assert (ver, t, tw, a, aw, mw) == (2, tc.T, tc.TW, tc.A, tc.AW, tc.MW), path
    off = 64
    obs = np.frombuffer(b, "<i2", n * t * tw, off).reshape(n, t, tw); off += obs.nbytes
    act = np.frombuffer(b, "<i2", n * a * aw, off).reshape(n, a, aw); off += act.nbytes
    pol = np.frombuffer(b, "<f4", n * a, off).reshape(n, a); off += pol.nbytes
    meta = np.frombuffer(b, "<f4", n * mw, off).reshape(n, mw)
    return obs, act, pol, meta


def load(path):
    if path.endswith(".zst"):
        b = subprocess.run(["zstd", "-dcq", path], check=True, capture_output=True).stdout
    else:
        b = open(path, "rb").read()
    return parse_chunk(b, path)


def load_many(paths, mx):
    parts, n = [], 0
    for p in paths:
        c = load(p); parts.append(c); n += len(c[0])
        if n >= mx:
            break
    return tuple(np.concatenate([c[i] for c in parts])[:mx] for i in range(4))


def value_rows(v, m):
    """Per-row end-HP loss [B], potion BCE [B, slots] and its mask [B, slots] (outcome head; train_combat.value_losses)."""
    surv = m[:, 1]
    hp_logits, keep_logits = v[:, :HP_BINS], v[:, HP_BINS:HP_BINS + POTION_SLOTS]
    tgt = tc.hl_gauss(m[:, 11], surv)
    hp = -(tgt * torch.log_softmax(hp_logits, -1)).sum(-1)
    held = m[:, 9].long(); end = m[:, 10].long()
    bits = 2 ** torch.arange(POTION_SLOTS, device=v.device)
    mask = (((held[:, None] & bits) > 0) & (surv[:, None] > 0)).float()
    end_b = ((end[:, None] & bits) > 0).float()
    keep = F.binary_cross_entropy_with_logits(keep_logits, end_b, reduction="none")
    return hp, keep, mask


def wmean_value(v, m, w):
    hp, keep, mask = value_rows(v, m)
    hl = (w * hp).sum() / w.sum().clamp(min=1e-9)
    km = mask * w[:, None]
    kl = (keep * km).sum() / km.sum().clamp(min=1e-9)
    return hl + kl, hl.item(), kl.item()


def evaluate(net, H, dev):
    obs, act, pol, meta = H
    net.eval(); acc = {}; w = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for s in range(0, len(obs), 2048):
            sl = slice(s, s + 2048)
            _, _, info = tc.losses(net, obs[sl], act[sl], pol[sl], meta[sl], dev, "outcome")
            k = len(obs[sl])
            for key, val in info.items():
                acc[key] = acc.get(key, 0.0) + val * k
            w += k
    return {k: round(v / w, 5) for k, v in acc.items() if k in ("kl", "pol_acc", "hp", "survive", "hp_mae", "keep")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--heldout-max", type=int, default=400_000)
    ap.add_argument("--init", default="")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--vdrop", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reuse", type=float, default=3.2)
    ap.add_argument("--window", type=int, default=1_000_000)
    ap.add_argument("--warm", type=int, default=1_000_000, help="decisions ingested before training starts")
    ap.add_argument("--surprise", type=float, default=0.5)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--vw", type=float, default=1.0)
    ap.add_argument("--fight-weight", action="store_true")
    ap.add_argument("--human", default="")
    ap.add_argument("--human-frac", type=float, default=0.0625)
    ap.add_argument("--human-vw", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--ref", nargs="*", default=[], help="NAME=CKPT measured once on the held-out set")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    vocab = co.load_vocab(os.path.join(ROOT, "src", "s0", "data", "combat_vocab_v113.json"))
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)

    H = load_many([l.strip() for l in open(a.heldout) if l.strip()], a.heldout_max)
    emit({"heldout": len(H[0]), "policy_rows": float((H[3][:, 0] > 0).mean()), "args": vars(a)})
    for spec in a.ref:
        name, p = spec.split("=", 1)
        ck = torch.load(p, map_location=dev, weights_only=False)
        emit({"ref": name, "path": p, "version": int(ck.get("version", 0)), **evaluate(load_net(vocab, ck).to(dev), H, dev)})
    if a.max_steps == 0:
        return

    extra = {"dropout": a.dropout, "vdrop": a.vdrop}
    ck0 = None
    if a.init:
        ck0 = torch.load(a.init, map_location=dev, weights_only=False)
        net = load_net(vocab, ck0, extra).to(dev)
        if "train_state_dict" in ck0:
            net.load_state_dict(ck0["train_state_dict"])
    else:
        net = load_net(vocab, None, {"d": a.d, "layers": a.layers, "head": "outcome", **extra}).to(dev)
    ema = copy.deepcopy(net); ema.requires_grad_(False)
    if ck0 is not None:
        ema.load_state_dict(ck0["state_dict"])
    ema_p, net_p = list(ema.parameters()), list(net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    npar = sum(p.numel() for p in net.parameters())

    hum = None
    if a.human:
        hum = load_many(a.human.split(","), 10**9)
        emit({"human_rows": len(hum[0]), "human_survived": float(hum[3][:, 1].mean())})
    kh = int(round(a.batch * a.human_frac)) if hum is not None else 0
    B = a.batch - kh

    files = [l.strip() for l in open(a.stream) if l.strip()]
    q = queue.Queue(maxsize=3)
    def reader():
        for p in files:
            q.put(load(p))
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    rb = tc.Replay(a.window, surprise=a.surprise > 0)
    step, budget, t0, acc, nacc = 0, 0.0, time.time(), {}, 0
    emit({"step": 0, "t": 0, "ingested": 0, "params": npar, **evaluate(ema, H, dev)})
    done = False
    while not done:
        c = q.get()
        if c is None:
            done = True
        else:
            rb.add(*c)
            if rb.surp is not None:
                rb.refresh_surprise()
            if rb.ingested > a.warm:
                budget += a.reuse * min(len(c[0]), rb.ingested - a.warm)
        while step * B < budget and step < a.max_steps:
            net.train()
            obs, act, pol, meta, idx, _ = rb.sample(B, rng, a.surprise, a.pool)
            if kh:
                j = rng.integers(0, len(hum[0]), kh)
                obs, act, pol, meta = (np.concatenate([x, h[j]]) for x, h in zip((obs, act, pol, meta), hum))
            o = torch.from_numpy(obs).to(dev, non_blocking=True).long()
            ac = torch.from_numpy(act).to(dev, non_blocking=True).long()
            p = torch.from_numpy(pol).to(dev, non_blocking=True)
            m = torch.from_numpy(meta).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits, v = net(o, ac)
            logits, v = logits.float(), v.float()
            has = m[:, 0] > 0
            logp = torch.log_softmax(logits, -1)
            ce = -torch.where(p > 0, p * logp, torch.zeros_like(logp)).sum(-1)
            ent = -torch.where(p > 0, p * torch.log(p.clamp(min=1e-12)), torch.zeros_like(p)).sum(-1)
            pol_loss = ce[has].mean()
            vs, ms = v[:B], m[:B]
            w = (1.0 / ms[:, 7].clamp(min=1)) if a.fight_weight else torch.ones(B, device=dev)
            w = w / w.mean()
            v_sp, hp_sp, keep_sp = wmean_value(vs, ms, w)
            loss = pol_loss + a.vw * v_sp
            if kh:
                v_h, hp_h, _ = wmean_value(v[B:], m[B:], torch.ones(kh, device=dev))
                loss = loss + a.human_vw * v_h
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); step += 1; continue
            if rb.surp is not None:
                k = (ce - ent)[:B].detach().float().cpu().numpy()
                ok = meta[:B, 0] > 0
                ok &= np.isfinite(k)
                rb.surp[idx[ok]] = np.maximum(k[ok], 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema_p, net_p, 1.0 - a.ema)
            step += 1
            if rb.surp is not None and step % 100 == 0:
                rb.refresh_surprise()
            for key, val in {"tr_kl": (ce - ent)[has].mean().item(), "tr_hp": hp_sp, "tr_keep": keep_sp,
                             **({"tr_hp_human": hp_h} if kh else {})}.items():
                acc[key] = acc.get(key, 0.0) + val
            nacc += 1
            if step % a.eval_every == 0:
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
                emit({"step": step, "t": round(time.time() - t0), "ingested": rb.ingested,
                      **{k: round(v_ / nacc, 5) for k, v_ in acc.items()}, **evaluate(ema, H, dev)})
                acc, nacc = {}, 0
    for be, bn in zip(ema.buffers(), net.buffers()):
        be.copy_(bn)
    emit({"step": step, "t": round(time.time() - t0), "ingested": rb.ingested, "final": True,
          **({k: round(v_ / nacc, 5) for k, v_ in acc.items()} if nacc else {}), **evaluate(ema, H, dev)})
    torch.save({"state_dict": ema.state_dict(), "train_state_dict": net.state_dict(), "config": net.config, "version": step},
               os.path.join(a.out, "final.pt"))


if __name__ == "__main__":
    main()
