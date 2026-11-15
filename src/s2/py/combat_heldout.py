"""Held-out losses of combat checkpoints (IMPROVEMENT_PLAN.md change C): the same losses as train_combat.py, measured
on chunks no trainer learned from (train_combat.py --skip-every K holds out chunks whose sequence number % K == 0).

usage: python -I research/s2/py/combat_heldout.py --data DIR --skip-every K [--max N] ARM=CKPT_OR_DIR [ARM=...]
  CKPT_OR_DIR: a checkpoint (.pt) or a directory of checkpoints (every v*.pt in it, plus weights.pt if present)
Prints one line per checkpoint: arm, version, steps of the trainer if known, policy KL to the search target, top-1
agreement, end-HP loss, survival log loss, end-HP mean abs error, potion-kept loss.
"""
import glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_combat as tc  # noqa: E402
from model import load_net, co  # noqa: E402


def main():
    args = sys.argv[1:]
    def opt(k, d):
        if k in args:
            i = args.index(k); v = args[i + 1]; del args[i:i + 2]; return v
        return d
    data, K, mx = opt("--data", None), int(opt("--skip-every", "10")), int(opt("--max", "400000"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(p for p in glob.glob(os.path.join(data, "*.s2d")) if K <= 0 or tc.chunk_seq(p) % K == 0)
    parts, n = [], 0
    for p in files:
        c = tc.read_chunk(p); parts.append(c); n += len(c[0])
        if n >= mx:
            break
    obs, act, pol, meta = (np.concatenate([c[i] for c in parts])[:mx] for i in range(4))
    print(f"held-out: {len(obs):,} decisions from {len(parts)} chunks ({len(files)} eligible); policy targets on "
          f"{(meta[:, 0] > 0).mean():.1%}", flush=True)
    vocab = co.load_vocab(os.path.join(HERE, "..", "..", "s0", "data", "combat_vocab_v1.json"))
    print(f"{'arm':12s} {'ver':>6s} {'params':>8s} {'pol_kl':>7s} {'acc':>6s} {'hp':>7s} {'surv':>7s} {'hp_mae':>7s} {'keep':>7s}")
    for spec in args:
        arm, path = spec.split("=", 1)
        cks = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "v*.pt")))
        for ck_path in cks:
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
            net = load_net(vocab, ck).to(dev).eval()
            acc = {}; w = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                for s in range(0, len(obs), 2048):
                    sl = slice(s, s + 2048)
                    _, _, info = tc.losses(net, obs[sl], act[sl], pol[sl], meta[sl], dev, net.config["head"])
                    k = len(obs[sl])
                    for key, v in info.items():
                        acc[key] = acc.get(key, 0.0) + v * k
                    w += k
            m = {k: v / w for k, v in acc.items()}
            npar = sum(p.numel() for p in net.parameters())
            print(f"{arm:12s} {int(ck.get('version', 0)):6d} {npar / 1e6:7.2f}M {m['kl']:7.4f} {m['pol_acc']:6.3f} "
                  f"{m.get('hp', float('nan')):7.4f} {m['survive']:7.4f} {m.get('hp_mae', float('nan')):7.4f} "
                  f"{m.get('keep', float('nan')):7.4f}", flush=True)


if __name__ == "__main__":
    main()
