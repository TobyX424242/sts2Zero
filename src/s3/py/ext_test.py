"""Layout-v4 / ext network check (CPU, a few seconds; no GPU, no server).

Usage: python -I src/s3/py/ext_test.py WEIGHTS.pt CHUNK.s2d [--n 256] [--vocab src/s0/data/combat_vocab_v1.json]
  WEIGHTS: a v3 combat checkpoint (e.g. src/s3/weights/s037/weights.pt); CHUNK: a training chunk written by v3 actors.

(a) blindness and identity at init, on n real decisions of the chunk:
    a0  the chunk's layout-v4 slots (combat_obs.V4_SLOTS) are all 0 (v3 data);
    a1  old network, new code == the same network with the mask switched off (= the code before layout v4), bit for bit;
    a2  old network on the batch with the v4 slots filled with random large values == on the original batch (blind);
    a3  ext network (v3 checkpoint + config ext=1) == old network, on the original and on the filled batch, bit for bit;
    a4  checkpoint round trip: an ext checkpoint carries "ext": 1 and rebuilds the ext network from the file alone (same
        outputs); an ext checkpoint does not load into a v3 network (strict).
(b) one AdamW step of the ext network (trainer losses, lr 2e-4, grad clip 1) on the filled batch: ext_feat's last layer
    becomes non-zero and the outputs on the filled batch change (it can learn); the old network's parameters and outputs
    are unchanged.
Prints one line per check and PASS / FAIL; exit code 1 on failure.
"""
import argparse, copy, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "s0", "py")); sys.path.insert(0, os.path.join(HERE, "..", "..", "s2", "py"))
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import combat_obs as co  # noqa: E402
from model import load_net  # noqa: E402
from train_combat import read_chunk, losses  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("chunk")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--vocab", default=os.path.join(HERE, "..", "..", "s0", "data", "combat_vocab_v1.json"))
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vocab = co.load_vocab(a.vocab)
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    assert int(ck.get("config", {}).get("ext", 0)) == 0, "give a v3 (non-ext) checkpoint"
    obs16, act16, pol, meta = read_chunk(a.chunk)
    n = min(a.n, len(obs16))
    obs = torch.from_numpy(obs16[:n].astype(np.int64)); act = torch.from_numpy(act16[:n].astype(np.int64))
    pol, meta = np.ascontiguousarray(pol[:n]), np.ascontiguousarray(meta[:n])
    fails = []

    def check(ok, name, extra=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + extra) if extra else ''}", flush=True)
        if not ok:
            fails.append(name)

    def fwd(net, o):
        net.eval()
        with torch.no_grad():
            return net(o, act)

    def same(x, y):
        return all(torch.equal(p, q) for p, q in zip(x, y))

    def diff(x, y):
        return max(float((p - q).abs().masked_fill(~torch.isfinite(p - q), 0).max()) for p, q in zip(x, y))

    # the batch with every v4 slot of its kind filled with random large values (int16 range, as in the chunks)
    filled = obs.clone()
    kind = obs[..., 0]
    for k, slots in co.V4_SLOTS.items():
        m = kind == k
        for f in slots:
            col = filled[..., 3 + f]
            col[m] = torch.from_numpy(rng.integers(-30000, 30001, int(m.sum()))).long()
    n_filled = int(sum(int((kind == k).sum()) * len(s) for k, s in co.V4_SLOTS.items()))

    # a0
    nz = sum(int((obs[..., 3 + f][kind == k] != 0).sum()) for k, s in co.V4_SLOTS.items() for f in s)
    check(nz == 0, "a0 v4 slots of the chunk are 0", f"{nz} non-zero values, {n} decisions, {n_filled} v4 slots filled for a2")

    old = load_net(vocab, ck)
    out_old = fwd(old, obs)
    # a1: mask switched off = the pre-v4 embedding (features used as stored)
    ref = copy.deepcopy(old)
    ref.fmask.fill_(1)
    out_ref = fwd(ref, obs)
    check(same(out_old, out_ref), "a1 old network: new code == old code on v3 data", f"max diff {diff(out_old, out_ref):.3g}")
    out_old_f = fwd(old, filled)
    check(same(out_old, out_old_f), "a2 old network blind to the v4 slots", f"max diff {diff(out_old, out_old_f):.3g}")
    out_ref_f = fwd(ref, filled)
    check(not same(out_ref, out_ref_f), "a2' control: without the mask the filled slots change the outputs",
          f"max diff {diff(out_ref, out_ref_f):.3g}")

    # a3
    ext = load_net(vocab, ck, {"ext": 1})
    check(ext.config.get("ext") == 1 and hasattr(ext, "ext_feat"), "a3 ext network built from the v3 checkpoint", str(ext.config))
    check(float(ext.ext_feat[2].weight.abs().sum()) == 0 and float(ext.ext_feat[2].bias.abs().sum()) == 0,
          "a3 ext_feat last layer zero at init")
    out_ext = fwd(ext, obs)
    out_ext_f = fwd(ext, filled)
    check(same(out_ext, out_old), "a3 ext == old at init (v3 batch)", f"max diff {diff(out_ext, out_old):.3g}")
    check(same(out_ext_f, out_old), "a3 ext == old at init (filled batch)", f"max diff {diff(out_ext_f, out_old):.3g}")

    # a4: checkpoint round trip
    bio = io.BytesIO()
    torch.save({"state_dict": ext.state_dict(), "config": ext.config, "version": 1}, bio)
    bio.seek(0)
    ck_ext = torch.load(bio, map_location="cpu", weights_only=False)
    re = load_net(vocab, ck_ext)
    check(ck_ext["config"].get("ext") == 1 and getattr(re, "ext", 0) == 1 and same(fwd(re, filled), out_ext_f),
          "a4 ext checkpoint rebuilds the ext network from the file alone")
    try:
        load_net(vocab, ck_ext, {"ext": 0})
        check(False, "a4 ext checkpoint refused by a v3 network (strict)")
    except RuntimeError:
        check(True, "a4 ext checkpoint refused by a v3 network (strict)")
    check("ext" not in old.config and not any(k.startswith("ext_feat") for k in old.state_dict()),
          "a4 v3 network: config and state dict unchanged (no ext keys)")

    # (b) one optimizer step of the ext network on the filled batch
    p_old = [p.detach().clone() for p in old.parameters()]
    ext.train()
    opt = torch.optim.AdamW(ext.parameters(), lr=2e-4, weight_decay=1e-4)
    head = ext.config["head"]
    pl, vl, info = losses(ext, filled.numpy().astype(np.int16), act16[:n], pol, meta, "cpu", head)
    opt.zero_grad(set_to_none=True)
    (pl + vl).backward()
    g_last = float(ext.ext_feat[2].weight.grad.abs().sum())
    torch.nn.utils.clip_grad_norm_(ext.parameters(), 1.0)
    opt.step()
    w_last = float(ext.ext_feat[2].weight.abs().sum())
    check(g_last > 0 and w_last > 0, "b ext_feat last layer non-zero after one step",
          f"|grad| {g_last:.3g}, |weight| {w_last:.3g}, loss {float(pl + vl):.4f}")
    out_ext_f1 = fwd(ext, filled)
    check(not same(out_ext_f1, out_ext_f), "b ext outputs on the filled batch changed", f"max diff {diff(out_ext_f1, out_ext_f):.3g}")
    check(all(torch.equal(p, q) for p, q in zip(p_old, old.parameters())), "b old network parameters unchanged")
    check(same(fwd(old, obs), out_old) and same(fwd(old, filled), out_old), "b old network outputs unchanged")

    print("PASS" if not fails else f"FAIL ({len(fails)}: {', '.join(fails)})")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
