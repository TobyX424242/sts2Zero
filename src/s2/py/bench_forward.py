"""Forward-pass time of CombatNet per batch size: eager bf16 vs torch.compile (CUDA graphs), on real v2 states."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import numpy as np, torch
from model import load_net, co
vocab = co.load_vocab(os.path.join(HERE, "../../s0/data/combat_vocab_v1.json"))
obs, act = co.load_arrays(os.path.join(HERE, "../../s0/data/obs_v3_1000"))
net = load_net(vocab).cuda().eval()
def bench(fn, B, reps=50):
    i = np.arange(B) % len(obs)
    o = torch.from_numpy(obs[i]).cuda().long(); a = torch.from_numpy(act[i]).cuda().long()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5): fn(o, a)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(reps): fn(o, a)
        torch.cuda.synchronize()
    return (time.time() - t) / reps * 1e3
mode = sys.argv[1] if len(sys.argv) > 1 else "eager"
fn = net.forward if mode == "eager" else torch.compile(net.forward, mode=mode, dynamic=False)
for B in [64, 256, 1024, 2048]:
    ms = bench(fn, B)
    print(f"{mode:16s} batch {B:5d}: {ms:6.2f} ms, {B / ms * 1e3:8.0f} states/s", flush=True)
