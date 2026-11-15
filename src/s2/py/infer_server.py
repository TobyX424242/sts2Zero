"""Multi-actor inference server for S2 (bridge protocol v1, client src/s0/S0.Sim/Bridge.cs).

Creates one shared-memory file per actor (<prefix><i>), each with the header of src/s0/py/bridge_server.py.
Loop: poll every actor's request sequence number; concatenate all ready requests into one batch, pad it to a fixed
bucket size (so CUDA graphs / torch.compile see few shapes), run CombatNet once, write each actor's rows back and
publish its response. Weights are re-loaded between batches when the --weights file changes (write + rename).

Usage: python -I src/s2/py/infer_server.py --prefix /dev/shm/s2_a --actors 16 [--capacity 256]
           [--weights src/s2/runs/X/weights.pt] [--compile] [--dtype bf16|fp16|fp32]
"""
import argparse, mmap, os, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "s0", "py"))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import combat_obs as co  # noqa: E402
from model import load_net  # noqa: E402

MAGIC = 0x4547444952423053
PROTOCOL, HEADER = 1, 256
BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
T_BUCKETS = [48, 64, 96, 128]  # token rows actually used (tokens are padded at the end of each state)


class Actor:
    def __init__(self, path, B, n_value):
        T, TW, A, AW = co.MAX_TOKENS, co.TOKEN_WIDTH, co.MAX_ACTIONS, co.ACTION_WIDTH
        OW = A + n_value
        obs_b, act_b, out_b = B * T * TW * 4, B * A * AW * 4, B * OW * 4
        size = HEADER + obs_b + act_b + out_b
        with open(path + ".tmp", "wb") as f:
            f.truncate(size)
        os.replace(path + ".tmp", path)
        self.path = path
        self.fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, size)
        self.hdr = np.frombuffer(self.mm, dtype="<i8", count=HEADER // 8)
        self.obs = np.frombuffer(self.mm, dtype="<i4", count=B * T * TW, offset=HEADER).reshape(B, T, TW)
        self.act = np.frombuffer(self.mm, dtype="<i4", count=B * A * AW, offset=HEADER + obs_b).reshape(B, A, AW)
        self.out = np.frombuffer(self.mm, dtype="<f4", count=B * OW, offset=HEADER + obs_b + act_b).reshape(B, OW)
        struct.pack_into("<iiiiiiii", self.mm, 8, PROTOCOL, co.VERSION, B, T, TW, A, AW, OW)
        self.last = 0
        self.done = False

    def ready(self):
        k = int(self.hdr[8])
        return k != self.last

    def n(self):
        return struct.unpack_from("<i", self.mm, 72)[0]

    def shutdown_flag(self):
        return struct.unpack_from("<i", self.mm, 192)[0] == 1

    def close(self):
        del self.hdr, self.obs, self.act, self.out
        self.mm.close(); os.close(self.fd)
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="/dev/shm/s2_a")
    ap.add_argument("--actors", type=int, default=1)
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--vocab", default=os.path.join(HERE, "..", "..", "s0", "data", "combat_vocab_v1.json"))
    ap.add_argument("--weights", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--head", default="outcome", help="value head when --weights does not exist yet (outcome|scalar)")
    ap.add_argument("--stats-every", type=float, default=60.0)
    ap.add_argument("--idle-exit", type=float, default=300.0, help="exit after this many seconds without any request "
                    "(an actor that died never sets its shutdown flag; without this the pipeline would wait forever)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = a.device
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    vocab = co.load_vocab(a.vocab)
    net, version, mtime = None, 0, None

    def maybe_reload(force=False):
        nonlocal net, version, mtime
        m = os.path.getmtime(a.weights) if a.weights and os.path.exists(a.weights) else None
        if force or (m is not None and m != mtime):
            if m is not None:
                for _ in range(5):
                    try:
                        sd = torch.load(a.weights, map_location=dev, weights_only=False)
                        break
                    except Exception:
                        time.sleep(0.05)
                else:
                    return
                version = int(sd.get("version", version + 1)) if isinstance(sd, dict) and "state_dict" in sd else version + 1
                if net is None:
                    net = load_net(vocab, sd).to(dev).eval()
                else:
                    net.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
            elif net is None:
                torch.manual_seed(0)
                net = load_net(vocab, None, {"head": a.head}).to(dev).eval()
                version = 1
            mtime = m
            for ac in actors:
                ac.hdr[5] = version
    actors = []
    maybe_reload(force=True)
    n_value = net.value[-1].out_features
    actors = [Actor(f"{a.prefix}{i}", a.capacity, n_value) for i in range(a.actors)]
    for ac in actors:
        ac.hdr[5] = version
    fwd = net.forward
    if a.compile:
        fwd = torch.compile(net.forward, mode="reduce-overhead", dynamic=False)
    T, TW, A, AW = co.MAX_TOKENS, co.TOKEN_WIDTH, co.MAX_ACTIONS, co.ACTION_WIDTH
    # staging buffers hold the largest padded batch: the total capacity rounded up to its bucket (sizing them to the
    # capacity itself crashed when a batch was padded past it, 2026-11-11)
    maxB = next((b for b in BUCKETS if b >= a.capacity * a.actors), a.capacity * a.actors)
    pin = dev == "cuda"
    h_obs_flat = torch.zeros(maxB * T * TW, dtype=torch.int32)
    h_obs_flat = h_obs_flat.pin_memory() if pin else h_obs_flat
    h_act = torch.zeros((maxB, A, AW), dtype=torch.int32).pin_memory() if pin else torch.zeros((maxB, A, AW), dtype=torch.int32)
    n_obs_flat, n_act = h_obs_flat.numpy(), h_act.numpy()
    for ac in actors:
        ac.hdr[0] = MAGIC
    print(f"infer server ready: {a.actors} actors at {a.prefix}0..; capacity {a.capacity} each; device {dev} {a.dtype}; "
          f"compile {a.compile}; model version {version}; value head {net.config['head']} ({n_value} outputs)", flush=True)
    served = batches = 0
    t_gpu = t_gather = t_scatter = 0.0
    t0 = t_last = t_req = time.time()
    idle_spins = 0
    live = len(actors)
    while live > 0:
        ready = [ac for ac in actors if not ac.done and ac.ready()]
        if not ready and time.time() - t_req > a.idle_exit:
            print(f"[infer] no request for {a.idle_exit:.0f} s from {live} live actors: exiting", flush=True)
            break
        if not ready:
            for ac in actors:
                if not ac.done and ac.shutdown_flag():
                    ac.done = True
                    live -= 1
            idle_spins += 1
            if idle_spins > 2000:
                time.sleep(0.0002)
            continue
        idle_spins = 0
        tq = t_req = time.time()
        seqs, ns, off, used = [], [], 0, 1
        for ac in ready:
            k = int(ac.hdr[8]); n = ac.n()
            seqs.append(k); ns.append(n)
            used = max(used, int((ac.obs[:n, :, 0] != 0).sum(1).max()))
            off += n
        Bp = next((b for b in BUCKETS if b >= off), off)
        Tp = next(t for t in T_BUCKETS if t >= used)
        v_obs = n_obs_flat[:Bp * Tp * TW].reshape(Bp, Tp, TW)   # contiguous staging for this batch shape
        v_act = n_act[:Bp]
        off = 0
        for ac, n in zip(ready, ns):
            v_obs[off:off + n] = ac.obs[:n, :Tp]; v_act[off:off + n] = ac.act[:n]
            off += n
        if Bp > off:
            v_obs[off:Bp] = 0; v_act[off:Bp] = 0
        tg = time.time()
        t_gather += tg - tq
        with torch.inference_mode(), torch.autocast("cuda", dtype=dt, enabled=dev == "cuda" and dt != torch.float32):
            o = h_obs_flat[:Bp * Tp * TW].view(Bp, Tp, TW).to(dev, non_blocking=True).long()
            ac_ = h_act[:Bp].to(dev, non_blocking=True).long()
            o[off:, 0, 0] = 1  # padded rows: one global token so pooling is defined
            logits, v = fwd(o, ac_)
            res = torch.cat([logits.float(), v.float()], 1)[:off].cpu().numpy()
        tw = time.time()
        t_gpu += tw - tg
        off = 0
        for ac, k, n in zip(ready, seqs, ns):
            ac.out[:n] = res[off:off + n]
            off += n
            ac.hdr[17] = version
            ac.hdr[16] = k  # publish (x86 TSO keeps the stores above ordered before this one)
            ac.last = k
        served += off; batches += 1
        now = time.time()
        t_scatter += now - tw
        if now - t_last > a.stats_every:
            print(f"[infer] {now - t0:.0f}s served {served} states in {batches} batches (mean {served / max(1, batches):.0f}), "
                  f"{served / (now - t0):.0f} states/s, busy: gather {100 * t_gather / (now - t0):.0f}% GPU+copy {100 * t_gpu / (now - t0):.0f}% "
                  f"scatter {100 * t_scatter / (now - t0):.0f}%, model version {version}", flush=True)
            t_last = now
        maybe_reload()
    print(f"infer server stopped after {served} states, {batches} batches", flush=True)
    for ac in actors:
        ac.close()


if __name__ == "__main__":
    main()
