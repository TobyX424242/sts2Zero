"""Shared-memory evaluation server, protocol v1 (client: src/s0/S0.Sim/Bridge.cs; layout described there).

Creates the bridge file, writes the header (magic last), then serves requests: spin on the request sequence number,
run CombatNet on the first n rows (GPU if available), write [policy logits (A), P(survive) logit, relative T_f]
per row, publish the response sequence number. Weights: --weights PATH is loaded at start (random init if absent)
and re-loaded between batches when its mtime changes (write a new checkpoint to a temp name and rename it over
PATH); every load bumps the model version, which is written to the header and echoed with each response.

Usage: ./.venv/bin/python -I src/s0/py/bridge_server.py [--path /dev/shm/s0_bridge] [--capacity 1024]
           [--vocab src/s0/data/combat_vocab_v1.json] [--weights PATH] [--device cuda] [--fp16]
"""
import argparse, mmap, os, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import combat_obs as co  # noqa: E402

MAGIC = 0x4547444952423053
PROTOCOL, HEADER = 1, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/dev/shm/s0_bridge")
    ap.add_argument("--capacity", type=int, default=1024)
    ap.add_argument("--vocab", default=os.path.join(os.path.dirname(HERE), "data/combat_vocab_v1.json"))
    ap.add_argument("--weights", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    B, T, TW, A, AW = a.capacity, co.MAX_TOKENS, co.TOKEN_WIDTH, co.MAX_ACTIONS, co.ACTION_WIDTH
    OW = A + 2
    obs_b, act_b, out_b = B * T * TW * 4, B * A * AW * 4, B * OW * 4
    size = HEADER + obs_b + act_b + out_b
    with open(a.path + ".tmp", "wb") as f:
        f.truncate(size)
    os.replace(a.path + ".tmp", a.path)
    fd = os.open(a.path, os.O_RDWR)
    mm = mmap.mmap(fd, size)
    hdr = np.frombuffer(mm, dtype="<i8", count=HEADER // 8)  # int64 view of the header
    obs = np.frombuffer(mm, dtype="<i4", count=B * T * TW, offset=HEADER).reshape(B, T, TW)
    act = np.frombuffer(mm, dtype="<i4", count=B * A * AW, offset=HEADER + obs_b).reshape(B, A, AW)
    out = np.frombuffer(mm, dtype="<f4", count=B * OW, offset=HEADER + obs_b + act_b).reshape(B, OW)
    struct.pack_into("<iiiiiiii", mm, 8, PROTOCOL, co.VERSION, B, T, TW, A, AW, OW)

    torch.manual_seed(0)
    net = co.CombatNet(co.load_vocab(a.vocab)).to(a.device).eval()
    version, mtime = 0, None

    def maybe_reload():
        nonlocal version, mtime
        m = os.path.getmtime(a.weights) if a.weights and os.path.exists(a.weights) else None
        if version == 0 or (m is not None and m != mtime):
            if m is not None:
                net.load_state_dict(torch.load(a.weights, map_location=a.device))
            mtime = m
            version += 1
            hdr[5] = version  # offset 40
    maybe_reload()
    hdr[0] = MAGIC  # header complete
    print(f"bridge server ready: {a.path} ({size / 1e6:.1f} MB), capacity {B}, device {a.device}, model version {version}", flush=True)

    last = int(hdr[8])  # request seq at offset 64
    served = 0
    while True:
        k = int(hdr[8])
        if k == last:
            if struct.unpack_from("<i", mm, 192)[0] == 1:
                break
            continue  # spin (one core); a production server would also poll other actors' files here
        n = struct.unpack_from("<i", mm, 72)[0]
        with torch.inference_mode(), torch.autocast(a.device, dtype=torch.float16, enabled=a.fp16 and a.device == "cuda"):
            o = torch.from_numpy(obs[:n].astype(np.int64)).to(a.device, non_blocking=True)
            ac = torch.from_numpy(act[:n].astype(np.int64)).to(a.device, non_blocking=True)
            logits, v = net(o, ac)
            res = torch.cat([logits, v], 1).float().cpu().numpy()
        out[:n] = res
        hdr[17] = version  # offset 136: version that answered
        hdr[16] = k        # offset 128: publish response (x86 TSO keeps the stores above ordered before this one)
        last = k
        served += n
        maybe_reload()
    print(f"bridge server stopped after {served} states", flush=True)
    del hdr, obs, act, out
    mm.close()
    os.close(fd)
    os.remove(a.path)


if __name__ == "__main__":
    main()
