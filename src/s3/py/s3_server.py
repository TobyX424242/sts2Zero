"""S3 inference server: the combat network and the macro ensemble for many actor processes, on one GPU.

Combat half: exactly src/s2/py/infer_server.py (bridge protocol v1, files <prefix><i>). Macro half: files
<mprefix><i> with the layout of src/s3/S3/MacroBridge.cs; rows as in macro_s3.py. One loop polls both kinds of
request; all ready combat requests go into one padded batch, all ready macro requests into another. Weights are
re-loaded when the --weights (combat) or --macro (ensemble) file changes.

GPU options for the combat half (off by default; with both off the combat loop is the original code, unchanged):
  --graphs 1    CUDA graph per (batch bucket, token bucket), captured at its first use and replayed (combat_pipe.py)
  --overlap 1   two batches in flight: pinned host buffers, copies on a side stream, results through CUDA events; the
                loop gathers / scatters one batch while the GPU computes the other. An actor whose request is in flight
                is not polled again until it has been answered (one request in flight per actor, as the bridge allows)
  --overlap-min K   launch a batch while another is in flight only if it has at least K states (default 1)

Weights of another structure (2026-11-15: e.g. a checkpoint with config "ext": 1 replacing a v3 one, or back): the combat
network is rebuilt from the checkpoint (combat_pipe.CombatPipe.replace_net: captured graphs are dropped and captured
again); same structure: loaded in place as before. A checkpoint with another output width (value head) is refused.

Usage: python -I src/s3/py/s3_server.py --prefix /dev/shm/s3_c --mprefix /dev/shm/s3_m --actors 8
           --weights src/s2/runs/X/weights.pt --macro src/s3/data/macro_s1.pt [--capacity 64] [--mcapacity 256]
"""
import argparse, mmap, os, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
S2PY = os.path.join(HERE, "..", "..", "s2", "py")
sys.path.insert(0, os.path.join(HERE, "..", "..", "s0", "py")); sys.path.insert(0, S2PY); sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import combat_obs as co  # noqa: E402
import infer_server as cs  # noqa: E402  (Actor, BUCKETS, T_BUCKETS)
import macro_s3 as ms  # noqa: E402
from model import load_net, config_of, arch  # noqa: E402

MMAGIC = 0x314F5243414D3353  # "S3MACRO1"
MBUCKETS = [8, 32, 128, 512, 2048, 8192]


class MacroActor:
    def __init__(self, path, B):
        ib, fb, ob = B * ms.IW * 4, B * ms.FW * 4, B * ms.OW * 4
        size = cs.HEADER + ib + fb + ob
        with open(path + ".tmp", "wb") as f:
            f.truncate(size)
        os.replace(path + ".tmp", path)
        self.path = path
        self.fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, size)
        self.hdr = np.frombuffer(self.mm, dtype="<i8", count=cs.HEADER // 8)
        self.ints = np.frombuffer(self.mm, dtype="<i4", count=B * ms.IW, offset=cs.HEADER).reshape(B, ms.IW)
        self.floats = np.frombuffer(self.mm, dtype="<f4", count=B * ms.FW, offset=cs.HEADER + ib).reshape(B, ms.FW)
        self.out = np.frombuffer(self.mm, dtype="<f4", count=B * ms.OW, offset=cs.HEADER + ib + fb).reshape(B, ms.OW)
        struct.pack_into("<iiii", self.mm, 8, B, ms.IW, ms.FW, ms.OW)
        self.last = 0
        self.done = False

    def ready(self):
        return int(self.hdr[8]) != self.last

    def n(self):
        return struct.unpack_from("<i", self.mm, 72)[0]

    def shutdown_flag(self):
        return struct.unpack_from("<i", self.mm, 192)[0] == 1

    def close(self):
        del self.hdr, self.ints, self.floats, self.out
        self.mm.close(); os.close(self.fd)
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="/dev/shm/s3_c"); ap.add_argument("--mprefix", default="/dev/shm/s3_m")
    ap.add_argument("--actors", type=int, default=1)
    ap.add_argument("--capacity", type=int, default=64); ap.add_argument("--mcapacity", type=int, default=256)
    ap.add_argument("--vocab", default=os.path.join(HERE, "..", "..", "s0", "data", "combat_vocab_v1.json"))
    ap.add_argument("--weights", required=True, help="combat network checkpoint (src/s2/py/model.py)")
    ap.add_argument("--macro", required=True, help="macro ensemble file (macro_s3.Ensemble)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--serve", default="both", choices=["both", "combat", "macro"],
                    help="which half this process serves (run one 'combat' and one 'macro' process so that macro "
                         "batches never delay combat batches; the unserved half's files are not created)")
    ap.add_argument("--first", type=int, default=0, help="index of this process's first actor (several combat processes "
                    "share the actors: --first 0 --actors 14 and --first 14 --actors 14)")
    ap.add_argument("--min-period", type=float, default=0.0, help="macro process: at least this many seconds between "
                    "batches, so that requests of several actors gather into one batch (actors do not block on them)")
    ap.add_argument("--stats-every", type=float, default=60.0)
    ap.add_argument("--idle-exit", type=float, default=300.0)
    ap.add_argument("--graphs", type=int, default=0, choices=[0, 1], help="combat: CUDA graphs per (batch, token) bucket")
    ap.add_argument("--overlap", type=int, default=0, choices=[0, 1], help="combat: two batches in flight")
    ap.add_argument("--overlap-min", type=int, default=1, help="combat with --overlap: least states of a batch launched "
                    "while another is in flight")
    ap.add_argument("--dump", default="", help="test aid: write the first --dump-batches combat batches (inputs and outputs) "
                    "to this .npz file")
    ap.add_argument("--dump-batches", type=int, default=200)
    ap.add_argument("--dump-every", type=int, default=1, help="with --dump: keep every K-th batch")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = a.device
    if dev == "cpu":
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    vocab = co.load_vocab(a.vocab)
    state = {"net": None, "cver": 0, "cm": None, "ens": None, "mm": None, "pipe": None}

    def load(path):
        for _ in range(20):
            try:
                return torch.load(path, map_location=dev, weights_only=False)
            except Exception:
                time.sleep(0.05)
        return None

    def maybe_reload():
        m = os.path.getmtime(a.weights) if os.path.exists(a.weights) else None
        if m is not None and m != state["cm"]:
            sd = load(a.weights)
            if sd is not None:
                if state["net"] is None:
                    state["net"] = load_net(vocab, sd).to(dev).eval()
                elif arch(config_of(sd)) != arch(state["net"].config):
                    new = load_net(vocab, sd).to(dev).eval()
                    if new.value[-1].out_features != state["net"].value[-1].out_features:
                        print(f"[s3srv] {a.weights}: value head width differs from the served network: not loaded", flush=True)
                        state["cm"] = m
                        return
                    if state["pipe"] is not None:
                        state["pipe"].replace_net(new)     # drains the GPU, drops the captured graphs
                    state["net"] = new
                    print(f"[s3srv] combat network rebuilt for config {new.config}", flush=True)
                elif state["pipe"] is not None:
                    # batches in flight finish with the old weights (they carry their version); the copy goes into the
                    # same parameter tensors, so captured graphs read the new weights at their next replay
                    state["pipe"].drain_gpu()
                    state["net"].load_state_dict(sd["state_dict"])
                    torch.cuda.synchronize()
                else:
                    state["net"].load_state_dict(sd["state_dict"])
                state["cver"] = int(sd.get("version", state["cver"] + 1)); state["cm"] = m
        m = os.path.getmtime(a.macro) if os.path.exists(a.macro) else None
        if m is not None and m != state["mm"]:
            try:
                if state["ens"] is None:
                    state["ens"] = ms.Ensemble.load(a.macro, dev)
                else:
                    state["ens"].load_members_from(a.macro)
                state["mm"] = m
            except Exception as e:  # file being replaced: retry at the next poll
                print(f"[s3srv] macro reload failed ({type(e).__name__}), retrying", flush=True)
    maybe_reload()
    assert state["net"] is not None and state["ens"] is not None, "weights not found"
    net, ens = state["net"], state["ens"]
    n_value = net.value[-1].out_features
    cact = [cs.Actor(f"{a.prefix}{a.first + i}", a.capacity, n_value) for i in range(a.actors)] if a.serve != "macro" else []
    mact = [MacroActor(f"{a.mprefix}{a.first + i}", a.mcapacity) for i in range(a.actors)] if a.serve != "combat" else []
    T, TW, A, AW = co.MAX_TOKENS, co.TOKEN_WIDTH, co.MAX_ACTIONS, co.ACTION_WIDTH
    maxB = next((b for b in cs.BUCKETS if b >= a.capacity * a.actors), a.capacity * a.actors)
    pin = dev == "cuda"
    h_obs_flat = torch.zeros(maxB * T * TW, dtype=torch.int32)
    h_act = torch.zeros((maxB, A, AW), dtype=torch.int32)
    if pin:
        h_obs_flat, h_act = h_obs_flat.pin_memory(), h_act.pin_memory()
    n_obs_flat, n_act = h_obs_flat.numpy(), h_act.numpy()
    mB = next((b for b in MBUCKETS if b >= a.mcapacity * a.actors), a.mcapacity * a.actors)
    m_ints = np.zeros((mB, ms.IW), np.int32); m_floats = np.zeros((mB, ms.FW), np.float32)
    pipe = None
    if (a.graphs or a.overlap) and cact:
        if dev != "cuda":
            print("[s3srv] --graphs/--overlap need CUDA: ignored", flush=True)
        else:
            from combat_pipe import CombatPipe
            pipe = state["pipe"] = CombatPipe(net, dev, dt, maxB, cact[0].out.shape[1], graphs=bool(a.graphs),
                                              depth=2 if a.overlap else 1)
    dump = {"obs": [], "act": [], "out": []} if a.dump and cact else None
    for ac in cact:
        ac.hdr[5] = state["cver"]; ac.hdr[0] = cs.MAGIC
    for ac in mact:
        ac.hdr[5] = ens.version; ac.hdr[0] = MMAGIC
    print(f"s3 server ready ({a.serve}): actors {a.first}..{a.first + a.actors - 1}; combat {a.prefix}* capacity {a.capacity} (model v{state['cver']}, head "
          f"{net.config['head']}); macro {a.mprefix}* capacity {a.mcapacity} ({len(ens.nets)} members, v{ens.version}); device {dev} {a.dtype}"
          + (f"; combat graphs {a.graphs}, overlap {a.overlap} (min {a.overlap_min})" if pipe is not None else ""), flush=True)
    served = batches = mserved = mbatches = 0

    def add_dump(obs, act, res):
        dump["seen"] = dump.get("seen", -1) + 1
        if len(dump["obs"]) < a.dump_batches and dump["seen"] % a.dump_every == 0:
            dump["obs"].append(obs.copy()); dump["act"].append(act.copy()); dump["out"].append(np.array(res))
            if len(dump["obs"]) == a.dump_batches:
                np.savez(a.dump, n=np.array([len(x) for x in dump["obs"]]), tp=np.array([x.shape[1] for x in dump["obs"]]),
                         **{f"{k}{i}": x for k in ("obs", "act", "out") for i, x in enumerate(dump[k])})
                print(f"[s3srv] dumped {a.dump_batches} combat batches to {a.dump}", flush=True)

    def scatter(res, user):
        rdy, seqs, ns, ver = user
        off = 0
        for ac, k, n in zip(rdy, seqs, ns):
            ac.out[:n] = res[off:off + n]; off += n
            ac.hdr[17] = ver
            ac.hdr[16] = k                 # publish (x86 TSO keeps the stores above ordered before this one)
        return off

    def pipe_step(rdy):
        """--graphs / --overlap: launch the ready requests if a slot is free, answer the oldest batch when it is done
        (or at once when no slot is free). Returns the number of states answered."""
        nonlocal batches
        if rdy and len(pipe.inflight) < pipe.depth:
            ns = [ac.n() for ac in rdy]
            if not pipe.inflight or sum(ns) >= a.overlap_min:
                seqs, off, used = [], 0, 1
                for ac, n in zip(rdy, ns):
                    seqs.append(int(ac.hdr[8]))
                    used = max(used, int((ac.obs[:n, :, 0] != 0).sum(1).max()))
                    off += n
                Bp = next((b for b in cs.BUCKETS if b >= off), off)
                Tp = next(t for t in cs.T_BUCKETS if t >= used)
                v_obs, v_act = pipe.views(pipe.slot(), Bp, Tp)
                off = 0
                for ac, n in zip(rdy, ns):
                    v_obs[off:off + n] = ac.obs[:n, :Tp]; v_act[off:off + n] = ac.act[:n]
                    off += n
                if Bp > off:
                    v_obs[off:Bp] = 0; v_obs[off:Bp, 0, 0] = 1; v_act[off:Bp] = 0   # padded rows: one global token
                for ac, k in zip(rdy, seqs):
                    ac.last = k            # in flight: not polled again until answered (the client waits for it)
                pipe.launch(Bp, Tp, off, (rdy, seqs, ns, state["cver"]))
                if dump is not None:
                    dump_q.append((v_obs[:off].copy(), v_act[:off].copy()))
        done = 0
        if pipe.inflight and (len(pipe.inflight) >= pipe.depth or pipe.head_done()):
            res, user = pipe.finish()
            if dump is not None:
                o, ac_ = dump_q.pop(0); add_dump(o, ac_, res)
            done = scatter(res, user); batches += 1
        return done
    dump_q = []
    t_c = t_m = 0.0
    t0 = t_last = t_req = time.time()
    idle = 0
    live = a.actors
    while live > 0:
        cready = [ac for ac in cact if not ac.done and ac.ready()]
        mready = [ac for ac in mact if not ac.done and ac.ready()]
        if not cready and not mready and not (pipe is not None and pipe.inflight):
            if time.time() - t_req > a.idle_exit:
                print(f"[s3srv] no request for {a.idle_exit:.0f} s from {live} live actors: exiting", flush=True)
                break
            for i in range(a.actors):
                acs = [x[i] for x in (cact, mact) if x]
                if not acs[0].done and any(x.shutdown_flag() for x in acs):
                    for x in acs:
                        x.done = True
                    live -= 1
            idle += 1
            if idle > 2000:
                time.sleep(0.0002)
            continue
        idle = 0
        if a.min_period > 0 and mready and not cready:
            wait = a.min_period - (time.time() - t_req)
            if wait > 0:
                time.sleep(wait)
                mready = [ac for ac in mact if not ac.done and ac.ready()]
        t_req = time.time()
        if pipe is not None:
            if cready or pipe.inflight:
                ts = time.time()
                served += pipe_step(cready)
                t_c += time.time() - ts
        elif cready:
            ts = time.time()
            seqs, ns, off, used = [], [], 0, 1
            for ac in cready:
                k = int(ac.hdr[8]); n = ac.n()
                seqs.append(k); ns.append(n)
                used = max(used, int((ac.obs[:n, :, 0] != 0).sum(1).max()))
                off += n
            Bp = next((b for b in cs.BUCKETS if b >= off), off)
            Tp = next(t for t in cs.T_BUCKETS if t >= used)
            v_obs = n_obs_flat[:Bp * Tp * TW].reshape(Bp, Tp, TW)
            v_act = n_act[:Bp]
            off = 0
            for ac, n in zip(cready, ns):
                v_obs[off:off + n] = ac.obs[:n, :Tp]; v_act[off:off + n] = ac.act[:n]
                off += n
            if Bp > off:
                v_obs[off:Bp] = 0; v_act[off:Bp] = 0
            with torch.inference_mode(), torch.autocast("cuda", dtype=dt, enabled=dev == "cuda" and dt != torch.float32):
                o = h_obs_flat[:Bp * Tp * TW].view(Bp, Tp, TW).to(dev, non_blocking=True).long()
                ac_ = h_act[:Bp].to(dev, non_blocking=True).long()
                o[off:, 0, 0] = 1
                logits, v = state["net"](o, ac_)
                res = torch.cat([logits.float(), v.float()], 1)[:off].cpu().numpy()
            if dump is not None:
                add_dump(v_obs[:off], v_act[:off], res)
            off = 0
            for ac, k, n in zip(cready, seqs, ns):
                ac.out[:n] = res[off:off + n]; off += n
                ac.hdr[17] = state["cver"]
                ac.hdr[16] = k
                ac.last = k
            served += off; batches += 1
            t_c += time.time() - ts
        if mready:   # after the combat batch: actors do not block on macro answers, and more requests gather meanwhile
            ts = time.time()
            seqs, ns, off = [], [], 0
            for ac in mready:
                k = int(ac.hdr[8]); n = ac.n()
                m_ints[off:off + n] = ac.ints[:n]; m_floats[off:off + n] = ac.floats[:n]
                seqs.append(k); ns.append(n); off += n
            Bp = next((b for b in MBUCKETS if b >= off), off)
            m_ints[off:Bp] = 0; m_floats[off:Bp] = 0
            m_ints[off:Bp, ms.OFF["ctx"][0] + 4] = 1     # padded rows: valid room index, nothing else
            res = ens.serve(m_ints[:Bp], m_floats[:Bp], dev)[:off].cpu().numpy()
            off = 0
            for ac, k, n in zip(mready, seqs, ns):
                ac.out[:n] = res[off:off + n]; off += n
                ac.hdr[17] = ens.version
                ac.hdr[16] = k
                ac.last = k
            mserved += off; mbatches += 1
            t_m += time.time() - ts
        now = time.time()
        if now - t_last > a.stats_every:
            el = now - t0
            print(f"[s3srv] {el:.0f}s combat {served} states in {batches} batches (mean {served / max(1, batches):.0f}), {served / el:.0f}/s, "
                  f"busy {100 * t_c / el:.0f}%; macro {mserved} rows in {mbatches} batches (mean {mserved / max(1, mbatches):.0f}), "
                  f"busy {100 * t_m / el:.0f}%; versions combat {state['cver']} macro {ens.version}"
                  + (f"; graphs captured {pipe.captures}" if pipe is not None and pipe.use_graphs else ""), flush=True)
            t_last = now
        maybe_reload()
    while pipe is not None and pipe.inflight:   # answers still in flight (e.g. an actor that died while waiting)
        res, user = pipe.finish(); served += scatter(res, user); batches += 1
    print(f"s3 server stopped: combat {served} states / {batches} batches, macro {mserved} rows / {mbatches} batches", flush=True)
    for ac in cact + mact:
        ac.close()


if __name__ == "__main__":
    main()
