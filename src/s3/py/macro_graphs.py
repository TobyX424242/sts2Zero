"""Macro batches of src/s3/py/s3_server.py with CUDA graphs (option --mgraphs 1; off by default, then the server
does not use this module at all) and optionally under autocast (--mdtype bf16).

Why: at layout v3 (attention blocks) one macro batch of ~18 rows is a few hundred small kernels per member, launched
one by one from Python (~33 ms per batch, the macro process 91% busy at 52 actors). A replayed graph launches them
all at once.

Per batch (MacroServe.serve), as Ensemble.serve: the rows are split into decision rows (policy member 0:
Ensemble.serve_decisions) and value rows (every member's value head: Ensemble.serve_values). Each half is padded on
its own up to the smallest bucket >= its row count (BUCKETS; padded rows: zeros with a valid room index, as the
server's padding) and runs the graph of (half, bucket): the rows go into pinned host buffers, are copied into the
graph's static input buffers, the graph is replayed, its output rows are copied back to pinned memory; one stream
synchronisation per batch, then the rows are scattered into the batch's order. A half larger than the largest bucket
runs eagerly (counted in `eager rows`). The server passes only its real rows (no MBUCKETS padding: that padding would
all land in the value half).

Graph capture: at the first use of a (half, bucket), after two warm-up forwards on a side stream; all graphs share one
memory pool (they never run concurrently, and every graph's output is copied out, stream-ordered, before another graph
is replayed). The captured forward is the eager one with b["static"] = True (macro_s3.py): every fight / map slot
instead of the batch's largest count, and option embeddings by masks instead of boolean indexing (no data-dependent
shapes, no host syncs). Same function per row; only matrix shapes differ, so TF32 kernel choice can move results by
~1e-6. Under autocast (--mdtype bf16) the weight-cast cache is off, so every replay casts the current fp32 weights.

Weights: a reload whose members fit the served ensemble (same count, vocabulary and parameter shapes) is copied into
the same parameter tensors (Ensemble.load_members_ck: load_state_dict, in place): the graphs stay valid and read the
new weights at their next replay. Otherwise (member count, vocabulary or architecture changed) a new Ensemble is built,
the graphs are dropped and captured again at first use. The milestone blend (Ensemble.milestones, alpha, boss1_share)
is supported inside the graphs; its settings are constants of the captured kernels, so a change of any of them (a
reload of a file with other settings) drops the graphs too (captured again at first use).

Without CUDA (or capture=False) the same static-shape forward runs eagerly on the same static buffers: the check
script's CPU test of this code path (mgraphs_check.py).
"""
import contextlib

import numpy as np
import torch

import macro_s3 as ms

BUCKETS = (8, 16, 32, 64, 128, 256)
CTX_ROOM = ms.OFF["ctx"][0] + 4         # padded rows: valid room index, nothing else (as s3_server.py)
HALVES = ("dec", "val")


def fits(ens, ck):
    """True if the members of a loaded file (dict) load in place into ens: same member count, vocabulary and special
    list, every parameter of the file present with the same shape (dt_emb of an S1 file is padded by the loader)."""
    if len(ck["members"]) != len(ens.nets) or ck["vocab"] != ens.vocab or ck["special"] != ens.special:
        return False
    if bool(ck.get("p2", False)) != bool(getattr(ens, "p2", False)) or bool(ck.get("p2_ema")) != (getattr(ens, "p2_ema_nets", None) is not None):
        return False
    for n, sd in zip(ens.nets, ck["members"]):
        own = n.state_dict()
        for k, v in sd.items():
            if k not in own or (k != "dt_emb.weight" and tuple(own[k].shape) != tuple(v.shape)):
                return False
    return True


class _Graph:
    """One half (decision rows / value rows) at one bucket size: static inputs ints [Bp, IW] int32, floats [Bp, FW],
    output [Bp, OW] float32. capture: a CUDA graph; else the same forward run eagerly on the static buffers."""

    def __init__(self, ens, half, Bp, dev, dt, capture, pool):
        self.ens, self.half, self.dt = ens, half, dt
        self.ints = torch.zeros((Bp, ms.IW), dtype=torch.int32, device=dev)
        self.ints[:, CTX_ROOM] = 1
        self.floats = torch.zeros((Bp, ms.FW), dtype=torch.float32, device=dev)
        self.g = self.out = None
        if capture:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):             # warm-up (cuBLAS workspaces, lazy inits) off the capture
                for _ in range(2):
                    self._fwd()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g, pool=pool):
                self.out = self._fwd()

    @torch.no_grad()
    def _fwd(self):
        b = ms.tensors_to_batch(self.ints.long(), self.floats)
        b["static"] = True
        ac = (torch.autocast("cuda", dtype=self.dt, cache_enabled=False) if self.dt != torch.float32
              else contextlib.nullcontext())
        with ac:
            o = self.ens.serve_decisions(b) if self.half == "dec" else self.ens.serve_values(b)
        return o.float()

    def run(self):
        if self.g is not None:
            self.g.replay()
            return self.out
        return self._fwd()


class MacroServe:
    """The macro half of the server: serve(ints, floats) -> numpy rows [B, OW], as Ensemble.serve(...).cpu().numpy().
    graphs=False: Ensemble.serve itself (under autocast if dtype is not fp32)."""

    def __init__(self, ens, dev, graphs=True, dtype=torch.float32, buckets=BUCKETS, capture=None):
        self.cuda = str(dev).startswith("cuda")
        self.ens, self.dev, self.dt = ens, dev, (dtype if self.cuda else torch.float32)   # autocast: CUDA only
        self.use_graphs = graphs
        self.capture = self.cuda if capture is None else (capture and self.cuda)
        self.buckets = tuple(sorted(int(x) for x in buckets))
        self.graphs, self.pool = {}, None
        self.captures = self.builds = self.rebuilds = self.eager_rows = 0      # builds: graphs made (captured or not)
        self.in_capture = False
        self.key = self._key()
        maxb = self.buckets[-1]

        def host(shape, dt):
            t = torch.zeros(shape, dtype=dt)
            return t.pin_memory() if self.cuda else t
        # one set per half: both halves' copies are queued before the one synchronisation of the batch
        self.h_ints = {h: host((maxb, ms.IW), torch.int32) for h in HALVES}
        self.h_floats = {h: host((maxb, ms.FW), torch.float32) for h in HALVES}
        self.h_out = {h: host((maxb, ms.OW), torch.float32) for h in HALVES}
        self.n_ints = {h: t.numpy() for h, t in self.h_ints.items()}
        self.n_floats = {h: t.numpy() for h, t in self.h_floats.items()}
        self.n_out = {h: t.numpy() for h, t in self.h_out.items()}

    def _key(self):
        """What the captured kernels depend on besides the weights' values."""
        e = self.ens
        return id(e), len(e.nets), bool(e.milestones), float(e.alpha), float(e.boss1_share), e.value_members, bool(getattr(e, "p2", False))

    def _ac(self):
        return (torch.autocast("cuda", dtype=self.dt) if self.cuda and self.dt != torch.float32
                else contextlib.nullcontext())

    def reset(self):
        """Drops the captured graphs (captured again at first use)."""
        if self.cuda:
            torch.cuda.synchronize()
        self.graphs, self.pool = {}, None
        if self.cuda:
            torch.cuda.empty_cache()
        self.key = self._key()

    def replace(self, ens):
        """Serve another Ensemble object from the next batch on (graphs dropped)."""
        self.ens = ens
        self.rebuilds += 1
        self.reset()

    def reload(self, path):
        """The server's reload of the macro file. In place if the file fits the served ensemble (graphs kept: they read
        the new weights at their next replay), else a new Ensemble (graphs dropped). Returns True if rebuilt."""
        if self.cuda:
            torch.cuda.synchronize()                  # nothing in flight (serve synchronises), kept as a guard
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        if fits(self.ens, ck):
            self.ens.load_members_ck(ck)
            return False
        old_vm = self.ens.value_members
        new_ens = ms.Ensemble.from_ck(ck, self.dev, path)
        if old_vm is not None:
            new_ens.set_value_members(old_vm)
        self.replace(new_ens)
        return True

    def _graph(self, half, Bp):
        g = self.graphs.get((half, Bp))
        if g is None:
            if self.capture:
                torch.cuda.synchronize()
            self.in_capture = self.capture
            g = _Graph(self.ens, half, Bp, self.dev, self.dt, self.capture, self.pool)
            self.in_capture = False
            self.builds += 1
            if self.capture:
                self.pool = g.g.pool()
                self.captures += 1
                torch.cuda.empty_cache()              # the warm-up's activations (the graphs have their own pool)
            self.graphs[(half, Bp)] = g
        return g

    def stats(self):
        return (f"macro graphs {self.captures} captured ({len(self.graphs)} live), eager rows {self.eager_rows}, "
                f"rebuilds {self.rebuilds}")

    @torch.no_grad()
    def serve(self, ints, floats):
        """Output rows [B, OW] (a new numpy float32 array) for bridge rows ints [B, IW'], floats [B, FW']
        (any layout width; no padding rows needed)."""
        if self.use_graphs:
            try:
                return self._serve_graphs(ints, floats)
            except RuntimeError as e:
                if not self.in_capture:
                    raise
                # a capture failed (e.g. an op that cannot be captured): eager from now on, said once
                print(f"[macro_graphs] graph capture failed ({type(e).__name__}: {str(e)[:200]}): serving eagerly", flush=True)
                self.use_graphs = self.in_capture = False
                self.reset()
        with self._ac():
            return self.ens.serve(ints, floats, self.dev).float().cpu().numpy()

    def _serve_graphs(self, ints, floats):
        ints, floats = ms.pad_rows(ints, floats)
        if self._key() != self.key:
            self.reset()
        dec = ms.decision_rows(ints)
        res = np.zeros((len(ints), ms.OW), np.float32)
        queued = []
        for half, idx in (("dec", np.flatnonzero(dec)), ("val", np.flatnonzero(~dec))):
            n = len(idx)
            if n == 0:
                continue
            Bp = next((x for x in self.buckets if x >= n), None)
            if Bp is None:                            # above the largest bucket: eager (rare)
                b = ms.rows_to_batch(ints[idx], floats[idx], self.dev)
                with self._ac():
                    o = self.ens.serve_decisions(b) if half == "dec" else self.ens.serve_values(b)
                res[idx] = o.float().cpu().numpy()
                self.eager_rows += n
                continue
            g = self._graph(half, Bp)
            hi, hf = self.n_ints[half], self.n_floats[half]
            hi[:n] = ints[idx]; hf[:n] = floats[idx]
            hi[n:Bp] = 0; hi[n:Bp, CTX_ROOM] = 1; hf[n:Bp] = 0
            g.ints.copy_(self.h_ints[half][:Bp], non_blocking=True)
            g.floats.copy_(self.h_floats[half][:Bp], non_blocking=True)
            out = g.run()
            self.h_out[half][:n].copy_(out[:n], non_blocking=True)
            queued.append((half, idx))
        if self.cuda and queued:
            torch.cuda.current_stream().synchronize()
        for half, idx in queued:
            res[idx] = self.n_out[half][:len(idx)]
        return res
