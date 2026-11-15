"""Combat batches of src/s3/py/s3_server.py with CUDA graphs and/or two batches in flight (options --graphs and
--overlap; both off by default, then the server does not use this module at all).

Per batch (one "slot" = pinned host buffers for inputs and outputs, device staging buffers and two CUDA events):
  host:  the server gathers the actors' rows into the slot's pinned obs/act buffers (padded rows: obs[., 0, 0] = 1, the
         same value the eager server writes on the device), then calls launch();
  copy stream:    pinned obs/act -> the slot's device staging (non-blocking), event `copied`;
  compute stream: waits for `copied`; eager: CombatNet on the staging tensors; graphs: staging -> the static inputs of the
         graph of this (batch bucket, token bucket), replay; then output -> the slot's pinned output (non-blocking),
         event `done`.
  host:  finish() waits for `done` of the oldest batch and returns its output rows; the server scatters them.
With depth 2 the server gathers and launches batch k+1 (and scatters batch k-1) while the GPU computes batch k.

Slot reuse is safe: a slot is launched again only after finish() returned it, i.e. after its `done` event, which is
recorded after every use of its host and staging buffers. Batches that share a graph are ordered on the compute stream:
batch k+1's copy into the static inputs is queued after batch k's replay and after the copy of k's static output.

CUDA graphs: captured on first use of a (batch bucket, token bucket) pair (the pipeline is drained first), with
autocast's weight-cast cache off, so every replay casts the current fp32 parameters: weights loaded in place with
load_state_dict (what the server does) are used by the next replay. Replacing the module or its tensors would not be;
the server never does that after the first load, except for weights of another structure (replace_net: new module,
graphs captured again). All graphs share one memory pool (they never run concurrently).
The layout-v4 feature mask of CombatNet (a non-persistent buffer, never reloaded) is part of the captured forward.
"""
from collections import deque

import numpy as np
import torch

import combat_obs as co

T, TW, A, AW = co.MAX_TOKENS, co.TOKEN_WIDTH, co.MAX_ACTIONS, co.ACTION_WIDTH


class _Slot:
    def __init__(self, maxB, ow, dev):
        self.h_obs = torch.zeros(maxB * T * TW, dtype=torch.int32).pin_memory()
        self.h_act = torch.zeros((maxB, A, AW), dtype=torch.int32).pin_memory()
        self.h_out = torch.zeros((maxB, ow), dtype=torch.float32).pin_memory()
        self.n_obs, self.n_act, self.n_out = self.h_obs.numpy(), self.h_act.numpy(), self.h_out.numpy()
        self.d_obs = torch.zeros(maxB * T * TW, dtype=torch.int32, device=dev)
        self.d_act = torch.zeros((maxB, A, AW), dtype=torch.int32, device=dev)
        self.copied = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.batch = None          # (Bp, rows, user data) while in flight


class _Graph:
    def __init__(self, net, Bp, Tp, dt, dev, pool):
        self.obs = torch.zeros((Bp, Tp, TW), dtype=torch.int32, device=dev)
        self.obs[:, 0, 0] = 1
        self.act = torch.zeros((Bp, A, AW), dtype=torch.int32, device=dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():     # warm-up (cuBLAS workspaces, lazy inits) off the capture
            for _ in range(2):
                self._fwd(net, dt)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g, pool=pool):
            self.out = self._fwd(net, dt)

    def _fwd(self, net, dt):
        with torch.autocast("cuda", dtype=dt, enabled=dt != torch.float32, cache_enabled=False):
            logits, v = net(self.obs.long(), self.act.long())
        return torch.cat([logits.float(), v.float()], 1)


class CombatPipe:
    def __init__(self, net, dev, dt, maxB, ow, graphs=False, depth=1):
        assert dev == "cuda", "the combat pipeline needs CUDA"
        self.net, self.dev, self.dt, self.maxB, self.ow = net, dev, dt, maxB, ow
        self.use_graphs, self.depth = graphs, depth
        self.slots = [_Slot(maxB, ow, dev) for _ in range(depth)]
        self.free = deque(self.slots)
        self.inflight = deque()
        self.xstream = torch.cuda.Stream()
        self.cstream = torch.cuda.Stream()
        self.graphs, self.pool = {}, None
        self.captures = 0

    # host side -------------------------------------------------------------------------------------------------
    def slot(self):
        """A free slot (call only when len(inflight) < depth)."""
        return self.free[0]

    def views(self, s, Bp, Tp):
        return s.n_obs[:Bp * Tp * TW].reshape(Bp, Tp, TW), s.n_act[:Bp]

    # GPU side --------------------------------------------------------------------------------------------------
    def _graph(self, Bp, Tp):
        g = self.graphs.get((Bp, Tp))
        if g is None:
            self.drain_gpu()
            g = _Graph(self.net, Bp, Tp, self.dt, self.dev, self.pool)
            self.pool = g.g.pool()
            self.graphs[(Bp, Tp)] = g
            self.captures += 1
        return g

    def launch(self, Bp, Tp, rows, user):
        """Queues the forward of the slot returned by slot() (filled up to Bp rows, token bucket Tp)."""
        if self.use_graphs:
            g = self._graph(Bp, Tp)          # before taking the slot: a capture drains the GPU, not the host queue
        s = self.free.popleft()
        n_in = Bp * Tp * TW
        with torch.cuda.stream(self.xstream):
            self.xstream.wait_event(s.done)  # no-op in practice (the slot was finished), kept as a guard
            s.d_obs[:n_in].copy_(s.h_obs[:n_in], non_blocking=True)
            s.d_act[:Bp].copy_(s.h_act[:Bp], non_blocking=True)
            s.copied.record(self.xstream)
        self.cstream.wait_event(s.copied)
        with torch.cuda.stream(self.cstream):
            o = s.d_obs[:n_in].view(Bp, Tp, TW)
            if self.use_graphs:
                g.obs.copy_(o); g.act.copy_(s.d_act[:Bp])
                g.g.replay()
                out = g.out
            else:
                with torch.inference_mode(), torch.autocast("cuda", dtype=self.dt, enabled=self.dt != torch.float32):
                    logits, v = self.net(o.long(), s.d_act[:Bp].long())
                    out = torch.cat([logits.float(), v.float()], 1)
            s.h_out[:Bp].copy_(out, non_blocking=True)
            s.done.record(self.cstream)
        s.batch = (Bp, rows, user)
        self.inflight.append(s)

    def head_done(self):
        return bool(self.inflight) and self.inflight[0].done.query()

    def finish(self):
        """Waits for the oldest batch; returns (output rows [rows, ow] numpy view of pinned memory, user data). The
        view stays valid until this slot is launched again (the caller copies it out before its next launch)."""
        s = self.inflight.popleft()
        s.done.synchronize()
        Bp, rows, user = s.batch
        s.batch = None
        self.free.append(s)
        return s.n_out[:rows], user

    def replace_net(self, net):
        """Serve another module from the next launch on (e.g. weights of another structure): waits for the batches in
        flight (they finish with the old module), drops the captured graphs (captured again at first use)."""
        self.drain_gpu()
        self.net = net
        self.graphs, self.pool = {}, None
        torch.cuda.synchronize()

    def drain_gpu(self):
        """Waits until all queued GPU work of the pipeline is complete (the host queue is kept)."""
        for s in self.inflight:
            s.done.synchronize()
        torch.cuda.synchronize()
