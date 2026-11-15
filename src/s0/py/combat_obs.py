"""Combat observation layout v5 (v4 shape; mirror of src/s0/S0.Sim/ObsLayout.cs and CombatObservation.cs).

Observation: int32[T=128, 11]; row = [kind, id, id2, f0..f7]. Actions: int32[A=64, 8]; row = [kind+1 (0 pad,
1 end, 2 card, 3 potion), source token, target token (-1 none, 0 = player/global), prompt code (0 none, 1 + number of
chosen cards), chosen card 1 (card vocab id, upgrade + 1), chosen card 2 (card vocab id, upgrade + 1)].
`id` indexes the vocabulary of the kind's primary category, `id2` the secondary one (0 = none/pad, 1 = <unk>).
Features per kind: FEATURES below. Layout v4 (2026-11-15) adds values in slots that were always 0 in v3 (V4_SLOTS):
card damage / block (hand, draw, discard, exhaust) and hidden power counters (enemy / player powers). Layout v5
(2026-11-15, opt-in) adds the relic counters in kind 11 f0/f1 (V5_SLOTS); the enemy intent damage (kind 6 f5) is
changed in place by the C# actor when obs=5 and cannot be masked. CombatNet always multiplies the features by the v3
slot mask (V3_MASK) before its `feat` MLP, so a network sees exactly its v3 input on v3 and v4 data; with ext=1 it adds
`ext_feat`, an MLP over the v4+v5 slots whose last layer starts at zero.
"""
import json, os

import numpy as np

VERSION = 4
VERSIONS_READ = (3, 4)   # encoded directories / chunks of these layouts are read (v3 = v4 with zeros in V4_SLOTS)
MAX_TOKENS, TOKEN_WIDTH, MAX_ACTIONS, ACTION_WIDTH, N_FEATURES, MAX_SEL = 128, 11, 64, 8, 8, 2
KINDS = ["pad", "global", "hand", "draw", "discard", "exhaust", "enemy", "enemy_power", "player_power", "orb", "potion", "relic", "context"]
CATEGORIES = {  # kind index -> (primary, secondary)
    1: ("", ""), 2: ("card", "enchantment"), 3: ("card", "enchantment"), 4: ("card", "enchantment"), 5: ("card", "enchantment"),
    6: ("monster", "intent"), 7: ("power", ""), 8: ("power", ""), 9: ("orb", ""), 10: ("potion", ""), 11: ("relic", ""), 12: ("", ""),
}
VOCAB_ORDER = ["card", "enchantment", "monster", "intent", "power", "orb", "potion", "relic"]
FEATURES = {  # kind index -> names of f0..f7 (layout v5; "" = always 0)
    1: ["round", "energy", "max_energy", "stars", "hp", "max_hp", "block", "orb_capacity"],
    2: ["hand_pos", "upgrade", "cost", "playable", "ench_amount", "damage", "block", ""],
    3: ["count", "upgrade", "ench_amount", "known_top_pos", "damage", "block", "", ""],
    4: ["count", "upgrade", "ench_amount", "damage", "block", "", "", ""],
    5: ["count", "upgrade", "ench_amount", "damage", "block", "", "", ""],
    6: ["slot", "hp", "max_hp", "block", "intent_type", "damage_per_hit", "hits", "alive"],
    7: ["enemy_slot", "amount", "counter", "", "", "", "", ""],
    8: ["amount", "counter", "", "", "", "", "", ""],
    9: ["position", "passive", "evoke", "", "", "", "", ""],
    10: ["slot", "marginal_tf", "", "", "", "", "", ""],
    11: ["counter", "shows_counter", "", "", "", "", "", ""],
    12: ["room", "act", "floor", "tf", "price_10hp", "potions_worth", "tmax_1000", "final_boss"],  # S2 Evaluator.cs Context
}
V4_SLOTS = {2: (5, 6), 3: (4, 5), 4: (3, 4), 5: (3, 4), 7: (2,), 8: (1,), 11: (0, 1)}  # kind -> ext_feat slots
# 11 f0/f1 = layout-v5 relic counters (2026-11-15): they are appended to this map because ext_feat consumes every slot
# listed here, and the live/v3 network must stay blind to them. Adding the two columns changes ext_feat's input width, so
# an ext checkpoint written before this change cannot load into the wider network: restart the shadow from the playing
# network (src/s3/run_s3.sh SHADOW_FROM_PLAYING=1 SHADOW_ARGS="--config ext=1", OBS=5).
V5_SLOTS = {11: (0, 1)}                                                     # kind -> feature slots introduced in v5
EXT_SLOTS = V4_SLOTS                                                        # every slot fed to ext_feat (v4 + v5)
V3_MASK = np.ones((len(KINDS), N_FEATURES), np.int64)                         # 1 = slot seen by every network
for _k, _sl in EXT_SLOTS.items():
    V3_MASK[_k, list(_sl)] = 0
EXT_COLS = [(k, f) for k in sorted(EXT_SLOTS) for f in EXT_SLOTS[k]]         # ext_feat input columns: (kind, slot)
EXT_SRC = np.full((len(KINDS), len(EXT_COLS)), N_FEATURES, np.int64)         # per kind: feature slot of each column
for _j, (_k, _f) in enumerate(EXT_COLS):                                      # (N_FEATURES = an all-zero column)
    EXT_SRC[_k, _j] = _f


def load_vocab(path):
    v = json.load(open(path))
    return {k: v[k] for k in VOCAB_ORDER}


def load_arrays(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["version"] in VERSIONS_READ and meta["max_tokens"] == MAX_TOKENS and meta["token_width"] == TOKEN_WIDTH
    assert meta["max_actions"] == MAX_ACTIONS and meta["action_width"] == ACTION_WIDTH
    n = meta["n"]
    obs = np.fromfile(os.path.join(d, "obs.bin"), dtype="<i4").reshape(n, MAX_TOKENS, TOKEN_WIDTH)
    act = np.fromfile(os.path.join(d, "act.bin"), dtype="<i4").reshape(n, MAX_ACTIONS, ACTION_WIDTH)
    return obs, act


def decode_state(obs_row, vocab):
    """int tokens -> list of (kind, id string, id2 string, features) for non-pad tokens (the C# ObsToken form)."""
    out = []
    for r in obs_row:
        k = int(r[0])
        if k == 0:
            continue
        c1, c2 = CATEGORIES[k]
        out.append((k, vocab[c1][r[1]] if c1 and r[1] else "", vocab[c2][r[2]] if c2 and r[2] else "", tuple(int(x) for x in r[3:])))
    return out


def multisets(tokens):
    """Per kind, the multiset of (id, id2, features) -- order-free view used by the round-trip test."""
    from collections import Counter
    m = {}
    for k, i, i2, f in tokens:
        m.setdefault(k, Counter())[(i, i2, tuple(f))] += 1
    return m


def global_offsets(vocab):
    """Offsets that put every category into one embedding table."""
    off, o = {}, 0
    for c in VOCAB_ORDER:
        off[c] = o
        o += len(vocab[c])
    return off, o


def kind_offsets(vocab):
    """Per kind, the global-table offset of its primary and secondary category (-1 = no category)."""
    off, _ = global_offsets(vocab)
    prim, sec = np.full(len(KINDS), -1, np.int64), np.full(len(KINDS), -1, np.int64)
    for k, (c1, c2) in CATEGORIES.items():
        if c1: prim[k] = off[c1]
        if c2: sec[k] = off[c2]
    return prim, sec


try:
    import torch
    from torch import nn

    class CombatObsDataset(torch.utils.data.Dataset):
        """Encoded states from an obs_v1 directory (obs.bin, act.bin, meta.json); optional targets arrays."""

        def __init__(self, d, targets=None):
            self.obs, self.act = load_arrays(d)
            self.targets = targets

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, i):
            item = {"obs": torch.from_numpy(self.obs[i].astype(np.int64)), "act": torch.from_numpy(self.act[i].astype(np.int64))}
            if self.targets is not None:
                item.update({k: torch.as_tensor(v[i]) for k, v in self.targets.items()})
            return item

    class CombatNet(nn.Module):
        """Small combat network over layout v1 (~1M parameters at d=128).
        Token embedding = kind + id + id2 embeddings + MLP(signed log features); 2-layer transformer encoder.
        Policy: one logit per action row from [h_source, h_target (or a learned 'no target'), action-kind embedding,
        pooled state]; padded actions get -inf. Value head: n_value outputs (2 = scalar head: P(survive) logit and
        relative T_f; 47 = outcome head: 41 end-HP bin logits + 6 potion-kept logits, src/s2/S2/Evaluator.cs Outcome).
        forward returns (policy logits [B, A], value outputs [B, n_value]).
        Features: the integer features are multiplied by V3_MASK[kind] before `feat` (exact: integer product, so on v3
        data, whose V4_SLOTS are 0, the input is unchanged bit for bit, and on v4 data the network sees its v3 input).
        ext=1: `ext_feat` = MLP over the V4_SLOTS values (sign-log1p, one input column per (kind, slot) of EXT_COLS) added
        to the token embedding; its last layer starts at zero, so a v3 checkpoint loaded into an ext network gives the
        same outputs (src/s2/py/model.py load_net)."""

        def __init__(self, vocab, d=128, layers=2, heads=4, n_value=2, dropout=0.0, vdrop=0.0, ext=0):
            super().__init__()
            self.vdrop = vdrop  # dropout on the pooled state before the value head (training only; 0 = off)
            self.ext = int(ext)
            prim, sec = kind_offsets(vocab)
            _, n_ids = global_offsets(vocab)
            self.register_buffer("prim", torch.from_numpy(prim))
            self.register_buffer("sec", torch.from_numpy(sec))
            # constants, not in the state dict (checkpoints of every version load unchanged)
            self.register_buffer("fmask", torch.from_numpy(V3_MASK.copy()), persistent=False)
            self.kind = nn.Embedding(len(KINDS), d)
            self.ids = nn.Embedding(n_ids + 1, d)  # last row = "no category"
            self.none_id = n_ids
            self.feat = nn.Sequential(nn.Linear(N_FEATURES, d), nn.GELU(), nn.Linear(d, d))
            enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers)
            self.act_kind = nn.Embedding(4, d)
            self.card_off = global_offsets(vocab)[0]["card"]
            self.sel_up = nn.Embedding(4, d)      # upgrade + 1 of a chosen card (0 = unused slot), clamped
            self.sel_count = nn.Embedding(6, d)   # prompt code, clamped
            self.no_target = nn.Parameter(torch.zeros(d))
            self.policy = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
            self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_value))
            if self.ext:   # created last: the other modules draw the same random init as without ext
                self.register_buffer("ext_src", torch.from_numpy(EXT_SRC.copy()), persistent=False)
                self.ext_feat = nn.Sequential(nn.Linear(len(EXT_COLS), d), nn.GELU(), nn.Linear(d, d))
                self.reset_ext()

        def reset_ext(self):
            """Fresh ext_feat: default init of the first layer, zero last layer (the network ignores the v4 slots)."""
            self.ext_feat[0].reset_parameters()
            nn.init.zeros_(self.ext_feat[2].weight)
            nn.init.zeros_(self.ext_feat[2].bias)

        def forward(self, obs, act):
            kind = obs[..., 0]
            pad = kind == 0
            def ids(col, offs):
                o = offs[kind]
                return torch.where((o >= 0) & (obs[..., col] > 0), o + obs[..., col], torch.full_like(o, self.none_id))
            f = (obs[..., 3:] * self.fmask[kind]).float()     # v3 slots only (integer product: exact)
            x = self.kind(kind) + self.ids(ids(1, self.prim)) + self.ids(ids(2, self.sec)) + self.feat(torch.sign(f) * torch.log1p(f.abs()))
            if self.ext:
                g = obs[..., 3:].float()
                g = torch.cat([torch.sign(g) * torch.log1p(g.abs()), torch.zeros_like(g[..., :1])], -1)
                x = x + self.ext_feat(torch.gather(g, -1, self.ext_src[kind]))
            h = self.encoder(x, src_key_padding_mask=pad)
            keep = (~pad).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
            akind = act[..., 0]
            src = act[..., 1].clamp(min=0)
            tgt = act[..., 2]
            gather = lambda idx: torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            h_src = gather(src)
            h_tgt = torch.where((tgt >= 0).unsqueeze(-1), gather(tgt.clamp(min=0)), self.no_target.expand_as(h_src))
            ak = self.act_kind(akind.clamp(max=3)) + self.sel_count(act[..., 3].clamp(0, 5))
            for j in range(MAX_SEL):
                cid, up = act[..., 4 + 2 * j], act[..., 5 + 2 * j]
                used = (up > 0).unsqueeze(-1)
                e = self.ids(torch.where(cid > 0, cid + self.card_off, torch.full_like(cid, self.none_id))) + self.sel_up(up.clamp(0, 3))
                ak = ak + torch.where(used, e, torch.zeros_like(e))
            a = torch.cat([h_src, h_tgt, ak, pooled.unsqueeze(1).expand_as(h_src)], -1)
            logits = self.policy(a).squeeze(-1).masked_fill(akind == 0, float("-inf"))
            vin = nn.functional.dropout(pooled, self.vdrop, self.training) if self.vdrop > 0 else pooled
            return logits, self.value(vin)
except ImportError:  # numpy-only use (decoder)
    pass
