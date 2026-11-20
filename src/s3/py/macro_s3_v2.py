"""FROZEN copy of macro_s3.py at layout v2 (2026-11-16, before layout v3). Do not edit: layout3_check.py compares the
current network (macro_s3.py, layout v3) against this one. Rows: ints 462, floats 344, outputs 221.

S3 macro network: the S1 MacroNet (src/s1/train_macro_v2.py) extended for full-game self-play.

Additions to S1 (all new parameters start at zero, so an S1 checkpoint loads into this class with unchanged outputs
on everything S1 covered, and the new decision types start as a uniform policy):
  - decision types 7 MAP (which map point to travel to) and 8 EVENT (event option), which have no human data;
  - option kinds 5 (map point: type embedding + linear map of path features), 6 (event option: embedding of the
    hashed option key) and 7 (a fixed logit of 0: LEAVE in the shop, where S1's per-item buy logits become a softmax
    against it, so one affordable item is bought with exactly S1's probability sigmoid(logit));
  - the policy query gets the HP ramp features (S1's policy head could not see HP: it was removed from the encoder to
    make the value monotone in HP), through a layer that starts at zero;
  - rest-site options are ordinary options of kind 4 (special vocabulary "REST:<NAME>"), which is the same function
    S1 used for its rest logits, restricted to the options actually offered.

Layout v2 (2026-11-16; append only, so every v1 offset is unchanged; again every new parameter starts at zero, so a
v1 file loads with exactly the outputs it had, whatever the new inputs hold):
  - ints: the act's boss and the act's second boss (act 3 at ascension >= 10; else 0), in the id space of the context
    encounter (ctx[5]). Each has its own embedding (index 0 = none stays the zero vector), added to the act context token;
  - floats: 8 more features per option (map options only: the order of rooms ahead, see RunAgent.MapFeatures2), through
    a second linear map added where the 12 path features enter; and 8 more state scalars (elites fought / rest sites /
    shops visited in this act, free potion slots, potion slots, map row, 2 reserved), through a linear map added to
    the scalar token. None of these depends on HP, so the state vector stays HP-free and the value stays monotone
    in HP through its ramp head. The linear maps have no bias: a row with zeros there (a padded v1 row) gets nothing;
  - v1 rows (ints 460 wide, floats 208 wide: old chunks, carry files, bridges of old actors) are zero-padded to the
    new widths by pad_rows(); rows_to_batch() does it for any rows it is given.

Row layout shared with the C# actor (src/s3/S3/MacroLayout.cs) and the run records:
  ints  [IW = 462]: d_ids 64, d_upg 64, d_ench 64, d_eamt 64, d_age 64, r_ids 40, r_age 40, p_ids 5, ctx 6, dtype 1,
                    opt_kind 16, opt_id 16, opt_upg 16 (v1 ends here: 460), boss 2
  floats[FW = 344]: scal 16, opt_feat 16 x 12 (v1 ends here: 208), opt_feat2 16 x 8 (from 208), scal2 8 (from 336)
  out   [OW = 110]: 0 a (ensemble mean of the value head's HP-free logit), 1..26 c (ensemble mean of the 26 monotone
                    ramp weights; value logit at HP h = a + sum_k c_k phi_k(h, max HP)), 27 ensemble value logit at the
                    row's own HP, 28 standard deviation of the members' value logits, 29 the policy member's own value
                    logit, 30..45 option logits, 46..109 deck-slot logits,
                    110..217 milestone heads (4 x [a, 26 c], ensemble means; see below), 218 alpha, 219 boss-1 share,
                    220 1 if the file has trained milestone heads (then 27 is the logit of the milestone blend, 28 its
                    spread over members, 29 the policy member's blend)

Milestone heads (2026-11-14, PROPOSAL.md D14, PROGRESS.md "why the run value gains only 0.007"): one HP-monotone head
per stage of a run, each conditional on reaching the stage before it:
  q0 = P(clear act 1), q1 = P(clear act 2 | act 1 cleared), q2 = P(beat the first boss of act 3 | act 2 cleared),
  q3 = P(win | first act-3 boss beaten): at ascension >= 10 (two bosses) only; below it the first act-3 boss ends the run.
The stage of a row is its act index, plus 1 on a boss floor (the state after that boss). The run outcome of D14 follows
outside the weights: reach probabilities R1..R4 (1 for stages already reached, products of q for later ones) and
  V = alpha R4 + (1 - alpha) (R1 + R2 + w R3 + (1 - w) R4) / 3,   w = the boss-1 share (0 = D14 as it is),
so V can never fall under the progress already secured, and each head learns at its own base rate (at A10 about 25% of
runs that beat the first act-3 boss win, against 3.6% of all runs).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "value_v1")); sys.path.insert(0, os.path.join(ROOT, "s1"))
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_macro_v2 as tm  # noqa: E402
import train_value_v2 as v2  # noqa: E402

S = v2.S
MAX_DECK, MAX_REL, MAX_POT, K, MF = 64, 40, 5, 16, 12
MF2, NS2 = 8, 8                                                   # layout v2: extra features per option, extra state scalars
IW_V1 = 5 * MAX_DECK + 2 * MAX_REL + MAX_POT + 6 + 1 + 3 * K       # 460
FW_V1 = 16 + K * MF                                               # 208
IW = IW_V1 + 2                                                    # 462: + the act boss, the act's second boss
F_OPT2 = FW_V1                                                    # 208: opt_feat2 [K, MF2]
F_SCAL2 = F_OPT2 + K * MF2                                        # 336: scal2 [NS2]
FW = F_SCAL2 + NS2                                                # 344
N_RAMPS = v2.N_RAMPS                                              # 26
N_MS = 4                                                          # milestone heads
OW = 1 + N_RAMPS + 3 + K + MAX_DECK + N_MS * (1 + N_RAMPS) + 3    # 221
O_A, O_C, O_V, O_STD, O_VPOL, O_OPT, O_SLOT = 0, 1, 27, 28, 29, 30, 46
O_MS, O_ALPHA, O_W1, O_HASMS = 110, 218, 219, 220
DOUBLE_BOSS_ASC, ROOM_BOSS = 10, 5
DT = tm.DT_NAMES + ["MAP", "EVENT"]                               # 0..8
DT_SLOT = (4, 5)                                                  # SMITH_TARGET, REMOVE_TARGET: choice among deck slots
N_EVENT_BUCKETS, N_MAP_TYPES = 4096, 12
ERA_V0111 = 4

# offsets into the int row
_o = 0
OFF = {}
for _k, _n in (("d_ids", MAX_DECK), ("d_upg", MAX_DECK), ("d_ench", MAX_DECK), ("d_eamt", MAX_DECK), ("d_age", MAX_DECK),
               ("r_ids", MAX_REL), ("r_age", MAX_REL), ("p_ids", MAX_POT), ("ctx", 6), ("dtype", 1),
               ("opt_kind", K), ("opt_id", K), ("opt_upg", K), ("boss", 2)):
    OFF[_k] = (_o, _o + _n); _o += _n
assert _o == IW and OFF["opt_upg"][1] == IW_V1
# parameters added by layout v2 (registered last, zero at the start; absent from v1 files)
V2_PARAMS = ("map_feat2", "boss_emb", "boss2_emb", "scal2")


class S3MacroNet(tm.MacroNet):
    def __init__(self, vocab, n_special, rest_ids):
        super().__init__(vocab, n_special, rest_ids, arch="noattn")
        d = self.card.embedding_dim
        old = self.dt_emb
        self.dt_emb = nn.Embedding(len(DT), d)
        with torch.no_grad():
            self.dt_emb.weight.zero_(); self.dt_emb.weight[:old.num_embeddings] = old.weight
        self.map_emb = nn.Embedding(N_MAP_TYPES, d); nn.init.zeros_(self.map_emb.weight)
        self.map_feat = nn.Linear(MF, d); nn.init.zeros_(self.map_feat.weight); nn.init.zeros_(self.map_feat.bias)
        self.event_emb = nn.Embedding(N_EVENT_BUCKETS, d); nn.init.zeros_(self.event_emb.weight)
        # S1 removed HP from the encoder (the value is monotone in HP through its own head), so the S1 policy cannot
        # see HP. In S3 the policy query gets the same 26 HP ramp features through a layer that starts at zero.
        self.hp_pol = nn.Linear(N_RAMPS, d); nn.init.zeros_(self.hp_pol.weight); nn.init.zeros_(self.hp_pol.bias)
        # milestone heads (see the header): HP-free logit + softplus ramp weights per head, as the S1 value head
        self.ms_win = nn.Linear(d, N_MS); nn.init.zeros_(self.ms_win.weight); nn.init.zeros_(self.ms_win.bias)
        self.ms_mono = nn.Linear(d, N_MS * N_RAMPS)
        nn.init.zeros_(self.ms_mono.weight); nn.init.constant_(self.ms_mono.bias, -4.0)
        # layout v2 (see the header). Registered after everything else: the trainer restores the optimizer state of an
        # older carry file by parameter position (train_macro_s3.load_carry), which needs new parameters at the end.
        n_enc = self.enc.num_embeddings
        self.map_feat2 = nn.Linear(MF2, d, bias=False); nn.init.zeros_(self.map_feat2.weight)
        self.boss_emb = nn.Embedding(n_enc, d, padding_idx=0); nn.init.zeros_(self.boss_emb.weight)
        self.boss2_emb = nn.Embedding(n_enc, d, padding_idx=0); nn.init.zeros_(self.boss2_emb.weight)
        self.scal2 = nn.Linear(NS2, d, bias=False); nn.init.zeros_(self.scal2.weight)

    def load_s1_or_s3(self, sd):
        """Loads an S1 MacroNet state dict (new parameters keep their zero start) or an S3 one."""
        sd = dict(sd)
        if sd["dt_emb.weight"].shape[0] < len(DT):
            w = torch.zeros(len(DT), sd["dt_emb.weight"].shape[1], dtype=sd["dt_emb.weight"].dtype)
            w[:sd["dt_emb.weight"].shape[0]] = sd["dt_emb.weight"]
            sd["dt_emb.weight"] = w
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(m.split(".")[0] in ("map_emb", "map_feat", "event_emb", "hp_pol", "ms_win", "ms_mono") + V2_PARAMS for m in missing), missing
        # a file without the layout v2 parameters is the function without them: zero, also when this network held
        # trained ones before (a server that reloads an older file)
        own = dict(self.named_parameters())
        with torch.no_grad():
            for m in missing:
                if m.split(".")[0] in V2_PARAMS:
                    own[m].zero_()

    def encode(self, b):
        """tm.MacroNet.encode (the same code) plus the layout v2 state inputs where the state context enters the encoder:
        the two boss embeddings are added to the act context token, the linear map of the extra scalars to the scalar
        token. With these parameters at zero the tokens are the old ones. No new token: the mean pooling would change.
        The new inputs are HP-free (as every encoder input), so z does not depend on HP and the value heads stay
        monotone in HP. A batch without the new fields (not built by rows_to_batch) is encoded as before."""
        b = v2.apply_drop(b, self.drop)
        scal = b["scal"] * self.mask_always
        phi = self._hp_features(scal)
        scal_enc = scal * self.mask_encoder
        B = b["ctx"].shape[0]; ctx = b["ctx"]
        era = ctx[:, 2]
        if self.training and self.era_dropout > 0:
            era = torch.where(torch.rand(B, device=era.device) < self.era_dropout, torch.full_like(era, 5), era)
        T = self.tok_type.weight
        act_tok = self.act(ctx[:, 3].clamp(0, 3))
        boss = b.get("boss")
        if boss is not None:
            n = self.boss_emb.num_embeddings
            act_tok = act_tok + self.boss_emb(boss[:, 0].clamp(0, n - 1)) + self.boss2_emb(boss[:, 1].clamp(0, n - 1))
        ctx_tok = torch.stack([self.char(ctx[:, 0]), self.asc(ctx[:, 1]), self.era(era), act_tok,
                               self.room(ctx[:, 4]), self.enc(ctx[:, 5])], 1) + T[1]
        scal_vec = self.scal(scal_enc.clamp(-10, 10))
        scal2 = b.get("scal2")
        if scal2 is not None:
            scal_vec = scal_vec + self.scal2(scal2.clamp(-10, 10)).to(scal_vec.dtype)
        scal_tok = (scal_vec + T[2]).unsqueeze(1)
        card_tok = (self.card(b["d_ids"]) + self.upg(b["d_upg"]) + self.ench(b["d_ench"])
                    + self.card_num(torch.stack([b["d_eamt"] / 20.0, b["d_age"] / 63.0], -1).float()) + T[3])
        relic_tok = self.relic(b["r_ids"]) + self.relic_num((b["r_age"] / 63.0).unsqueeze(-1).float()) + T[4]
        pot_tok = self.potion(b["p_ids"]) + T[5]
        x = torch.cat([self.cls.expand(B, -1, -1) + T[0], ctx_tok, scal_tok, card_tok, relic_tok, pot_tok], 1)
        pad = torch.cat([torch.zeros(B, 8, dtype=torch.bool, device=x.device), b["d_ids"] == 0, b["r_ids"] == 0, b["p_ids"] == 0], 1)
        if not self.use_cls:
            pad[:, 0] = True                      # CLS excluded from pooling
        h = self.encoder(x, src_key_padding_mask=pad) if self.use_cls else self.encoder(x)
        valid = (~pad).unsqueeze(-1).to(h.dtype)
        mean = (h * valid).sum(1) / valid.sum(1)
        pooled = torch.cat([h[:, 0], mean], -1) if self.use_cls else mean
        return self.trunk(self.norm(pooled)), phi, h

    def opt_emb_s3(self, kind, oid, upg, feat, feat2=None):
        e = torch.zeros(*kind.shape, self.card.embedding_dim, device=kind.device, dtype=torch.float32)
        for k, fn, n in ((1, self.card, self.card.num_embeddings), (2, self.relic, self.relic.num_embeddings),
                         (3, self.potion, self.potion.num_embeddings), (4, self.special, self.special.num_embeddings),
                         (5, self.map_emb, N_MAP_TYPES), (6, self.event_emb, N_EVENT_BUCKETS)):
            m = kind == k
            if m.any():
                e[m] = fn(oid[m].clamp(0, n - 1)).float()
        m = kind == 1
        if m.any():
            e[m] = e[m] + self.upg(upg[m].clamp(0, 3)).float()
        m = kind == 5
        if m.any():
            e[m] = e[m] + self.map_feat(feat[m]).float()
            if feat2 is not None:
                e[m] = e[m] + self.map_feat2(feat2[m]).float()
        return e

    def forward_ms(self, b, detach=False):
        """Milestone heads: a [B, N_MS], c [B, N_MS, 26] (softplus), logits at the row's own HP [B, N_MS]."""
        z, phi, h = self.encode(b)
        return self.ms_from(z.detach() if detach else z, phi)

    def ms_from(self, z, phi):
        a = self.ms_win(z).float()
        c = F.softplus(self.ms_mono(z).float()).view(-1, N_MS, N_RAMPS)
        return a, c, a + (c * phi.float().unsqueeze(1)).sum(-1)

    def forward_rows(self, b, policy=True, detach=False):
        """b: dict from rows_to_batch. Returns a [B], c [B, 26], logit at own HP [B], and (if policy) option logits
        [B, K] (kind 7 = 0, padding = -1e9) and deck-slot logits [B, 64] (unmasked). detach: the value heads read the
        state vector without sending gradients into the encoder (used for the policy member's value loss)."""
        z, phi, h = self.encode(b)
        if detach:
            z = z.detach()
        a = self.win_head(z).squeeze(-1).float()
        c = F.softplus(self.mono(z).float())
        v = a + (c * phi).sum(-1)
        if not policy:
            return a, c, v, None, None
        q = z + self.dt_emb(b["dtype"].clamp(0, len(DT) - 1)) + self.hp_pol(phi.float()).to(z.dtype)
        kind = b["opt_kind"]
        e = self.opt_emb_s3(kind, b["opt_id"], b["opt_upg"], b["opt_feat"], b.get("opt_feat2"))
        qe = q.unsqueeze(1).expand_as(e).float()
        opt = self.opt_mlp(torch.cat([qe, e, qe * e], -1)).squeeze(-1).float()
        opt = torch.where(kind == 7, torch.zeros_like(opt), opt).masked_fill(kind == 0, -1e9)
        hs = h[:, 8:8 + MAX_DECK].float()
        qs = q.unsqueeze(1).expand_as(hs).float()
        slot = self.slot_mlp(torch.cat([qs, hs, qs * hs], -1)).squeeze(-1).float()
        return a, c, v, opt, slot


def stage_of(ctx_act, ctx_room):
    """Stage of a row (0..3): its act index, plus 1 on a boss floor (the state after that boss)."""
    return ctx_act + (ctx_room == ROOM_BOSS).astype(ctx_act.dtype) if isinstance(ctx_act, np.ndarray) else \
        ctx_act + (ctx_room == ROOM_BOSS).to(ctx_act.dtype)


def blend(q, stage, double, alpha, w):
    """Run value of D14 from milestone probabilities. q [..., N_MS] (conditional probabilities), stage [...] (0..3),
    double [...] (bool: two act-3 bosses), alpha and w scalars. Works on torch tensors (float)."""
    q = q.clone()
    q[..., 3] = torch.where(double, q[..., 3], torch.ones_like(q[..., 3]))   # below A10 the first act-3 boss is the win
    R, run = [], torch.ones_like(q[..., 0])
    for m in range(N_MS):                                     # R_{m+1}: reached stage m + 1
        run = torch.where(stage > m, torch.ones_like(run), run * q[..., m])
        R.append(run)
    R1, R2, R3, R4 = R
    return alpha * R4 + (1 - alpha) * (R1 + R2 + w * R3 + (1 - w) * R4) / 3


def banked(stage, alpha, w):
    """The outcome of losing the run at this stage (what V cannot fall under)."""
    s = np.minimum(stage, 2) if isinstance(stage, np.ndarray) else stage.clamp(max=2)
    return (1 - alpha) * (s + w * (stage >= 3)) / 3


def pad_rows(ints, floats):
    """Rows of layout v1 (ints IW_V1 = 460 wide, floats FW_V1 = 208 wide) zero-padded to the current widths; rows of the
    current widths are returned as they are (no copy). Zeros are what a v1 actor knew about the v2 inputs: nothing."""
    ints, floats = np.asarray(ints), np.asarray(floats)
    if ints.shape[-1] == IW_V1:
        p = np.zeros(ints.shape[:-1] + (IW,), ints.dtype); p[..., :IW_V1] = ints; ints = p
    if floats.shape[-1] == FW_V1:
        p = np.zeros(floats.shape[:-1] + (FW,), floats.dtype); p[..., :FW_V1] = floats; floats = p
    assert ints.shape[-1] == IW and floats.shape[-1] == FW, ("macro row widths", ints.shape, floats.shape)
    return ints, floats


def rows_to_batch(ints, floats, dev):
    """ints [B, IW] (any int dtype), floats [B, FW] float32 -> the dict MacroNet.encode expects (+ option fields).
    Rows of the v1 widths are accepted (pad_rows)."""
    ints, floats = pad_rows(ints, floats)
    it = torch.as_tensor(np.ascontiguousarray(ints)).to(dev).long()
    fl = torch.as_tensor(np.ascontiguousarray(floats)).to(dev).float()
    g = lambda k: it[:, OFF[k][0]:OFF[k][1]]
    return {
        "d_ids": g("d_ids"), "d_upg": g("d_upg").clamp(0, 3), "d_ench": g("d_ench"), "d_eamt": g("d_eamt").float(), "d_age": g("d_age").float(),
        "r_ids": g("r_ids"), "r_age": g("r_age").float(), "p_ids": g("p_ids"), "ctx": g("ctx"), "dtype": g("dtype").squeeze(1),
        "opt_kind": g("opt_kind"), "opt_id": g("opt_id"), "opt_upg": g("opt_upg"),
        "scal": fl[:, :16], "opt_feat": fl[:, 16:FW_V1].reshape(-1, K, MF),
        "opt_feat2": fl[:, F_OPT2:F_SCAL2].reshape(-1, K, MF2), "scal2": fl[:, F_SCAL2:FW], "boss": g("boss"),
    }


def load_checkpoint(path, dev):
    """S1 model.pt ({"state_dict", "vocab", "special"}) or an S3 member file of the same form."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    special = ck["special"]
    rest = [i for i, s in enumerate(special) if s.startswith("REST:")]
    net = S3MacroNet(ck["vocab"], len(special), rest).to(dev)
    net.load_s1_or_s3(ck["state_dict"])
    return net, ck


def s1_member_dirs():
    base = os.path.join(ROOT, "s1", "runs", "macro")
    return [os.path.join(base, "noattn_light")] + [os.path.join(base, f"noattn_light_seed{i}") for i in range(1, 5)]


class Ensemble:
    """The macro ensemble as one file: {"members": [state_dict...], "vocab", "special", "version"}. Member 0 is the
    policy member (PPO); every member's value head is trained on the agent's run outcomes."""

    def __init__(self, nets, vocab, special, version=1):
        self.nets, self.vocab, self.special, self.version = nets, vocab, special, version
        self.milestones, self.alpha, self.boss1_share = False, 0.0, 0.0    # set by the trainer (--milestones)

    @staticmethod
    def from_s1(dev, dirs=None):
        nets, ck = [], None
        for d in dirs or s1_member_dirs():
            n, ck = load_checkpoint(os.path.join(d, "model.pt"), dev)
            nets.append(n.eval())
        return Ensemble(nets, ck["vocab"], ck["special"])

    @staticmethod
    def load(path, dev):
        ck = torch.load(path, map_location=dev, weights_only=False)
        rest = [i for i, s in enumerate(ck["special"]) if s.startswith("REST:")]
        nets = []
        for sd in ck["members"]:
            n = S3MacroNet(ck["vocab"], len(ck["special"]), rest).to(dev)
            n.load_s1_or_s3(sd); nets.append(n.eval())
        e = Ensemble(nets, ck["vocab"], ck["special"], int(ck.get("version", 1)))
        e.milestones, e.alpha, e.boss1_share = bool(ck.get("milestones", False)), float(ck.get("alpha", 0.0)), float(ck.get("boss1_share", 0.0))
        return e

    def save(self, path):
        ck = {"members": [n.state_dict() for n in self.nets], "vocab": self.vocab, "special": self.special, "version": self.version,
              "milestones": self.milestones, "alpha": self.alpha, "boss1_share": self.boss1_share}
        torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)

    def load_members_from(self, path):
        ck = torch.load(path, map_location=next(self.nets[0].parameters()).device, weights_only=False)
        for n, sd in zip(self.nets, ck["members"]):
            n.load_s1_or_s3(sd)                           # also a file from before layout v2 (its new parameters: zero)
        self.version = int(ck.get("version", self.version + 1))
        self.milestones, self.alpha, self.boss1_share = bool(ck.get("milestones", False)), float(ck.get("alpha", 0.0)), float(ck.get("boss1_share", 0.0))

    def _ms_fill(self, o, b, nets):
        """Milestone columns of output rows o for batch b from `nets` (member means of a and c per head); returns the
        members' blended values [members, B] (probabilities) at each row's own HP."""
        A, C, V = [], [], []
        stage = stage_of(b["ctx"][:, 3], b["ctx"][:, 4]); double = b["ctx"][:, 1] >= DOUBLE_BOSS_ASC
        for n in nets:
            a, c, lg = n.forward_ms(b)
            A.append(a); C.append(c); V.append(blend(torch.sigmoid(lg), stage, double, self.alpha, self.boss1_share))
        A, C = torch.stack(A).mean(0), torch.stack(C).mean(0)
        for m in range(N_MS):
            o[:, O_MS + m * (1 + N_RAMPS)] = A[:, m]
            o[:, O_MS + m * (1 + N_RAMPS) + 1:O_MS + (m + 1) * (1 + N_RAMPS)] = C[:, m]
        o[:, O_ALPHA] = self.alpha; o[:, O_W1] = self.boss1_share; o[:, O_HASMS] = 1.0
        phi = nets[0]._hp_features(b["scal"] * nets[0].mask_always)    # as in encode(), without encoding again
        lg_mean = A + (C * phi.float().unsqueeze(1)).sum(-1)              # the formula the actor uses for its tables
        v_mean = blend(torch.sigmoid(lg_mean), stage, double, self.alpha, self.boss1_share)
        return torch.stack(V), v_mean

    @torch.no_grad()
    def serve(self, ints, floats, dev):
        """Output rows [B, OW] for the bridge. Rows that carry a decision (options, or a deck-slot decision type) are
        answered by the policy member alone (their value fields are that member's; the trainer recomputes values);
        rows without one (T_f table rows) by every member's value head."""
        ints, floats = pad_rows(ints, floats)
        B = len(ints)
        out = torch.zeros(B, OW, device=dev)
        dec = (ints[:, OFF["opt_kind"][0]] != 0) | np.isin(ints[:, OFF["dtype"][0]], DT_SLOT)
        di, vi = np.flatnonzero(dec), np.flatnonzero(~dec)
        if len(di):
            a, c, v, opt, slot = self.nets[0].forward_rows(rows_to_batch(ints[di], floats[di], dev), policy=True)
            idx = torch.from_numpy(di).to(dev)
            o = torch.zeros(len(di), OW, device=dev)
            o[:, O_A] = a; o[:, O_C:O_C + N_RAMPS] = c; o[:, O_V] = v; o[:, O_VPOL] = v
            o[:, O_OPT:O_OPT + K] = opt; o[:, O_SLOT:O_SLOT + MAX_DECK] = slot
            if self.milestones:
                _, vm = self._ms_fill(o, rows_to_batch(ints[di], floats[di], dev), self.nets[:1])
                o[:, O_V] = torch.logit(vm.clamp(1e-6, 1 - 1e-6)); o[:, O_VPOL] = o[:, O_V]
            out[idx] = o
        if len(vi):
            b = rows_to_batch(ints[vi], floats[vi], dev)
            A, C, V = [], [], []
            for n in self.nets:
                a, c, v, _, _ = n.forward_rows(b, policy=False)
                A.append(a); C.append(c); V.append(v)
            V = torch.stack(V)
            idx = torch.from_numpy(vi).to(dev)
            o = torch.zeros(len(vi), OW, device=dev)
            o[:, O_A] = torch.stack(A).mean(0); o[:, O_C:O_C + N_RAMPS] = torch.stack(C).mean(0)
            o[:, O_V] = V.mean(0); o[:, O_STD] = V.std(0) if len(self.nets) > 1 else 0.0; o[:, O_VPOL] = V[0]
            if self.milestones:
                Vm, vm = self._ms_fill(o, b, self.nets)
                L = torch.logit(Vm.clamp(1e-6, 1 - 1e-6))
                o[:, O_V] = torch.logit(vm.clamp(1e-6, 1 - 1e-6)); o[:, O_STD] = L.std(0) if len(self.nets) > 1 else 0.0; o[:, O_VPOL] = L[0]
            out[idx] = o
        return out

    @torch.no_grad()
    def value_all(self, ints, floats, dev):
        """Ensemble value logits [B] (every member) for any rows."""
        b = rows_to_batch(ints, floats, dev)
        return torch.stack([n.forward_rows(b, policy=False)[2] for n in self.nets]).mean(0)


def export_vocab(path, vocab, special):
    """Vocabulary for the C# encoder (src/s3/S3/MacroLayout.cs)."""
    import json
    json.dump({"card": vocab["card"], "ench": vocab["ench"], "relic": vocab["relic"], "potion": vocab["potion"],
               "encounter": vocab["encounter"], "special": special}, open(path, "w"))


if __name__ == "__main__":
    dev = "cpu"
    ens = Ensemble.from_s1(dev)
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
    os.makedirs(out, exist_ok=True)
    export_vocab(os.path.join(out, "macro_vocab.json"), ens.vocab, ens.special)
    ens.save(os.path.join(out, "macro_s1.pt"))
    print(f"wrote {out}/macro_vocab.json and macro_s1.pt ({len(ens.nets)} members, "
          f"{sum(p.numel() for p in ens.nets[0].parameters()):,} parameters each)")
