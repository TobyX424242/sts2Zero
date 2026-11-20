"""S3 macro network: the S1 MacroNet (src/s1/train_macro_v2.py) extended for full-game self-play.

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

Layout v3 (2026-11-16; append only again; every new path from a new input or a new parameter to an output passes a
layer that starts at zero, so a v1 / v2 file - e.g. segment 49 - computes exactly what it computed before, whatever the
new inputs hold; the inner layers of the new paths start random, ReZero / LoRA style, so that they can learn):
  - fight tokens (rows: ints f_enc / f_room, floats f_num 8 x 6): the run's last 8 live fights, most recent first
    (encounter in the id space of ctx[5], room kind 1 hallway / 2 elite / 3 boss; HP lost / max HP, HP share at the
    start, potions used / 3, turns / 20, floors ago / 20, same act). A token = encounter embedding (the existing one) +
    room embedding + linear map of the 6 numbers + a token-type vector;
  - map tokens (rows: floats map 80, each ONE integer < 2^24 stored exactly in a float32, unpacked by unpack_map): the
    nodes of the current act's map, type / row / column embeddings + a linear map of (visited, current, reachable,
    rows ahead of the current point / 16, reachable from some option) + a token-type vector; the option-reach mask
    (bit j: reachable from map option slot j) is used by the policy's cross-attention;
  - new tokens go through the existing per-token blocks on their own (the old tokens' tensor keeps its shape, so their
    encoding is bit-identical), then 2 pre-norm self-attention blocks (4 heads, d 128, FF 512) run over ALL state tokens
    with key padding; their output projection and FF second layer start at zero. The mean over the old token set is
    unchanged; fight and map tokens are pooled separately (masked means) and enter the trunk input through zero linear maps;
  - content properties (src/s3/data/content_props.json, from the simulator's model classes: `S3 props`): per card
    (at its upgrade level, plus what an upgrade adds), relic and potion a feature vector, added to the token and option
    embeddings through zero linear maps;
  - deck counts (deck_counts: deck size, Strikes, Defends, upgraded, attacks / skills / powers, curses, cost buckets,
    orb / focus / exhaust / draw / block / damage cards, relics, potions), computed here from the row's ids and the
    property table (so the lookahead's cloned rows stay consistent), enter the trunk input and the policy query
    directly through zero linear maps (the mean pooling cannot see deck size);
  - policy cross-attention: every option's representation (option embedding + query) attends over the unpooled state
    tokens; map options get a per-head bonus on the map tokens of their own subtree (option-reach bit); the result is
    added to the option embedding through a zero projection;
  - shop options: price / 300 and price / gold in opt_feat2 (a zero linear map, shop decisions only); scal2[6] is the
    price of the next card removal / 200.
  None of this depends on the row's own HP (past fights' HP is about fights already over), so z stays HP-free and the
  value stays monotone in HP. v1 and v2 rows are zero-padded by pad_rows() (no fights, no map).

Row layout shared with the C# actor (src/s3/S3/MacroLayout.cs) and the run records:
  ints  [IW = 478]: d_ids 64, d_upg 64, d_ench 64, d_eamt 64, d_age 64, r_ids 40, r_age 40, p_ids 5, ctx 6, dtype 1,
                    opt_kind 16, opt_id 16, opt_upg 16 (v1 ends here: 460), boss 2 (v2 ends here: 462), f_enc 8, f_room 8
  floats[FW = 472]: scal 16, opt_feat 16 x 12 (v1 ends here: 208), opt_feat2 16 x 8 (from 208), scal2 8 (from 336;
                    v2 ends here: 344), f_num 8 x 6 (from 344), map 80 (from 392)
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
NF, FFW, NMAP = 8, 6, 80                                          # layout v3: fights, numbers per fight, map nodes
IW_V1 = 5 * MAX_DECK + 2 * MAX_REL + MAX_POT + 6 + 1 + 3 * K       # 460
FW_V1 = 16 + K * MF                                               # 208
IW_V2 = IW_V1 + 2                                                 # 462: + the act boss, the act's second boss
F_OPT2 = FW_V1                                                    # 208: opt_feat2 [K, MF2]
F_SCAL2 = F_OPT2 + K * MF2                                        # 336: scal2 [NS2]
FW_V2 = F_SCAL2 + NS2                                             # 344
IW = IW_V2 + 2 * NF                                               # 478: + fight encounters, fight room kinds
F_FIGHT = FW_V2                                                   # 344: f_num [NF, FFW]
F_MAP = F_FIGHT + NF * FFW                                        # 392: map [NMAP] (packed integers)
FW = F_MAP + NMAP                                                 # 472
WIDTHS = {1: (IW_V1, FW_V1), 2: (IW_V2, FW_V2), 3: (IW, FW)}       # row widths of each layout
# map node packing (MacroLayout.cs MapNode.Pack): type + 1 (4 bits), row (5), column (3), visited, current, reachable,
# option-reach mask (MAP_SLOTS bits)
MAP_ROW_SHIFT, MAP_COL_SHIFT, MAP_VIS_BIT, MAP_CUR_BIT, MAP_REACH_BIT, MAP_MASK_SHIFT, MAP_SLOTS = 4, 9, 12, 13, 14, 15, 9
X_REMOVE_PRICE = 6                                                # scal2 column of the next removal price / 200
PROPS_PATH = os.path.join(HERE, "..", "data", "content_props_v113.json")    # ic113: v0.113 simulator (v0.111: content_props.json)
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
# ic113 (2026-11-18): the era token of the v0.113 simulator content. S1 eras: 0-4 from the human data (4 = v0.111), 5 = the
# era-dropout 'unknown' token; 6 = v0.113. S3MacroNet grows the era embedding to N_ERA rows; a file with 6 rows loads with
# row 6 = a copy of row 4 (load_s1_or_s3; src/ic113/convert_v113.py writes it into the file).
ERA_V0113, N_ERA = 6, 7

# offsets into the int row
_o = 0
OFF = {}
for _k, _n in (("d_ids", MAX_DECK), ("d_upg", MAX_DECK), ("d_ench", MAX_DECK), ("d_eamt", MAX_DECK), ("d_age", MAX_DECK),
               ("r_ids", MAX_REL), ("r_age", MAX_REL), ("p_ids", MAX_POT), ("ctx", 6), ("dtype", 1),
               ("opt_kind", K), ("opt_id", K), ("opt_upg", K), ("boss", 2), ("f_enc", NF), ("f_room", NF)):
    OFF[_k] = (_o, _o + _n); _o += _n
assert _o == IW and OFF["opt_upg"][1] == IW_V1 and OFF["boss"][1] == IW_V2
# parameters added by layout v2 (registered last, zero at the start; absent from v1 files)
V2_PARAMS = ("map_feat2", "boss_emb", "boss2_emb", "scal2")
# parameters added by layout v3 (registered after the v2 ones). V3_ZERO: the layers every new path passes, zero at the
# start and set to zero when a file without them is loaded (then the new paths do nothing, whatever the inner layers
# V3_INNER hold, which start random).
# (Names are module names: a parameter belongs to V3_ZERO if any component of its name is listed, e.g.
# "attn_blocks.1.attn_out.weight".)
V3_ZERO = ("card_prop", "relic_prop", "pot_prop", "attn_out", "attn_ff2", "fight_pool", "map_pool", "count_z", "count_q",
           "x_out", "shop_feat")
V3_TOP = ("card_prop", "relic_prop", "pot_prop", "fight_room", "fight_num", "fight_tok", "map_type", "map_row", "map_col",
          "map_flags", "map_tok", "attn_blocks", "fight_pool", "map_pool", "count_z", "count_q", "x_lnq", "x_lnkv", "x_q", "x_kv",
          "x_reach", "x_out", "shop_feat")


def is_v3(name):
    return name.split(".")[0] in V3_TOP


def is_v3_zero(name):
    return is_v3(name) and any(c in V3_ZERO for c in name.split("."))
ATT_HEADS, ATT_BLOCKS, ATT_FF = 4, 2, 512


# ---------------------------------------------------------------- layout v3: content properties and deck counts

CARD_TYPES = ("Attack", "Skill", "Power", "Curse", "Status", "Quest")
CARD_RARITIES = ("Basic", "Common", "Uncommon", "Rare", "Ancient", "Event", "Curse", "Quest")
CARD_TARGETS = ("AnyEnemy", "AllEnemies", "RandomEnemy", "Self", "None")
CARD_KEYWORDS = ("Exhaust", "Ethereal", "Innate", "Retain", "Sly", "Unplayable", "Eternal")
ORB_TYPES = ("LightningOrb", "FrostOrb", "DarkOrb", "PlasmaOrb", "GlassOrb")
CARD_PROBE_KEYS = ("DamageDealt", "BlockGained", "CardsDrawn", "EnergyGain", "SelfExhausted", "CardsCreated", "CreatedAttack",
                   "CreatedSkill", "CreatedPower", "CreatedOther", "Selections", "OthersExhausted", "OthersDiscarded", "HpLost",
                   "StarsChange", "PotionsGained", "Power.Enemy.WEAK_POWER", "Power.Enemy.VULNERABLE_POWER",
                   "Power.Enemy.POISON_POWER", "Power.Enemy.DOOM_POWER", "Power.Enemy.STRENGTH_POWER", "Power.Self.STRENGTH_POWER",
                   "Power.Self.DEXTERITY_POWER", "Power.Self.FOCUS_POWER", "Power.Self.ENERGY_NEXT_TURN_POWER",
                   "Power.Self.DRAW_CARDS_NEXT_TURN_POWER", "Power.Self.PLATING_POWER", "Power.Self.INTANGIBLE_POWER",
                   "Power.Self.THORNS_POWER", "Power.Self.BLOCK_NEXT_TURN_POWER", "Power.Self.VIGOR_POWER")
RELIC_RARITIES = ("Starter", "Common", "Uncommon", "Rare", "Shop", "Event", "Ancient")
POTION_RARITIES = ("Common", "Uncommon", "Rare", "Event")
POTION_USAGES = ("CombatOnly", "AnyTime", "Automatic")
POTION_TARGETS = ("AnyPlayer", "AnyEnemy", "AllEnemies", "Self")
N_ITEM_KEYS = 40                     # most frequent probe keys of relics / potions (ties: by name)
# deck counts (deck_counts; each a count over the deck's cards, then scaled)
COUNT_NAMES = ("deck", "strikes", "defends", "upgraded", "attacks", "skills", "powers", "curses", "cost0", "cost1", "cost2",
               "cost3plus", "costX", "orb_cards", "focus_cards", "exhaust_cards", "draw_cards", "block_cards", "damage_cards",
               "cost_sum", "relics", "potions")
COUNT_SCALE = (40, 5, 5, 20, 20, 20, 10, 5, 20, 20, 20, 10, 5, 10, 5, 10, 10, 20, 20, 40, 20, 5)
NC = len(COUNT_NAMES)


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _onehot(v, names):
    return [1.0 if v == n else 0.0 for n in names] + [0.0 if v in names else 1.0]


def _card_level_feats(c, L):
    """Feature vector of one card at one upgrade level (content_props.json entry c, level entry L) and its count
    contributions (COUNT_NAMES order, before scaling; deck / relic / potion counts are filled elsewhere)."""
    cost, x, unpl = int(L.get("cost", 0)), bool(L.get("x_cost")), bool(L.get("unplayable"))
    kw = set(L.get("keywords") or []); tags = set(c.get("tags") or [])
    pv = dict((L.get("probe") or {}).get("values") or {})
    pv["EnergyGain"] = pv.get("EnergyChange", 0.0) + (0 if x or unpl else cost)     # energy gained besides the cost paid
    orbs = L.get("orbs") or {}
    cost_b = [float(not unpl and not x and cost == 0), float(not unpl and not x and cost == 1), float(not unpl and not x and cost == 2),
              float(not unpl and not x and cost >= 3), float(x), float(unpl)]
    f = (cost_b + _onehot(c.get("type"), CARD_TYPES) + _onehot(c.get("rarity"), CARD_RARITIES) + _onehot(c.get("target"), CARD_TARGETS)
         + [float(k in kw) for k in CARD_KEYWORDS] + [float("Strike" in tags), float("Defend" in tags), float(bool(c.get("colorless")))]
         + [float(_slog(pv.get(k, 0.0))) for k in CARD_PROBE_KEYS]
         + [float(_slog(sum(v for k, v in pv.items() if k.startswith("Power.Self.") and k not in CARD_PROBE_KEYS))),
            float(_slog(sum(v for k, v in pv.items() if k.startswith("Power.Enemy.") and k not in CARD_PROBE_KEYS)))]
         + [float(orbs.get(o, 0)) / 3 for o in ORB_TYPES] + [float(sum(v for k, v in orbs.items() if k in ORB_TYPES)) / 3]
         + [float((L.get("probe") or {}).get("unsupported") is not None)])
    t = c.get("type"); basic = c.get("rarity") == "Basic"
    cnt = np.zeros(NC, np.float32)
    cnt[1] = float(basic and "Strike" in tags); cnt[2] = float(basic and "Defend" in tags)
    cnt[4] = float(t == "Attack"); cnt[5] = float(t == "Skill"); cnt[6] = float(t == "Power"); cnt[7] = float(t in ("Curse", "Status"))
    cnt[8:13] = cost_b[:5]
    cnt[13] = float(sum(v for k, v in orbs.items() if k in ORB_TYPES) > 0); cnt[14] = float(pv.get("Power.Self.FOCUS_POWER", 0) > 0)
    cnt[15] = float("Exhaust" in kw or pv.get("SelfExhausted", 0) > 0); cnt[16] = float(pv.get("CardsDrawn", 0) > 0)
    cnt[17] = float(pv.get("BlockGained", 0) > 0); cnt[18] = float(pv.get("DamageDealt", 0) > 0)
    cnt[19] = float(cost if not (x or unpl) else 0)
    return np.asarray(f, np.float32), cnt


def _item_feats(d, names, cats):
    """Relic / potion features: one-hots of the categorical fields `cats` [(field, values)], the probe keys `names`
    (signed log), and flags: probe unsupported, id missing in the simulator."""
    pv = (d.get("probe") or {}).get("values") or {}
    f = []
    for field, vals in cats:
        f += _onehot(d.get(field), vals)
    f += [float(_slog(pv.get(k, 0.0))) for k in names]
    f += [float((d.get("probe") or {}).get("unsupported") is not None), float(bool(d.get("missing")))]
    return np.asarray(f, np.float32)


def load_props(vocab, path=PROPS_PATH):
    """The property tables for a vocabulary: card [V_card, 2, Dc] (upgrade level 0 / 1: the level's features followed by
    what an upgrade adds, level 1 - level 0), card counts [V_card, 2, NC], relic [V_relic, Dr], potion [V_pot, Dp], and a
    short digest of the file. Ids without an entry (padding, unknown, missing in the simulator) get zeros except their
    flags. Without the file every table is zero (the network then has no property input) and the digest is ''."""
    import hashlib, json
    if not os.path.exists(path):
        print(f"[macro_s3] {path} not found: content properties are zero", flush=True)
        d = {"card": {}, "relic": {}, "potion": {}}; digest = ""
    else:
        raw = open(path, "rb").read(); d = json.loads(raw); digest = hashlib.sha1(raw).hexdigest()[:12]
    from collections import Counter

    def top_keys(kind):
        c = Counter(k for v in d[kind].values() for k in ((v.get("probe") or {}).get("values") or {}))
        return [k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:N_ITEM_KEYS]]
    # ic113: a file may pin the relic / potion probe-key columns ("item_keys", written by src/ic113/convert_v113.py from
    # the v0.111 file) so that a network trained on the old columns reads the same features from the new content
    pinned = d.get("item_keys") or {}
    rk, pk = pinned.get("relic") or top_keys("relic"), pinned.get("potion") or top_keys("potion")
    dummy_c = {"type": None, "rarity": None, "target": None}
    base_len = len(_card_level_feats(dummy_c, {})[0])
    card = np.zeros((len(vocab["card"]), 2, 2 * base_len), np.float32); counts = np.zeros((len(vocab["card"]), 2, NC), np.float32)
    for i, cid in enumerate(vocab["card"]):
        c = d["card"].get(cid)
        if not c or not c.get("levels"):
            continue
        lv = [_card_level_feats(c, L) for L in c["levels"]]
        if len(lv) == 1:
            lv.append(lv[0])
        delta = lv[1][0] - lv[0][0]
        for u in range(2):
            card[i, u, :base_len] = lv[u][0]; card[i, u, base_len:] = delta; counts[i, u] = lv[u][1]
    rcats = (("rarity", RELIC_RARITIES),)
    pcats = (("rarity", POTION_RARITIES), ("usage", POTION_USAGES), ("target", POTION_TARGETS))
    relic = np.stack([_item_feats(d["relic"].get(r, {}), rk, rcats) if r in d["relic"] else np.zeros(len(_item_feats({}, rk, rcats)), np.float32)
                      for r in vocab["relic"]])
    pot = np.stack([_item_feats(d["potion"].get(r, {}), pk, pcats) if r in d["potion"] else np.zeros(len(_item_feats({}, pk, pcats)), np.float32)
                    for r in vocab["potion"]])
    return {"card": card, "counts": counts, "relic": relic, "potion": pot, "digest": digest}


def unpack_map(m):
    """Packed map nodes (long tensor [..., NMAP], from the float columns) -> dict of long tensors: type (map point type
    + 1; 0 = no node), row, col, visited, current, reach (from the current point), mask (option-reach bits)."""
    return {"type": m & 15, "row": (m >> MAP_ROW_SHIFT) & 31, "col": (m >> MAP_COL_SHIFT) & 7, "visited": (m >> MAP_VIS_BIT) & 1,
            "current": (m >> MAP_CUR_BIT) & 1, "reach": (m >> MAP_REACH_BIT) & 1, "mask": (m >> MAP_MASK_SHIFT) & ((1 << MAP_SLOTS) - 1)}


class Vec(nn.Module):
    """One learned vector as a module of its own: a parameter set directly on the network would be listed before every
    submodule's parameters and break the order the trainer restores optimizer states by."""

    def __init__(self, x):
        super().__init__()
        self.v = nn.Parameter(x)


class AttnBlock(nn.Module):
    """Pre-norm self-attention block over the state tokens (key padding); output projection and FF second layer start at
    zero, so the block is the identity until they move."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.attn_ln1 = nn.LayerNorm(d); self.attn_qkv = nn.Linear(d, 3 * d); self.attn_out = nn.Linear(d, d)
        self.attn_ln2 = nn.LayerNorm(d); self.attn_ff1 = nn.Linear(d, ff); self.attn_ff2 = nn.Linear(ff, d)
        for m in (self.attn_out, self.attn_ff2):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, keep):
        """x [B, L, d]; keep [B, L] bool (True = a real token: a key that may be attended to)."""
        B, L, d = x.shape
        q, k, v = self.attn_qkv(self.attn_ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        x = x + self.attn_out(a.transpose(1, 2).reshape(B, L, d))
        return x + self.attn_ff2(F.gelu(self.attn_ff1(self.attn_ln2(x))))


class P2Adapter(nn.Module):
    """Opt-in P2 residual adapter (REPR_TEST_PLAN.md / LAPTOP_GPU_PROPOSAL.md T2): small zero-init residual MLPs
    reading stopgrad(mean of members 1-4's state z, option embeddings e, and deck-slot tokens hs).
    Every output layer starts at exact zero so loading an existing checkpoint gives bit-identical policy logits at step 0."""

    def __init__(self, d):
        super().__init__()
        self.ln_z = nn.LayerNorm(d); self.z_fc1 = nn.Linear(d, d); self.z_fc2 = nn.Linear(d, d)
        self.ln_e = nn.LayerNorm(d); self.e_fc1 = nn.Linear(d, d); self.e_fc2 = nn.Linear(d, d)
        self.ln_s = nn.LayerNorm(d); self.s_fc1 = nn.Linear(d, d); self.s_fc2 = nn.Linear(d, d)
        for m in (self.z_fc2, self.e_fc2, self.s_fc2):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def delta_z(self, z_val):
        return self.z_fc2(F.gelu(self.z_fc1(self.ln_z(z_val))))

    def delta_e(self, e_val):
        return self.e_fc2(F.gelu(self.e_fc1(self.ln_e(e_val))))

    def delta_s(self, s_val):
        return self.s_fc2(F.gelu(self.s_fc1(self.ln_s(s_val))))


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
        # layout v3 (see the header), registered after the v2 parameters, in this order (optimizer states of older carry
        # files are restored by position). The property tables are buffers built from content_props.json (not saved).
        pr = load_props(vocab)
        self.props_digest = pr["digest"]
        self.register_buffer("card_props", torch.from_numpy(pr["card"]), persistent=False)
        self.register_buffer("card_counts", torch.from_numpy(pr["counts"]), persistent=False)
        self.register_buffer("relic_props", torch.from_numpy(pr["relic"]), persistent=False)
        self.register_buffer("pot_props", torch.from_numpy(pr["potion"]), persistent=False)
        self.register_buffer("count_scale", torch.tensor(COUNT_SCALE, dtype=torch.float32), persistent=False)
        z = lambda m: (nn.init.zeros_(m.weight), m)[1]
        self.card_prop = z(nn.Linear(pr["card"].shape[-1], d, bias=False))
        self.relic_prop = z(nn.Linear(pr["relic"].shape[-1], d, bias=False))
        self.pot_prop = z(nn.Linear(pr["potion"].shape[-1], d, bias=False))
        self.fight_room = nn.Embedding(4, d); self.fight_num = nn.Linear(FFW, d)
        self.fight_tok = Vec(torch.randn(d) * 0.02)
        self.map_type = nn.Embedding(16, d, padding_idx=0); self.map_row = nn.Embedding(32, d); self.map_col = nn.Embedding(8, d)
        self.map_flags = nn.Linear(5, d); self.map_tok = Vec(torch.randn(d) * 0.02)
        self.attn_blocks = nn.ModuleList(AttnBlock(d, ATT_HEADS, ATT_FF) for _ in range(ATT_BLOCKS))
        self.fight_pool = z(nn.Linear(d, d, bias=False)); self.map_pool = z(nn.Linear(d, d, bias=False))
        self.count_z = z(nn.Linear(NC, d, bias=False)); self.count_q = z(nn.Linear(NC, d, bias=False))
        self.x_lnq = nn.LayerNorm(d); self.x_lnkv = nn.LayerNorm(d); self.x_q = nn.Linear(d, d); self.x_kv = nn.Linear(d, 2 * d)
        self.x_reach = Vec(torch.full((ATT_HEADS,), 2.0))     # logit bonus of a map option's own subtree, per head
        self.x_out = nn.Linear(d, d); nn.init.zeros_(self.x_out.weight); nn.init.zeros_(self.x_out.bias)
        self.shop_feat = z(nn.Linear(MF2, d, bias=False))
        # ic113: era embedding grown from S1's 6 rows to N_ERA (row ERA_V0113 = a copy of row ERA_V0111). Done last so the
        # random init of every other parameter is drawn as before; the module keeps its place in the parameter order.
        if self.era.num_embeddings < N_ERA:
            old_era = self.era
            self.era = nn.Embedding(N_ERA, d)
            with torch.no_grad():
                self.era.weight[:old_era.num_embeddings] = old_era.weight
                self.era.weight[ERA_V0113] = old_era.weight[ERA_V0111]
        self.p2 = None   # opt-in P2Adapter (enable_p2); absent by default so state_dict & param order are unchanged

    def enable_p2(self):
        """Attach the zero-init P2 residual adapter at the end of the module list (preserving earlier param order)."""
        if self.p2 is None:
            dev = next(self.parameters()).device
            self.p2 = P2Adapter(self.card.embedding_dim).to(dev)
        return self.p2

    def v3_zero_params(self):
        return [p for k, p in self.named_parameters() if is_v3_zero(k)]

    def load_s1_or_s3(self, sd):
        """Loads an S1 MacroNet state dict (new parameters keep their zero start) or an S3 one."""
        sd = dict(sd)
        if "era.weight" in sd and sd["era.weight"].shape[0] < self.era.num_embeddings:
            # a v0.111 file (6 era rows): the v0.113 era starts as the v0.111 era (ic113)
            w = sd["era.weight"]; e = torch.zeros(self.era.num_embeddings, w.shape[1], dtype=w.dtype)
            e[:w.shape[0]] = w; e[ERA_V0113] = w[ERA_V0111]
            sd["era.weight"] = e
        if sd["dt_emb.weight"].shape[0] < len(DT):
            w = torch.zeros(len(DT), sd["dt_emb.weight"].shape[1], dtype=sd["dt_emb.weight"].dtype)
            w[:sd["dt_emb.weight"].shape[0]] = sd["dt_emb.weight"]
            sd["dt_emb.weight"] = w
        if any(k.startswith("p2.") for k in sd):
            self.enable_p2()
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(m.split(".")[0] in ("map_emb", "map_feat", "event_emb", "hp_pol", "ms_win", "ms_mono", "p2") + V2_PARAMS or is_v3(m)
                   for m in missing), missing
        # a file without the layout v2 / v3 parameters is the function without them: their zero layers are zero, also
        # when this network held trained ones before (a server that reloads an older file); the inner v3 layers keep
        # whatever they hold (they cannot reach an output through a zero layer)
        own = dict(self.named_parameters())
        with torch.no_grad():
            for m in missing:
                if m.split(".")[0] in V2_PARAMS or is_v3_zero(m):
                    own[m].zero_()

    def encode(self, b):
        """State vector z, HP ramp features phi, per-token outputs h (see encode_full)."""
        z, phi, h, _ = self.encode_full(b)
        return z, phi, h

    def deck_counts(self, b):
        """Layout v3 deck counts [B, NC] (COUNT_NAMES, scaled) from the row's card ids / upgrades, relics and potions."""
        ids = b["d_ids"].long().clamp(0, self.card_counts.shape[0] - 1)
        c = self.card_counts[ids, (b["d_upg"] > 0).long()].sum(1)                     # [B, NC]
        real = (ids > 0).float()
        c[:, 0] = real.sum(1); c[:, 3] = ((b["d_upg"] > 0).float() * real).sum(1)
        c[:, NC - 2] = (b["r_ids"] > 0).float().sum(1); c[:, NC - 1] = (b["p_ids"] > 0).float().sum(1)
        return c / self.count_scale

    def encode_full(self, b):
        """tm.MacroNet.encode (the same code) with the layout v2 state inputs (the boss embeddings are added to the act
        context token, the extra scalars to the scalar token) and the layout v3 inputs (header). The old tokens are
        encoded in a tensor of their old shape, so that with the v3 zero layers at zero every output is bit-identical to
        the v2 network's. The inputs are HP-free (as every encoder input), so z does not depend on the row's HP and the
        value heads stay monotone in HP. A batch without the v2 / v3 fields (not built by rows_to_batch) is encoded as
        before. Returns z, phi, h [B, L, d] (old tokens first: CLS, 6 context, scalar, 64 deck, 40 relic, 5 potion; then
        the fight and map tokens of the batch), and aux (keep mask [B, L], map unpacked, token offsets, counts)."""
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
        v3 = "map" in b
        card_tok = (self.card(b["d_ids"]) + self.upg(b["d_upg"]) + self.ench(b["d_ench"])
                    + self.card_num(torch.stack([b["d_eamt"] / 20.0, b["d_age"] / 63.0], -1).float()) + T[3])
        relic_tok = self.relic(b["r_ids"]) + self.relic_num((b["r_age"] / 63.0).unsqueeze(-1).float()) + T[4]
        pot_tok = self.potion(b["p_ids"]) + T[5]
        if v3:                                            # content properties (zero layers at the start)
            card_tok = card_tok + self.card_prop(self.card_props[b["d_ids"].long(), (b["d_upg"] > 0).long()]).to(card_tok.dtype)
            relic_tok = relic_tok + self.relic_prop(self.relic_props[b["r_ids"].long()]).to(relic_tok.dtype)
            pot_tok = pot_tok + self.pot_prop(self.pot_props[b["p_ids"].long()]).to(pot_tok.dtype)
        x = torch.cat([self.cls.expand(B, -1, -1) + T[0], ctx_tok, scal_tok, card_tok, relic_tok, pot_tok], 1)
        pad = torch.cat([torch.zeros(B, 8, dtype=torch.bool, device=x.device), b["d_ids"] == 0, b["r_ids"] == 0, b["p_ids"] == 0], 1)
        if not self.use_cls:
            pad[:, 0] = True                      # CLS excluded from pooling
        h = self.encoder(x, src_key_padding_mask=pad) if self.use_cls else self.encoder(x)
        valid = (~pad).unsqueeze(-1).to(h.dtype)
        mean = (h * valid).sum(1) / valid.sum(1)
        pooled = torch.cat([h[:, 0], mean], -1) if self.use_cls else mean
        if not v3:
            return self.trunk(self.norm(pooled)), phi, h, None
        # ---- layout v3: fight and map tokens (their own pass through the per-token blocks), self-attention over all
        # tokens, separate pools and the deck counts into the trunk input
        L0 = h.shape[1]
        # b["static"] (macro_graphs.py, CUDA graph capture): every fight and map slot, no host sync on the batch's own
        # largest count; the slots beyond a row's count are masked as everywhere else (same function, other shapes)
        static = bool(b.get("static", False))
        nf = NF if static else max(1, int((b["f_room"] > 0).sum(1).max()))
        f_room = b["f_room"][:, :nf].clamp(0, 3); f_keep = f_room > 0
        f_tok = (self.enc(b["f_enc"][:, :nf].clamp(0, self.enc.num_embeddings - 1)) + self.fight_room(f_room)
                 + self.fight_num(b["f_num"][:, :nf].float()) + self.fight_tok.v)
        mp = b["map"]
        nm = NMAP if static else max(1, int((mp > 0).sum(1).max()))
        u = unpack_map(mp[:, :nm]); m_keep = u["type"] > 0
        cur_row = torch.where(u["current"] > 0, u["row"], torch.zeros_like(u["row"])).max(1, keepdim=True).values
        has_cur = (u["current"] > 0).any(1, keepdim=True)
        rel = torch.where(has_cur, (u["row"] - cur_row).float() / 16.0, torch.zeros_like(u["row"], dtype=torch.float32))
        flags = torch.stack([u["visited"].float(), u["current"].float(), u["reach"].float(), rel, (u["mask"] > 0).float()], -1)
        m_tok = self.map_type(u["type"]) + self.map_row(u["row"]) + self.map_col(u["col"]) + self.map_flags(flags) + self.map_tok.v
        hn = self.encoder(torch.cat([f_tok, m_tok], 1).to(h.dtype))
        h = torch.cat([h, hn], 1)
        keep = torch.cat([~pad, f_keep, m_keep], 1)
        for blk in self.attn_blocks:
            h = blk(h, keep)
        kf, km = f_keep.unsqueeze(-1).to(h.dtype), m_keep.unsqueeze(-1).to(h.dtype)
        mean = (h[:, :L0] * valid).sum(1) / valid.sum(1)          # the old pooling over the old token set
        pooled = torch.cat([h[:, 0], mean], -1) if self.use_cls else mean
        f_pool = (h[:, L0:L0 + nf] * kf).sum(1) / kf.sum(1).clamp(min=1)
        m_pool = (h[:, L0 + nf:] * km).sum(1) / km.sum(1).clamp(min=1)
        counts = self.deck_counts(b)
        zin = self.norm(pooled) + self.fight_pool(f_pool) + self.map_pool(m_pool) + self.count_z(counts)
        aux = {"keep": keep, "map": u, "m0": L0 + nf, "nm": nm, "counts": counts}
        return self.trunk(zin), phi, h, aux

    def opt_emb_s3_static(self, kind, oid, upg, feat, feat2=None, dtype=None):
        """opt_emb_s3 without boolean indexing (no data-dependent shapes or host syncs, for CUDA graph capture): every
        term is computed for every option and added where the option is of its kind (adding 0 elsewhere). Per option
        the same sums in the same order; only the matrix shapes differ (TF32 kernel choice)."""
        e = torch.zeros(*kind.shape, self.card.embedding_dim, device=kind.device, dtype=torch.float32)
        at = lambda m, x: torch.where(m.unsqueeze(-1), x.float(), 0.0)
        for k, fn, n in ((1, self.card, self.card.num_embeddings), (2, self.relic, self.relic.num_embeddings),
                         (3, self.potion, self.potion.num_embeddings), (4, self.special, self.special.num_embeddings),
                         (5, self.map_emb, N_MAP_TYPES), (6, self.event_emb, N_EVENT_BUCKETS)):
            e = e + at(kind == k, fn(oid.clamp(0, n - 1)))
        e = e + at(kind == 1, self.upg(upg.clamp(0, 3)))
        m = kind == 5
        e = e + at(m, self.map_feat(feat))
        if feat2 is not None:
            e = e + at(m, self.map_feat2(feat2))
        if dtype is not None:
            for k, tab, lin in ((1, None, self.card_prop), (2, self.relic_props, self.relic_prop), (3, self.pot_props, self.pot_prop)):
                i = oid.clamp(0, (self.card_props if k == 1 else tab).shape[0] - 1)
                pf = self.card_props[i, (upg > 0).long()] if k == 1 else tab[i]
                e = e + at(kind == k, lin(pf))
            m = (dtype == 6).unsqueeze(1) & (kind >= 1) & (kind <= 4)
            if feat2 is not None:
                e = e + at(m, self.shop_feat(feat2))
        return e

    def opt_emb_s3(self, kind, oid, upg, feat, feat2=None, dtype=None):
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
        if dtype is not None:                             # layout v3 (zero layers at the start)
            for k, tab, lin in ((1, None, self.card_prop), (2, self.relic_props, self.relic_prop), (3, self.pot_props, self.pot_prop)):
                m = kind == k
                if m.any():
                    i = oid[m].clamp(0, (self.card_props if k == 1 else tab).shape[0] - 1)
                    pf = self.card_props[i, (upg[m] > 0).long()] if k == 1 else tab[i]
                    e[m] = e[m] + lin(pf).float()
            m = (dtype == 6).unsqueeze(1) & (kind >= 1) & (kind <= 4)
            if m.any() and feat2 is not None:
                e[m] = e[m] + self.shop_feat(feat2[m]).float()
        return e

    def cross(self, e, q, h, aux, kind):
        """Layout v3 policy cross-attention: each option (embedding e [B, K, d] + query q) attends over the state tokens
        h [B, L, d]; a map option (kind 5) in slot j < MAP_SLOTS gets x_reach[head] added on the map tokens whose
        option-reach bit j is set (its own subtree). Returns the zero-initialised projection of the result [B, K, d]."""
        B, Kk, d = e.shape; L = h.shape[1]; H = ATT_HEADS; dh = d // H
        Q = self.x_q(self.x_lnq(e + q.unsqueeze(1))).view(B, Kk, H, dh).transpose(1, 2)           # [B, H, K, dh]
        kv = self.x_kv(self.x_lnkv(h)).view(B, L, 2, H, dh).permute(2, 0, 3, 1, 4)              # [2, B, H, L, dh]
        bias = torch.zeros(B, 1, Kk, L, device=h.device, dtype=Q.dtype)
        bias = bias.masked_fill(~aux["keep"][:, None, None, :], float("-inf"))
        mask = aux["map"]["mask"]                                                             # [B, nm]
        js = torch.arange(Kk, device=h.device)
        bit = ((mask.unsqueeze(1) >> js.clamp(max=MAP_SLOTS - 1)[None, :, None]) & 1).bool()     # [B, K, nm]
        bit &= ((kind == 5) & (js < MAP_SLOTS)[None, :]).unsqueeze(-1)
        reach = torch.zeros(B, Kk, L, device=h.device, dtype=Q.dtype)
        reach[:, :, aux["m0"]:aux["m0"] + aux["nm"]] = bit.to(Q.dtype)
        bias = bias + self.x_reach.v.view(1, H, 1, 1).to(Q.dtype) * reach.unsqueeze(1)
        a = F.scaled_dot_product_attention(Q, kv[0], kv[1], attn_mask=bias)
        return self.x_out(a.transpose(1, 2).reshape(B, Kk, d))

    def forward_ms(self, b, detach=False):
        """Milestone heads: a [B, N_MS], c [B, N_MS, 26] (softplus), logits at the row's own HP [B, N_MS]."""
        z, phi, h = self.encode(b)
        return self.ms_from(z.detach() if detach else z, phi)

    def ms_from(self, z, phi):
        a = self.ms_win(z).float()
        c = F.softplus(self.ms_mono(z).float()).view(-1, N_MS, N_RAMPS)
        return a, c, a + (c * phi.float().unsqueeze(1)).sum(-1)

    def forward_rows(self, b, policy=True, detach=False, p2_ctx=None):
        """b: dict from rows_to_batch. Returns a [B], c [B, 26], logit at own HP [B], and (if policy) option logits
        [B, K] (kind 7 = 0, padding = -1e9) and deck-slot logits [B, 64] (unmasked). detach: the value heads read the
        state vector without sending gradients into the encoder (used for the policy member's value loss).
        p2_ctx: optional tuple (z_val, e_val, s_val) from stopgrad(EMA(mean members 1-4)) when P2 is enabled."""
        z, phi, h, aux = self.encode_full(b)
        if detach:
            z = z.detach()
        a = self.win_head(z).squeeze(-1).float()
        c = F.softplus(self.mono(z).float())
        v = a + (c * phi).sum(-1)
        if not policy:
            return a, c, v, None, None
        q = z + self.dt_emb(b["dtype"].clamp(0, len(DT) - 1)) + self.hp_pol(phi.float()).to(z.dtype)
        if p2_ctx is not None and self.p2 is not None:
            q = q + self.p2.delta_z(p2_ctx[0].to(q.dtype)).to(q.dtype)
        kind = b["opt_kind"]
        if aux is not None:                               # layout v3: deck counts into the query, option cross-attention
            q = q + self.count_q(aux["counts"]).to(q.dtype)
            emb = self.opt_emb_s3_static if b.get("static", False) else self.opt_emb_s3
            e = emb(kind, b["opt_id"], b["opt_upg"], b["opt_feat"], b.get("opt_feat2"), b["dtype"])
            if p2_ctx is not None and self.p2 is not None:
                e = e + self.p2.delta_e(p2_ctx[1].float()).float()
            e = e + self.cross(e, q, h, aux, kind).float()
        else:
            e = self.opt_emb_s3(kind, b["opt_id"], b["opt_upg"], b["opt_feat"], b.get("opt_feat2"))
            if p2_ctx is not None and self.p2 is not None:
                e = e + self.p2.delta_e(p2_ctx[1].float()).float()
        qe = q.unsqueeze(1).expand_as(e).float()
        opt = self.opt_mlp(torch.cat([qe, e, qe * e], -1)).squeeze(-1).float()
        opt = torch.where(kind == 7, torch.zeros_like(opt), opt).masked_fill(kind == 0, -1e9)
        hs = h[:, 8:8 + MAX_DECK].float()
        if p2_ctx is not None and self.p2 is not None:
            hs = hs + self.p2.delta_s(p2_ctx[2].float()).float()
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
    """Rows of layout v1 (ints 460, floats 208) or v2 (462, 344) zero-padded to the current widths; rows of the current
    widths are returned as they are (no copy). Zeros are what an older actor knew about the newer inputs: nothing (no
    boss ids, no fights, no map nodes)."""
    ints, floats = np.asarray(ints), np.asarray(floats)
    if ints.shape[-1] in (IW_V1, IW_V2):
        p = np.zeros(ints.shape[:-1] + (IW,), ints.dtype); p[..., :ints.shape[-1]] = ints; ints = p
    if floats.shape[-1] in (FW_V1, FW_V2):
        p = np.zeros(floats.shape[:-1] + (FW,), floats.dtype); p[..., :floats.shape[-1]] = floats; floats = p
    assert ints.shape[-1] == IW and floats.shape[-1] == FW, ("macro row widths", ints.shape, floats.shape)
    return ints, floats


def decision_rows(ints):
    """Bool [B]: rows that carry a decision (options, or a deck-slot decision type); the others are value rows."""
    return (ints[:, OFF["opt_kind"][0]] != 0) | np.isin(ints[:, OFF["dtype"][0]], DT_SLOT)


def rows_to_batch(ints, floats, dev):
    """ints [B, IW] (any int dtype), floats [B, FW] float32 -> the dict MacroNet.encode expects (+ option fields).
    Rows of the v1 / v2 widths are accepted (pad_rows). The packed map nodes are float32 columns holding integers below
    2^24 (exact); they are read back as integers here."""
    ints, floats = pad_rows(ints, floats)
    it = torch.as_tensor(np.ascontiguousarray(ints)).to(dev).long()
    fl = torch.as_tensor(np.ascontiguousarray(floats)).to(dev).float()
    return tensors_to_batch(it, fl)


def tensors_to_batch(it, fl):
    """rows_to_batch for rows already on the device: it [B, IW] long, fl [B, FW] float32 (macro_graphs.py builds the
    batch from its static input buffers inside the captured graph)."""
    g = lambda k: it[:, OFF[k][0]:OFF[k][1]]
    return {
        "d_ids": g("d_ids"), "d_upg": g("d_upg").clamp(0, 3), "d_ench": g("d_ench"), "d_eamt": g("d_eamt").float(), "d_age": g("d_age").float(),
        "r_ids": g("r_ids"), "r_age": g("r_age").float(), "p_ids": g("p_ids"), "ctx": g("ctx"), "dtype": g("dtype").squeeze(1),
        "opt_kind": g("opt_kind"), "opt_id": g("opt_id"), "opt_upg": g("opt_upg"),
        "scal": fl[:, :16], "opt_feat": fl[:, 16:FW_V1].reshape(-1, K, MF),
        "opt_feat2": fl[:, F_OPT2:F_SCAL2].reshape(-1, K, MF2), "scal2": fl[:, F_SCAL2:FW_V2], "boss": g("boss"),
        "f_enc": g("f_enc"), "f_room": g("f_room"), "f_num": fl[:, F_FIGHT:F_MAP].reshape(-1, NF, FFW),
        "map": fl[:, F_MAP:FW].round().long(),
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


def parse_value_members(spec, n):
    """Parse --value-members spec ('' / 'all' -> None = all members; '1,2,3,4' -> tuple of member indices)."""
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        idx = tuple(int(i) for i in spec)
    else:
        s = str(spec).strip()
        if not s or s.lower() == "all":
            return None
        idx = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    if not idx or any(i < 0 or i >= n for i in idx):
        raise ValueError(f"invalid value_members {spec!r} for {n} members")
    return idx


class Ensemble:
    """The macro ensemble as one file: {"members": [state_dict...], "vocab", "special", "version"}. Member 0 is the
    policy member (PPO); every member's value head is trained on the agent's run outcomes."""

    def __init__(self, nets, vocab, special, version=1, value_members=None):
        self.nets, self.vocab, self.special, self.version = nets, vocab, special, version
        self.milestones, self.alpha, self.boss1_share = False, 0.0, 0.0    # set by the trainer (--milestones)
        self.value_members = parse_value_members(value_members, len(nets))
        self.p2, self.p2_ema_nets = False, None                            # opt-in P2 (--p2 / --p2-ema)

    def set_value_members(self, spec):
        self.value_members = parse_value_members(spec, len(self.nets))
        return self.value_members

    def enable_p2(self, use_ema=False):
        """Enable opt-in P2 adapter on member 0 (and optional frozen EMA copies of members 1-4)."""
        import copy
        self.nets[0].enable_p2()
        self.p2 = True
        if use_ema and self.p2_ema_nets is None and len(self.nets) > 1:
            self.p2_ema_nets = [copy.deepcopy(n).eval() for n in self.nets[1:]]
            for n in self.p2_ema_nets:
                for p in n.parameters():
                    p.requires_grad_(False)
        return self

    @torch.no_grad()
    def p2_context(self, b, src_nets=None):
        """Mean (z, e, hs) across value members (EMA copies when present, else members 1-4), detached."""
        src = src_nets if src_nets is not None else (self.p2_ema_nets if self.p2_ema_nets is not None else self.nets[1:])
        if not src:
            return None
        zs, es, ss = [], [], []
        static = bool(b.get("static", False))
        for n in src:
            z, _, h, aux = n.encode_full(b)
            emb = n.opt_emb_s3_static if (aux is not None and static) else n.opt_emb_s3
            e = emb(b["opt_kind"], b["opt_id"], b["opt_upg"], b["opt_feat"], b.get("opt_feat2"),
                    b["dtype"] if aux is not None else None)
            zs.append(z.float()); es.append(e.float()); ss.append(h[:, 8:8 + MAX_DECK].float())
        return torch.stack(zs).mean(0).detach(), torch.stack(es).mean(0).detach(), torch.stack(ss).mean(0).detach()

    @staticmethod
    def from_s1(dev, dirs=None):
        nets, ck = [], None
        for d in dirs or s1_member_dirs():
            n, ck = load_checkpoint(os.path.join(d, "model.pt"), dev)
            nets.append(n.eval())
        return Ensemble(nets, ck["vocab"], ck["special"])

    @staticmethod
    def load(path, dev):
        return Ensemble.from_ck(torch.load(path, map_location=dev, weights_only=False), dev, path)

    @staticmethod
    def from_ck(ck, dev, path=""):
        """The ensemble of a loaded file (dict); path only for the message."""
        rest = [i for i, s in enumerate(ck["special"]) if s.startswith("REST:")]
        nets = []
        for sd in ck["members"]:
            n = S3MacroNet(ck["vocab"], len(ck["special"]), rest).to(dev)
            n.load_s1_or_s3(sd); nets.append(n.eval())
        e = Ensemble(nets, ck["vocab"], ck["special"], int(ck.get("version", 1)))
        e.milestones, e.alpha, e.boss1_share = bool(ck.get("milestones", False)), float(ck.get("alpha", 0.0)), float(ck.get("boss1_share", 0.0))
        if ck.get("p2"):
            e.p2 = True
            e.nets[0].enable_p2()
            if ck.get("p2_ema"):
                e.p2_ema_nets = []
                for sd in ck["p2_ema"]:
                    n = S3MacroNet(ck["vocab"], len(ck["special"]), rest).to(dev)
                    n.load_s1_or_s3(sd); n.eval()
                    for p in n.parameters():
                        p.requires_grad_(False)
                    e.p2_ema_nets.append(n)
        if ck.get("props_digest") and ck["props_digest"] != nets[0].props_digest:
            print(f"[macro_s3] warning: {path} was trained with content properties {ck['props_digest']}, this tree has "
                  f"{nets[0].props_digest or 'none'}", flush=True)
        return e

    def save(self, path):
        ck = {"members": [n.state_dict() for n in self.nets], "vocab": self.vocab, "special": self.special, "version": self.version,
              "milestones": self.milestones, "alpha": self.alpha, "boss1_share": self.boss1_share,
              "layout": 3, "props_digest": self.nets[0].props_digest}
        if self.p2:
            ck["p2"] = True
            if self.p2_ema_nets is not None:
                ck["p2_ema"] = [n.state_dict() for n in self.p2_ema_nets]
        torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)

    def load_members_from(self, path):
        self.load_members_ck(torch.load(path, map_location=next(self.nets[0].parameters()).device, weights_only=False))

    def load_members_ck(self, ck):
        """Weights of a loaded file (dict) copied into the existing parameter tensors (load_state_dict: in place, so
        captured CUDA graphs read them at their next replay)."""
        for n, sd in zip(self.nets, ck["members"]):
            n.load_s1_or_s3(sd)                           # also a file from before layout v2 (its new parameters: zero)
        if self.p2 and self.p2_ema_nets is not None and ck.get("p2_ema"):
            for n, sd in zip(self.p2_ema_nets, ck["p2_ema"]):
                n.load_s1_or_s3(sd)
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
        rows without one (T_f table rows) by every member's value head. (macro_graphs.MacroServe runs the same two
        halves, serve_decisions and serve_values, as CUDA graphs.)"""
        ints, floats = pad_rows(ints, floats)
        B = len(ints)
        out = torch.zeros(B, OW, device=dev)
        dec = decision_rows(ints)
        di, vi = np.flatnonzero(dec), np.flatnonzero(~dec)
        if len(di):
            out[torch.from_numpy(di).to(dev)] = self.serve_decisions(rows_to_batch(ints[di], floats[di], dev))
        if len(vi):
            out[torch.from_numpy(vi).to(dev)] = self.serve_values(rows_to_batch(ints[vi], floats[vi], dev))
        return out

    def serve_decisions(self, b):
        """Output rows [B, OW] of decision rows (batch b): the policy member alone (or + P2 value-member context)."""
        p2_ctx = self.p2_context(b) if self.p2 else None
        a, c, v, opt, slot = self.nets[0].forward_rows(b, policy=True, p2_ctx=p2_ctx)
        o = torch.zeros(len(a), OW, device=a.device)
        o[:, O_A] = a; o[:, O_C:O_C + N_RAMPS] = c; o[:, O_V] = v; o[:, O_VPOL] = v
        o[:, O_OPT:O_OPT + K] = opt; o[:, O_SLOT:O_SLOT + MAX_DECK] = slot
        if self.milestones:
            _, vm = self._ms_fill(o, b, self.nets[:1])
            o[:, O_V] = torch.logit(vm.clamp(1e-6, 1 - 1e-6)); o[:, O_VPOL] = o[:, O_V]
        return o

    def _value_nets(self):
        return self.nets if self.value_members is None else [self.nets[i] for i in self.value_members]

    def serve_values(self, b):
        """Output rows [B, OW] of rows without a decision (T_f table rows, batch b): value head of all members
        (or self.value_members when restricted, e.g. (1,2,3,4) to exclude Member 0 from T_f and lookahead)."""
        vnets = self._value_nets()
        A, C, V = [], [], []
        for n in vnets:
            a, c, v, _, _ = n.forward_rows(b, policy=False)
            A.append(a); C.append(c); V.append(v)
        V = torch.stack(V)
        o = torch.zeros(V.shape[1], OW, device=V.device)
        o[:, O_A] = torch.stack(A).mean(0); o[:, O_C:O_C + N_RAMPS] = torch.stack(C).mean(0)
        o[:, O_V] = V.mean(0); o[:, O_STD] = V.std(0) if len(vnets) > 1 else 0.0; o[:, O_VPOL] = V[0]
        if self.milestones:
            Vm, vm = self._ms_fill(o, b, vnets)
            L = torch.logit(Vm.clamp(1e-6, 1 - 1e-6))
            o[:, O_V] = torch.logit(vm.clamp(1e-6, 1 - 1e-6)); o[:, O_STD] = L.std(0) if len(vnets) > 1 else 0.0; o[:, O_VPOL] = L[0]
        return o

    @torch.no_grad()
    def value_all(self, ints, floats, dev):
        """Ensemble value logits [B] (value members) for any rows."""
        b = rows_to_batch(ints, floats, dev)
        return torch.stack([n.forward_rows(b, policy=False)[2] for n in self._value_nets()]).mean(0)


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
    export_vocab(os.path.join(out, "macro_vocab.json"), ens.vocab, ens.special)   # the S1 (v0.111) vocabulary
    ens.save(os.path.join(out, "macro_s1.pt"))
    print(f"wrote {out}/macro_vocab.json and macro_s1.pt ({len(ens.nets)} members, "
          f"{sum(p.numel() for p in ens.nets[0].parameters()):,} parameters each)")
