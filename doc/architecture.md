# Architecture

This document explains how the agent makes decisions. It describes the system as it ran in segment 29 of the
self-play chain `s3ic` (2026-11-20), the checkpoint of release `v0.2.0-ic113-s029`. Training is in
[training.md](training.md); the simulator is in [simulator.md](simulator.md).

## 1. Overview

A run of Slay the Spire 2 has two kinds of decisions. *Macro decisions* happen outside combat: which map node to
visit, which card to take, what to buy, whether to rest or upgrade, which event option to pick. *Combat decisions*
are the cards and potions played in a fight. A run at A10 has a few hundred of each, and the only reward is win or
loss at the end.

The agent uses one network for each kind, and links them with a price table.

```text
 run state (deck, relics, potions, gold, HP, act, floor, map, recent fights)
        |
        v
 +-------------------------------+    macro action: map node, card, purchase,
 | Macro network (5-member       |--> rest option, event option, relic
 | ensemble): policy + V_run     |
 +-------------------------------+
        | at the start of each fight: T_f(h, P) = V_run(state after the fight, HP h, belt P)
        v
 +-------------------------------+    combat action: card (+ target, + prompt answer),
 | Gumbel search over resampled  |--> potion, end turn
 | clones, guided by the combat  |
 | network (policy + outcome)    |
 +-------------------------------+
   search leaves are priced by T_f
```

Terms:

- **V_run**: the macro network's estimate of the chance to win the run from a state (with the training target used
  here, a blend of win and progress; see §2.4).
- **T_f(h, P)**: the price table of one fight. It gives V_run of the state just after the fight if the fight ends with
  `h` HP and the potion belt `P` (the potions still held). It is rebuilt at the start of every fight from the current
  macro network.
- **T_max**: the largest entry of a fight's table (full HP, every potion kept). The search works with T_f / T_max.

Why this split:

1. **Combat has hidden information and chance.** The draw order and enemy rolls are unknown, so the search samples
   possible worlds and never reads the true random streams.
2. **The macro horizon is long and the reward is sparse.** A value function is needed; rollouts to the end of a run
   are too slow.
3. **HP and potions are the currency between fights.** A fight should spend them when the rest of the run can afford
   it. The price comes from the macro network and lives outside the combat network's weights, so it changes as soon as
   the macro network learns, without retraining the combat network.

## 2. Macro network

The macro network is an ensemble of five members (2,165,194 parameters each, 10.8M in all). Member 0 also carries the
policy heads. Its input is one *macro row*: the state after a floor, plus the decision being asked, if any. The row
layout is shared between C# (`src/s3/S3/MacroLayout.cs`) and Python (`src/s3/py/macro_s3.py`); the current layout is
**v3**.

### 2.1 Inputs

| Group | Content |
|---|---|
| Context (6 tokens) | character (Ironclad = 0), ascension, version era (6 = v0.113), act, room type, encounter |
| Scalars | floor, act, max HP, gold, deck size, upgrades, enchantments, relic and potion counts, and others. HP itself is **not** an encoder input (§2.3). |
| Cards (≤ 64) | id, upgrade level, enchantment and amount, age (floors since the card was added) |
| Relics (≤ 40), potions (≤ 5) | id (relics also age) |
| Act bosses (layout v2) | the act's boss and, in act 3 at A10, the second boss |
| Act scalars (layout v2) | elites fought, rest sites and shops visited in this act, free potion slots, map row, price of the next card removal |
| Recent fights (layout v3) | the last 8 fights: encounter, room kind, HP lost, HP share at the start, potions used, turns, floors ago |
| Map (layout v3) | up to 80 nodes of the current act's map: type, row, column, visited, current, reachable, and which offered map option reaches each node |
| Content properties (layout v3) | fixed feature vectors per card, relic and potion, read from the simulator's model classes (type, rarity, cost, keywords, measured damage, block, draw, powers applied) |
| Deck counts (layout v3) | 22 counts computed from the deck (Strikes, Defends, attacks, skills, powers, curses, cost buckets, damage, block, draw and others) |
| Decision (if any) | a decision type and up to 16 options, or a choice among deck slots |

Ids come from `src/s3/data/macro_vocab_v113.json` (721 cards, 23 enchantments, 318 relics, 69 potions, 163
encounters).

### 2.2 Encoder

Each token goes through a stack of per-token residual MLP blocks (d = 128) and the token sets are mean-pooled into a
512-256-128 trunk. Layout v3 adds two pre-norm self-attention blocks (4 heads, d = 128) over all state tokens, and lets
each decision option attend over the state tokens. Every v3 addition started at zero, so an older network loaded into
the v3 code gives exactly its old outputs. (The first version of this encoder, in the Defect phase, had no attention;
see branch `v0.111-defect`.)

### 2.3 Value head: monotone in HP

The value logit at HP `h` is

```text
logit V(h) = a(z) + sum_k softplus(b_k(z)) * phi_k(h, max_hp)
```

where `z` is the trunk output and `phi_k` are 26 fixed increasing ramps (16 on absolute HP, 10 on the HP fraction).
More HP can never lower V. This matters because the search looks for states where the value is mispriced; a value
that could fall with HP would be exploited.

Because of this form, one evaluation gives the value at every HP: the server returns `a` and the 26 coefficients
`c_k = softplus(b_k)`, and the actor computes the whole HP axis. The actor checks this against the network's own output
in every run (largest difference seen: under 1e-6).

### 2.4 Milestone heads and the training target

The training target of a run is `outcome = alpha * won + (1 - alpha) * acts_cleared / 3`, with `alpha = 0.85` in
segment 29 (0.70 before). The progress term gives the value a signal even when wins are rare.

Since segment 20 the trainer also trains four **milestone heads** (`--milestones`), each monotone in HP like the
value head:

| Head | Estimates |
|---|---|
| q0 | P(clear act 1) |
| q1 | P(clear act 2, given act 1 cleared) |
| q2 | P(beat the first act-3 boss, given act 2 cleared) |
| q3 | P(win, given the first act-3 boss beaten); A10 only, where act 3 has two bosses |

V_run is then computed outside the weights from the chance of reaching each stage (R1 to R4, products of the q's,
with 1 for stages already reached):

```text
V = alpha * R4 + (1 - alpha) * (R1 + R2 + w * R3 + (1 - w) * R4) / 3
```

with the boss-1 share w = 0 in this chain, so the progress term is (R1 + R2 + R4) / 3.

Each head learns at its own base rate, and V never falls below the progress already secured. The released macro file
has trained milestone heads, so tables and lookahead use this blend.

### 2.5 Policy heads

Member 0 scores the offered options of each decision type.

| Decision type | Options |
|---|---|
| CARD_REWARD | the offered cards and SKIP |
| ANCIENT | the offered relics |
| REST | the offered rest options (heal, smith, and others; v0.113's new `restore` and `evolve` enter as hashed keys) |
| SMITH_TARGET, REMOVE_TARGET | deck slots |
| SHOP | each item's buy logit against a fixed LEAVE logit, one purchase at a time |
| MAP | map nodes: room type, 12 path features, 8 route features, and attention over the map tokens |
| EVENT | a hashed embedding of the `"event:option"` key |

The policy query also receives the 26 HP ramps, so the policy can depend on HP while the value stays monotone.

### 2.6 Ensemble

T_f uses the mean of the five members' logits. Member 0 trains its value heads only (its encoder serves the policy);
members 1–4 train fully. Probability averaging is used for advantages and logs (`--prob-avg`).

## 3. Combat network

| Item | Value |
|---|---|
| Input | observation layout v5: up to 128 tokens of 11 integers each (kind, id, second id, 8 features) |
| Token kinds | global, hand, draw pile, discard, exhaust, enemy, enemy power, player power, orb, potion, relic, context |
| Encoder | token embeddings + an MLP over signed-log features; 2 transformer layers, d = 128, 4 heads, FF 512; masked mean pooling |
| Extra features (`ext = 1`) | card damage and block in pile tokens, hidden power counters, 14 relic counters, read through a separate input layer |
| Policy | scores each legal action row (source token, target token, action kind, prompt answer, pooled state) |
| Value | outcome head (below), value dropout 0.3 |
| Size | 724,810 parameters |

Piles are multisets: the draw order is not observable. An action row is `[kind, source token, target token, prompt
code, chosen card 1, upgrade, chosen card 2, upgrade]`, so different answers to a card prompt are different actions.

The **context token** carries signals from the price table, all in thousandths of T_max: room kind, act, floor, T_f
at the current HP and belt, the price of 10 HP, the worth of the potions held, T_max itself, and a final-boss flag.
Each potion token carries its marginal T_f.

### 3.1 Outcome head

The value head does not predict a value. It predicts the fight's outcome:

- 41 logits for the end-HP distribution (bin 0 = death, bin b = HP in ((b−1)/40, b/40] of max HP);
- 6 logits for P(starting potion i is still held at the end | survive).

The search prices this outcome with the fight's T_f table (§4.3). The network therefore learns combat only (HP lost,
potions spent); the prices stay outside it.

## 4. The link: T_f

### 4.1 The post-fight state

For a fight on floor `g` that ends with HP `h` and belt `P`, the post-fight state is: floor `g`'s state, the deck and
relics at the fight's start, HP = heal(h), belt = P, gold + the room's expected gold reward, no reward card added. Then

```text
T_f(h, P) = V_run(post-fight state)    for h >= 1
T_f(0, .) = 0                          death
T_f(h, .) = 1                          if this fight wins the run (the final boss)
```

**Post-combat heals (`tf_heal=1`, new in this phase).** `heal(h)` applies the relics that heal after a won fight, in
the simulator's order: Chosen Cheese (max HP +1, heal 1), Meat on the Bone (+12 if HP ≤ 50% of max HP), Burning Blood
(+6), Black Blood (+12), capped at max HP. The search's terminal states and the combat network's end-HP target both use
the HP before these heals, so the heal is counted exactly once, through the table. Verified against the simulator on
1,065 won fights (`S3 healtest`) and on at least 2,607 live fights; see [simulator.md](simulator.md).

### 4.2 Building the table

At a fight's first decision the agent sends one macro row per subset of the potions held (at most 2^5 = 32 rows). The
HP axis of each row follows from the monotone head (§2.3). Two corrections:

- **Free disposal**: holding an extra potion can never be worse than not holding it, so each entry is raised to the
  best entry of the subsets it contains.
- **T_max** = the largest entry. Search values are T_f / T_max, between 0 and 1.

### 4.3 Pricing a search leaf

At a leaf the combat network gives `p(b)` (end-HP bins) and `q_i` (potion i kept). The leaf value is

```text
V(leaf) = sum over subsets Q of the held potions of  prod_i (q_i if i in Q else 1 - q_i)
          * sum_{b >= 1} p(b) * Tbar_f(b, Q) / T_max
```

`Tbar_f(b, Q)` is the table's mean over bin b's HP range. Death contributes 0. End HP and potions kept are treated as
independent; that is an approximation.

## 5. Search

The search is Gumbel MuZero-style (Danihelka et al. 2022), written in C# next to the simulator
(`src/s2/S2/NetSearch.cs`).

**Sampled worlds.** Every simulation starts from a *resampling clone* of the live fight (`ResamplingClone.Resampled`):
all random streams are reseeded, and the unknown part of the draw pile is reshuffled. Cards whose position the player
knows (placed on top or bottom by a card effect, Innate cards) keep their place. The search never sees the true future.

**Tree.** Nodes are keyed by a hash of what the player can observe. An end turn, or any random effect, leads to a
sampled outcome with its own node; the same state reached by different card orders is one node.

**Root.** Gumbel-top-k over the network's policy picks the candidates (4 in self-play, 16 in evaluation); sequential
halving spends the budget on them. The training target for the policy is `softmax(logits + sigma(completed Q))`.
Below the root, selection is deterministic (`argmax(pi'(a) - N(a) / (1 + sum N))`).

**Budget.**

| Setting | Self-play | Evaluation |
|---|---|---|
| Simulations per move | 64 on 25% of moves (these give policy targets), 16 on the rest | 64 on every move |
| Hallway fights (`hall_sims=0.5`) | budget halved | — |
| Boss fights (`boss_pfull=1`) | every move at 64 | 64 |
| Root candidates | 4 | 16 |
| Gumbel noise | on | off |

**End-turn prompts.** If ending the turn opens a prompt (for example Knowledge Demon's curse), the root splits End Turn
into one action per answer.

**Stalemate guard (`win_bound=0.001`).** Without it, the agent sometimes refused to kill the last weak enemy: the
network valued the open fight a few thousandths above the exact value of winning now, nothing charged for time, and
the fight ran into the round cap. With `win_bound`, the search caps the value of a non-terminal node at the exact
T_f of winning now minus 0.001. Self-play stalemates fell from 269 to 1 per segment (segment 2 to 3).

**Lethal backstop (`lethal_backstop=1`).** After round 50 the search is switched off and the policy's top move plays.
With the backstop, a root move that wins the fight at once on a resampled clone is played instead.

**Practice fights (self-play only).** At the start of a live boss fight from act 3 on (`practice_min_act=3`), the
agent plays 3 extra copies of the fight to the end on resampled clones (`practice_boss=3`), with half of their moves at
full search (`practice_pfull=0.5`). The copies are training data; the live run is not affected. (Before segment 29:
2 copies from act 2, 25% full.)

## 6. The run agent

`RunAgent` (`src/s3/S3/RunAgent.cs`) plays a whole run. The simulator's run driver asks it for one decision at a time.

| Decision | How it is made |
|---|---|
| Card reward, "choose a card" offers | CARD_REWARD head, plus a one-step lookahead (`look_beta=20`, below) |
| Ancient relic choice | ANCIENT head |
| Rest site | REST head; then SMITH_TARGET for the upgrade |
| Shop | SHOP head, one purchase at a time; REMOVE_TARGET for the removal service |
| Deck prompts with a purpose (upgrade, remove, transform) | SMITH_TARGET or REMOVE_TARGET head; the purpose comes from the simulator patch |
| Deck prompts without a purpose (`deck_route=1`) | routed to the closest head: cards that get better (enchant, duplicate) to SMITH_TARGET, cards lost or replaced to REMOVE_TARGET, new cards to CARD_REWARD one pick at a time. Bundles (Scroll Boxes) stay random. |
| Map node | MAP head (sampled in self-play, argmax in evaluation with `greedy_map=1`) |
| Event option | EVENT head (sampled, also in evaluation: its argmax can repeat a repeatable option until death) |
| Fake Merchant event (`fake_merchant=1`) | its relics are offered to the SHOP head; with a Foul Potion the EVENT head first chooses between throwing it and shopping |
| Full potion belt and a new potion | `potion_mix` (§7) |
| Pumpkin Candle's Kindle at a rest site (`kindle_cap=1`, live play only) | not offered while the candle's charges already cover an upper bound of the fights left |
| Fight | build T_f (§4), then the search (§5) plays every move |
| Crystal Sphere | simulator default (the hidden contents are not policy inputs) |
| Gold, relic rewards | taken |

**One-step lookahead.** For decisions whose result is known without chance, the option's logit gets a bonus
`beta * (z(post) - z(now))`, where `z` is the ensemble's value logit and `post` the state after the option. In
self-play only card rewards use it (`look_beta=20`); evaluation and live play also use it for upgrades
(`look_smith=20`) and shop purchases (`look_shop=20`).

**Self-play versus evaluation.** Self-play samples every macro decision and uses Gumbel noise in the search.
Evaluation takes the argmax for the decision types learned from human data and for the map, samples events, and turns
search noise off.

**Perturbed fight starts (self-play).** In 10% of self-play runs, each fight has a 35% chance to start with HP shifted
by up to ±30% of max HP and a potion added or removed. These runs are flagged and left out of win-rate statistics. They
give V_run HP and potion variation that does not come from deck strength, so its prices become causal for this agent.

**Records.** Combat decisions go to `*.s2d` chunks (observation, action rows, policy target on fully searched moves,
end-HP bin, potion masks). Macro rows go to `*.s3r` chunks (the state after every floor for value training, every
decision with its option and log-probability for policy training).

## 7. potion_mix: which potion to keep

When the belt is full and a potion reward is about to be taken, the agent decides which potion, if any, to drop.
`potion_mix` mixes a population price with a state-specific one:

```text
dV(s, k)     = V_run(s with the belt replaced by potion k alone) - V_run(s with an empty belt)
Vbar_act(k)  = mean of dV(s, k) over sampled self-play states of that act
V_mix(s, k)  = Vbar_act(k) + 0.2 * (dV(s, k) - Vbar_act(k))
```

The held potion with the lowest V_mix is dropped when the offered one scores higher. The rule also covers extra potion
rewards of events and relics (`potion_mix_extra=1`).

`Vbar` is recomputed between segments by `src/ic113/potion_mix.py vbar`: up to 500 states per act from the newest
20,000 runs in the finished segment's macro carry, priced by the macro file the next segment starts from. The result is
a JSON file per segment (`vbar_sNNN.json`); the release's `ic113_s029_vbar.json` is the file the segment-29 evaluation
used. A potion kind without a mean (too few states) is scored by its state term alone. The C# and Python versions of
dV gave identical values on a logged decision in the chain's smoke test (`potion_mix.py check`).

A second phase that blends in a measured final-boss price per potion is coded but off; a paired test on 320 seeds
showed no gain (see [experiments.md](experiments.md)).

## 8. Actor keys in use

The actor (`S3.dll play`) takes `key=value` options; the full list is at the top of `src/s3/S3/Program.cs`. These are
the ones that define the segment-29 agent. Most keys added in this phase default to off (`tf_heal` and `era` default
to the v0.113 settings). For `deck_route` and `lethal_backstop`, runs with the keys off were checked byte-identical to
the build before them.

| Key | Self-play | Evaluation / live | Meaning |
|---|---|---|---|
| `character` | ironclad | ironclad | character of new runs and the context token |
| `era` | 6 | 6 | version-era token of the macro network (6 = v0.113) |
| `tf_heal` | 1 | 1 | post-combat relic heals inside T_f |
| `hall_sims`, `boss_pfull` | 0.5, 1 | — | search budget factors |
| `practice_boss`, `practice_min_act`, `practice_pfull` | 3, 3, 0.5 | — | practice copies of act-3 boss fights |
| `candidates` | 4 | 16 (default) | root candidates |
| `greedy_map` | — | 1 | argmax on the map head |
| `look_beta` | 20 | 20 | card-reward lookahead |
| `look_smith`, `look_shop` | — | 20 | upgrade and shop lookahead |
| `potion_mix`, `potion_mix_shrink`, `potion_mix_extra` | file, 0.2, 1 | file, 0.2, 1 | §7 |
| `fake_merchant` | 1 | 1 | Fake Merchant purchases by the shop head |
| `win_bound` | 0.001 | 0.001 | stalemate guard |
| `deck_route` | 1 | 1 | deck prompts without a purpose go to a head |
| `lethal_backstop` | 1 | 1 | immediate kill past the round cap |
| `kindle_cap` | — | live only | Kindle cap at rest sites |
| `asc_weights`, `potion_mix` file | per segment | — | written by the chain hook ([training.md](training.md)) |

## 9. Shadow combat trainer

The playing combat network stopped improving early in the chain: from about segment 3 its training loss, policy
accuracy and gradient norm stayed flat. One suspected cause is that it inherited shrunken embedding and LayerNorm
scales from the Defect phase, where weight decay hit every parameter. A larger network cannot be swapped in directly,
so a **shadow trainer** runs beside the playing one:

- It reads the same training chunks and writes `shadow.pt`, which actors do not use.
- Since segment 29 it trains a network from random weights: d = 256, 3 layers, 8 heads, `ext = 1`, 3.26M parameters,
  learning rate 3e-4 with cosine decay to 5e-5 over 340,000 steps, weight decay 0.01 on weight matrices only, a 4M
  replay window preloaded from the playing trainer's carry.
- Its optimizer and replay state go into `carry_shadow.bin`, so it continues across segments.
- It replaces the playing network only after a paired evaluation on the same seeds shows a gain
  (`src/ic113/shadow_eval.sh`, then `src/s3/switch_combat.sh`). As of segment 29 no switch has been made; the shadow
  network is not part of the release.

## 10. Inference and serving

C# actors never touch the GPU. Each actor shares memory files with Python inference servers
(`src/s3/py/s3_server.py`): the actor writes a request row and waits; the server collects ready rows from all actors,
runs one batch, and writes the answers back. Each answer carries a model version, so every training example records
which weights produced it. Servers reload weights when the trainers publish new ones.

- Combat batches use fixed size buckets with one CUDA graph per bucket and two batches in flight (`--graphs 1
  --overlap 1`). The chain ran 2 combat servers for 60 actors.
- The macro server also runs as CUDA graphs (`--mgraphs 1`); it answers T_f table rows, lookahead rows and policy
  queries. Actors keep playing other fights while they wait for macro answers.
- With CUDA MPS (`MPS=1`) trainers and servers share the GPU without time-slicing.

## 11. Known limits

- Potions gained during a fight and permanent in-fight changes (max HP, cards added) are not priced by T_f.
- Heals at reward time (for example Book of Five Rings) are not in T_f.
- dV compares "this potion alone" with an empty belt, not the potion's marginal value next to the others.
- `Vbar` is computed once per segment, while the macro network is reloaded about every 2,560 runs within it.
- Decks over 64 cards are truncated in the macro row.
- After round 50 the search is off; at round 151 the run is stopped as a stalemate and counted as lost.
- Crystal Sphere and a few prompts without a known purpose are answered by default or at random.
- The simulator ends the run when the final boss dies; it does not model the ending scene.
