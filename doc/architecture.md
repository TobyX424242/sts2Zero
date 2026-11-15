# Architecture

The agent is a hybrid of a run-level value/policy network and a learned search over combat. The two layers exchange
exactly one object: a table `T_f(h, P)` that says what the rest of the run is worth after the current fight ends with
`h` HP and potion belt `P`. Everything else follows from that choice.

```text
 out-of-campaign state
 (deck, relics, potions, gold, act, floor, map, ...)
            |
            v
   +--------------------------+        macro action
   |  Macro network (S1/S3)   |--------------------------> map / reward / shop /
   |  no-attention set encoder|                            rest / event / ancient
   |  V_run = P(win), monotone|
   |  in HP; policy head      |
   +------------+-------------+
                | at fight start: T_f(h, P) = V_run(post-fight state)
                v
   +--------------------------+        action
   |  Combat search (S2/S3)   |--------------------------> card / potion / end turn
   |  Gumbel root, sequential |
   |  halving, chance nodes   |
   |  combat network: policy  |
   |  + outcome head          |
   +--------------------------+
        leaves priced by T_f
```

The design rests on three constraints of the game:

1. **Combat is stochastic with hidden information.** The agent does not know the draw order or future enemy rolls.
   Search must sample the unknown world, never read the true RNG streams.
2. **The macro horizon is long and sparse.** A run has roughly 180 macro decisions and 640 combat decisions; the
   reward is one win/loss bit. The macro network needs a value function, not a rollout.
3. **HP and potions are the currency between the layers.** A fight should spend them exactly when the run value says
   they are worth spending. That price must be outside the combat network's weights, so it can be re-estimated as the
   macro network improves.

## 1. Macro network

### Inputs

A state after a floor is a set of entities plus context, tokenised as:

| Token group | Count | Content |
|---|---|---|
| CLS | 1 | set summary (transformer variant only) |
| Context | 6 | character (Defect = 2), ascension, game-version era, act index, room type, encounter id |
| Scalars | 1 | floor/50, act/3, HP/100, max HP/100, HP fraction, gold/500, deck size/40, upgraded/20, enchanted/10, relics/20, potions/5, is-combat, floor-in-act/20; damage taken / HP healed / turns taken are always zero |
| Cards | <= 64 | card id, upgrade level, enchantment id + amount, age (floors since added, 0-63) |
| Relics | <= 40 | relic id, age |
| Potions | <= 5 | potion id |

HP is deliberately **removed from the encoder input**. It enters only through the value head (§1.2), which makes the
value monotone in HP by construction. The policy query gets HP back in S3 (§1.4).

### Encoder

The selected architecture is **no-attention** ("Deep Sets"): per-token residual MLP blocks (3 blocks, d = 128,
ff = 512), masked mean pooling, and a 512-256-128 trunk. A set transformer (CLS + mean pooling, d = 128, 4 layers,
4 heads) was the alternative. The no-attention model won or tied at every data size in the scaling curve, was better
calibrated, and trains about twice as fast. The transformer is kept behind `--arch transformer` for ablation.

### Heads

- **Value.** `logit V = a(z) + sum_k softplus(b_k(z)) * phi_k(hp, max_hp)`, with 26 fixed increasing ramp features
  (16 absolute-HP ramps of width 10, 10 HP-fraction ramps of width 0.1). Raising HP with everything else fixed cannot
  lower V. This matters because the search actively looks for states where a higher HP is mispriced.
- **Floors left.** An auxiliary HL-Gauss head; not used by the agent at inference, but it regularises the state
  representation.
- **Policy.** For an offered option, a logit from `MLP([q, e, q*e])`, where `q` is the state vector plus a decision-type
  embedding and `e` is the option embedding. For a deck-slot choice, the logits come from the deck token states. For
  rest sites the offered set was not logged, so S1 scores every rest option in a global vocabulary; S3 restricts the
  same function to the options actually offered.

### Ensemble

Five seeds. Each member is a full network. `T_f` uses the **mean logit** of the members (the raw ensemble was the best
calibrated; temperature scaling made it slightly worse and was dropped). The training code can also average the
members' probabilities (`--prob-avg`), which reduced overconfidence in S3; tables still use the mean logit unless the
milestone heads are enabled.

### S3 additions

`S3MacroNet` extends the S1 `MacroNet` for full-game self-play. All new parameters start at zero, so an S1 checkpoint
loads into it with unchanged behaviour on the decision types S1 covered:

- two new policy heads and decision types: **MAP** (which map point to take) and **EVENT** (which event option);
- map option = room type + 12 path features (row, and fewest/most rooms of each type on any path to the boss);
- event option = embedding of a hash of the `"event:option"` key;
- the policy query receives the 26 HP ramp features through a zero-initialised layer;
- rest-site options are the offered set, not the global vocabulary;
- shop purchases are S1's per-item logits as a softmax against a fixed LEAVE logit (0), one purchase at a time;
- optional **milestone heads**: P(clear act 1), P(clear act 2 | act 1), P(beat act-3 boss 1 | act 2), P(win | boss 1)
  for A10. The D14 outcome is blended outside the weights. Built and tested; off by default because it did not
  improve play with frozen networks.

The macro row layout is shared between C# and Python: ints [460], floats [208], out [110] for the base heads and
[221] when the milestone heads are present. `src/s3/S3/MacroLayout.cs` and `src/s3/py/macro_s3.py` are the two
definitions; they are checked by the encoder round trip.

## 2. Combat network

### Observation layout

The combat observation is `int32[128, 11]`; each token is `[kind, id, id2, f0..f7]`. Kinds: pad, global, hand, draw,
discard, exhaust, enemy, enemy power, player power, orb, potion, relic, context. Piles are multisets: draw order is not
observable. Actions are `int32[64, 8]`: `[kind+1, source token, target token, prompt code, chosen card 1 id, upgrade,
chosen card 2 id, upgrade]`. Prompt answers are part of the action, so two answers to the same prompt are distinct.

Layouts:

- **v3**: the original 11-int layout. Feature slots that became v4 are zero.
- **v4** (2026-11-15): card damage / block in the hand, draw, discard and exhaust tokens; hidden power counters (Wither
  remaining cards, Slither, Panache, ...) in power tokens. A network without `"ext": 1` multiplies those slots by the
  v3 mask, so its input is bit-for-bit the old one. An `"ext": 1` network adds `ext_feat`, whose last layer starts at
  zero; a v3 checkpoint loaded into it gives identical outputs.
- **v5** (`obs=5`, opt-in): counters of 14 relics in the masked Relic f0/f1 slots, extending `ext_feat` from 10 to 12
  input columns. The playing v3 network stays blind to them.
- The search fingerprint includes the new values when `fp_ext=1` (default), so the search does not merge states that
  differ in a visible card damage or counter.

Context token (kind 12), all in thousandths of the fight's `T_max`: room (monster/elite/boss), act, floor, `T_f` at
the current HP and belt, `T_f(HP) - T_f(HP-10)` (price of 10 HP), `T_f(HP, held) - T_f(HP, none)` (worth of the held
potions), `1000 * T_max`, final-boss flag. Potion tokens carry their marginal `T_f` in feature 1 (-1 for a potion
gained during the fight).

### Network

A token-transformer: kind + primary-id + secondary-id embeddings + an MLP over signed-log features; 2 encoder layers,
d = 128, 4 heads, ff = 512; masked mean pooling; about 677k parameters (702k with the v4/v5 `ext` head). The policy
scores each action row with `MLP([h_source, h_target or a learned no-target vector, action-kind + prompt-selection
embedding, pooled state])`; padded rows get `-inf`. The value head is an MLP over the pooled state.

### Value heads

Two designs were built:

- **Outcome head (adopted):** 41 logits for the end-HP distribution (bin 0 = death, bin b = HP in
  `((b-1)/40, b/40]` of max HP) and 6 logits for `P(starting potion i is still held at the end | survive)`. The search
  prices this outcome with the fight's `T_f` table at every leaf. The network therefore learns combat only (HP lost,
  potions spent), and the macro prices stay outside its weights. Precedent: KataGo's score-distribution head priced by
  a utility outside the network; successor features priced by separate weights.
- **Scalar head (comparison):** P(survive) logit + relative `T_f`. Kept behind `--head scalar` for the A20 ablation.

Training targets for the outcome head: end-HP bin from the fight end (HL-Gauss around the exact HP for survivors,
sigma = 0.75 bin; death = bin 0), and potion-kept labels for survivors. The outcome head and the scalar head tied in
the A20 experiment; the outcome head was kept for the structural property (re-pricing without retraining), which was
verified separately.

## 3. The macro -> combat link: `T_f`

### The post-fight state

For a fight on floor `g`, ending with HP `h` and potion belt `P`, define the state

```text
s_g(h, P): floor/act/room/encounter of floor g;
           deck and relics as at fight start, plus permanent in-fight changes;
           HP = h; potion belt = P;
           gold = gold at fight start + expected gold reward of the room;
           no reward card added (the "skip" state)
T_f(h, P) = V_run(s_g(h, P))       for h >= 1
T_f(0, .) = 0                      death (revive effects already resolved by the simulator)
T_f(h, .) = 1                      if this fight wins the run (final boss)
```

Why post-fight rather than pre-fight: the pre-fight state prices HP as if the fight were still to come. On 12,076 boss
fights the pre-fight state gave +1.9 / +3.6 / +5.1 points per 10 HP in acts 1 / 2 / 3, against +1.0 / +2.1 / +3.7 from
the post-fight state, and a level about 10 points lower. The post-fight state is the causal object the search edits.

### Building the table

At the first decision of a fight the agent sends one macro row per subset of the potions held (at most 2^5 = 32 rows),
all with the same deck, relics and context. The value head is HP-monotone, so the server returns the HP-free logit `a`
and the 26 ramp coefficients `c`; the whole HP axis follows from

```text
logit(h) = a + sum_k c_k * phi_k(h, max_hp)
```

The actor checks this formula against the network's own logit in every run; the largest difference observed is 9e-7.
The resulting table is installed in `TfScorer`.

- **T_max** is the largest table entry (full HP, whole belt). Search values are `T_f / T_max`.
- **Free disposal:** holding an extra potion can never be worse than not holding it, so every entry is raised to the
  best entry of the subsets it contains. This corrected 0.18% of held-out entries (72 of 13,384 starts, worst
  -2.5 points), which the search would otherwise exploit.
- **Pessimism:** `kappa` can subtract `kappa * spread` from the ensemble logit. Off in the final runs.
- **Death value options:** `death_value=progress` prices a death as `d = (1-alpha) * acts cleared / 3`, capped at
  `T_f(1 HP, no potions)`; `boss2_model` replaces the first of two act-3 boss tables by a fitted logistic in HP share
  and potions held. Both are off by default; neither improved wins with frozen networks.

### Pricing a leaf

The outcome head gives `p(b | s)` and `q_i(s)`. The search computes

```text
V(s) = sum_{Q subset of held} prod_i (q_i if i in Q else 1-q_i)
       * sum_{b>=1} p(b | s) * Tbar_f(b, Q) / T_max
       + p(death | s) * death_value / T_max
```

`Tbar_f(b, Q)` is the table's mean over bin `b`'s HP range. With at most 5 potions this is at most 32 x 40 terms, a
few microseconds. End HP and potions kept are treated as independent; the correlation (drinking potions when HP is
low) is the approximation. The scalar head instead learns `P(survive)` and a relative `T_f` directly; it tied with
the outcome head in play.

## 4. Search

The search is Gumbel MuZero-style (Danihelka et al. 2022), implemented in C# next to the simulator.

### Information sets and chance

Every simulation clones the live combat with `ResamplingClone.Resampled(state, seed)`: all run/player/monster RNG
streams are reseeded and the unknown part of the draw pile is reshuffled; cards whose position the player knows
(Headbutt, Thinking Ahead, Innate, ...) keep their position. The exact clone (`--exact 1`) is kept only as the G2
upper-bound arm, because it sees the true future.

Tree nodes are keyed by `ObservableState.Hash` of the state the player sees. An end turn (or any random effect) leads to
a sampled chance outcome with its own node; the same state reached by different card orders is one node
(transpositions). Statistics live on the edges.

### Root and selection

- Root: Gumbel-top-k over the network prior, then sequential halving over the remaining candidates. The completed-Q
  transform is `sigma(q) = (c_visit + max N) * c_scale * q`, with `q` min-max normalised over the root's completed Q
  values (`c_visit = 50`, `c_scale = 0.1`; `qspan` can floor the span). The improved policy target is
  `softmax(logits + sigma(completed Q))` over the root actions, and only fully searched moves give policy targets.
- Below the root: deterministic Gumbel selection `argmax(pi'(a) - N(a) / (1 + sum N))`.
- Leaf: a new node is evaluated by the network; a finished fight is scored exactly by the table; values are relative
  `T_f` in [0, 1].

The search budget is allocated per move. S2/S3 use 64 simulations on a quarter of the moves (these give policy
targets) and 16 on the rest. In S3, hallway moves are halved (`hall_sims=0.5`) and every boss move is searched at 64
(`boss_pfull=1`); the allocation test showed the hallway cut costs nothing and the boss budget gains +0.064 bosses per
run at 103% of the simulations. Confidence-gated skipping exists (`cskip`) but is off.

### Practice fights (S3)

At the first decision of a live boss (or elite) fight of act >= `practice_min_act`, the agent plays K copies of the
fight to the end on resampled clones, with the live fight's table, signals and search settings, and records them like
live fights. The live run is unchanged; the copies are training data. Cost: about 24-29% of all simulations before
the 64/16 mix was applied to practice moves, and less after.

### Tree reuse (optional)

The fight keeps its node table between moves. The root keeps its own node, but root edges start with the visits and
value sums of the matching edges of the kept node for the root's observable state. Fresh simulations are unchanged;
the target uses reused + fresh statistics. Nodes untouched for two searches are dropped; the table is cleared above
`reuse_cap`. Tree reuse was on and off during S3; it costs about 5% throughput.

### End-turn prompts (default since 2026-11-15)

A prompt that opens during End Turn (Knowledge Demon's curse, start-of-turn power prompts) used to be answered at
random. The search now probes End Turn once on a resampled clone and, if it opens a prompt, splits End Turn into one
root action per answer. Answers reach the live prompt through the action key script. `S3 prompttest` checks the
Knowledge Demon case.

## 5. The S3 agent

`RunAgent : IRunDecisionSource` plays a whole run. The run driver asks for one decision at a time; the agent routes it
to the macro network, a rule, or the combat search.

| Decision | Type | How |
|---|---|---|
| Card reward / choose-one-card | CARD_REWARD | S1 head; options = cards + SKIP |
| Ancient relic choice | ANCIENT | S1 head; options = relics |
| Rest site | REST, then SMITH_TARGET | S1 heads; S3 sees the offered set |
| Shop | SHOP, then REMOVE_TARGET | per-item logits vs LEAVE, one purchase at a time |
| Deck selections (event/relic) | SMITH_TARGET / REMOVE_TARGET | S1 heads; purpose from the simulator patch; other prompts random |
| Map point | MAP (new) | option = room type + 12 path features; uniform at the start |
| Event option | EVENT (new) | option = hash of "event:option"; uniform at the start |
| Potion reward, gold, relic | - | taken; a potion is lost if the belt is full |
| Crystal Sphere, Fake Merchant | - | simulator default |

At a fight's first decision the agent builds the table (§3), installs `TfScorer` and `FightSignals`, and calls the S2
search on the live `CombatState`. The chosen action is applied to the real fight through `RunDriver`; clones are only
for simulations. Potions are actions at the root and below it; card prompts are compound actions.

**Training vs evaluation.** Training samples all macro decisions from the policy and uses Gumbel noise in search.
Evaluation takes the argmax for S1 decision types, samples the two new heads (their argmax repeats itself on
repeatable event options: a run was lost by holding on at Slippery Bridge until death), and turns search noise off.
Evaluation always uses 64 simulations on every move, or the arm's budget.

**Perturbed fight starts.** In 10% of self-play runs, with probability 0.35 before a fight, HP is shifted by up to
+/-30% of max HP and a potion is added or removed. The run is flagged and excluded from win-rate summaries. The point
is to give V_run HP and potion variation that is independent of deck strength and play, so its prices become causal
for this agent rather than observational correlations.

**Records.** Combat decisions go to `*.s2d` chunks: observation, action rows, policy target on fully searched moves,
model version, end-HP bin, potion masks. Macro rows go to `*.s3r`: the state after every floor (value examples) and
every decision (policy examples with chosen option and log-probability, plus legal deck slots).

## 6. Training

### Macro network

- **Outcome target:** `alpha * won + (1-alpha) * acts_cleared / 3`, `alpha = win rate of the window / 0.2`, capped at
  1. A pure win target could not start from a near-zero win rate: V_run collapsed to 0 within two updates and neither
  T_f nor the advantages carried signal.
- **Policy:** PPO with clipped ratios against the log-probabilities recorded at play time, GAE (gamma = 1,
  lambda = 0.95), entropy bonus, and a KL penalty toward the frozen S1 policy on the S1 decision types. The KL weight
  falls linearly to zero at `--kl-runs` runs. An update stops when KL(policy at start || policy now) exceeds
  1.5 x `--target-kl`. Advantages are divided by `max(std, 0.02)`.
- **Value:** cross-entropy against the outcome on floor and decision states sampled from the last `--window-runs`
  runs; each of the five members draws its own samples, Monte Carlo (lambda = 1). The policy member trains only its
  value heads here (state vector detached): a value gradient into the shared encoder moved the policy far outside
  PPO's trust region (behaviour-lag KL 0.41 after four updates). Members 1-4 train fully.
- **Window hygiene:** a large window (50k runs), at most 16 rows per run for value training, dropout + weight decay on
  members 1-4, and probability averaging. Without these the value memorised individual runs (fresh-run loss worse
  than a constant predictor after segment 9).

### Combat network

Expert iteration: the trainer reads the chunks, keeps a replay window (6M decisions in the final design), and trains
the policy by cross-entropy to the improved-policy target on fully searched moves, and the value head to the end-HP
bin / potion-kept labels. Surprise weighting (0.5) and an EMA of the weights (0.999) are used; the learning rate is
5e-5 in the final design. Old fights are re-searched with the current network rather than replayed. A second "shadow"
trainer can train a larger network (d = 256, 4 layers, 3.9M parameters) on the same chunks; it is compared on fresh
chunks and paired evaluation seeds before it replaces the playing network.

### Warm start and hand-off

S1 and S2 are warm starts only. S3 starts from their weights; from then on every target comes from the agent's own
runs. The KL-to-S1 weight reaches zero in the first quarter of training. T_f is rebuilt from the current V_run at
every fight start, so the combat network's prices track the macro network without retraining. The macro V_run was
never trained on the real game or on AlphaDefect data in the final runs; human data only initialised the weights.

## 7. Inference and serving

C# actors never call the GPU directly. Each actor has a shared-memory file with a header, an input row buffer and an
output buffer. The actor writes a request and waits; a Python server polls all actors, pads the ready rows into one
batch, runs both networks, and writes the answers back. The header carries a model version, so every trajectory
records which weights produced it; the server reloads a checkpoint when its mtime changes.

- One combat inference process saturates at about 27k states/s; S3 runs 3-4 combat processes plus one macro process.
- Combat batches use fixed buckets and, optionally, a CUDA graph per (batch, token) bucket plus two batches in flight
  (`--graphs 1 --overlap 1`), which gave +18% runs/hour at the same window and roughly doubled combat-trainer
  capacity.
- CUDA MPS (`MPS=1`) lets the inference servers and trainers share one GPU context instead of time-slicing.
- The macro server handles the many small rows of the `T_f` tables and the per-decision policy queries. Actors do not
  wait for macro answers while their fights can continue.

## 8. What is not modelled

- Potions gained during a fight and permanent in-fight changes (max HP, cards added) have no `T_f`; they are ignored
  at fight end. A full belt loses a potion reward (no discard decision).
- Card-reward alternatives granted by relics are never chosen.
- V_run has no map input, so map choices rest on the MAP head alone; the path features are a proxy, not a map encoder.
- The combat input lacks relic counters (v5 adds them only as an opt-in ext head), multi-intent moves' later intents,
  and engine forecasts (expected HP loss, lethal).
- Ties are not broken toward shorter fights; fights are capped at 50 rounds, and the agent fails a run at round 151.
- The simulator does not model the victory event after the final boss; it declares the win when the final boss dies.
- Fidelity is discussed in `doc/simulator.md`; the four bosses that looked suspicious were reviewed and no
  single-player difference was found, but the simulator is not byte-exact.
