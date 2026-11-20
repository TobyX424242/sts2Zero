# S3: full-game self-play (PROPOSAL.md §6.5)

Whole Defect runs in iRyougi. Every out-of-combat decision is made by the macro network, every fight is played by the
network-guided search of S2 on the live combat, scored by a T_f table built at the fight start from the current macro
value network. Both networks are trained on the agent's own runs. Builds on `src/s0` and `src/s2`.
Results and decisions are in `PROGRESS.md`; this file describes the code.

| Path | Content |
|---|---|
| `S3/MacroLayout.cs` | Row layout shared with Python (`ML`), macro vocabulary, `MacroEncoder`: the S1 state after a floor from a live `RunState` (and the post-fight state of §4.2) |
| `S3/MacroBridge.cs` | Shared-memory client for the macro network; `MacroBridgeEvaluator` batches rows of all runs of a process |
| `S3/RunAgent.cs` | `RunAgent : IRunDecisionSource`: macro decisions, live T_f table per fight, battle search, perturbed fight starts, records |
| `S3/Records.cs` | Run record, macro training chunks (`*.s3r`) |
| `S3/Program.cs` | `S3 play key=value ...` (self-play or evaluation; keys at the top of the file), `S3 encode` (round-trip test) |
| `S3/EncodeTest.cs` | Writes simulated runs as Spire Codex style run records plus the encoder's rows, for `py/roundtrip_macro.py` |
| `py/macro_s3.py` | `S3MacroNet` (S1 `MacroNet` + MAP and EVENT decisions), `Ensemble` file format, row layout, vocabulary export |
| `py/s3_server.py` | Inference server for both networks (`--serve combat`, `--serve macro`: run one process of each) |
| `py/combat_pipe.py` | Combat batches of `s3_server.py` with CUDA graphs per (batch, token) bucket and two batches in flight (`--graphs 1`, `--overlap 1`; off by default) |
| `py/train_macro_s3.py` | Macro trainer: PPO for the policy member, V_run for all members |
| `py/roundtrip_macro.py` | Encoder round trip against `src/value_v1/build_floor_dataset.py` |
| `py/layout2_check.py`, `py/macro_s3_v1.py` | Checks of row layout v2 (below) against the frozen pre-change network (`macro_s3_v1.py`: do not edit) |
| `mps.sh` | CUDA MPS control daemon for training segments (`MPS=1` in `run_s3.sh` / `extra.env`): `start`, `status`, `stop` |
| `shadow_watch.sh`, `shadow_switch_check.sh`, `switch_combat.sh` | Shadow combat network (`SHADOW_ARGS`, `SHADOW_FROM_PLAYING=1` in extra.env): fresh-chunk loss comparison per segment, paired evaluation on the supervisor's seeds and the switch rule, the switch itself |
| `eval_asc_sweep.sh`, `asc_sweep_report.py` | One agent at A0-A10 on the same seeds; win rate and where runs die |
| `run_s3.sh`, `eval_s3.sh`, `summarize_runs.py` | Training run, evaluation on fixed seeds, summary with intervals and paired tests |
| `s3_supervise.sh`, `s3_resume.sh` | Server: unattended run as a chain of guarded segments with evaluations and Nextcloud status; restart of a dead supervisor |
| `carry_backup.sh`, `restore_from_nextcloud.sh` | Interruptible servers (2026-11-19): trainer carries of a segment to Nextcloud (CARRY_BACKUP_EVERY), and a chain rebuilt from Nextcloud on a fresh tree; adoption of interrupted segments is in the `s3_supervise.sh` header |
| `backup_loop.sh`, `server_status.sh`, `s3_pipeline*.sh`, `curve_runs.py` | Server: Nextcloud backup of training chunks, watchdog status line, test pipelines, self-play curve from run logs |
| `panel/` | Laptop: watch panel for a supervised run. `src/s3/panel/run_panel.sh [port]` serves http://127.0.0.1:8733/ from the local copy in `src/server_backup/s3/`; its Sync buttons pull the logs from Nextcloud with rclone (read-only). Usage in the header of `panel/panel.py` |
| `iryougi_s3.patch` | Simulator change (on top of the S0 patch): `CardSelectCmd.CurrentDeckPurpose` |
| `data/macro_vocab.json`, `data/macro_s1.pt` | Macro vocabulary for C#, the five S1 members as one ensemble file (`python -I py/macro_s3.py`) |
| `runs/<name>/` | `weights.pt` (combat), `macro.pt` (ensemble), `runs/` and `fights/` (logs), `cdata/` `mdata/` (chunks), `ckpt/` `mckpt/`, `logs/` |

The combat trainer is `src/s2/py/train_combat.py --init <S2 weights>`. Changes outside this directory:
`TfScorer.Set/Remove` (S0: tables built at run time), `BridgeEvaluator.Submit/Complete` (S2), `train_combat.py --init`.

## How a run is played (`RunAgent`)

**Macro decisions.** The row sent to the network is the current state in the S1 conventions (floor counted from the
Ancient of act 1 = 1, room type of the map point, model id of the floor's first room, card age = floor − floor added)
plus the decision: a type and up to 16 options, or a choice among deck slots.

| Decision | Type | How |
|---|---|---|
| Card reward, other "choose one card" offers | CARD_REWARD | S1 head; options = cards + SKIP |
| Ancient (relic choice at act start) | ANCIENT | S1 head; options = relics |
| Rest site | REST, then SMITH_TARGET | S1 heads; options = the options offered (S1 could not see which were offered) |
| Shop | SHOP, then REMOVE_TARGET | S1's per-item buy logits as a softmax against LEAVE (logit 0), one purchase at a time: with one affordable item it is bought with exactly S1's probability |
| Deck selections of events and relics | SMITH_TARGET (upgrade), REMOVE_TARGET (remove, transform) | S1 heads; the purpose comes from the simulator patch; other selections are answered at random |
| Map point | MAP (new) | New head, uniform at the start. Option = room type + 12 path features (row; fewest and most rooms of each type on any path to the boss) |
| Event option | EVENT (new) | New head, uniform at the start. Option = embedding of the hashed "event:option" key |
| Potion reward, gold, relic | – | Taken (a potion is lost if the belt is full) |
| Crystal Sphere, Fake Merchant | – | Simulator default |

Training samples from the policy. Evaluation (`greedy=1`) takes the most probable option for the S1 decision types and
samples the two new heads (their argmax repeats itself on repeatable event options: runs were lost by holding on at
Slippery Bridge until death).

**HP and the policy.** S1 removed HP from the encoder so that the value is monotone in HP through its own head; the S1
policy head therefore could not see HP. `S3MacroNet` adds the 26 HP ramp features to the policy query through a layer
that starts at zero (S1 behaviour at the start, HP dependence learned by PPO).

**Fights.** At the first decision of a fight the agent builds the table T_f(h, P) for every HP h and every subset P of
the potions held: one macro row per subset (the post-fight state: this floor's state with the expected gold reward and
that potion belt). The value head is monotone in HP by construction, logit(h) = a + Σ c_k φ_k(h, max HP), so the server
returns (a, c) and the whole HP axis follows from one row (checked in every run: the actor reports the largest difference
between this formula and the network's own logit, 9e-7). T_f = 1 at the fight that wins the run. The table goes into
`TfScorer`, and the S2 search (`NetSearch`, `FightSignals`) runs unchanged on the live combat state: clones for
simulations, the chosen action applied to the real fight through `RunDriver`. `scorer=hp` replaces the table by
end HP / max HP (the comparison arm of gate G3). `kappa` subtracts κ × the members' spread from the logit (off by default).

**Perturbed fight starts** (`perturb`, 10% of self-play runs): before a fight, with probability 0.35, HP is shifted by
up to ±30% of max HP and a potion is added or removed. Flagged in the run record; excluded from win-rate summaries.

**Practice fights** (`practice_boss=K`, `practice_elite=K`, `practice_min_act=A`; self-play only, off by default): at the
first decision of a live boss (elite) fight of act ≥ A, K copies of the fight are played to the end before the live
fight goes on, one after another, each from a resampling clone of the live combat (hidden information drawn afresh,
own chance seed and random stream), with the live fight's search settings, T_f table, signals and T_max. They are
recorded to the combat chunks like a live fight (policy targets on full-search moves, outcome targets from the copy's
own end). The live run is unchanged (checked: identical run and fight logs and macro chunks with K = 2 vs 0 on fixed
seeds); copies are not counted in the run's fight statistics (`practice_*` keys in the run log; fight-log lines with
`"practice": true`, id `<fight id>:p<k>`); a copy still running at round 3 × `max_rounds` is abandoned (not recorded).

**Tree reuse** (`reuse=1`, `reuse_cap`): the fight's search keeps its node table between moves. The root keeps its own
node, but root edges start with the visits and value sums of the matching edges of the kept node for the root's
observable state; the fresh budget and sequential halving are unchanged, the target uses reused + fresh statistics,
recorded `sims` count fresh simulations only. Nodes untouched for two searches are dropped; the table is cleared above
`reuse_cap` nodes and at fight end. Fight-log lines carry `reused_visits`.

**Death value and the double boss** (off by default; defaults are byte-identical to the old build on fixed seeds):
- `death_value=progress`: a death in a T_f-scored fight is worth d = (1 − α) · acts cleared before the fight / 3, the
  run outcome of dying there under the macro value target (D14), instead of 0; d is capped at T_f(1 HP, no potions) so
  that dying is never worth more than surviving (at A10 the act-3 tables of the macro network lie below d at every HP in
  most fights: its level error there is larger than the HP effect). `progress_raw`: d uncapped, table entries below d
  raised to d. The value enters terminal states (`TfScorer.Score/Value`), the outcome head's death probability at leaves
  (`FightSignals.OutcomeValue`, bin 0) and the tables' HP-0 entries; T_max, the network's context inputs, the recorded
  `score`/`rel` (0 for a lost fight) and the combat training targets are unchanged. α: `death_alpha=X` (0.05), or
  `death_alpha_log=FILE` (the last `alpha` of a `macro_train_log.jsonl`; falls back to `death_alpha`).
- `boss2_model=src/s3/data/boss2_w2.json`: the first of two act-3 bosses (A10) is scored by
  T_f(h, P) = d + (1 − d) · W2(h / max HP, |P|) instead of the macro network's table, W2 = P(win the second boss) from
  a logistic in HP share and potions held at its start, fitted by `fit_boss2.py` on self-play logs (d = 0 without
  `death_value`).
- `tf_dump=FILE` (`{tag}` = actor tag), `tf_dump_k=K`: the tables of the first K act-3 boss and hallway fights per act;
  `tf_dump_report.py` prices a held potion under base, F1 and F1+F2 from it. `S3 tftest` checks the death-value plumbing.

**End-turn prompts** (`endturn_prompts=1`, default since 2026-11-15): a prompt that opens during End Turn (Knowledge
Demon's curse choice, start-of-turn power prompts) is chosen by the search: the root probes End Turn once on a
resampled clone and splits it into one action per answer of the first prompt. `0` = answered at random (old build).

**Combat input layout v4** (2026-11-15): card damage / block and hidden power counters in feature slots that were 0 in v3
(`src/s0/py/combat_obs.py` FEATURES / V4_SLOTS). Networks without `"ext": 1` mask them out (identical outputs);
`"ext": 1` networks read them through `ext_feat` (zero-initialised, so they start equal to the v3 network they were
copied from). `fp_ext=1` (default) adds the same values to the search's state fingerprint.

**Records.** Combat decisions go to the S2 chunk format (`*.s2d`, unchanged). Macro rows go to `*.s3r`: the state after
every floor (value examples, as in S1: not the last floor, not after death) and every decision with the chosen option,
its log-probability, and the legal deck slots.

## Training (`run_s3.sh`)

Processes: combat trainer, macro trainer, `CSERVERS` combat server processes (one Python loop saturates at about 27k
states/s; with 26 actors use 3), one macro server, N actors (each plays `CONC` runs at once on one thread). An actor does
not wait for macro answers while fights can continue. `ASC` is the target ascension, `ASC_MIX` the share of runs played
at it (the rest uniformly at the levels below; decision D15: 0.5).

**Macro trainer.** Every `--runs-per-iter` runs:
- *Outcome.* outcome = α · won + (1 − α) · acts cleared / 3. α = win rate of the window / 0.2, at most 1, so the
  target becomes the plan's P(win) once the agent wins a fifth of its runs. Reason: a new agent wins almost no runs
  (0 of 1,024 in the first test); with the pure win target V_run went to 0 everywhere within two updates and neither
  T_f nor the advantages carried any signal.
- *Policy (member 0).* PPO, clipped ratio against the recorded log-probabilities, GAE (γ = 1, λ = 0.95) with the outcome
  as the terminal reward, entropy bonus, KL penalty toward the frozen S1 policy on the S1 decision types with a weight
  falling linearly to zero at `--kl-runs` runs. Steps stop when KL(policy at the iteration start ‖ policy now) exceeds
  1.5 × `--target-kl`. Advantages are divided by max(their standard deviation, 0.02).
- *Value (all five members).* Cross-entropy against the outcome on floor states and decision states sampled from the
  last `--window-runs` runs; each member draws its own samples. The policy member trains only its value heads here
  (state vector detached): a value gradient into the encoder it shares with the policy moved the policy far outside
  PPO's trust region (behaviour-lag KL 0.41 after four updates in the first server run). Members 1–4 train fully. Monte Carlo targets (λ = 1), so V stays calibrated for
  this agent. The log reports the value's calibration on new runs before each update (`value_mean_floor_rows` vs
  `outcome_mean_floor_rows`).
- The log line's `calibration` gives, per ascension group (A0-8, A9, A10) and act, floor rows of the new runs: mean
  outcome, ensemble value and each member's value (log only; `value_calib.py` does the same offline on stored chunks).
- Set `--runs-per-iter` to at least the number of runs in flight (actors × `CONC`), so a run spans one or two versions.

**Value-fit and other options (2026-11-13).** `--value-rows-per-run`, `--value-steps`, `--value-dropout`,
`--value-wd`, `--prob-avg`, `--policy-warmup-s` (see the trainer's docstring; chain s3a uses them through
`EXTRA_MTRAIN_ARGS` in `extra.env` from segment 24). Offline test of the value fit: `value_replay.sh` +
`value_replay_report.py`. Combat side: `train_combat.py --tee-dir/--skip-every`, `capacity_test.sh`,
`src/s2/py/combat_heldout.py`, `SHADOW_ARGS` in `run_s3.sh`; search keys `candidates` (4 from segment 24),
`cscale`, `cvisit`, `qspan`; `src/s2/search_params_test.sh`.

**Combat trainer.** As in S2 (improved-policy targets on fully searched moves, end-HP bin and potions kept), started from
the S2 weights, lr 2e-4, replay window 500k decisions.

## Commands

```bash
W=/mnt/4dbcffbd-bd6e-43c4-bc27-9274605ea5e4/AI/Slay_the_Spire_2_agent; cd $W
export NUGET_PACKAGES=$W/.cache/nuget DOTNET_CLI_HOME=$W/.cache/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$W/.cache/tmp
dotnet build src/s3/S3/S3.csproj -c Release -m:8
# encoder round trip
dotnet src/s3/S3/bin/Release/net9.0/S3.dll encode runs=300 ascension=0 out=src/s3/out/encode_a0
./.venv/bin/python -I src/s3/py/roundtrip_macro.py src/s3/out/encode_a0
# training run: name, actors, seconds
ASC=0 CONC=16 MTRAIN_ARGS="--runs-per-iter 256" src/s3/run_s3.sh NAME 12 2400
# unattended on the server: name, hours (settings as environment variables, see the script)
N=26 CSERVERS=3 CONC=32 ASC=10 ASC_MIX=0.5 setsid nohup src/s3/s3_supervise.sh NAME 24 > s3_supervise_NAME.log 2>&1 < /dev/null &
# evaluation on fixed seeds: combat weights, macro file, output, actors, arms
COUNT=210 ASC=0 src/s3/eval_s3.sh src/s3/weights/combat_s2p1.pt src/s3/data/macro_s1.pt src/s3/out/eval_X 8 "tf64:64 hp64:64:scorer=hp"
```

## Row layout v2 (2026-11-16)

Append-only extension of the macro row, so that the network can learn routes and prepare for the known boss. Every v1
offset is unchanged; `ML` in `S3/MacroLayout.cs` and the constants of `py/macro_s3.py` mirror each other.

| | v1 | v2 |
|---|---|---|
| ints | 460 | 462: + the act boss at 460 (`OBoss`), the act's second boss at 461 (`OBoss2`; act 3 at A10, else 0), both in the id space of the context encounter |
| floats | 208 | 344: + 16 x 8 option features from 208 (`FOptFeat2`), + 8 state scalars from 336 (`FScal2`) |
| outputs | 221 | unchanged |

Option features (map options only; computed on the act map by map point type, an Unknown point counts as nothing;
`RunAgent.MapFeatures2`): 0 rows to the nearest rest site / 8 (rows to the boss if there is none); 1 rows to the nearest
shop / 8, at most 1 (1 = none, or 8 rows and more); 2 fewest elites up to the first rest site (or the boss) / 3;
3 fewest fights up to there / 5; 4 most elites within three rows / 3; 5 an elite followed by a rest site; 6 a rest site
followed by an elite; 7 rows to the boss / 16. State scalars (every row): elites fought / 4, rest sites / 4 and shops / 3
visited in this act (the floor's own room included), free potion slots / 3, potion slots / 5, map row / 16, two reserved.

Network: the two boss ids have an embedding each (added to the act context token), the scalars a linear map (added to
the scalar token), the option features a second linear map next to the one of the 12 path features. All of it starts at
zero, has no bias, and index 0 of the boss embeddings stays zero, so a v1 file gives exactly its old outputs and rows
with zeros in the new columns are never affected. None of the inputs depends on HP (the value stays monotone in HP).
Old data: chunks and carry files of the v1 widths are read and zero-padded (`macro_s3.pad_rows`); the trainer restores
a v1 carry's Adam states for the old parameters. Old actor builds need `s3_server.py --mlayout 1` (bridges of the v1
widths); a v2 actor needs a server and trainer of this code (the widths are checked when the bridge opens).

```bash
# network side (CPU, seconds): a macro file, optionally run chunks of either layout
./.venv/bin/python -I src/s3/py/layout2_check.py src/s3/runs/NAME/macro.pt [src/s3/runs/NAME/mdata_archive]
# encoder side: prints 8 map decisions of the first run with the act maps (maps=K), for reading against the map
dotnet src/s3/S3/bin/Release/net9.0/S3.dll encode runs=2 ascension=10 maps=8 out=src/s3/out/encode_layout2
```

## Row layout v3 (2026-11-16)

Append-only again (every v1 / v2 offset unchanged); `ML` in `S3/MacroLayout.cs` and the constants of `py/macro_s3.py`
mirror each other; `py/macro_s3_v2.py` is the frozen v2 network (the reference of `layout3_check.py`).

| | v2 | v3 |
|---|---|---|
| ints | 462 | 478: + encounter token of the last 8 live fights at 462 (`OFightEnc`, most recent first, id space of ctx[5]), + room kind at 470 (`OFightRoom`: 1 hallway, 2 elite, 3 boss, 0 none) |
| floats | 344 | 472: + 8 x 6 fight numbers from 344 (`FFights`), + 80 map nodes from 392 (`FMap`) |
| scal2 | 6 used | column 6 = price of the next card removal / 200 (`XRemovePrice`) |
| opt_feat2 | map options | also shop options: price / 300, price / gold |
| outputs | 221 | unchanged |
| bytes per row (chunk / carry: ints int16, floats float32, meta 64) | 2,364 | 2,908 (+23%) |

Fight numbers (per fight, `RunAgent.EndFight` -> `FloorFacts.Fights`): HP lost / max HP at the start (-1..1), HP share
at the start, potions used / 3, turns / 20, floors ago / 20, 1 if in the current act. Practice fights are not listed.
T_f table rows list the fights before the current one (the current fight's HP loss is what the table varies).

Map nodes (`MacroEncoder.MapNodes`): every node of the current act's map (grid points with an edge, the start, the
boss(es)), sorted by (row, column), each ONE integer < 2^24 stored exactly in a float32 (`MapNode.Pack`; 0 = no node):
bits 0-3 map point type + 1, 4-8 row, 9-11 column, 12 visited (this act), 13 current, 14 reachable from the current
point (strict descendant; every node when there is no current point), 15-23 option-reach mask (bit j: reachable,
inclusive, from map option slot j; slots = the row's map options on a map decision, else the points the run can travel
to next, in the same column-then-row order). Act maps have up to ~70 nodes (24 encode-test runs: 70); with more than 80
the nodes behind the current row go first, then those no longer reachable. Why a float column: chunks and carries store
ints as int16, too narrow for 24 bits; a float32 holds the integer exactly and costs the same as two int16 columns.

Content properties (`dotnet S3.dll props out=src/s3/data/content_props.json mvocab=...`, `ContentProps.cs`; seconds;
commit the JSON with the code): per card id and upgrade level 0 / 1 the simulator's type, rarity, target, energy cost
(X / unplayable flagged), keywords, tags, its `CardEffectProbe` (damage, block, draw, energy, created cards, powers
applied with amounts: focus, strength, weak, vulnerable, ...; Silent sandbox) and the orbs it channels (played once in a
Defect sandbox: the probe's Silent has no orb slots); per relic / potion the rarity (potions: usage, target) and the
`ItemEffectProbe` effects. Only the simulator's MIT model classes are read. 532 cards (37 levels channel orbs), 296 relics,
64 potions; 5 vocabulary ids are missing in the simulator. Known gaps: effects the probes cannot observe (orb slots,
conditional effects) are absent; `macro_s3.load_props` turns the JSON into fixed feature vectors (card: 78 per level + the
upgrade delta = 156; relic 50, potion 56: one-hots + the 40 most frequent probe keys, signed log).

Network (`macro_s3.py` header): fight tokens (existing encounter embedding + room + 6 numbers) and map tokens (type, row,
column, flags, rows ahead of the current point) are encoded by the existing per-token blocks in a tensor of their own
(the old tokens keep their tensor shape: bit-identical outputs); 2 pre-norm self-attention blocks (4 heads, d 128,
FF 512) run over all state tokens with key padding; the old mean pooling is unchanged, fight and map tokens are pooled
separately into the trunk input; deck counts (22: deck size, Strikes, Defends, upgraded, attacks / skills / powers,
curses, cost buckets 0 / 1 / 2 / 3+ / X, orb / focus / exhaust / draw / block / damage cards, cost sum, relics, potions;
computed in Python from the row's ids and the property table, so the lookahead's cloned rows stay consistent) go into the
trunk input and the policy query; content properties are added to the card / relic / potion token and option
embeddings; every option attends over the unpooled state tokens (query: option embedding + policy query), map options
get a per-head bonus (start 2.0) on the map tokens of their own subtree. Every path ends in a zero layer (`V3_ZERO`:
attention output and FF second layers, pools, counts, properties, cross-attention output, shop prices); the inner layers
start random. A file without v3 parameters loads with its zero layers at zero (also on a hot reload over trained ones).
The value stays monotone in HP: nothing new depends on the row's own HP.

Trainer / server: chunks and carries of v1 / v2 / v3 widths are read (older rows zero-padded); Adam states of older
carries are restored for the old parameters (v3 parameters are registered last); `--freeze-v3` keeps the zero layers at
zero (control arm). `s3_server.py --mlayout 3` (default) / `2` / `1` serves bridges of the v3 / v2 / v1 widths.
**Switch builds together**: a v3 actor needs this Python (the bridge checks the widths); a v3 macro file cannot be loaded
by the layout2 tree; a layout2 actor needs `--mlayout 2`. Old macro files (segment 49) are served unchanged by v3.

Checks (server CPU, `/root/exp/layout3`, segment 49; `layout3_check.py ... --random-rows 70 --hp-rows 8 --encode
src/s3/out/encode_l3 --combat /root/exp/w/combat_s049.pt`: ALL CHECKS PASSED, 88 s): with random values in every
v3 input the outputs equal the frozen v2 network's bit for bit (max |difference| 0.0: values, T_f serve rows with and
without milestone blending, option logits of all 7 decision types present, slot logits, milestone heads); v1 / v2 width
rows padded (0.0); a hot reload of segment 49 over random v3 parameters gives its function (0.0). Each zero layer, made
random, moves what it should (value and policy; count_q and x_out the option logits only, value exactly unchanged; shop
prices the shop logits only); fight and map inputs reach the outputs through the pools, the self-attention and the
cross-attention; flipping option-reach bit 0 moves only map option logits (6.9). Value monotone in HP with every v3
parameter random (largest fall 0.0, (a, c) spread over HP 1.5e-5). v1 / v2 / v3 chunks in one window; v1 and v2 carries
restore their Adam states (71 / 75 of 130 parameters), gradients reach all 18 zero-layer tensors, `--freeze-v3` keeps
them at zero. C# round trip on 24 encode-test runs (2,758 rows: 1,647 decisions, 350 map decisions): 153,251 map nodes
and 18,716 fight tokens unpacked exactly as the encoder meant them, at most one current node, every map option slot
reaches its own subtree. `s3_server.py --mlayout 3 / 2 / 1` answers bridges of 478/472, 462/344, 460/208 (vs
Ensemble.serve 1.7e-6, batch padding). End to end (`l3_smoke.sh 3`, CPU, sims 8): 6 runs, 0 errors, v3 chunks with map
nodes on 100% of rows (51-70 nodes) and fights on 97%, look_beta=20 active (18 of 70 eval card picks changed), T_f
HP-axis check 3.9e-7. v3 parameters per member: 545,668 (of 2,136,650); a 5-member file is 43 MB (segment 49: 31 MB).

```bash
# server tree (copies /root/sts2, overlays /root/exp/l3_upload, builds): src/exp/layout3_setup.sh
# checks (CPU): network, trainer files, C# round trip, server bridges
./.venv/bin/python -I src/s3/py/layout3_check.py /root/exp/w/macro_s049.pt --encode src/s3/out/encode_l3 --combat /root/exp/w/combat_s049.pt
dotnet src/s3/S3/bin/Release/net9.0/S3.dll encode runs=24 ascension=10 maps=0 out=src/s3/out/encode_l3
src/exp/l3_smoke.sh 3                       # end to end on CPU: actors write v3 rows, look_beta=20
./.venv/bin/python -I src/s3/py/macro_cpu_bench.py /root/exp/w/macro_s049.pt --compare-v2 --no-cuda --batch 64,256 --threads 8
# experiment (GPU: coordinate first). 1 data: frozen segment 49 self-play writing v3 rows, no trainer
cd /root/exp/layout3 && (setsid nohup src/exp/l3_gen.sh 3600 56 2 > /root/exp/out/l3gen.log 2>&1 &)
# 2 offline value arms v3 / ctl from segment 49 (+ the segment-49 carry rows, zero-padded, training only)
MDATA=/root/exp/out/l3gen/mdata src/exp/l3_value.sh /root/exp/out/l3value --steps 4000 --carry /root/exp/s049init/carry_macro.pt
# 3 play test of each arm's file (same seeds, lookahead beta 20)
B="greedy_map=1:potion_values=/root/exp/w/potion_values.json:look_beta=20"
for A in v3 ctl; do COUNT=512 FIRST=0 SEEDS=XV ASC=10 CONC=64 OBS=5 src/exp/eval_l3.sh /root/exp/w/combat_s049.pt \
  /root/exp/out/l3value/macro_$A.pt /root/exp/out/l3eval/$A 8 "look20$A:64:$B"; done
```

## One-step lookahead for the other deterministic decisions (2026-11-16, branch `lookx`)

The card-reward lookahead (`look_beta`, +2.8 points over 4,110 fresh seeds at beta 20) extended to every macro
decision whose result is known without chance. Actor options (eval; each a beta, 0 = off, all off = the old build bit
for bit; `look_act` gates all of them): a candidate's logit gets beta x (z(post) - z(state)), z = the ensemble value
logit of the row, the post-state built from the state row by `MacroEncoder.With*` exactly as `Encode` would encode the
player after the change (deck kept contiguous, deck / upgrade / enchant / relic / potion scalars recomputed, so the
Python deck counts follow). All rows of a decision are evaluated in one batch.

| Option | Decision | Post-state | No bonus |
|---|---|---|---|
| `look_remove` | removal target (shop service, "remove" prompts) | that slot removed, later slots move down | (transform prompts: not deterministic) |
| `look_smith` | upgrade target (rest-site smith, "upgrade" prompts) | that card's upgrade level + 1 | |
| `look_rest` | rest-site option | heal: the state row's value at HP + heal (30% max HP through `Hook.ModifyRestSiteHealAmount`, e.g. Regal Pillow; read off the HP ramps, no evaluation); smith: the smith head's argmax target upgraded (one evaluation) | lift, dig, cook, hatch, ... |
| `look_shop` | each purchase | card added / relic added / potion in the first free slot, gold - price | removal service (its target: `look_remove`), LEAVE |
| `look_relic` | Ancient relic choice | relic added; only when no offered relic has a pickup effect (`AfterObtained` override or `HasUponPickupEffect`); `look_relic_any=1`: always, the token alone | |

Not modelled: relic side effects of a change (eggs upgrading an added card, Lucky Fysh's 15 gold per added card, Stone
Humidifier's max HP after a heal); a multi-card prompt scores each pick against the state before the earlier picks. Relic rewards are taken
automatically and STS2 has no boss-relic choice, so DRelic is never asked. Notes per type: `lookahead_TYPE`,
`look_changed_TYPE` / `look_same_TYPE` (greedy pick with vs without the bonus), `look_relic_skipped_pickup`; the run
record gets `look` counts (only when non-empty); `look types` line at the actor's end (mean |dz|). `look_dump=FILE`
writes each decision's state row, post-state rows, pick and the state row at the next callback for
`src/exp/look_rowcheck.py`.

Checks (server CPU, segment 49, sims 8, base = map argmax + belt rule + look_beta 20; `lx_check.sh`, servers pinned to
2 cores each: `s3_server.py` on CPU takes as many torch threads as its affinity has cores, 64 unpinned):
- (a) all look_* off vs the branch base build, same seeds, one server per actor: 3 eval runs (run records identical,
  67 fight-log lines identical), 1 self-play run (record, 24 fight lines and the 369,444-byte macro chunk identical).
- (b) `look_rowcheck.py` on 315 lookahead decisions (`lxcheck` + `lxcheck2`: 15 eval runs, 3 self-play runs with
  `look_relic_any=1`): every post-state changes only its fields (remove: deck slots from the removed one on + deck /
  upgrade counts; smith: one d_upg + upg; shop card: next deck slot + deck + gold; relic: next relic slot + relics
  (+ gold); potion: potion slots, potions, free slots, gold), counts equal the ids everywhere, 0 errors. Picked
  post-state vs the next state: 90 heals predicted exactly; 7 mismatches in 159 picks, all expected: the
  token-only relics with pickup effects (Cursed Pearl, Large Capsule, Archaic Tooth), a 4-card removal prompt (the
  last pick sees all four removed), Lucky Fysh's +15 gold on two shop cards.
- (c) 0 errors; per 12 eval runs at beta 20: shop 76 decisions (15 picks changed), rest 72 (1), smith 22 (1),
  remove 9 (0); Ancient choices with only plain relics are rare (0 of 27 in eval, `look_relic_skipped_pickup`).

```bash
# side tree (copies /root/sts2, overlays a git archive of the branch, builds); ref = the branch base for the identity check
src/exp/lookx_setup.sh /root/exp/lookx /root/exp/lx_upload.tgz; src/exp/lookx_setup.sh /root/exp/lookx/ref /root/exp/lx_ref_upload.tgz
src/exp/lx_check.sh 4 2          # CPU: (a) identity with all off, (b) row check, (c) smoke with all look_* = 20
src/exp/g12.sh                   # experiment 30: XV 0-511, base (look_beta 20) and one arm per look_* (GPU)
./.venv/bin/python -I src/exp/g12_cmp.py /root/exp/out/g12
```

## Side tasks of 2026-11-17 (branch `side1`; PROPOSAL §6.1 steps 8-10)

All off by default; off = the base build bit for bit (checked below).

| Option | What |
|---|---|
| `potion_reserve=W` (1), `potion_reserve_min` (12), `potion_reserve_hp_lo` / `_hp_hi` (0.2 / 0.3) | Reservation price for the top potions (potion value >= min; needs `potion_values`; refused with `boss2_model`, `tf_stake`, `potion_scale`, `hp_scale`, `potion_kind_scale`, `scorer=hp`). In every fight scored by the macro table except the act-3 bosses: T'(h, P) = min(1, T + (W - 1) ramp(h / max HP) max(0, T(h, P) - T(h, P without its top potions))), ramp = clamp((x - lo) / (hi - lo), 0, 1) on the end-HP axis; applied before free disposal (TfScorer.Set keeps it). Note `potion_reserve_fight`; fight-log key `reserve_t_max0` (T_max before); actor line `potion reserve: fights, entries raised, mean T_max ratio` |
| `tf_dump_all=1` | `tf_dump` covers every room kind of every act (keys monsterA / eliteA / bossA, boss3) |
| (always) `root_v0`, `root_q0` in the fight log | The network's root value of the fight's first evaluated move and the root Q of its first searched move (relative T_f; x `t_max` for value units), S3 and S2 fight logs; null when unset |
| `clone_probe=1|2`, `clone_dump=FILE` | At every non-Ancient event decision with two or more open options: `RunState.CloneReseeded` (timed, dropped); 2: each open option also chosen once on a clone of its own, and again in a second world (same visible result = the option does not depend on the hidden future). Notes `clone_ok`, `clone_reject:Reason:event`; play unchanged |

Checks (server CPU, everything pinned to 4 cores, segment 49, sims 8, deployment base incl. `look_smith` / `look_shop`;
`src/exp/side1_check.sh`): see the PROGRESS entry of 2026-11-17 for the numbers.

```bash
src/exp/lookx_setup.sh /root/exp/side1 /root/exp/side1_upload.tgz; src/exp/lookx_setup.sh /root/exp/side1_ref /root/exp/side1_ref_upload.tgz
src/exp/side1_check.sh 3 1        # CPU: identity (a), clone_probe=2 identity (b), potion_reserve=3 smoke + reserve_check.py (c)
src/exp/side1_probe.sh 50 2 4     # CPU: Task 3, clone probes on 50 eval runs; src/exp/clone_report.py
src/exp/side1_luckdata.sh         # CPU: Task 2 rows (s3c segment 1 from Nextcloud) -> /root/exp/side1_luck/s3c_s001.npz
src/exp/g14.sh                    # GPU (not launched): base / W 2 / W 4 on XV 0-511; src/exp/reserve_cmp.py /root/exp/out/g14
```

## Macro server with CUDA graphs (2026-11-17, branch `mgraphs`)

With layout v3 the macro process (`s3_server.py --serve macro`) needed ~33 ms for a batch of ~18 rows (v2: ~11 ms), was
91% busy at 52 actors, and actors lost ~15% throughput waiting for it. A batch is a few hundred small kernels per member
launched from Python, so `--mgraphs 1` runs it as CUDA graphs (`macro_graphs.py`; default 0 = the old code path, outputs
byte-identical: the refactored `Ensemble.serve` was checked bit-identical against the previous one on CPU):

- each batch is split as before into decision rows (member 0's policy forward) and value rows (every member's value
  forward); each half is padded to its own bucket (`--mbuckets`, default 8,16,32,64,128,256) and replays the graph of
  (half, bucket), captured at its first use after a warm-up on a side stream. A half larger than the largest bucket
  runs eagerly. The server hands over only its real rows (no MBUCKETS padding). One stream synchronisation per batch.
- the captured forward is the eager one with `b["static"]`: all 8 fight and 80 map slots instead of the batch's largest
  count, option embeddings by masks instead of boolean indexing (same function per row; only matrix shapes differ).
- milestone blending is inside the graphs; a change of `milestones` / `alpha` / `boss1_share` drops the graphs.
- reloads: a file that fits the served ensemble (member count, vocabulary, parameter shapes) is copied into the same
  parameter tensors (graphs kept, they read the new weights); anything else builds a new ensemble and the graphs are
  captured again (`[s3srv] macro ensemble rebuilt`). The stats line ends with `macro graphs N captured (M live), eager
  rows R, rebuilds K`.
- `--mdtype bf16` runs the macro forward under autocast (default fp32). Values feed the lookahead and the T_f prices:
  use it only if the check shows a large extra gain and an acceptable error.
- memory: the graphs have their own pool (about one forward's activations at the largest bucket) next to the eager cache;
  `mgraphs_check.py` step (m) prints both.

```bash
# check + benchmark (GPU; caps itself at 1.4 GB allocator memory, ~2 min): correctness per bucket against eager,
# weight swaps / milestone switch / member-count rebuild through MacroServe.reload, ms per call at 8 / 18 / 32 / 64 rows
./.venv/bin/python -I src/s3/py/mgraphs_check.py /root/exp/w/macro_s049.pt /root/exp/out/l3gen/mdata
# CPU smoke of the same code path (no capture: the static-shape forward runs eagerly)
CUDA_VISIBLE_DEVICES= ./.venv/bin/python -I src/s3/py/mgraphs_check.py src/s3/data/macro_s1.pt --rows 400
# serving
python -I src/s3/py/s3_server.py --serve macro ... --mgraphs 1
```

## Checks that exist

| Check | Result |
|---|---|
| S3MacroNet vs S1 MacroNet on the same input | value and policy logits identical |
| HP axis from one row vs direct evaluation | 3e-7 (Python), 9e-7 (C#, checked in every run) |
| Encoder round trip, 300 runs / 13,891 floors | 13,871 floors equal; the rest are belts of more than five potions (19) and one deck case; value difference ≤ 1.2e-7 |
| Actor log-probabilities vs trainer recomputation | ≤ 0.0035, all decision types |
| Macro vocabulary on the agent's runs | 0 unknown cards, relics, potions, enchantments, encounters |
| Supervisor on the server | a trainer killed on purpose: segment stopped within a minute, chain continued from validated weights; supervisor killed: `s3_resume.sh` restarted it with the stored settings and end time |

The round trip also showed three properties of the S1 data (not of the encoder): a relic that is later removed appears
on every earlier floor of that run (age 63); when a card id has several copies, upgrades are attributed to an arbitrary
copy; enchantment amounts are the final amounts on every floor.

## Not covered yet

Potions gained during a fight and in-fight permanent changes have no T_f; a full belt loses a potion reward (no discard
decision); card-reward alternatives given by relics are never chosen; V_run has no map input, so map choices rest on the
MAP head alone; relic counters and forecasts are not in the combat input; in-flight runs are finished, not aborted, at
the time limit (a tail of 1–2 minutes).
