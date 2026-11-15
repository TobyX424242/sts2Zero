# S3: full-game self-play

Whole Defect runs in iRyougi. Every out-of-combat decision is made by the macro network; every fight is played by
the network-guided search of S2 on the live combat, scored by a T_f table built at the fight start from the current
macro value network. Both networks are trained on the agent's own runs. Builds on `src/s0` and `src/s2`.

Full architecture, experiment history and results are in `doc/` (`architecture.md`, `experiments.md`, `findings.md`).
This file is the code guide.

| Path | Content |
|---|---|
| `S3/MacroLayout.cs` | Row layout shared with Python (`ML`), macro vocabulary, `MacroEncoder`: the S1 state after a floor from a live `RunState` (and the post-fight state) |
| `S3/MacroBridge.cs` | Shared-memory client for the macro network; `MacroBridgeEvaluator` batches rows of all runs of a process |
| `S3/RunAgent.cs` | `RunAgent : IRunDecisionSource`: macro decisions, live T_f table per fight, battle search, perturbed fight starts, records |
| `S3/Records.cs` | Run record, macro training chunks (`*.s3r`) |
| `S3/Program.cs` | `S3 play key=value ...` (self-play or evaluation; keys at the top of the file), `S3 encode`, `tftest`, `prompttest` |
| `S3/EncodeTest.cs` | Writes simulated runs as Spire Codex style run records plus the encoder's rows, for `py/roundtrip_macro.py` |
| `S3/PromptTest.cs` | End-turn prompt expansion test (Knowledge Demon: Disintegration vs Mind Rot) |
| `py/macro_s3.py` | `S3MacroNet` (S1 `MacroNet` + MAP and EVENT decisions), `Ensemble` file format, row layout, vocabulary export |
| `py/s3_server.py` | Inference server for both networks (`--serve combat`, `--serve macro`: run one process of each) |
| `py/combat_pipe.py` | Combat batches with CUDA graphs per (batch, token) bucket and two batches in flight (`--graphs 1`, `--overlap 1`; off by default) |
| `py/train_macro_s3.py` | Macro trainer: PPO for the policy member, V_run for all members |
| `py/roundtrip_macro.py` | Encoder round trip against `src/value_v1/build_floor_dataset.py` |
| `py/ext_test.py`, `py/ms_check.py`, `py/macro_cpu_bench.py` | Layout-v4/v5 ext-head test, milestone-head check, macro CPU benchmark |
| `run_s3.sh` | Local training run: combat trainer + macro trainer + inference servers + N actors |
| `eval_s3.sh` | Evaluation on fixed run seeds |
| `eval_asc_sweep.sh` | One checkpoint at A0-A10 on the same seeds |
| `data/macro_vocab.json` | Macro vocabulary for C# (the S1 checkpoint itself is not redistributed) |
| `data/boss2_w2.json` | Logistic double-boss model fitted on self-play logs (`fit_boss2.py`) |
| `tools` in the parent | `a10val_*.py`, `alloc_compare.py`, `curve_runs.py`, `fit_boss2.py`, `value_calib.py`, ...: analysis scripts named after the experiments in `doc/experiments.md` |

## How a run is played (`RunAgent`)

**Macro decisions.** The row sent to the network is the current state in the S1 conventions (floor counted from the
Ancient of act 1 = 1, room type of the map point, model id of the floor's first room, card age = floor - floor added)
plus the decision: a type and up to 16 options, or a choice among deck slots.

| Decision | Type | How |
|---|---|---|
| Card reward, other "choose one card" offers | CARD_REWARD | S1 head; options = cards + SKIP |
| Ancient (relic choice at act start) | ANCIENT | S1 head; options = relics |
| Rest site | REST, then SMITH_TARGET | S1 heads; options = the options offered |
| Shop | SHOP, then REMOVE_TARGET | S1's per-item buy logits as a softmax against LEAVE (logit 0), one purchase at a time |
| Deck selections of events and relics | SMITH_TARGET / REMOVE_TARGET | S1 heads; the purpose comes from the simulator patch; other selections are answered at random |
| Map point | MAP (new) | New head, uniform at the start. Option = room type + 12 path features |
| Event option | EVENT (new) | New head, uniform at the start. Option = embedding of the hashed "event:option" key |
| Potion reward, gold, relic | - | Taken (a potion is lost if the belt is full) |
| Crystal Sphere, Fake Merchant | - | Simulator default |

Training samples from the policy. Evaluation (`greedy=1`) takes the most probable option for the S1 decision types and
samples the two new heads (their argmax repeats itself on repeatable event options).

**HP and the policy.** S1 removed HP from the encoder so that the value is monotone in HP through its own head; the S1
policy head therefore could not see HP. `S3MacroNet` adds the 26 HP ramp features to the policy query through a layer
that starts at zero (S1 behaviour at the start, HP dependence learned by PPO).

**Fights.** At the first decision of a fight the agent builds the table T_f(h, P) for every HP h and every subset P of
the potions held: one macro row per subset (the post-fight state: this floor's state with the expected gold reward and
that potion belt). The value head is monotone in HP by construction, so the server returns (a, c) and the whole HP axis
follows from one row (checked in every run: the actor reports the largest difference between this formula and the
network's own logit, 9e-7). T_f = 1 at the fight that wins the run. The table goes into `TfScorer`, and the S2 search
(`NetSearch`, `FightSignals`) runs unchanged on the live combat state. `scorer=hp` replaces the table by end HP /
max HP (the comparison arm of gate G3). `kappa` subtracts kappa x the members' spread from the logit (off by default).

**Perturbed fight starts** (`perturb`, 10% of self-play runs): before a fight, with probability 0.35, HP is shifted by
up to +/-30% of max HP and a potion is added or removed. Flagged in the run record; excluded from win-rate summaries.

**Practice fights** (`practice_boss=K`, `practice_elite=K`, `practice_min_act=A`; self-play only, off by default): at
the first decision of a live boss (elite) fight of act >= A, K copies of the fight are played to the end before the
live fight goes on, each from a resampling clone of the live combat. They are recorded to the combat chunks like a live
fight; the live run is unchanged.

**Tree reuse** (`reuse=1`, `reuse_cap`): the fight's search keeps its node table between moves. The fresh budget and
sequential halving are unchanged; the target uses reused + fresh statistics.

**Death value and the double boss** (off by default): `death_value=progress` prices a death as the run progress already
secured; `boss2_model=data/boss2_w2.json` scores the first of two act-3 bosses with a fitted W2 instead of the macro
table. Both were tested; neither improved play with frozen networks (`doc/experiments.md`).

**End-turn prompts** (`endturn_prompts=1`, default): a prompt that opens during End Turn (Knowledge Demon's curse
choice, start-of-turn power prompts) is chosen by the search: the root probes End Turn once on a resampled clone and
splits it into one action per answer of the first prompt.

**Combat input layout v4/v5.** v4 (2026-11-15) puts card damage / block and hidden power counters in feature slots
that were 0 in v3. Networks without `"ext": 1` mask them out (identical outputs); `"ext": 1` networks read them through
`ext_feat`, whose last layer starts at zero. v5 (`obs=5`) adds counters of 14 relics in the masked Relic f0/f1 slots.
`fp_ext=1` adds the same values to the search's state fingerprint.

**Records.** Combat decisions go to the S2 chunk format (`*.s2d`, unchanged). Macro rows go to `*.s3r`: the state after
every floor (value examples, as in S1) and every decision with the chosen option, its log-probability, and the legal
deck slots.

## Training (`run_s3.sh`)

Processes: combat trainer, macro trainer, `CSERVERS` combat server processes, one macro server, N actors (each plays
`CONC` runs at once on one thread). `ASC` is the target ascension, `ASC_MIX` the share of runs played at it (the rest
at the levels below; `asc_weights` sets their weights).

**Macro trainer.** Every `--runs-per-iter` runs:
- *Outcome.* outcome = alpha x won + (1 - alpha) x acts cleared / 3, alpha = win rate of the window / 0.2, at most 1.
  A new agent wins almost no runs; with a pure win target V_run collapsed to 0 within two updates.
- *Policy (member 0).* PPO, clipped ratio against the recorded log-probabilities, GAE (gamma = 1, lambda = 0.95) with the
  outcome as the terminal reward, entropy bonus, KL penalty toward the frozen S1 policy on the S1 decision types with a
  weight falling linearly to zero at `--kl-runs` runs. Steps stop when KL(policy at the iteration start || policy now)
  exceeds 1.5 x `--target-kl`. Advantages are divided by max(their standard deviation, 0.02).
- *Value (all five members).* Cross-entropy against the outcome on floor states and decision states sampled from the
  last `--window-runs` runs; each member draws its own samples. The policy member trains only its value heads here
  (state vector detached): a value gradient into the encoder it shares with the policy moved the policy far outside
  PPO's trust region. Members 1-4 train fully. Monte Carlo targets (lambda = 1), so V stays calibrated for this agent.
- The log line's `calibration` gives, per ascension group and act, floor rows of the new runs: mean outcome, ensemble
  value and each member's value.

**Combat trainer.** As in S2 (improved-policy targets on fully searched moves, end-HP bin and potions kept), started
from the S2 weights; lr 5e-5 in the final design, replay window 6M decisions, surprise weighting and EMA.

## Commands

```bash
W=$(pwd); cd "$W"
export NUGET_PACKAGES=$W/.cache/nuget DOTNET_CLI_HOME=$W/.cache/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$W/.cache/tmp
dotnet build src/s3/S3/S3.csproj -c Release -m:8
# encoder round trip
dotnet src/s3/S3/bin/Release/net9.0/S3.dll encode runs=300 ascension=0 out=runs/encode_a0
./.venv/bin/python -I src/s3/py/roundtrip_macro.py runs/encode_a0
# training run: name, actors, seconds (COMBAT_INIT / MACRO_INIT must point at checkpoints; not included)
ASC=0 CONC=16 MTRAIN_ARGS="--runs-per-iter 256" src/s3/run_s3.sh NAME 12 2400
# evaluation on fixed seeds: combat weights, macro file, output, actors, arms
COUNT=210 ASC=0 src/s3/eval_s3.sh COMBAT.pt MACRO.pt runs/eval_X 8 "tf64:64 hp64:64:scorer=hp"
```

## Checks that exist

| Check | Result |
|---|---|
| S3MacroNet vs S1 MacroNet on the same input | value and policy logits identical |
| HP axis from one row vs direct evaluation | 3e-7 (Python), 9e-7 (C#, checked in every run) |
| Encoder round trip, 300 runs / 13,891 floors | 13,871 floors equal; the rest are belts of more than five potions (19) and one deck case; value difference <= 1.2e-7 |
| Actor log-probabilities vs trainer recomputation | <= 0.0035, all decision types |
| Macro vocabulary on the agent's runs | 0 unknown cards, relics, potions, enchantments, encounters |
| End-turn prompt expansion (`prompttest`) | all pass (Knowledge Demon: Disintegration vs Mind Rot) |
| Layout v4/v5 off switch | identical runs and fight logs on 6 seeds; v5 smoke 124 runs, 0 errors |

## Not covered yet

Potions gained during a fight and in-fight permanent changes have no T_f; a full belt loses a potion reward (no discard
decision); card-reward alternatives given by relics are never chosen; V_run has no map input, so map choices rest on
the MAP head alone; relic counters and forecasts are not in the combat input; in-flight runs are finished, not aborted,
at the time limit.

## Not part of this repository

The server-side supervisor, status/backup tooling and the local training panel are intentionally excluded. Training
chunks and checkpoints are not redistributed. See `doc/code-map.md`.
