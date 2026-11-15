# Training pipeline

This document is the practical companion to `doc/architecture.md`: what was trained, on what data, with which
hyperparameters, and how a run is evaluated. The code is in `src/s1`, `src/s2`, `src/s3`; the simulator is in
`src/sims`.

## 0. Environment

- .NET SDK 9.0.118, Release build.
- Python 3.12 venv at `./.venv` with PyTorch 2.6 (CUDA 12.4), NumPy 2.5, orjson 3.13.
- The scripts assume the repository root as the working directory and use `./.venv/bin/python -I`.
- Training data, battle-start pools, T_f tables and checkpoints are **not** included (see `README.md`).

Build:

```bash
export NUGET_PACKAGES=$PWD/.cache/nuget DOTNET_CLI_HOME=$PWD/.cache/dotnet
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$PWD/.cache/tmp
dotnet build src/s0/S0Tests/S0Tests.csproj -c Release -m:8
dotnet build src/s2/S2/S2.csproj -c Release -m:8
dotnet build src/s3/S3/S3.csproj -c Release -m:8
```

## 1. S0: simulator extensions

S0 adds what the simulator lacked for search:

1. **Resampling clone.** `ResamplingClone.Resampled(state, seed)` reseeds every run/player/monster RNG stream in the
   clone and reshuffles the unknown region of the draw pile. Known top/bottom positions are tracked by the simulator
   patch. The exact clone (`state.Clone()`) is kept for the true-future upper-bound arm.
2. **Card choices inside clones.** `SearchCardSelectionSource` answers card-selection prompts in clones from a
   script, a policy, a suspend/replay mechanism, or a pause/resolve mechanism.
3. **Battle-start import.** `BattleStartLoader` rebuilds a fight from a Spire Codex floor (deck, relics, potions, HP,
   encounter) through the simulator's transplant importer.
4. **Legal actions and observation encoders.** `ActionSpace` enumerates identity-addressed actions including potions
   and prompt answers; `CombatObservation`/`ObsLayout` encode the tokens; `Bridge` is the shared-memory client.

Validation: 10,548 exact-clone comparisons with 0 mismatches; resampling leaves the observable state unchanged and
matches the posterior for the next draw (20.57% vs 21.12% expected, z = -1.63); 2,174 card prompts handled with 0
errors; 998/1,000 human battle starts imported and played to the end. Full numbers in `src/s0/README.md`.

## 2. S1: macro warm start

### Data

- 1.2M clean single-player standard-mode unmodded A0-A10 release runs from the Spire Codex dump; 965,021 training
  runs after dedupe and split.
- Pinned cell: Defect A10 v0.111.0: 16,302 runs, 471k floors, 15.5% win rate.
- Per-floor example: state after floor g; labels = run won and floors remaining. Decks and relics rebuilt backward
  from the final state; potions rebuilt forward (a backward rebuild leaked future potions). Terminal floor excluded;
  cards/relics added after floor g dropped; no act-boss identity; 158 impossible runs dropped.
- Vocabulary: ids seen in >= 20 training runs; unseen ids map to `<unk>`.

### Value model

- No-attention encoder (3 residual MLP blocks, d = 128), mean pooling, 512-256-128 trunk.
- 26 HP ramps; value logit `a(z) + sum softplus(b_k(z)) phi_k(hp, max_hp)`.
- Uniform floor sampling, state-only inputs, natural frequencies, early stopping on validation log loss.
- Ensemble of 5 seeds. Light warm start: 2 epochs with the learning rate fully decayed (human data is only a warm
  start; S3 retrains on the agent's own outcomes).

### Policy model

The same encoder plus a policy head over offered options and deck slots. Decision dataset: 31.3M decisions
(card reward 16.2M, rest 6.0M, smith target 3.1M, ancient 2.4M, shop 2.2M, removal 1.4M, relic 74k). Training is
behaviour cloning on the human choice, with a per-decision-type softmax.

### Results

See `doc/results.md` (G1 table) and `doc/data/s1_final.json`.

## 3. S2: combat warm start

### Data

- Battle-start pool from all Defect A10 v0.111.0 combat floors in the dump: 253,485 training starts (19,098 runs) and
  13,244 held-out starts (984 runs), split by run hash (5% held out).
- A `T_f` table per battle start from the S1 ensemble's post-fight state: every HP from 1 to max HP and every subset of
  the starting potion belt. Free disposal applied; `T = 1` for the final boss; `T(0) = 0`.
- The server later rebuilt the pool at 256,785 / 13,384 starts; the held-out numbers in the results use 9,461 starts
  after filtering.

### Self-play

- 30 actors x 33,334 fights (1M fights total), 33 fights in flight per actor.
- Search budget 64 simulations on 25% of moves (policy targets), 16 on the rest; Gumbel noise on.
- Each actor plays a battle start several times with different resampled chance (`group=3`).
- Training: 1M-fight run, about 163k trainer steps, replay window 1M decisions, cosine learning rate 1e-3 -> 1e-5
  over the run. Outcome head from the start; the scalar head was run for the A20 comparison.
- Final weights `s2p1` (not redistributed); they are the S3 combat initialisation.

### Results

G2 on 9,461 held-out starts (search 64 wins 91.8%, network alone 87.2%, one-turn 90.9%, true future 93.9%; hidden
information costs 0.026). The full table and the human comparison are in `doc/results.md`.

## 4. S3: full-game self-play

### One training run

`src/s3/run_s3.sh NAME N_ACTORS SECONDS` starts:

- one combat trainer (`src/s2/py/train_combat.py`) and one macro trainer (`src/s3/py/train_macro_s3.py`);
- `CSERVERS` combat inference processes and one macro inference process (`src/s3/py/s3_server.py`);
- N C# actors (`S3.dll play mode=selfplay`), each playing `CONC` whole runs on one thread.

Actors write run logs, fight logs, combat chunks (`*.s2d`) and macro chunks (`*.s3r`). Trainers publish versioned
weights; actors pick up changes between runs. Trainers can carry their replay window and optimizer state across
segments (`CARRY=1`); chunks are deleted after reading unless `KEEP=1`.

### Final S3 configuration

| Item | Value |
|---|---|
| Actors / runs in flight | 85 actors x 32 runs |
| Combat inference processes | 4 + 1 macro process |
| Search | 64 sims on 25% of moves, 16 otherwise; hallway factor 0.5; every boss move at 64 |
| Root candidates | 4 (down from 16) |
| Ascension | target A10; 21.25% of runs at A10, the rest weighted to A7-A9 (weights 0,0,0,0,0,0,0,2.762,3.181,4.057) |
| Perturbed starts | 10% of runs, 35% chance per fight, HP +/-30% and a potion +/- |
| Practice fights | boss copies from act 2, 25% of their moves at full search |
| Macro PPO | every 2,560 runs; target KL 0.03; KL-to-S1 weight to zero over 75k runs |
| Macro value | window 50k runs, 16% of each run's rows, 150 steps/update, dropout + weight decay 0.1, probability averaging |
| Combat | lr 5e-5, replay window 6M decisions, carry 3M, surprise 0.5, EMA 0.999 |
| Segments | 1.5 h, each followed by an evaluation on 300 fixed A10 seeds |
| GPU sharing | CUDA MPS + CUDA graphs + two batches in flight |

The ascension mix is a curriculum: A10 alone gave too little experience of acts 2-3 early, and A0 alone does not
calibrate the target level. The weights changed over the run (A10 share 0.5 -> 0.6 -> 0.2-0.225; A6 dropped from
segment 40). See `doc/experiments.md`.

### Macro trainer

Per `--runs-per-iter` runs:

- **Outcome:** `alpha * won + (1-alpha) * acts_cleared / 3`, with `alpha = win rate of the window / 0.2`, capped at 1.
- **Policy (member 0):** PPO, clipped ratio, GAE (gamma = 1, lambda = 0.95), entropy bonus, KL penalty to the frozen
  S1 policy on S1 decision types, linearly decayed to zero. The update stops at KL > 1.5 x target KL. Advantages are
  normalised by `max(std, 0.02)`.
- **Value (all members):** cross-entropy to the outcome on floor and decision rows from the last window. Each member
  samples its own rows. Member 0 trains only its value heads (state detached); members 1-4 train fully.
- **Logs:** fresh-run value loss, table and constant baselines, per-ascension/act calibration, per-member spread,
  explained variance, and the share of the achievable value gain over the table.

### Combat trainer

Expert iteration over the actor chunks:

- Policy: cross-entropy to the improved-policy target on moves searched at full budget.
- Value: 41 end-HP bins with HL-Gauss targets, 6 potion-kept logits; death is bin 0.
- Replay window 6M decisions, batch shuffling, surprise weighting (0.5; upweights rare outcomes), EMA 0.999,
  learning rate 5e-5.
- A **shadow trainer** can train a second network (d = 256, 4 layers, 3.9M parameters) on the same chunks. It is
  compared with the playing network on fresh chunks (policy KL, end-HP/survival/potion losses) and on paired
  evaluation seeds; only then does it replace the playing network (`SHADOW_ARGS`, `SHADOW_FROM_PLAYING=1`).

### Weight and data flow

```text
actors (C#) --*.s2d / *.s3r--> trainers --> versioned weights --> inference servers --> actors
   |                              |
   |                              +--> carry_combat.bin / carry_macro.pt at segment end
   +--> run logs / fight logs (evaluation and analysis)
```

Every trajectory records the model version that produced it; the actor's log-probabilities are checked against the
trainer's recomputation (<= 0.0035 in absolute value, all decision types).

## 5. Evaluation

- Fixed seed sets, never used in training. Evaluation seeds are strings `EVAL000000`, `EVAL000001`, ... so both arms
  see the same run seeds.
- Primary metric: Defect A10 win rate on 300 fixed seeds for the curve, 1,000 for a gate. The ascension sweep uses 154
  seeds per level.
- Greedy macro (argmax on S1 decision types; the two new heads are sampled, because their argmax loops), no search
  noise, 64 simulations on every move.
- Statistics: Wilson 95% intervals for win rates; McNemar's test for paired win/loss differences; paired bootstrap for
  mean relative T_f; spread over training seeds where a result depends on a training run.
- Secondary metrics: mean floor, act 1/2 clear rate, deaths by act and encounter, potions used, bosses won per run.

Gates:

| Gate | Requirement | Result |
|---|---|---|
| G1 | macro value calibrated (ECE bands <= 0.02), HP monotone, better than the 100k baselines, T_f builder tests pass; potion dominance <= 1% | passed except +1 potion 2.82%, accepted by the owner |
| G2 | search 64 > network alone and > one-turn search, paired CI excluding 0; true-future arm reported | passed (a, b); hidden-information cost reported |
| G3 | A10 win rate > human 15.5% (CI excluding), and T_f-scored search > HP-scored search with the macro fixed | human threshold passed at segment 35+; T_f vs HP tested separately (no measurable difference at A0 with the early agent) |

## 6. Compute and throughput

- One combat inference process saturates at about 27k states/s. S3 with 26 actors early: 29.4k simulations/s,
  6.3 runs/s (23k runs/h) while runs were short. Later 60-85 actors on a 64-physical-core box: about 25k runs/h.
- Raw combat chunks: about 25 GB/h; zstd-compressed about 0.45 GB/h.
- Simulator: clone 27 us, card play 52 us, end turn 239 us on the laptop; 8.6k clone+action/s in one process,
  115k/s at 24 processes. The perf patch gave 1.31x simulator throughput with real networks. The server (9950X)
  measured 144k/s at 24 processes.
- GPU: the small combat network needs a fraction of an RTX 4080/5080; S3 is CPU-bound. The later 128-vCPU box with an
  RTX PRO 6000 96 GB ran 85 actors and 4 combat servers with the GPU at its 400 W cap only when the shadow trainer
  was also running.
