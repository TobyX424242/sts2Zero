# Training

This document describes how the segment-29 agent was produced, and how to start a new self-play chain from the
release weights. The agent itself is described in [architecture.md](architecture.md). Numbers come from the project's
logs, the chain's settings files and its evaluation reports.

## 1. The four steps

| Step | Date | What | Result |
|---|---|---|---|
| 1. Port | 2026-11-18 | Simulator moved to v0.113.1.1; character and version made configurable; Defect networks converted to the new vocabularies | All port checks pass ([simulator.md](simulator.md)) |
| 2. Day-0 anchors | 2026-11-18 | Converted Defect networks evaluated on v0.113 (Defect and untrained Ironclad) | Port sane; Ironclad at A10 wins 0% before training |
| 3. Combat warm-up | 2026-11-18 | 4.1 GPU hours of fight self-play on human Ironclad battle starts | Held-out elite and boss fights won 59.8% → 85.1% |
| 4. Self-play chain `s3ic` | from 2026-11-18 | Full runs, both networks trained on the agent's own games | Segment 29: 50.7% A10 |

The macro network was **not** refreshed on human data in this phase. It started from the converted Defect network.
Human data entered only through the combat warm-up's battle starts.

## 2. Environment

- .NET SDK 9 (Release builds), Python 3.12 with PyTorch, NumPy and orjson (`scripts/setup.sh` creates `.venv` in
  the repository root; the scripts call `./.venv/bin/python -I`).
- Linux with an NVIDIA GPU for training and fast evaluation. CUDA MPS is optional (`MPS=1`).
- Run every script from the repository root (they find the root from their own path).
- The chain moved between several rented machines (30 to 184 CPU threads, one RTX 5090 or RTX PRO 6000 GPU each).
  Segment 29 ran on a 64-thread CPU with one RTX 5090 (32 GB).

Build (also builds S0, S2 and the simulator):

```bash
dotnet build src/s3/S3/S3.csproj -c Release -m:8
```

## 3. Starting point: the converted Defect agent

The previous phase ended with a Defect agent on v0.111.0. Its best chain, `s3v3` (macro layout v3), scored 45.0% at A10
at segment 5. Both networks of `s3v3_s005` were converted to the v0.113 vocabularies by `src/ic113/convert_v113.py`:

- **Ids are remapped by name**, because the vocabularies are sorted id lists and new content shifts every index.
- **Combat network:** rows copied by id. New cards, and the 90 Ironclad cards that the Defect network never trained,
  start at the mean embedding of trained cards of the same type (attack, skill, power, curse, status). Other new ids
  start at their category's mean.
- **Macro network:** new ids start as the `<unk>` row; the content properties tell them apart. The v0.113 era token
  (row 6) starts as a copy of the v0.111 row.
- **Not converted:** replay windows, optimizer states and chunks of the Defect chains (their indices are stale).

Check (verified): on 794 stored combat states and 1,480 macro rows, converted and original networks give identical
outputs (largest difference 0).

## 4. Day-0 anchors

| Arm | Seeds | Result |
|---|---|---|
| Port sanity: converted Defect networks, Defect A10 on v0.113, no post-combat heals in T_f | `Y000000`–`Y001023` | 46.6% (475 / 1,019; CI 43.6–49.7) vs 44.8% and 45.0% for the same agent on v0.111 |
| Same networks, Ironclad A10 | 512 runs | 0.0%, mean floor 10.3 (deaths mostly act-1 elites) |
| Same networks, Ironclad A0 | 254 runs | 5.1% |

The Defect level carried over, so the port showed no sign of a bug. The Ironclad needed training before anything else.

## 5. Combat warm-up

Goal: teach the combat network Ironclad cards before full-run self-play. Scripts: `src/ic113/warmup_*.sh`,
`warmup_pool.py`, `warmup_report.py`; the fight player is `S2.dll play`.

**Data.** Ironclad A10 battle starts exported from Spire Codex runs (`src/s2/export_starts.py --standard --macro-ctx`):
v0.111.0 368,357 train / 18,725 held out, v0.113.0 1,307 / 74, v0.113.1 798 / 23; in all 372,680 / 18,984. The held-out
set is 5% of runs, split by run hash. `--standard` drops modded runs, non-standard modes and modifiers (40% of v0.113 A10
starts were Dread mode).

**Prices.** Every start got a T_f table from the converted macro network, built by the same code as in a live run
(`S3 tables`, `src/s3/S3/HumanTables.cs`). Run facts a battle start lacks (recent fights, act counters, boss) come from
the human run record. The map is missing: the injected run has none.

**Coverage.** 20% of training starts had 1–2 deck cards swapped for rarely held Ironclad cards, upgraded with
chance 0.3. One mistake: 5 of the swap cards are multiplayer-only in the simulator, so about 6.5% of all training starts
held a card a single-player run never sees. The pool was fixed afterwards (`multiplayer_only` in
`src/s3/data/card_info_v113.json`) but the run used the old one.

**Training.** 52 self-play actors, 3 fights per loaded start, search 64 / 16 with 25% full searches, 4 root
candidates, round cap 50. Trainer: lr 5e-5 constant, weight decay 0.05 (on all parameters, the old setting), value
dropout 0.3, surprise weighting 0.5, EMA 0.999, batch 512, replay window 1M decisions. 4.1 GPU hours, 2,314,287 fights,
357,600 trainer steps.

**Stop rule** (written before the run): stop after hour 3 or later when the held-out losses move less than 1% and the
fight win rate less than 2 points from the previous hour; at most 6 hours. The run stopped at 4.1 hours as a plateau.
Hourly win rate on 1,000 held-out elite and boss starts: 61.5, 85.6, 86.8, 85.0, 87.3%. Almost all of the gain came in
the first hour.

**Result** (verified, paired by start, search 64):

| Held-out fights | Step 0 | Warmed | Humans (same starts) |
|---|---|---|---|
| Elites (4,868) | 70.2% | 92.3% | 90.8% |
| Bosses (2,284) | 37.7% | 69.9% | 78.7% |
| All (7,152) | 59.8% | 85.1% (+25.3, CI +24.2 to +26.4) | |
| Full runs, Ironclad A10 (512 seeds) | 0.0% | 3.1% | |
| Full runs, Ironclad A0 (256 seeds) | 5.1% | 52.9% | |

Issue found: 5 of 52 actors hung during the run (cause not found). The chain supervisor got a memory guard for this
(`ACTOR_MEM_GB`).

## 6. The s3ic self-play chain

### 6.1 Terms

| Term | Meaning |
|---|---|
| chain | A long training run made of segments. Its directory is `src/s3/runs/s3ic/`; segment `K` lives in `src/s3/runs/s3ic_sKKK/`. |
| segment | One training block of fixed length (2 h, from segment 28 on 3 h). It starts from the last good segment's weights and carries, and ends with saved weights, carries and an evaluation. |
| supervisor | `src/s3/s3_supervise.sh`: starts each segment, guards it, validates its output, runs the hook and the evaluation, uploads backups |
| guard | A check every minute: trainers and servers alive, enough actors alive, new runs arriving, disk below 90%, actor memory below `ACTOR_MEM_GB` |
| hook | `src/ic113/s3ic_hook.sh`: chain-specific work between segments (curriculum, potion prices, staged builds, evaluation report) |
| carry | `carry_combat.bin` and `carry_macro.pt` (plus `carry_shadow.bin`): replay windows and optimizer states, so training continues across segments |
| last good | The newest segment that passed validation; the next segment starts from it |
| curriculum stage | Which ascensions self-play uses; set by the hook from self-play win rates |

### 6.2 What runs inside a segment

`src/s3/run_s3.sh` starts, for the segment's length:

- **60 actors** (`S3.dll play mode=selfplay`), each playing 32 runs at once on one thread;
- **2 combat inference servers** and **1 macro inference server** (`src/s3/py/s3_server.py`);
- the **combat trainer** (`src/s2/py/train_combat.py`), the **macro trainer** (`src/s3/py/train_macro_s3.py`) and,
  since segment 29, the **shadow combat trainer**.

Actors write run logs, fight logs, combat chunks (`*.s2d`) and macro chunks (`*.s3r`). Trainers read the chunks,
publish versioned weights, and the servers reload them. A run started under one weight version can finish under the
next.

### 6.3 The cycle between segments

1. The segment ends at its time limit. Runs in flight finish.
2. The supervisor validates it: at least 100 runs, both weight files load and are finite, 5 macro members, and (with
   `CARRY=1`) both carries complete. A failed segment does not advance the chain; three failures in a row stop it.
3. The **hook** runs for the next segment:
   - swaps in a staged simulator/agent build (`src/s3/S3/bin_next`) if one is waiting, so code changes land only
     between segments;
   - updates the **curriculum** from the finished segment's self-play runs (§6.4);
   - computes the next **potion prices** (`Vbar`, §6.5);
   - writes `segment.env`: ascension mix, `asc_weights` and `potion_mix=FILE` for the actors, and the evaluation
     arguments.
4. The next segment starts. Beside it, the finished segment is **evaluated** on the fixed panel (§7), and its chunks,
   logs and weights are backed up.

### 6.4 Ascension curriculum

Self-play does not play only A10: early on, A10 runs gave too little experience of acts 2 and 3. The stage moves up
after a segment whose self-play win rates meet the condition (each rate over at least 300 runs; perturbed runs left out;
errors count as losses). It never moves down. Evaluation is always at A10.

| Stage | Share of runs at A10 (`asc_mix`) | Levels below A10 | Moves up when | Reached |
|---|---|---|---|---|
| 1 | 20% | A0–A9 uniform | A10 ≥ 5% or A0–A4 ≥ 50% | chain start |
| 2 | 30% | A5–A9 uniform | A10 ≥ 12% | after segment 0 (A0–A4 66.5%) |
| 3 | 27% | A8 32%, A9 41% of all runs | A10 ≥ 20% (threshold added 2026-11-20) | after segment 3 (A10 13.9%) |
| 4 | 55% | A8 10%, A9 35% of all runs | — | after segment 28 (A10 26.6%) |

Code: `src/ic113/chain_stats.py curriculum` (thresholds `CURRIC_*` in `config.env`). Each reading is a line in the
chain's `curriculum.log`.

### 6.5 potion_mix prices per segment

Before each segment the hook runs `src/ic113/potion_mix.py vbar` on the finished segment's macro carry: up to 500
states per act from the newest 20,000 runs (all ascensions), priced by the macro file the next segment starts from. The
output `vmix/vbar_sNNN.json` is used by the next segment's actors and by the finished segment's evaluation. The first
segment had no self-play states, so its file had no means and potions were ranked by the state term alone. A failed job
keeps the previous file. See [architecture.md §7](architecture.md#7-potion_mix-which-potion-to-keep).

### 6.6 Settings at segment 29

The full files are [configs/s3ic_config.env](../configs/s3ic_config.env) (read by the supervisor) and
[configs/s3ic_extra.env](../configs/s3ic_extra.env) (read by `run_s3.sh` at every segment start, so changes apply at
the next segment without a restart).

**Chain**

| Setting | Value |
|---|---|
| Actors × runs in flight (`N`, `CONC`) | 60 × 32 |
| Segment length (`SEG`) | 10,800 s (7,200 s up to segment 27) |
| Combat servers (`CSERVERS`) | 2, CUDA graphs, two batches in flight (`--overlap-min 48`) |
| Macro server | CUDA graphs (`--mgraphs 1`) |
| GPU sharing | CUDA MPS |
| Carries | on (`CARRY=1`), written every 600 s (`CARRY_EVERY`), backed up after every good segment (`CARRY_BACKUP_EVERY=1`) |
| Actor memory guard | 8 GB |
| Perturbed runs | 10% (run_s3.sh default) |

**Search and actors** (`EXTRA_ACTOR_ARGS`; meanings in [architecture.md §8](architecture.md#8-actor-keys-in-use))

`character=ironclad era=6 tf_heal=1 hall_sims=0.5 boss_pfull=1 practice_boss=3 practice_min_act=3 practice_pfull=0.5
candidates=4 look_beta=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 deck_route=1
lethal_backstop=1`, plus `potion_mix=FILE` and `asc_weights` from the hook. Search budget: 64 simulations on 25% of
moves, 16 on the rest (run_s3.sh defaults).

**Macro trainer** (`train_macro_s3.py`)

| Setting | Value |
|---|---|
| Update interval | every 2,560 runs (`RUNS_PER_ITER`) |
| Outcome target | `0.85 * won + 0.15 * acts_cleared / 3` (`--alpha 0.85`; 0.70 up to segment 28) |
| Milestone heads | trained, their blend is V_run (`--milestones`, from segment 20) |
| Policy (member 0) | PPO: clip 0.2, GAE γ = 1, λ = 0.95, entropy 0.01, target KL 0.03 (stop at 1.5 × target), advantages divided by max(std, 0.02), lr 1e-4 |
| KL to the human-data policy | off (`KL_RUNS=1`); `--bc` names the converted S1 file, which is loaded but has weight 0 |
| Value (all members) | cross-entropy to the outcome on rows from the last 50,000 runs (`--window-runs`), 16% of each run's rows kept (`--value-row-share 0.16`), 150 steps per member per update, dropout and weight decay 0.1 on members 1–4, probability averaging |
| Warm-up | first 600 s of a segment: value only (`--policy-warmup-s 600`) |
| Memory | `--eval-batch 1024` (no-grad forwards in chunks; freed GPU memory for the shadow trainer) |

Member 0 trains only its value heads in the value step (its encoder serves the policy). An earlier test showed that a
value gradient into the shared encoder pushed the policy far outside PPO's trust region.

**Combat trainer** (`train_combat.py`)

| Setting | Value |
|---|---|
| Policy loss | cross-entropy to the search's improved policy, on fully searched moves |
| Value loss | 41 end-HP bins with HL-Gauss targets (death = bin 0), 6 potion-kept logits |
| Learning rate | 5e-5 |
| Weight decay | AdamW 0.05 on weight matrices only; none on embeddings, LayerNorm and biases (`--wd-groups matrices`) |
| Replay window | 6M decisions (`COMBAT_WINDOW`), 1M carried to the next segment (`--carry-keep`) |
| Other | surprise weighting 0.5, EMA of weights 0.999, value dropout 0.3 |
| Optimizer state at chain start | fresh (the warm-up's Adam state was not reused) |

**Shadow combat trainer** (since segment 29; [architecture.md §9](architecture.md#9-shadow-combat-trainer))

`--d 256 --layers 3 --config heads=8 --config ext=1 --config vdrop=0.3 --config dropout=0.1 --wd-groups matrices
--wd 0.01 --lr 3e-4 --lr-schedule cosine --lr-min 5e-5 --total-steps 340000 --lr-continue --surprise 0.5 --ema 0.999
--window 4000000`, 1M entries carried, replay window preloaded from the playing trainer's carry at its first segment.

### 6.7 Changes during the chain

The recipe at launch was the Defect `s3v3` chain's, with the ic113 changes (character, era, `tf_heal`, potion_mix,
`fake_merchant`, combat weight decay on matrices only). Later changes:

| From segment | Date | Change | Why |
|---|---|---|---|
| 0 | 2026-11-18 | Launch: stage 1, 2 h segments, practice 2 copies of each boss from act 2 (25% full search), `--alpha 0.70` | — |
| 3 (eval of 2) | 2026-11-18 | `win_bound=0.001` | Stalemates: 269 per segment, mostly one weak slime left on floor 2 |
| 8 (eval of 7) | 2026-11-19 | `deck_route=1`, `lethal_backstop=1` | About 0.67 deck prompts per run were answered at random; residual Knights stalemates |
| 20 | 2026-11-19 | `--milestones` | Offline replay test: better held-out value loss, more completed PPO steps |
| 28 | 2026-11-20 | 3 h segments | — |
| 29 | 2026-11-20 | Stage 4; `--alpha 0.85`; practice 3 copies of act-3 bosses only, 50% full search; shadow trainer from scratch; `--eval-batch 1024` | Act-3 bosses cause most deaths; A10 self-play reached 26.6% |

The evidence behind each change is in [experiments.md](experiments.md).

**Gaps in the curve.** Segments 9, 10 and 13–17 have no evaluation. They were cut short by restarts and moves to
other machines; one (segment 14) was lost with its host. Segment 20 was stopped early when an inference server died.
After a restore from backup the chain counted only the run records that had reached the backup, so its run count fell
from 314,598 to 294,723 and then rose again. The 528,810 runs at segment 29 are therefore a lower bound of the runs
actually played.

### 6.8 Interruption safety

The chain ran partly on interruptible rented machines, which can stop at any moment without a signal. What protects
it (all in `src/s3/s3_supervise.sh` and `src/s3/s3_resume.sh`):

| Risk | Protection |
|---|---|
| Carries written only at a segment's end | Both trainers also write their carry every `CARRY_EVERY` seconds (temp file, fsync, rename; about 1 s per save) |
| A restart threw away the interrupted segment | **Adoption**: at restart, the interrupted segment is validated like a finished one; if both carries are complete it becomes last good and the chain continues from it. Otherwise the chain continues from the previous last good segment, losing at most `CARRY_EVERY` seconds. |
| The evaluation beside the next segment died | Missing evaluations are run again at restart |
| The time budget ran on during downtime | A heartbeat file; `s3_resume.sh` moves the end time by the downtime |
| Supervisor not restarted | `s3_resume.sh` from cron every minute, with a lock |
| Host lost | `CARRY_BACKUP_EVERY`: carries uploaded after each good segment (zstd, 90 MB parts, SHA-256 manifest). `src/s3/restore_from_nextcloud.sh` rebuilds the chain in a fresh tree. |

These paths were tested with deliberate kills (mid-segment, during an evaluation, and a restore into a fresh tree);
one restore bug was found and fixed.

### 6.9 Backups

The supervisor uploads to an rclone remote named by `NC_ROOT`. Per segment: 1 in 4 training chunks (`KEEP_EVERY=4`),
all fight logs and mid-segment checkpoints with the segment's meta copy, then local pruning once the upload is confirmed
(`PRUNE_KEEP=0`). Run records, logs and segment weights stay on disk. The combat carry compresses about 44× (4.15 GB →
about 95 MB), the macro carry about 15×.

### 6.10 Throughput

| Measurement | Value |
|---|---|
| 60 actors on a 184-thread CPU + RTX PRO 6000 | 19,800 runs per hour (minutes 20–30 of a segment) |
| 84 actors, 3 combat servers, same machine | 17,900 runs per hour: slower, GPU at its power cap; reverted |
| Segment 28 (3 h, stage 3) | 52,820 runs |
| Segment 29 (3 h, stage 4, 3 practice copies, shadow trainer) | 38,599 runs |

## 7. Evaluation protocol

After every good segment, beside the next one:

| Item | Value |
|---|---|
| Seeds | `Y100000`–`Y100299` (300 runs; segments 22–26 played 304 because of how runs were split over actors) |
| Ascension | A10 only |
| Macro | greedy: argmax on the human-learned decision types and the map (`greedy_map=1`), events sampled |
| Search | 64 simulations on every move, 16 root candidates, no Gumbel noise |
| Rules | as in self-play, plus `look_smith=20 look_shop=20`; `potion_mix` with the next segment's `Vbar` file |
| Arguments | `character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20 look_smith=20 look_shop=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 deck_route=1 lethal_backstop=1 potion_mix=FILE` |
| Report | `src/ic113/chain_stats.py eval`: win rate with Wilson CI, act funnel (runs entering and winning each boss), HP share and potions entering each boss, stalemates, deaths by encounter, potion_mix decisions per run |

Outputs: the chain's `curve.txt` (one line per segment) and `eval_report.txt`; copies are in
[results/s3ic/](../results/s3ic/).

The 300-seed panel has a 95% interval of about ±5.5 points, so neighbouring segments often cannot be told apart.
Self-play A10 win rate over 5,000–19,000 runs per segment is a finer signal of progress. A claimed gain from a change is
confirmed on fresh paired seeds (300–1,024), not on this panel.

## 8. Launching a new chain from the release weights

This path follows the scripts as they are, but **was not run end to end from the release files** (status: not
verified). It starts a new chain with empty replay windows and fresh optimizer states: training continues from the
segment-29 weights, but it is not an exact continuation of `s3ic` (the carries are not released).

1. Set up the environment and build (§2). Download the release (`scripts/fetch_weights.sh`) into a folder, here
   `$WEIGHTS`, and check the sums ([checkpoints.md](checkpoints.md)).
2. Create the chain directory and copy the settings. Adjust `NC_ROOT` to your rclone remote, and `N`, `EVAL_N`,
   `CSERVERS` to your machine.

   ```bash
   mkdir -p src/s3/runs/s3ic/vmix
   cp configs/s3ic_config.env src/s3/runs/s3ic/config.env
   cp configs/s3ic_extra.env  src/s3/runs/s3ic/extra.env
   ```

3. Start at the segment-29 curriculum stage and potion prices. Without these two files the hook starts at stage 1 and
   with an empty price file.

   ```bash
   echo "4 asc_mix 0.55 asc_weights 0,0,0,0,0,0,0,0,1,3.5" > src/s3/runs/s3ic/curriculum_stage
   cp $WEIGHTS/ic113_s029_vbar.json src/s3/runs/s3ic/vmix/vbar_s029.json
   ```

   The hook keeps the newest earlier `vbar_s*.json` when the finished segment has no carry, which is the case at a
   chain start.

4. Launch. `COMBAT0` and `MACRO0` are the release files. `BC0` is the macro trainer's `--bc` reference; its weight is 0
   in this recipe, so the release macro file can stand in for the converted S1 file (assumption, not tested).

   ```bash
   COMBAT0=$WEIGHTS/ic113_s029_weights.pt MACRO0=$WEIGHTS/ic113_s029_macro.pt BC0=$WEIGHTS/ic113_s029_macro.pt \
     src/ic113/s3ic_launch.sh 168
   ```

   The script copies the three files to `src/s3/runs/s3ic_init/`, checks that the combat network uses observation
   layout 5 and the macro file has 5 members, keeps the `config.env` and `extra.env` you placed, and starts the
   supervisor for 168 hours. `DRY=1` writes the files and starts nothing.

5. Add the resume line to cron (the script prints it but does not install it). On an interruptible machine, run it
   every minute:

   ```text
   * * * * * <repo>/src/s3/s3_resume.sh >> <repo>/s3_resume.log 2>&1
   ```

6. Watch `src/s3/runs/s3ic/status.txt`, `supervisor.log`, `curve.txt` and `curriculum.log`. Pause before the next
   segment with `touch src/s3/runs/s3ic/pause`; resume with `src/s3/resume_chain.sh s3ic`.

Notes:

- The first segment's shadow trainer starts from random weights; with no carry, it has nothing to preload.
- `src/ic113/s3ic_setup.sh` builds a separate server tree from an uploaded archive. It is specific to the original
  servers and is not needed for the steps above.
- The watch panel (`src/s3/panel/`) reads a local copy of the chain's logs synced from the backup remote.

## 9. Open issues

- PPO and the card lookahead: self-play records the lookahead-shifted card probability as the behaviour policy, so
  the policy ratio sees a large lag (behaviour-lag KL about 0.09 against a target of 0.03; 75.6% of it from card
  rewards) and most PPO updates stop early. A distillation fix was tested offline and rejected
  ([experiments.md](experiments.md)).
- The playing combat network's training statistics have been flat since about segment 3; the shadow trainer is the
  current answer (§6.6).
- The A10 value after the first act-3 boss was too flat in HP when measured at segment 8 (before the milestone
  heads): on fresh runs the actual win rate spanned 82.7 points across HP fifths, the prediction only 54.5. No re-measurement
  with the milestone heads is recorded.
- Actor hangs (seen in the warm-up, about one per 40 actor-hours) are contained by the memory guard; the cause is not
  known.
