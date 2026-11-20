# Experiments: Ironclad A10 on game v0.113.1 (ic113)

This file records every experiment of the ic113 phase, oldest first. The phase trains **Ironclad at Ascension 10 on
game v0.113.1** (standard mode) in the simulator iRyougi/sts2-sim. It ran from 2026-11-18 to 2026-11-20. The
previous phase (Defect, v0.111.0) has its own record on branch `v0.111-defect` (`doc/experiments.md` there).

Each entry starts with its verdict. Then come the date, the question, the setup, the numbers and what was deployed.
The headline numbers and the learning curve are in [results.md](results.md). The lessons are in
[findings.md](findings.md). The system itself is described in [architecture.md](architecture.md),
[training.md](training.md) and [simulator.md](simulator.md). Live play has its own page,
[live-play.md](live-play.md).

## Terms used here

| Term | Meaning |
|---|---|
| A0 ... A10 | Ascension level. A10 is the hardest level used. At A10, act 3 ends with two bosses in a row. |
| Macro network | The network that makes every decision outside combat (map, cards, shop, rest, events). It also predicts V_run. It is an ensemble of 5 members; member 0 also carries the policy heads. |
| V_run | The macro network's predicted value of the current run. Its training target is `alpha * won + (1 - alpha) * acts_cleared / 3`, with `alpha` = 0.70 (0.85 from segment 29). With `alpha` near 1 it is the predicted win probability. |
| T_f | The fight score. `T_f(h, P) = V_run(post-fight state with HP h and potion belt P)`. The search in each fight maximises it. "Rel. T_f" is the fight's T_f divided by the table's maximum. |
| Combat network, search | The combat network guides a Gumbel MuZero-style search in every fight. "Search 64" means 64 simulations per move. Evaluations use 16 root candidates; self-play uses 4. |
| Chain `s3ic` | The self-play training run of this phase. |
| Segment | One block of self-play and training: 2 hours up to segment 27, 3 hours from segment 28. After each segment, its weights are evaluated on the panel below while the next segment trains. |
| Eval panel | 300 fixed A10 seeds, `Y100000`-`Y100299`, never used in training. Greedy macro, map argmax, search 64 on every move, the deployment rules (below). Segments 22-26 ran 304 seeds (`Y100000`-`Y100303`). |
| Deployment rules | Evaluation settings beyond the networks: card / smith / shop lookahead (`look_beta=20` etc.), the potion belt rule `potion_mix`, Fake Merchant purchases (`fake_merchant=1`), `win_bound=0.001` (from the segment-2 eval), `deck_route=1` and `lethal_backstop=1` (from the segment-7 eval). |
| Self-play A10 rate | Win rate of the chain's own A10 self-play runs in one segment (thousands of runs; excludes runs with perturbed fight starts). |
| Curriculum stage | The ascension mix of self-play (see experiment 8). Evaluation is always A10. |
| Carry | A trainer's replay window and optimizer state, saved to disk so the next segment can continue from it. |
| Stalemate | A fight that reaches round 151. The actor stops the run and records an error. |
| pp | Percentage points. |
| Paired | Both arms play the same seeds (or the same battle starts with the same chance seed). "+a / -b" counts the seeds only one arm won; McNemar's test gives the p-value. |
| CI | 95% confidence interval: Wilson for a single rate, normal approximation for a paired difference. |

## Overview

| # | Date (UTC) | Experiment | Verdict | Deployed |
|---|---|---|---|---|
| 1 | 2026-11-18 | Human data and baselines | Humans win 21.2% of Ironclad A10 runs on v0.111.0; v0.113 data is too thin to use alone | v0.111 human battle starts accepted for the combat warm-up |
| 2 | 2026-11-18 | Simulator port to v0.113.1 and parity checks | Every check passed; two open items closed on day 0 | New simulator, character and version as config, `tf_heal=1` |
| 3 | 2026-11-18 | Day-0 checks | No sign of a port bug (46.6% vs 44.8%); untrained Ironclad wins 0% at A10 | Combat warm-up required |
| 4 | 2026-11-18 | Combat warm-up on human battle starts | Harness 59.8% -> 85.1%; full runs A10 0% -> 3.1%, A0 5.1% -> 52.9% | Warmed combat network starts the chain |
| 5 | 2026-11-18 | Chain setup and launch | Smoke test passed; audit bugs fixed | Chain `s3ic` launched 12:19 |
| 6 | 2026-11-18 | Stalemates and `win_bound` | Agent bug, not simulator; fix removes stalemates | `win_bound=0.001` from the segment-2 eval / segment-3 self-play |
| 7 | 2026-11-18 to 2026-11-20 | Ascension curriculum | Stages 1 -> 2 -> 3 -> 4 by the written rule | Stage 4 since segment 29 |
| 8 | 2026-11-19 to 2026-11-20 | Interruptible server and server moves | Periodic carries and adoption saved 9 interrupted segments; 7 segment numbers produced nothing | `CARRY_EVERY=600`, carry backups every segment |
| 9 | 2026-11-19 | `deck_route` and `lethal_backstop` | Bit-identical when off; routes 0.67 random prompts per run when on | On from the segment-7 eval / segment-8 self-play; not paired-tested |
| 10 | 2026-11-19 | 84 actors vs 60 | 84 actors were slower (17.9k vs 19.8k runs/h) | Reverted to 60 |
| 11 | 2026-11-19 | T1: value calibration | After act-3 boss 1, V_run's HP slope is far too flat; member 0 is flatter still | Fed T2 |
| 12 | 2026-11-19 | T2: offline macro replay | Milestone heads pass; P2 and look-distill fail | `--milestones` |
| 13 | 2026-11-19 | `--milestones` in the chain | Switched on from segment 20 without a fork test; no controlled reading | On |
| 14 | 2026-11-19 | T3: boss search budget | 128 or 256 simulations do not beat 64 on act-3 bosses | Not deployed |
| 15 | 2026-11-19 to 2026-11-20 | Improvement plan review | 7 items done or tested, others open | See the table |
| 16 | 2026-11-20 | T4: members 1-4 for T_f, potion phase 2 | No arm beats the baseline | Not deployed |
| 17 | 2026-11-20 | Stage-4 companions: `alpha` 0.85, act-3 boss practice | Segment 29 evaluates at 50.7%; not attributable to one change | On |
| 18 | 2026-11-20 | Shadow combat trainer (d=256) | Trains beside the playing network since segment 29; no play test yet | Running, not playing |
| 19 | 2026-11-19 to 2026-11-20 | Live play in the real game | 3 finished runs, 3 wins (A0, A6, A10) | Live agent uses segment-29 weights |

---

## 1. Human data and baselines

**Verdict.** Humans win 21.2% of single-player standard Ironclad A10 runs on v0.111.0. On v0.113 there are only
134 such runs (about 12.7% won), too few to train on or to use as a firm yardstick. The project accepted v0.111 human
data for the warm start and relies on self-play for the rest.

- **Date.** 2026-11-18.
- **Question.** How much human Ironclad data is there for v0.113, and how strong are human players?
- **Setup.** Spire Codex exports: v0.113.0 (27,468 runs) and v0.113.1 (13,015 runs) in full, and the Ironclad runs
  of v0.111.0. Filter: single player, standard mode, unmodded, not abandoned. `src/ic113/fetch_human.sh`.

| Build | Ironclad A10 runs | Won | Reached act 3 |
|---|---|---|---|
| v0.111.0 | 24,048 | 21.2% | 37.7% |
| v0.113.0 | 75 | 13.3% | 32.0% |
| v0.113.1 | 59 | 11.9% | 18.6% |

Other v0.113 A10 characters: Fool 395 runs (about 22.8%), Regent 92 (about 39%), Defect 158 (about 10%). An earlier
summary table, used in the plan, gave 20.9% for v0.111 Ironclad A10; the 21.2% above is the later measurement with
the stated filter.

Battle starts (the state at the start of each human fight) were exported for the combat warm-up: v0.111.0 A10
401,968 train / 20,671 held-out starts (30,974 runs), plus A8/A9 and 5,244 / 291 v0.113 A8-A10 starts. Held-out
starts are 5% of runs by run hash.

**Decisions.** The v0.111 human starts are used for the combat warm-up, because the differences between the
versions are small and self-play is the main signal. The macro network gets **no** human refresh: it starts from the
converted Defect network (experiment 3).

## 2. Simulator port to v0.113.1 and parity checks

**Verdict.** The port passed every check that could fail. Two items stayed open and were closed on day 0
(experiment 3). The patched simulator is about 1.18x faster than the unpatched v0.113 release on the same runs.

- **Date.** 2026-11-18 (finished about 06:00 UTC).
- **Question.** Does our simulator patch work on iRyougi's v0.113.1.1 tag, and can the system switch to Ironclad
  without hidden-information leaks or broken decisions?
- **Setup.** New simulator clone at tag `v0.113.1.1`; the v0.111 clone is untouched. Our patch was rebased (three
  conflicts resolved by hand). New: a `character=` option, era 6 for v0.113 (a copy of the v0.111 era), v0.113
  vocabularies, an id-remapping converter for checkpoints (`src/ic113/convert_v113.py`), and T_f with post-combat
  heals (`tf_heal=1`: Burning Blood, Black Blood, Meat on the Bone, Chosen Cheese). All checks ran on CPU
  (`src/ic113/checks.sh`, `src/ic113/smoke.sh`).

| Check | Result |
|---|---|
| Upstream test suite with our patch | 5,040 tests: 5,021 pass, 19 skipped, 0 fail |
| Resampling clone (10,000 Defect states) | 0 / 9,969 digest mismatches; encoder identical 10,000 / 10,000 |
| Leak test (Ironclad 300 states, Defect 200) | Clones never saw the true draw order (clone == true reshuffle 0 / 8,700 and 0 / 5,300) |
| Root immutability (16 Ironclad A10 runs, 7,780 searches of 64 simulations) | 0 differences in the live game state before and after each search |
| Every Ironclad card in prompt states (90 cards x base / upgraded x 6 starts) | 1,456 card actions, 0 errors, 0 missing legal actions |
| Card token damage vs executed damage | 4,250 / 4,262 match; 12 mismatches (closed on day 0: Pen Nib, a check artefact) |
| Post-combat heal table vs the simulator | 1,065 / 1,065 won fights exact |
| Ironclad encoder round trip | 21 floors differed (closed on day 0: a test-recorder bug) |
| CPU smoke (16 full runs, 4 modes) | 0 errors |
| Speed | Patched v0.113 = 0.92x patched v0.111 per simulation (Defect); our patch = 1.18-1.19x the pristine v0.113 tag on the same runs |

**Deployed.** The new simulator, `character=ironclad`, era 6 and `tf_heal=1` for all ic113 work. Details of the port
are in [simulator.md](simulator.md).

## 3. Day-0 checks

**Verdict.** The converted Defect networks play Defect on v0.113 at their v0.111 level (46.6% vs 44.8%), so there is
no sign of a port bug. The same networks win **0%** of Ironclad A10 runs. A combat warm-up is required.

- **Date.** 2026-11-18, 05:00-06:10 UTC.
- **Questions.** (a) Is the port sound? (b) How strong is an untrained Ironclad? (c) Do human battle starts load in
  the v0.113 simulator? (d) What were the two open port items?
- **Setup.** Networks: the Defect layout-v3 chain at segment 5 (`s3v3_s005`), converted by id. Search 64 on every
  move, 16 root candidates, greedy macro with map argmax, lookahead 20. New panel `Y000000`-`Y001023`.
  `src/ic113/day0.sh`, `src/ic113/eval_ic113.sh`.

**(a) Port sanity (Defect A10, `tf_heal=0`).**

| | v0.113, panel Y 0-1023 | v0.111 reference, other panel |
|---|---|---|
| Won | **46.6%** (475 / 1,019; CI 43.6-49.7) | 44.8% (458 / 1,020; CI 41.8-47.9) |
| Mean floor | 42.4 | 43.4 |
| Act 1 / act 2 cleared | 88% / 76% | 89% / 81% |
| Boss / elite / hallway fights won | 87.6 / 98.1 / 99.8% | 87.1 / 98.2 / 99.8% |

The panels and segments differ, so this is unpaired. Act 2 is cleared 5 points less often on v0.113; this was not
investigated.

**(b) Ironclad step 0 (`character=ironclad`, `tf_heal=1`).**

| Arm | Runs | Won (CI) | Mean floor | Elite fights won |
|---|---|---|---|---|
| A10, converted Defect macro | 512 | **0.0%** (0-0.8) | 10.3 | 38.8% |
| A10, same + Defect potion table | 512 | 0.0% (0-0.7) | 10.3 | 38.0% |
| A0, converted Defect macro | 254 | **5.1%** (3.0-8.6) | 19.3 | 80.9% |
| A10, S1 human macro | 512 | 0.0% (0-0.8) | 9.5 | 30.1% |
| A0, S1 human macro | 251 | 2.4% (1.1-5.1) | 17.8 | 68.3% |

At A10, at least 74% of deaths were act-1 elites. The Defect-trained map policy routes into elites that the untrained
Ironclad combat play cannot win. The S1 human macro is no better. Seven runs ended in `SLIMES_WEAK` stalemates (about
4x Defect's rate); see experiment 6.

**(c) Battle-start import.** 99.4% of clean v0.111 A10 starts and 99.5% of clean v0.113 starts reach a playable first
decision; the failures are mod content. The export had not filtered non-standard runs: 40% of v0.113 A10 starts were
Dread mode. `export_starts.py --standard` now filters them.

**(d) Open port items, closed.** The 12 damage mismatches all came from two states with Pen Nib armed (the check
could not see its x2); after the fix 4,262 / 4,262 plays match. The encoder round-trip differences were a test-recorder
bug (a card stolen by Thieving Hopper and returned); after the fix Ironclad A10 1,754 / 1,754 and A0 1,349 / 1,349
floors match.

## 4. Combat warm-up on human battle starts

**Verdict.** 4.1 GPU hours of self-play from human Ironclad battle starts lifted the fight harness from 59.8% to
85.1% (+25.3 pp, CI +24.2 to +26.4). In full runs it took A10 from 0% to 3.1% and A0 from 5.1% to 52.9%. Almost all
of the gain came in the first hour. The warmed network became the chain's starting combat network.

- **Date.** 2026-11-18, 07:11-11:17 UTC.
- **Question.** Can S2-style combat self-play on human battle starts make the converted combat network usable for
  Ironclad?
- **Setup.**
  - Data: Ironclad A10 standard starts, v0.111.0 plus v0.113: 372,680 train and 18,984 held-out starts.
  - T_f prices: built with the converted Defect macro network in the v0.113 simulator.
  - Coverage mutation: 20% of training starts had 1-2 cards swapped for rarely seen Ironclad cards.
  - Init: the converted Defect combat network. Trainer: the Defect chain's live settings (lr 5e-5, weight decay 0.05
    on all parameters).
  - Actors: 52 self-play actors plus one held-out actor. Search 64/16, 4 root candidates, 3 fights per start.
  - Stop rule, written before the run: stop after hour K >= 3 once held-out losses and the harness both stop
    improving.
  - Scripts `src/ic113/warmup_*.sh`, `warmup_pool.py`, `warmup_report.py`.
- **Training.** 2,314,287 fights, 357,600 trainer steps. Hourly harness (1,000 fixed held-out elite and boss starts):
  61.5 / 85.6 / 86.8 / 85.0 / 87.3% for hours 0-4. The run stopped at a plateau after 4.1 h.

**Fight harness: all 7,152 held-out elite and boss starts, paired by start, search 64.**

| Room | Starts | Step 0 | Warmed | Difference (CI) | Humans survived |
|---|---|---|---|---|---|
| Elite | 4,868 | 70.2% | 92.3% | +22.1 (+20.8, +23.3) | 90.8% |
| Boss | 2,284 | 37.7% | 69.9% | +32.3 (+30.1, +34.4) | 78.7% |
| All | 7,152 | 59.8% | **85.1%** | **+25.3 (+24.2, +26.4)** | |

Per act: [results/ic113/warmup_harness.csv](../results/ic113/warmup_harness.csv). The warmed network wins human
elite starts about as often as the humans did, and boss starts 5-16 points less often.

**Full runs, paired by seed with day 0 (only the combat weights changed).**

| Arm | Step 0 | Warmed | Paired difference (CI) | Mean floor | Act 1 cleared |
|---|---|---|---|---|---|
| A10, Y0-511 | 0.0% (0 / 505) | **3.1%** (16 / 512; CI 1.9-5.0) | +3.2 (+1.6, +4.7) | 10.3 -> 19.9 | 2% -> 25% |
| A0, Y0-255 | 5.1% (13 / 254) | **52.9%** (135 / 255; CI 46.8-59.0) | +47.8 (+41.4, +54.3) | 19.3 -> 39.7 | 26% -> 83% |

At A10, act 1 was still the wall: only 25% of runs cleared it.

**Problems found.**
- 5 of 52 actors hung (cause not found; two grew to 33 GB / 6 GB). The chain supervisor got an actor memory guard.
- The mutation pool held 5 multiplayer-only cards (Blaze and Demonic Shield cannot be played solo), so about 6.5%
  of training starts got a card a single-player run never sees. The pool was fixed for later use; this run used
  the old pool.
- A first attempt ran out of RAM (each actor read the whole 1.3 GB table file); per-actor shards fixed it.

## 5. Chain setup and launch

**Verdict.** The chain `s3ic` launched at 2026-11-18 12:19 UTC after a CPU smoke test of every part (supervisor,
trainers, curriculum hook, potion values, evaluation, backup). An audit of the Defect chain found two agent bugs and
one trainer problem; all three were fixed for `s3ic`.

- **Setup.** The Defect layout-v3 chain's recipe with these changes:

| Setting | Value |
|---|---|
| Start weights | macro: converted Defect `s3v3_s005`; combat: the warm-up output (experiment 4) |
| Value target | `alpha` 0.70 with the acts term |
| Combat trainer | lr 5e-5; AdamW weight decay 0.05 **on weight matrices only** (none on embeddings, LayerNorm or biases); fresh Adam state |
| Actors | 60 x 32 runs in flight; 2 combat servers; search 64/16, 25% full searches, every boss move full, 4 root candidates |
| Self-play rules | `look_beta=20`; map head sampled (argmax only in evaluation); potion belt rule `potion_mix` (below) |
| Boss practice | 2 extra copies of each live boss fight from act 2 on, 25% full search |
| Segments | 2 h; evaluation beside the next segment |

- **Audit fixes.**
  1. The Fake Merchant event always left by the default option, so its relics were never bought (11.8% of runs
     entered it). With `fake_merchant=1` the shop head now buys from it; a forced-event check bought all 9 kinds.
  2. Combat weight decay hit every parameter, including embeddings and LayerNorm. Now it hits weight matrices only.
  3. The macro trainer's value members already decayed only weight matrices; unchanged.
- **Potion belt rule `potion_mix`, phase 1.** With a full belt and a potion on offer, each potion is scored by
  `V_mix(s, k) = Vbar_act(k) + 0.2 (dV(s, k) - Vbar_act(k))`. Here `dV` is the change in V_run from holding potion
  `k` alone versus an empty belt, and `Vbar_act` is its mean over 500 states per act from the previous segment. The
  lowest-scoring held potion is dropped if the offer scores higher. Phase 2 (a measured potion table) was built but
  left off; see experiment 16.
- **Smoke test.** Two 5-minute CPU segments: stage change, Vbar job, segment hand-off, evaluation and backup all
  worked; a re-derived `dV` matched the C# value exactly.

## 6. Stalemates and `win_bound`

**Verdict.** The stalemates were an agent bug, not a simulator bug. The fix, `win_bound=0.001`, caps every searched
value at the exact value of winning now. Replayed stalemate seeds went from 6 stalemates to 0, and self-play
stalemates fell from 269 to 1 per segment.

- **Date.** 2026-11-18 (evening). On from the segment-2 evaluation and segment-3 self-play.
- **Question.** Why did self-play stalemates grow from 149 to 251 to 269 per segment, mostly against `SLIMES_WEAK` on
  floor 2?
- **Finding.** The fight gets down to one small slime that one Defend blocks fully. The agent holds a Strike that would
  kill it, but plays Defend or ends the turn, round after round. A kill ends the fight and is scored exactly by T_f.
  Any other move leads to a leaf the network scores, and the network sits a few thousandths too high. Nothing charges
  for time. Min-max normalisation at the root turns the tiny gap into a hard policy target ("kill = 0.00"). After
  round 50 the search is off and the prior's argmax plays, so the fight never ends. Stalemated fights are dropped
  from training, so nothing pushes back.
- **Fix.** `v = min(v_net, T_f(HP now, potions held) / T_max - 0.001)` for every non-terminal node. Opt-in key, off by
  default (bit-identical when off). Replay test on stalemate seeds, `s3ic_s001` weights:

| Replay | Fights | Stalemates | Median rounds | Won fights over 20 rounds |
|---|---|---|---|---|
| Seeds 21-60, chain settings | 40 | 6 | 31.5 | 20 |
| Same seeds, `win_bound=0.001` | 40 | **0** | 6 | 0 |

Mean end HP was 58.6 with the fix and 59.1 without, the same within noise.

**In the chain.** Self-play stalemates: 269 (segment 2) -> 1 (segment 3) -> at most 3 per segment after that, all
`KNIGHTS_ELITE`. Evaluation stalemates: 4 (segments 0, 1) -> 1 (segment 2) -> 0 in every later evaluation except
segment 5 (1, `KNIGHTS_ELITE`). So the fix explains at most about 1 point of the step from segment 1 to segment 2
(27.4% -> 34.8%); the rest is training.

## 7. Ascension curriculum

**Verdict.** The chain moved through four stages by a rule written before training: stage 2 after segment 0, stage 3
after segment 3, stage 4 after segment 28. The rule never moves down.

- **Dates.** Stages 1-3 written 2026-11-18 before the launch; stage 4 added 2026-11-20.
- **Question.** Can early self-play at lower ascensions smooth the start of a new character?
- **Setup.** After each good segment, `src/ic113/chain_stats.py curriculum` reads that segment's self-play runs
  (stalemates and other errors count as losses) and picks the highest stage whose condition holds, never lower than
  the current one. Evaluation stays at A10.

| Stage | A10 share | Other levels | Move up when | Entered |
|---|---|---|---|---|
| 1 | 20% | A0-A9 uniform | A10 self-play >= 5%, or A0-A4 >= 50% | segment 0 |
| 2 | 30% | A5-A9 uniform | A10 self-play >= 12% | segment 1 (A0-A4 66.5% in segment 0) |
| 3 | 27% | A8 32%, A9 41% of all runs (the Defect chain's final mix) | A10 self-play >= 20% (added 2026-11-20) | segment 4 (A10 13.9% in segment 3) |
| 4 | 55% | A9 35%, A8 10% | (none) | segment 29 (A10 26.6% in segment 28) |

Self-play A10 win rate by segment: 3.9% (0), 8.1% (1), 11.4% (2), 13.9% (3), 16.7% (4), 20.6% (6), 16.7-20.0%
(segments 7-21, many of them short; segment 19 had only 87 A10 runs), 22.5-24.8% (22-27), 26.6% (28), 29.3% (29, the
first at stage 4). Full table: [results/s3ic/curriculum.csv](../results/s3ic/curriculum.csv).

The design was not tested against a chain without a curriculum, so whether it helped is unknown.

## 8. Interruptible server and server moves

**Verdict.** On an interruptible server, only what is on disk survives a stop. Periodic carries every 10 minutes and
"adoption" of an interrupted segment let 9 segments survive a stop and keep their training. 7 segment numbers produced
nothing usable: 3 were interrupted before their first carry, 1 was lost with its host, and 3 failed after a restore
because the restored code tree lacked a Python module. This is why the segment numbers have gaps (9-10, 13-17).

- **Dates.** Hardening and test 2026-11-19 01:12-02:10 UTC; the chain ran on rented servers that changed several times
  between 2026-11-19 03:11 and 2026-11-20 01:32 UTC.
- **Question.** Can the chain run on a rented server that can be stopped at any moment without a signal?
- **Changes** (`src/s3/s3_supervise.sh`, `s3_resume.sh`, `carry_backup.sh` and a restore script):
  - both trainers write their carry every `CARRY_EVERY` seconds (temp file, fsync, rename; about 1 s per save);
  - after a restart the supervisor validates the interrupted segment and, if both carries are complete, adopts it
    (weights, curriculum, evaluation) instead of discarding up to 2 hours;
  - a missed evaluation is re-run; downtime is added to the chain's end time;
  - after every good segment the carries go to backup storage (compressed, checksummed); a restore script rebuilds a
    chain on a fresh machine;
  - a resume job runs every minute.
- **Test.** A throwaway chain with 15-minute segments and four hard kills: mid-segment, during an evaluation, and a
  host loss with a restore into a fresh tree. Every path worked; the restore found one bug, which was fixed.
- **In use** (records in [results/s3ic/segments.csv](../results/s3ic/segments.csv)):

| Outcome | Segments |
|---|---|
| Recovered after an interruption | 11, 12 |
| Recovered after a planned stop (setting change or machine move) | 7, 8, 18, 19, 21, 22, 27 |
| Interrupted before the first carry; not adopted | 9 (281 runs), 10, 13 |
| Lost with the host | 14 |
| Failed after the restore (missing Python module; 0 runs) | 15, 16, 17 |
| Stopped early by the guard (an inference server ran out of GPU memory) but kept | 20 |

- **Moves.** After segment 6 the chain paused for 2 h 55 min to move to the interruptible server. At 60 actors that
  server played about 1.5x as many runs as the first one (19.8k runs/h). After the host loss in segment 14 the chain
  was restored on another machine and then moved several more times. The actor count followed each machine's CPU
  count (60, 36, 42, 32 and again 60). From segment 28 the segments are 3 hours long.
- **Run counter.** The run total counts run records that reached backup storage. At the restore it fell from 314,598
  to 294,723; runs of lost and discarded segments are not counted. So the 528,810 runs after segment 29 are a lower
  bound on the runs played.

## 9. `deck_route` and `lethal_backstop`

**Verdict.** About 0.67 run-level card prompts per run used to get a uniform random answer. `deck_route=1` sends them
to the existing smith, removal or card-reward heads. `lethal_backstop=1` plays a move that wins at once past round 50.
With both keys off the new build is byte-identical to the old one. Both are on in self-play and evaluation. A paired
play test was never run, so their effect on the win rate is unknown.

- **Date.** 2026-11-19, 06:35 UTC (build); on from the segment-7 evaluation and segment-8 self-play.
- **Question.** Which run-level card prompts fell through to a random answer, and can they reuse a trained head?
- **Setup.** Prompts were counted in segment 6 (38,748 runs). Rules by what happens to the card: an in-deck card that
  gets better (enchant, upgrade, copy) -> smith head; removed -> removal head; replaced (transform) -> removal head with
  `removal: false`; new cards -> card-reward head one pick at a time. Bundles stay random.

| Source (examples) | Prompts per 1k runs | Old | New |
|---|---|---|---|
| Self-Help Book (enchant) | 235.4 | random | smith head |
| Symbiote (enchant or transform) | 178.8 | random | smith / removal head |
| Spiraling Whirlpool (enchant) | 69.4 | random | smith head |
| Lord's Parasol (free removal) | 46.0 | random | removal head |
| Kifuda, Tri-Boomerang, Gnarled Hammer (enchant up to 3) | 57.5 | random | smith head x3 |
| Room Full of Cheese (2 new cards) | 27.6 | random | card head x2 |
| All newly routed | **669** (0.67 per run) | random | a head |

- **Tests (CPU, one actor, uniform networks).**

| Test | Runs | Result |
|---|---|---|
| Old build vs new build, keys off | 24 and 96 | Runs, fights, macro data and combat data byte-identical |
| `deck_route=1`, A0 | 2 x 240 | 0 errors; 7 sources routed |
| `max_rounds=5`, backstop off vs on | 48 each | 27 moves replaced; won fights 56 -> 65 |

Act-2/3 sources were not reached by uniform-policy runs; they were first exercised in the chain itself.

## 10. 84 actors vs 60

**Verdict.** More actors made the chain slower. With 84 actors and 3 combat servers the chain played 17.9k runs/h
against 19.8k with 60 actors and 2 servers. The chain went back to 60 / 2.

- **Date.** 2026-11-19, 06:39-07:12 UTC (segment 8).
- **Question.** Does a machine with more CPU cores gain throughput from more actors?
- **Result** (minutes 20-30 of a segment, no evaluation running beside it):

| | 60 actors, 2 combat servers | 84 actors, 3 combat servers |
|---|---|---|
| Runs per hour | 19.8k | 17.9k |
| Combat trainer steps/s | 22 | 18 |
| Actor time waiting on the inference bridge | 30% | 46% |

The GPU ran at its power cap, so more actors only queued for it.

## 11. T1: value calibration

**Verdict.** Confirmed. After the first act-3 boss at A10, the actual win rate spans 82.7 pp from the lowest to the
highest HP fifth, but V_run's predicted win rate spans only 54.5 pp. Low-HP states are overpriced by about 12 pp and
high-HP states underpriced by 14-16 pp. Member 0 of the ensemble is much flatter than members 1-4. At A10 in act 1 the
level is calibrated (within 1-1.5 pp) but the signal is weak.

T1-T4 are offline tests on a separate GPU, away from the training chain. Their code and raw results are in
`src/ic113/laptop/`.

- **Date.** 2026-11-19, 08:20 UTC.
- **Question.** Is V_run miscalibrated where it matters: after act-3 boss 1 by HP, and in act 1 at A10?
- **Setup.** `s3ic_s008` macro network (version 967). Data: (a) runs of segments 6-8 scored before training on them
  (n = 5,896 "after boss 1" rows); (b) the segment-8 training window (in-window); (c) 117 strictly later runs of
  segment 11 (held out). Predicted win = `(V - 0.20) / 0.80`, since 0.20 is the progress already banked at that point.

| HP fifth after act-3 boss 1 | Runs (a) | Actual win (CI) | Predicted | Gap |
|---|---|---|---|---|
| 0-20% | 620 | 4.5% (3.1-6.5) | 16.5% | +12.0 pp |
| 20-40% | 1,295 | 18.0% (16.0-20.2) | 30.6% | +12.6 pp |
| 40-60% | 1,640 | 47.2% (44.8-49.6) | 47.1% | -0.1 pp |
| 60-80% | 1,451 | 73.1% (70.8-75.3) | 59.1% | -14.0 pp |
| 80-100% | 890 | 87.2% (84.8-89.2) | 71.0% | -16.2 pp |

- **Member 0.** In-window, member 0 predicts 28.3% -> 66.0% across the fifths; members 1-4 predict 9.8% -> 79.9%.
  Held out (segment 11), the gaps are larger: +22.7 pp (CI +10.2 to +35.1) in the second fifth and -22.1 pp
  (-31.6 to -12.6) in the top fifth.
- **Overfitting.** On the training window the network uses 34.4% of the possible log-loss improvement over a 9-cell
  table; on held-out runs only 8.9%.

All tables: [results/laptop/t1_after_boss1.csv](../results/laptop/t1_after_boss1.csv).

## 12. T2: offline macro replay

**Verdict.** Milestone heads (arm M1, `--milestones`) passed every gate and were recommended for a fork test. P2
(arm M2) and look-distill (arm M3) failed and were not used.

- **Date.** 2026-11-19, 10:45 UTC.
- **Question.** Which macro-trainer changes are worth GPU time on the chain?
- **Setup.** Start: `s3ic_s005` macro network, fresh Adam and an empty value window in every arm. Five updates on 180
  chunks of segment 6 (5,760 runs). Held out: 120 strictly later chunks (3,840 runs, 150,732 floor rows). Live recipe
  flags in every arm. Script `src/ic113/laptop/t2_run_replay.sh`, `t2_eval.py`.
  - **M0** control (live recipe).
  - **M1** M0 + milestone heads: separate value heads for act 1 cleared, act 2 cleared, act-3 boss 1 beaten and win.
  - **M2** M1 + P2: member 0's policy also reads a stop-gradient moving average of members 1-4's encoders.
  - **M3** M0 + look-distill: card-reward rows trained toward the lookahead distribution by a KL loss instead of PPO.
  - Gates written before the run (held-out log loss -0.005 nats; residual AUC +0.015; member spread on one-card
    value differences must not rise by > 15%; PPO steps must not fall by > 30%; card agreement +3 pp).
- **Caveat.** All arms train on a small cold window, so all overfit compared with the starting network. The paired
  differences between arms are still clean.

| Metric (held out) | M0 | M1 | M2 | M3 |
|---|---|---|---|---|
| Value log loss (probability average) | 0.5779 | **0.5625** (-0.0154) | 0.5688 | 0.5782 |
| Value log loss (logit average, as T_f uses) | 0.7186 | **0.6475** (-0.0711) | 0.6680 | 0.7149 |
| Train-test gap | 0.3156 | **0.2541** (-0.0614) | 0.2609 | 0.3161 |
| Residual AUC within (floor, HP), all | 0.6478 | **0.6664** (+0.0186) | 0.6541 | 0.6508 |
| Member spread of one-card value difference (logit SD) | 0.440 | **0.304** (-30.8%) | 0.307 | 0.440 |
| Completed PPO minibatches (5 updates) | 1,163 | 1,066 | 731 (-31.4% vs M1) | 1,232 |
| Raw card choice agrees with own lookahead | 74.1% | 81.1% | 80.5% | 66.3% |

- **M2.** The P2 adapter caused no drift before PPO (lag KL 2.4e-5 to 3.9e-5), but it cut completed PPO minibatches
  by 31.4% (kill gate 30%) and gained only +0.07 pp in card agreement (gate +3 pp).
- **M3.** It confirmed that 75.6% of PPO's initial behaviour-lag KL comes from `look_beta=20` on card rewards. But
  forward KL toward a noisy lookahead target halved the card head's logit range (9.42 -> 4.35), doubled its entropy
  and cut agreement by 7.8-12.8 pp.

Per-arm metrics: [results/laptop/t2_arms.csv](../results/laptop/t2_arms.csv).

## 13. `--milestones` in the chain

**Verdict.** The chain switched milestone heads on at once, without the fork test that T2 recommended. There is no
controlled measurement of their effect on play.

- **Date.** 2026-11-19, about 11:05 UTC (segment 19 was cut after 6 minutes to apply it); first full use in segment 20.
- **What can be read.** The evaluation panel gave a mean of 40.3% over segments 5-19 (8 evaluations) and 41.5% over
  segments 20-28 (9 evaluations). Self-play A10 rose from 16.7-20.0% (segments 7-21) to 22.5-26.6% (segments
  22-28). Other things changed at the same time (machines, actor counts, segment lengths), so neither difference can
  be credited to the milestone heads.

## 14. T3: boss search budget

**Verdict.** Do not raise the search budget in boss fights. On 904 paired act-3 boss fights, 128 simulations won
+0.22 pp (CI -1.99 to +2.43) and 256 simulations +0.55 pp (-1.69 to +2.80) over 64. The final boss did not improve at
all. The act-3 boss bottleneck is not search depth.

- **Date.** 2026-11-19, 11:15 UTC.
- **Question.** Does more search in act-3 boss fights raise the final-boss win rate without retraining?
- **Setup.** `s3ic_s008` weights and the chain's exact evaluation settings. Step 1: 320 full A10 runs on fresh seeds
  `Z200000`-`Z200319` (37.8% won, CI 32.7-43.2; act-3 bosses caused 118 of 199 deaths, 59.3%) collected 452 act-3
  boss starts. Step 2: a fight harness replayed each start twice per arm with the same chance seeds; arms 64, 128 and
  256 simulations. 0 errors, 0 capped fights.

| Slice | Paired fights | 64 sims | 128 sims (diff, p) | 256 sims (diff, p) |
|---|---|---|---|---|
| All act-3 bosses | 904 | 76.2% | 76.4% (+0.22, 0.92) | 76.8% (+0.55, 0.70) |
| Act-3 boss 1 | 478 | 87.0% | 87.7% (+0.63, 0.75) | 88.9% (+1.88, 0.20) |
| Act-3 boss 2 (final) | 426 | 64.1% | 63.8% (-0.23, 1.00) | 63.1% (-0.94, 0.72) |

- **Cost.** 1.93x (128) and 3.84x (256) wall time per boss fight. All boss fights are 28.7% of an evaluation's
  simulations, so doubling them costs +28.7% of evaluation compute and quadrupling +86.1%.
- **The only measurable gain** is end HP: +1.36 HP (CI +0.37 to +2.34) at 128 simulations.

Per-boss and per-slot rows: [results/laptop/t3_boss_search.csv](../results/laptop/t3_boss_search.csv).

## 15. Improvement plan review

**Verdict.** A review of the chain at segment 6 listed four phases of possible changes. By segment 29, seven items
were deployed or tested. The phase-1 play test, marginal potion values, observation layout 6 and Ironclad deck counts
were not done.

- **Date.** Written 2026-11-19 (at segment 6); status as of 2026-11-20.
- **Rules it set.** Every change opt-in and off by default; never touch a running chain's binaries; CPU checks before
  GPU; screen on 300-512 paired seeds; confirm adoptions on a fresh 1,024-seed panel.

| Item | Status | Where |
|---|---|---|
| Route random run-level card prompts to trained heads | Done, deployed | Experiment 9 |
| Lethal backstop past round 50 | Done, deployed | Experiment 9 |
| Safer `win_bound` near in-fight heals (Meat on the Bone, Feed, Reaper) | Not built | |
| Paired play test of the phase-1 fixes | Not run | |
| Potion phase 2 (measured table), `VMIX_ASC=10` | Tested with an 80-start table: no gain, not deployed | Experiment 16 |
| Marginal potion value (`dV` given the rest of the belt) | Not built | |
| Look-distill (decouple lookahead from PPO) | Tested offline: failed | Experiment 12 (M3) |
| Milestone heads | Passed offline; deployed | Experiments 12, 13 |
| P2 encoder for member 0 | Tested offline: failed | Experiment 12 (M2) |
| Members 1-4 only for T_f | Tested: no gain | Experiment 16 |
| Ironclad deck-count features | Not built | |
| Stage 4 curriculum, higher `alpha` | Deployed in segment 29 | Experiments 7, 17 |
| Larger combat network from a clean start (shadow) | Running since segment 29 | Experiment 18 |
| Observation layout 6 (modified intent damage) | Not tested | |
| Act-3 boss practice | Deployed in segment 29 | Experiment 17 |
| More root candidates in self-play | Not tested | |
| Actor scaling | 84 actors tested: slower | Experiment 10 |

## 16. T4: members 1-4 for T_f, and potion phase 2

**Verdict.** Neither change helped. Pricing T_f with members 1-4 lost 2.2 pp (p = 0.53); the phase-2 potion table
gained 0.0 pp; both together +0.6 pp (p = 0.92). Neither was deployed.

- **Date.** 2026-11-20, before segment 29.
- **Questions.** (a) Does dropping the flat member 0 (T1) from the T_f table improve play? (b) Does a measured
  potion table (phase 2) beat the phase-1 belt rule?
- **Setup.**
  - Potion probe: 80 A10 final-boss starts from T3, each played with an empty belt and with each of the 46
    potions seen in Ironclad belts (3,760 paired fights, search 64). The empty belt won 56.2%. The value of a potion is
    T(k) = 100 x the paired difference in win rate. Each value has a standard error of about 5-6 points.
  - Play test: 320 paired A10 seeds `Z300000`-`Z300319`, `s3ic_s008` weights, the chain's evaluation settings,
    4 arms, 0 errors in 1,280 runs.
- **Potion table, ends of the ranking.** Entropic Brew +17.5, Fairy in a Bottle +15.0, Heart of Iron +13.8 ... Liquid
  Bronze -6.2, Ashwater -6.2 points ([results/laptop/t4_potion_probe.csv](../results/laptop/t4_potion_probe.csv)).

| Arm | Won (CI) | Paired diff | +/- | p | Act-3 boss 1 won | Final boss: HP share / potions entering | Final boss won |
|---|---|---|---|---|---|---|---|
| Base (5 members, phase 1) | 117 / 320 = 36.6% (31.5-42.0) | | | | 201 / 231 (87%) | 0.592 / 1.35 | 117 / 201 (58%) |
| Members 1-4 | 34.4% (29.4-39.7) | -2.2 | 43 / 50 | 0.534 | 196 / 232 (84%) | 0.610 / 1.51 | 110 / 196 (56%) |
| Phase 2, `VMIX_ASC=10` | 36.6% (31.5-42.0) | 0.0 | 27 / 27 | 1.000 | 189 / 230 (82%) | 0.616 / 1.37 | 117 / 189 (62%) |
| Both | 37.2% (32.1-42.6) | +0.6 | 47 / 45 | 0.917 | 187 / 229 (82%) | 0.616 / 1.66 | 119 / 187 (64%) |

- **Reading.** The new prices did what they were built to do: the agent entered the final boss with more HP and
  more potions and won it more often. But it saved them by playing act-3 boss 1 more cautiously and died there 12-14
  more times. The net effect was zero. The same pattern appeared in the Defect phase (finding 14 there): a better
  price does not help while the combat network cannot act on it.

## 17. Stage-4 companions: `alpha` 0.85 and act-3 boss practice

**Verdict.** Segment 29 changed four things at once and evaluated at **50.7%** (152 / 300, CI 45.0-56.3), the best
reading of the chain. On the same 300 seeds it beat segment 28 by +8.0 pp (CI +1.9 to +14.1, McNemar p = 0.014) and
segment 26, the previous best, by +3.7 pp (CI -3.1 to +10.4, p = 0.33). The gain cannot be split between the changes,
and the best of 23 readings on one panel is biased upward.

- **Date.** 2026-11-20; settings written at about 03:04 UTC, effective from segment 29 (04:40 UTC).
- **Changes in segment 29.**
  1. Curriculum stage 4: A10 55%, A9 35%, A8 10% (experiment 7).
  2. `alpha` 0.70 -> 0.85: the value target weights the win more and the acts term less.
  3. Boss practice moved to act 3: 3 copies of each act-3 boss fight, 50% full search (was 2 copies from act 2,
     25%).
  4. The shadow combat trainer started beside the playing network (experiment 18). It does not play, but it
     shares the GPU.
- **Throughput.** Segment 29 played 38,599 runs in 3 hours; segment 28 played 52,820 on the same machine.
- **Self-play.** A10 self-play 29.3% (5,511 / 18,832), the highest of the chain.
- **Paired readings** (computed from the per-seed evaluation records):

| Comparison on seeds Y100000-Y100299 | Earlier | Segment 29 | Diff (CI) | +/- | p |
|---|---|---|---|---|---|
| vs segment 26 | 141 (47.0%) | 152 (50.7%) | +3.7 (-3.1, +10.4) | 59 / 48 | 0.33 |
| vs segment 27 | 115 (38.3%) | 152 (50.7%) | +12.3 (+5.3, +19.3) | 78 / 41 | 0.001 |
| vs segment 28 | 128 (42.7%) | 152 (50.7%) | +8.0 (+1.9, +14.1) | 56 / 32 | 0.014 |

A fresh panel would be needed to confirm the level (see [results.md](results.md#what-the-numbers-do-not-show)).

## 18. Shadow combat trainer (d=256)

**Verdict.** A larger combat network (3.26M parameters, 4.6x the playing one) trains from a clean start on the live
data since segment 29. After one segment its training-batch losses already match or beat the playing network's. It
has not been tested in play and does not play yet.

- **Dates.** Built and benchmarked 2026-11-20; running from segment 29.
- **Question.** Is the combat network the bottleneck? The playing network's training metrics have been flat since
  segment 3 (policy KL 0.64-0.66, top-1 0.55-0.58, HP error 0.055-0.058 in full-length segments). Its pre-clip gradient norm sits at 60-130
  against a clip of 1.0. The working hypothesis is that the network inherited shrunken embedding and LayerNorm
  scales from the Defect phase, when weight decay hit every parameter. T3 showed that more search does not help.
- **Setup.** A shadow trainer reads the same data stream as the playing trainer but does not serve the actors. It
  starts from random weights, with the playing trainer's replay window preloaded.

| Network | Parameters | Train steps/s* | Inference states/s, batch 256* |
|---|---|---|---|
| Playing: d=128, 2 layers, 4 heads | 0.71M | 44.1 | 164.7k |
| d=192, 3 layers, 6 heads, ext | 1.91M | 16.9 | 75.7k |
| **Shadow: d=256, 3 layers, 8 heads, ext** | **3.26M** | **13.4** | **56.3k** |
| d=256, 4 layers, 8 heads, ext | 4.04M | 11.2 | 44.9k |

\* Benchmark on a 16 GB GPU, bf16, batch 512. "ext" adds the layout-v5 extra card, power and relic-counter features.

  Shadow trainer settings: weight decay 0.01 on matrices only, lr 3e-4 cosine to 5e-5 over 340,000 steps (about
  3 segments), replay window 4M decisions.

- **After segment 29** (same 140,000 steps and 17.8M decisions for both; training-batch metrics, not held out):

| | Policy KL | Top-1 | End-HP loss | Survival loss | HP error | Pre-clip gradient norm |
|---|---|---|---|---|---|---|
| Playing | 0.671 | 0.556 | 2.026 | 0.075 | 0.0571 | 83.5 |
| Shadow | 0.673 | 0.546 | 1.950 | 0.070 | 0.0536 | 5.1 |

- **Plan.** After 2-3 segments: a paired 300-seed screen, then a fresh 1,024-seed paired panel, and only then a
  switch of the playing network (`src/s3/switch_combat.sh`, `src/ic113/shadow_eval.sh`). Per-segment trainer
  numbers: [results/s3ic/combat_trainer.csv](../results/s3ic/combat_trainer.csv).

## 19. Live play in the real game

**Verdict.** The simulator-trained agent finished three runs in the real game and won all three: A0 and A6 with
segment-21 weights, A10 with segment-29 weights. Three runs say that the agent works end to end in the real game, but
they are far too few to estimate a win rate.

- **Dates.** 2026-11-19 (A0, A6) and 2026-11-20 (A10), UTC.
- **Setup.** A game mod (AgentBridge) exposes the game state and accepts actions. A controller (LiveAgent) imports
  the run from the game's save into the simulator and plays it with the evaluation settings: search 64 on every move,
  16 root candidates, greedy macro, lookahead 20, `potion_mix`, `deck_route`, `lethal_backstop`, `win_bound`. See
  [live-play.md](live-play.md).

| Weights | Ascension | Outcome | Floor | Last fight | HP after it |
|---|---|---|---|---|---|
| Segment 21 | A0 | Won | 48 | Queen (final boss) | 95 / 138 |
| Segment 21 | A6 | Won | 48 | Test Subject (final boss) | 82 / 97 |
| Segment 29 | A10 | Won | 49 | Queen (second act-3 boss) | 52 / 80 |

- **Harness fixes found by these runs.** A first A0 attempt stopped in its first fight because the inference bridge
  was called from the wrong thread (fixed). The ending scene after the final boss (the Architect) was not in the
  simulator and paused the controller (now clicked through). The Crystal Sphere minigame is played with random cell
  clicks. The game-over screen shows HP 0 after a win too; the controller now reports each run as won or lost from
  the ending scene.
- **Not counted.** A run in progress at the time of writing.
