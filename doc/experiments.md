# Experiments

Chronological record of what was tried, with the result that mattered. The original project log has thousands of
lines of server metadata; this document keeps the experiment, the comparison, the number, and the decision. Raw
per-segment logs are not redistributed.

Conventions: win rates are Defect A10 on fixed evaluation seeds unless stated otherwise; "rel. T_f" is the mean
T_f at fight end divided by the fight's table maximum; paired intervals are 95% unless stated. "Owner" marks a
decision made by the project owner rather than a result.

## 0. Early POC (one paragraph)

Before S1, a 1.0M-parameter set transformer was trained on a balanced 100k-run sample of the Spire Codex dump. It
reached test AUC 0.775 (0.745 within a floor) and transferred to a strong agent's states in ranking but not in scale
(AUC 0.670 within floor, predicted 39% wins where the true rate was 74%). The review found three defects: floor
sampling biased early floors low, the fight table used the pre-fight state (overpricing HP at bosses), and
fight-outcome inputs inflated the apparent HP price. Those findings shaped S1, and the POC was then replaced. Its
code, probes and checkpoints are intentionally not part of this repository.

## 1. S0: simulator extensions and M1 skeleton (2026-11-10 to 11)

### S0 experiments

| Experiment | Result |
|---|---|
| Exact clone vs original after one real action | 10,548 comparisons, 0 mismatches |
| Two exact clones under the same 5 random actions | 46,791 steps, 0 mismatches |
| Resampled clone: observable fingerprint / encoder | identical in 10,000 states; draw order differs where it should |
| Resampled clone: next draw vs posterior | 20.57% vs 21.12% expected (z = -1.63); negative control 96.87% |
| Clone cost | 25.6 us exact, 1.0 us resample, ~110 us action; 7.2k/s vs 7.4k/s interleaved |
| Card choices in clones | 0 errors with random choices (2,174 prompts); suspend/replay and pause/resolve reproduce policy exactly |
| Battle-start import | 998/1,000 human starts imported and played to the end; starting enemies matched 924, subset 74 (summoners); 2 failures were a modded relic |
| iRyougi test suite with patch | 1,774 passed, 0 failed (pile/clone/shuffle/draw/innate/transplant) |

S0 also built the legal-action space, the observation encoder/vocabulary, the shared-memory bridge, and the M1
rollout search.

### M1: T_f vs HP scoring on 998 human battle starts

| Arm (1,000 simulations) | Fights won | Mean T_f at end | Potions used |
|---|---|---|---|
| T_f-scored | 90.0% | 0.297 | 74.6% |
| HP-scored | 90.5% | 0.288 | 97.5% |

No win difference (p = 0.49), but T_f used potions as if they had a future price and scored higher on T_f. At 200
simulations the difference was noise. Speed with rollout leaves: ~800 simulations/s per process, ~19k/s on a 7950X.
The result was "the link changes behaviour; whether it wins more runs needs a learned fighter and full runs" (G3).

## 2. S1: macro network warm start (2026-11-10 to 11)

### Data and scaling curve

Full natural-frequency dataset: 1.2M clean runs, 965,021 training runs. Scaling runs at 25k / 50k / 100k / 300k / all:

| Model | Test log loss (all) | Within-floor AUC (all) | Test log loss at 100k | Within-floor AUC at 100k |
|---|---|---|---|---|
| No attention | **0.4604** | **0.8054** | **0.4800** | **0.7864** |
| Set transformer | 0.4628 | 0.8036 | 0.4841 | 0.7830 |

The no-attention model won or tied at every size, was better calibrated, and trained about twice as fast. It became
S1. Uniform floor sampling removed the early-floor bias; state-only inputs cost nothing measurable.

### G1 (5-member ensemble, Defect A10 v0.111.0 target cell)

| Requirement | Result |
|---|---|
| Calibration | ECE 0.0038 in floor bands (target <= 0.02); worst band gap 0.20 pt |
| HP monotone | 0 violations in 39,850 +10 HP probes |
| Accuracy | test log loss 0.4655, within-floor AUC 0.8002, both better than the 100k baselines |
| T_f builder | 2,000 fights; batched vs direct max difference 0.007 pt; final boss = 1; 0 HP steps decreasing |
| +1 potion dominance | 2.82% wrong by > 0.5 pt (target <= 1%) — **accepted by the owner**, because T_f fixes potion prices in S3 |
| Policy vs pick-rate | card reward 53.3% vs 40.6%; ancient 82.6%, removal 73.2%, rest 67.2%, relic 54.9%, smith 47.0%, shop 34.8% |

The +1-potion violation was the only failed check. It was accepted rather than trained away because the important
prices are computed by T_f over explicit potion subsets, and S3 was expected to fix them with its own data.

## 3. S2: combat network warm start (2026-11-10 to 11)

### Preparation

- Layout v2 added the cards a prompt answer picks; v3 added the Context token and per-potion marginal T_f.
- Table diagnostics on 400 held-out starts: the five S1 members agreed on the HP price (full vs half) to 9.8 +/- 1.0
  points and on the whole belt to 5.2 +/- 0.85 points, with the same sign in 99-100% of starts; free-disposal
  violations were 0.18% of entries (worst -2.5 points). The tables were a usable signal; the weak part was how the
  network received them.
- D13: the outcome head (end-HP distribution + potion-kept probabilities, priced by T_f at every leaf) was adopted;
  the scalar head stayed as the A20 comparison.

### First learning run (1 hour, from scratch, 30 actors)

Self-play by version: v1 43.6% won / rel. 0.178 / potions 87%; v2-10 78.8% / 0.428 / 74%; v11-30 83.8% / 0.483 / 78%;
v31-60 84.9% / 0.505 / 68%; v61+ 86.1% / 0.520 / 61%. Held-out evaluation (988 starts, no noise): network alone 85.6%
won / rel. 0.544; search 64 89.8% / 0.581; one-turn 88.5% / 0.569; exact-clone 93.3% / 0.618. The agent learned from
scratch in minutes, but the curve flattened after about 40 minutes.

### Tuning tests at equal compute (1 hour each)

| Test | Change | Result |
|---|---|---|
| A | training search 200/32 instead of 64/16 | worse or equal: network alone -0.019 (CI -0.030 to -0.009); search 64 -0.002 n.s.; only ~61k fights vs ~160k |
| B | cosine lr 1e-3 -> 1e-5 over the hour | slightly better, not significant: network alone +0.010 (CI -0.001 to +0.020), search 64 +0.006 (CI -0.003 to +0.015); adopted |
| C | d192 x 3 layers, 1.84M params | no better: network alone -0.010, search 64 -0.008; GPU-bound, ~10% fewer fights |

Conclusion: keep 64/16, keep the small network, add the cosine schedule. The larger network moved to the S3 shadow
experiment later.

### A20: outcome head vs scalar head (1 hour each, 988 starts, paired)

| Arm | Outcome - scalar (rel. T_f) |
|---|---|
| Network alone | -0.004 (-0.015 to +0.007) |
| Search 16 | -0.002 |
| Search 64 | -0.006 (-0.016 to +0.004) |
| One-turn | -0.001 |
| True future | +0.002 |

A tie; the outcome head used 5-7 points fewer potions. Re-pricing without retraining (potions worth nothing / 3x):
both networks followed the new tables (potion use 95-99% when worthless, 28-34% at 3x), and with search the heads were
equal. So D13's expected faster learning did not show, but the structural property (prices outside the weights) did.
The outcome head was kept.

### S2 phase 1 (1M fights, ~6.5 h)

30 actors x 33,334 fights, 163k trainer steps, cosine lr over the run. Self-play rel. T_f rose from 0.53 at 50 min to
0.56-0.57 at the end; checkpoints were flat at 0.59-0.60 (search 64) for hours and reached 0.611 at the last
checkpoint as the cosine reached its floor.

Gate G2 on 9,461 held-out starts:

| Arm | Won | Rel. T_f | vs search 64 (paired) |
|---|---|---|---|
| Network alone | 87.2% | 0.554 | -0.040 (-0.044 to -0.037) |
| **Search 64** | **91.8%** | **0.594** | - |
| One-turn search 64 | 90.9% | 0.585 | -0.009 (-0.012 to -0.006) |
| True-future search 64 | 93.9% | 0.620 | +0.026 (+0.024 to +0.029) |

(a) and (b) passed: search beats the network and MCTS beats one-turn search. Hidden information costs 0.026 rel. T_f
(about 2.1 win points in this comparison). Against the 1-hour run, the six hours added +0.027 to +0.030 in every arm.

### Human reference (same 988 starts, search 64)

| | Won | Rel. T_f | Potions used |
|---|---|---|---|
| Human logged fight | 94.5% | 0.625 | 24% |
| Agent, network alone | 85.6% | 0.544 | 34% |
| Agent, search 64 | 89.8% | 0.581 | 46% |
| Agent, true future | 93.3% | 0.618 | 43% |

By fight type: hallway 98.4% vs human 97.3%, elite 91.2% vs 92.5%, **boss 65.8% vs 83.8%**. The boss gap was the
main weakness. A dedicated boss test (117 held-out boss fights): search 64 51.3%, search 1,024 56.4%, true-future
search 64 65.8%, true-future search 1,024 76.1%, human 83.8%. With enough search most bosses are beatable at human
rates; the gap is the network's judgement under hidden information, not an impossible simulator.

## 4. S3: full-game self-play (2026-11-11 onward)

### Baseline (G3 reference, before training)

S1 macro policy greedy + S2 phase-1 combat network, 210 fixed seeds, search 64 on every move:

| Ascension, scoring | Runs won | Act 1 cleared | Act 2 cleared | Hallway / elite / boss fights won |
|---|---|---|---|---|
| A0, T_f | 3.3% (1.6-6.7) | 71% | 25% | 97.8 / 89.1 / 71.7% |
| A0, end HP | 2.9% (1.3-6.1) | 69% | 19% | 97.6 / 89.7 / 68.7% |
| A10, T_f | 0% (0-1.8) | 26% | 2% | 96.6 / 71.5 / 56.1% |

T_f vs HP at A0: 4 vs 3 seeds won (p = 1.0), no measurable win difference at this size; T_f was ahead on acts cleared
and used potions as intended (1.40 vs 0.78 per boss fight; 0.29 vs 0.47 per hallway fight).

### First 24-hour run and the four findings that changed the design

1. **The pure win target cannot start.** 0 wins in the first 1,024 training runs; V_run went to ~0 within two
   updates, and neither T_f nor the advantages carried signal. Fixed by D14 (blended outcome: alpha x win +
   (1-alpha) x acts/3, alpha = win rate / 0.2). With it V stayed calibrated (0.336 predicted vs 0.333 observed after
   4,600 runs).
2. **The S1 policy head cannot see HP.** S1 removed HP from the encoder for monotonicity, so the policy query had no
   HP input; after 40 minutes the policy rested 92% of the time. S3 adds the 26 HP ramps to the policy query through
   a zero-initialised layer.
3. **Argmax on the new event head loops.** 26% of evaluation runs died outside combat, e.g. holding on at Slippery
   Bridge page after page from full HP. Evaluation now samples the MAP and EVENT heads and uses argmax only for the
   S1 decision types.
4. **The value loss was moving the policy.** In the first minutes of the server run the behaviour-lag KL reached 0.41
   and the KL to S1 0.23 after four updates. The policy member's value loss trained the encoder it shares with the
   policy, outside PPO's trust region. The policy member now trains only its value heads (state detached); lag KL fell
   to 0.004-0.006 and PPO completed its steps.

Other fixes in the first days: a fight that could not finish looped forever past round 150 (fail the run at round
151; `max_rounds`); combat carry files were truncated by a 2.1 GB single-write limit (write in a loop, check size
before rename); a 64 GB disk filled with raw chunks (trainer archives/deletes, disk guard); straggler actors and
inference-server crashes (fail fast, watchdog).

### Learning curve

Selected A10 evaluations on 300 fixed seeds (full series in `doc/data/s3_curve.csv`):

| Segment | Self-play runs | Won | Mean floor | Act 1 | Act 2 |
|---|---|---|---|---|---|
| 4 | 53,516 | 0.0% | 27.4 | 77% | 8% |
| 10 | 187,037 | 0.0% | 39.7 | 93% | 60% |
| 14 | 228,733 | 2.0% | 41.8 | 95% | 71% |
| 20 | 352,962 | 4.0% | 42.3 | 95% | 73% |
| 24 | 499,071 | 9.0% | 43.5 | 97% | 79% |
| 30 | 710,244 | 13.7% | 43.2 | 96% | 80% |
| 35 | 898,763 | 19.7% | 44.1 | 96% | 83% |
| 37 | 989,082 | 23.7% | 44.2 | 97% | 82% |
| 40 | 1,080,985 | 25.7% | 42.6 | 92% | 76% |
| 42 | 1,160,296 | 25.7% | 45.0 | 96% | 87% |
| 44 | 1,237,521 | 28.0% | 43.4 | 92% | 81% |
| 45 | 1,276,289 | **31.3%** | 43.6 | 93% | 80% |

The curve is not monotone at 150 seeds early (0 / 0.7 / 1.3 / 2.0% are within noise of each other); it becomes
informative after 300 seeds from segment 24. The 24-hour run was extended several times by the owner; the best
checkpoint at the snapshot is segment 45.

### Ascension sweep of checkpoint s037 (154 seeds per level)

| A0 | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 92.9% | 95.5% | 96.1% | 96.7% | 86.3% | 91.5% | 89.6% | 85.1% | 81.2% | 48.1% | 19.5% |

At A10, deaths were boss 1 24.7%, boss 2 28.6%, act-2 boss 11.7%, act-3 elite 5.8%, act-3 hallway 3.2%, act-2 elite
2.6%. Top killers over all ascensions: Aeonglass 69, Test Subject 63, Queen 53, Knowledge Demon 28. A9 lost more runs
before the act-3 boss than A10 (35.5% vs 27.1% of runs), not significant at 154 seeds. Three runs with errors were
excluded (two Bowlbugs at the round limit, one simulator `OrbQueue is full`).

### A10 value analysis (segments 19-28, 28,673 runs, analysis only)

The run value beat the 9-cell table by only 0.007 log loss at A10. Why:

- **The target is mostly noise.** At A10 60% of runs die in act 3 (label 0.55), 32% in act 2 (0.275), 3.6% win. A
  predictor that knows every run's outcome could beat the table by only 0.086 / 0.061 / 0.043 in acts 1 / 2 / 3
  (A5-8: 0.254 / 0.229 / 0.187; A9: 0.211 / 0.180 / 0.163). The network takes about 10% of that room at A10 and 11% at
  A0-9.
- **Ranking within a floor is modest.** AUC for the run win at A10: 0.575 / 0.736 / 0.820 in acts 1 / 2 / 3; A9
  0.573 / 0.699 / 0.755; HP share alone 0.517 / 0.589 / 0.700.
- **Act-3 values sit on the banked floor.** Mean 0.578, sd 0.086; 40.4% of act-3 decision rows are below the 0.55
  already secured by reaching act 3 (A9 7.5%, A8 1.5%). At HP < 20% the value is 0.490 against a true 0.553.
- **Fights are priced almost only for survival.** Act-3 hallway T_max: A8 0.851, A9 0.731, A10 0.594, of which 0.55 is
  banked. The combat network has no ascension input and the enemies are the same at A9/A10 until the first act-3
  boss, yet at A10 act 3 (segment 28): 0.50 potions per hallway fight vs 0.36 at A9; 1.45/1.61 at elite/boss-1 start
  vs 1.80/1.98 at A9; elite fights won 77.3% vs 85.4%; first boss 41.0% vs 46.8%; act-3 runs reaching the boss 54.7%
  vs 70.5%.
- **The first of two bosses has almost no data.** After boss 1 (2,193 runs), the run is won 1.9% at HP < 20% (27% of
  cases), 13% / 34% / 52% / 71% in the next HP fifths; 17% / 47% / 69% with 0 / 1 / 2 potions; 79% hold none. The
  boss-1 table implies 5.5-39% over the same HP range. These rows are about 0.07% of the value's training rows.
- **New bias from 16 rows per run.** A run dying in act 1 has 33 rows, a won run 120, so 16 rows per run overweights
  short runs 3.6x among act-1 rows. Value minus outcome on fresh runs, act 1: -0.049 (A10), -0.070 (A9), -0.072 (A0-8)
  before the fix.

This analysis produced the improvement plan below. It also explains why the earlier death-value / double-boss tests
could not show the bias: they changed only the death value, while tables, T_max and the combat network's price inputs
stayed as they were.

### Improvement plan (2026-11-13)

| Change | Result | Decision |
|---|---|---|
| A. Stop the value memorising runs | 50k window x 16 rows, 150 steps, dropout + weight decay, probability averaging: fresh loss 0.718 -> 0.640, minus table +0.077 -> -0.005 | adopted |
| B. Less noisy search targets | 4 root candidates +0.0075 rel. T_f at 64 sims (+0.004 to +0.011), +0.0125 at 8 sims; 8 candidates +0.001 n.s.; 2 candidates -0.015 | adopted (4) |
| C. Combat capacity / lr | lr 5e-5 better than 2e-4 on every held-out loss; d256 x 4 learns faster per sample but far from trained in 1.5 h | lr 5e-5 adopted; larger net as shadow |
| D. Combat replay window | 2M -> 6M decisions (~30 min at 75 actors), carry 3M | adopted |
| E. PPO housekeeping | 2,560 runs per update, target KL 0.03, policy warm-up 600 s | adopted |
| F. One-step macro lookahead | not built | open |
| G. Evaluation that can see changes | 300 seeds, per-segment curve, value cells | adopted |

Ablation details for A:

| Arm | Fresh loss | Minus 9-cell table | Train-fresh gap | Member logit sd A0-8 |
|---|---|---|---|---|
| Base (4,096-run window, all rows) | 0.718 | +0.077 | 0.137 | 2.42 |
| 50k window, 16 rows/run, 150 steps | 0.657 | +0.012 | 0.050 | 1.34 |
| + dropout, weight decay 0.1 | 0.644 | -0.002 | 0.029 | 0.80 |
| **+ probability averaging (chosen)** | **0.640** | **-0.005** | 0.025 | 0.80 |
| 4 rows/run | 0.648 | +0.003 | 0.041 | 0.99 |
| 100k window, 8 rows/run | 0.645 | +0.000 | 0.034 | 0.87 |
| dropout + decay only (4,096 window) | 0.676 | +0.035 | 0.088 | 1.72 |

The memorisation was almost all in A0-9 runs; A10 matched the table in every arm. The final row-mix fix used
`--value-row-share 0.16` (keep 16% of each run's rows, stochastic rounding): replay of segments 26-28 improved act-1
value-minus-outcome from -0.049 to +0.005 (A10), -0.070 to +0.001 (A9), -0.072 to -0.021 (A0-8); fresh loss
0.5984 -> 0.5944. The weighted variant was within 0.005 and was stopped.

Search settings (B): mean relative T_f over held-out battle starts, paired by start, noise on, s3a_s023 network:

| Arm | 64 sims | 8 sims |
|---|---|---|
| 4 root candidates | **+0.0075 (+0.004 to +0.011)** | **+0.0125 (+0.010 to +0.015)** |
| 8 root candidates | +0.001 n.s. | - |
| 2 root candidates | -0.015 | - |
| c_scale 0.05 | -0.014 | - |
| Q-span floor 0.1 | -0.008 / -0.009 | - |

The plan's idea of a weaker sigma was wrong; fewer root candidates was the win. The evaluation keeps 16 candidates so
the curve stays comparable (the candidates gain does not show in evaluation).

Capacity (C), held-out after 32.8k steps on the same stream (cont = trained net at lr 2e-4, contlo = 5e-5):

| Arm | Policy KL | End-HP loss | Survival loss | Potions kept |
|---|---|---|---|---|
| cont | 0.259 | 2.75 | 0.167 | 0.33 |
| contlo | 0.256 | 2.61 | 0.158 | 0.27 |

From scratch, d256 x 4 (3.9M) learned the policy faster per sample (KL 0.293 vs 0.310) but neither came near the
trained network in 1.5 h. **Finding:** the combat value head overfits (training survival loss 0.05 vs held-out
0.16-0.18). Not solved; a shadow trainer is the current mitigation/experiment.

### Search allocation and practice

Allocation test (2,016 fixed A10 seeds, segment-9 weights):

| Arm | Acts cleared | Bosses won per run | Simulations |
|---|---|---|---|
| Hallway search halved | -0.001 (CI +/-0.03) | - | 69% |
| Every boss move at 64 (hallway halved) | +0.052 (+0.022 to +0.085) | **+0.064 (+0.031 to +0.100)** | 103% |
| Boss search doubled instead | - | +0.046 | 95% |

Adopted: hallway 8 simulations, every boss move 64. Act-2 boss won 50.7% -> 56.3%, act-3 boss 17.9% -> 21.5%.

Practice fights (one extra copy of each boss fight, from segment 12; from act 2 with 25% full-search moves from
segment 14) were never tested alone for strength. They went live with a package (boss search, surprise weighting,
averaged weights, tree reuse). Cost: 24-29% of all simulations before the 64/16 practice mix. A clean A/B (practice
vs no practice, same checkpoint, per hour) was proposed and not run ("not now").

### Boss-related tests (record kept on request)

| Idea | Status | Evidence |
|---|---|---|
| Practice boss fights | on since segment 12 | never isolated; package evaluation floor 39.7 -> 41.8, act 2 60% -> 71% |
| Every boss move searched at 64 | on | +0.064 bosses/run at 103% simulations |
| Death valued as progress (F1) | off | bosses/run -0.042 to +0.004, all CIs include 0 (1,000 seeds, s016 weights) |
| Double-boss table (F2) | off | same test: no gain |
| F1 + F2 | off | no gain |
| Stake-priced tables (`tf_stake=1`) | off | 18.3% vs 18.0% wins (-0.2 pts, -2.8 to +2.6); bosses/run +0.04 (-0.03 to +0.10); potions in act-3 elite -0.14; no win gain |
| Milestone value heads | off | 16.2% vs 14.5% (-1.6 pts, -4.2 to +0.9); replay loss -0.0201 -> -0.0208, AUC gain ~0.005, act-3 below banked 21.7% -> 0%; no play gain |

The A9-vs-A10 estimate for stake pricing (+20-45%) did not hold in evaluation: it came from self-play (hallway fights
at 8 simulations, sampled macro policy), where only 55% of A10 act-3 runs reach boss 1; in evaluation (64 simulations,
greedy macro) 82% already do.

### End-turn prompts (2026-11-15)

A boss fidelity review found that Knowledge Demon's Curse of Knowledge (Disintegration vs Mind Rot / Sloth / Waste
Away) is a prompt inside End Turn, which `ActionSpace` never expanded; live fights answered it at random. The same
held for start-of-turn power prompts. Fix: the search root probes End Turn once on a resampled clone and splits it
into one action per answer (`endturn_prompts=1`, default). `S3 prompttest` passes; the old build is byte-identical
with `endturn_prompts=0`.

### Observation layouts v4/v5 and the shadow network (2026-11-15)

- **v4:** card damage/block and hidden power counters in previously zero slots; `ext_feat` with a zero-initialised
  last layer, so v3 checkpoints are bit-exact. Checks: 6 A10 seeds, new build with `endturn_prompts=0 fp_ext=0`
  vs old build: identical run and fight logs (6 runs, 129 fights); `ext_test.py`: playing network blind to the new
  slots (0 difference), ext = playing at init, learns after one step. Cost: hash share of actor time 11% -> 14%.
- **v5:** 14 relic counters in the masked Relic f0/f1 slots (ext input 10 -> 12 columns), `obs=5`. Checks: `obsv5`,
  `obsv4`, `prompttest` pass; 6 seeds identical at `obs=4` and `obs=5` (one fight counter 5,643 vs 5,642); smoke
  segment 124 runs, 0 errors; shadow 702,384 parameters.
- **Shadow switch:** a second ext network trained on the same chunks, compared on fresh-chunk losses and paired
  evaluation seeds. Segment 42: playing 25.7% vs shadow 23.3% (p 0.51); segment 43: 24.3% vs 26.0% (p 0.64);
  segment 44: 28.0% vs 29.0% (p 0.81). After 43: KEEP (600 pairs, 76 vs 78); after 44: **SWITCH** (600 pairs,
  shadow-only wins 74 vs 66, +1.3 points, McNemar p ~0.5; fresh losses within the 2% tolerance). Segment 46 played
  with segment 45's shadow.
- **Fidelity review:** Test Subject, Queen, Aeonglass and Knowledge Demon were reviewed against the decompiled
  v0.111.0 game (code reading only). Ascension thresholds and all HP/damage values match; no difference that changes
  a single-player fight was found. Minor engine deviations (hook order, no "late" end-of-turn pass, extra guards) do
  not affect these fights. Remaining observation gaps found: Wither cards encoded by id/upgrade/cost but not growing
  damage; Withering Presence shows the threshold, not the cards left (both addressed by v4).

### Simulator speed (measurement only)

S3 actor profile (20 actors, 600 s): inside simulations 91%: engine steps 55% (end turns 33%, replayed paths 16%,
the reaching step 39%), clone 11%, observable-state hash 11-12%, leaf encode 9%, legal actions 8%, selection/backup
1%. Outside the search only 4%. Per simulation: 2.08 steps, 0.93 new leaves, 0.07 fight ends, 1.08 tree hits;
713-738 us at 20 actors vs 514 us at 4. Memory: ~300 MB/s allocated per actor, GC pauses 5.5% of wall time.
Trace hot spots: hook-listener lists rebuilt per dispatch (~12%), list growth ~14%, LINQ inside the engine 8-9%.
Consequences: replay avoidance is worth at most ~16%; the cheap wins are a string-free streaming hash (~10%), cached
vocabulary ids (~5-7%), cheaper legal-action keys, then the iRyougi hook-listener fixes. The perf patch implemented
the last group (1.31x with real networks; upstream suite 4,662 passed). CUDA MPS + graphs + overlap gave +18%
runs/hour at the same window and ~20 -> ~45 combat-trainer steps/s.

## 5. Ablations and their status

| ID | Question | Status / result |
|---|---|---|
| A1 | Floor sampling | done: uniform floors cut log loss by 40x seed noise, removed early-floor bias |
| A2 | Model type / size | done: no-attention wins or ties at every size; transformer kept as option |
| A3 | Floors-left auxiliary head | not run as a separate arm (used throughout at weight 0.25) |
| A4 | Forward-in-time generalization | done: one era forward costs ~0.004 AUC |
| A7 | Potions as input | not run as a separate arm |
| A8 | Version data | done: v0.111.0-only target cell; all versions for the base |
| A9 | HP handling | done: monotone head, 0 violations; log loss within 0.002 of unconstrained |
| A10 | Terminal score in MCTS | partly: T_f vs HP in M1 (no win difference at 998 starts); G3 T_f vs HP at A0 (4 vs 3 seeds, p=1.0); needs a stronger agent/larger N |
| A11 | Chance handling | G2 (c): true future +0.026 rel. T_f; resampling is the deployed choice |
| A12 | Search budget at deployment | 16 / 64 / 200 / 1,024 tested at various points; 64 is the knee |
| A13 | Search budget in S3 | 64/16 chosen; 200/32 at S2 was worse at equal compute |
| A14 | Fight-outcome inputs | done: dropped; no measurable cost |
| A15 | Lookahead rule | not built (one-step macro lookahead proposed, F) |
| A16 | BC weighting | not run |
| A17 | Perturbed fight starts | on (10%); causal-price effect not isolated |
| A18 | Warm start | not ablated end-to-end; S1/S2 warm starts used, arm R dropped by the owner |
| A19 | Confidence-gated search | implemented (`cskip`), off (threshold > 1) |
| A20 | Outcome vs scalar head | tie; outcome kept for re-pricing; both follow altered tables |

## 6. Open experiments

- Is practice worth its simulations? Two chains from one checkpoint, `practice_boss=1` vs `0`, compared per hour.
- T_f vs HP on wins with the current (31%) agent and N >= 1,000; the early test was underpowered.
- A map encoder or V_run map input; currently the MAP head has no value signal.
- Price potions gained mid-fight and in-fight permanent changes.
- A value head conditioned on potions used, to remove the independence approximation.
- The larger combat network as the playing network (shadow method is set up; one switch happened at segment 46).
- Combat value overfitting: stronger regularisation, more data, or a different value target.
