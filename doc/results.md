# Key results

All numbers are from the project's own runs. Evaluation seeds are fixed and never used in training. "Rel. T_f" is the
mean fight-end T_f divided by the fight's table maximum. Confidence intervals are 95%. Larger tables are in
`doc/data/`.

## S1: macro warm start

Dataset: 1.2M clean human runs; 965,021 training runs. Target cell Defect A10 v0.111.0: 16,302 runs, 15.5% human
win rate.

### Data-scaling curve (test split)

| Training runs | No-attention log loss | No-attention within-floor AUC | Transformer log loss | Transformer within-floor AUC |
|---|---|---|---|---|
| 25,000 | 0.5056 | 0.7646 | 0.5239 | 0.7620 |
| 50,000 | 0.4883 | 0.7784 | 0.4936 | 0.7752 |
| 100,000 | 0.4800 | 0.7864 | 0.4841 | 0.7830 |
| 300,000 | 0.4707 | 0.7954 | 0.4731 | 0.7938 |
| 965,021 | **0.4604** | **0.8054** | 0.4628 | 0.8036 |

### G1 (5-member ensemble)

| Metric | Value |
|---|---|
| Test log loss | 0.4655 |
| AUC / within-floor AUC | 0.8365 / 0.8002 |
| ECE (floor bands) | 0.0038 |
| Target-cell log loss / within-floor AUC / ECE | 0.4301 / 0.7814 / 0.019 |
| HP monotonicity (+10 HP) | 0 violations / 39,850 |
| +1 potion wrong by > 0.5 pt | 2.82% (accepted) |
| +100 gold wrong by > 0.5 pt | 0.02% |
| Curse (Clumsy/Injury/Regret/Doubt) wrong by > 0.5 pt | 0.00-0.01% |
| Card-reward top-1 / pick-rate baseline | 53.3% / 40.6% |
| Ancient / relic / removal / rest / smith / shop top-1 | 82.6 / 54.9 / 73.2 / 67.2 / 47.0 / 34.8% |

## S2: combat warm start

1M fights, 163k trainer steps, search 64/16 during self-play.

### Gate G2 (9,461 held-out battle starts)

| Arm | Won | Rel. T_f | Difference vs search 64 |
|---|---|---|---|
| Network alone | 87.2% | 0.554 | -0.040 (-0.044 to -0.037) |
| **Search 64** | **91.8%** | **0.594** | - |
| One-turn search 64 | 90.9% | 0.585 | -0.009 (-0.012 to -0.006) |
| True-future search 64 | 93.9% | 0.620 | +0.026 (+0.024 to +0.029) |

### Human reference (988 starts, search 64)

| Arm | Won | Rel. T_f | Potions used |
|---|---|---|---|
| Human logged fight | 94.5% | 0.625 | 24% |
| Network alone | 85.6% | 0.544 | 34% |
| Search 64 | 89.8% | 0.581 | 46% |
| True future, 64 | 93.3% | 0.618 | 43% |

By fight type at search 64: hallway 98.4% vs human 97.3%; elite 91.2% vs 92.5%; **boss 65.8% vs 83.8%**.

### Boss test (117 held-out boss fights)

| Arm | Won | Rel. T_f |
|---|---|---|
| Human | 83.8% | 0.656 |
| Search 64 | 51.3% | 0.370 |
| Search 1,024 | 56.4% | 0.423 |
| True future, 64 | 65.8% | 0.477 |
| True future, 1,024 | **76.1%** | **0.574** |

## S3: full-game self-play

The training run reached 1.28M self-play runs. Evaluations are A10, 300 fixed seeds, greedy macro, 64 simulations on
every move. The full series is in [`doc/data/s3_curve.csv`](data/s3_curve.csv).

| Segment | Self-play runs | Won | 95% CI | Mean floor | Act 1 | Act 2 |
|---|---|---|---|---|---|---|
| 4 | 53,516 | 0.0% | 0.0-2.5 | 27.4 | 77% | 8% |
| 10 | 187,037 | 0.0% | 0.0-2.5 | 39.7 | 93% | 60% |
| 14 | 228,733 | 2.0% | 0.7-5.7 | 41.8 | 95% | 71% |
| 20 | 352,962 | 4.0% | 1.8-8.5 | 42.3 | 95% | 73% |
| 24 | 499,071 | 9.0% | 6.3-12.8 | 43.5 | 97% | 79% |
| 30 | 710,244 | 13.7% | 10.2-18.0 | 43.2 | 96% | 80% |
| 35 | 898,763 | 19.7% | 15.6-24.5 | 44.1 | 96% | 83% |
| 37 | 989,082 | 23.7% | 19.2-28.8 | 44.2 | 97% | 82% |
| 40 | 1,080,985 | 25.7% | 21.1-30.9 | 42.6 | 92% | 76% |
| 42 | 1,160,296 | 25.7% | 21.1-30.9 | 45.0 | 96% | 87% |
| 44 | 1,237,521 | 28.0% | 23.2-33.3 | 43.4 | 92% | 81% |
| 45 | 1,276,289 | **31.3%** | **26.3-36.8** | 43.6 | 93% | 80% |

### Ascension sweep of checkpoint s037 (154 seeds per level)

| A0 | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 92.9% | 95.5% | 96.1% | 96.7% | 86.3% | 91.5% | 89.6% | 85.1% | 81.2% | 48.1% | 19.5% |

A10 deaths (% of all A10 runs): act-3 boss 1 24.7%, act-3 boss 2 28.6%, act-2 boss 11.7%, act-3 elite 5.8%, act-3
hallway 3.2%, act-2 elite 2.6%, act-2 hallway 0.6%, act-1 elite 1.3%, other 1.3%. Top killers over all ascensions:
Aeonglass 69, Test Subject 63, Queen 53, Knowledge Demon 28.

The s037 checkpoint is earlier than segment 45 (19.5% vs 31.3% at A10).

## Improvement experiments

### A: value fit (replay of segments 19-22, 140k runs)

| Arm | Fresh loss | Minus 9-cell table | Train-fresh gap | Member sd A0-8 |
|---|---|---|---|---|
| Base | 0.718 | +0.077 | 0.137 | 2.42 |
| 50k window, 16 rows, 150 steps | 0.657 | +0.012 | 0.050 | 1.34 |
| + dropout + wd 0.1 | 0.644 | -0.002 | 0.029 | 0.80 |
| **+ probability averaging** | **0.640** | **-0.005** | 0.025 | 0.80 |

### B: search settings (held-out battle starts, s3a_s023, paired by start)

| Arm | 64 sims | 8 sims |
|---|---|---|
| 4 root candidates | +0.0075 (+0.004 to +0.011) | +0.0125 (+0.010 to +0.015) |
| 8 root candidates | +0.001 n.s. | - |
| 2 root candidates | -0.015 | - |
| c_scale 0.05 | -0.014 | - |
| Q-span floor 0.1 | -0.008 / -0.009 | - |

### C: combat capacity and learning rate (held-out, same stream, after 32.8k steps)

| Arm | Policy KL | End-HP loss | Survival loss | Potions kept |
|---|---|---|---|---|
| lr 2e-4 | 0.259 | 2.75 | 0.167 | 0.33 |
| **lr 5e-5** | **0.256** | **2.61** | **0.158** | **0.27** |

From scratch, d256 x 4 (3.9M) learned faster per sample (KL 0.293 vs 0.310) but did not approach the trained network
in 1.5 h. The combat value head overfits: training survival loss 0.05 vs held-out 0.16-0.18.

### Search allocation (2,016 fixed A10 seeds, s009 weights)

| Arm | Acts cleared | Bosses won per run | Simulations |
|---|---|---|---|
| Hallway search halved | -0.001 (CI +/-0.03) | - | 69% |
| Every boss move at 64 | +0.052 (+0.022 to +0.085) | **+0.064 (+0.031 to +0.100)** | 103% |
| Boss search doubled | - | +0.046 | 95% |

### Negative or neutral results

| Test | Result |
|---|---|
| Death value as progress (F1) | bosses/run -0.042 to +0.004, CIs include 0 |
| Double-boss table (F2) | no gain; F1+F2 no gain |
| Stake-priced tables | 18.3% vs 18.0% wins (-0.2 pts, -2.8 to +2.6); no win gain |
| Milestone value heads | 16.2% vs 14.5% (-1.6 pts, -4.2 to +0.9); no win gain |
| Outcome vs scalar head (A20) | tie in every arm; outcome kept for re-pricing |
| Larger S2 network (d192 x 3) | no better at equal compute |

## Throughput

| Metric | Measured |
|---|---|
| Simulator clone / card / end turn (laptop) | 27 us / 52 us / 239 us |
| Clone + one action, 1 process / 24 processes | 8.6k/s / 115k/s |
| Simulator on 9950X server, 24 processes | 144k/s |
| S3, 26 actors | 29.4k simulations/s, 6.3 runs/s, ~23k runs/h early |
| S3, 60-85 actors | ~25k runs/h |
| Raw / zstd combat chunks | ~25 GB/h / ~0.45 GB/h |
| Simulator perf patch | 1.31x with real networks; 4,662 upstream tests pass |
| MPS + CUDA graphs + overlap | +18% runs/h; combat trainer ~20 -> ~45 steps/s |

## Data and checkpoints

The human-run data, battle-start pools, T_f tables, training chunks and checkpoints are not included (see the
top-level README). The JSON/CSV files in `doc/data/` are summaries only:
[`s1_final.json`](data/s1_final.json), [`s1_scaling.csv`](data/s1_scaling.csv),
[`s3_curve.csv`](data/s3_curve.csv), [`ascension_sweep.csv`](data/ascension_sweep.csv).
