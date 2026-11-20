# Results: Ironclad A10 on game v0.113.1 (ic113)

All numbers come from the project's own runs. Evaluation seeds are fixed and never used in training. Confidence
intervals (CI) are 95% Wilson intervals unless stated. Terms (V_run, T_f, segment, panel, deployment rules) are
defined at the top of [experiments.md](experiments.md). The data behind every table is in [`results/`](../results/),
described in [results/README.md](../results/README.md). The previous phase's results (Defect, v0.111.0) are on branch
`v0.111-defect`.

## Headline

| | Value |
|---|---|
| Released weights | chain `s3ic`, segment 29 (see [checkpoints.md](checkpoints.md)) |
| A10 win rate | **152 / 300 = 50.7% (CI 45.0-56.3)** |
| Self-play runs before these weights | 528,810 (a lower bound; see "Segment numbers" below) |
| Evaluation | 300 fixed A10 seeds `Y100000`-`Y100299`, greedy macro with map argmax, search 64 on every move, 16 root candidates, deployment rules; 0 errors |
| Mean floor | 44.0 |
| Act 1 / act 2 cleared | 92% / 82% |
| Final boss (act-3 boss 2) | entered by 216 runs (72%), won 152 (70%) |
| Live play in the real game | 3 finished runs, 3 wins (A0, A6, A10) |

## Learning curve

One evaluation per adopted segment, on the same 300 seeds (304 for segments 22-26). "Runs" is the chain's run
counter after the segment. The win rate here leaves out runs that ended in an error; the per-segment report in
[results/s3ic/funnel.csv](../results/s3ic/funnel.csv) counts them as losses (it differs only for segments 0, 1, 2, 5
and 24, by at most 0.4 pp).

| Segment | Self-play runs | Won | Win rate (CI) | Mean floor | Act 1 | Act 2 | Self-play A10 |
|---|---|---|---|---|---|---|---|
| 0 | 45,637 | 69 | 23.3% (18.9-28.5) | 38.8 | 82% | 63% | 3.9% |
| 1 | 85,063 | 81 | 27.4% (22.6-32.7) | 40.8 | 87% | 69% | 8.1% |
| 2 | 123,219 | 104 | 34.8% (29.6-40.3) | 40.8 | 83% | 73% | 11.4% |
| 3 | 161,046 | 105 | 35.0% (29.8-40.6) | 42.4 | 89% | 77% | 13.9% |
| 4 | 200,596 | 101 | 33.7% (28.6-39.2) | 42.4 | 87% | 76% | 16.7% |
| 5 | 239,523 | 130 | 43.5% (38.0-49.1) | 43.7 | 90% | 81% | 18.1% |
| 6 | 278,271 | 122 | 40.7% (35.3-46.3) | 43.3 | 89% | 81% | 20.6% |
| 7 | 288,591 | 126 | 42.0% (36.6-47.7) | 43.5 | 89% | 82% | 19.1% |
| 8 | 298,378 | 118 | 39.3% (34.0-45.0) | 42.5 | 86% | 78% | 20.0% |
| 11 | 304,782 | 110 | 36.7% (31.4-42.3) | 41.5 | 86% | 72% | 18.9% |
| 12 | 314,598 | 125 | 41.7% (36.2-47.3) | 42.9 | 88% | 79% | 17.8% |
| 18 | 298,996 | 115 | 38.3% (33.0-43.9) | 42.1 | 87% | 76% | 17.7% |
| 19 | 299,310 | 121 | 40.3% (34.9-46.0) | 42.4 | 87% | 77% | (87 runs) |
| 20 | 302,721 | 111 | 37.0% (31.7-42.6) | 42.5 | 88% | 77% | 16.7% |
| 21 | 310,118 | 112 | 37.3% (32.1-42.9) | 42.6 | 87% | 77% | 16.8% |
| 22 | 341,785 | 124 / 304 | 40.8% (35.4-46.4) | 43.6 | 89% | 82% | 22.5% |
| 23 | 362,467 | 131 / 304 | 43.1% (37.6-48.7) | 44.2 | 91% | 83% | 24.0% |
| 24 | 382,849 | 123 / 304 | 40.6% (35.2-46.2) | 44.7 | 92% | 84% | 24.5% |
| 25 | 403,353 | 142 / 304 | 46.7% (41.2-52.3) | 43.8 | 90% | 82% | 24.8% |
| 26 | 424,275 | 142 / 304 | 46.7% (41.2-52.3) | 43.9 | 91% | 81% | 24.2% |
| 27 | 437,391 | 115 | 38.3% (33.0-43.9) | 42.9 | 89% | 78% | 24.0% |
| 28 | 490,211 | 128 | 42.7% (37.2-48.3) | 43.4 | 90% | 80% | 26.6% |
| **29** | **528,810** | **152** | **50.7% (45.0-56.3)** | **44.0** | **92%** | **82%** | **29.3%** |

Data: [results/s3ic/eval_curve.csv](../results/s3ic/eval_curve.csv) and
[results/s3ic/curriculum.csv](../results/s3ic/curriculum.csv) (self-play).

**Reading the curve.**
- The chain rose fast in its first six segments: from 23.3% to 43.5% in about 13 hours and 240k runs. The
  stalemate fix (from the segment-2 evaluation) accounts for at most about 1 point of that.
- From segment 5 to segment 28 the panel moved between 36.7% and 46.7% with no clear trend. Self-play A10, which is
  measured on about 5,000-19,000 A10 runs per full segment, kept rising: 18.1% (segment 5), 22.5-24.8% (22-27), 26.6% (28).
- Segment 29 changed four things at once (curriculum stage 4, `alpha` 0.85, act-3 boss practice, a shadow combat
  trainer beside the playing one) and gave the best reading, 50.7%. On the same seeds it beat segment 28 by +8.0 pp
  (paired CI +1.9 to +14.1, p = 0.014) and segment 26 by +3.7 pp (CI -3.1 to +10.4, p = 0.33). See experiment 17.

### Segment numbers and gaps

The segment numbers skip 9-10 and 13-17. Those segments produced no usable weights:
- 9, 10 and 13 were interrupted on the rented server before their first saved carry;
- 14 was lost together with its host machine;
- 15, 16 and 17 failed at once after the restore, because the restored code tree lacked a Python module.

The run counter counts run records that reached backup storage. At the restore it fell from 314,598 (segment 12) to
294,723, so segment 18 shows fewer runs than segment 12. Runs played in lost or discarded segments are not counted.
The full timeline is in [results/s3ic/segments.csv](../results/s3ic/segments.csv); the reasons are in
[experiments.md](experiments.md#8-interruptible-server-and-server-moves).

Segments 0-27 are 2 hours long (some recovered segments are shorter); segments 28 and 29 are 3 hours. The actor count
changed with the machine (60, 84, 36, 42, 32, 60).

## Act funnel over the chain

Runs entering and winning each boss fight, per 300 runs. Selected segments; all segments are in
[results/s3ic/funnel.csv](../results/s3ic/funnel.csv).

| Segment | Act-1 boss won | Act-2 boss won | Act-3 boss 1 won / entered | Act-3 boss 2 won / entered | HP share entering act-3 boss 2 |
|---|---|---|---|---|---|
| 0 | 245 | 186 | 133 / 171 (78%) | 69 / 133 (52%) | 0.588 |
| 5 | 269 | 243 | 205 / 237 (86%) | 130 / 205 (63%) | 0.628 |
| 12 | 263 | 238 | 200 / 230 (87%) | 125 / 200 (62%) | 0.631 |
| 26 | 278 | 245 | 213 / 236 (90%) | 142 / 213 (67%) | 0.663 |
| 28 | 271 | 239 | 199 / 230 (87%) | 128 / 199 (64%) | 0.621 |
| 29 | 277 | 246 | 216 / 242 (89%) | 152 / 216 (70%) | 0.678 |

The early gains came from reaching act 3 more often. The later gains came mostly from the final boss: its win rate
went from 52% (segment 0) to 55-64% (segments 2-24, 27 and 28) and to 67-70% (segments 25, 26 and 29).

## Segment 29 in detail

### Deaths by encounter (148 losses in 300 runs)

| Encounter | Room | Act | Deaths | Share of deaths |
|---|---|---|---|---|
| Test Subject | boss | 3 | 33 | 22.3% |
| Aeonglass | boss | 3 | 30 | 20.3% |
| Queen | boss | 3 | 27 | 18.2% |
| Kaiser Crab | boss | 2 | 8 | 5.4% |
| Entomancer | elite | 2 | 7 | 4.7% |
| Bygone Effigy | elite | 1 | 5 | 3.4% |
| Knowledge Demon | boss | 2 | 5 | 3.4% |
| Phrog Parasite | elite | 1 | 5 | 3.4% |
| Soul Fysh | boss | 1 | 4 | 2.7% |
| 14 other encounters | | | 24 | 16.2% |

The three act-3 bosses cause 90 of 148 deaths (60.8%). By act: 23 deaths in act 1, 31 in act 2, 94 in act 3. All 148
losses happened in a fight. Full list: [results/s3ic/deaths_s029.csv](../results/s3ic/deaths_s029.csv).

### Fights by type and act

| Room | Act | Fights | Won | Mean rel. T_f | Mean HP lost | Potions used per fight |
|---|---|---|---|---|---|---|
| Hallway | 1 | 1,310 | 99.9% | 0.900 | 5.1 | 0.28 |
| Hallway | 2 | 998 | 99.7% | 0.929 | 6.9 | 0.36 |
| Hallway | 3 | 836 | 100.0% | 0.960 | 6.2 | 0.19 |
| Elite | 1 | 762 | 98.6% | 0.800 | 20.0 | 0.67 |
| Elite | 2 | 663 | 98.2% | 0.846 | 20.6 | 0.60 |
| Elite | 3 | 539 | 99.3% | 0.895 | 21.6 | 0.40 |
| Boss | 1 | 288 | 96.2% | 0.771 | 36.2 | 1.07 |
| Boss | 2 | 262 | 93.9% | 0.801 | 33.5 | 0.94 |
| Boss | 3 (both) | 458 | 80.3% | 0.738 | 38.6 | 1.16 |
| **All** | | **6,116** | **97.6%** | | | |

"HP lost" is HP at the start minus HP at the end (0 for a lost fight). Per run: 20.4 fights, 6.55 elites, 3.64 rests,
4.82 smiths, 3.61 shops, 14.1 cards taken and 5.8 skipped, a final deck of 29.1 cards, 10.3 potions used, 25,544
simulations and 430 seconds. Data: [results/s3ic/fights_s029.csv](../results/s3ic/fights_s029.csv),
[results/s3ic/eval_s029.txt](../results/s3ic/eval_s029.txt).

### Act 3

| Fight | Entered | Won | HP share entering | Potions entering |
|---|---|---|---|---|
| Act-3 boss 1 | 242 (81%) | 216 (89%) | 0.945 | 2.24 |
| Act-3 boss 2 (final) | 216 (72%) | 152 (70%) | 0.678 | 1.27 |

Winning act-3 boss 2 wins the run: 152 of the 216 runs that beat boss 1 won the game.

| Boss | As act-3 boss 1: won / fought | As act-3 boss 2: won / fought |
|---|---|---|
| Aeonglass | 65 / 72 (90.3%) | 58 / 81 (71.6%) |
| Queen | 72 / 81 (88.9%) | 39 / 57 (68.4%) |
| Test Subject | 79 / 89 (88.8%) | 55 / 78 (70.5%) |

The three bosses are about equally hard, and each is harder as the second fight. The agent enters the second fight
with about two thirds of its HP (0.678) and 1.27 potions, against 0.945 and 2.24 for the first. Act-1 and act-2 bosses: per encounter in
[results/s3ic/bosses_s029.csv](../results/s3ic/bosses_s029.csv).

## Live play in the real game

| Date (UTC) | Weights | Ascension | Outcome | Floor | Last fight | HP after it |
|---|---|---|---|---|---|---|
| 2026-11-19 | Segment 21 | A0 | Won | 48 | Queen | 95 / 138 |
| 2026-11-19 | Segment 21 | A6 | Won | 48 | Test Subject | 82 / 97 |
| 2026-11-20 | Segment 29 | A10 | Won | 49 | Queen (act-3 boss 2) | 52 / 80 |

Settings: the evaluation settings above. Three runs show that the agent works in the real game; they cannot measure a
win rate. Data: [results/live/live_runs.csv](../results/live/live_runs.csv); setup in [live-play.md](live-play.md).

## Comparison anchors

| Reference | A10 win rate | Notes |
|---|---|---|
| **This agent, segment 29** | **50.7%** (45.0-56.3) | Ironclad, v0.113.1 simulator, 300 seeds |
| Human Ironclad players, v0.111.0 | 21.2% | 24,048 single-player standard runs (Spire Codex uploads). An earlier summary gave 20.9%. |
| Human Ironclad players, v0.113 | about 12.7% | 134 runs only |
| Zero-search PPO agent (sts2ai), Ironclad | 15.6% | its own report |
| Previous release of this project (Defect, v0.111.0, segment 45) | 31.3% (26.3-36.8) | release v0.1.0-s045, branch `v0.111-defect` |
| Later Defect networks (layout-v3 chain, segment 4) | 44.8% (41.8-47.9) | fresh 1,020-seed panel; segment 5 of that chain (45.0% on its own panel) is this phase's starting point |
| Project target | 50% | set in the Defect phase |

The comparisons across characters and game versions are rough. Defect and Ironclad play differently, and the human
numbers come from uploaders, who are not a random sample of players. The segment-29 reading reaches the 50% target on
one panel, but its CI reaches down to 45.0%.

Before training, the same system won 0% of Ironclad A10 runs (day 0) and 3.1% after the combat warm-up
([results/ic113/full_runs.csv](../results/ic113/full_runs.csv)).

## What the numbers do not show

- **One panel per segment.** Each segment is read on the same 300 seeds. The CI of one reading is about +-5.5 pp. The
  panel moved by 6-8 pp between neighbouring segments (40.6% -> 46.7% from segment 24 to 25, 46.7% -> 38.3% from 26
  to 27) while self-play A10 stayed at 24.0-24.8%. Most segment-to-segment changes are within noise.
- **The best reading is biased upward.** Segment 29 is the best of 23 readings on the same seeds. Choosing the
  maximum favours a lucky draw. The release reading needs a fresh panel.
- **Panels disagree.** In the Defect phase a 1,024-seed arm was once 5 points off its replication. Deployment
  decisions there were confirmed on a second fresh panel; no fresh panel has been run for segment 29 yet.
- **Changes are confounded.** The chain changed settings at segment boundaries (`win_bound`, `deck_route`,
  milestone heads, curriculum stages, `alpha`, boss practice) and moved between machines. Only the offline tests
  (T1-T4) and the warm-up are controlled comparisons.
- **Simulator, not game.** All rates except live play are measured in the simulator. The port checks and three won
  live runs support the transfer, but they do not measure the gap.
- **Self-play rates mix settings.** Self-play uses sampled map choices, 4 root candidates and lower ascensions; its
  A10 rate is a trend signal, not a comparable win rate.
