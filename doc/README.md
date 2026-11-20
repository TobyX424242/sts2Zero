# Documentation

This folder describes the Ironclad A10 agent on game v0.113.1 (project phase "ic113"): how it works, how it was
trained, what it achieved, and how to run it. The previous phase (Defect on v0.111.0) is documented on branch
`v0.111-defect`.

## Where to start

| If you want to... | Read |
|---|---|
| watch the agent play the real game | [live-play.md](live-play.md) |
| understand how the agent decides | [architecture.md](architecture.md) |
| see the numbers | [results.md](results.md) |
| train it further or reproduce the chain | [training.md](training.md), then [checkpoints.md](checkpoints.md) |
| find a file or a command | [code-map.md](code-map.md) |

## All documents

| Document | Content |
|---|---|
| [architecture.md](architecture.md) | Macro network, combat network, the T_f price table, Gumbel search, run-level decisions, potion_mix, actor keys, shadow trainer, inference serving |
| [training.md](training.md) | Simulator port, day-0 anchors, combat warm-up, the s3ic self-play chain (segments, supervisor, hook, curriculum, carries, interruption safety), evaluation protocol, settings, launching a chain |
| [simulator.md](simulator.md) | Upstream source, every change of our patch, port checks, fidelity evidence, speed |
| [code-map.md](code-map.md) | Entry points per directory, build commands, data and weights the scripts expect |
| [checkpoints.md](checkpoints.md) | Release assets, checksums, evaluation and continued training from the weights |
| [live-play.md](live-play.md) | Game mod and controller: setup on Windows and Linux, playing A0–A10 |
| [results.md](results.md) | Headline numbers and tables |
| [experiments.md](experiments.md) | Experiments of this phase and their outcomes |
| [findings.md](findings.md) | Conclusions and the mistakes behind them |
| [history.md](history.md) | Timeline |
| [references.md](references.md) | Literature and related Slay the Spire 2 work |

## Terms used throughout

| Term | Meaning |
|---|---|
| run | One game from the first floor to a win or a death |
| A10 | Ascension 10, the difficulty level the agent targets. At A10 act 3 ends with two bosses in a row. |
| macro decision | Any decision outside combat: map, card reward, shop, rest site, event, relic choice |
| V_run | The macro network's estimate of the chance to win the run from a given state |
| T_f(h, P) | The price table of one fight: V_run of the state right after the fight, if it ends with h HP and potion belt P |
| segment | One block of the self-play chain (2 h, later 3 h) that ends with saved weights and an evaluation |
| carry | A trainer's replay window and optimizer state, saved at a segment's end and loaded by the next segment |
| potion_mix | The rule that decides which potion to drop when the belt is full and a new potion is offered |
| s3ic | The name of the Ironclad self-play chain |

## Evidence rules

- Win rates come from fixed evaluation seeds that are never used in training. The chain's panel is seeds
  `Y100000`–`Y100299` at A10.
- Intervals are 95% Wilson intervals. Paired comparisons use McNemar's test on the same seeds.
- Dates are YYYY-MM-DD. Times, where given, are UTC.
- Each document marks what was checked (verified) and what was not.
