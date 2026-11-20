# sts2Zero

sts2Zero is a game-playing agent for Slay the Spire 2. It plays the **Ironclad at Ascension 10 on game version
v0.113.1**. Two neural networks share the work. A *macro network* makes every decision outside combat (map, card
rewards, shops, rest sites, events) and estimates the chance of winning the run. A *combat network* guides an
AlphaZero-style tree search that plays every fight. The two are linked by one price table: before each fight, the
macro network says what each possible end state of the fight (HP left, potions kept) is worth for the rest of the run,
and the search plays to maximise that worth. Both networks are trained by self-play in
[iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim), an MIT-licensed reimplementation of the game. The agent can
also play the real game through a game mod.

## Status (2026-11-20)

| What | Result |
|---|---|
| Latest evaluated checkpoint | segment 29 of the self-play chain `s3ic` |
| A10 win rate in the simulator | **152 of 300 runs = 50.7%** (95% CI 45.0–56.3) on 300 fixed seeds, 0 errors |
| Mean floor reached / act 1 cleared / act 2 cleared | 44.0 / 92% / 82% |
| Self-play runs before this checkpoint | 528,810 (as counted by the chain; see [doc/training.md](doc/training.md)) |
| Real game | 3 finished live runs, 3 wins: A0 and A6 with segment-21 weights (2026-11-19), A10 with the segment-29 weights (2026-11-20, floor 49); see [doc/results.md](doc/results.md) |

How to read these numbers:

- The 300-seed panel has a 95% interval about ±5.5 points wide. Neighbouring segments scored 46.7% (25, 26),
  38.3% (27) and 42.7% (28) on the same seeds. Segment 29 is the best single reading so far; it has not yet been
  repeated on a second, fresh seed panel.
- Three real-game runs show that the agent plays end to end in the real game. They are not a win-rate measurement.
- Most losses happen at the act-3 bosses. In the segment-29 evaluation, 90 of the 148 losses were to Test Subject,
  Aeonglass and Queen.

Full tables: [doc/results.md](doc/results.md) and [results/s3ic/eval_curve.csv](results/s3ic/eval_curve.csv).

## Comparison anchors

| Reference | Ironclad A10 win rate | Notes |
|---|---|---|
| Human uploaders, v0.111.0 | 21.2% | Spire Codex runs: single player, standard mode, no mods, not abandoned (24,048 runs). An earlier summary gave 20.9%. |
| Human uploaders, v0.113.0 + v0.113.1 | about 12.7% | Only 134 runs, too few for a firm reference |
| This agent, segment 29 | 50.7% | Simulator, 300 fixed seeds |

The previous phase of this project, a Defect agent on v0.111.0, reached 31.3% at A10 (release `v0.1.0-s045`, branch
`v0.111-defect`). Human Defect A10 players won 15.5% on that version.

## Quick start

**Play the real game (Windows or Linux).** Follow [doc/live-play.md](doc/live-play.md). In short:

1. `scripts/setup.sh` (or `scripts/setup.ps1` on Windows) sets up Python and checks for .NET 9.
2. `scripts/fetch_weights.sh` (or `.ps1`) downloads the release weights and checks their SHA-256 sums.
3. Install the AgentBridge mod, start the game, then run `src/live/play_live.sh` (or `.ps1`) and pick A0–A10.

**Evaluate in the simulator, or continue training.** Build the C# projects, then follow
[doc/checkpoints.md](doc/checkpoints.md) (evaluation) or [doc/training.md](doc/training.md) (a new self-play chain
from the release weights). Training needs a Linux machine with an NVIDIA GPU and many CPU cores.

## Weights

The weights are in the GitHub release
[**v0.2.0-ic113-s029**](https://github.com/TobyX424242/sts2Zero/releases/tag/v0.2.0-ic113-s029), not in the git history.

| Asset | What it is |
|---|---|
| `ic113_s029_weights.pt` | Combat network of segment 29 (724,810 parameters) |
| `ic113_s029_macro.pt` | Macro network of segment 29: an ensemble of 5 members, 2.17M parameters each |
| `ic113_s029_vbar.json` | Potion price table (`potion_mix`) that the segment-29 evaluation used |
| `manifest.json` | Machine-readable record: simulator commit, evaluation settings and result, configs, hashes |
| `SHA256SUMS` | Checksums of all release files |
| `AgentBridge-mod.zip` | The game mod for live play (built from `src/live/mod`) |
| `LiveAgent.zip` | The live-play controller (built from `src/live/agent`) |

Details and checksums: [doc/checkpoints.md](doc/checkpoints.md).

## Repository layout

```text
sts2Zero/
  README.md            this file
  doc/                 documentation (index: doc/README.md)
  src/
    sims/              the simulator: iRyougi/sts2-sim v0.113.1.1 with our patch applied, and the patch file
    s0/                simulator extensions: resampling clone, action space, combat encoder, shared-memory bridge
    s1/                macro-network warm start on human runs (from the Defect phase)
    value_v1/          run reconstruction and base trainer that s1 and s3 import
    s2/                combat network, Gumbel search, combat trainer
    s3/                full-game self-play: run agent, macro trainer, inference server, chain scripts, watch panel
    ic113/             the v0.113 / Ironclad port, combat warm-up and the s3ic chain scripts
    live/              game mod and controller for the real game
    pipeline/          Spire Codex data selection helpers
  configs/             settings of the s3ic chain at segment 29
  checkpoints/         manifest.json and SHA256SUMS of the release (the weights are release assets)
  results/             evaluation curve and reports of the s3ic chain
  scripts/             setup and weight download
```

## Branches

| Branch | Content |
|---|---|
| `main` | Ironclad A10 on v0.113.1 (this phase) |
| `v0.111-defect` | Previous phase: Defect A10 on v0.111.0, release `v0.1.0-s045` (31.3% A10) |

## Documentation

| Document | Content |
|---|---|
| [doc/architecture.md](doc/architecture.md) | How the agent decides: networks, the T_f price table, search, run-level rules |
| [doc/training.md](doc/training.md) | Port, combat warm-up and the s3ic self-play chain; settings; how to launch a chain |
| [doc/simulator.md](doc/simulator.md) | Simulator source, our patch, port checks, fidelity and speed |
| [doc/code-map.md](doc/code-map.md) | Entry points per directory, build commands, what data and weights the scripts expect |
| [doc/checkpoints.md](doc/checkpoints.md) | Release assets, checksums, loading and evaluation |
| [doc/live-play.md](doc/live-play.md) | Playing the real game with the agent |
| [doc/results.md](doc/results.md) | Headline numbers and tables |
| [doc/experiments.md](doc/experiments.md) | Experiments of this phase, with results |
| [doc/findings.md](doc/findings.md) | What was learned |
| [doc/history.md](doc/history.md) | Timeline of the project |
| [doc/references.md](doc/references.md) | Literature and related Slay the Spire work |

## Data and provenance

- **Simulator.** [iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim), MIT, tag `v0.113.1.1` (commit `1394c37`), with
  our patch (`src/sims/patches/iryougi-sts2-sim-v113.patch`). The upstream license is kept in
  `src/sims/iRyougi-sts2-sim-v113/LICENSE`. See [doc/simulator.md](doc/simulator.md).
- **Game.** Slay the Spire 2 is a commercial game. This repository contains no game files. The simulator is a
  reimplementation; the live-play mod loads into a game you own.
- **Human data.** [Spire Codex](https://spire-codex.com/) runs, used with the maintainer's confirmation. They were used
  for warm starts only: the macro network's original warm start in the Defect phase, and the Ironclad combat warm-up
  of this phase (human battle starts). Human win rates above come from the same data. The raw runs and derived
  datasets are not part of this repository. Self-play training itself uses no human data.
- **AlphaDefect.** The published replays of
  [AlphaDefect](https://github.com/hmty-CHEN/alphadefect-evaluation-runs) were used only for the early feasibility
  analysis, not for training.

## License

The code in this repository is released under the MIT License; see [LICENSE](LICENSE). The vendored simulator keeps
its own MIT license. The released weights are research artifacts trained in the simulator; if you redistribute them,
keep the attribution and check the terms that apply to the training data and the game.
