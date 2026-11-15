# sts2Zero

**A hybrid Slay the Spire 2 agent: a macro value network plus AlphaZero-style search in every fight.**

`sts2Zero` plays **Defect, Ascension 10, game v0.111.0** in the MIT-licensed
[iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim) simulator. It combines:

- a **macro network** that makes every out-of-combat decision and predicts the run value `V_run` (P(win));
- a **combat network** guiding **Gumbel MuZero-style search** in every fight;
- one economic link between the two: a fight is scored by
  `T_f(h, P) = V_run(post-fight state with HP h and potion belt P)`, and the combat network predicts the fight
  outcome that the search prices with that table.

Training is staged. **S1** warms the macro network up on human runs; **S2** warms the combat network up on human
battle starts; **S3** trains both networks on the agent's own full runs with search in every fight. The final
training signal is self-play only.

## Status

The last validated S3 checkpoint is **segment 45** (2026-11-15): **31.3% A10 win rate** on 300 fixed seeds
(94/300, 95% CI 26.3-36.8), after about **1.28M** self-play runs. The human Defect A10 uploader population on
v0.111.0 wins 15.5%; the best published end-to-end STS2 agent (AlphaDefect) claims 70.9% in its own engine. The
project target is 50%.

| Stage | Result |
|---|---|
| S1 macro warm-up (1.2M clean human runs, 965k training runs) | ensemble value log loss 0.4655, within-floor AUC 0.800, ECE 0.0038; card-reward policy top-1 53.3% vs 40.6% pick-rate |
| S2 combat warm-up (1M self-play fights) | search 64 wins 91.8% of 9,461 held-out battle starts, rel. T_f 0.594; network alone 87.2% / 0.554; true-future upper bound 93.9% / 0.620 |
| S3 full self-play (1.28M runs) | A10 win rate 0% at 53k runs -> 31.3% at 1.28M runs; mean floor 27.4 -> 43.6; act 2 cleared 8% -> 80% |
| Ascension sweep of an earlier S3 checkpoint (s037) | A0 92.9%, A5 91.5%, A8 81.2%, A9 48.1%, A10 19.5% (154 seeds each) |
| Last validated checkpoint | segment 45: 94/300 A10 wins (31.3%, 95% CI 26.3-36.8) |

Key data: [`doc/results.md`](doc/results.md), [`doc/data/s3_curve.csv`](doc/data/s3_curve.csv),
[`doc/data/ascension_sweep.csv`](doc/data/ascension_sweep.csv), [`doc/data/s1_final.json`](doc/data/s1_final.json).

## Checkpoints

The code is in this repository; the weights are in the GitHub release
[**v0.1.0-s045**](https://github.com/TobyX424242/sts2Zero/releases/tag/v0.1.0-s045):

| Asset | What it is |
|---|---|
| `s3_s045_weights.pt` | S3 combat network at segment 45 (layout v3, the network that scored 31.3%) |
| `s3_s045_macro.pt` | S3 macro ensemble + policy at segment 45 |
| `s3_s045_shadow.pt` | layout-v4/v5 `ext` combat network; the one segment 46 continues from |
| `SHA256SUMS` | checksums |
| `manifest.json` | stage, code commit, configs, parameter counts, evaluation |

Use them as the initialisation of a new S3 run:

```bash
COMBAT_INIT=/path/s3_s045_weights.pt \
MACRO_INIT=/path/s3_s045_macro.pt \
ASC=10 CONC=32 src/s3/run_s3.sh NAME 26 7200
```

`shadow.pt` uses the opt-in `ext` input and observation layout v5. `run_s3.sh` detects the layout from the
checkpoint; if you use it directly, set `OBS=5`. `weights.pt` is the v3 playing network. The checkpoints are
weights only: replay buffers and optimizer states are not included, so a new run starts with empty replay/optimizer
state. Training continues, but it is not an exact continuation of the original chain.

## Repository layout

```text
sts2Zero/
  README.md
  research -> src            # compatibility symlink; historical research/... paths still resolve
  src/
    sims/iRyougi-sts2-sim/   # simulator source, MIT, patched (see src/sims/NOTICE.md)
    s0/                      # simulator extensions: resampling clone, action space, encoders, bridge
    s1/                      # macro-network warm start (src/value_v2 is a compatibility symlink)
    value_v1/                # shared reconstruction + base trainer used by S1/S2/S3
    s2/                      # combat network + Gumbel search, warm start on human battle starts
    s3/                      # full-game self-play: macro + combat training on the agent's own runs
    pipeline/                # Spire Codex scan / sample / verify helpers
  doc/                       # architecture, training, experiments, findings, references, simulator, code map
```

`src/` keeps the original relative layout, so the C# project references and the Python `sys.path` inserts work
unchanged. `research/` at the repository root is a symlink to `src/` only so historical commands and comments keep
working. New work should use `src/`.

## Build

```bash
cd sts2Zero
export NUGET_PACKAGES=$PWD/.cache/nuget DOTNET_CLI_HOME=$PWD/.cache/dotnet
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$PWD/.cache/tmp
dotnet build src/s2/S2/S2.csproj -c Release -m:8
dotnet build src/s3/S3/S3.csproj -c Release -m:8
```

A Python 3.12 venv at `./.venv` with PyTorch, NumPy and orjson is expected by the scripts; see
`requirements.txt`. `run_s3.sh` / `run_selfplay.sh` take `COMBAT_INIT`, `MACRO_INIT`, `STARTS` and `TABLES`
environment variables. See [`doc/code-map.md`](doc/code-map.md) for the entry points and data paths.

## Data and provenance

- **Human warm-start data:** [Spire Codex](https://spire-codex.com/) runs, with the maintainer's confirmation for
  this project. The raw runs and the datasets derived from them are not part of the repository; S1/S2 code and
  documentation explain how they were built. S3 does not need human data.
- **Simulator:** [iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim), MIT, commit `db844ad` plus four local
  patches. The upstream license is retained; see [`src/sims/NOTICE.md`](src/sims/NOTICE.md) and
  [`doc/simulator.md`](doc/simulator.md).
- **Game:** Slay the Spire 2 is a commercial game. No game files are included; the simulator is a reimplementation.
- **AlphaDefect:** the published replay data of
  [AlphaDefect](https://github.com/hmty-CHEN/alphadefect-evaluation-runs) was used only for the early feasibility
  analysis, not for the final training.

## Documentation

| Document | Content |
|---|---|
| [`doc/architecture.md`](doc/architecture.md) | Macro network, combat network, T_f link, search, S3 agent |
| [`doc/training.md`](doc/training.md) | S0-S3 training pipeline, losses, hyperparameters, inference system |
| [`doc/experiments.md`](doc/experiments.md) | Every experiment and ablation, with dates and results |
| [`doc/findings.md`](doc/findings.md) | The major findings and the mistakes that produced them |
| [`doc/results.md`](doc/results.md) | Headline numbers and tables |
| [`doc/references.md`](doc/references.md) | Annotated literature and related STS2 work |
| [`doc/simulator.md`](doc/simulator.md) | Upstream source, patches, fidelity and performance evidence |
| [`doc/code-map.md`](doc/code-map.md) | File-by-file map, entry points, data/checkpoint expectations |
| [`doc/checkpoints.md`](doc/checkpoints.md) | Release assets, checksums and loading instructions |

## License

The code in this repository is released under the **MIT License**; see [`LICENSE`](LICENSE). The vendored simulator
keeps its own MIT license in `src/sims/iRyougi-sts2-sim/LICENSE`. The released checkpoints are research artifacts
trained in the simulator; if you redistribute them, keep the attribution and check the terms that apply to the
training data and the game.
