# Slay the Spire 2 agent: hybrid macro value network + MCTS combat

A research codebase for a **Defect, Ascension 10, game v0.111.0** agent in the MIT-licensed
[iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim) simulator. The agent combines:

- a **macro network** that makes every out-of-combat decision and predicts the run value `V_run` (P(win));
- a **combat network** guiding a **Gumbel MuZero-style search** in every fight;
- a single economic link between them: a fight is scored by `T_f(h, P) = V_run(post-fight state with HP h and potion
  belt P)`, and the combat network predicts the fight outcome that the search prices with that table.

Training is staged: S1 warms the macro network up on human runs, S2 warms the combat network up on human battle
starts, and S3 trains both networks on the agent's own full runs with search in every fight.

**Status at the snapshot (2026-11-15):** S1 and S2 are complete and passed their gates. S3 is training on a rented
box. The last recorded evaluation (segment 45, 300 fixed A10 seeds) won **31.3%** of runs, against **15.5%** for the
human Defect A10 uploader population on v0.111.0. The run had processed about 1.28M self-play runs. The project target
is 50%; the best published end-to-end agent (AlphaDefect) claims 70.9% in its own engine.

## Headline results

| Stage | Result |
|---|---|
| S1 macro warm-up (1.2M clean human runs, 965k training runs) | ensemble value log loss 0.4655, within-floor AUC 0.800, ECE 0.0038; card-reward policy top-1 53.3% vs 40.6% pick-rate |
| S2 combat warm-up (1M self-play fights) | search 64 wins 91.8% of 9,461 held-out battle starts, rel. T_f 0.594; network alone 87.2% / 0.554; true-future upper bound 93.9% / 0.620 |
| S3 full self-play (1.28M runs) | A10 win rate 0% at 53k runs -> 31.3% at 1.28M runs; mean floor 27.4 -> 43.6; act 2 cleared 8% -> 80% |
| Ascension sweep of an earlier S3 checkpoint (s037) | A0 92.9%, A5 91.5%, A8 81.2%, A9 48.1%, A10 19.5% (154 seeds each) |
| Last recorded checkpoint | segment 45: 94/300 A10 wins (31.3%, 95% CI 26.3-36.8) |

Key data: [`doc/results.md`](doc/results.md), [`doc/data/s3_curve.csv`](doc/data/s3_curve.csv),
[`doc/data/ascension_sweep.csv`](doc/data/ascension_sweep.csv), [`doc/data/s1_final.json`](doc/data/s1_final.json).

## Repository layout

```text
sts2-agent/
  README.md
  research -> src            # compatibility symlink; the original code paths (research/...) still resolve
  src/
    sims/iRyougi-sts2-sim/   # simulator source, MIT, patched (see src/sims/NOTICE.md)
    s0/                      # simulator extensions: resampling clone, action space, encoders, bridge
    s1/                      # macro-network warm start (formerly value_v2; src/value_v2 is a symlink)
    value_v1/                # shared reconstruction + base trainer used by S1/S2/S3 (trial-1 POC not included)
    s2/                      # combat network + Gumbel search, warm start on human battle starts
    s3/                      # full-game self-play: macro + combat training on the agent's own runs
    pipeline/                # Spire Codex scan / sample / verify helpers
  doc/                       # architecture, training, experiments, findings, references, simulator, code map
```

`src/` keeps the original relative layout, so the C# project references and the Python `sys.path` inserts work
unchanged. `research/` at the repository root is a symlink to `src/` only so that historical commands and comments
keep working. All new work should use `src/`.

## Build and run (no training data or checkpoints are included)

```bash
cd sts2-agent
export NUGET_PACKAGES=$PWD/.cache/nuget DOTNET_CLI_HOME=$PWD/.cache/dotnet
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$PWD/.cache/tmp
dotnet build src/s2/S2/S2.csproj -c Release -m:8
dotnet build src/s3/S3/S3.csproj -c Release -m:8
```

A Python 3.12 venv at `./.venv` with PyTorch, NumPy and orjson is expected by the scripts; see
`requirements.txt`. The battle-start pools,
T_f tables, training chunks and checkpoints are **not** in the repository; `run_s3.sh` and `run_selfplay.sh` take
`MACRO_INIT`, `COMBAT_INIT`, `STARTS` and `TABLES` environment variables. See [`doc/code-map.md`](doc/code-map.md).

## What is intentionally excluded

- **Human-run data** from Spire Codex and everything derived from it: run files, per-floor/decision datasets,
  battle-start pools, T_f tables and the warm-up checkpoints. The use of the data was confirmed with the maintainer,
  but it is not redistributed here.
- **The early POC** (trial-1 value transformer, review probes, AlphaDefect replay experiments). It is summarised in
  one paragraph in [`doc/experiments.md`](doc/experiments.md); its code and results are not part of this repository.
- **Server operations**: the Nextcloud backup workflow, the unattended training supervisor, deployment/restart scripts
  and the training watch panel.
- **Game files** and the locally decompiled game assembly.

The simulator **is** included, with its MIT license and a clear statement of what we changed:
[`src/sims/NOTICE.md`](src/sims/NOTICE.md) and [`doc/simulator.md`](doc/simulator.md).

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

## Provenance and licensing

- Simulator: iRyougi/sts2-sim, MIT, commit `db844ad` plus four local patches (see the NOTICE). The upstream `LICENSE`
  is retained.
- Slay the Spire 2 is a commercial game; no game files are included. The simulator is a reimplementation and its
  fidelity is discussed in [`doc/simulator.md`](doc/simulator.md).
- The agent code in `src/s0`, `src/s1`, `src/s2`, `src/s3` and `src/value_v1` is this project's work. No license is
  declared for it in this snapshot; add one before redistributing.
