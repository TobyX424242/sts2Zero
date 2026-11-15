# Code map and running the code

## Tree

```text
src/
  sims/
    iRyougi-sts2-sim/        upstream simulator source (MIT) + our patches applied
    patches/                 the four patches, for provenance
    NOTICE.md                what changed and why
  s0/
    S0.Sim/                  resampling clone, search card selection, battle-start import,
                             action space, combat observation, bridge, TfScorer
    S0Tests/                 test runner (resample, choices, import, actions, encode, bridge,
                             obs v4/v5, m1, prompt test)
    py/                      combat vocabulary/encoder, round-trip test, bridge server
    export_battle_starts.py  human floor -> battle-start JSON
    make_tf_tables.py        battle-start -> T_f table (S1 ensemble)
    analyze_m1.py            M1 summary
  s1/                        macro warm start (formerly value_v2)
    build_dataset_v2.py      per-floor dataset
    build_decisions_v2.py    decision dataset for the policy head
    train_value_v2.py        value model (+ _r1 frozen version)
    train_macro_v2.py        MacroNet: value + policy
    s1_finalize.py           G1 report
    run_*.sh                 pipeline drivers
  value_v1/                  shared reconstruction + base trainer (only these two files)
  pipeline/                  Spire Codex scan / sample / verify
  s2/
    S2/                      NetSearch (Gumbel), Evaluator, Fights, Program
    py/                      infer_server, train_combat, model, benchmark, held-out eval
    *.py                     pool export, tables, human reference, diagnostics, summaries
    *.sh                     self-play, checkpoint eval, actor benchmark, search-params test
  s3/
    S3/                      RunAgent, MacroLayout, MacroBridge, Records, Program,
                             EncodeTest, PromptTest
    py/                      s3_server, combat_pipe, macro_s3, train_macro_s3,
                             roundtrip_macro, ext_test, ms_check, macro_cpu_bench
    *.py                     analysis/report scripts for the experiments
    *.sh                     run_s3, eval_s3, eval_asc_sweep
    data/                    macro_vocab.json, boss2_w2.json (small, redistributable)
```

`src/value_v2` is a symlink to `src/s1` for the original imports. `research/` at the repository root is a symlink
to `src/` for the historical `research/...` paths inside the code and comments. All new work should use `src/`.

## Entry points

### Build

```bash
export NUGET_PACKAGES=$PWD/.cache/nuget DOTNET_CLI_HOME=$PWD/.cache/dotnet
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$PWD/.cache/tmp
dotnet build src/s0/S0Tests/S0Tests.csproj -c Release -m:8
dotnet build src/s2/S2/S2.csproj -c Release -m:8
dotnet build src/s3/S3/S3.csproj -c Release -m:8
```

### S0 checks (no data needed for most)

```bash
T=src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll
dotnet $T resample 60 10000 L 3000      # clone / resample / encoder checks
dotnet $T choices 10 10 4 B 3000        # card prompts inside clones
```

The `import` and `m1` commands need a battle-start JSON; the checks and their expected output are in
`src/s0/README.md`.

### S2 self-play / evaluation

```bash
# self-play: name, actors, seconds; STARTS and TABLES point at the battle-start pool
STARTS=/path/starts_train.jsonl TABLES=/path/tables_train.jsonl \
  src/s2/run_selfplay.sh NAME 16 3600

# evaluate a checkpoint on held-out starts
STARTS=/path/starts_eval.jsonl TABLES=/path/tables_eval.jsonl \
  COUNT=1000 src/s2/eval_ckpt.sh runs/s2/NAME/weights.pt runs/eval_NAME 8 \
  "net:0 s16:16 s64:64 ot64:64:oneturn=1 ex64:64:exact=1"
```

### S3 training / evaluation

```bash
# training; COMBAT_INIT and MACRO_INIT are the released S3 checkpoints
COMBAT_INIT=/path/combat.pt MACRO_INIT=/path/macro.pt \
  ASC=10 CONC=32 src/s3/run_s3.sh NAME 26 7200

# evaluation on fixed seeds; pass the released combat and macro checkpoints
COUNT=300 ASC=10 src/s3/eval_s3.sh COMBAT.pt MACRO.pt runs/eval_X 8 \
  "tf64:64 hp64:64:scorer=hp"
```

### Encoder round trip and prompt test

```bash
dotnet src/s3/S3/bin/Release/net9.0/S3.dll prompttest
dotnet src/s3/S3/bin/Release/net9.0/S3.dll encode runs=300 ascension=0 out=runs/encode_a0
./.venv/bin/python -I src/s3/py/roundtrip_macro.py runs/encode_a0
```

## What the scripts expect

The repository contains **code only**. The original working tree had these large/derived artifacts; they are not
redistributed:

| Artifact | Produced by | Used by |
|---|---|---|
| Spire Codex dump (`runs-latest.jsonl.gz`, `meta_all.csv.gz`) | download from Spire Codex (use confirmed with the maintainer) | `src/pipeline/scan_runs.py`, `src/s1/*`, `src/s2/export_starts.py`, `human_reference.py` |
| Per-floor arrays `runs/floors_v2/` | `src/s1/build_dataset_v2.py` | `src/s1/train_*.py`, `src/s0/make_tf_tables.py`, `src/s2/make_tables.py` |
| Decision arrays | `src/s1/build_decisions_v2.py` | `src/s1/train_macro_v2.py` |
| Battle-start pool `starts_*.jsonl` | `src/s2/export_starts.py` | S2 self-play/eval, `make_tables.py` |
| T_f tables `tables_*.jsonl` | `src/s0/make_tf_tables.py` / `src/s2/make_tables.py` | S2 training/eval |
| S1 ensemble `macro_s1.pt` | `src/s1/run_s1_finalize.sh` | S3 `MACRO_INIT` |
| S2 combat checkpoint `combat_s2p1.pt` | S2 phase 1 | S3 `COMBAT_INIT` |
| S3 chunks `*.s2d`, `*.s3r` | S3 actors | S2/S3 trainers |
| S3 checkpoints | S3 trainers | evaluation |

The S3 code defaults still use the historical paths (`research/s3/data/macro_s1.pt`,
`research/s3/weights/combat_s2p1.pt`); the `research` symlink makes them resolve to `src/`. If you put a checkpoint
there, or pass the path explicitly, no code change is needed.

## Environment variables used by the launchers

### `src/s2/run_selfplay.sh`

`NAME N_ACTORS SECONDS`, then trainer args. `STARTS`, `TABLES`, `CONC` (64), `SIMS_FULL` (64), `SIMS_FAST` (16),
`PFULL` (0.25), `FIGHTS`, `SERVER_CPUS` (default `0,1`), `KEEP` (0/1; keep chunks). The script uses
`./.venv/bin/python -I` and `dotnet` from `PATH` or `./dotnet`.

### `src/s3/run_s3.sh`

`NAME N_ACTORS SECONDS`. `ASC` (10), `CONC` (32), `RUNS`, `SIMS_FULL` (64), `SIMS_FAST` (16), `PFULL` (0.25),
`PERTURB` (0.1), `COMBAT_INIT`, `MACRO_INIT`, `COMBAT_WINDOW`, `MTRAIN_ARGS`, `CTRAIN_ARGS`, `ACTOR_ARGS`,
`CSERVERS`, `MACRO_PERIOD`, `ASC_MIX`, `KEEP`, `MPS`, `CARRY`, `SHADOW_ARGS`, `SHADOW_FROM_PLAYING`.
The chain's `runs/<chain>/extra.env` is sourced at every segment start and can set `EXTRA_ACTOR_ARGS`,
`EXTRA_CTRAIN_ARGS`, `EXTRA_MTRAIN_ARGS`, `EXTRA_SERVER_ARGS`, `EXTRA_N`, `CSERVERS`, `ASC_MIX`, `MPS`.

### `src/s3/eval_s3.sh`

`COMBAT_WEIGHTS MACRO_FILE OUTDIR N_ACTORS "name:sims[:key=value...] ..."`. `COUNT` (200), `FIRST` (0), `ASC` (10),
`CONC` (16), `SEEDS` (`EVAL`), `CSERVERS` (1). The observation layout is read from the checkpoint's `config.obs`;
`OBS` overrides it.

## Analysis scripts

The `*.py` files at the top of `src/s3` and `src/s2` are the analysis tools quoted in `doc/experiments.md`:
`a10val_*` (value analysis), `alloc_compare.py` (search allocation), `asc_sweep_report.py`,
`curve_runs.py`, `diag_runs.py`, `fit_boss2.py`, `summarize_runs.py`, `tf_dump_report.py`, `value_calib.py`,
`value_replay_report.py`, `table_diagnostics.py`, `human_reference.py`, `compare_heads.py`, `curve_by_time.py`.
They read run/fight logs or chunks.

## Common pitfalls

- Run the scripts **from the repository root**. They compute the root from their own path, but the C# actors and
  Python imports assume `src/` as a sibling tree.
- `taskset -c ${SERVER_CPUS:-0,1}` in `eval_ckpt.sh` assumes CPU 0 and 1 exist; on a container with a restricted
  cpuset, set `SERVER_CPUS` to an allowed range or remove the `taskset`.
- The shared-memory bridge uses `/dev/shm`; a container with a tiny `/dev/shm` will fail. The actor/server stop after
  300 s without an answer rather than hanging forever.
- `MPS=1` requires `nvidia-cuda-mps-control`; `src/s3/mps.sh start|status|stop` manages the daemon. It leaves the
  daemon running between segments by design; stop it only while no segment is running.
- The S3 training loop is CPU-bound. One Python inference process saturates at about 27k states/s; use 3-4 combat
  processes for 60-90 actors.
- Do not use `--exact 1` for training or deployment; it is the true-future upper-bound arm and leaks hidden
  information.
