# Code map

Where things are, how to build them, and what each entry point expects. Run every script from the repository root;
the scripts find the root from their own path and call `./.venv/bin/python -I` and `dotnet` (from `PATH`, or a
`./dotnet` folder if present).

## 1. Build

```bash
scripts/setup.sh                                     # Python venv in ./.venv (uv), checks for .NET 9
dotnet build src/s3/S3/S3.csproj -c Release -m:8     # S3 agent; also builds S2, S0 and the simulator
dotnet build src/s0/S0Tests/S0Tests.csproj -c Release -m:8   # S0 test runner (optional)
```

Outputs: `src/s3/S3/bin/Release/net9.0/S3.dll`, `src/s2/S2/bin/Release/net9.0/S2.dll`,
`src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll`. The live-play builds are described in [live-play.md](live-play.md).

To keep the .NET caches inside the repository, set before building:

```bash
export NUGET_PACKAGES=$PWD/.cache/nuget DOTNET_CLI_HOME=$PWD/.cache/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
```

## 2. Directories

| Directory | Role | Main entry points |
|---|---|---|
| `src/sims/` | Simulator (iRyougi/sts2-sim v0.113.1.1, patched) and the patch file | see [simulator.md](simulator.md) |
| `src/s0/` | Simulator extensions for search | `S0.Sim/` library, `S0Tests` runner, `py/` |
| `src/s1/` | Macro warm start on human runs (Defect phase) | `train_macro_v2.py`, `run_*.sh` |
| `src/value_v1/` | Run reconstruction and base trainer imported by s1 and s3 | `build_floor_dataset.py`, `train_value.py` |
| `src/s2/` | Combat search, combat trainer, battle-start tools | `S2.dll play`, `py/train_combat.py` |
| `src/s3/` | Full-run agent, macro network and trainer, inference server, chain scripts | `S3.dll play`, `run_s3.sh`, `s3_supervise.sh`, `eval_s3.sh` |
| `src/ic113/` | Port to v0.113 and Ironclad, warm-up, s3ic chain hooks and tests | `s3ic_launch.sh`, `s3ic_hook.sh`, `potion_mix.py`, `chain_stats.py`, `convert_v113.py` |
| `src/live/` | Game mod and controller | see [live-play.md](live-play.md) |
| `src/pipeline/` | Spire Codex run selection | `scan_runs.py`, `sample_runs.py`, `verify_sample.py` |
| `configs/` | s3ic chain settings at segment 29 | `s3ic_config.env`, `s3ic_extra.env` |
| `checkpoints/` | Release manifest and checksums | `manifest.json`, `SHA256SUMS` |
| `results/s3ic/` | Evaluation curve and reports | `eval_curve.csv`, `eval_report.txt` |

## 3. src/s0: simulator extensions

| File | Content |
|---|---|
| `S0.Sim/ResamplingClone.cs` | `Exact(state)` and `Resampled(state, seed)`: clone a fight, reseed every random stream, reshuffle the unknown part of the draw pile |
| `S0.Sim/SearchCardSelection.cs` | Card prompts inside clones (script, policy, suspend/replay, pause/resolve) |
| `S0.Sim/ActionSpace.cs`, `CombatActions.cs` | Legal actions (cards, targets, potions, prompt answers) and the observable-state hash |
| `S0.Sim/CombatObservation.cs`, `ObsLayout.cs`, `HiddenValues.cs` | Combat observation encoder (layouts v3–v5), vocabulary |
| `S0.Sim/Bridge.cs` | Shared-memory client to the Python inference server |
| `S0.Sim/BattleStart.cs` | Rebuild a fight from a human run's floor (`BattleStartLoader`) |
| `S0Tests/` | `S0Tests.dll resample | choices | import | importfirst | actions | obsv4 | obsv5 | leak113 | encode | bridge | m1` |
| `py/combat_obs.py` | Python mirror of the observation layout |
| `py/bridge_server.py`, `py/test_roundtrip.py` | Bridge test server and encoder round trip |
| `export_battle_starts.py`, `make_tf_tables.py`, `analyze_m1.py` | Defect-phase battle-start export, T_f tables, an analysis |
| `data/combat_vocab_v113.json` | Combat vocabulary of v0.113 (the default); `combat_vocab_v1.json` is v0.111 |

Example checks (CPU, no data needed):

```bash
T=src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll
dotnet $T resample 60 10000 L 3000      # clone, resample and encoder checks
dotnet $T choices 10 10 4 B 3000        # card prompts inside clones
```

## 4. src/s2: combat search and trainer

| File | Content |
|---|---|
| `S2/NetSearch.cs` | Gumbel search: root Gumbel-top-k + sequential halving, chance nodes, transpositions, `win_bound` |
| `S2/Evaluator.cs` | Bridge evaluator (batches rows to the server), scheduler for many fights, `FightSignals` (T_f context) |
| `S2/Fights.cs` | Fight runner, training chunk writer (`*.s2d`) |
| `S2/Program.cs` | `S2.dll play key=value ...`: fight self-play or evaluation on battle starts (keys at the top of the file) |
| `py/train_combat.py` | Combat trainer (replay window, policy and outcome losses, carries, `--wd-groups`, shadow options) |
| `py/infer_server.py`, `py/model.py`, `py/combat_heldout.py`, `py/bench_forward.py` | Fight-only server, checkpoint helper, held-out loss comparison, benchmark |
| `export_starts.py` | Battle starts from the Spire Codex dump (`--character IRONCLAD --build ... --standard --macro-ctx`) |
| `run_selfplay.sh`, `eval_ckpt.sh` | Fight self-play and checkpoint evaluation on battle starts (`STARTS`, `TABLES`) |
| other `*.py` | Analysis and reports of the Defect phase |

## 5. src/s3: full runs

| File | Content |
|---|---|
| `S3/RunAgent.cs` | The run agent: macro decisions, T_f tables, fight search, potion_mix, deck routing, practice fights, records |
| `S3/MacroLayout.cs`, `MacroBridge.cs` | Macro row layout (v3) shared with Python; macro shared-memory client |
| `S3/PotionMix.cs`, `PostCombatHeal.cs` | potion_mix scoring; post-combat heal model for T_f |
| `S3/Records.cs` | Run records and macro chunks (`*.s3r`) |
| `S3/HumanTables.cs` | `S3.dll tables`: T_f tables of human battle starts (combat warm-up) |
| `S3/PortChecks.cs`, `EncodeTest.cs`, `PromptTest.cs` | Port checks, encoder round-trip data, end-turn prompt test |
| `S3/Program.cs` | `S3.dll play | encode | props | vocab | cardinfo | prompttest | tftest | healtest | portcheck | tables` (all keys documented at the top) |
| `py/s3_server.py` | Inference server: `--serve combat` or `--serve macro` (defaults: v113 vocabularies) |
| `py/macro_s3.py` | Macro network (`S3MacroNet`), ensemble file format, row layout, milestone blend |
| `py/train_macro_s3.py` | Macro trainer (PPO for member 0, value for all members, milestones, carries, offline `--replay`) |
| `py/combat_pipe.py`, `py/macro_graphs.py` | CUDA-graph batching for the combat and macro servers |
| `py/roundtrip_macro.py`, `py/layout2_check.py`, `py/layout3_check.py`, `py/mgraphs_check.py` | Layout and server checks |
| `run_s3.sh` | One training segment: actors, servers, trainers (environment variables in its header) |
| `s3_supervise.sh`, `s3_resume.sh` | Chain supervisor and its cron restarter (interruption handling in the headers) |
| `eval_s3.sh` | Evaluation on fixed seeds |
| `carry_backup.sh`, `restore_from_nextcloud.sh` | Carry upload to the backup remote; rebuild a chain from it |
| `pause_chain.sh`, `resume_chain.sh`, `stop_segment.sh`, `switch_combat.sh`, `shadow_*.sh` | Chain control and the shadow-network switch |
| `mps.sh` | CUDA MPS daemon control (`start`, `status`, `stop`) |
| `panel/` | Watch panel for a chain: `panel/run_panel.sh [port]` serves a local page from synced logs |
| `data/` | `macro_vocab_v113.json`, `content_props_v113.json`, `card_info_v113.json`, `sim_ids_v113.json` (v0.113, defaults) and the v0.111 versions |
| other `*.py`, `*.sh` | Analysis and experiment scripts, mostly from the Defect phase |

## 6. src/ic113: this phase's port and chain

See [src/ic113/README.md](../src/ic113/README.md) for the file list. The scripts used by a running chain are
`s3ic_hook.sh`, `chain_stats.py` and `potion_mix.py`.

## 7. Common commands

```bash
# evaluate the release on the chain's panel (see doc/checkpoints.md for the full command)
COUNT=300 FIRST=100000 SEEDS=Y ASC=10 EVAL_ACTOR_ARGS="..." src/s3/eval_s3.sh COMBAT.pt MACRO.pt OUTDIR 20 "tf64:64"

# one training segment by hand: name, actors, seconds
COMBAT_INIT=COMBAT.pt MACRO_INIT=MACRO.pt ASC=10 CONC=32 src/s3/run_s3.sh NAME 60 7200

# a chain: see doc/training.md section 8
src/ic113/s3ic_launch.sh 168

# summary of run logs with Wilson intervals (eval_s3.sh already writes OUTDIR/summary.txt this way);
# two or more arms on the same seeds also get paired McNemar tests
./.venv/bin/python -I src/s3/summarize_runs.py OUTDIR/*.jsonl
```

## 8. What the scripts expect

The repository has code and small data files only. Large or derived files are not included:

| Artifact | Made by | Used by | Included? |
|---|---|---|---|
| Release weights (`ic113_s029_*.pt`, `ic113_s029_vbar.json`) | the s3ic chain | evaluation, live play, a new chain | GitHub release |
| Combat and macro vocabularies, content properties, card info (v0.113) | `S3.dll vocab`, `props`, `cardinfo`; `convert_v113.py` | actors, servers, trainers | yes (`src/s0/data`, `src/s3/data`) |
| Chain directory `src/s3/runs/s3ic/` and segments `s3ic_sNNN/` | supervisor | everything of a chain | no (created at launch) |
| Chunks `*.s2d`, `*.s3r` | actors | trainers | no |
| Carries `carry_combat.bin`, `carry_macro.pt`, `carry_shadow.bin` | trainers | the next segment | no (several GB each) |
| Spire Codex dump and per-version exports | download (use confirmed with the maintainer) | `src/pipeline`, `src/s1`, `src/s2/export_starts.py`, `src/ic113/fetch_human.sh` | no |
| Battle starts and T_f tables of the warm-up | `export_starts.py`, `warmup_tables.sh` | `warmup_run.sh` | no |
| S1 macro ensemble, converted Defect networks | Defect phase; `convert_v113.py` | the original chain start | no |

Defaults to know:

- `run_s3.sh` falls back to `src/s3/weights/combat_s2p1.pt` and `src/s3/data/macro_s1.pt` when `COMBAT_INIT` and
  `MACRO_INIT` are not set. These files are not in the repository; always pass the paths.
- `eval_s3.sh` reads the observation layout from the combat checkpoint (`config.obs`, 5 for the release); `OBS`
  overrides it.
- `s3_server.py` refuses a macro file whose vocabulary differs from `--mvocab` (default the v113 file), so a v0.111
  file cannot be served by mistake.
- Some script headers still name files of the private research tree (`src/ic113/CHAIN.md`, `PLAN.md` and others) or
  its server paths. The content of those notes is in this `doc/` folder.

## 9. Pitfalls

- The bridge uses `/dev/shm`. A container with a small `/dev/shm` fails; actors and servers give up after 300 s
  without an answer.
- One Python combat server saturates at roughly 27k states per second; use 2–3 for 60 actors.
- `MPS=1` needs `nvidia-cuda-mps-control`. `src/s3/mps.sh` leaves the daemon running between segments; stop it only
  while no segment runs.
- Never rebuild `S3.dll` in place while a chain's actors run from it. Build into `src/s3/S3/bin_next`; the hook swaps it
  in between segments.
- Do not use `exact=1` in the search for training or play: it sees the true future (it exists only as an upper-bound
  test arm).
