# s3: full-game self-play

Whole runs in the simulator: the run agent, the macro network and its trainer, the inference server, and the scripts
that run a self-play chain. Design: [doc/architecture.md](../../doc/architecture.md). Training and the chain:
[doc/training.md](../../doc/training.md).

| Path | Content |
|---|---|
| `S3/RunAgent.cs` | The run agent: every macro decision, the T_f table at each fight start, the fight search, potion_mix, deck routing, Fake Merchant, practice fights, perturbed starts, records |
| `S3/MacroLayout.cs`, `S3/MacroBridge.cs` | Macro row layout v3 (mirrored in `py/macro_s3.py`), macro encoder, shared-memory client |
| `S3/PotionMix.cs`, `S3/PostCombatHeal.cs` | potion_mix scoring; post-combat heal model used by T_f |
| `S3/Records.cs` | Run records and macro chunks (`*.s3r`) |
| `S3/ContentProps.cs`, `S3/HumanTables.cs`, `S3/PortChecks.cs`, `S3/EncodeTest.cs`, `S3/PromptTest.cs` | Content properties, warm-up tables, v0.113 port checks, encoder round-trip data, prompt test |
| `S3/Program.cs` | `S3.dll play | encode | props | vocab | cardinfo | prompttest | tftest | healtest | portcheck | tables`; every `play` key is documented at the top |
| `py/s3_server.py` | Inference server, `--serve combat` or `--serve macro`; CUDA graphs (`--graphs`, `--mgraphs`) |
| `py/macro_s3.py` | Macro network `S3MacroNet`, ensemble file, row layout, milestone blend |
| `py/train_macro_s3.py` | Macro trainer: PPO for member 0, value for all members, milestone heads, carries, offline `--replay` |
| `py/combat_pipe.py`, `py/macro_graphs.py` | CUDA-graph batching for the two servers |
| `py/*_check.py`, `py/roundtrip_macro.py` | Layout, server and encoder checks |
| `run_s3.sh` | One segment: actors, servers, trainers, optional shadow trainer (environment variables in the header) |
| `s3_supervise.sh`, `s3_resume.sh` | The chain supervisor (guard, validation, adoption after interruptions, hook, evaluation, backups) and its cron restarter |
| `eval_s3.sh`, `summarize_runs.py` | Evaluation on fixed seeds; summary with Wilson intervals and paired tests |
| `carry_backup.sh`, `restore_from_nextcloud.sh`, `backup_loop.sh` | Uploads to the backup remote and a restore into a fresh tree |
| `pause_chain.sh`, `resume_chain.sh`, `stop_segment.sh`, `switch_combat.sh`, `shadow_*.sh`, `mps.sh` | Chain control, shadow-network switch, CUDA MPS daemon |
| `panel/` | Watch panel: `panel/run_panel.sh [port]` serves a local page from logs synced from the backup remote |
| `data/` | Macro vocabulary, content properties, card info and simulator ids for v0.113 (`*_v113.json`, the defaults) and v0.111 |
| other `*.py`, `*.sh` | Analysis and experiment scripts, mostly from the Defect phase |

```bash
dotnet build src/s3/S3/S3.csproj -c Release            # also builds S2, S0 and the simulator
src/s3/eval_s3.sh COMBAT.pt MACRO.pt OUTDIR 20 "tf64:64"   # see doc/checkpoints.md for the full arguments
src/s3/run_s3.sh NAME 60 7200                           # one segment; COMBAT_INIT / MACRO_INIT required in practice
```

A chain keeps its state in `src/s3/runs/<chain>/` (settings, curve, logs) and one directory per segment,
`src/s3/runs/<chain>_sNNN/` (weights, carries, run and fight logs, chunks).
