# S2: combat network + network-guided search (PROPOSAL.md §6.4)

Builds on `research/s0` (iRyougi + S0 patch, resampling clone, action space, encoder, bridge). Observation layout v3
(`research/s0/S0.Sim/ObsLayout.cs`, mirrored in `research/s0/py/combat_obs.py`).

| Path | Content |
|---|---|
| `S2/NetSearch.cs` | Gumbel MuZero search: root Gumbel-top-k + sequential halving, improved-policy target; nodes keyed by `ObservableState.Hash` (chance outcomes and transpositions); network value at new leaves, T_f / T_max at fight end. Arms: `OneTurn` (G2 b), `ExactWorlds` (G2 c) |
| `S2/Evaluator.cs` | `IEvaluator`; `BridgeEvaluator` (encodes into the shared-memory rows, flushes as one batch); `Pump` (single-thread scheduler for many fights); `FightSignals` (macro-network signals, below) |
| `S2/Fights.cs` | Fight runner (playout cap randomisation, records), `ChunkWriter` (training chunks `*.s2d`), timing counters |
| `S2/Program.cs` | `S2 play key=value ...` (self-play or eval; keys documented at the top) |
| `py/infer_server.py` | One GPU server for many actors; bucketed batch and token sizes; reloads weights on change |
| `py/train_combat.py` | Trainer: replay window, policy CE (fully searched moves), outcome-head losses (end-HP bins with HL-Gauss targets, potions kept) or the scalar head (`--head scalar`); writes versioned weights |
| `py/model.py`, `py/bench_forward.py` | Checkpoint helper; forward-pass benchmark |
| `export_starts.py` | Battle-start pool from the full dump (Defect A10 v0.111.0), held-out 5% of runs |
| `make_tables.py` | Batched T_f tables for the pool (S1 macro ensemble; same construction as `s0/make_tf_tables.py`) |
| `run_selfplay.sh`, `eval_ckpt.sh`, `summarize_eval.py`, `bench_actors.sh` | Run scripts |
| `data/` | `starts_*_{train,eval}.jsonl`, `tables_*_{train,eval}.jsonl` (run-derived; **not included** in this repository) |
| `runs/<name>/` | weights.pt (current), ckpt/, data/ (chunks), fights/ (per-fight logs), logs/ (created at run time) |

This repository contains the code only. The battle-start pool, T_f tables and all checkpoints are derived from
Spire Codex runs and are not redistributed; see `doc/experiments.md` for how they were built and `doc/code-map.md`
for the paths the scripts expect.

## How the macro network guides the combat network (PROPOSAL.md §4.2, D13)

1. **Fight end:** a finished fight is scored T_f(HP, potions held) = V_run of the post-fight state (S1 ensemble), table per
   battle start. Tables obey free disposal (TfScorer raises an entry to the best of its potion subsets at load).
2. **Leaves (outcome head, default):** the network predicts the end-HP distribution (41 bins, 0 = death) and, per starting
   potion held, P(still held at the end); `FightSignals.OutcomeValue` prices this with the same table, relative to T_max.
   `--head scalar` (trainer) gives the first design, P(survive) + relative T_f, for comparison (A20).
3. **Inputs (layout v3):** Context token after the global token (room, act, floor, T_f at the current HP and potions, the
   price of 10 HP, the worth of the potions held, 1000 T_max, final boss); potion tokens' feature 1 = marginal T_f
   (−1 for a potion gained in the fight). All in thousandths of T_max.

Not yet covered: potions gained during the fight have no T_f (ignored at fight end), in-fight permanent changes (max HP,
added cards) are not priced, ties are not broken toward shorter fights.

Training chunks are version 2 (meta width 12: end-HP bin, held-slot masks at the decision and at the end, end HP / max HP).

## Commands

```bash
W=$(pwd)  # repository root; cd $W
export NUGET_PACKAGES=$W/.cache/nuget DOTNET_CLI_HOME=$W/.cache/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$W/.cache/tmp
dotnet build research/s2/S2/S2.csproj -c Release -m:8
research/s2/run_selfplay.sh NAME 16 3600                      # self-play training, 16 actors, 1 hour
research/s2/eval_ckpt.sh research/s2/runs/NAME/weights.pt research/s2/out/eval_NAME 8 "net:0 s16:16 s64:64 ot64:64:oneturn=1 ex64:64:exact=1"
```
