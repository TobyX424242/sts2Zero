# s2: combat network and search

The fight player: the Gumbel search guided by the combat network, the combat trainer, and tools that play battle starts
outside full runs (used for the combat warm-up and for fight-level tests). The full-run agent in `src/s3` calls the
same search. Design: [doc/architecture.md](../../doc/architecture.md) §3–5.

| Path | Content |
|---|---|
| `S2/NetSearch.cs` | Gumbel search: root Gumbel-top-k and sequential halving, completed-Q policy target, chance nodes and transpositions keyed by the observable state, leaves priced by T_f; `win_bound`; test arms `oneturn` and `exact` (the latter sees the true future: tests only) |
| `S2/Evaluator.cs` | `BridgeEvaluator` (batches rows to the inference server), `Pump` (one thread runs many fights), `FightSignals` (T_f context for the network) |
| `S2/Fights.cs` | Fight runner, training chunks (`*.s2d`), timing counters |
| `S2/Program.cs` | `S2.dll play key=value ...`: self-play or evaluation on battle starts; keys at the top of the file (incl. the warm-up's `mutate`, `mutate_pool`, `card_counts`) |
| `py/train_combat.py` | Combat trainer: replay window, policy cross-entropy on fully searched moves, end-HP bins (HL-Gauss) and potions kept, EMA, surprise weighting, carries, `--wd-groups matrices`, cosine schedule with `--lr-continue` |
| `py/combat_heldout.py` | Compares checkpoints' losses on the same held-out chunks (used for the shadow network) |
| `py/infer_server.py`, `py/model.py`, `py/bench_forward.py` | Fight-only inference server, checkpoint helper, forward benchmark |
| `export_starts.py` | Battle starts from Spire Codex runs: `--character IRONCLAD --build v0.111.0,v0.113.0,v0.113.1 --standard --macro-ctx`; 5% of runs held out by run hash |
| `make_tables.py` | T_f tables for battle starts from the S1 ensemble (Defect phase; the warm-up used `S3.dll tables` instead) |
| `run_selfplay.sh`, `eval_ckpt.sh`, `bench_actors.sh`, `search_params_test.sh` | Fight self-play, checkpoint evaluation on held-out starts, actor benchmark, search-parameter test |
| other `*.py` | Reports and diagnostics of the Defect phase |

```bash
dotnet build src/s2/S2/S2.csproj -c Release
# fight self-play on a battle-start pool: name, actors, seconds
STARTS=starts_train.jsonl TABLES=tables_train.jsonl src/s2/run_selfplay.sh NAME 16 3600
```

Battle-start pools and their tables are not in the repository (they come from the Spire Codex data).
