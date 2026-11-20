# ic113: Ironclad A10 on game v0.113.1

The code of this phase: the port from v0.111 to v0.113, the combat warm-up, and the scripts the `s3ic` self-play chain
runs between segments. The process is described in [doc/training.md](../../doc/training.md). Some script headers refer
to notes of the private research tree (`CHAIN.md`, `PLAN.md`, `WARMUP.md` and others) and to its server paths; their
content is in `doc/`.

## Used by a running chain

| File | Role |
|---|---|
| `s3ic_launch.sh` | Starts the chain: copies the start weights to `src/s3/runs/s3ic_init/`, writes default settings if none exist, starts the supervisor |
| `s3ic_hook.sh` | Called by the supervisor between segments: swaps in a staged build, runs the curriculum, computes the potion table, writes `segment.env`; after each evaluation, writes the report |
| `chain_stats.py` | Curriculum stages and rule (`curriculum`, `stage-args`); per-segment evaluation report (`eval`) |
| `potion_mix.py` | Potion price table per segment (`vbar`), empty table (`empty`), check against actor logs (`check`), `show` |

## Port and checks

| File | Role |
|---|---|
| `convert_v113.py` | Converts checkpoints and vocabularies from v0.111 to v0.113 by id; `check-combat`, `check-macro` |
| `checks.sh` | Port checks on CPU: S0 tests and leak test, `prompttest` / `tftest` / `healtest`, `portcheck` runs / cards / events |
| `smoke.sh`, `s3ic_smoke.sh` | CPU smoke runs: a few full runs; the whole chain machinery with tiny settings |
| `day0.sh`, `eval_ic113.sh` | Day-0 anchor evaluations |
| `fetch_human.sh`, `run_classes.py`, `import_starts.sh`, `import_report.py` | Human data per version, run classification, battle-start import test |

## Combat warm-up

| File | Role |
|---|---|
| `warmup_tables.sh` | T_f tables of human battle starts (`S3.dll tables`) |
| `warmup_pool.py` | Card pool for coverage mutation; held-out evaluation sets; card counts |
| `warmup_run.sh`, `warmup_watch.sh`, `warmup_memguard.sh` | The warm-up run, its watchdog and a guard against hung actors |
| `warmup_eval.sh`, `warmup_fullruns.sh`, `warmup_final.sh`, `warmup_report.py` | Fight harness and full-run evaluations, reports |

## Tests and investigations

| File | Role |
|---|---|
| `stalemate_rep.sh`, `stalemate_summary.py` | Replays of stalemated fights (led to `win_bound`) |
| `drtest_act.sh`, `drtest_cmp.py` | Identity and smoke tests of `deck_route` and `lethal_backstop` |
| `potion_probe.sh`, `potion_probe.py` | Paired final-boss potion probe for potion_mix phase 2 |
| `shadow_eval.sh` | Paired evaluation of the shadow combat network against the playing one (report only) |
| `laptop/` | Offline single-GPU tests T1–T4: value calibration (`t1_*`), offline macro replay of trainer variants (`t2_*`), act-3 boss search budget (`t3_*`), value members 1–4 and potion_mix phase 2 (`t4_*`); `*_results.json` hold the results |
| `clean_selfplay.sh`, `s3ic_setup.sh` | Housekeeping and building a chain tree on the original servers (server-specific) |

Results of these tests are summarised in [doc/experiments.md](../../doc/experiments.md).
