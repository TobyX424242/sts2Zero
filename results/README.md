# Result data

Small, plain data files behind [doc/results.md](../doc/results.md) and [doc/experiments.md](../doc/experiments.md).
All of them were extracted from the chain records, the evaluation outputs and the reports of the ic113 phase
(Ironclad A10 on game v0.113.1). Times are UTC. Win rates are percentages. "CI" is a 95% interval (Wilson for a
single rate). Encounter names are the simulator's ids without the `ENCOUNTER.` prefix.

The raw results of the offline tests T1-T3 are JSON files next to their scripts in
[`src/ic113/laptop/`](../src/ic113/laptop/) (`t1_results.json`, `t2_results.json`, `t3_results.json`); the CSV files
in `laptop/` here are summaries of them. T4 has no raw file in the repository; its CSV files hold the numbers of its
report.

## Files

| File | What it holds | Source |
|---|---|---|
| `s3ic/eval_curve.csv` | One row per evaluated segment: the learning curve | chain `curve.txt` |
| `s3ic/eval_report.txt` | Per-segment evaluation report: win rate, boss funnel, HP entering bosses, stalemates, deaths, potion-rule counts | chain `eval_report.txt` |
| `s3ic/eval_s029.txt` | Summary of the segment-29 evaluation | segment-29 evaluation `summary.txt` |
| `s3ic/funnel.csv` | The boss funnel of every evaluated segment, as columns | parsed from `eval_report.txt` |
| `s3ic/segments.csv` | Timeline of every segment number 0-29: times, actors, runs, status, setting changes | chain supervisor log and segment logs |
| `s3ic/curriculum.csv` | Self-play win rates per segment by ascension group and the curriculum decision | chain `curriculum.log` (last reading per segment) |
| `s3ic/combat_trainer.csv` | End-of-segment training-batch metrics of the playing combat trainer, and of the shadow trainer in segment 29 | per-segment `train_log.jsonl`, `shadow_train_log.jsonl` |
| `s3ic/deaths_s029.csv` | Every cause of death in the segment-29 evaluation | segment-29 per-run and per-fight records |
| `s3ic/fights_s029.csv` | Fight statistics of the segment-29 evaluation by room type and act | segment-29 per-fight records |
| `s3ic/bosses_s029.csv` | Boss fights of the segment-29 evaluation by boss slot and encounter | segment-29 per-fight records |
| `ic113/full_runs.csv` | Full-run results before the chain: port sanity, untrained Ironclad, after the combat warm-up | day-0 and warm-up reports |
| `ic113/warmup_harness.csv` | Combat warm-up fight harness by room and act, step 0 vs warmed | warm-up report |
| `laptop/t1_after_boss1.csv` | T1: V_run calibration after act-3 boss 1 by HP fifth, three data sets | T1 report / `t1_results.json` |
| `laptop/t2_arms.csv` | T2: held-out metrics of the replay arms | `t2_results.json` |
| `laptop/t3_boss_search.csv` | T3: paired boss-search comparisons by slice | `t3_results.json` |
| `laptop/t4_paired_eval.csv` | T4: 4-arm paired A10 evaluation | T4 report |
| `laptop/t4_potion_probe.csv` | T4: the potions at both ends of the measured potion table | T4 report |
| `live/live_runs.csv` | Finished runs in the real game | live controller logs |

## Column dictionary

### `s3ic/eval_curve.csv`

| Column | Meaning |
|---|---|
| `segment` | Segment number (gaps: segments without usable weights, see `segments.csv`) |
| `self_play_runs` | The chain's run counter after the segment (counts run records that reached backup storage; it was rebased at the restore before segment 18) |
| `eval_runs` | Evaluation runs (300; 304 for segments 22-26) |
| `errors` | Evaluation runs that ended in an error (stalemates and others) |
| `wins` | Runs won |
| `win_rate`, `ci95_low`, `ci95_high` | Win rate and CI with errored runs left out of the denominator |
| `mean_floor` | Mean floor reached |
| `act1_cleared`, `act2_cleared` | Percent of runs that cleared act 1 / act 2 |

### `s3ic/funnel.csv`

| Column | Meaning |
|---|---|
| `segment`, `eval_runs`, `wins`, `errors`, `mean_floor` | As above |
| `win_rate`, `ci95_low`, `ci95_high` | Win rate and CI with errored runs counted as losses (differs from `eval_curve.csv` by at most 0.4 pp) |
| `stalemates` | Evaluation runs stopped at round 151 of a fight |
| `<boss>_entered`, `<boss>_won` | Runs that entered / won the act-1 boss, act-2 boss, first act-3 boss and second act-3 boss |
| `<boss>_hp_share` | Mean HP / max HP when entering that boss fight |
| `<boss>_potions` | Mean potions held when entering that boss fight |

### `s3ic/segments.csv`

| Column | Meaning |
|---|---|
| `segment` | Segment number |
| `start_utc`, `end_utc` | Start and end (for interrupted segments: the last saved weights or heartbeat, to the minute) |
| `actors` | Self-play actor processes |
| `planned_length_h` | Planned segment length in hours |
| `segment_runs` | Runs the segment added to the run counter (empty if unknown) |
| `status` | `completed`; `recovered (...)` = adopted after a stop, with its saved weights and carries; `not adopted`; `lost with the host`; `failed` |
| `evaluated` | Whether the segment has an evaluation on the panel |
| `cumulative_runs_after` | The run counter after the segment |
| `changes_and_notes` | Settings that changed and other notes |

### `s3ic/curriculum.csv`

| Column | Meaning |
|---|---|
| `segment` | Segment whose self-play runs were read |
| `reading_time` | When the curriculum rule ran |
| `a10_*`, `a0_a4_*`, `a5_a9_*`, `all_*` | Wins, runs and win rate per ascension group (runs with perturbed fight starts left out; errors count as losses; empty rate = no runs) |
| `errors`, `stalemates`, `stalemates_by_encounter` | Self-play runs that ended in an error; stalemates by encounter |
| `stage_decision` | The rule's output: `STAGE a -> b (reason)` or `stage stays` |

### `s3ic/combat_trainer.csv`

| Column | Meaning |
|---|---|
| `segment` | Adopted segment |
| `network` | `playing` (serves the actors) or `shadow` (trains beside it, segment 29 on) |
| `steps`, `decisions_ingested` | Trainer steps and new decisions read in that segment |
| `replay_window`, `lr` | Replay window size (decisions) and learning rate at the end |
| `policy_kl`, `policy_top1` | KL from the search's policy target to the network's policy, and top-1 agreement (training batches) |
| `end_hp_loss`, `survive_loss`, `hp_mae` | End-HP distribution loss, survival loss and end-HP mean absolute error (fraction of max HP) |
| `grad_norm_preclip` | Gradient norm before clipping at 1.0 |

All metrics are the last log line of the segment and come from training batches, not held-out data.

### `s3ic/deaths_s029.csv`, `fights_s029.csv`, `bosses_s029.csv`

| Column | Meaning |
|---|---|
| `encounter`, `room_type` | Encounter id; `monster` (hallway), `elite` or `boss` |
| `act` | Act of the fight, counted by bosses beaten before it in the same run (act 3 includes both act-3 bosses) |
| `deaths`, `share_of_deaths` | Runs lost in that fight; percent of the 148 losses |
| `fights`, `won`, `win_rate` | Fights fought and won |
| `mean_rel_tf` | Mean of the fight's T_f divided by its table maximum |
| `mean_hp_lost` | Mean of start HP minus end HP (0 if the fight was lost) |
| `mean_potions_used` | Mean potions used per fight |
| `boss_slot` | `act-1 boss`, `act-2 boss`, `act-3 boss 1`, `act-3 boss 2` (the order of boss fights in the run) |
| `mean_hp_share_entering` | Mean start HP / max HP |

### `ic113/full_runs.csv`

| Column | Meaning |
|---|---|
| `stage` | `day0_port_sanity`, `day0_step0` (untrained Ironclad) or `warmup` (after the combat warm-up) |
| `arm` | Networks used |
| `character`, `ascension`, `seeds` | Setting and seed range |
| `runs_counted`, `wins`, `win_rate`, `ci95_low`, `ci95_high` | Runs in the rate and the result |
| `mean_floor`, `act1_cleared`, `act2_cleared` | As above |
| `note` | Comparison or caveat |

### `ic113/warmup_harness.csv`

| Column | Meaning |
|---|---|
| `room`, `act`, `starts` | Held-out human battle starts (elite or boss) by act |
| `step0_win`, `warmed_win` | Win rate before and after the warm-up, search 64, paired by start |
| `diff_pp`, `diff_ci95_low`, `diff_ci95_high` | Paired difference and CI |
| `step0_only`, `warmed_only` | Starts won by only one of the two networks |
| `step0_mean_end_hp`, `warmed_mean_end_hp` | Mean end HP (0 for a loss) |
| `human_survived` | Share of these starts the human player survived (one logged attempt with the human's own draws) |

### `laptop/t1_after_boss1.csv`

| Column | Meaning |
|---|---|
| `dataset` | `fresh_s006_s008` (runs of segments 6-8 scored before training on them), `inwindow_s008_carry` (the segment-8 training window), `heldout_s011` (later runs of segment 11) |
| `hp_fifth` | HP / max HP after act-3 boss 1, in fifths |
| `n`, `actual_win`, `actual_ci95_low`, `actual_ci95_high` | Runs and their actual win rate |
| `pred_prob_avg`, `gap_prob_avg_pp` | Predicted win rate (probability average of the 5 members) and predicted minus actual |
| `pred_logit_avg` | Predicted win rate from the logit average (what T_f uses) |
| `pred_member0`, `pred_members1_4` | Predictions of member 0 and of members 1-4 |

### `laptop/t2_arms.csv`

| Column | Meaning |
|---|---|
| `arm` | `base` (start network), `m0` control, `m1` milestones, `m2` milestones + P2, `m3` look-distill |
| `heldout_ll_prob`, `heldout_ll_logit` | Held-out value log loss (nats), probability average / logit average |
| `gain_over_table` | Held-out log-loss gain over a 9-cell (ascension group x act) table |
| `train_test_gap` | Held-out minus training log loss |
| `heldout_a10_*`, `cross_seg_ll_prob` | The same on A10 rows / on the segment-7 subset |
| `res_auc_*` | Residual AUC within (floor, HP band): all wins, A10 next milestone, A10 wins |
| `after_boss1_*` | After act-3 boss 1 at A10: runs, actual win rate, predictions, and the top-minus-bottom HP fifth spread |
| `card_dv_logit_sd` | Spread across members of one-card value differences on 1,500 card-reward decisions |
| `agree_raw_vs_own_look`, `agree_raw_vs_fixed_target` | Share of card decisions where the raw policy's choice equals the lookahead choice |

### `laptop/t3_boss_search.csv`

| Column | Meaning |
|---|---|
| `comparison` | `s128_vs_s64`, `s256_vs_s64`, `s256_vs_s128` (simulations per move) |
| `slice` | All act-3 bosses, boss 1, boss 2 (final), or one encounter (`enc:`, `boss1:`, `boss2:`) |
| `paired_fights` | Fights in the slice (each start played twice with fixed chance seeds) |
| `base_win`, `arm_win`, `diff_win_pp`, `diff_ci95_*` | Win rates of the two arms and the paired difference |
| `arm_only_wins`, `base_only_wins`, `mcnemar_p` | Discordant fights and McNemar p-value |
| `diff_end_hp*` | Paired difference in end HP and its CI |
| `diff_tf_score` | Paired difference in T_f score |
| `base_sec_per_fight`, `arm_sec_per_fight` | Wall time per fight (concurrent harness) |

### `laptop/t4_paired_eval.csv`

| Column | Meaning |
|---|---|
| `arm`, `description` | The tested setting |
| `runs`, `wins`, `win_rate`, `ci95_*` | Full A10 runs on seeds `Z300000`-`Z300319` |
| `diff_vs_base_pp`, `arm_only_wins`, `base_only_wins`, `mcnemar_p` | Paired comparison with `base` |
| `act1_cleared`, `act2_cleared` | Percent of runs |
| `act3_boss1_won`, `act3_boss1_entered` | First act-3 boss |
| `act3_boss2_entered_hp_share`, `act3_boss2_entered_potions` | HP share and potions entering the final boss |
| `act3_boss2_won`, `act3_boss2_entered` | Final boss |

### `laptop/t4_potion_probe.csv`

| Column | Meaning |
|---|---|
| `potion` | Potion id |
| `t_win_points` | 100 x (win rate holding only this potion - win rate with an empty belt) on 80 paired final-boss starts; standard error about 5-6 points |
| `note` | `top` or `bottom` of the ranking; the report lists only these ends |

### `live/live_runs.csv`

| Column | Meaning |
|---|---|
| `date_utc` | Day the run was played |
| `weights_segment` | Chain segment of the weights |
| `ascension`, `seed` | Game setting and the game's seed |
| `outcome` | `won` or `lost` (won = the game's ending scene after the final boss was reached) |
| `final_floor`, `last_boss` | Floor and encounter of the last fight |
| `hp_after_last_boss`, `max_hp` | HP after the last fight |
