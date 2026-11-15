# Result data

Curated summaries of the result data.

| File | Content | Source |
|---|---|---|
| `s1_final.json` | S1 G1 report: members, calibration, dominance probes, T_f builder tests, policy accuracy | `s1_finalize.py` on the 5-member ensemble |
| `s1_scaling.csv` | S1 data-scaling curve (no-attention vs transformer at 25k-965k runs) | `train_value_v2.py` runs on the full dataset |
| `s3_curve.csv` | S3 evaluation at A10, one row per segment: self-play runs, 150/300 fixed seeds, wins, mean floor, act 1/2 | the continuous S3 run's evaluation lines |
| `ascension_sweep.csv` | S3 checkpoint s037 at A0-A10, 154 seeds per level | `eval_asc_sweep.sh` + `asc_sweep_report.py` |

Notes:

- `s3_curve.csv` has gaps where a segment had no evaluation (15, 31) and duplicate evaluation lines were collapsed to
  the last one for a segment.
- The S3 win percentages in the CSV are rounded as the evaluation printed them; confidence intervals are in
  `doc/results.md` for the selected points.
- The ascension sweep is from an earlier checkpoint (s037). The final curve point (segment 45) is 31.3% at A10, later
  than the sweep.
