# s1: macro warm start on human runs

The first version of the macro network, trained on human Spire Codex runs in the Defect phase (v0.111.0). It is kept
because the current macro network descends from it and because `src/s3/py/macro_s3.py` imports its model classes
(`train_value_v2.py`, `train_macro_v2.py`). This phase did not retrain it: the Ironclad chain started from the
converted Defect self-play network, and no human-data refresh was run.

| File | Content |
|---|---|
| `build_dataset_v2.py` | Per-floor dataset from all clean runs (uses `src/value_v1` and `src/pipeline`) |
| `build_decisions_v2.py` | Decision dataset for the policy head: card reward, relic, ancient, rest, upgrade target, removal target, shop |
| `train_value_v2.py` | Value trainer and model: per-token MLP encoder, mean pooling, HP-monotone value head |
| `train_value_v2_r1.py` | Frozen earlier version used for the scaling curve |
| `train_macro_v2.py` | `MacroNet`: shared encoder + value head + policy head |
| `s1_finalize.py` | Ensemble calibration, dominance checks, T_f builder tests, the warm-start report |
| `summarize_scaling.py` | Scaling-curve table |
| `run_*.sh` | Dataset, scaling, ensemble and finalize drivers |

Results of the Defect phase (1.2M clean human runs, 965k for training; ensemble test log loss 0.4655, within-floor
AUC 0.800) are documented on branch `v0.111-defect`. The human data is not part of this repository.
