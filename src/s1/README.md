# S1: macro network (human-data warm start)

The macro network predicts the run outcome V_run = P(win) from the state after a floor and a policy over the
out-of-combat choices. S1 is the warm start; S3 retrains the same network on the agent's own runs.

Code origin: the working tree's `value_v2` stage. The directory is named `s1` here; `src/value_v2` is a
compatibility symlink for the original imports. The shared reconstruction and base trainer live in
`src/value_v1`.

| File | What |
|---|---|
| `build_dataset_v2.py` | Per-floor dataset from all clean runs at natural frequencies; leak guards and run selection from `src/value_v1/build_floor_dataset.py` and `src/pipeline/sample_runs.py` |
| `build_decisions_v2.py` | Decision dataset for the policy head: card reward, relic pick, ancient, rest, smith target, removal target, shop |
| `train_value_v2.py` | Value trainer: uniform floors, state-only inputs, HP-monotone head, natural frequencies, no-attention or transformer encoder |
| `train_value_v2_r1.py` | Frozen pre-speedup version used for the scaling curve |
| `train_macro_v2.py` | `MacroNet`: shared no-attention encoder + HP-monotone value head + policy head |
| `s1_finalize.py` | Ensemble calibration, dominance checks, T_f builder tests, G1 report |
| `summarize_scaling.py` | Scaling-curve table |
| `run_*.sh` | Data build / scaling / ensemble / finalize drivers (paths and `PY` via environment) |

## Model

- Entity-set encoder over deck cards (id, upgrade, enchantment and amount, age), relics (id, age), potions, context
  (character, ascension, era, act, room, encounter) and 16 scalars. The no-attention variant is a stack of per-token
  residual MLPs followed by masked mean pooling and a 512-256-128 trunk. It won or tied at every data size and trains
  about twice as fast as the set transformer.
- HP is removed from the encoder input. The value logit is
  `a(z) + sum_k softplus(b_k(z)) * phi_k(hp, max_hp)` with 26 fixed increasing ramps, so V cannot fall when HP rises.
- Ensemble of five seeds; T_f uses the mean logit (the raw ensemble was best calibrated; temperature scaling was
  dropped). The policy head scores each offered option with `[q, e, q*e]` and deck slots with the deck token states.
- The S1 policy does not see HP. S3 adds the HP ramps to the policy query through a zero-initialised layer.

## Results (G1)

5 members, test split of the 965,021 training-run dataset; target cell Defect A10 v0.111.0:

| Metric | Value |
|---|---|
| Ensemble test log loss | 0.4655 |
| AUC / within-floor AUC | 0.8365 / 0.8002 |
| ECE (all floor bands) | 0.0038 |
| Target cell log loss / within-floor AUC / ECE | 0.4301 / 0.7814 / 0.019 |
| HP violations (+10 HP lowers V by >0.5 pt) | 0 / 39,850 |
| +1 potion wrong by >0.5 pt | 2.82% (accepted; T_f fixes potion prices in S3) |
| Card-reward top-1 vs human | 53.3% (pick-rate baseline 40.6%) |
| Ancient / removal / rest / relic / smith / shop top-1 | 82.6 / 73.2 / 67.2 / 54.9 / 47.0 / 34.8% |

Data-scaling curve (test log loss, full data): no-attention 0.4604 / within-floor 0.8054; transformer 0.4628 / 0.8036.
At 25k-300k runs the no-attention model also wins or ties. Full table in `doc/data/s1_scaling.csv`.

## Commands

```bash
# D points at your runs-latest.jsonl.gz + meta_all.csv.gz (Spire Codex)
D=data/spire_codex F=runs/floors_v2 src/s1/run_s1_scaling.sh
F=runs/floors_v2 src/s1/run_macro_full.sh
F=runs/floors_v2 src/s1/run_macro_ensemble.sh
F=runs/floors_v2 src/s1/run_s1_finalize.sh
```

`PY` defaults to `./.venv/bin/python`; the scripts run from the repository root. The resulting five members are
packaged as one ensemble checkpoint for S3.
