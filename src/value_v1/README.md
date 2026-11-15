# Shared S1 base modules

Only two files from the original `value_v1` stage are kept, because S1 and S3 import them:

- `build_floor_dataset.py`: the per-run reconstruction of decks, relics, potions and scalars from Spire Codex logs,
  including the leak guards and the sanity filter. It is used by the S1 dataset builder and by
  `src/s3/py/roundtrip_macro.py`.
- `train_value.py`: the base `ValueNet` class, data pipeline and training utilities that
  `src/s1/train_value_v2.py` subclasses.

The early POC (the trial-1 set transformer, its probes, AlphaDefect transfer experiments and checkpoints) is **not**
part of this repository. It is summarised in one paragraph in `doc/experiments.md`. The files here are plumbing, not
the POC result.
