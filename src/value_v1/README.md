# Shared S1 base modules

Only two files from the original `value_v1` stage are kept, because S1 and S3 import them:

- `build_floor_dataset.py`: the per-run reconstruction of decks, relics, potions and scalars from Spire Codex logs,
  including the leak guards and the sanity filter. It is used by the S1 dataset builder and by
  `src/s3/py/roundtrip_macro.py`.
- `train_value.py`: the base `ValueNet` class, data pipeline and training utilities that
  `src/s1/train_value_v2.py` subclasses.

These files are plumbing shared by the later stages, not a standalone experiment.
