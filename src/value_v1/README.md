# value_v1: shared reconstruction and base trainer

Two files from the project's first value-network stage, kept because other code imports them:

| File | Used by | Content |
|---|---|---|
| `build_floor_dataset.py` | `src/s1` dataset builder, `src/s0/export_battle_starts.py`, `src/s3/py/roundtrip_macro.py` | Rebuilds decks, relics, potions and scalars of every floor from a Spire Codex run record, with the leak guards (no future cards or relics) and the sanity filter |
| `train_value.py` | `src/s1/train_value_v2.py` (subclasses it) | Base `ValueNet`, data pipeline and training utilities |

They are plumbing, not a standalone stage.
