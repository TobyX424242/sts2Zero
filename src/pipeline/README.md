# pipeline: Spire Codex data selection

Helpers that read the Spire Codex run dump and choose the runs used for the human-data warm start.

| File | Content |
|---|---|
| `scan_runs.py` | Pass 1: streams a runs export (`.jsonl.gz`) and writes one metadata row per run with a single exclusion reason (empty = a clean single-player official run) |
| `sample_runs.py` | Pass 2: stratified sample of clean runs (character × ascension × outcome, then game-version era), per-uploader caps, inclusion weights, and train/val/test splits by player or seed; its selection and era rules are imported by `src/s1/build_dataset_v2.py` |
| `verify_sample.py` | Independent checks of a sample: re-parses every run, checks filters, balance and split leakage |

The data comes from [Spire Codex](https://spire-codex.com/) and was used with the maintainer's confirmation. The dump
is not part of this repository; the files are kept so the dataset construction can be audited and repeated with your
own copy.
