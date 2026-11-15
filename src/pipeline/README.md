# Spire Codex data helpers

`scan_runs.py`, `sample_runs.py` and `verify_sample.py` are the data-selection code used by S1. They read the
Spire Codex dump. The project used the data with the maintainer's confirmation; see `https://spire-codex.com/` for
the data source and terms.

The S1 dataset builder imports `sample_runs.py` for run selection, dedupe, eras and splits. The files are kept so the
dataset construction is auditable and reproducible if you have your own licensed copy of the dump.
