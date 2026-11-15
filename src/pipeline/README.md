# Spire Codex data helpers

`scan_runs.py`, `sample_runs.py` and `verify_sample.py` are the data-selection code used by S1. They read the
Spire Codex dump; the dump and any derived arrays are not redistributed (see the top-level README). The project used
the data with the maintainer's confirmation, but the data is not ours to redistribute.

The S1 dataset builder imports `sample_runs.py` for run selection, dedupe, eras and splits. The files are kept so the
dataset construction is auditable and reproducible if you have your own licensed copy of the dump.
