# S1 v2: value network on all clean Spire Codex runs

Implements PROPOSAL.md §6.3 items 1, 2, 4 and 5: uniform floor sampling, state-only inputs, an HP-monotone output head, and natural frequencies, with the no-attention model as an option. The rebuild, leak guards and metrics are reused from `../value_v1/`; run selection and eras from `../pipeline/sample_runs.py`.

| File | What | Tested |
|---|---|---|
| `build_dataset_v2.py` | Streams the dump, keeps every clean unique run, builds per-floor arrays (+ `build.npy`, `builds.json`, `index.csv`) | 2026-11-10 on the server: 20k-run smoke test (583,707 floors, start-deck match 99.7%); full build launched |
| `train_value_v2.py` | Trainer, `--arch transformer|noattn`, `--max-train-runs` for the scaling curve; reports log loss, AUC, within-floor AUC, ECE in floor bands, calibration by floor, target cell (Defect A10 v0.111.0), HP probe | 2026-11-10 smoke test, 2 epochs each: HP probe 0% decreasing (monotone head works); 17k samples/s (transformer), 49–53k/s (no attention) on the RTX 4080 |
| `run_s1_scaling.sh` | Full dataset build, then the scaling curve: 25k / 50k / 100k / 300k / all training runs × both model types, one job at a time | Launched 2026-11-10 |

**Where it runs:** the rented server (`/root/sts2`; access and transfer rules in the memory notes and PROPOSAL.md §11.2). The full dataset is about 23 GB of arrays, so it needs a machine with 64 GB+ RAM.

**Server layout:** `/root/sts2/env.sh` (cache variables, `$PY`, `$D`); data in `src/data/spire_codex/` (dump downloaded on the server 2026-11-10, 9,417,747,533 bytes, sha256 `42c8fc90cd222140bac56d46093d73af5468b78dd9eefd580a1bc9c741fb1a91`; scan: 2,710,489 lines, 1,209,796 clean); outputs in `runs/floors_v2/` and `runs/s1_scaling/<arch>_<runs>/`.

**Server software:** Python 3.12.3, torch 2.6.0+cu124, numpy 2.5.4, orjson 3.13.0, pigz (apt).

## Status 2026-11-11

| File | What | Tested |
|---|---|---|
| `train_value_v2.py` (r2) | Value trainer with packed, length-sorted batches, sorted prediction, no per-step syncs, optional `--compile`, `--drop` input ablations | On the 5090: same result as r1 on noattn 300k (test log loss 0.4707 both), 136k samples/s, validation 1.2 s per 1M floors |
| `train_value_v2_r1.py` | The scaling-curve version, frozen | Produced `runs/s1_scaling_5090/` |
| `build_decisions_v2.py` | Decision dataset for the policy head: 7 decision types; each is a floor example + edits that restore the pre-choice state | Full build on the 5090: 31,341,873 decisions in 326 s |
| `train_macro_v2.py` | Macro network: shared no-attention encoder (mean pooling, no CLS) + value head (HP-monotone) + policy head | Smoke test (20k runs, 2 epochs) passed; full run started 13:59 |
| `summarize_scaling.py`, `stop_s1.sh`, `restart_s1.sh`, `s1_server_bootstrap.sh`, `run_*.sh`, `smoke_macro.sh` | Server tooling | Used on the 9950X and 5090 servers |

**S1 decisions (2026-11-11):** no-attention architecture; inputs and tokenizer unchanged (unknown tokens 0.004–0.033%, deck truncation 0.08%). Results in `runs/s1_scaling_5090/summary.txt`.
