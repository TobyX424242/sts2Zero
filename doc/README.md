# Documentation

This directory is the curated record of the project. It deliberately keeps the experiment narrative and the key
numbers, and drops the per-segment server logs, backup metadata and panel data.

## Reading order

1. **`architecture.md`** — what the agent is: macro network, combat network, T_f, search, S3 actor.
2. **`training.md`** — how it is trained and served: S0-S3, losses, hyperparameters, inference.
3. **`experiments.md`** — what was tried, in order, with the result that changed the design.
4. **`findings.md`** — the conclusions and the mistakes behind them.
5. **`results.md`** — the headline numbers and tables.
6. **`simulator.md`** — what simulator is included, what we changed, fidelity and speed evidence.
7. **`code-map.md`** — where every file is and how to run it.
8. **`references.md`** — annotated literature and related STS2 work.

## Scope and evidence rules

- Target cell: **Defect, Ascension 10, game v0.111.0**, in iRyougi/sts2-sim.
- Win rates are on fixed evaluation seeds, never used in training. The S3 curve uses 150 seeds per point before segment 24 and 300 after; the S2
  gate uses 9,461 held-out battle starts; the ascension sweep uses 154 seeds per level.
- Paired comparisons use McNemar (win/loss) or a paired bootstrap (rel. T_f); intervals are 95%.
- "Rel. T_f" is the fight-end T_f divided by the fight's table maximum.
- Dates are the project's local working dates. "Segment" means one 1.5-2 h block of the continuous S3 run.
- Numbers quoted from memory in older notes were re-checked against the stored reports where possible; the main
  sources are the S1 final report, the S2 G2 evaluations, the S3 curve, the ascension sweep, and the improvement
  reports.

## What is not here

- The early POC (trial-1 transformer, review probes, AlphaDefect replay analysis): one paragraph in
  `experiments.md`; code/results excluded.
- Human-run data and derived datasets/checkpoints: see the top-level README.
- Nextcloud backup, unattended supervisor, deployment scripts, watch panel: excluded by request.
- Game files and decompiled game code: excluded (commercial game; local only).
