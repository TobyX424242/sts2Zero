# Simulator provenance and modifications

`iRyougi-sts2-sim/` is a source snapshot of the MIT-licensed simulator
[iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim), taken at commit `db844ad` (2026-11-10), with four local
patches applied. The upstream `LICENSE`, `README.md` and test suite are kept in the directory. Build outputs and the
upstream git history are not included.

This tree contains no game files. It is a reimplementation used for training and evaluation; it is not a byte-exact
copy of the game. See `doc/simulator.md` for the fidelity evidence and the known gaps.

## Applied changes

| Patch | Purpose | Behaviour |
|---|---|---|
| `patches/iryougi_local.patch` | Parentheses in one pattern expression so the tree compiles on .NET SDK 9.0.118 | none |
| `patches/iryougi_s0.patch` | Draw-pile position knowledge (`KnownTopCount` / `KnownBottomCount`), clone support, `InternalsVisibleTo` for `Sts2Sim.S0` / `S0Tests` | Game behaviour unchanged; only counters are added |
| `patches/iryougi_s3.patch` | `CardSelectCmd.CurrentDeckPurpose` (transform / upgrade / remove) readable by the decision source | none; no game behaviour depends on it |
| `patches/iryougi_perf.patch` | Hook-listener caching / skip index, pooled buffers, exact-size list capacities, `ModelDb` id cache | Measured 1.31x simulator throughput; checked against identical logs and chunks and the upstream test suite (4,662 passed) |

The working tree in this repository already has all four applied. For a fresh clone, apply them in the order
`local`, `s0`, `s3`, `perf`; `perf` was generated last. The staged tree is authoritative if a patch does not apply
cleanly because of context drift.

## Attribution

- Upstream: iRyougi/sts2-sim, MIT License, Copyright (c) 2026 iRyougi and contributors. The license text is in
  `iRyougi-sts2-sim/LICENSE`.
- Our changes: simulator extensions required by the search (resampling, deck-purpose signal) and a throughput patch.
  They are documented in `doc/simulator.md` and in the patch headers.
