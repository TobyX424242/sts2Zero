# Checkpoints

The S3 checkpoints are published as GitHub release assets in
[**sts2Zero v0.1.0-s045**](https://github.com/TobyX424242/sts2Zero/releases/tag/v0.1.0-s045). They are not stored in
the git history.

## Assets

| Asset | Size | SHA256 | What it is |
|---|---|---|---|
| [`s3_s045_weights.pt`](https://github.com/TobyX424242/sts2Zero/releases/download/v0.1.0-s045/s3_s045_weights.pt) | 5,505,333 B | `ad1ef93f86ae3a33fd0ab9299b36c38ec1864ad4c8211f2915e66df7282e872e` | Playing combat network at segment 45, the one scored at 31.3% A10 |
| [`s3_s045_macro.pt`](https://github.com/TobyX424242/sts2Zero/releases/download/v0.1.0-s045/s3_s045_macro.pt) | 31,122,456 B | `1d5194cbaa9cb24043a7f943edd60abd5a6346ef14ef6a1805b5871e6843e1b7` | Macro ensemble + policy at segment 45 (5 members) |
| [`s3_s045_shadow.pt`](https://github.com/TobyX424242/sts2Zero/releases/download/v0.1.0-s045/s3_s045_shadow.pt) | 5,653,327 B | `37a70f224092643956b5e8dab9c16436966e2b07a82262cfe8b5d756ba1d1572` | layout-v4/v5 `ext` combat network; the one segment 46 continues from |
| [`SHA256SUMS`](https://github.com/TobyX424242/sts2Zero/releases/download/v0.1.0-s045/SHA256SUMS) | 252 B | – | checksum file |
| [`manifest.json`](https://github.com/TobyX424242/sts2Zero/releases/download/v0.1.0-s045/manifest.json) | – | – | stage, evaluation, configs and hashes in machine-readable form |

Configs:

| Asset | `config` |
|---|---|
| `s3_s045_weights.pt` | `d=128, layers=2, heads=4, head=outcome, obs=5` (non-`ext`; the v4/v5 slots are masked, so it plays exactly the v3 network) |
| `s3_s045_shadow.pt` | `d=128, layers=2, heads=4, head=outcome, ext=1, obs=5` |
| `s3_s045_macro.pt` | 5-member ensemble, no milestone heads |

## Evaluation

The release corresponds to the last recorded evaluation of the `s3a` chain:

- ascension 10, 300 fixed seeds (`EVAL000000`+), greedy macro, 64 simulations on every move;
- 94 wins = 31.3% (95% CI 26.3-36.8);
- mean floor 43.6, act 1 cleared 93%, act 2 cleared 80%;
- 1,276,289 self-play runs when the segment ended.

## Loading

Use the checkpoints as the initialisation of a new S3 run:

```bash
git clone https://github.com/TobyX424242/sts2Zero.git
cd sts2Zero
# download the three assets from the release page into checkpoints/

COMBAT_INIT=$PWD/checkpoints/s3_s045_weights.pt \
MACRO_INIT=$PWD/checkpoints/s3_s045_macro.pt \
ASC=10 CONC=32 src/s3/run_s3.sh s3new 26 7200
```

To continue from the `ext` network that segment 46 used:

```bash
COMBAT_INIT=$PWD/checkpoints/s3_s045_shadow.pt \
MACRO_INIT=$PWD/checkpoints/s3_s045_macro.pt \
OBS=5 ASC=10 CONC=32 src/s3/run_s3.sh s3new 26 7200
```

`run_s3.sh` reads `obs` from the checkpoint config; setting `OBS=5` explicitly is harmless and makes the intent
clear. The macro checkpoint needs `src/s3/data/macro_vocab.json`, which is in the repository.

## What is and is not included

- These are **network weights only**. The `train_state_dict` field inside each combat checkpoint is the trainer's
  copy of the weights, not an optimizer state.
- No replay buffer, optimizer state or `carry_*.bin` is included. A new run starts with an empty replay window and
  fresh optimizers; it can continue training from these weights, but it is **not** an exact continuation of the
  original chain.
- The S1 and S2 warm-start checkpoints are separate artefacts; they are not needed to continue S3 from these weights.

Verify a download:

```bash
sha256sum -c SHA256SUMS
```

## Provenance and terms

- Training data for the S1/S2 warm start: [Spire Codex](https://spire-codex.com/) runs, used with the maintainer's
  confirmation. S3 training itself uses only the simulator.
- Simulator: [iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim), MIT, commit `db844ad` plus the patches in
  `src/sims/patches/`.
- The code is MIT (`LICENSE`). The checkpoints are research artefacts trained in the simulator; keep the attribution
  and check the terms that apply to the training data and the game if you redistribute them.
