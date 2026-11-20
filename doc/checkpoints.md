# Checkpoints

The weights of segment 29 of the `s3ic` chain are published as assets of the GitHub release
[**v0.2.0-ic113-s029**](https://github.com/TobyX424242/sts2Zero/releases/tag/v0.2.0-ic113-s029). They are not in the
git history. This repository keeps the release's `checkpoints/manifest.json` and `checkpoints/SHA256SUMS`.

## 1. Assets

| Asset | Bytes | SHA-256 | What it is |
|---|---|---|---|
| `ic113_s029_weights.pt` | 5,832,693 | `57c5f949ed084ce07969e16fe86770357c86ba59203d2af36f8ff713d6adc41c` | Combat network (the playing network of segment 29) |
| `ic113_s029_macro.pt` | 43,557,435 | `4d63f80c6d135f219a1e4f4d45fda1037633be762823ce29296c5824292ee6ba` | Macro ensemble (5 members) with policy and milestone heads |
| `ic113_s029_vbar.json` | 15,821 | `53099bb1f8973d16c4c4415651a5d9f5569afd6f4398876940b7f4d06d28b079` | potion_mix price table used by the segment-29 evaluation (the chain's `vbar_s030.json`) |
| `manifest.json` | 2,454 | `f3ed1535c028e4ffabdc0afe08084613df84aa1c3edaedd0e46f4199cb9d0435` | Release record: simulator commit, evaluation settings and result, configs, hashes |
| `SHA256SUMS` | 507 | — | Checksums of the other six files |
| `AgentBridge-mod.zip` | 49,299 | `712ad9d6526d525d970e26b0c8172f6a7b558d6f66822f3f0ea3f04e74ef0c56` | Game mod for live play ([live-play.md](live-play.md)) |
| `LiveAgent.zip` | 2,038,057 | `ebb4b9562cd948a0ece965d02da273c07d545cdab40058bb85ca93ee7f19da70` | Live-play controller ([live-play.md](live-play.md)) |

All sizes and hashes were checked against the files staged for the release (`sha256sum -c SHA256SUMS`: all OK). The
zips were built with `scripts/build_release_binaries.sh` and include the event-key fix of the live controller.

Verify a download in the folder that holds the files:

```bash
sha256sum -c SHA256SUMS                 # Linux
```

```powershell
Get-FileHash ic113_s029_weights.pt -Algorithm SHA256   # Windows, compare with the table
```

`scripts/fetch_weights.sh` (and `.ps1`) downloads the files and checks them for you.

## 2. What the files contain

**Combat network** (`ic113_s029_weights.pt`): a dictionary with `state_dict` (the exponential moving average of
the weights, which servers and evaluations load), `train_state_dict` (the trainer's raw weights, not an optimizer state;
a new trainer resumes from them), `config` and `version` (44145).

| Config key | Value | Meaning |
|---|---|---|
| `d`, `layers`, `heads` | 128, 2, 4 | transformer width, depth, attention heads |
| `head` | outcome | end-HP bins + potions kept ([architecture.md §3.1](architecture.md#31-outcome-head)) |
| `obs` | 5 | observation layout v5 |
| `ext` | 1 | reads card damage / block, hidden counters and relic counters |
| `vdrop` | 0.3 | value dropout |

724,810 parameters.

**Macro ensemble** (`ic113_s029_macro.pt`): a dictionary with `members` (5), `vocab` (721 cards, 23 enchantments, 318
relics, 69 potions, 163 encounters), `special` (rest options and other special tokens), `version` (1052), `milestones`
(true), `alpha` (0.85), `boss1_share` (0.0), `layout` (3) and `props_digest` (`59a4974d5b77`, the content-property
table it was trained with). 2,165,194 parameters per member, 10,825,970 in all. It needs
`src/s3/data/macro_vocab_v113.json` and `src/s3/data/content_props_v113.json`, which are in the repository.

**Potion table** (`ic113_s029_vbar.json`): per act and potion kind, the mean price `Vbar` that potion_mix uses
([architecture.md §7](architecture.md#7-potion_mix-which-potion-to-keep)).

Loading in Python:

```python
import sys, torch
sys.path.insert(0, "src/s3/py")
import macro_s3 as ms

combat = torch.load("ic113_s029_weights.pt", map_location="cpu", weights_only=False)
print(combat["config"], combat["version"])
ens = ms.Ensemble.load("ic113_s029_macro.pt", "cpu")     # five S3MacroNet members
```

## 3. Evaluation result

From `manifest.json` and the chain's evaluation report:

| Item | Value |
|---|---|
| Seeds | `Y100000`–`Y100299` (300 fixed A10 seeds, never used in training) |
| Settings | greedy macro, map argmax, 64 simulations on every move, 16 root candidates, deployment rules (below) |
| Result | 152 / 300 = 50.7% (95% CI 45.0–56.3), 0 errors |
| Mean floor; act 1 / act 2 cleared | 44.0; 92% / 82% |
| Act-3 bosses | boss 1 reached 242, won 216; boss 2 reached 216, won 152; HP share at boss 2 start 0.678, potions 1.27 |
| Self-play runs before this checkpoint | 528,810 (as counted by the chain) |

## 4. Reproduce the evaluation

```bash
W=/path/to/release/files
ARGS="character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20 look_smith=20 look_shop=20 \
potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 deck_route=1 lethal_backstop=1 \
potion_mix=$W/ic113_s029_vbar.json"
COUNT=300 FIRST=100000 SEEDS=Y ASC=10 CSERVERS=2 EVAL_ACTOR_ARGS="$ARGS" \
  src/s3/eval_s3.sh $W/ic113_s029_weights.pt $W/ic113_s029_macro.pt runs/eval_s029 20 "tf64:64"
cat runs/eval_s029/summary.txt
```

Use fewer actors (the `20`) on a smaller machine. Expect a result within the panel's noise of 50.7%. GPU batching is
not bit-deterministic, so an exact repeat of 152 wins is not guaranteed. This command follows the chain's evaluation
settings but was not re-run from the release files (not verified).

## 5. Continue training

Use the release as the start of a new chain: [training.md §8](training.md#8-launching-a-new-chain-from-the-release-weights).
In short:

```bash
COMBAT0=$W/ic113_s029_weights.pt MACRO0=$W/ic113_s029_macro.pt BC0=$W/ic113_s029_macro.pt \
  src/ic113/s3ic_launch.sh 168
```

after copying `configs/s3ic_config.env` and `configs/s3ic_extra.env` into `src/s3/runs/s3ic/` and setting the
curriculum stage and potion table as described there. For a single segment without the chain machinery:

```bash
COMBAT_INIT=$W/ic113_s029_weights.pt MACRO_INIT=$W/ic113_s029_macro.pt ASC=10 ASC_MIX=0.55 OBS=5 \
  ACTOR_ARGS="character=ironclad era=6 tf_heal=1 potion_mix=$W/ic113_s029_vbar.json asc_weights=0,0,0,0,0,0,0,0,1,3.5" \
  src/s3/run_s3.sh trial 60 7200
```

Add the other self-play keys and trainer options of `configs/s3ic_extra.env` (as `ACTOR_ARGS`, `CTRAIN_ARGS`,
`MTRAIN_ARGS`) to match the chain's recipe.

## 6. Not included

- Replay windows and optimizer states (the carries, several GB each). A new chain starts with empty windows and fresh
  optimizers, so it continues from these weights but is not an exact continuation of `s3ic`.
- The shadow combat network (d = 256, 3 layers), trained from scratch during segment 29. It has not replaced the
  playing network.
- The combat warm-up output, the converted Defect networks and the S1 human-data ensemble.
- Training chunks, run logs and fight logs.

## 7. Previous release

The Defect A10 agent on v0.111.0 is release
[v0.1.0-s045](https://github.com/TobyX424242/sts2Zero/releases/tag/v0.1.0-s045) (31.3% A10), documented on branch
`v0.111-defect`. Its files do not load into this branch's code without conversion (`src/ic113/convert_v113.py`), because
the vocabularies changed.
