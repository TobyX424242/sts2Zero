#!/bin/bash
# S1 macro network (value + policy), no-attention, all training runs, lightly trained (owner decision 2026-11-11):
# human data is only a warm start, so 2 epochs with the cosine schedule fully decayed over them.
# env: F (built dataset dir, default runs/floors_v2), PY
set -e
W=$(cd "$(dirname "$0")/../.." && pwd); cd "$W"
PY=${PY:-./.venv/bin/python}; F=${F:-runs/floors_v2}
$PY -I src/s1/train_macro_v2.py "$F" runs/macro/noattn_light --arch noattn --epochs 2 --patience 9
echo MACRO_DONE
