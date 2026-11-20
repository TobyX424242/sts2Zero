#!/bin/bash
# S1 macro network (value + policy), no-attention, all training runs, LIGHTLY trained (owner decision 2026-11-11):
# human data is only a warm start, so 2 epochs with the cosine schedule fully decayed over them.
. /root/sts2/env.sh
$PY src/s1/train_macro_v2.py runs/floors_v2 runs/macro/noattn_light --arch noattn --epochs 2 --patience 9 || echo FAILED
echo MACRO_DONE
