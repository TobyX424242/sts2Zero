#!/bin/bash
# S1 input ablations and second-seed repeats (PROPOSAL.md §6.3, experiment X3/A2).
# usage: src/s1/run_s1_inputs.sh
# env: F (built dataset dir, default runs/floors_v2), O (output dir, default runs/s1_inputs), PY
set -e
W=$(cd "$(dirname "$0")/../.." && pwd); cd "$W"
PY=${PY:-./.venv/bin/python}; F=${F:-runs/floors_v2}; O=${O:-runs/s1_inputs}
run() { o=$O/$1; shift; [ -f "$o/results.json" ] && return; $PY -I src/s1/train_value_v2.py "$F" "$o" "$@" || echo "FAILED $o"; }
run noattn_300000_seed1      --arch noattn --max-train-runs 300000 --epochs 15 --seed 1
for g in relics potions card_meta era encounter room_act scalars; do
  run noattn_300000_drop_$g  --arch noattn --max-train-runs 300000 --epochs 15 --drop "$g"
done
run noattn_0_seed1           --arch noattn --epochs 15 --seed 1
run transformer_0_seed1      --arch transformer --epochs 15 --seed 1
echo ALL_DONE
