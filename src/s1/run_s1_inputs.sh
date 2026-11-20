#!/bin/bash
# S1 input ablations (300k training runs, no-attention model) and second-seed repeats of the key runs.
# Run on the server from /root/sts2:  setsid nohup bash src/s1/run_s1_inputs.sh > runs/s1_inputs.log 2>&1 < /dev/null &
. /root/sts2/env.sh
F=runs/floors_v2; O=runs/s1_inputs
run() { o=$O/$1; shift; [ -f $o/results.json ] && return; $PY src/s1/train_value_v2.py $F $o "$@" || echo "FAILED $o"; }
run noattn_300000_seed1      --arch noattn --max-train-runs 300000 --epochs 15 --seed 1
for g in relics potions card_meta era encounter room_act scalars; do
  run noattn_300000_drop_$g  --arch noattn --max-train-runs 300000 --epochs 15 --drop $g
done
run noattn_0_seed1           --arch noattn --epochs 15 --seed 1
run transformer_0_seed1      --arch transformer --epochs 15 --seed 1
echo ALL_DONE
