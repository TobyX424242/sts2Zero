#!/bin/bash
# S1 ensemble: light macro networks with seeds 1-4 (seed 0 = runs/macro/noattn_light). One at a time.
. /root/sts2/env.sh
for s in 1 2 3 4; do
  o=runs/macro/noattn_light_seed$s
  [ -f $o/results.json ] && continue
  $PY src/s1/train_macro_v2.py runs/floors_v2 $o --arch noattn --epochs 2 --patience 9 --seed $s || echo "FAILED $o"
done
echo ENSEMBLE_DONE
