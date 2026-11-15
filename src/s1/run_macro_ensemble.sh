#!/bin/bash
# S1 ensemble: light macro networks with seeds 1-4 (seed 0 = runs/macro/noattn_light). One at a time.
# env: F (built dataset dir, default runs/floors_v2), PY
set -e
W=$(cd "$(dirname "$0")/../.." && pwd); cd "$W"
PY=${PY:-./.venv/bin/python}; F=${F:-runs/floors_v2}
for s in 1 2 3 4; do
  o=runs/macro/noattn_light_seed$s
  [ -f "$o/results.json" ] && continue
  $PY -I src/s1/train_macro_v2.py "$F" "$o" --arch noattn --epochs 2 --patience 9 --seed "$s" || echo "FAILED $o"
done
echo ENSEMBLE_DONE
