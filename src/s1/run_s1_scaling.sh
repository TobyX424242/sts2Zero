#!/bin/bash
# S1 v2 on the server: full natural-frequency dataset, then the data-scaling curve (X3) for both model types.
# Run on the server from /root/sts2:  setsid nohup bash src/s1/run_s1_scaling.sh > runs/s1_scaling.log 2>&1 < /dev/null &
# One training job at a time. Small sizes get more epochs so early stopping, not the cap, ends them.
. /root/sts2/env.sh
F=runs/floors_v2
if [ ! -f $F/stats.json ]; then
  $PY src/s1/build_dataset_v2.py $D/runs-latest.jsonl.gz $D/meta_all.csv.gz $F --workers 28 --min-runs 100 || exit 1
fi
for spec in 25000:60 50000:40 100000:30 300000:15 0:15; do
  n=${spec%%:*}; ep=${spec##*:}
  for a in noattn transformer; do
    o=runs/s1_scaling/${a}_${n}
    [ -f $o/results.json ] && continue
    $PY src/s1/train_value_v2.py $F $o --arch $a --max-train-runs $n --epochs $ep || echo "FAILED $o"
  done
done
echo ALL_DONE
