#!/bin/bash
# Search-parameter test (IMPROVEMENT_PLAN.md change B): src/s2/eval_ckpt.sh once per arm, all arms in parallel
# (each with its own inference server and N actors), on the same held-out battle starts with the same chance seeds;
# then src/s2/summarize_eval.py compares every arm with the first, paired by start.
# usage: src/s2/search_params_test.sh OUT WEIGHTS N_PER_ARM "name:sims[:key=value...] ..."
#   e.g. "c16:64:noise=1 c8:64:noise=1:candidates=8 c8q:64:noise=1:candidates=8:qspan=0.05"
# env: COUNT (starts, default all 13244), CONC (16)
O=$1; WT=$2; N=$3; ARMS=$4
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
mkdir -p $O
PIDS=""
for arm in $ARMS; do
  name=${arm%%:*}
  SERVER_CPUS=0-$(( $(nproc) - 1 )) COUNT=${COUNT:-13244} CONC=${CONC:-16} src/s2/eval_ckpt.sh $WT $O/$name $N "$arm" > $O/$name.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
for arm in $ARMS; do name=${arm%%:*}; cp $O/$name/$name.jsonl $O/; done
python3 src/s2/summarize_eval.py $O $ARMS | tee $O/summary.txt
