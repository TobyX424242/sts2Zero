#!/bin/bash
# eval_ic113.sh (2026-11-18): src/exp/eval_side1.sh (the evaluator of the v0.111 deployment panels, e.g. experiment
# 35 / g19 on XF 4096-5119) run in the ic113 server tree (/root/exp/ic113: v0.113.1.1 simulator, its S3.dll and Python
# servers, venv and .NET of /root/sts2). Arms, servers and actors exactly as there: greedy macro policy, no search noise,
# every combat move searched with the arm's budget (16 root candidates = the eval default), one arm after another.
# usage: src/ic113/eval_ic113.sh COMBAT_WEIGHTS MACRO_FILE OUTDIR N_ACTORS "name:sims[:key=value...] ..."
#   e.g. "base:64:greedy_map=1:potion_values=/root/exp/w/potion_values.json:look_beta=20:look_smith=20:look_shop=20:tf_heal=0"
# env: COUNT (seeds, 200), FIRST (first seed index, 0), ASC (10), CONC (16), SEEDS (seed prefix, EVAL; runs use
#      <prefix><index:D6>), CSERVERS (1), OBS (5), MLAYOUT (3), CHARACTER (defect; passed as character=)
CW=$1; MW=$2; O=$3; N=$4; ARMS=$5
W=${TREE:-/root/exp/ic113}; cd $W
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1
OBS=${OBS:-5}
mkdir -p $O; COUNT=${COUNT:-200}; PER=$(( (COUNT + N - 1) / N ))
for arm in $ARMS; do
  IFS=: read -r -a parts <<< "$arm"; name=${parts[0]}; sims=${parts[1]}; extra="${parts[@]:2}"; P=/dev/shm/s3e_${name}_$$_
  CS=${CSERVERS:-1}; PERS=$(( (N + CS - 1) / CS ))
  for k in $(seq 0 $((CS-1))); do
    F=$((k * PERS)); NA=$(( N - F < PERS ? N - F : PERS )); [ $NA -le 0 ] && continue
    ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first $F --actors $NA \
      --capacity ${CONC:-16} --weights $CW --macro $MW > $O/server_${name}_$k.txt 2>&1 &
  done
  ./.venv/bin/python -I src/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --actors $N --min-period 0.02 \
    --capacity ${CONC:-16} --weights $CW --macro $MW --mlayout ${MLAYOUT:-3} > $O/mserver_$name.txt 2>&1 &
  rm -f $O/$name.jsonl $O/${name}_fights.jsonl
  for i in $(seq 0 $((N-1))); do
    first=$(( ${FIRST:-0} + i * PER )); cnt=$(( ${FIRST:-0} + COUNT - first )); cnt=$(( cnt < PER ? cnt : PER ))
    dotnet src/s3/S3/bin/Release/net9.0/S3.dll play mode=eval bridge=${P}c$i mbridge=${P}m$i concurrent=${CONC:-16} tag=$name$i \
      character=${CHARACTER:-defect} ascension=${ASC:-10} seeds=${SEEDS:-EVAL} first=$first count=$cnt sims_full=$sims obs=$OBS $extra \
      out=$O/${name}_$i.jsonl fights=$O/${name}_fights_$i.jsonl shutdown=1 > $O/${name}_$i.txt 2>&1 &
  done
  wait
  cat $O/${name}_[0-9]*.jsonl > $O/$name.jsonl; cat $O/${name}_fights_*.jsonl > $O/${name}_fights.jsonl; rm -f $O/${name}_[0-9]*.jsonl $O/${name}_fights_[0-9]*.jsonl
done
./.venv/bin/python -I src/s3/summarize_runs.py $O/*.jsonl 2>/dev/null | tee $O/summary.txt
