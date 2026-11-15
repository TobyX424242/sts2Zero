#!/bin/bash
# Evaluates one checkpoint on held-out battle starts, one arm after another (no Gumbel noise, fixed chance per start).
# usage: research/s2/eval_ckpt.sh WEIGHTS|random OUTDIR N_ACTORS "net:0 s16:16 s64:64 ot64:64:oneturn=1 ex64:64:exact=1"
# env: STARTS (default: held-out eval pool), TABLES, COUNT (first COUNT starts, default 1000), CONC (32)
WT=$1; O=$2; N=$3; ARMS=$4
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
STARTS=${STARTS:-research/s2/data/starts_v0.111.0_defect_a10_eval.jsonl}
TABLES=${TABLES:-research/s2/data/tables_v0.111.0_defect_a10_eval.jsonl}
mkdir -p $O; head -n ${COUNT:-1000} $STARTS > $O/starts.jsonl
for arm in $ARMS; do
  IFS=: read -r name sims extra <<< "$arm"; extra=${extra//:/ }; P=/dev/shm/s2e_${name}_$$_
  taskset -c ${SERVER_CPUS:-0,1} ./.venv/bin/python -I research/s2/py/infer_server.py --prefix $P --actors $N --capacity ${CONC:-32} \
    $( [ "$WT" != "random" ] && echo --weights $WT ) > $O/infer_$name.txt 2>&1 &
  rm -f $O/$name.jsonl
  for i in $(seq 0 $((N-1))); do
    dotnet research/s2/S2/bin/Release/net9.0/S2.dll play mode=eval bridge=$P$i concurrent=${CONC:-32} tag=$name$i shard=$i/$N \
      starts=$O/starts.jsonl tables=$TABLES sims_full=$sims $extra out=$O/${name}_$i.jsonl shutdown=1 > $O/${name}_$i.txt 2>&1 &
  done
  wait
  cat $O/${name}_*.jsonl > $O/$name.jsonl; rm -f $O/${name}_*.jsonl
done
python3 research/s2/summarize_eval.py $O $ARMS | tee $O/summary.txt
