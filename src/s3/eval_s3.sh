#!/bin/bash
# Evaluates an S3 agent on fixed run seeds (EVAL000000, EVAL000001, ...), one arm after another: greedy macro policy,
# no search noise, every combat move searched with the arm's budget.
# usage: research/s3/eval_s3.sh COMBAT_WEIGHTS MACRO_FILE OUTDIR N_ACTORS "name:sims[:key=value...] ..."
#   e.g. "tf64:64 hp64:64:scorer=hp net:0"
# env: COUNT (seeds, 200), FIRST (first seed index, 0), ASC (10), CONC (16), SEEDS (seed prefix, EVAL), CSERVERS (1)
CW=$1; MW=$2; O=$3; N=$4; ARMS=$5
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
# Match the actors to the checkpoint's observation layout (config "obs", default 4); OBS=... overrides.
if [ -z "${OBS:-}" ]; then
  OBS=$(./.venv/bin/python -I -c "import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('config',{});print(int(c.get('obs',4)))" "$CW" 2>/dev/null || echo 4)
fi
mkdir -p $O; COUNT=${COUNT:-200}; PER=$(( (COUNT + N - 1) / N ))
for arm in $ARMS; do
  IFS=: read -r name sims e1 e2 e3 e4 <<< "$arm"; P=/dev/shm/s3e_${name}_$$_
  CS=${CSERVERS:-1}; PERS=$(( (N + CS - 1) / CS ))
  for k in $(seq 0 $((CS-1))); do
    F=$((k * PERS)); NA=$(( N - F < PERS ? N - F : PERS )); [ $NA -le 0 ] && continue
    ./.venv/bin/python -I research/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first $F --actors $NA \
      --capacity ${CONC:-16} --weights $CW --macro $MW > $O/server_${name}_$k.txt 2>&1 &
  done
  ./.venv/bin/python -I research/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --actors $N --min-period 0.02 \
    --capacity ${CONC:-16} --weights $CW --macro $MW > $O/mserver_$name.txt 2>&1 &
  rm -f $O/$name.jsonl $O/${name}_fights.jsonl
  for i in $(seq 0 $((N-1))); do
    dotnet research/s3/S3/bin/Release/net9.0/S3.dll play mode=eval bridge=${P}c$i mbridge=${P}m$i concurrent=${CONC:-16} tag=$name$i \
      ascension=${ASC:-10} seeds=${SEEDS:-EVAL} first=$(( ${FIRST:-0} + i * PER )) count=$PER sims_full=$sims obs=$OBS $e1 $e2 $e3 $e4 \
      out=$O/${name}_$i.jsonl fights=$O/${name}_fights_$i.jsonl shutdown=1 > $O/${name}_$i.txt 2>&1 &
  done
  wait
  cat $O/${name}_[0-9]*.jsonl > $O/$name.jsonl; cat $O/${name}_fights_*.jsonl > $O/${name}_fights.jsonl; rm -f $O/${name}_[0-9]*.jsonl $O/${name}_fights_[0-9]*.jsonl
done
./.venv/bin/python -I research/s3/summarize_runs.py $O/*.jsonl 2>/dev/null | tee $O/summary.txt
