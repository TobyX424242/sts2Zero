#!/bin/bash
# ic113 combat warm-up: S2 fight harness on held-out battle starts (src/exp/swap_run.sh in the ic113 tree): each arm
# plays every start once (eval mode: no Gumbel noise, logged monsters matched, the same chance seed per start in every
# arm, so arms are paired by start), search SIMS (64) on every move, 16 root candidates. Arms one after another.
# usage: src/ic113/warmup_eval.sh OUTDIR STARTS TABLES N_ACTORS "name=WEIGHTS ..."
# env: SIMS (64), CONC (fights in flight per actor, 32), EXTRA (extra S2 key=value)
O=$1; STARTS=$2; TABLES=$3; N=$4; ARMS=$5
W=${TREE:-/root/exp/ic113}; cd $W
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1
mkdir -p $O
for arm in $ARMS; do
  name=${arm%%=*}; CW=${arm#*=}; P=/dev/shm/icwe_${name}_$$_
  ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --actors $N --capacity ${CONC:-32} \
    --weights $CW --macro $W/w/macro_s3v3_s005_v113.pt > $O/${name}_server.txt 2>&1 &
  SPID=$!; PIDS=""
  rm -f $O/$name.jsonl
  for i in $(seq 0 $((N-1))); do
    dotnet src/s2/S2/bin/Release/net9.0/S2.dll play mode=eval bridge=${P}c$i concurrent=${CONC:-32} tag=$name$i shard=$i/$N obs=5 \
      starts=$STARTS tables=$TABLES sims_full=${SIMS:-64} $EXTRA out=$O/${name}_$i.jsonl shutdown=1 > $O/${name}_$i.txt 2>&1 &
    PIDS="$PIDS $!"
  done
  wait $PIDS
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  cat $O/${name}_[0-9]*.jsonl > $O/$name.jsonl; rm -f $O/${name}_[0-9]*.jsonl
  echo "$(date -u +%FT%TZ) done $name ($CW): $(wc -l < $O/$name.jsonl) fights, load failures $(cat $O/${name}_[0-9]*.txt | grep -c 'load failed')" >> $O/progress.log
done
