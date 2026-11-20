#!/bin/bash
# Search-allocation test: plays the same fixed seeds with several per-room search budgets, one arm after another, with
# the self-play search settings (sims 64/16, a quarter of moves at 64, Gumbel noise on) and a greedy macro policy.
# Compare the arms with src/s3/alloc_compare.py.
# usage: src/s3/alloc_test.sh COMBAT_WEIGHTS MACRO_FILE OUTDIR N_ACTORS "name[:key=value...] ..."
#   e.g. "base hb2:hall_sims=0.5:boss_sims=2 hbf:hall_sims=0.5:boss_pfull=1"
# env: COUNT (seeds, 1000), FIRST (first seed index, 10000: away from the evaluation curve's seeds), ASC (10), CONC (12), CSERVERS (1),
#      SEEDS (seed prefix, EVAL), BASE ("sims_full=64 sims_fast=16 pfull=0.25 noise=1")
CW=$1; MW=$2; O=$3; N=$4; ARMS=$5
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
mkdir -p $O; COUNT=${COUNT:-1000}; PER=$(( (COUNT + N - 1) / N ))
BASE=${BASE:-"sims_full=64 sims_fast=16 pfull=0.25 noise=1"}
for arm in $ARMS; do
  name=${arm%%:*}; extra=""; [ "$arm" != "$name" ] && extra=${arm#*:}; extra=${extra//:/ }
  [ -s $O/$name.jsonl ] && { echo "[$(date +%T)] arm $name: already done, skipped"; continue; }
  P=/dev/shm/s3a_${name}_$$_
  echo "[$(date +%T)] arm $name: $COUNT seeds from ${FIRST:-10000}, $N actors, ${CSERVERS:-1} combat servers, $BASE $extra" | tee -a $O/log.txt
  # CSERVERS combat processes share the actors (one Python loop saturates at a few thousand states/s with small batches)
  CS=${CSERVERS:-1}; PERS=$(( (N + CS - 1) / CS )); SP=""
  for k in $(seq 0 $((CS-1))); do
    F=$((k * PERS)); NA=$(( N - F < PERS ? N - F : PERS )); [ $NA -le 0 ] && continue
    ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first $F --actors $NA \
      --capacity ${CONC:-12} --weights $CW --macro $MW > $O/server_${name}_$k.txt 2>&1 &
    SP="$SP $!"
  done
  ./.venv/bin/python -I src/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --actors $N --min-period 0.02 \
    --capacity ${CONC:-12} --weights $CW --macro $MW > $O/mserver_$name.txt 2>&1 &
  MP=$!
  t0=$(date +%s); APIDS=""
  for i in $(seq 0 $((N-1))); do
    dotnet src/s3/S3/bin/Release/net9.0/S3.dll play mode=eval bridge=${P}c$i mbridge=${P}m$i concurrent=${CONC:-12} tag=$name$i \
      ascension=${ASC:-10} seeds=${SEEDS:-EVAL} first=$(( ${FIRST:-10000} + i * PER )) count=$PER $BASE $extra \
      out=$O/${name}_$i.jsonl fights=$O/${name}_fights_$i.jsonl shutdown=1 > $O/${name}_$i.txt 2>&1 &
    APIDS="$APIDS $!"
  done
  wait $APIDS
  # the servers exit on the actors' shutdown message; do not leave one behind if an actor died
  sleep 5; kill $SP $MP 2>/dev/null; wait $SP $MP 2>/dev/null
  rm -f ${P}*
  cat $O/${name}_[0-9]*.jsonl > $O/$name.jsonl.tmp && mv $O/$name.jsonl.tmp $O/$name.jsonl
  cat $O/${name}_fights_*.jsonl > $O/${name}_fights.jsonl; rm -f $O/${name}_[0-9]*.jsonl $O/${name}_fights_[0-9]*.jsonl
  echo "[$(date +%T)] arm $name done: $(wc -l < $O/$name.jsonl) runs in $(( $(date +%s) - t0 )) s" | tee -a $O/log.txt
done
echo "[$(date +%T)] ALLOC_TEST_DONE" | tee -a $O/log.txt
