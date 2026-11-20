#!/bin/bash
# Step 0 of the simulator-speed work (2026-11-13): where an S3 actor's CPU goes, with frozen weights and the live run's
# self-play settings (A10 share 0.6, the rest A5-A9 weighted toward A7-A9; hall_sims 0.5, every boss move at 64, one
# practice copy of act 2/3 bosses, 64/16 with pfull 0.25, combat and macro records written). No trainers run, so the
# actors have the CPU to themselves (the server also runs two trainers: compare with care).
# Every actor prints a "timing" line next to its usual report (src/s3/S3/Program.cs, TimingLine).
# usage: src/s3/prof_step0.sh OUTDIR N_ACTORS SECONDS
# env: CW / MW (combat weights / macro file; default the s3a_s016 copies in src/s3/weights), CONC (32), CSERVERS (1),
#      TRACE=1 (actor 0 runs under dotnet-trace, sampled thread time, nettrace + speedscope in OUTDIR), TRACE_S (trace seconds, 120)
O=$1; N=$2; S=$3
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
export DOTNET_CLI_HOME=$W/.cache/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$W/.cache/tmp
CW=${CW:-src/s3/weights/s3a_s016/weights.pt}; MW=${MW:-src/s3/weights/s3a_s016/macro.pt}
CONC=${CONC:-32}; CS=${CSERVERS:-1}
EXTRA="hall_sims=0.5 boss_pfull=1 practice_boss=1 practice_min_act=2 practice_pfull=0.25 asc_weights=0,0,0,0,0,1,1,2.667,2.667,2.667 asc_mix=0.6"
mkdir -p $O/cdata $O/mdata $O/runs $O/logs
P=/dev/shm/prof0_$$_
echo "[$(date +%T)] $N actors x $CONC runs, $S s, $CS combat servers, $CW | $MW" | tee $O/logs/run.txt
PERS=$(( (N + CS - 1) / CS )); SP=""
for k in $(seq 0 $((CS-1))); do
  F=$((k * PERS)); NA=$(( N - F < PERS ? N - F : PERS )); [ $NA -le 0 ] && continue
  ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first $F --actors $NA \
    --capacity $CONC --weights $CW --macro $MW --stats-every 60 > $O/logs/server$k.txt 2>&1 &
  SP="$SP $!"
done
./.venv/bin/python -I src/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --actors $N --min-period 0.03 \
  --capacity $CONC --weights $CW --macro $MW --stats-every 60 > $O/logs/mserver.txt 2>&1 &
SP="$SP $!"
APIDS=""
for i in $(seq 0 $((N-1))); do
  CMD="dotnet src/s3/S3/bin/Release/net9.0/S3.dll play mode=selfplay bridge=${P}c$i mbridge=${P}m$i seconds=$S concurrent=$CONC tag=a$i
    seed=$((7000 + i)) ascension=10 sims_full=64 sims_fast=16 pfull=0.25 perturb=0.1 data=$O/cdata mdata=$O/mdata
    out=$O/runs/a$i.jsonl shutdown=1 $EXTRA"
  if [ "$i" = 0 ] && [ "${TRACE:-0}" = 1 ]; then
    # start tracing after a minute of warm-up (JIT, first fights), for TRACE_S seconds
    # the runtime's diagnostics socket lives in TMPDIR, whose path here is too long for a Unix socket (108 chars):
    # actor 0 and dotnet-trace use /dev/shm instead (IPC only, nothing is installed there)
    TMPDIR=/dev/shm $CMD > $O/logs/a$i.txt 2>&1 &
    APID=$!; APIDS="$APIDS $APID"
    ( sleep 60; TMPDIR=/dev/shm $W/.cache/tools/dotnet-trace collect -p $APID --profile dotnet-sampled-thread-time --duration 00:00:$(printf %02d:%02d $(( ${TRACE_S:-120} / 60 )) $(( ${TRACE_S:-120} % 60 ))) \
        --format Speedscope -o $O/a0.nettrace > $O/logs/trace.txt 2>&1 ) &
  else
    $CMD > $O/logs/a$i.txt 2>&1 &
    APIDS="$APIDS $!"
  fi
done
# actors stop starting runs after S seconds; runs in flight then finish one by one (stragglers): stop them at S + 60
( sleep $((S + 60)); kill $APIDS 2>/dev/null ) & KP=$!
wait $APIDS 2>/dev/null
kill $KP 2>/dev/null
sleep 3; kill $SP 2>/dev/null; wait $SP 2>/dev/null
rm -f ${P}*
echo "[$(date +%T)] done" | tee -a $O/logs/run.txt
