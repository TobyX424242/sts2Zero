#!/bin/bash
# Offline test of the macro value fit (IMPROVEMENT_PLAN.md change A): replays stored macro chunks of a run, in their
# original order, through src/s3/py/train_macro_s3.py --value-only, once per arm, all arms in parallel on the GPU.
# Every trainer iteration logs the loss of the current value on the next 1,024 runs before training on them
# (value_logloss) next to a constant and a 9-cell table (value_logloss_const/_table). Compare with value_replay_report.py.
# usage: src/s3/value_replay.sh OUT INIT_MACRO "SEGMENTS" "name[:--flag value ...] ..."
#   SEGMENTS: segment numbers of chain s3a whose chunks are fetched from nextcloud:sts2_backup/s3/s3a_sNNN/data
#   arms: flags after the name are passed to the trainer (':' separates words), e.g.
#     "base w:--window-runs:50000:--value-rows-per-run:16:--value-steps:150"
# env: CHAIN (s3a), DATA (OUT/data: decompressed chunks, kept for later arms), RUNS_PER_ITER (1024; the live chain uses
#      2560), SEQ=1 (arms one after another instead of all at once: for a small GPU)
O=$1; INIT=$2; SEGS=$3; ARMS=$4
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
CHAIN=${CHAIN:-s3a}; D=${DATA:-$O/data}; mkdir -p $O $D
RUNS_PER_ITER=${RUNS_PER_ITER:-1024}; SEQ=${SEQ:-0}
for s in $SEGS; do
  n=$(printf "%03d" $s); dd=$D/s$n
  if [ ! -f $dd/.ok ]; then
    mkdir -p $dd
    echo "[$(date +%T)] fetching ${CHAIN}_s$n macro chunks" | tee -a $O/log.txt
    rclone copy nextcloud:sts2_backup/s3/${CHAIN}_s$n/data $dd --include "*.s3r.zst" --transfers 16 >> $O/rclone.log 2>&1 || { echo "fetch failed"; exit 1; }
    for f in $dd/*.zst; do zstd -d -q --rm "$f"; done
    touch $dd/.ok
  fi
done
# order: segment, then chunk sequence number, then actor (a<actor>_<seq>.s3r): the actors write at about the same pace
LIST=$O/list.txt; : > $LIST
for s in $SEGS; do
  n=$(printf "%03d" $s)
  ls $D/s$n/*.s3r | awk -F/ '{f=$NF; split(f,p,/[_.]/); a=substr(p[1],2); print p[2]+0, a+0, $0}' | sort -n -k1,1 -k2,2 | awk '{print $3}' >> $LIST
done
echo "[$(date +%T)] $(wc -l < $LIST) chunks from segments $SEGS; init $INIT" | tee -a $O/log.txt
PIDS=""
for arm in $ARMS; do
  name=${arm%%:*}; extra=""; [ "$arm" != "$name" ] && extra=${arm#*:}; extra=${extra//:/ }
  mkdir -p $O/$name; rm -f $O/$name/log.jsonl
  echo "[$(date +%T)] arm $name: $extra" | tee -a $O/log.txt
  ./.venv/bin/python -I src/s3/py/train_macro_s3.py --mdata $O/$name/none --macro $O/$name/macro.pt --init $INIT \
    --log $O/$name/log.jsonl --target-asc 10 --runs-per-iter $RUNS_PER_ITER --value-only --replay $LIST --seed 1 $extra > $O/$name/out.txt 2>&1 &
  PIDS="$PIDS $!"
  [ "$SEQ" = 1 ] && { wait $!; echo "[$(date +%T)] arm $name done" | tee -a $O/log.txt; }
done
wait $PIDS
echo "[$(date +%T)] all arms done" | tee -a $O/log.txt
./.venv/bin/python -I src/s3/value_replay_report.py $O $(for arm in $ARMS; do echo ${arm%%:*}; done) | tee $O/report.txt
