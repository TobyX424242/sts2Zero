#!/bin/bash
# Restarts chain s3a after its supervisor stopped (2026-11-14: the disk was full at the end of segment 29, so segments
# 30-32 failed at once and the supervisor ended), and runs the evaluation the stopped supervisor skipped.
#   setsid nohup src/s3/restart_after_stop.sh OLD_SUPERVISOR_PID EVAL_SEG >> /root/sts2/restart_after_stop.log 2>&1 < /dev/null &
SUP=$1; K=$2
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
kill $SUP 2>/dev/null; sleep 2; kill -0 $SUP 2>/dev/null && kill -9 $SUP
source $S/config.env
read CW MW < $S/last_good
log "supervisor restarted by hand (after the disk-full stop; three empty failed segments do not count): next segment starts from $CW | $MW"
setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
sleep 5
E=$S/eval/seg_$(printf %03d $K); truns=$(cat src/s3/runs/s3a_s*/runs/*.jsonl 2>/dev/null | wc -l)
rm -rf $E
COUNT=${EVAL_COUNT:-300} ASC=${ASC:-10} CONC=12 CSERVERS=${CSERVERS:-5} src/s3/eval_s3.sh $CW $MW $E ${EVAL_N:-30} "tf64:64" > $S/eval_$K.log 2>&1 < /dev/null
rm -f /dev/shm/s3e_*
echo "seg $K runs $truns | $(grep -m1 '^== tf64' $E/summary.txt 2>/dev/null | cut -c1-200)" >> $S/curve.txt
log "eval: seg $K runs $truns | $(grep -m1 '^== tf64' $E/summary.txt 2>/dev/null | cut -c1-200)"
