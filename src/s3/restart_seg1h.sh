#!/bin/bash
# One-off (owner, 2026-11-13): switch the running s3a chain to 1-hour segments and the 2026-11-14 10:08:28 UTC end.
# The supervisor reads SEG and end_time only at start, so it is restarted at a segment boundary: after it has logged
# "segment K ok" (weights validated, last_good written) and started the evaluation, it and the extension waiter are
# stopped; config.env (SEG) and end_time are updated; a new supervisor starts the next segment at once. Segment K's
# evaluation keeps running on its own; its curve line is appended when it finishes (the old supervisor would have).
#   setsid nohup src/s3/restart_seg1h.sh K SUPERVISOR_PID WAITER_PID >> /root/sts2/restart_seg1h.log 2>&1 < /dev/null &
K=$1; SUP=$2; WAIT=$3
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log; NEWEND=1794650908
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
echo "[$(date -u +%T)] waiting for 'segment $K ok' or 'segment $K FAILED'"
until grep -qE "\] segment $K (ok|FAILED)" $L; do
  sleep 10
  kill -0 $SUP 2>/dev/null || { echo "supervisor $SUP is gone: nothing done"; exit 1; }
done
grep -qE "\] segment $K ok" $L || { echo "segment $K failed: not restarting, the supervisor goes on as it is"; exit 1; }
sleep 15   # the supervisor starts the evaluation right after the ok line
runs_at_end=$(cat src/s3/runs/s3a_s*/runs/*.jsonl 2>/dev/null | wc -l)
kill $WAIT 2>/dev/null; kill $SUP 2>/dev/null; sleep 2
kill -0 $SUP 2>/dev/null && { kill -9 $SUP; sleep 1; }
sed -i "s/^export SEG=.*/export SEG='3600'/" $S/config.env
echo $NEWEND > $S/end_time.tmp && mv $S/end_time.tmp $S/end_time
log "supervisor restarted by hand (owner): 1-hour segments from segment $((K + 1)); run ends $(date -u -d @$NEWEND +%F\ %T) UTC (extension applied directly, waiter stopped); segment $K's evaluation finishes alongside"
setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
sleep 5
echo "[$(date -u +%T)] new supervisor: $(ps -eo pid=,args= | awk '$2 ~ /bash$/ && $3 ~ /s3_supervise\.sh$/ && $4 == "s3a" {print $1}')"
# segment K's evaluation: wait for it (at most 40 min), then write the curve line as the supervisor does
E=$S/eval/seg_$(printf %03d $K)
for i in $(seq 240); do
  pgrep -f "eval_s3.sh .*$E" > /dev/null || break
  sleep 10
done
line=$(grep -m1 '^== tf64' $E/summary.txt 2>/dev/null | cut -c1-200)
if [ -n "$line" ]; then
  echo "seg $K runs $runs_at_end | $line" >> $S/curve.txt
  log "eval: $(tail -n 1 $S/curve.txt)"
else
  log "evaluation of segment $K produced no summary line"
fi
rm -f /dev/shm/s3e_*
echo "[$(date -u +%T)] RESTART_SEG1H_DONE"
