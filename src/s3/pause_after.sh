#!/bin/bash
# Pauses the s3a chain after segment K (owner, 2026-11-13: the server is needed for experiments). Waits until the
# supervisor has logged "segment K ok" (weights validated, last_good written, carry files saved by the trainers) and
# started K's evaluation, stops the supervisor before it can start segment K+1, waits for the evaluation, writes its
# curve line (as the supervisor would) and a pause note. The chain resumes later with
#   setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
# (it continues from last_good with the next segment number; move end_time first if the run should last longer).
#   setsid nohup src/s3/pause_after.sh K SUPERVISOR_PID >> /root/sts2/pause_s3a.log 2>&1 < /dev/null &
K=$1; SUP=$2
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
echo "[$(date -u +%T)] waiting for 'segment $K ok' or 'segment $K FAILED'"
until grep -qE "\] segment $K (ok|FAILED)" $L; do
  sleep 5
  kill -0 $SUP 2>/dev/null || { echo "supervisor $SUP is gone: nothing done"; exit 1; }
done
sleep 10   # the supervisor starts the evaluation right after the ok line
runs_at_end=$(cat src/s3/runs/s3a_s*/runs/*.jsonl 2>/dev/null | wc -l)
kill $SUP 2>/dev/null; sleep 2; kill -0 $SUP 2>/dev/null && kill -9 $SUP
N=$(printf %03d $((K + 1)))
if ps -eo args | grep -q "^/bin/bash src/s3/run_s3.sh s3a_s$N "; then
  log "WARNING: segment $((K + 1)) had already started; stop it by hand"
fi
log "chain paused after segment $K (owner: server needed for experiments, IMPROVEMENT_PLAN.md); last good weights: $(cat $S/last_good)"
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
{ echo "S3 run s3a on $(hostname): PAUSED after segment $K (owner: server used for experiments)"
  echo "updated $(date -u +%F\ %T) UTC; segment $K; runs so far $runs_at_end"
  echo "weights in use: $(cat $S/last_good)"
  echo "--- learning curve"; tail -n 6 $S/curve.txt
  echo "--- supervisor log"; tail -n 6 $L; } > $S/status.txt.tmp && mv $S/status.txt.tmp $S/status.txt
timeout 900 rclone copy $S nextcloud:sts2_backup/s3/s3a --update --transfers 8 --exclude "*.tmp" 2>&1 | tail -n 2
echo "[$(date -u +%T)] PAUSE_DONE"
