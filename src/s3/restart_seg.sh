#!/bin/bash
# Restarts the supervisor of chain s3a at a segment boundary with a new segment length (it reads SEG only at start)
# and the current s3a_supervise.sh. After "segment K ok" the old supervisor is in its evaluation of segment K: it is
# stopped (the evaluation goes on), config.env gets SEG, a new supervisor starts segment K+1 at once, and segment K's
# curve line is written when its evaluation ends (as the old supervisor would have). The end time is unchanged.
#   setsid nohup src/s3/restart_seg.sh K SUPERVISOR_PID SECONDS >> /root/sts2/restart_seg.log 2>&1 < /dev/null &
K=$1; SUP=$2; SEGS=$3
# UNSAFE since evaluations run next to the next segment (2026-11-13): the supervisor starts segment K+1 within ~2 s of
# "segment K ok", before this script stops it, and that segment then runs unsupervised next to the new supervisor's
# (2026-11-14 04:02: segments 31 and 32 ran in parallel and filled the disk). Refuses to run; end-time changes need no
# restart any more (s3_supervise.sh re-reads end_time every segment).
echo "restart_seg.sh is disabled (see the comment in the script)"; exit 1
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
echo "[$(date -u +%T)] waiting for 'segment $K ok' or 'segment $K FAILED'"
until grep -qE "\] segment $K (ok|FAILED)" $L; do
  sleep 5
  kill -0 $SUP 2>/dev/null || { echo "supervisor $SUP is gone: nothing done"; exit 1; }
done
grep -qE "\] segment $K ok" $L || { echo "segment $K failed: not restarting, the supervisor goes on as it is"; exit 1; }
sleep 10   # the supervisor starts the evaluation right after the ok line
runs_at_end=$(cat src/s3/runs/s3a_s*/runs/*.jsonl 2>/dev/null | wc -l)
kill $SUP 2>/dev/null; sleep 2
kill -0 $SUP 2>/dev/null && { kill -9 $SUP; sleep 1; }
N=$(printf %03d $((K + 1)))
ps -eo args | grep -q "^/bin/bash src/s3/run_s3.sh s3a_s$N " && log "WARNING: segment $((K + 1)) had already started"
sed -i "s/^export SEG=.*/export SEG='$SEGS'/" $S/config.env
log "supervisor restarted by hand (owner): $SEGS s segments from segment $((K + 1)); evaluations now run next to the next segment; segment $K's evaluation finishes alongside"
setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
sleep 5
echo "[$(date -u +%T)] new supervisor started"
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
