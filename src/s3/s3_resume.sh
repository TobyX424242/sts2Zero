#!/bin/bash
# Restarts the supervisor of every S3 run that has time left, has not ended and has no supervisor running: after a
# container restart, or if the supervisor process itself died. Idempotent; run from cron (main box: every 10 minutes):
#   */10 * * * * /root/sts2/src/s3/s3_resume.sh >> /root/sts2/s3_resume.log 2>&1
# Interruptible servers (2026-11-19; src/ic113/CHAIN.md "Interruptible server"): every minute, so a chain is back
# within ~1 min of the container start (cron is started by supervisord at boot; this needs no @reboot line):
#   * * * * * /root/exp/ic113/chain/src/s3/s3_resume.sh >> /root/exp/ic113/chain/s3_resume.log 2>&1
# - One instance at a time (flock on .s3_resume.lock in the tree root; a second one exits at once). The lock is not
#   passed on to the supervisor it starts.
# - Downtime does not count: the supervisor writes runs/CHAIN/heartbeat (epoch seconds) every minute. With no
#   supervisor running, a chain that is not paused (no end_time.paused) and whose heartbeat is older than 3 minutes
#   gets end_time moved later by (now - heartbeat), logged as "downtime N min: end moved to ...", and the heartbeat is
#   set to now, so the next tick adds nothing again. Chains without a heartbeat file (older supervisors) are not moved.
# - A pause flag (pause_chain.sh) whose supervisor died before the segment boundary is completed here (end_time parked
#   as the supervisor would have done): the chain stays paused (resume_chain.sh continues it). From branch resilient.
# - No supervisor is started while the GPU does not answer (right after a boot); the next tick tries again.
# - Leftover evaluation bridges (/dev/shm/s3e_*) are removed only when no process maps them (other evaluations may run).
# The restarted supervisor adopts an interrupted segment and re-runs a missed evaluation (s3_supervise.sh header).
cd "$(dirname "$0")/../.."
exec 9> .s3_resume.lock
flock -n 9 || { echo "[$(date -u +%F\ %T)] another s3_resume.sh holds the lock: nothing done"; exit 0; }
for e in src/s3/runs/*/end_time; do
  [ -f "$e" ] || continue
  d=$(dirname $e); n=$(basename $d)
  [ -f $d/end_time.paused ] && continue
  ps -eo cmd | grep -q "[s]3_supervise.sh $n " && continue
  # ended = the LAST supervisor run logged its end (2026-11-19: an old "supervisor end" line kept a chain restarted by
  # hand from ever being resumed again)
  tac $d/supervisor.log 2>/dev/null | sed '/supervisor start/q' | grep -q "supervisor end" && continue
  now=$(date +%s); hb=$(cat $d/heartbeat 2>/dev/null)
  if [[ "$hb" =~ ^[0-9]+$ ]] && [ $(( now - hb )) -gt 180 ]; then
    end=$(( $(cat $e) + now - hb ))
    echo $end > $e.tmp && mv $e.tmp $e && echo $now > $d/heartbeat
    echo "[$(date -u +%F\ %T)] downtime $(( (now - hb) / 60 )) min (last heartbeat $(date -u -d @$hb +%F\ %T) UTC): end moved to $(date -u -d @$end +%F\ %T) UTC" | tee -a $d/supervisor.log
  fi
  if [ -f $d/pause ]; then
    rm -f $d/pause; cp $e $d/end_time.paused; date +%s > $d/paused_at; date +%s > $e
    echo "[$(date -u +%F\ %T)] no supervisor for $n and its pause flag is set: chain paused (resume_chain.sh continues it)" | tee -a $d/supervisor.log
    continue
  fi
  [ $(( $(cat $e) - $(date +%s) )) -gt 1200 ] || continue
  nvidia-smi -L > /dev/null 2>&1 || { echo "[$(date -u +%F\ %T)] $n: GPU not answering yet; next tick"; continue; }
  echo "[$(date -u +%F\ %T)] no supervisor for $n: stopping leftovers and resuming" | tee -a $d/supervisor.log
  # a segment or evaluation left without its supervisor would run unguarded beside the new one: stop them first
  left=$(ps -eo pid,cmd | grep -E "[r]un_s3.sh ${n}_s|[e]val_s3.sh .*runs/$n/" | awk '{print $1}')
  for pid in $left; do kill -TERM -- -$pid 2>/dev/null; done
  [ -n "$left" ] && sleep 8
  for pid in $(ps -eo pid,cmd | grep -E "[r]un_s3.sh ${n}_s|[e]val_s3.sh .*runs/$n/|runs/${n}_s[0-9]|s3_${n}_s[0-9]|runs/$n/eval" | grep -v "backup_loop\|rclone" | awk '{print $1}'); do kill -KILL $pid 2>/dev/null; done
  rm -f /dev/shm/s3_${n}_*; rm -rf /dev/shm/push_$n
  inuse=" $(cat /proc/[0-9]*/maps 2>/dev/null | grep -o '/dev/shm/s3e_[^ ]*' | sort -u | tr '\n' ' ') "
  for f in /dev/shm/s3e_*; do [ -e "$f" ] && [[ "$inuse" != *" $f "* ]] && rm -f "$f"; done
  setsid nohup src/s3/s3_supervise.sh $n 0 >> s3_supervise_$n.log 2>&1 < /dev/null 9>&- &
done
