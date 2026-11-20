#!/bin/bash
# Pause a chain at its next segment boundary (2026-11-16). Sets the pause flag that s3_supervise.sh checks before it
# starts a segment: the running segment finishes normally (weights validated, carries saved, evaluation started), then
# the supervisor parks end_time (kept in end_time.paused, so the cron resume does not restart it), waits for that
# evaluation and exits. No race: the first version killed the supervisor from outside after "segment K ok" and was 14 s
# too late (segment K+1 had started; it had to be stopped by hand with stop_segment.sh).
#   src/s3/pause_chain.sh CHAIN [wait]     ("wait": block until the supervisor logged the pause)
# Resume: src/s3/resume_chain.sh CHAIN [NEW_CONFIG_ENV] [NEW_EXTRA_ENV]
C=$1
cd "$(dirname "$0")/../.."
S=src/s3/runs/$C
[ -d $S ] || { echo "no chain $C"; exit 1; }
touch $S/pause; echo "[$(date -u +%F\ %T)] pause flag set for $C (takes effect at the next segment boundary)" | tee -a $S/supervisor.log
if [ "${2:-}" = "wait" ]; then
  until grep -q "chain paused before segment" $S/supervisor.log 2>/dev/null && [ ! -f $S/pause ]; do sleep 10; done
  echo "[$(date -u +%T)] PAUSED"
fi
