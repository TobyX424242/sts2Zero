#!/bin/bash
# Resume a chain paused by pause_chain.sh: restores end_time moved later by the length of the pause, restarts the
# supervisor (it continues from last_good with the next segment). Optional: CONFIG / EXTRA files replace the chain's
# config.env / extra.env first (old copies kept with a timestamp), for a resume with changed settings.
#   src/s3/resume_chain.sh CHAIN [NEW_CONFIG_ENV] [NEW_EXTRA_ENV]
C=$1; NC=${2:-}; NX=${3:-}
cd "$(dirname "$0")/../.."
S=src/s3/runs/$C; ts=$(date -u +%Y%m%dT%H%M%S)
[ -f $S/end_time.paused ] || { echo "no end_time.paused in $S: not paused by pause_chain.sh"; exit 1; }
ps -eo cmd | grep -q "[s]3_supervise.sh $C " && { echo "a supervisor for $C is running"; exit 1; }
[ -n "$NC" ] && { cp $S/config.env $S/config.env.$ts; cp $NC $S/config.env; }
[ -n "$NX" ] && { cp $S/extra.env $S/extra.env.$ts; cp $NX $S/extra.env; }
end=$(( $(cat $S/end_time.paused) + $(date +%s) - $(cat $S/paused_at) )); echo $end > $S/end_time; rm -f $S/end_time.paused $S/paused_at
echo "[$(date -u +%F\ %T)] chain resumed (ends $(date -u -d @$end +%F\ %T) UTC)${NC:+, config replaced}${NX:+, extra.env replaced}" >> $S/supervisor.log
setsid nohup src/s3/s3_supervise.sh $C 0 >> s3_supervise_$C.log 2>&1 < /dev/null &
sleep 3; tail -n 2 $S/supervisor.log
