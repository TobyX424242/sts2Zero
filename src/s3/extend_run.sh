#!/bin/bash
# Extends a running S3 chain without touching what runs now: waits until the run's supervisor logs its normal end
# ("supervisor end: FINISHED"), stops that supervisor (by then it only waits for uploads; the segments' backup loops
# are separate processes and finish on their own), moves the run's end time HOURS later and starts the supervisor
# again with the same NAME, which resumes the chain from the last good weights with the next segment number.
# A run that ended with "STOPPED AFTER REPEATED FAILURES" is not extended.
#   setsid nohup src/s3/extend_run.sh NAME HOURS >> /root/sts2/extend_NAME.log 2>&1 < /dev/null &
# env: POLL (seconds between checks, 30)
n=$1; H=$2
cd "$(dirname "$0")/../.."
S=src/s3/runs/$n; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
[ -f $S/end_time ] && [ -f $L ] || { echo "no run $n"; exit 1; }
# the supervisor's own process only: its command starts with bash and runs the script (a shell whose command text
# merely contains the name, e.g. the ssh command that started this script, must not match)
sup() { ps -eo pid=,args= | awk -v n="$n" '$2 ~ /^(\/usr)?(\/bin\/)?bash$/ && $3 ~ /s3_supervise\.sh$/ && $4 == n {print $1}'; }
P=$(sup)
[ -n "$P" ] || { echo "no supervisor running for $n"; exit 1; }
[ $(echo "$P" | wc -w) -eq 1 ] || { echo "more than one supervisor process for $n: $P"; exit 1; }
before=$(grep -c "supervisor end" $L)
old=$(cat $S/end_time); new=$(( old + $(awk "BEGIN{printf \"%d\", $H * 3600}") ))
log "extension armed: when the supervisor (pid $P) ends, the end moves from $(date -u -d @$old +%F\ %T) to $(date -u -d @$new +%F\ %T) UTC and the chain resumes"
while true; do
  sleep ${POLL:-30}
  if [ "$(grep -c "supervisor end" $L)" -gt "$before" ]; then break; fi
  if ! kill -0 $P 2>/dev/null; then log "extension: the supervisor exited without its end line: not extended"; exit 1; fi
done
line=$(grep "supervisor end" $L | tail -n 1)
case "$line" in *FINISHED*) ;; *) log "extension: the run did not finish normally ($line): not extended"; exit 1 ;; esac
sleep 20   # it copies the final weights right after its end line
kill -TERM $P 2>/dev/null; sleep 2; kill -0 $P 2>/dev/null && kill -KILL $P 2>/dev/null
[ $new -gt $(( $(date +%s) + 1800 )) ] || { log "extension: new end $(date -u -d @$new +%F\ %T) is too close: not extended"; exit 1; }
echo $new > $S/end_time.tmp && mv $S/end_time.tmp $S/end_time
log "run extended by $H h: ends $(date -u -d @$new +%F\ %T) UTC; supervisor restarted"
setsid nohup src/s3/s3_supervise.sh $n 0 >> s3_supervise_$n.log 2>&1 < /dev/null &
sleep 5; log "extension: new supervisor pid $(sup)"
