#!/bin/bash
# Restarts the supervisor of chain s3a without a race (restart_seg.sh lost one on 2026-11-14): the old supervisor is
# stopped while segment K runs; this script waits for that segment to end (with the disk guard), does what the
# supervisor does after a segment (shared memory, weights check, last_good, latest/), starts a new supervisor (which
# reads end_time and the current s3_supervise.sh), and runs segment K's evaluation next to it.
#   setsid nohup src/s3/adopt_restart.sh SUPERVISOR_PID K "reason" >> /root/sts2/adopt_restart.log 2>&1 < /dev/null &
SUP=$1; K=$2; WHY=$3
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
KP=$(printf %03d $K); SN=s3a_s$KP;   # not N: config.env (sourced below) sets N = actors
 R=src/s3/runs/$SN
G=$(ps -eo pid,args | awk -v n="$SN" '$2=="/bin/bash" && $3=="src/s3/run_s3.sh" && $4==n {print $1}')
[ -n "$G" ] || { echo "segment $K is not running: nothing done"; exit 1; }
kill $SUP 2>/dev/null; sleep 2; kill -0 $SUP 2>/dev/null && kill -9 $SUP
log "supervisor stopped during segment $K ($WHY); src/s3/adopt_restart.sh finishes segment $K, then starts a new supervisor"
while kill -0 $G 2>/dev/null; do
  sleep 30
  d=$(df / | awk 'NR==2{print $5}' | tr -d %)
  if [ "$d" -ge 90 ]; then log "segment $K stopped by adopt_restart.sh: disk ${d}%"; kill -TERM -- -$G; sleep 8; kill -KILL -- -$G 2>/dev/null; fi
done
rm -f /dev/shm/s3_${SN}_*
nruns=$(cat $R/runs/*.jsonl 2>/dev/null | wc -l)
if [ "$nruns" -ge 100 ] && ./.venv/bin/python -I -c "
import sys, torch
a = torch.load(sys.argv[1], map_location='cpu', weights_only=False); b = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
assert 'state_dict' in a and len(b['members']) == 5
assert all(torch.isfinite(v).all() for v in a['state_dict'].values() if v.is_floating_point())
assert all(torch.isfinite(v).all() for m in b['members'] for v in m.values() if v.is_floating_point())
" $R/weights.pt $R/macro.pt 2>> $L; then
  echo "$R/weights.pt $R/macro.pt" > $S/last_good
  mkdir -p $S/latest; cp $R/weights.pt $S/latest/weights.pt; cp $R/macro.pt $S/latest/macro.pt
  log "segment $K ok: $nruns runs (finished under adopt_restart.sh); $(tail -n 1 $R/macro_train_log.jsonl 2>/dev/null | cut -c1-300)"
  good=1
else
  log "segment $K FAILED ($nruns runs, finished under adopt_restart.sh); weights not advanced"; good=0
fi
setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
sleep 5
[ $good = 1 ] || exit 0
source $S/config.env
read CW MW < $S/last_good
E=$S/eval/seg_$KP;   # was seg_$N: N came from config.env (26), so evaluations landed in eval/seg_26 (fixed 2026-11-15)
 truns=$(cat src/s3/runs/s3a_s*/runs/*.jsonl 2>/dev/null | wc -l)
COUNT=${EVAL_COUNT:-300} ASC=${ASC:-10} CONC=12 CSERVERS=${CSERVERS:-4} src/s3/eval_s3.sh $CW $MW $E ${EVAL_N:-30} "tf64:64" > $S/eval_$K.log 2>&1 < /dev/null
rm -f /dev/shm/s3e_*
echo "seg $K runs $truns | $(grep -m1 '^== tf64' $E/summary.txt 2>/dev/null | cut -c1-200)" >> $S/curve.txt
log "eval: seg $K runs $truns | $(grep -m1 '^== tf64' $E/summary.txt 2>/dev/null | cut -c1-200)"
