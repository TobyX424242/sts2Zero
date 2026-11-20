#!/bin/bash
# Pauses chain s3a now (owner, 2026-11-15: "pause the training and stop it, some experiments need to be done"): stops the
# supervisor, ends the running segment gracefully (actors and inference servers only; run_s3.sh then stops the trainers
# with SIGTERM, which save weights and carry files), validates the weights as the supervisor does, writes last_good and
# latest/, stops the shadow watcher and the CUDA MPS daemon. No new segment, no evaluation.
# Resume: move end_time if needed, then
#   setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
#   setsid nohup src/s3/pause_now.sh > /root/sts2/pause_now.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
SUP=$(ps -eo pid,args | awk '$2=="/bin/bash" && $3=="src/s3/s3_supervise.sh" && $4=="s3a" {print $1}')
G=$(ps -eo pid,args | awk '$2=="/bin/bash" && $3=="src/s3/run_s3.sh" && $4 ~ /^s3a_s[0-9]+$/ {print $1}')
SN=$(ps -eo pid,args | awk '$2=="/bin/bash" && $3=="src/s3/run_s3.sh" && $4 ~ /^s3a_s[0-9]+$/ {print $4}')
[ -n "$SUP" ] && [ -n "$G" ] || { echo "supervisor ($SUP) or segment ($G) not found: nothing done"; exit 1; }
R=src/s3/runs/$SN; K=$((10#${SN#s3a_s}))
kill $SUP; sleep 2; kill -0 $SUP 2>/dev/null && kill -9 $SUP
log "owner: training paused (experiments); supervisor stopped during segment $K, the segment is ended gracefully and no new segment starts"
A=$(pgrep -g $G -f "S3.dll play"); SV=$(cat $R/logs/server_pids.txt)
echo "stopping $(echo $A | wc -w) actors and inference servers $SV"
kill -TERM $A $SV
for i in $(seq 120); do kill -0 $G 2>/dev/null || break; sleep 5; done
kill -0 $G 2>/dev/null && { log "segment $K still running 10 min after the stop: stopping its process group"; kill -TERM -- -$G; sleep 30; kill -KILL -- -$G 2>/dev/null; }
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
  log "segment $K ok: $nruns runs (ended early for the pause); last_good = $R; carry files: $(ls $R/carry_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
else
  log "segment $K FAILED validation ($nruns runs); last_good unchanged: $(cat $S/last_good)"
fi
for p in $(ps -eo pid,args | awk '$2=="/bin/bash" && ($3=="src/s3/shadow_watch.sh" || $3=="src/s3/shadow_switch_check.sh") {print $1}'); do kill $p; done
src/s3/mps.sh stop >> $R/logs/mps.txt 2>&1 && echo "MPS daemon stopped"
log "paused: no training processes left (resume: src/s3/s3_supervise.sh s3a 0; move end_time first if needed, currently $(date -u -d @$(cat $S/end_time) +%F\ %T) UTC)"
