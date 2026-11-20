#!/bin/bash
# ic113 combat warm-up: watchdog and hourly progress checks for src/ic113/warmup_run.sh NAME.
# Every minute: actors / trainer / servers alive, new fights and trainer steps within STALL_MIN (15) minutes, disk; problems
# go to $R/logs/watch.txt as ALERT lines (disk above 92%: the run is stopped). Every EVERY_S (3600) seconds: the current
# weights are kept as $R/hourly/hK.pt, then (in the background, one at a time)
#   - fight harness on the fixed held-out subset (eval_eb_sub.jsonl, 1,000 elite / boss starts, search 64) -> $R/evals/hK
#   - held-out losses of h0..hK on the held-out actor's chunks of the last hour -> $R/evals/heldout_hK.txt
# usage: src/ic113/warmup_watch.sh NAME   (start right after warmup_run.sh; exits when the run is done)
NAME=$1; W=${TREE:-/root/exp/ic113}; cd $W
R=$W/warm/$NAME; D=/root/exp/ic113data/warm; EV=${EVERY_S:-3600}; EN=${EVAL_N:-8}
mkdir -p $R/hourly $R/evals
log() { echo "[$(date -u +%FT%TZ)] $*" >> $R/logs/watch.txt; }
for i in $(seq 1 300); do [ -f $R/actors.pid ] && break; sleep 2; done
[ -f $R/actors.pid ] || { log "ALERT no actors.pid after 10 min"; exit 1; }
cp $R/weights.pt $R/hourly/h0.pt; log "watch start; h0 = $(./.venv/bin/python -I -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['version'])" $R/hourly/h0.pt)"
hourly() {   # K
  local K=$1 O=$R/evals/h$1
  SIMS=64 CONC=32 src/ic113/warmup_eval.sh $O $D/eval_eb_sub.jsonl $D/tables_eval.jsonl $EN "h$K=$R/hourly/h$K.pt" > $O.log 2>&1
  local T=$R/evals/ho_tmp; rm -rf $T; mkdir -p $T
  find $R/heldout -name '*.s2d' -mmin -60 -exec ln {} $T/ \; 2>/dev/null
  local A=""; for k in $(seq 0 $K); do A="$A h$k=$R/hourly/h$k.pt"; done
  ./.venv/bin/python -I src/s2/py/combat_heldout.py --data $T --skip-every 0 --max 200000 $A > $R/evals/heldout_h$K.txt 2>&1
  rm -rf $T
  log "hourly h$K: harness $(python3 -c "
import json,sys
f=[json.loads(l) for l in open(sys.argv[1])]
e=[x for x in f if x['room_type']=='elite']; b=[x for x in f if x['room_type']=='boss']
w=lambda g: 100*sum(x['result']=='won' for x in g)/max(1,len(g))
print(f'{len(f)} fights, win elite {w(e):.1f}% ({len(e)}), boss {w(b):.1f}% ({len(b)}), all {w(f):.1f}%')" $O/h$K.jsonl 2>/dev/null); held-out: $(tail -n1 $R/evals/heldout_h$K.txt)"
}
t0=$(date +%s); K=0; lastf=-1; lastfT=$t0; lasts=-1; lastsT=$t0
# the step-0 network on the same subset (the reference of the hourly harness readings)
( SIMS=64 CONC=32 src/ic113/warmup_eval.sh $R/evals/h0 $D/eval_eb_sub.jsonl $D/tables_eval.jsonl $EN "h0=$R/hourly/h0.pt" > $R/evals/h0.log 2>&1
  log "hourly h0 harness done" ) & HP=$!
while true; do
  sleep 60
  [ -f $R/.done ] && { log "run done; watch exits"; break; }
  now=$(date +%s)
  alive=0; for p in $(cat $R/actors.pid); do kill -0 $p 2>/dev/null && alive=$((alive + 1)); done
  tp=$(cat $R/trainer.pid); kill -0 $tp 2>/dev/null || log "ALERT trainer $tp not running"
  srv=$(pgrep -fc "s3_server.py --serve combat --prefix /dev/shm/icw_${NAME}_c")
  nf=$(cat $R/fights/a*.jsonl 2>/dev/null | wc -l); [ "$nf" != "$lastf" ] && { lastf=$nf; lastfT=$now; }
  st=$(tail -n1 $R/train_log.jsonl 2>/dev/null | python3 -c "import json,sys;d=json.loads(sys.stdin.read() or '{}');print(d.get('step',-1))" 2>/dev/null)
  [ "$st" != "$lasts" ] && { lasts=$st; lastsT=$now; }
  disk=$(df --output=pcent / | tail -1 | tr -dc 0-9)
  [ $((now - lastfT)) -gt $((${STALL_MIN:-15} * 60)) ] && log "ALERT no new fight for $(( (now - lastfT) / 60 )) min"
  [ $((now - lastsT)) -gt $((${STALL_MIN:-15} * 60)) ] && log "ALERT no trainer step for $(( (now - lastsT) / 60 )) min"
  [ "$alive" -lt $(wc -w < $R/actors.pid) ] && log "ALERT only $alive of $(wc -w < $R/actors.pid) actors alive"
  mem=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ "$mem" -lt 6 ] && log "ALERT only $mem GB of RAM available"
  [ "$disk" -gt 92 ] && { log "ALERT disk $disk%: stopping the run"; kill $(cat $R/actors.pid) 2>/dev/null; }
  [ $(( (now - t0) / 60 % 10 )) -eq 0 ] && log "status: actors $alive, servers $srv, fights $nf, trainer step $st, disk $disk%, RAM free $mem GB, data $(ls $R/data | wc -l) chunks, held-out $(du -sh $R/heldout | cut -f1)"
  if [ $((now - t0)) -ge $(( (K + 1) * EV )) ]; then
    K=$((K + 1)); cp $R/weights.pt $R/hourly/h$K.pt
    log "h$K saved (step $st, fights $nf)"
    if [ -n "$HP" ] && kill -0 $HP 2>/dev/null; then log "previous hourly check still running; h$K harness skipped"
    else hourly $K & HP=$!; fi
  fi
done
[ -n "$HP" ] && wait $HP
