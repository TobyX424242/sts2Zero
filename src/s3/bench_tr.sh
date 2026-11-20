#!/bin/bash
# Throughput benchmark of real S3 self-play (run_s3.sh, trainers on, BACKUP=0) on a new server, one configuration at a time.
#   cd /root/sts2 && setsid nohup bash src/s3/bench_tr.sh > bench/driver.log 2>&1 < /dev/null &
# Configurations come from bench/queue.txt, one per line, popped from the top (lines can be appended while it runs):
#   TAG N CSERVERS MACRO_PERIOD [extra combat-trainer flags...]      e.g.  n90c6 90 6 0.03 --batch 1024
# A TAG containing "mps" runs the configuration under the CUDA MPS daemon (started before, stopped after).
# Each configuration is stopped TAIL (45) s after SECS: in-flight runs are not finished (they are outside the window).
# Settings = the live run s3a (extra.env + supervisor environment). Each config runs SECS (720) s; the guard stops it on
# a dead trainer/server, a Traceback, or the time limit; afterwards bench_report.py appends a line to bench/results.jsonl
# and the chain's cdata, mdata, archive and carry files are deleted. bench/STOP stops the driver after the current config.
W=/root/sts2; cd $W; mkdir -p bench
export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
SECS=${SECS:-720}; WD=src/s3/weights/s3a_s017
echo $$ > bench/driver.pid
log() { echo "[$(date -u +%T)] $*" | tee -a bench/driver.log.txt; }
cpu_idle_sample() { awk '/^cpu /{print $2+$3+$4+$6+$7+$8+$9, $5}' /proc/stat; }
while true; do
  [ -f bench/STOP ] && { log "STOP file: driver ends"; break; }
  line=$(grep -v '^\s*$' bench/queue.txt 2>/dev/null | head -1)
  [ -z "$line" ] && { log "queue empty: driver ends"; break; }
  grep -v '^\s*$' bench/queue.txt | tail -n +2 > bench/queue.tmp; mv bench/queue.tmp bench/queue.txt
  set -- $line; TAG=$1; N=$2; CS=$3; MP=$4; shift 4; CT="$*"
  CH=src/s3/runs/tb_$TAG; NAME=tb_${TAG}_s001; R=src/s3/runs/$NAME
  rm -rf $CH $R; mkdir -p $CH
  cat > $CH/extra.env <<'EOF'
EXTRA_ACTOR_ARGS="hall_sims=0.5 boss_pfull=1 practice_boss=1 practice_min_act=2 practice_pfull=0.25 asc_weights=0,0,0,0,0,1,1,2.667,2.667,2.667"
EXTRA_CTRAIN_ARGS="--surprise 0.5 --ema 0.999"
CARRY=1
ASC_MIX=0.6
EOF
  log "config $TAG: N=$N CSERVERS=$CS MACRO_PERIOD=$MP CTRAIN_ARGS='$CT' for $SECS s"
  MPS=0; case $TAG in *mps*) MPS=1;; esac
  if [ $MPS = 1 ]; then nvidia-cuda-mps-control -d && log "MPS daemon started" || log "MPS start FAILED"; fi
  du0=$(df -B1M / | awk 'NR==2{print $3}')
  ASC=10 CONC=32 COMBAT_WINDOW=2000000 BACKUP=0 CSERVERS=$CS MACRO_PERIOD=$MP CTRAIN_ARGS="$CT" SEED=5000 \
    COMBAT_INIT=$WD/weights.pt MACRO_INIT=$WD/macro.pt \
    MTRAIN_ARGS="--runs-per-iter 1024 --kl-runs 75000 --bc src/s3/data/macro_s1.pt --epochs 10 --target-kl 0.012" \
    ACTOR_ARGS="chunk=16384 mchunk=32" \
    setsid src/s3/run_s3.sh $NAME $N $SECS > bench/$TAG.log 2>&1 < /dev/null &
  SP=$!; echo $SP > bench/current.pid; t0=$(date +%s); why=""
  read a0 i0 < <(cpu_idle_sample)
  : > bench/sys_$TAG.txt
  while kill -0 $SP 2>/dev/null; do
    sleep 20; el=$(( $(date +%s) - t0 ))
    read a1 i1 < <(cpu_idle_sample); idle=$(awk -v a=$((a1-a0)) -v i=$((i1-i0)) 'BEGIN{printf "%.1f", 100*i/(a+i)}'); a0=$a1; i0=$i1
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
    ram=$(free -m | awk '/^Mem:/{print $3}'); disk=$(( $(df -B1M / | awk 'NR==2{print $3}') - du0 ))
    act=$(ps -eo cmd | grep -c "[S]3.dll play mode=selfplay bridge=/dev/shm/s3_${NAME}_")
    shm=$(du -sm /dev/shm 2>/dev/null | cut -f1)
    echo "$el idle=$idle gpu=$g ram_mb=$ram disk_mb=$disk actors=$act shm_mb=$shm load=$(cut -d' ' -f1 /proc/loadavg)" >> bench/sys_$TAG.txt
    if [ $el -ge 150 ] && [ $el -lt $SECS ]; then
      for f in $R/logs/trainer_pids.txt $R/logs/server_pids.txt; do
        for p in $(cat $f 2>/dev/null); do kill -0 $p 2>/dev/null || why="process $p of $(basename $f) died"; done
      done
      [ "$act" -lt $(( N * 9 / 10 )) ] && why="only $act of $N actors alive"
    fi
    grep -lE "Traceback|Unhandled exception|CUDA out of memory" $R/logs/*.txt > /dev/null 2>&1 && why="error in $(grep -lE 'Traceback|Unhandled exception|CUDA out of memory' $R/logs/*.txt | head -3 | tr '\n' ' ')"
    [ $el -gt $(( SECS + ${TAIL:-45} )) ] && why="end of measurement (in-flight runs not finished)"
    [ $(free -m | awk '/^Mem:/{print $7}') -lt 4000 ] && why="RAM available below 4 GB"
    if [ -n "$why" ]; then
      log "config $TAG stopped by the guard at ${el}s: $why"
      kill -TERM -- -$SP 2>/dev/null; sleep 8; kill -KILL -- -$SP 2>/dev/null; break
    fi
  done
  wait $SP 2>/dev/null
  log "config $TAG ended after $(( $(date +%s) - t0 )) s${why:+ (guard: $why)}"
  ./.venv/bin/python -I src/s3/bench_report.py $R bench/sys_$TAG.txt $TAG $N $CS $MP $SECS "$CT" "$why" >> bench/results.jsonl 2>> bench/driver.log.txt
  tail -n 1 bench/results.jsonl | cut -c1-600 | tee -a bench/driver.log.txt
  grep -hE "Traceback|Error|Unhandled" $R/logs/*.txt 2>/dev/null | grep -v UserWarning | sort | uniq -c | sort -rn | head -5 >> bench/driver.log.txt
  du -sm $R/* 2>/dev/null | sort -n | tail -4 | tr '\n' ' ' >> bench/driver.log.txt; echo >> bench/driver.log.txt
  rm -rf $R/cdata $R/mdata $R/archive $R/carry_* $CH/carry_* /dev/shm/s3_${NAME}_*
  if [ $MPS = 1 ]; then sleep 3; echo quit | nvidia-cuda-mps-control; sleep 3; log "MPS daemon stopped"; fi
  sleep 5
done
rm -f bench/current.pid
