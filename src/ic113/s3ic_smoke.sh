#!/bin/bash
# CPU smoke of the whole s3ic chain machinery (src/ic113/CHAIN.md): the real supervisor, run_s3.sh, trainers, servers,
# chain hook and evaluation, with tiny settings and no GPU (CUDA_VISIBLE_DEVICES=""): chain s3icsmoke, two segments of
# SEG (300) s, then the pause flag stops it. Starting networks: the converted s3v3_s005 pair (the warm-up output is not
# needed for a smoke). Backups go to a local folder instead of Nextcloud. Curriculum thresholds are set so that the stage
# moves 1 -> 2 at the first boundary (A0-A4 >= 0%), to exercise the move, its log and the stage-2 actor arguments.
# usage (server, chain tree): src/ic113/s3ic_smoke.sh run | check | clean   (env SEG (300), SMOKE_CPUS (56-63))
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
NAME=s3icsmoke; S=src/s3/runs/$NAME; I=src/s3/runs/${NAME}_init; NC=$W/smoke_nc
export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1
case "$1" in
run)
  [ -d $S ] && { echo "$S exists: run '$0 clean' first"; exit 1; }
  mkdir -p $S $I $NC
  cp /root/exp/ic113/w/combat_s3v3_s005_v113.pt $I/weights.pt; cp /root/exp/ic113/w/macro_s3v3_s005_v113.pt $I/macro.pt
  cp /root/exp/ic113/w/macro_s1_v113.pt $I/macro_s1_v113.pt
  cat > $S/config.env <<EOF
export N='3'
export SEG='${SEG:-300}'
export ASC='10'
export ASC_MIX='0.20'
export CONC='4'
export CSERVERS='1'
export RUNS_PER_ITER='16'
export KL_RUNS='1'
export PPO_TARGET_KL='0.03'
export MIN_RUNS='1'
export COMBAT_WINDOW='20000'
export EVAL_COUNT='4'
export EVAL_N='2'
export EVAL_ARM='tf8:8'
export EVAL_SEEDS='Y'
export EVAL_FIRST='100000'
export COMBAT0='$I/weights.pt'
export MACRO0='$I/macro.pt'
export NC_ROOT='$NC'
export CHAIN_HOOK='src/ic113/s3ic_hook.sh'
export EVAL_BASE_ARGS='character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20 look_smith=20 look_shop=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 look_dump=$S/lookdump_eval/{tag}.jsonl'
export CURRIC_T1_A10='0.05'
export CURRIC_T1_LOW='0.0'
export CURRIC_T2_A10='0.12'
export CURRIC_MIN_RUNS='1'
export VMIX_ROWS='40'
export VMIX_RECENT='0'
export VMIX_ASC='all'
export VMIX_DEVICE='cpu'
export VMIX_THREADS='8'
EOF
  cat > $S/extra.env <<EOF
EXTRA_ACTOR_ARGS="character=ironclad era=6 tf_heal=1 hall_sims=0.5 boss_pfull=1 practice_boss=1 practice_min_act=2 practice_pfull=0.25 candidates=4 look_beta=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 look_dump=$S/lookdump_sp/{tag}.jsonl"
COMBAT_WINDOW=20000
EXTRA_CTRAIN_ARGS="--surprise 0.5 --ema 0.999 --carry-keep 10000 --lr 5e-5 --wd 0.05 --wd-groups matrices --config vdrop=0.3"
EXTRA_MTRAIN_ARGS="--window-runs 2000 --value-row-share 0.16 --value-steps 5 --value-dropout --value-wd 0.1 --prob-avg --policy-warmup-s 60 --alpha 0.70 --bc $I/macro_s1_v113.pt"
CARRY=1
MPS=0
CSERVERS=1
OBS=5
EXTRA_SERVER_ARGS="--device cpu --dtype fp32"
EXTRA_MSERVER_ARGS="--device cpu"
EOF
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SIMS_FULL=8 SIMS_FAST=4
  t0=$(date +%s)
  # pinned to a few cores (SMOKE_CPUS): s3_server.py on CPU takes one torch thread per core of its affinity, and 64 threads
  # on a loaded box spend their time waiting for each other
  setsid nohup taskset -c ${SMOKE_CPUS:-56-63} nice -n 10 src/s3/s3_supervise.sh $NAME 1 > $W/s3_supervise_$NAME.log 2>&1 < /dev/null &
  SP=$!; echo "supervisor pid $SP"
  until grep -q "segment 1 ($NAME" $S/supervisor.log 2>/dev/null || ! kill -0 $SP 2>/dev/null; do sleep 15; done
  touch $S/pause; echo "pause flag set at $(( $(date +%s) - t0 )) s (segment 1 running)"
  while kill -0 $SP 2>/dev/null; do sleep 15; done
  echo "supervisor ended after $(( $(date +%s) - t0 )) s"; cat $S/supervisor.log
  ;;
check)
  echo "== supervisor / hook lines"; grep -E "hook:|settings from segment.env|segment [0-9] ok|FAILED|eval:" $S/supervisor.log
  echo "== curriculum.log"; cat $S/curriculum.log; echo "== curriculum_stage: $(cat $S/curriculum_stage)"
  echo "== segment.env"; cat $S/segment.env
  echo "== Vbar files"; for f in $S/vmix/*.json; do ./.venv/bin/python -I src/ic113/potion_mix.py show $f | cut -c1-300; done
  echo "== Vbar job logs"; cat $S/vmix/*.log | cut -c1-400
  for k in 0 1; do R=src/s3/runs/${NAME}_s00$k
    echo "== segment $k: actor arguments and potion_mix lines"
    grep -ho "potion_mix=[^ ]*\|asc_mix=[^ ]*\|asc_weights=[^ ]*" $W/s3_supervise_$NAME.log $R/logs/run.txt 2>/dev/null | sort | uniq -c
    grep -h "potion_mix\|\] mode" $R/logs/a*.txt | cut -c1-330
    echo "   runs $(cat $R/runs/*.jsonl | wc -l), ascensions $(cat $R/runs/*.jsonl | grep -o '"ascension":[0-9]*' | sort | uniq -c | tr '\n' ' ')"
    ls -la $R/carry_macro.pt $R/carry_combat.bin 2>&1 | awk '{print "   " $5, $9}'
  done
  echo "== evaluations"; for E in $S/eval/seg_*; do echo "-- $E"; grep -h "potion_mix\|\] mode" $E/tf8_*.txt | cut -c1-300; cat $E/report.txt; done
  echo "== eval_report.txt"; cat $S/eval_report.txt
  echo "== dV check (actor vs Python) on the evaluation decisions of segment 0 (macro fixed: $NAME s000's macro.pt)"
  ls $S/lookdump_eval/*.jsonl >/dev/null 2>&1 && ./.venv/bin/python -I src/ic113/potion_mix.py check --dump "$S/lookdump_eval/*.jsonl" --macro src/s3/runs/${NAME}_s000/macro.pt
  echo "== self-play potion_mix records: $(cat $S/lookdump_sp/*.jsonl 2>/dev/null | grep -c '"potion_mix"')"
  echo "== local backup folder"; find $NC -maxdepth 2 | head -20; du -sh $NC
  ;;
clean)
  for p in $(ps -eo pid,cmd | grep -E "[s]3_supervise.sh $NAME |runs/${NAME}_s|s3_${NAME}_s|runs/$NAME/" | awk '{print $1}'); do kill $p 2>/dev/null; done
  sleep 3
  rm -rf src/s3/runs/${NAME} src/s3/runs/${NAME}_s[0-9][0-9][0-9] $I $NC $W/s3_supervise_$NAME.log /dev/shm/s3_${NAME}_* /dev/shm/s3e_tf8_* /dev/shm/push_$NAME
  echo "cleaned: $(ls src/s3/runs 2>/dev/null | tr '\n' ' ')"
  ;;
*) echo "usage: $0 run|check|clean"; exit 2;;
esac
