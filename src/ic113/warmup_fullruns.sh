#!/bin/bash
# ic113 combat warm-up: full runs with a warmed combat network, the day-0 Ironclad arms (src/ic113/day0.sh ic10 / ic0)
# with only the combat weights changed: converted s3v3_s005 macro, character=ironclad, tf_heal=1, belt rule off, search 64
# on every move, 16 root candidates, greedy macro + map argmax, look_beta / look_smith / look_shop 20, obs 5, CONC 64,
# 2 combat servers, 16 actors; panel Y (A10 Y0-511, A0 Y0-255: the day-0 seeds, so runs are paired by seed).
# usage: src/ic113/warmup_fullruns.sh COMBAT_WEIGHTS OUTDIR [arm ...]   (arms: ic10 ic0; default both)
CW=$1; O=$2; shift 2
L=/root/exp/ic113; E=$L/src/ic113/eval_ic113.sh; MW=$L/w/macro_s3v3_s005_v113.pt; mkdir -p $O
B="greedy_map=1:look_beta=20:look_smith=20:look_shop=20"
log() { echo "$(date -u +%FT%TZ) fullruns: $*" >> $O/progress.log; }
arm() {  # name character asc count settings
  local D=$O/$1; rm -rf $D; mkdir -p $D; log "start $1 with $CW"
  CHARACTER=$2 ASC=$3 COUNT=$4 FIRST=0 SEEDS=Y CONC=64 CSERVERS=2 OBS=5 MLAYOUT=3 $E $CW $MW $D 16 "$1:64:$5" > $D/eval.log 2>&1
  log "done $1 rc=$?: $(grep -m1 '^== ' $D/summary.txt)"; }
for a in ${*:-ic10 ic0}; do
  case $a in
    ic10) arm ic10 ironclad 10 512 "$B:tf_heal=1" ;;
    ic0)  arm ic0  ironclad 0  256 "$B:tf_heal=1" ;;
    *) log "unknown arm $a" ;;
  esac
done
log "all done"
