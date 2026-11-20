#!/bin/bash
# Day-0 anchors on the v0.113 simulator (2026-11-18; PLAN step 5a + the Ironclad untrained-transfer baseline). GPU, one
# arm after another (one GPU job at a time). Server tree /root/exp/ic113; outputs /root/exp/ic113/out/day0/<arm>.
# Networks: the v0.111 v3 networks of segment s3v3_s005 converted by id (src/ic113/convert_v113.py), and the S1
# human macro ensemble converted the same way (macro_s1_v113.pt) with the converted s3v3_s005 combat net.
# Panel Y (new seed prefix, never used before): runs use seeds Y000000, Y000001, ... (S3 eval: <prefix><index:D6>).
# Deployment settings of the last v0.111 fresh-panel evaluation (experiment 35, /root/exp/g19.sh: XF 4096-5119,
# 44.8% with s3v3_s004): search 64 on every move, 16 root candidates (eval default), greedy macro policy with the map
# head's argmax, belt rule with the Defect potion table, look_beta / look_smith / look_shop 20, obs 5, CONC 64,
# 2 combat servers. Actors 16 (g19: 8 per arm with two arms at once).
#   defect   : Defect A10 Y0-1023, tf_heal=0 (the v0.111 T_f, comparable with the v0.111 reading)
#   ic10     : Ironclad A10 Y0-511, tf_heal=1, belt rule off (no Ironclad potion table)
#   ic10belt : the same with the Defect table's belt rule (potions outside the table get its median)
#   ic0      : Ironclad A0 Y0-255, tf_heal=1, belt rule off
#   s1ic10 / s1ic0 : ic10 / ic0 with the S1 human macro ensemble
# usage: src/ic113/day0.sh [arm ...]   (default: all, in the order above)
L=/root/exp/ic113; O=$L/out/day0; W=$L/w; E=$L/src/ic113/eval_ic113.sh; mkdir -p $O
CW=$W/combat_s3v3_s005_v113.pt; MW=$W/macro_s3v3_s005_v113.pt; M1=$W/macro_s1_v113.pt
PV=/root/exp/w/potion_values.json
B="greedy_map=1:look_beta=20:look_smith=20:look_shop=20"
log() { echo "$(date -u +%FT%TZ) day0: $*" >> $O/progress.log; }
wdog() {  # stall / disk watchdog for one arm: run lines must grow within 20 minutes
  local D=$1 last=-1 t0=$(date +%s) n
  while sleep 60; do
    n=$(cat $D/*_[0-9]*.jsonl 2>/dev/null | grep -c '"seed"'); [ "$n" != "$last" ] && { last=$n; t0=$(date +%s); }
    [ $(( $(date +%s) - t0 )) -gt 1200 ] && log "STALL $(basename $D): $n runs, no new run for 20 min"
    [ $(df --output=pcent / | tail -1 | tr -dc 0-9) -gt 92 ] && log "DISK $(df --output=pcent / | tail -1)"
  done; }
arm() {  # name character asc count macro settings
  local D=$O/$1; rm -rf $D; mkdir -p $D; log "start $1 (GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB)"
  wdog $D & local wp=$!
  CHARACTER=$2 ASC=$3 COUNT=$4 FIRST=0 SEEDS=Y CONC=64 CSERVERS=2 OBS=5 MLAYOUT=3 $E $CW $5 $D 16 "$1:64:$6" > $D/eval.log 2>&1
  local rc=$?; kill $wp 2>/dev/null; log "done $1 rc=$rc: $(grep -m1 '^== ' $D/summary.txt)"; }
ARMS=${*:-defect ic10 ic10belt ic0 s1ic10 s1ic0}
for a in $ARMS; do
  case $a in
    defect)   arm defect   defect   10 1024 $MW "$B:potion_values=$PV:tf_heal=0" ;;
    ic10)     arm ic10     ironclad 10 512  $MW "$B:tf_heal=1" ;;
    ic10belt) arm ic10belt ironclad 10 512  $MW "$B:potion_values=$PV:tf_heal=1" ;;
    ic0)      arm ic0      ironclad 0  256  $MW "$B:tf_heal=1" ;;
    s1ic10)   arm s1ic10   ironclad 10 512  $M1 "$B:tf_heal=1" ;;
    s1ic0)    arm s1ic0    ironclad 0  256  $M1 "$B:tf_heal=1" ;;
    *) log "unknown arm $a" ;;
  esac
done
log "all done"
