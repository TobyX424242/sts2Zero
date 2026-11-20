#!/bin/bash
# Combat network capacity test (IMPROVEMENT_PLAN.md change C1). Self-play with frozen networks (run_s3.sh with
# FREEZE_COMBAT=1 FREEZE_MACRO=1, the live actor settings from src/s3/runs/cap/extra.env) produces one stream of
# combat chunks; several trainers learn from that same stream at the live reuse, each with its own settings:
# the first reads the run's cdata and hard-links every chunk to the others (train_combat.py --tee-dir); all skip
# chunks whose sequence number is a multiple of 10 (--skip-every 10), which are kept as the held-out set.
# After the actors stop, the trainers get EXTRA seconds to use their remaining budget; then
# src/s2/py/combat_heldout.py measures every checkpoint on the held-out chunks.
# usage: src/s3/capacity_test.sh OUT N_ACTORS SECONDS "name|trainer args" ...   (first arm = the reader)
# env: COMBAT0, MACRO0 (frozen networks), EXTRA (1200 s)
O=$1; N=$2; S=$3; shift 3
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
mkdir -p $O; NAME=cap_s000; R=src/s3/runs/$NAME
rm -rf $R; mkdir -p src/s3/runs/cap
names=(); args=()
for a in "$@"; do names+=("${a%%|*}"); args+=("${a#*|}"); done
TEE=$O/heldout
for i in "${!names[@]}"; do [ $i -gt 0 ] && TEE="$TEE,$O/d_${names[$i]}"; done
TP=""
for i in "${!names[@]}"; do
  n=${names[$i]}; mkdir -p $O/$n/ckpt $O/d_$n
  if [ $i -eq 0 ]; then DATA="$R/cdata --tee-dir $TEE"; mkdir -p $R/cdata; else DATA=$O/d_$n; fi
  ./.venv/bin/python -I src/s2/py/train_combat.py --data $DATA --skip-every 10 --weights $O/$n/weights.pt \
    --ckpt-dir $O/$n/ckpt --log $O/$n/train_log.jsonl --keep-every 10 ${args[$i]} > $O/$n/out.txt 2>&1 &
  TP="$TP $!"
  echo "[$(date +%T)] trainer $n: ${args[$i]}" | tee -a $O/log.txt
done
# the held-out directory keeps only the held-out chunks (every chunk is teed there)
( while true; do
    for f in $O/heldout/*.s2d; do [ -e "$f" ] || continue; s=$(basename $f .s2d); s=$((10#${s##*_})); [ $((s % 10)) -ne 0 ] && rm -f $f; done
    sleep 10
  done ) & CL=$!
COMBAT_INIT=${COMBAT0:-src/s3/runs/s3a_s023/weights.pt} MACRO_INIT=${MACRO0:-src/s3/runs/s3a_s023/macro.pt} \
  FREEZE_COMBAT=1 FREEZE_MACRO=1 BACKUP=0 ASC=10 CONC=32 src/s3/run_s3.sh $NAME $N $S >> $O/log.txt 2>&1
echo "[$(date +%T)] self-play done; trainers get ${EXTRA:-1200} s more" | tee -a $O/log.txt
sleep ${EXTRA:-1200}
kill $TP; wait $TP 2>/dev/null; kill $CL
rm -rf $R/cdata $R/mdata
echo "[$(date +%T)] trainers stopped; held-out chunks: $(ls $O/heldout | wc -l)" | tee -a $O/log.txt
SPEC=""; for n in "${names[@]}"; do SPEC="$SPEC $n=$O/$n/ckpt"; done
./.venv/bin/python -I src/s2/py/combat_heldout.py --data $O/heldout --skip-every 10 $SPEC | tee $O/heldout.txt
