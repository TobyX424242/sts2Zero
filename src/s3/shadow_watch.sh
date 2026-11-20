#!/bin/bash
# Compares the playing combat network with the shadow (larger) network at every segment boundary of a chain, on fresh
# data: the first FRESH chunks the next segment's actors write (played with the previous segment's network; neither
# network has trained on them). Results are appended to src/s3/runs/<chain>/shadow_compare.txt.
# usage: setsid nohup src/s3/shadow_watch.sh CHAIN [FIRST_SEGMENT] > log 2>&1 &     env: FRESH (60 chunks)
C=${1:-s3a}; K=${2:-25}; FRESH=${FRESH:-60}
cd "$(dirname "$0")/../.."
OUT=src/s3/runs/$C/shadow_compare.txt
while true; do
  P=src/s3/runs/${C}_s$(printf %03d $((K - 1))); R=src/s3/runs/${C}_s$(printf %03d $K)
  until [ -d $R/archive ]; do sleep 30; done
  D=src/s3/out/shadow_cmp/s$(printf %03d $K); mkdir -p $D
  # link the first FRESH chunks as they are archived (backup_loop.sh compresses and moves them within ~2 minutes)
  for i in $(seq 1800); do
    for f in $R/archive/*.s2d; do [ -e "$f" ] || continue; b=$(basename $f); [ -e $D/$b ] || ln $f $D/$b 2>/dev/null; done
    [ $(ls $D | wc -l) -ge $FRESH ] && break
    sleep 2
  done
  if [ -f $P/shadow.pt ]; then
    { echo "== segment $K start, $(date -u +%F\ %T) UTC: $P/weights.pt (playing) vs $P/shadow.pt (shadow), fresh chunks of $R"
      ./.venv/bin/python -I src/s2/py/combat_heldout.py --data $D --skip-every 0 --max 200000 \
        playing=$P/weights.pt shadow=$P/shadow.pt 2>&1 | grep -v -E "Warning|self.encoder"; } >> $OUT
  else
    echo "== segment $K: no $P/shadow.pt" >> $OUT
  fi
  rm -rf $D
  K=$((K + 1))
done
