#!/bin/bash
# Starts the shadow combat trainer (larger network) inside a segment that was started without SHADOW_ARGS: hard-links
# the chunks the main trainer archives (before backup_loop.sh compresses them) into <seg>/shadow_data, runs the
# shadow trainer on them like run_s3.sh would, and stops it with SIGTERM (carry saved) when the segment's run is done.
# The next segment (SHADOW_ARGS in extra.env) continues from <seg>/shadow.pt and <seg>/carry_shadow.bin.
# usage: src/s3/shadow_now.sh SEG_DIR "shadow trainer args"
R=$1; ARGS=$2
cd "$(dirname "$0")/../.."
mkdir -p $R/shadow_data $R/shadow_ckpt
./.venv/bin/python -I src/s2/py/train_combat.py --data $R/shadow_data --weights $R/shadow.pt --ckpt-dir $R/shadow_ckpt \
  --log $R/shadow_train_log.jsonl $ARGS --save-carry $R/carry_shadow.bin --carry-keep 0 > $R/logs/shadow.txt 2>&1 &
SP=$!
echo "[$(date +%T)] shadow combat trainer (started inside the segment, pid $SP): $ARGS" >> $R/logs/run.txt
while ! grep -q "run $(basename $R) done" $R/logs/run.txt 2>/dev/null; do
  for f in $R/archive/*.s2d; do [ -e "$f" ] || continue; b=$(basename $f); [ -e $R/shadow_data/$b ] || ln $f $R/shadow_data/$b 2>/dev/null; done
  kill -0 $SP 2>/dev/null || { echo "[$(date +%T)] shadow trainer exited early" >> $R/logs/run.txt; break; }
  sleep 3
done
kill $SP 2>/dev/null; wait $SP 2>/dev/null
rm -rf $R/shadow_data
echo "[$(date +%T)] shadow trainer stopped: $(tail -n 1 $R/logs/shadow.txt | cut -c1-200)" >> $R/logs/run.txt
