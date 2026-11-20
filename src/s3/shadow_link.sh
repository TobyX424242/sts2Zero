#!/bin/bash
# Linker for a shadow trainer started inside a segment (replaces the loop of shadow_now.sh, which re-linked chunks the
# shadow had already read and deleted, so they piled up: 44 GB in segment 24). Links every newly archived chunk once
# (names remembered in shadow_linked.txt), and at the end of the segment stops the shadow trainer SP (carry saved).
# usage: src/s3/shadow_link.sh SEG_DIR SHADOW_PID
R=$1; SP=$2
cd "$(dirname "$0")/../.."
L=$R/shadow_linked.txt; touch $L
declare -A done_; while read -r b; do done_[$b]=1; done < $L
while ! grep -q "run $(basename $R) done" $R/logs/run.txt 2>/dev/null; do
  for f in $R/archive/*.s2d; do
    [ -e "$f" ] || continue; b=$(basename $f)
    [ -n "${done_[$b]}" ] && continue
    ln $f $R/shadow_data/$b 2>/dev/null; done_[$b]=1; echo $b >> $L
  done
  kill -0 $SP 2>/dev/null || { echo "[$(date +%T)] shadow trainer exited early" >> $R/logs/run.txt; break; }
  sleep 3
done
kill $SP 2>/dev/null; while kill -0 $SP 2>/dev/null; do sleep 1; done
rm -rf $R/shadow_data
echo "[$(date +%T)] shadow trainer stopped: $(grep -E "carry" $R/logs/shadow.txt | tail -n 1 | cut -c1-200)" >> $R/logs/run.txt
