#!/bin/bash
# Keeps the fight logs of 1 actor in K (a0, aK, a2K, ...) in a finished segment directory and deletes the rest, so only
# that sample is uploaded with the segment's meta copy (owner, 2026-11-18: "backup 1/4 of them to nextcloud as sample").
# usage: src/s3/thin_fights.sh SEGMENT_DIR K LOG
R=$1; K=$2; L=$3
[ "$K" -gt 1 ] 2>/dev/null || exit 0
n=0; k=0
for f in $R/fights/a*.jsonl; do
  [ -e "$f" ] || continue
  i=${f##*/a}; i=${i%.jsonl}
  [[ $i =~ ^[0-9]+$ ]] || continue
  if [ $((10#$i % K)) -ne 0 ]; then rm -f "$f"; n=$((n + 1)); else k=$((k + 1)); fi
done
echo "[$(date +%T)] fight logs: kept $k actor files (1 in $K), deleted $n" >> $L
