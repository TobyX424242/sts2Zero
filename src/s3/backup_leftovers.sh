#!/bin/bash
# One pass over finished or stopped S3 segments whose training chunks were not uploaded (their backup loop was stopped
# with them): compress and move the chunks to Nextcloud, copy logs and weights. Skips segments that are running.
# usage: src/s3/backup_leftovers.sh NAME      (segments src/s3/runs/NAME_sNNN)
cd "$(dirname "$0")/../.."
for R in src/s3/runs/$1_s[0-9][0-9][0-9]; do
  n=$(basename $R)
  ps -eo cmd | grep -q "[r]un_s3.sh $n \|[b]ackup_loop.sh $R " && continue
  D=nextcloud:sts2_backup/s3/$n
  mkdir -p $R/archive; mv $R/cdata/*.s2d $R/mdata/*.s3r $R/archive/ 2>/dev/null
  [ -z "$(ls $R/archive 2>/dev/null)" ] && continue
  for f in $R/archive/*.s2d $R/archive/*.s3r; do [ -e "$f" ] || continue; nice -n 19 zstd -3 -q --rm "$f" -o "$f.zst"; done
  timeout 1800 rclone move $R/archive $D/data --include "*.zst" --transfers 4 --retries 5 2>&1 | tail -n 2
  timeout 900 rclone copy $R $D/meta --update --exclude "archive/**" --exclude "cdata/**" --exclude "mdata/**" --exclude "carry_*" --transfers 4 2>&1 | tail -n 2
  echo "[$(date -u +%T)] $n: leftovers uploaded, $(ls $R/archive | wc -l) files remain"
done
