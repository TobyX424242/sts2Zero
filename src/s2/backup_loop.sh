#!/bin/bash
# Backs up a run's self-play training chunks to Nextcloud (owner, 2026-11-11: the data is paid for, keep it).
# Every 2 min: zstd-compress the chunks the trainer has archived (~56x smaller) and rclone-move them to
# nextcloud:sts2_backup/s2/<name>/data (local copies go only after a confirmed upload). Disk guard: above 80% use,
# uncompressed archived chunks are dropped (logged) so actors cannot crash on a full disk. When <run>/.done appears:
# a last pass, then the run's weights, checkpoints, logs and fight records go to .../<name>/meta.
# usage: src/s2/backup_loop.sh RUN_DIR NAME
R=$1; NAME=$2; DEST=nextcloud:sts2_backup/s2/$NAME; A=$R/archive; L=$R/logs/backup.txt
mkdir -p $A
pass_() {
  for f in $A/*.s2d; do [ -e "$f" ] || break; zstd -3 -q --rm "$f" -o "$f.zst" 2>>$L || echo "[$(date +%T)] zstd failed $f" >> $L; done
  rclone move $A $DEST/data --include "*.zst" --transfers 4 --retries 5 >> $L 2>&1 || echo "[$(date +%T)] rclone move failed (will retry)" >> $L
  d=$(df / | awk 'NR==2{print $5}' | tr -d %)
  if [ "$d" -ge 80 ]; then n=$(ls $A/*.s2d 2>/dev/null | wc -l); rm -f $A/*.s2d; echo "[$(date +%T)] disk ${d}%: dropped $n unarchived chunks" >> $L; fi
}
echo "[$(date +%T)] backup loop start -> $DEST" >> $L
while [ ! -f $R/.done ]; do pass_; sleep 120; done
pass_; pass_
rclone copy $R $DEST/meta --exclude "archive/**" --exclude "data/**" --transfers 4 --retries 5 >> $L 2>&1 && echo "[$(date +%T)] meta uploaded" >> $L
echo "[$(date +%T)] backup loop done: $(rclone size $DEST/data 2>/dev/null | tr '\n' ' ')" >> $L
