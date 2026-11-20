#!/bin/bash
# Backs up an S3 run's training chunks (combat *.s2d and macro *.s3r, archived by the trainers) to Nextcloud; same
# rules as src/s2/backup_loop.sh: zstd, upload, delete local copies only after a confirmed upload, drop
# unarchived chunks above 80% disk use (logged), final pass and run outputs when <run>/.done appears.
# usage: src/s3/backup_loop.sh RUN_DIR NAME   (env KEEP_EVERY, below; NC_ROOT: the Nextcloud folder, default
# nextcloud:sts2_backup/s3)
R=$1; NAME=$2; DEST=${NC_ROOT:-nextcloud:sts2_backup/s3}/$NAME; A=$R/archive; L=$R/logs/backup.txt
mkdir -p $A
# KEEP_EVERY=K (default 4; owner, 2026-11-14: "only upload 1/4 of training data"): only chunks whose sequence number
# is a multiple of K are uploaded; the others are deleted (the trainers have read them). 1 = upload everything.
# Logs, configuration and weights (the meta copy at the end) are always uploaded in full.
KEEP_EVERY=${KEEP_EVERY:-4}
thin_() {
  [ "$KEEP_EVERY" -gt 1 ] || return 0
  local n=0 f b
  for f in $A/*.s2d $A/*.s3r $A/*.s2d.zst $A/*.s3r.zst; do
    [ -e "$f" ] || continue
    b=${f##*_}; b=${b%%.*}
    [[ $b =~ ^[0-9]+$ ]] || continue
    if [ $((10#$b % KEEP_EVERY)) -ne 0 ]; then rm -f "$f"; n=$((n + 1)); fi
  done
  [ $n -gt 0 ] && echo "[$(date +%T)] not uploaded (1 in $KEEP_EVERY kept): $n chunks deleted" >> $L
  return 0
}
pass_() {
  thin_
  for f in $A/*.s2d $A/*.s3r; do [ -e "$f" ] || continue; zstd -3 -q --rm "$f" -o "$f.zst" 2>>$L || echo "[$(date +%T)] zstd failed $f" >> $L; done
  rclone move $A $DEST/data --include "*.zst" --transfers 8 --retries 5 >> $L 2>&1 || echo "[$(date +%T)] rclone move failed (will retry)" >> $L
  d=$(df / | awk 'NR==2{print $5}' | tr -d %)
  if [ "$d" -ge 80 ]; then n=$(ls $A/*.s2d $A/*.s3r 2>/dev/null | wc -l); rm -f $A/*.s2d $A/*.s3r; echo "[$(date +%T)] disk ${d}%: dropped $n unarchived chunks" >> $L; fi
}
echo "[$(date +%T)] backup loop start -> $DEST" >> $L
# a pass every 2 minutes; the end of the run is noticed within 5 s (the run's shell waits for this loop)
while [ ! -f $R/.done ]; do pass_; for i in $(seq 24); do [ -f $R/.done ] && break; sleep 5; done; done
pass_; pass_
# FIGHT_KEEP_EVERY=K (> 1): only 1 actor's fight log in K is kept and uploaded (owner, 2026-11-18); default 1 = all
[ "${FIGHT_KEEP_EVERY:-1}" -gt 1 ] && src/s3/thin_fights.sh $R $FIGHT_KEEP_EVERY $L
# rclone's output goes to a separate file: logs/backup.txt changing while it is uploaded failed the copy (s3ic_s000,
# "request body larger than specified"), and then the segment was never confirmed as uploaded (prune_old)
ML=$R/../$(basename $R)_meta_rclone.txt
rclone copy $R $DEST/meta --exclude "archive/**" --exclude "cdata/**" --exclude "mdata/**" --exclude "carry_*" --exclude "logs/backup.txt" --transfers 4 --retries 5 > $ML 2>&1 \
  && echo "[$(date +%T)] meta uploaded" >> $L || { echo "[$(date +%T)] meta upload failed: $(grep -c ERROR $ML) errors (prune_old retries)" >> $L; cat $ML >> $L; }
rm -f $ML
echo "[$(date +%T)] backup loop done: $(rclone size $DEST/data 2>/dev/null | tr '\n' ' ')" >> $L
rclone copy $L $DEST/meta/logs --retries 3 > /dev/null 2>&1
