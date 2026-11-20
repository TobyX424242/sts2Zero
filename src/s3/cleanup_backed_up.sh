#!/bin/bash
# Owner 2026-11-15: "if data has been backed up to Nextcloud, clean up for space" (fight logs stay local, owner earlier).
# For every segment directory of chain s3a except last_good: deletes local checkpoint folders (ckpt, mckpt, shadow_ckpt),
# and for segments < KEEP_FROM the top-level .pt files, but only when the segment's Nextcloud backup is complete (.done)
# and `rclone check --one-way --size-only` confirms every local file is on Nextcloud with the same size. Carry files are
# deleted only if the same check passes for them. Everything not verified is kept and listed.
#   setsid nohup src/s3/cleanup_backed_up.sh [KEEP_FROM] > /root/sts2/cleanup.log 2>&1 < /dev/null &
KEEP_FROM=${1:-44}
cd /root/sts2/src/s3/runs
LG=$(basename $(dirname $(cut -d" " -f1 s3a/last_good)))
echo "[$(date -u +%T)] start; last_good $LG (untouched); top-level .pt kept from segment $KEEP_FROM; disk $(df -h / | awk 'NR==2{print $5}')"
freed=0; kept=""
for d in s3a_s[0-9][0-9][0-9]; do
  [ "$d" = "$LG" ] && continue
  R=nextcloud:sts2_backup/s3/$d/meta
  if ! timeout 120 rclone lsf $R/.done > /dev/null 2>&1 || [ -z "$(timeout 120 rclone lsf $R/.done 2>/dev/null)" ]; then
    kept="$kept $d(no .done)"; continue
  fi
  k=$((10#${d#s3a_s}))
  items="ckpt mckpt shadow_ckpt"
  [ $k -lt $KEEP_FROM ] && items="$items weights.pt macro.pt shadow.pt shadow_v4_stopped.pt"
  items="$items carry_combat.bin carry_macro.pt carry_shadow.bin"
  for it in $items; do
    [ -e $d/$it ] || continue
    sz=$(du -sk $d/$it | cut -f1)
    if [ -d $d/$it ]; then
      [ -z "$(ls -A $d/$it)" ] && { rmdir $d/$it; continue; }
      timeout 900 rclone check $d/$it $R/$it --one-way --size-only > /tmp/rc_$$.txt 2>&1 && ok=1 || ok=0
    else
      timeout 900 rclone check $d $R --one-way --size-only --include "/$it" > /tmp/rc_$$.txt 2>&1 && ok=1 || ok=0
    fi
    if [ $ok = 1 ]; then rm -rf $d/$it; freed=$((freed + sz)); echo "deleted $d/$it ($((sz / 1024)) MB, verified)"
    else kept="$kept $d/$it"; fi
  done
done
rm -f /tmp/rc_$$.txt
echo "[$(date -u +%T)] done: freed $((freed / 1024)) MB; disk $(df -h / | awk 'NR==2{print $5}')"
echo "kept (not verified on Nextcloud):$kept"
echo "[$(date -u +%F\ %T)] owner: local checkpoint/weight/carry files verified on Nextcloud deleted (fight logs kept): $((freed / 1024)) MB freed; see /root/sts2/cleanup.log" >> s3a/supervisor.log
