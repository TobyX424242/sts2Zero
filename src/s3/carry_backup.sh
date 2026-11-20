#!/bin/bash
# Off-host copy of a segment's trainer carries (2026-11-19, interruptible servers; s3_supervise.sh CARRY_BACKUP_EVERY):
# carry_combat.bin and carry_macro.pt of SEG_DIR go to DEST/<segment>/ as zstd streams split into 90 MB parts (Nextcloud
# refuses single uploads over ~100 MB), plus manifest.txt (segment, per file: size, sha256 of the uncompressed file,
# parts, compressed bytes), uploaded LAST: a set without manifest.txt is incomplete. After the parts are verified on the
# remote (count and bytes per file) and the manifest is up, the other sets in DEST are purged. The files are hard-linked
# into STAGE first, so a carry deleted meanwhile by run_s3.sh (two segments later) stays readable until the copy ends.
#   src/s3/carry_backup.sh SEG_DIR DEST STAGE      e.g. ... src/s3/runs/s3ic_s006 nextcloud:sts2_backup/s3ic/s3ic/carry src/s3/runs/s3ic_carryup
# Restore: src/s3/restore_from_nextcloud.sh. Last output line = the result (the supervisor logs it).
R=${1%/}; DEST=$2; ST=$3; SN=$(basename $R)
cd "$(dirname "$0")/../.."
say() { echo "[$(date -u +%F\ %T)] $*"; }
fail() { say "FAILED $SN: $*"; rm -rf $ST; exit 1; }
t0=$(date +%s)
rm -rf $ST; mkdir -p $ST/up || fail "cannot create $ST"
FILES="carry_combat.bin carry_macro.pt"
[ -f $R/carry_shadow.bin ] && FILES="$FILES carry_shadow.bin"
for f in $FILES; do ln $R/$f $ST/$f 2>/dev/null || cp $R/$f $ST/$f || fail "$R/$f missing"; done
M=$ST/up/manifest.txt
{ echo "segment $SN"; echo "created $(date -u +%FT%TZ) on $(hostname)"; } > $M
info=""
for f in $FILES; do
  sz=$(stat -c %s $ST/$f); sha=$(sha256sum $ST/$f | cut -c1-64)
  P=$ST/up/$f.zst.parts; mkdir -p $P
  zstd -q -3 -T8 -c $ST/$f | split -b 90M -d -a 4 - $P/part. || fail "zstd/split of $f"
  n=$(ls $P | wc -l); zb=$(cat $P/part.* | wc -c)
  echo "file $f size $sz sha256 $sha parts $n zst_bytes $zb" >> $M
  info="$info $f $((sz / 1000000)) MB -> $((zb / 1000000)) MB ($n parts);"
  rm -f $ST/$f
done
say "compressed in $(( $(date +%s) - t0 )) s:$info uploading to $DEST/$SN"
rclone copy $ST/up $DEST/$SN --exclude manifest.txt --transfers 4 --retries 5 --low-level-retries 10 || fail "rclone copy"
while read -r k f x sz y sha z n w zb; do
  [ "$k" = "file" ] || continue
  rn=$(rclone lsf $DEST/$SN/$f.zst.parts 2>/dev/null < /dev/null | grep -c '^part\.')
  rb=$(rclone lsl $DEST/$SN/$f.zst.parts 2>/dev/null < /dev/null | awk '{s+=$1} END{print s+0}')
  [ "$rn" = "$n" ] && [ "$rb" = "$zb" ] || fail "remote check of $f: $rn/$n parts, $rb/$zb bytes"
done < $M
rclone copyto $M $DEST/$SN/manifest.txt --retries 5 || fail "manifest upload"
old=""
for d in $(rclone lsf --dirs-only $DEST 2>/dev/null); do
  d=${d%/}; [ "$d" = "$SN" ] && continue
  rclone purge $DEST/$d && old="$old $d"
done
rm -rf $ST
say "ok $SN -> $DEST/$SN in $(( $(date +%s) - t0 )) s:$info older sets removed:${old:- none}"
