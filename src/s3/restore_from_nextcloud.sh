#!/bin/bash
# Restores a supervised S3 chain from Nextcloud onto a fresh server (2026-11-19; for the loss of an interruptible host).
#   NC_ROOT=nextcloud:sts2_backup/s3ic src/s3/restore_from_nextcloud.sh NAME [TREE]
# TREE (default: the tree this script is in) must hold the code (src/, .venv, dotnet, the S3 build) and whatever
# config.env / extra.env reference outside the chain (e.g. src/s3/runs/s3ic_init/); the script warns about missing
# ones. Read-only on Nextcloud. Steps:
#   1. $NC_ROOT/NAME (the chain directory the supervisor pushes: config.env, extra.env, end_time, last_good, heartbeat,
#      curriculum, vmix, segment.env, hook_env, eval/, curve.txt, logs ...) -> TREE/src/s3/runs/NAME (carry/ excluded);
#   2. every segment folder $NC_ROOT/NAME_sNNN: its meta copy without fights/, ckpt/, mckpt/, shadow_ckpt/ (weights, macro,
#      runs/, logs, train logs) -> TREE/src/s3/runs/NAME_sNNN; a segment without meta (interrupted on the old host)
#      becomes an empty directory, so the numbering continues after it. Every restored segment gets a .verdict, so the
#      supervisor does not try to adopt one (the last good one: ok, the others: restored);
#   3. the carry set $NC_ROOT/NAME/carry/<last good segment> (carry_backup.sh) when its manifest names that segment:
#      parts downloaded, counted, decompressed, sha256-checked and renamed into the segment directory, where run_s3.sh
#      loads them. No matching set: the next segment starts with empty replay windows and fresh optimizer states (the
#      trainers log "no carry file ...: fresh start"); OLDER_CARRY=1 puts the newest older set into the last good segment
#      instead (replay window and Adam state from that older segment, weights from the last good one).
#   4. checks: last_good weights load and are finite (5 macro members), referenced files exist.
# Then start it like any chain: install the s3_resume.sh cron line (it moves end_time by the downtime since the last
# pushed heartbeat, at most ~5 min stale) or run src/s3/s3_resume.sh once.
# Refuses when TREE/src/s3/runs/NAME/end_time exists (FORCE=1 overwrites files there) or its supervisor runs.
NAME=$1; T=${2:-$(cd "$(dirname "$0")/../.." && pwd)}; NC=${NC_ROOT:-nextcloud:sts2_backup/s3}
[ -n "$NAME" ] || { echo "usage: NC_ROOT=... $0 NAME [TREE]"; exit 2; }
T=$(cd $T && pwd) || exit 1; cd $T
S=src/s3/runs/$NAME; RL=$T/restore_$NAME.log
say() { echo "[$(date -u +%F\ %T)] restore $NAME: $*" | tee -a $RL; }
ps -eo cmd | grep -q "[s]3_supervise.sh $NAME " && { say "a supervisor of $NAME is running here: nothing done"; exit 1; }
[ -f $S/end_time ] && [ "${FORCE:-0}" != "1" ] && { say "$S/end_time exists (FORCE=1 to overwrite)"; exit 1; }
# (2026-11-19: Nextcloud answered 502 Bad Gateway for a while, which used to read as "no end_time": retried, and an
# error other than "not found" is reported as such)
for i in 1 2 3 4 5; do
  err=$(rclone lsf $NC/$NAME/end_time 2>&1 >/dev/null) && break
  echo "$err" | grep -qiE "not found|doesn't exist|directory not found" && { say "no $NC/$NAME/end_time on Nextcloud"; exit 1; }
  [ $i = 5 ] && { say "Nextcloud not answering ($(echo "$err" | tail -n 1 | cut -c1-200)): try again later"; exit 1; }
  sleep 30
done
say "from $NC into $T"
mkdir -p $S
rclone copy $NC/$NAME $S --exclude "carry/**" --transfers 8 --retries 5 >> $RL 2>&1 || { say "chain directory download FAILED"; exit 1; }
[ -f $S/last_good ] || { say "no last_good: the chain will start from COMBAT0 / MACRO0 of config.env"; }
CW=""; MW=""; LG=""; [ -f $S/last_good ] && { read CW MW < $S/last_good; LG=$(basename $(dirname $CW)); }
say "chain directory: $(du -sh $S | cut -f1); last good: ${LG:-none} ($CW | $MW)"
n=0; nm=0
for sd in $(rclone lsf --dirs-only $NC 2>/dev/null | grep -E "^${NAME}_s[0-9]{3}/$" | tr -d /); do
  mkdir -p src/s3/runs/$sd; n=$((n + 1))
  if rclone lsf $NC/$sd/meta --max-depth 1 2>/dev/null | grep -q .; then
    rclone copy $NC/$sd/meta src/s3/runs/$sd --exclude "fights/**" --exclude "ckpt/**" --exclude "mckpt/**" \
      --exclude "shadow_ckpt/**" --transfers 8 --retries 5 >> $RL 2>&1 || say "segment $sd: meta download FAILED"
    nm=$((nm + 1))
  fi
  [ "$sd" = "$LG" ] && echo ok > src/s3/runs/$sd/.verdict || echo restored > src/s3/runs/$sd/.verdict
done
say "segments: $n folders, $nm with meta ($(du -sch src/s3/runs/${NAME}_s[0-9][0-9][0-9] 2>/dev/null | tail -n 1 | cut -f1))"
# carries
carry_from() {   # $1 = carry set (segment name) on Nextcloud -> the last good segment directory
  local C=$NC/$NAME/carry/$1 D=src/s3/runs/$LG X=src/s3/runs/$LG/.carry_dl ok=1
  rm -rf $X; mkdir -p $X
  rclone copyto $C/manifest.txt $X/manifest.txt > /dev/null 2>&1 || { say "carry set $1: no manifest"; rm -rf $X; return 1; }
  while read -r k f x sz y sha z np w zb; do
    [ "$k" = "file" ] || continue
    rclone copy $C/$f.zst.parts $X/$f.zst.parts --transfers 8 --retries 5 >> $RL 2>&1 < /dev/null || { ok=0; break; }
    [ "$(ls $X/$f.zst.parts | grep -c '^part\.')" = "$np" ] && [ "$(cat $X/$f.zst.parts/part.* | wc -c)" = "$zb" ] || { say "carry $f: part count / bytes differ from the manifest"; ok=0; break; }
    cat $X/$f.zst.parts/part.* | zstd -q -d -T8 -o $D/$f.tmp -f || { ok=0; break; }
    rm -rf $X/$f.zst.parts
    [ "$(stat -c %s $D/$f.tmp)" = "$sz" ] && [ "$(sha256sum $D/$f.tmp | cut -c1-64)" = "$sha" ] || { say "carry $f: size or sha256 differs"; ok=0; break; }
    mv $D/$f.tmp $D/$f; say "carry $f restored ($((sz / 1000000)) MB, sha256 ok) from set $1"
  done < $X/manifest.txt
  rm -rf $X
  [ $ok = 1 ] || { rm -f $D/carry_combat.bin* $D/carry_macro.pt* $D/carry_shadow.bin*; say "carry set $1 NOT restored"; return 1; }
}
if [ -n "$LG" ]; then
  sets=$(rclone lsf --dirs-only $NC/$NAME/carry 2>/dev/null | tr -d / | sort | tr '\n' ' ')
  if [[ " $sets " == *" $LG "* ]] && carry_from $LG; then :
  elif [ "${OLDER_CARRY:-0}" = "1" ] && [ -n "$sets" ] && carry_from $(echo $sets | tr ' ' '\n' | tail -n 1); then
    say "OLDER_CARRY=1: the last good segment $LG continues with the replay windows / optimizer states of an older set"
  else
    say "no usable carry set for $LG on Nextcloud (sets: ${sets:-none}): the next segment starts with EMPTY replay windows and fresh optimizer states"
  fi
fi
# checks
if [ -n "$CW" ]; then
  ./.venv/bin/python -I -c "
import sys, torch
a = torch.load(sys.argv[1], map_location='cpu', weights_only=False); b = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
assert 'state_dict' in a and len(b['members']) == 5
assert all(torch.isfinite(v).all() for v in a['state_dict'].values() if v.is_floating_point())
assert all(torch.isfinite(v).all() for m in b['members'] for v in m.values() if v.is_floating_point())
" $CW $MW && say "last good weights load and are finite" || say "WARNING: last good weights do not load ($CW | $MW)"
fi
for f in $(cat $S/config.env $S/extra.env 2>/dev/null | grep -o "src/[A-Za-z0-9_./-]*\.\(pt\|bin\|sh\|json\)" | sort -u); do
  [ -e "$f" ] || say "WARNING: $f (referenced in config.env / extra.env) is missing in $T"
done
rm -f $S/pause src/s3/runs/${NAME}_s[0-9][0-9][0-9]/.finishing src/s3/runs/${NAME}_s[0-9][0-9][0-9]/.prune_skipped
hb=$(cat $S/heartbeat 2>/dev/null)
# pause files are never deleted on Nextcloud (pushes are rclone copy --update): a pause older than the last heartbeat
# was resumed long ago (2026-11-19: a stale 03:11 pause made resume_chain.sh add 6.7 h instead of the real downtime)
pa=$(cat $S/paused_at 2>/dev/null)
if [ -f $S/end_time.paused ] && [[ "$hb" =~ ^[0-9]+$ ]] && [[ "$pa" =~ ^[0-9]+$ ]] && [ $hb -gt $((pa + 120)) ]; then
  rm -f $S/end_time.paused $S/paused_at
  say "stale pause files (paused $(date -u -d @$pa +%F\ %T), heartbeat later) removed: the chain is not paused"
fi
say "done. end_time $(date -u -d @$(cat $S/end_time) +%F\ %T) UTC, last heartbeat $([[ "$hb" =~ ^[0-9]+$ ]] && date -u -d @$hb +%F\ %T || echo none) UTC (s3_resume.sh adds the downtime since then)$([ -f $S/end_time.paused ] && echo '; the chain is PAUSED: src/s3/resume_chain.sh')"
