#!/bin/bash
# Unattended S3 training on the rented server (no laptop watchdog needed).
#   setsid nohup src/s3/s3_supervise.sh NAME HOURS > /root/sts2/s3_supervise_NAME.log 2>&1 < /dev/null &
# The run is a chain of segments (run_s3.sh NAME_sNNN), each started from the previous segment's weights. Per segment:
#   - a guard checks every minute: trainers and servers alive, enough actors alive, new runs arriving, disk below 90%,
#     and a hard time limit; on a violation the whole segment (its process group) is stopped and the next one starts
#     from the last good weights;
#   - afterwards: weights are validated (a segment that produced no usable weights or fewer than MIN_RUNS (100) runs does not
#     advance the chain; three such segments in a row stop the supervisor), leftover training chunks are compressed and
#     uploaded, the agent is evaluated on fixed seeds at the target ascension (learning curve in curve.txt), and status,
#     logs, curve, evaluations and weights are copied to nextcloud:sts2_backup/s3/NAME/.
# status.txt (also on Nextcloud, refreshed every 5 minutes) says what is running and when the run ends.
# The end time is stored in the run directory: starting the supervisor again with the same NAME resumes the chain.
# src/s3/s3_resume.sh (cron, every 10 minutes; every minute on an interruptible server) restarts the supervisor of a
# run that has time left if the container was restarted or the supervisor process died (section "Interruptible servers").
# At the end: STOP_INSTANCE=1 stops the Vast.ai instance (GPU charges end; the disk is kept and still billed).
# env: N (actors, 26), SEG (segment seconds, 7200), ASC (target ascension, 10), ASC_MIX (share of runs at ASC, 0.5),
#      CONC (32), CSERVERS (2), RUNS_PER_ITER (1024), KL_RUNS (KL-to-S1 weight reaches zero after this many runs, 150000),
#      EVAL_COUNT (seeds per evaluation, 120; 0 = none), EVAL_EVERY (segments between evaluations, 1), EVAL_N (evaluation actors, 30),
#      PPO_EPOCHS (10) and PPO_TARGET_KL (0.012): PPO passes over each batch of runs until KL(policy at the start of the
#      update || now) exceeds 1.5 x the target, at most PPO_EPOCHS passes,
#      COMBAT0 / MACRO0 (starting weights), COMBAT_WINDOW (combat replay window, 2000000), STOP_INSTANCE (0)
#      CTRAIN_ARGS (extra combat-trainer arguments, e.g. "--surprise 0.5 --ema 0.999"; stored with the run)
# Chain options (2026-11-18, ic113 chain s3ic; all unset = the behaviour above, unchanged):
#      NC_ROOT (Nextcloud folder of the run and its segments, default nextcloud:sts2_backup/s3; exported for backup_loop.sh),
#      KEEP_EVERY (chunks uploaded 1 in K, 4), FIGHT_KEEP_EVERY (fight logs kept/uploaded 1 actor in K, 1 = all),
#        PRUNE_KEEP (keep the bulk of only the newest N segments, 0 = none; unset = keep all), see prune_old,
#      ACTOR_MEM_GB (kill a self-play actor whose resident memory exceeds this many GB, checked every minute; unset = off;
#        the ic113 warm-up saw hung simulator actors grow to 33 GB, src/ic113/WARMUP.md),
#      EVAL_SEEDS / EVAL_FIRST (seed prefix and first index of the evaluation panel; EVAL / 0), EVAL_ARM (the evaluation arm,
#        eval_s3.sh "name:sims"; tf64:64),
#      CHAIN_HOOK (script, e.g. src/ic113/s3ic_hook.sh): `CHAIN_HOOK segment S NEXT_SEG LAST_GOOD_SEG_DIR MACRO` runs
#        before a segment whose settings it has not written yet (chain start, after each good segment, and on a restart
#        whose segment.env is for another segment) and writes $S/segment.env, which is read before every segment and
#        evaluation: ASC_MIX (overrides the stored one), SEG_ACTOR_ARGS (added to the actors' arguments), EVAL_ARGS (added
#        to the evaluation actors' arguments, eval_s3.sh EVAL_ACTOR_ARGS), HOOK_FOR_SEG; its output lines are logged;
#        `CHAIN_HOOK eval S EVAL_DIR SEG` runs after each evaluation. Each segment.env is also kept as
#        $S/hook_env/seg_NNN.env (the evaluation of segment K uses the one written for K + 1, also when re-run).
# Interruptible servers (2026-11-19; src/ic113/CHAIN.md "Interruptible server"): the container can be stopped at any
# moment without a signal (all processes gone, /dev/shm empty, disk kept) and started again later.
#   - Periodic carries: with CARRY=1 both trainers also write their carry files every CARRY_EVERY seconds (run_s3.sh,
#     default 1800).
#   - Adoption: at a (re)start, the newest segment directory that has no verdict (.verdict, written for every segment
#     this supervisor judged), is not the last good one and has no live process is checked like a finished segment
#     (>= MIN_RUNS runs, both weight files load and are finite, 5 macro members) and, with CARRY=1, must also have both
#     carry files (complete: combat trailer, macro zip directory). Then it becomes last_good ("segment K recovered after
#     interruption"), its leftover chunks and meta are uploaded (finish_backup: its backup loop died with it), the chain
#     hook runs for K + 1 from it and it is evaluated. Without its carries it is not adopted and the chain continues
#     from the previous last_good (whose carries run_s3.sh kept): the next segment loads its carries from
#     dirname(COMBAT_INIT), so weights of K cannot be combined with carries of K - 1; at most CARRY_EVERY s of training
#     are lost that way. A segment that fails the check is logged and skipped (as a failed segment).
#   - Missed evaluations: at a start, the last good segment (and the one before an adoption) is evaluated again in the
#     background when it is due (EVAL_COUNT > 0, segment % EVAL_EVERY = 0) and eval/seg_NNN/summary.txt has no line of
#     the evaluation arm (interrupted evaluation); the partial output is removed first.
#   - Heartbeat: $S/heartbeat (epoch seconds) is written every minute while the supervisor lives; s3_resume.sh moves
#     end_time later by the time since the heartbeat when it finds no supervisor (downtime does not count).
#   - Segment numbers continue after the highest segment directory (not their count: a restored chain may lack some).
#   - Stale state at a start: *.tmp carry files and finish_backup markers of the chain's segments are removed.
#   - CARRY_BACKUP_EVERY=K (config.env; unset/0 = off): after a good segment whose number % K == 0 its two carry files
#     go to $NC_ROOT/$NAME/carry/<segment>/ in the background (src/s3/carry_backup.sh: zstd, 90 MB parts,
#     manifest with sizes and sha256 written last; older sets deleted after a verified upload), for a restore on a new
#     host after the old one is lost (src/s3/restore_from_nextcloud.sh). Never blocks the next segment; one at a time.
NAME=$1; HOURS=$2
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
# the settings of a run are stored with it, so a restart (same NAME, any HOURS) continues with the same settings
CFG=src/s3/runs/$NAME/config.env
if [ -f $CFG ]; then . $CFG; else
  mkdir -p src/s3/runs/$NAME
  for v in N SEG ASC ASC_MIX CONC CSERVERS RUNS_PER_ITER KL_RUNS EVAL_COUNT EVAL_EVERY EVAL_N COMBAT0 MACRO0 STOP_INSTANCE MIN_RUNS CHUNK MCHUNK COMBAT_WINDOW PPO_EPOCHS PPO_TARGET_KL CTRAIN_ARGS NC_ROOT EVAL_SEEDS EVAL_FIRST EVAL_ARM CHAIN_HOOK KEEP_EVERY FIGHT_KEEP_EVERY PRUNE_KEEP ACTOR_MEM_GB CARRY_EVERY CARRY_BACKUP_EVERY EVAL_GPU; do
    [ -n "${!v}" ] && echo "export $v='${!v}'" >> $CFG
  done
  touch $CFG
fi
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
N=${N:-26}; SEG=${SEG:-7200}; ASC=${ASC:-10}; ASC_MIX=${ASC_MIX:-0.5}; CONC=${CONC:-32}; CSERVERS=${CSERVERS:-2}
export COMBAT_WINDOW=${COMBAT_WINDOW:-2000000}
RUNS_PER_ITER=${RUNS_PER_ITER:-1024}; KL_RUNS=${KL_RUNS:-150000}; EVAL_COUNT=${EVAL_COUNT:-120}; EVAL_EVERY=${EVAL_EVERY:-1}
EARM=${EVAL_ARM:-tf64:64}
export NC_ROOT=${NC_ROOT:-nextcloud:sts2_backup/s3}
export KEEP_EVERY=${KEEP_EVERY:-4} FIGHT_KEEP_EVERY=${FIGHT_KEEP_EVERY:-1}
S=src/s3/runs/$NAME; mkdir -p $S/eval; ST=$S/status.txt; DEST=$NC_ROOT/$NAME
CW=${COMBAT0:-src/s3/weights/combat_s2p1.pt}; MW=${MACRO0:-src/s3/data/macro_s1.pt}
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $S/supervisor.log; }
# heartbeat (2026-11-19): epoch seconds every minute while this supervisor lives; s3_resume.sh adds the time since the
# last beat to end_time when it finds the chain without a supervisor (container stopped, server lost and restored)
heartbeat_loop() { while kill -0 $1 2>/dev/null; do date +%s > $S/heartbeat.tmp && mv $S/heartbeat.tmp $S/heartbeat; sleep 60; done; }
heartbeat_loop $$ &
HB_PID=$!
[ -f $S/end_time ] || echo $(( $(date +%s) + $(awk "BEGIN{printf \"%d\", $HOURS * 3600}") )) > $S/end_time
END=$(cat $S/end_time)
[ -f $S/last_good ] && { read CW MW < $S/last_good; }
seg_dirs() { ls -d src/s3/runs/${NAME}_s[0-9][0-9][0-9] 2>/dev/null; }
seg_of() { local b=${1%/}; b=${b##*_s}; echo $((10#$b)); }   # segment number of a segment directory
# next segment = highest existing number + 1 (2026-11-19; was the count of directories, the same for a gapless chain)
last_dir=$(seg_dirs | tail -n 1); seg=0; [ -n "$last_dir" ] && seg=$(( $(seg_of $last_dir) + 1 )); fails=0
total_runs() { cat src/s3/runs/${NAME}_s*/runs/*.jsonl 2>/dev/null | wc -l; }
runs_upto() { local d n=0; for d in $(seg_dirs); do [ $(seg_of $d) -le $1 ] && n=$((n + $(cat $d/runs/*.jsonl 2>/dev/null | wc -l))); done; echo $n; }
status() {
  {
    echo "S3 run $NAME on $(hostname): $1"
    echo "updated $(date -u +%F\ %T) UTC; ends $(date -u -d @$END +%F\ %T) UTC ($(( (END - $(date +%s)) / 60 )) min left); segment $seg; runs so far $(total_runs)"
    echo "weights in use: $CW | $MW"
    [ -f $S/hook_status.txt ] && echo "chain settings: $(cat $S/hook_status.txt)"
    echo "disk $(df / | awk 'NR==2{print $5}'), load $(cut -d' ' -f1 /proc/loadavg), gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)"
    echo "--- learning curve (fixed seeds, ascension $ASC, greedy macro, search 64)"; cat $S/curve.txt 2>/dev/null
    echo "--- self-play by training order, last two segments (runs with errors or perturbed starts excluded)"
    python3 src/s3/curve_runs.py $(ls -d src/s3/runs/${NAME}_s[0-9][0-9][0-9] 2>/dev/null | tail -n 2) --buckets 6 --asc $ASC 2>/dev/null | cut -c1-330
    echo "--- last macro-trainer lines"; tail -n 2 src/s3/runs/${NAME}_s*/macro_train_log.jsonl 2>/dev/null | cut -c1-700 | tail -n 4
    echo "--- supervisor log"; tail -n 12 $S/supervisor.log
  } > $ST.tmp; mv $ST.tmp $ST
}
# copies status, logs, curve and evaluations to Nextcloud in the background (WebDAV takes seconds per file; training
# must not wait for it); a copy still running is not started twice. `push wait` blocks, for the end of the run.
# 2026-11-17: uploads go from a snapshot (rsync to /dev/shm, mtimes kept) instead of the live directory: files that grow
# during the upload (rclone.log itself, evaluation outputs) failed with "request body larger than specified content
# length" / "corrupted on transfer: sha1 hash differ". rclone.log is uploaded as its last 400 lines (rclone_tail.log).
push_once() {
  local G=/dev/shm/push_$NAME
  rm -rf $G; mkdir -p $G
  rsync -a --exclude "*.tmp" --exclude rclone.log --exclude pause $S/ $G/ 2>/dev/null
  tail -n 400 $S/rclone.log > $G/rclone_tail.log 2>/dev/null
  timeout 900 rclone copy $G $DEST --update --transfers 8 --retries 3 >> $S/rclone.log 2>&1
  rm -rf $G
}
push() {
  if [ "$1" = "wait" ]; then [ -n "$PUSH_PID" ] && wait $PUSH_PID 2>/dev/null; push_once; return; fi
  [ -n "$PUSH_PID" ] && kill -0 $PUSH_PID 2>/dev/null && return
  push_once &
  PUSH_PID=$!
}
PUSH_PID=""
kill_group() {   # $1 = process-group leader
  kill -TERM -- -$1 2>/dev/null; sleep 8; kill -KILL -- -$1 2>/dev/null; sleep 1
}
alive_all() { for p in $(cat $1 2>/dev/null); do kill -0 $p 2>/dev/null || return 1; done; return 0; }
finish_backup() {   # $1 = segment directory, $2 = segment name: chunks the segment's own backup loop did not upload
  local R=$1 D=$NC_ROOT/$2
  touch $R/.finishing   # prune_old leaves the directory alone while this runs (it used to delete archive/ under it)
  mkdir -p $R/archive; mv $R/cdata/*.s2d $R/mdata/*.s3r $R/archive/ 2>/dev/null
  for f in $R/archive/*.s2d $R/archive/*.s3r; do   # 1 chunk in KEEP_EVERY (4) is uploaded, as backup_loop.sh
    [ -e "$f" ] || continue; b=${f##*_}; b=${b%%.*}
    [[ $b =~ ^[0-9]+$ ]] && [ $((10#$b % ${KEEP_EVERY:-4})) -ne 0 ] && rm -f "$f"
  done
  for f in $R/archive/*.s2d $R/archive/*.s3r; do [ -e "$f" ] || continue; zstd -3 -q --rm "$f" -o "$f.zst" 2>/dev/null; done
  timeout 1800 rclone move $R/archive $D/data --include "*.zst" --transfers 4 --retries 5 >> $S/rclone.log 2>&1
  [ "${FIGHT_KEEP_EVERY:-1}" -gt 1 ] && src/s3/thin_fights.sh $R $FIGHT_KEEP_EVERY $R/logs/backup.txt
  timeout 900 rclone copy $R $D/meta --update --exclude "archive/**" --exclude "cdata/**" --exclude "mdata/**" --exclude "carry_*" --exclude "logs/backup.txt" --exclude .finishing --transfers 4 >> $S/rclone.log 2>&1 \
    && echo "[$(date +%T)] meta uploaded (finish_backup)" >> $R/logs/backup.txt
  local d=$(df / | awk 'NR==2{print $5}' | tr -d %)
  [ "$d" -ge 85 ] && { n=$(ls $R/archive 2>/dev/null | wc -l); rm -f $R/archive/*; log "disk ${d}%: dropped $n unuploaded chunk files of $2"; }
  rm -f $R/.finishing
}
# PRUNE_KEEP=N (set): after each segment, segment directories older than the newest N lose their fight logs, mid-segment
# checkpoints (ckpt, mckpt) and leftover chunks, once their meta upload is confirmed in logs/backup.txt; runs/, logs,
# weights and carries stay (carries are managed by run_s3.sh). Owner, 2026-11-18: self-play data deleted after each
# segment, a 1/4 sample kept on Nextcloud (chunks: KEEP_EVERY, fight logs: FIGHT_KEEP_EVERY).
prune_old() {
  [ -n "$PRUNE_KEEP" ] || return 0
  local d
  for d in $(ls -d src/s3/runs/${NAME}_s[0-9][0-9][0-9] 2>/dev/null | head -n -$PRUNE_KEEP); do
    [ -d $d/fights ] || [ -d $d/ckpt ] || [ -d $d/mckpt ] || [ -d $d/shadow_ckpt ] || continue
    if grep -q "meta uploaded" $d/logs/backup.txt 2>/dev/null; then
      local sz=$(du -sh $d/fights $d/ckpt $d/mckpt $d/shadow_ckpt $d/archive 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
      rm -rf $d/fights $d/ckpt $d/mckpt $d/shadow_ckpt $d/archive; log "pruned $(basename $d) (uploaded): fights ckpt mckpt shadow_ckpt archive [$sz]"
    elif ! ps -eo cmd | grep -q "[b]ackup_loop.sh $d " && [ ! -f $d/.finishing ]; then
      # not confirmed and its backup loop has ended (its upload failed): upload the meta again (--update), prune on success
      if timeout 900 rclone copy $d $NC_ROOT/$(basename $d)/meta --update --exclude "archive/**" --exclude "cdata/**" --exclude "mdata/**" --exclude "carry_*" --exclude "logs/backup.txt" --transfers 4 --retries 3 >> $S/rclone.log 2>&1; then
        echo "[$(date +%T)] meta uploaded (prune_old retry)" >> $d/logs/backup.txt
        local sz=$(du -sh $d/fights $d/ckpt $d/mckpt $d/shadow_ckpt $d/archive 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
        rm -rf $d/fights $d/ckpt $d/mckpt $d/shadow_ckpt $d/archive; log "pruned $(basename $d) (uploaded on retry): fights ckpt mckpt shadow_ckpt archive [$sz]"
      else
        log "not pruned $(basename $d): meta upload failed again (see $S/rclone.log); retried at the next boundary"
      fi
    elif [ ! -f $d/.prune_skipped ]; then
      touch $d/.prune_skipped; log "not pruned $(basename $d) yet: its backup loop / finish_backup is still uploading; retried at the next boundary"
    fi
  done
}
# chain hook (CHAIN_HOOK): writes $S/segment.env for segment $1 from the last good segment's directory (if it is a
# segment of this chain) and the macro file; output lines go to the supervisor log
hook_segment() {
  local R0=""; case "$MW" in src/s3/runs/${NAME}_s[0-9][0-9][0-9]/*) R0=$(dirname $MW);; esac
  $CHAIN_HOOK segment $S $1 "$R0" $MW 2>> $S/hook.log | while IFS= read -r l; do log "hook: $l"; done
  [ -f $S/segment.env ] && { mkdir -p $S/hook_env; cp $S/segment.env $S/hook_env/seg_$(printf %03d $1).env; }
}
seg_env() {   # the hook's settings for the next segment / evaluation (none without CHAIN_HOOK)
  SEG_ACTOR_ARGS=""; EVAL_ARGS=""; HOOK_FOR_SEG=""
  [ -n "$CHAIN_HOOK" ] && [ -f $S/segment.env ] && . $S/segment.env
}
# --- segment verdicts, evaluations, carry backup (2026-11-19: shared by the segment end and the restart paths)
validate() {   # $1 = segment directory: >= MIN_RUNS runs and both weight files load and are finite (VRUNS = its runs)
  VRUNS=$(cat $1/runs/*.jsonl 2>/dev/null | wc -l)
  [ "$VRUNS" -ge ${MIN_RUNS:-100} ] && ./.venv/bin/python -I -c "
import sys, torch
a = torch.load(sys.argv[1], map_location='cpu', weights_only=False); b = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
assert 'state_dict' in a and len(b['members']) == 5
assert all(torch.isfinite(v).all() for v in a['state_dict'].values() if v.is_floating_point())
assert all(torch.isfinite(v).all() for m in b['members'] for v in m.values() if v.is_floating_point())
" $1/weights.pt $1/macro.pt 2>> $S/supervisor.log
}
carries_ok() {   # $1 = segment directory: both carry files present and complete (cheap checks; the trainers check fully)
  ./.venv/bin/python -I -c "
import os, sys, zipfile, json
c, m = sys.argv[1] + '/carry_combat.bin', sys.argv[1] + '/carry_macro.pt'
with open(c, 'rb') as f:
    h = f.read(16); assert h[:8] == b'S2CARRY1'
    hdr = json.loads(f.read(int.from_bytes(h[8:16], 'little'))); assert hdr['total'] == os.path.getsize(c)
    f.seek(-16, 2); assert f.read(16) == b'S2CARRYEND000000'
assert zipfile.is_zipfile(m)
" $1 2>/dev/null
}
chain_carry() { ( [ -f $S/extra.env ] && . $S/extra.env > /dev/null 2>&1; echo ${CARRY:-0} ); }
accept_good() {   # $1 = segment directory: the chain continues from it
  CW=$1/weights.pt; MW=$1/macro.pt; echo "$CW $MW" > $S/last_good; echo ok > $1/.verdict
  mkdir -p $S/latest; cp $CW $S/latest/weights.pt; cp $MW $S/latest/macro.pt
}
EARMN=${EARM%%:*}
eval_done() { grep -q "^== $EARMN" $S/eval/seg_$(printf %03d $1)/summary.txt 2>/dev/null; }
eval_due() { [ "$EVAL_COUNT" -gt 0 ] && [ $(( $1 % EVAL_EVERY )) -eq 0 ]; }
eval_one() {   # $1 = segment number: evaluation of that segment's weights (blocks; run it in the background)
  local j=$1 jn=$(printf %03d $1); local R=src/s3/runs/${NAME}_s$jn E=$S/eval/seg_$jn truns=$(runs_upto $1) f EA
  # actor arguments of the hook settings written for the next segment (as at the segment end), else the current ones
  f=$S/hook_env/seg_$(printf %03d $((j + 1))).env; [ -f $f ] || f=$S/segment.env
  EA=$( EVAL_ARGS=""; [ -n "$CHAIN_HOOK" ] && [ -f $f ] && . $f; echo "$EVAL_ARGS" )
  rm -rf $E   # partial output of an interrupted evaluation
  ( COUNT=$EVAL_COUNT ASC=$ASC CONC=12 CSERVERS=$CSERVERS SEEDS=${EVAL_SEEDS:-EVAL} FIRST=${EVAL_FIRST:-0} EVAL_ACTOR_ARGS="$EA" \
      setsid env ${EVAL_GPU:+CUDA_VISIBLE_DEVICES=$EVAL_GPU} src/s3/eval_s3.sh $R/weights.pt $R/macro.pt $E ${EVAL_N:-30} "$EARM" > $S/eval_$j.log 2>&1 < /dev/null &
    EP=$!; t0=$(date +%s)
    while kill -0 $EP 2>/dev/null; do
      sleep 20
      [ $(( $(date +%s) - t0 )) -gt 2400 ] && { log "evaluation of segment $j over 40 min: stopped"; kill_group $EP; break; }
    done
    wait $EP 2>/dev/null; rm -f /dev/shm/s3e_${EARMN}_${EP}_*   # this evaluation's files only (eval_s3.sh: s3e_<arm>_<its pid>_)
    echo "seg $j runs $truns | $(grep -m1 "^== $EARMN" $E/summary.txt 2>/dev/null | cut -c1-200)" >> $S/curve.txt
    log "eval: seg $j runs $truns | $(grep -m1 "^== $EARMN" $E/summary.txt 2>/dev/null | cut -c1-200)"
    [ -n "$CHAIN_HOOK" ] && $CHAIN_HOOK eval $S $E $j 2>> $S/hook.log | while IFS= read -r l; do log "hook: $l"; done )
}
start_evals() {   # segment numbers: evaluated one after another in the background, beside the next segment
  [ $# -gt 0 ] || return 0
  [ $(( $(date +%s) + 1500 )) -lt $END ] || { log "evaluation of segment(s) $* skipped: the run ends within 25 min"; return 0; }
  # one evaluation at a time (2026-11-19: an evaluation re-run after a restart overlapped the next segment's evaluation;
  # with the macro trainer's peak that ran the 32 GB GPU out of memory and killed an inference server): a later call
  # waits on the lock until the running one ends
  ( flock 8; for j in "$@"; do eval_one $j; done ) 8> $S/.eval.lock &
}
CB_PID=""
carry_backup() {   # $1 = segment number, $2 = its directory (a good segment), $3 = "check": only if Nextcloud has no
  # complete set (manifest) for it (supervisor start: an upload cut off by an interruption is done again); header
  [ "${CARRY_BACKUP_EVERY:-0}" -gt 0 ] 2>/dev/null || return 0
  [ $(( $1 % CARRY_BACKUP_EVERY )) -eq 0 ] || return 0
  [ -f $2/carry_combat.bin ] && [ -f $2/carry_macro.pt ] || { log "carry backup of segment $1 skipped: carry files missing"; return 0; }
  if [ -n "$CB_PID" ] && kill -0 $CB_PID 2>/dev/null; then log "carry backup of segment $1 skipped: the previous one is still running"; return 0; fi
  ( [ "$3" = check ] && rclone lsf $NC_ROOT/$NAME/carry/$(basename $2)/manifest.txt 2>/dev/null | grep -q . && exit 0
    [ "$3" = check ] && log "carry backup of segment $1 not on Nextcloud (cut off by an interruption?): uploading it again"
    src/s3/carry_backup.sh $2 $NC_ROOT/$NAME/carry src/s3/runs/${NAME}_carryup >> $S/carry_backup.log 2>&1
    log "carry backup: $(tail -n 1 $S/carry_backup.log | cut -c1-300)" ) &
  CB_PID=$!
}
# --- (re)start: stale state, adoption of an interrupted segment, missed evaluations
rm -f src/s3/runs/${NAME}_s[0-9][0-9][0-9]/carry_*.tmp src/s3/runs/${NAME}_s[0-9][0-9][0-9]/.finishing
log "supervisor start: $NAME, ends $(date -u -d @$END +%F\ %T) UTC, actors $N, segment $SEG s, ascension $ASC (share $ASC_MIX), conc $CONC, start weights $CW | $MW"
EVQ=""; adopted=""
case "$CW" in src/s3/runs/${NAME}_s[0-9][0-9][0-9]/*) EVQ=$(seg_of $(dirname $CW));; esac   # last good before any adoption
if [ -n "$last_dir" ] && [ ! -f $last_dir/.verdict ] && ! [ "$last_dir" -ef "$(dirname $CW)" ]; then
  A=$last_dir; AN=$(basename $A); ak=$(seg_of $A)
  if ps -eo cmd | grep -v -e backup_loop -e rclone | grep -qE "[r]un_s3.sh $AN |[r]uns/$AN/|[s]3_${AN}_"; then
    log "segment $ak ($AN) has live processes: not adopted (stop them, or let s3_resume.sh do it, and restart the supervisor)"
  else
    for p in $(ps -eo pid,cmd | grep "[b]ackup_loop.sh $A " | awk '{print $1}'); do kill $p 2>/dev/null; done   # loops forever without .done
    rm -f /dev/shm/s3_${AN}_*
    cc=$(chain_carry)
    if validate $A && { [ "$cc" != "1" ] || carries_ok $A; }; then
      accept_good $A
      log "segment $ak recovered after interruption: $VRUNS runs, weights $(date -u -r $A/weights.pt +%T) / macro $(date -u -r $A/macro.pt +%T)$([ "$cc" = 1 ] && echo ", carries $(date -u -r $A/carry_combat.bin +%T) / $(date -u -r $A/carry_macro.pt +%T)") UTC; $(tail -n 1 $A/macro_train_log.jsonl 2>/dev/null | cut -c1-300)"
      EVQ="$EVQ $ak"; adopted=$ak
      carry_backup $ak $A
    else
      echo failed > $A/.verdict
      if [ "$VRUNS" -ge ${MIN_RUNS:-100} ] && [ "$cc" = "1" ] && ! carries_ok $A; then why="no complete carry files (killed before the first periodic carry?)"; else why="$VRUNS runs or unusable weights"; fi
      log "segment $ak ($AN) interrupted and NOT adopted: $why; the chain continues from $CW | $MW"
    fi
    finish_backup $A $AN &   # its backup loop died with it
  fi
fi
[ -z "$adopted" ] && [ -n "$EVQ" ] && carry_backup $EVQ $(dirname $CW) check
if [ -n "$CHAIN_HOOK" ]; then seg_env; [ "$HOOK_FOR_SEG" = "$seg" ] || hook_segment $seg; fi
seg_env
EV=""
for j in $EVQ; do
  eval_due $j || continue; eval_done $j && continue
  ps -eo cmd | grep -q "[e]val_s3.sh .*eval/seg_$(printf %03d $j)" && continue
  EV="$EV $j"
done
[ -n "$EV" ] && { log "evaluation of segment(s)$EV missing or incomplete: run again in the background"; start_evals $EV; }
while [ $(( $(date +%s) + 900 )) -lt $END ]; do
  # pause flag (2026-11-16): `touch $S/pause` makes the supervisor stop here, before the next segment starts, with no race
  # (an outside kill after "segment K ok" came too late: the next segment starts within a second). end_time is parked
  # so the cron resume leaves the chain alone; src/s3/resume_chain.sh restores it and restarts the supervisor.
  if [ -f $S/pause ]; then
    rm -f $S/pause; cp $S/end_time $S/end_time.paused; date +%s > $S/paused_at; date +%s > $S/end_time
    log "chain paused before segment $seg (pause flag); last good weights: $CW | $MW"; status "PAUSED before segment $seg"; push
    kill $HB_PID 2>/dev/null; wait; exit 0   # waits for the previous segment's evaluation (background) so its curve line is written
  fi
  END=$(cat $S/end_time)   # re-read each segment (2026-11-14): the end can be moved without restarting the supervisor
  SEG=$(. $CFG > /dev/null 2>&1; echo $SEG)   # segment length from config.env, also re-read each segment (2026-11-14)
  left=$(( END - $(date +%s) )); dur=$(( left - 600 < SEG ? left - 600 : SEG ))
  # combat network switch (2026-11-13): src/s3/switch_combat.sh writes $S/combat_switch; the next segment starts
  # from the shadow (larger) network of the last good segment instead of its playing network
  if [ -f $S/combat_switch ] && [ -f $(dirname $CW)/shadow.pt ]; then
    CW=$(dirname $CW)/shadow.pt; rm -f $S/combat_switch
    log "combat network switched to the shadow network $CW ($(cat $S/combat_switch_reason 2>/dev/null))"
  fi
  SN=${NAME}_s$(printf %03d $seg); R=src/s3/runs/$SN; off=$(total_runs)
  seg_env; [ -n "$CHAIN_HOOK" ] && [ -f $S/segment.env ] && log "segment $seg settings from segment.env (written for segment $HOOK_FOR_SEG): ASC_MIX $ASC_MIX, actor args '$SEG_ACTOR_ARGS'"
  log "segment $seg ($SN): $dur s, from $CW | $MW, runs so far $off"
  CTRAIN_ARGS="$CTRAIN_ARGS" ASC=$ASC ASC_MIX=$ASC_MIX CONC=$CONC CSERVERS=$CSERVERS BACKUP=1 BACKUP_WAIT=0 COMBAT_INIT=$CW MACRO_INIT=$MW SEED=$(( 1000 + 100 * seg )) \
    MTRAIN_ARGS="--runs-per-iter $RUNS_PER_ITER --kl-runs $KL_RUNS --runs-offset $off --bc src/s3/data/macro_s1.pt --epochs ${PPO_EPOCHS:-10} --target-kl ${PPO_TARGET_KL:-0.012}" \
    ACTOR_ARGS="chunk=${CHUNK:-16384} mchunk=${MCHUNK:-32}${SEG_ACTOR_ARGS:+ $SEG_ACTOR_ARGS}" \
    setsid src/s3/run_s3.sh $SN $N $dur > $S/seg_$seg.log 2>&1 < /dev/null &
  SPID=$!; t0=$(date +%s); lastruns=0; lastchange=$t0; why=""; tick=0
  while kill -0 $SPID 2>/dev/null; do
    sleep 60; tick=$((tick + 1)); el=$(( $(date +%s) - t0 ))
    [ $((tick % 5)) -eq 0 ] && { status "segment $seg running ($el of $dur s)"; push; }
    if [ -n "$ACTOR_MEM_GB" ]; then
      for p in $(pgrep -g $SPID -f "S3.dll play"); do
        rss=$(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
        [ -n "$rss" ] && [ "$rss" -gt $((ACTOR_MEM_GB * 1048576)) ] && { kill $p; log "segment $seg: memory guard killed actor $p ($((rss / 1048576)) GB > $ACTOR_MEM_GB GB)"; }
      done
    fi
    [ $el -lt 240 ] && continue
    kill -0 $SPID 2>/dev/null || break
    r=$(cat $R/runs/*.jsonl 2>/dev/null | wc -l); [ "$r" != "$lastruns" ] && { lastruns=$r; lastchange=$(date +%s); }
    act=$(pgrep -g $SPID -fc "S3.dll play")
    d=$(df / | awk 'NR==2{print $5}' | tr -d %)
    if [ $el -lt $dur ]; then
      alive_all $R/logs/trainer_pids.txt || why="a trainer died"
      alive_all $R/logs/server_pids.txt || why="an inference server died"
      [ "$act" -lt $(( N / 2 )) ] && why="only $act of $N actors alive"
      [ $(( $(date +%s) - lastchange )) -gt 900 ] && why="no new run for 15 min"
    fi
    [ "$d" -ge 90 ] && why="disk ${d}%"
    [ $el -gt $(( dur + 1200 )) ] && why="segment over its time limit by 20 min"
    if [ -n "$why" ]; then log "segment $seg stopped by the guard: $why"; kill_group $SPID; break; fi
  done
  wait $SPID 2>/dev/null
  rm -f /dev/shm/s3_${SN}_*
  if validate $R; then
    nruns=$VRUNS; accept_good $R; fails=0
    log "segment $seg ok: $nruns runs${why:+ (stopped early: $why)}; $(tail -n 1 $R/macro_train_log.jsonl 2>/dev/null | cut -c1-420)"
    carry_backup $seg $R
  else
    nruns=$VRUNS; echo failed > $R/.verdict
    fails=$((fails + 1)); log "segment $seg FAILED ($nruns runs${why:+, $why}); weights not advanced; consecutive failures $fails"
    grep -hE "Traceback|Error|Unhandled|FAILED" $R/logs/*.txt $S/seg_$seg.log 2>/dev/null | sort | uniq -c | sort -rn | head -5 >> $S/supervisor.log
  fi
  [ -n "$why" ] && { finish_backup $R $SN & }   # a stopped segment's backup loop was stopped with it; otherwise it finishes by itself
  [ $fails -eq 0 ] && [ -n "$CHAIN_HOOK" ] && hook_segment $((seg + 1))   # settings of the next segment and of this evaluation
  seg_env
  # the evaluation runs next to the next segment (2026-11-13: the GPU used to sit idle while it ran)
  [ $fails -eq 0 ] && eval_due $seg && start_evals $seg
  status "between segments"; push
  prune_old
  seg=$((seg + 1))
  if [ $fails -ge 3 ]; then log "three failed segments in a row: stopping"; break; fi
done
[ $fails -ge 3 ] && FINAL="STOPPED AFTER REPEATED FAILURES" || FINAL="FINISHED"
log "supervisor end: $FINAL; total runs $(total_runs); last good weights $CW | $MW"
mkdir -p $S/final; cp $CW $S/final/weights.pt 2>/dev/null; cp $MW $S/final/macro.pt 2>/dev/null
# the last segments' backup loops may still be uploading: wait for them (at most 30 min) before the server may stop
for i in $(seq 180); do pgrep -f "backup_loop.sh src/s3/runs/${NAME}_s" > /dev/null || break; sleep 10; done
kill $HB_PID 2>/dev/null; wait
status "$FINAL (nothing is running; the server is idle unless it stops itself below)"; push wait
if [ "${STOP_INSTANCE:-0}" = "1" ] && [ -n "$CONTAINER_ID" ] && [ -n "$CONTAINER_API_KEY" ]; then
  log "stopping the instance (STOP_INSTANCE=1)"; push wait
  vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY >> $S/supervisor.log 2>&1
fi
