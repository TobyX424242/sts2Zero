#!/bin/bash
# Chain hook of s3ic (src/ic113/CHAIN.md), called by src/s3/s3_supervise.sh when config.env sets CHAIN_HOOK.
#   s3ic_hook.sh segment CHAIN_DIR NEXT_SEG FINISHED_SEG_DIR MACRO
#       before segment NEXT_SEG (FINISHED_SEG_DIR = the last good segment, "" at the chain's start; MACRO = the macro file
#       the next segment starts from):
#       1. curriculum (PLAN step 7): self-play statistics of the finished segment -> stage (never down), written to
#          CHAIN_DIR/curriculum_stage, every reading appended to CHAIN_DIR/curriculum.log;
#       2. mixed potion valuation (PLAN step 8): Vbar per act and kind from the finished segment's macro carry (its value
#          window: the newest self-play runs) and MACRO -> CHAIN_DIR/vmix/vbar_sNNN.json (src/ic113/potion_mix.py);
#          no carry (chain start) = a file without means (state term alone); a failed job keeps the previous file;
#       3. CHAIN_DIR/segment.env (read by the supervisor before each segment and evaluation): ASC_MIX, SEG_ACTOR_ARGS
#          (asc_weights, potion_mix=FILE), EVAL_ARGS (EVAL_BASE_ARGS + potion_mix=FILE), HOOK_FOR_SEG.
#       Prints one line per step (the supervisor logs them); details in CHAIN_DIR/hook.log.
#   s3ic_hook.sh eval CHAIN_DIR EVAL_DIR SEG
#       after the evaluation of segment SEG: src/ic113/chain_stats.py eval -> EVAL_DIR/report.txt, appended with a
#       header to CHAIN_DIR/eval_report.txt.
# Settings (config.env, exported by the supervisor): CURRIC_T1_A10 (0.05), CURRIC_T1_LOW (0.50), CURRIC_T2_A10 (0.12),
# CURRIC_MIN_RUNS (300), VMIX_ROWS (500), VMIX_RECENT (20000 runs), VMIX_ASC (all), VMIX_DEVICE (cpu), VMIX_THREADS (16),
# VMIX_TABLE (phase-2 table JSON; off), VMIX_W ("0,0.35,0.70"), EVAL_BASE_ARGS (actor key=value of the evaluation).
MODE=$1; shift
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
PY=./.venv/bin/python
if [ "$MODE" = "eval" ]; then
  S=$1; E=$2; SEG=$3
  $PY -I src/ic113/chain_stats.py eval $E > $E/report.txt 2>> $S/hook.log
  { echo "== segment $SEG ($(date -u +%F\ %T) UTC, $E)"; cat $E/report.txt; } >> $S/eval_report.txt
  echo "eval report seg $SEG: $(head -n 1 $E/report.txt)"
  exit 0
fi
[ "$MODE" = "segment" ] || { echo "usage: s3ic_hook.sh segment|eval ..."; exit 2; }
S=$1; NEXT=$2; R=$3; MW=$4
NN=$(printf %03d $NEXT); mkdir -p $S/vmix
exec 2>> $S/hook.log
echo "[$(date -u +%F\ %T)] segment hook for segment $NEXT (finished: ${R:-none}, macro $MW)" >&2
# config.env is re-read here, so a change of EVAL_BASE_ARGS / CURRIC_* / VMIX_* takes effect at the next segment without
# a supervisor restart (2026-11-18: win_bound added to the evaluation, src/ic113/STALEMATE.md)
[ -f $S/config.env ] && . $S/config.env
# 0. a staged S3 build (src/s3/S3/bin_next, built while a segment ran) is swapped in here, between segments, when no
#    actor or evaluation runs from the old one; the old build is kept as net9.0_prev
B=src/s3/S3/bin/Release
if [ -f src/s3/S3/bin_next/S3.dll ]; then
  rm -rf $B/net9.0_prev
  if mv $B/net9.0 $B/net9.0_prev && mv src/s3/S3/bin_next $B/net9.0; then
    echo "build: swapped in the staged S3 build ($(cat $B/net9.0/BUILD_NOTE 2>/dev/null))"
  else
    [ -d $B/net9.0 ] || mv $B/net9.0_prev $B/net9.0; echo "build: swap FAILED, old build kept"
  fi
fi
# 1. curriculum
if [ -n "$R" ] && [ -d "$R/runs" ]; then
  $PY -I src/ic113/chain_stats.py curriculum --runs $R/runs --stage-file $S/curriculum_stage --log $S/curriculum.log \
    --t1-a10 ${CURRIC_T1_A10:-0.05} --t1-low ${CURRIC_T1_LOW:-0.50} --t2-a10 ${CURRIC_T2_A10:-0.12} \
    --t3-a10 ${CURRIC_T3_A10:-0} --min-runs ${CURRIC_MIN_RUNS:-300} --segment $(basename $R) | sed 's/^\[[^]]*\] /curriculum: /'
fi
[ -f $S/curriculum_stage ] || { echo "1 asc_mix 0.2 asc_weights uniform A0-A9" > $S/curriculum_stage; echo "curriculum: chain start, stage 1"; }
STAGE=$(cut -d' ' -f1 $S/curriculum_stage)
read MIX ASCW < <($PY -I src/ic113/chain_stats.py stage-args $STAGE)
# 2. Vbar
F=$S/vmix/vbar_s$NN.json; L=$S/vmix/vbar_s$NN.log
ok=0
if [ -n "$R" ] && [ -f "$R/carry_macro.pt" ]; then
  $PY -I src/ic113/potion_mix.py vbar --carry $R/carry_macro.pt --macro $MW --out $F --rows ${VMIX_ROWS:-500} \
    --recent-runs ${VMIX_RECENT:-20000} --asc ${VMIX_ASC:-all} --device ${VMIX_DEVICE:-cpu} --threads ${VMIX_THREADS:-16} \
    --seed $NEXT ${VMIX_VALUE_MEMBERS:+--value-members $VMIX_VALUE_MEMBERS} ${VMIX_TABLE:+--table $VMIX_TABLE --w ${VMIX_W:-0,0.35,0.70}} > $L 2>&1 && ok=1
  [ $ok = 1 ] && echo "vmix: means for acts $(grep -o 'means for acts .*\]' $L | tr -d "[]'," | cut -d' ' -f4-) -> $F ($(tail -n 1 $L | sed 's/.*(\(.*\))$/\1/'))" \
              || echo "vmix: Vbar job FAILED for segment $NEXT ($(tail -n 1 $L | cut -c1-200)); see $L"
else
  [ -n "$R" ] && echo "vmix: no carry_macro.pt in $R (CARRY=0?)"
fi
if [ $ok = 0 ]; then
  PREV=$(ls $S/vmix/vbar_s[0-9][0-9][0-9].json 2>/dev/null | grep -v "_s$NN.json" | tail -n 1)
  if [ -n "$PREV" ]; then F=$PREV; echo "vmix: keeping the previous file $F"
  else $PY -I src/ic113/potion_mix.py empty --macro $MW --out $F --why "segment $NEXT: no self-play states yet" > $L 2>&1
       echo "vmix: no means for segment $NEXT (state term alone) -> $F"; fi
fi
# 3. segment.env (relative paths: the supervisor, actors and evaluation run from the tree root)
F=${F#$W/}
{
  echo "# written by src/ic113/s3ic_hook.sh at $(date -u +%F\ %T) UTC for segment $NEXT"
  echo "HOOK_FOR_SEG=$NEXT"
  echo "CUR_STAGE=$STAGE"
  echo "ASC_MIX=$MIX"
  echo "VMIX_FILE=$F"
  echo "SEG_ACTOR_ARGS=\"potion_mix=$F$([ "$ASCW" != "-" ] && echo " asc_weights=$ASCW")\""
  echo "EVAL_ARGS=\"${EVAL_BASE_ARGS:-character=ironclad} potion_mix=$F\""
} > $S/segment.env.tmp && mv $S/segment.env.tmp $S/segment.env
echo "stage $STAGE $(cut -d' ' -f2- $S/curriculum_stage), potion_mix $F" > $S/hook_status.txt
echo "segment $NEXT: stage $STAGE (asc_mix $MIX, asc_weights ${ASCW/-/uniform}), potion_mix $F"
