#!/bin/bash
# Paired evaluation of a segment's shadow combat network vs its playing network on s3ic deployment settings.
# Report-only (never auto-switches the live chain). Respects $S/.eval.lock and EVAL_GPU so it will not collide with the
# supervisor's background evaluation on a 32 GB GPU.
#
# usage: src/ic113/shadow_eval.sh SEG [CHAIN]
#   e.g. src/ic113/shadow_eval.sh 31 s3ic
# env:
#   COUNT (seeds; default from config.env EVAL_COUNT, typically 300; set 1024 for confirmation gate)
#   FIRST (first seed index; default from config.env EVAL_FIRST, typically 100000)
#   SEEDS (seed prefix; default from config.env EVAL_SEEDS, typically Y)
#   EVAL_N (actors; default from config.env EVAL_N or 16)
#   FRESH_DIR (optional dir of fresh *.s2d chunks from segment SEG+1 for held-out combat loss comparison)
set -e
SEG=${1:?usage: $0 SEG [CHAIN]}
C=${2:-s3ic}
W=$(cd "$(dirname "$0")/../.." && pwd); cd "$W"
S=src/s3/runs/$C
[ -f "$S/config.env" ] && . "$S/config.env"
JN=$(printf %03d "$SEG")
P=src/s3/runs/${C}_s$JN
[ -f "$P/weights.pt" ] || { echo "missing $P/weights.pt"; exit 1; }
[ -f "$P/shadow.pt" ] || { echo "missing $P/shadow.pt"; exit 1; }
[ -f "$P/macro.pt" ] || { echo "missing $P/macro.pt"; exit 1; }

# Exact deployment actor args from hook_env/seg_{SEG+1}.env (same as s3_supervise.sh eval_one), fallback to segment.env
F=$S/hook_env/seg_$(printf %03d $((SEG + 1))).env; [ -f "$F" ] || F=$S/segment.env
EA=$(EVAL_ARGS=""; [ -f "$F" ] && . "$F"; echo "$EVAL_ARGS")
[ -n "$EA" ] || EA="${EVAL_BASE_ARGS:-character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20 look_smith=20 look_shop=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 deck_route=1 lethal_backstop=1}"

EC=${COUNT:-${EVAL_COUNT:-300}}
EN=${EVAL_N:-16}
CS=${CSERVERS:-2}
ASCL=${ASC:-10}
SPFX=${SEEDS:-${EVAL_SEEDS:-Y}}
SFST=${FIRST:-${EVAL_FIRST:-100000}}
EARM=${EVAL_ARM:-tf64:64}
EARMN=${EARM%%:*}

OUT=$S/eval/shadow_cmp_$JN
mkdir -p "$OUT"

if [ -n "${FRESH_DIR:-}" ] && [ -d "$FRESH_DIR" ]; then
  echo "=== Held-out combat loss on $FRESH_DIR ===" | tee "$OUT/heldout.txt"
  env ${EVAL_GPU:+CUDA_VISIBLE_DEVICES=$EVAL_GPU} ./.venv/bin/python -I src/s2/py/combat_heldout.py \
    --data "$FRESH_DIR" --skip-every 0 --max 200000 \
    playing="$P/weights.pt" shadow="$P/shadow.pt" | tee -a "$OUT/heldout.txt"
fi

# Reuse supervisor's playing eval if seeds/count match and summary is complete, else run both paired under .eval.lock
PLAY_DIR=$S/eval/seg_$JN
SHAD_DIR=$OUT/shadow
run_eval() {
  local cw=$1 dir=$2 log=$3
  rm -rf "$dir"
  COUNT=$EC ASC=$ASCL CONC=12 CSERVERS=$CS SEEDS=$SPFX FIRST=$SFST EVAL_ACTOR_ARGS="$EA" \
    env ${EVAL_GPU:+CUDA_VISIBLE_DEVICES=$EVAL_GPU} src/s3/eval_s3.sh "$cw" "$P/macro.pt" "$dir" "$EN" "$EARM" > "$log" 2>&1
}

(
  flock 8
  if [ "$EC" = "${EVAL_COUNT:-300}" ] && [ "$SPFX" = "${EVAL_SEEDS:-Y}" ] && [ "$SFST" = "${EVAL_FIRST:-100000}" ] \
     && grep -q "^== $EARMN" "$PLAY_DIR/summary.txt" 2>/dev/null; then
    echo "[shadow_eval] reusing supervisor playing eval in $PLAY_DIR"
    PJSON=$PLAY_DIR/$EARMN.jsonl
    PDIR=$PLAY_DIR
  else
    echo "[shadow_eval] running playing eval ($EC seeds ${SPFX}${SFST}+, $EN actors)..."
    run_eval "$P/weights.pt" "$OUT/playing" "$OUT/playing.log"
    PJSON=$OUT/playing/$EARMN.jsonl
    PDIR=$OUT/playing
  fi
  echo "[shadow_eval] running shadow eval ($EC seeds ${SPFX}${SFST}+, $EN actors)..."
  run_eval "$P/shadow.pt" "$SHAD_DIR" "$OUT/shadow.log"

  cp "$PJSON" "$OUT/playing.jsonl"
  cp "$SHAD_DIR/$EARMN.jsonl" "$OUT/shadow.jsonl"
  {
    echo "=== Paired comparison: ${C}_s$JN playing vs shadow ($EC seeds ${SPFX}${SFST}+) ==="
    echo "actor args: $EA"
    ./.venv/bin/python -I src/s3/summarize_runs.py "$OUT/playing.jsonl" "$OUT/shadow.jsonl"
    echo ""
    echo "--- playing funnel ---"
    ./.venv/bin/python -I src/ic113/chain_stats.py eval "$PDIR"
    echo ""
    echo "--- shadow funnel ---"
    ./.venv/bin/python -I src/ic113/chain_stats.py eval "$SHAD_DIR"
    echo ""
    echo "To switch live chain on owner approval:"
    echo "  src/s3/switch_combat.sh $C \"shadow s$JN beat playing\""
  } | tee "$OUT/report.txt"
) 8> "$S/.eval.lock"
