#!/bin/bash
# T2: offline macro replay test on the laptop GPU (LAPTOP_GPU_PROPOSAL.md, REPR_TEST_PLAN.md Stage 1).
# Runs M0 (control), M1 (--milestones), M2 (M1 + --p2), M3 (M0 + --look-distill) sequentially under flock .laptop_gpu.lock.
set -euo pipefail
W=$(cd "$(dirname "$0")/../../.." && pwd); cd "$W"
OUT=src/ic113/laptop/data/t2_out
mkdir -p "$OUT"
INIT=src/ic113/laptop/data/weights/s005/macro.pt
BC=src/ic113/laptop/data/macro_s1_v113.pt
TRAIN_LIST=src/ic113/laptop/data/t2_train_s006_4_8_12.txt
HELDOUT_LIST=src/ic113/laptop/data/t2_heldout_s006_16_s007_4.txt
CROSS_LIST=src/ic113/laptop/data/t2_cross_s007_4.txt

# Build deterministic chronological steady-state lists (skipping seq 0 short-run censoring and seq 20 tail flush)
./.venv/bin/python - <<'EOF'
import glob, os
def by_seq(seg, seqs):
    files = glob.glob(f"src/ic113/laptop/data/chunks_raw/{seg}/*.s3r")
    out = []
    for f in files:
        bn = os.path.basename(f)[:-4]
        tag, seq = bn.rsplit("_", 1)
        si = int(seq)
        if si in seqs:
            anum = int(tag[1:]) if tag.startswith("a") and tag[1:].isdigit() else 999
            out.append(((si, anum), f))
    return [f for _, f in sorted(out)]

tr = by_seq("s006", {4, 8, 12})
ho = by_seq("s006", {16}) + by_seq("s007", {4})
cr = by_seq("s007", {4})
open("src/ic113/laptop/data/t2_train_s006_4_8_12.txt", "w").write("\n".join(tr) + "\n")
open("src/ic113/laptop/data/t2_heldout_s006_16_s007_4.txt", "w").write("\n".join(ho) + "\n")
open("src/ic113/laptop/data/t2_cross_s007_4.txt", "w").write("\n".join(cr) + "\n")
print(f"lists ready: train={len(tr)} chunks (5760 runs), heldout={len(ho)} chunks (3840 runs), cross={len(cr)} chunks (1920 runs)")
EOF

COMMON="--mdata /dev/null --replay $TRAIN_LIST --init $INIT --bc $BC --runs-per-iter 1152 --eval-batch 1024 --kl-runs 1 --window-runs 50000 --value-row-share 0.16 --value-steps 150 --value-dropout --value-wd 0.1 --prob-avg --alpha 0.70 --epochs 10 --target-kl 0.03 --seed 0"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_arm() {
  local name=$1; shift
  echo "=== [T2] starting arm $name ($(date -u +%FT%TZ)) ==="
  rm -f "$OUT/${name}.pt" "$OUT/${name}.jsonl"
  flock .laptop_gpu.lock ./.venv/bin/python -I src/s3/py/train_macro_s3.py $COMMON \
    --macro "$OUT/${name}.pt" --log "$OUT/${name}.jsonl" "$@" > "$OUT/${name}.txt" 2>&1
  echo "=== [T2] finished arm $name ($(date -u +%FT%TZ)) ==="
}

run_arm m0
run_arm m1 --milestones --ms-warmup-iters 3
run_arm m2 --milestones --ms-warmup-iters 3 --p2 --p2-ema 0.95
run_arm m3 --look-distill 0.5 --look-distill-beta 20.0 --look-distill-replace

echo "=== [T2] running held-out evaluation ($(date -u +%FT%TZ)) ==="
flock .laptop_gpu.lock ./.venv/bin/python -I src/ic113/laptop/t2_eval.py \
  --ref "$INIT" --train-list "$TRAIN_LIST" --heldout-list "$HELDOUT_LIST" --cross-list "$CROSS_LIST" \
  --arms "base=$INIT" "m0=$OUT/m0.pt:$OUT/m0.jsonl" "m1=$OUT/m1.pt:$OUT/m1.jsonl" "m2=$OUT/m2.pt:$OUT/m2.jsonl" "m3=$OUT/m3.pt:$OUT/m3.jsonl" \
  --out src/ic113/laptop/t2_results.json
echo "=== [T2] ALL DONE ($(date -u +%FT%TZ)) ==="
