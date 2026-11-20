#!/bin/bash
# T4: Phase 2 final-boss potion probe + paired A10 evaluation of Members 1-4 T_f and Phase 2 potion_mix.
# Uses updated Python code in .worktrees/ic113 and built S2.dll / S3.dll on s008 weights.
# Step 1: Paired final-boss potion probe on Z200000-Z200319 starts (ENTRIES=80 x 47 = 3,760 fights, sims=64).
# Step 2: Build calibrated Vbar files (Phase 1 m1-4, Phase 2 A10 all-5, Phase 2 A10 m1-4) from carry_macro_s008.pt.
# Step 3: 4-arm paired A10 eval on 320 strictly disjoint seeds (Z300000-Z300319): base, m14, p2, both.
set -euo pipefail
W=$(cd "$(dirname "$0")/../../.." && pwd); cd "$W"
WT="$W/.worktrees/ic113"
OUT="$W/src/ic113/laptop/data/t4_out"
mkdir -p "$OUT/probe" "$OUT/vmix" "$OUT/eval"

CW="$W/src/ic113/laptop/data/weights/s008/weights.pt"
MW="$W/src/ic113/laptop/data/weights/s008/macro.pt"
CARRY="$W/src/ic113/laptop/data/carry_macro_s008.pt"
STARTS="$W/src/ic113/laptop/data/t3_out/starts/starts.jsonl"
VBAR_BASE="$W/src/server_backup/s3ic/s3ic/vmix/vbar_s009.json"

N=${N:-16}
CONC=${CONC:-16}
ENTRIES=${ENTRIES:-80}
COUNT=${COUNT:-320}
FIRST=${FIRST:-300000}

BASE_ARGS="character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20 look_smith=20 look_shop=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 deck_route=1 lethal_backstop=1"

# ---- Step 1: Phase 2 Final-Boss Potion Probe ----
if [ ! -f "$OUT/probe/potion_table.json" ]; then
  echo "=== [T4 Step 1] Phase 2 final-boss potion probe (ENTRIES=$ENTRIES, sims=64) ($(date -u +%FT%TZ)) ==="
  ./.venv/bin/python -I "$WT/src/ic113/potion_probe.py" prep "$STARTS" "$OUT/probe" \
    --entries "$ENTRIES" --asc 10 --kinds all --seed 0 | tee "$OUT/probe/prep.txt"

  flock .laptop_gpu.lock bash -c "
    set -euo pipefail
    P=/dev/shm/t4pp_\$\$_
    trap 'rm -f \${P}*' EXIT
    ./.venv/bin/python -I $WT/src/s3/py/s3_server.py --serve combat --prefix \${P}c --mprefix \${P}m --actors $N --capacity $CONC \
      --weights $CW --macro $MW --graphs 1 --overlap 1 --overlap-min 32 > $OUT/probe/server.txt 2>&1 &
    SP=\$!
    sleep 2
    PIDS=''
    rm -f $OUT/probe/probe_[0-9]*.jsonl
    for i in \$(seq 0 \$(($N - 1))); do
      dotnet $W/src/s2/S2/bin/Release/net9.0/S2.dll play mode=eval bridge=\${P}c\$i concurrent=$CONC tag=pp\$i shard=\$i/$N obs=5 \
        starts=$OUT/probe/probe.jsonl tables=$OUT/probe/probe_tables.jsonl sims_full=64 candidates=16 win_bound=0.001 \
        out=$OUT/probe/probe_\$i.jsonl shutdown=1 > $OUT/probe/probe_\$i.txt 2>&1 &
      PIDS=\"\$PIDS \$!\"
    done
    wait \$PIDS
    kill \$SP 2>/dev/null || true; wait \$SP 2>/dev/null || true
    cat $OUT/probe/probe_[0-9]*.jsonl > $OUT/probe/probe_fights.jsonl
    rm -f $OUT/probe/probe_[0-9]*.jsonl
    echo \"[T4 Step 1] fights=\$(wc -l < $OUT/probe/probe_fights.jsonl), load_fail=\$(cat $OUT/probe/probe_[0-9]*.txt | grep -c 'load failed' || true)\"
  "
  ./.venv/bin/python -I "$WT/src/ic113/potion_probe.py" report \
    "$OUT/probe/probe_fights.jsonl" "$OUT/probe/potion_table.json" --min-pairs 20 | tee "$OUT/probe/report.txt"
fi

# ---- Step 2: Build Vbar files ----
echo "=== [T4 Step 2] Building Vbar files ($(date -u +%FT%TZ)) ==="
flock .laptop_gpu.lock bash -c "
  set -euo pipefail
  ./.venv/bin/python -I $WT/src/ic113/potion_mix.py vbar --carry $CARRY --macro $MW \
    --out $OUT/vmix/vbar_p1_m14.json --rows 500 --recent-runs 20000 --asc all --device cuda --seed 9 \
    --value-members 1,2,3,4 | tee $OUT/vmix/vbar_p1_m14.log
  ./.venv/bin/python -I $WT/src/ic113/potion_mix.py vbar --carry $CARRY --macro $MW \
    --out $OUT/vmix/vbar_p2_a10.json --rows 500 --recent-runs 20000 --asc 10 --device cuda --seed 9 \
    --table $OUT/probe/potion_table.json --w 0,0.35,0.70 | tee $OUT/vmix/vbar_p2_a10.log
  ./.venv/bin/python -I $WT/src/ic113/potion_mix.py vbar --carry $CARRY --macro $MW \
    --out $OUT/vmix/vbar_p2_a10_m14.json --rows 500 --recent-runs 20000 --asc 10 --device cuda --seed 9 \
    --value-members 1,2,3,4 --table $OUT/probe/potion_table.json --w 0,0.35,0.70 | tee $OUT/vmix/vbar_p2_a10_m14.log
"

# ---- Step 3: 4-Arm Paired A10 Evaluation (Z300000-Z300319) ----
run_arm() {
  local arm=$1 margs=$2 extra=$3
  local dir="$OUT/eval/$arm"
  mkdir -p "$dir"
  if [ -f "$dir/tf64.jsonl" ] && [ "$(wc -l < "$dir/tf64.jsonl")" -ge "$COUNT" ]; then
    echo "--- [T4 Step 3] arm $arm already complete ($(wc -l < "$dir/tf64.jsonl") runs) ---"
    return 0
  fi
  echo "--- [T4 Step 3] running arm $arm ($COUNT seeds Z${FIRST}+) ($(date -u +%FT%TZ)) ---"
  flock .laptop_gpu.lock bash -c "
    set -euo pipefail
    P=/dev/shm/t4e_${arm}_\$\$_
    trap 'rm -f \${P}*' EXIT
    ./.venv/bin/python -I $WT/src/s3/py/s3_server.py --serve combat --prefix \${P}c --mprefix \${P}m --actors $N --capacity $CONC \
      --weights $CW --macro $MW --graphs 1 --overlap 1 --overlap-min 32 > $dir/cserver.txt 2>&1 &
    CP=\$!
    ./.venv/bin/python -I $WT/src/s3/py/s3_server.py --serve macro --prefix \${P}c --mprefix \${P}m --actors $N --capacity $CONC \
      --min-period 0.02 --weights $CW --macro $MW --mgraphs 1 $margs > $dir/mserver.txt 2>&1 &
    MP=\$!
    sleep 2
    PER=\$(( ($COUNT + $N - 1) / $N ))
    PIDS=''
    rm -f $dir/tf64_[0-9]*.jsonl $dir/tf64_fights_[0-9]*.jsonl
    for i in \$(seq 0 \$(($N - 1))); do
      first=\$(( $FIRST + i * PER ))
      cnt=\$(( $FIRST + $COUNT - first )); [ \$cnt -gt \$PER ] && cnt=\$PER
      [ \$cnt -le 0 ] && continue
      dotnet $W/src/s3/S3/bin/Release/net9.0/S3.dll play mode=eval bridge=\${P}c\$i mbridge=\${P}m\$i concurrent=$CONC tag=${arm}\$i \
        ascension=10 seeds=Z first=\$first count=\$cnt sims_full=64 obs=5 $BASE_ARGS $extra \
        out=$dir/tf64_\$i.jsonl fights=$dir/tf64_fights_\$i.jsonl shutdown=1 > $dir/tf64_\$i.txt 2>&1 &
      PIDS=\"\$PIDS \$!\"
    done
    wait \$PIDS
    kill \$CP \$MP 2>/dev/null || true; wait \$CP \$MP 2>/dev/null || true
    cat $dir/tf64_[0-9]*.jsonl > $dir/tf64.jsonl
    cat $dir/tf64_fights_[0-9]*.jsonl > $dir/tf64_fights.jsonl
    rm -f $dir/tf64_[0-9]*.jsonl $dir/tf64_fights_[0-9]*.jsonl
  "
  ./.venv/bin/python -I "$WT/src/ic113/chain_stats.py" eval "$dir" | tee "$dir/report.txt"
}

echo "=== [T4 Step 3] Paired A10 Eval ($COUNT seeds Z${FIRST}+) ($(date -u +%FT%TZ)) ==="
run_arm base "" "potion_mix=$VBAR_BASE"
run_arm m14  "--value-members 1,2,3,4" "potion_mix=$OUT/vmix/vbar_p1_m14.json"
run_arm p2   "" "potion_mix=$OUT/vmix/vbar_p2_a10.json potion_mix_phase=2"
run_arm both "--value-members 1,2,3,4" "potion_mix=$OUT/vmix/vbar_p2_a10_m14.json potion_mix_phase=2"

echo "=== [T4 Summary] Paired comparisons vs base ($(date -u +%FT%TZ)) ==="
{
  cp "$OUT/eval/base/tf64.jsonl" "$OUT/eval/base.jsonl"
  cp "$OUT/eval/base/tf64_fights.jsonl" "$OUT/eval/base_fights.jsonl"
  for arm in m14 p2 both; do
    echo ">>> $arm vs base <<<"
    cp "$OUT/eval/$arm/tf64.jsonl" "$OUT/eval/$arm.jsonl"
    cp "$OUT/eval/$arm/tf64_fights.jsonl" "$OUT/eval/${arm}_fights.jsonl"
    ./.venv/bin/python -I "$WT/src/s3/summarize_runs.py" "$OUT/eval/base.jsonl" "$OUT/eval/$arm.jsonl"
    echo ""
  done
  echo ">>> Funnels <<<"
  for arm in base m14 p2 both; do
    echo "-- $arm --"
    cat "$OUT/eval/$arm/report.txt"
  done
} | tee "$OUT/summary.txt"
echo "=== [T4] ALL DONE ($(date -u +%FT%TZ)) ==="
