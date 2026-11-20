#!/bin/bash
# T3: Act-3 boss search budget at deployment (LAPTOP_GPU_PROPOSAL.md).
# Step 1: Collect ~300+ Act-3 boss starts from 320 fresh A10 runs (seeds Z200000-Z200319, s008 weights, exact deployment rules).
# Step 2: Paired S2 fight harness on the Act-3 boss starts with sims_full = 64, 128, 256 (boss_sims = 1, 2, 4), reps=2.
set -euo pipefail
W=$(cd "$(dirname "$0")/../../.." && pwd); cd "$W"
OUT=src/ic113/laptop/data/t3_out
mkdir -p "$OUT/collect" "$OUT/starts" "$OUT/harness"

CW=src/ic113/laptop/data/weights/s008/weights.pt
MW=src/ic113/laptop/data/weights/s008/macro.pt
VBAR=src/server_backup/s3ic/s3ic/vmix/vbar_s009.json
EVAL_ARGS="character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20 look_smith=20 look_shop=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 deck_route=1 lethal_backstop=1 potion_mix=$VBAR"

N_COL=${N_COL:-16}
COUNT=${COUNT:-320}
CONC_COL=${CONC_COL:-16}
N_HAR=${N_HAR:-16}
CONC_HAR=${CONC_HAR:-16}
REPS=${REPS:-2}

if [ ! -f "$OUT/starts/starts.jsonl" ]; then
  echo "=== [T3] Step 1: collecting Act-3 boss starts ($COUNT A10 runs, Z200000+, $N_COL actors) ($(date -u +%FT%TZ)) ==="
  flock .laptop_gpu.lock bash -c "
    set -euo pipefail
    P=/dev/shm/t3col_\$\$_
    trap 'rm -f \${P}*' EXIT
    ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix \${P}c --mprefix \${P}m --actors $N_COL --capacity $CONC_COL \
      --weights $CW --macro $MW --graphs 1 --overlap 1 --overlap-min 32 > $OUT/collect/cserver.txt 2>&1 &
    CP=\$!
    ./.venv/bin/python -I src/s3/py/s3_server.py --serve macro --prefix \${P}c --mprefix \${P}m --actors $N_COL --capacity $CONC_COL \
      --min-period 0.02 --weights $CW --macro $MW --mgraphs 1 > $OUT/collect/mserver.txt 2>&1 &
    MP=\$!
    sleep 2
    PER=\$(( ($COUNT + $N_COL - 1) / $N_COL ))
    PIDS=''
    rm -f $OUT/collect/a*.jsonl $OUT/collect/starts_*.jsonl
    for i in \$(seq 0 \$(($N_COL - 1))); do
      first=\$(( 200000 + i * PER ))
      cnt=\$(( 200000 + $COUNT - first )); [ \$cnt -gt \$PER ] && cnt=\$PER
      [ \$cnt -le 0 ] && continue
      dotnet src/s3/S3/bin/Release/net9.0/S3.dll play mode=eval bridge=\${P}c\$i mbridge=\${P}m\$i concurrent=$CONC_COL tag=a\$i \
        ascension=10 seeds=Z first=\$first count=\$cnt sims_full=64 obs=5 $EVAL_ARGS \
        starts_dump=$OUT/collect/starts_{tag}.jsonl starts_dump_elite=0 \
        out=$OUT/collect/a\$i.jsonl fights=$OUT/collect/fights_\$i.jsonl shutdown=1 > $OUT/collect/a\$i.txt 2>&1 &
      PIDS=\"\$PIDS \$!\"
    done
    wait \$PIDS
    kill \$CP \$MP 2>/dev/null || true; wait \$CP \$MP 2>/dev/null || true
    cat $OUT/collect/a[0-9]*.jsonl > $OUT/collect/runs.jsonl
  "
  ./.venv/bin/python -I src/s3/summarize_runs.py "$OUT/collect/runs.jsonl" | tee "$OUT/collect/summary.txt"
  ./.venv/bin/python -I src/ic113/laptop/t3_boss_search.py prep "$OUT/collect/starts_*.jsonl" "$OUT/starts"
fi

echo "=== [T3] Step 2: paired S2 fight harness (sims=64, 128, 256; reps=$REPS; $N_HAR actors) ($(date -u +%FT%TZ)) ==="
for sims in 64 128 256; do
  arm="s${sims}"
  echo "--- [T3] running arm $arm (sims_full=$sims, reps=$REPS) ($(date -u +%FT%TZ)) ---"
  flock .laptop_gpu.lock bash -c "
    set -euo pipefail
    P=/dev/shm/t3h_${arm}_\$\$_
    trap 'rm -f \${P}*' EXIT
    ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix \${P}c --mprefix \${P}m --actors $N_HAR --capacity $CONC_HAR \
      --weights $CW --macro $MW --graphs 1 --overlap 1 --overlap-min 32 > $OUT/harness/${arm}_server.txt 2>&1 &
    SP=\$!
    sleep 2
    PIDS=''
    rm -f $OUT/harness/${arm}.jsonl $OUT/harness/${arm}_[0-9]*.jsonl
    for i in \$(seq 0 \$(($N_HAR - 1))); do
      dotnet src/s2/S2/bin/Release/net9.0/S2.dll play mode=eval bridge=\${P}c\$i concurrent=$CONC_HAR tag=${arm}_\$i shard=\$i/$N_HAR obs=5 \
        starts=$OUT/starts/starts.jsonl tables=$OUT/starts/tables.jsonl sims_full=$sims candidates=16 reps=$REPS seed=1 \
        win_bound=0.001 out=$OUT/harness/${arm}_\$i.jsonl shutdown=1 > $OUT/harness/${arm}_\$i.txt 2>&1 &
      PIDS=\"\$PIDS \$!\"
    done
    wait \$PIDS
    kill \$SP 2>/dev/null || true; wait \$SP 2>/dev/null || true
    cat $OUT/harness/${arm}_[0-9]*.jsonl > $OUT/harness/${arm}.jsonl
    rm -f $OUT/harness/${arm}_[0-9]*.jsonl
    echo \"[T3] $arm done: \$(wc -l < $OUT/harness/${arm}.jsonl) fights, load_fail=\$(cat $OUT/harness/${arm}_[0-9]*.txt | grep -c 'load failed' || true)\"
  "
done

echo "=== [T3] Step 3: paired report ($(date -u +%FT%TZ)) ==="
./.venv/bin/python -I src/ic113/laptop/t3_boss_search.py report "$OUT/starts/meta.json" \
  --arms "s64=$OUT/harness/s64.jsonl" "s128=$OUT/harness/s128.jsonl" "s256=$OUT/harness/s256.jsonl" \
  --out src/ic113/laptop/t3_results.json | tee "$OUT/report.txt"
echo "=== [T3] ALL DONE ($(date -u +%FT%TZ)) ==="
