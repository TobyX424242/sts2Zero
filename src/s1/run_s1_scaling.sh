#!/bin/bash
# S1 data build + data-scaling curve (PROPOSAL.md §6.3 item 5, experiment X3).
# Data are not included (Spire Codex licence); point D at a directory with runs-latest.jsonl.gz and meta_all.csv.gz.
# usage: src/s1/run_s1_scaling.sh
# env: D (data dir, default data/spire_codex), F (built dataset dir, default runs/floors_v2), PY (python, default ./.venv/bin/python)
set -e
W=$(cd "$(dirname "$0")/../.." && pwd); cd "$W"
PY=${PY:-./.venv/bin/python}; D=${D:-data/spire_codex}; F=${F:-runs/floors_v2}
if [ ! -f "$F/stats.json" ]; then
  $PY -I src/s1/build_dataset_v2.py "$D/runs-latest.jsonl.gz" "$D/meta_all.csv.gz" "$F" --workers "${WORKERS:-24}" --min-runs 100
fi
for spec in 25000:60 50000:40 100000:30 300000:15 0:15; do
  n=${spec%%:*}; ep=${spec##*:}
  for a in noattn transformer; do
    o=runs/s1_scaling/${a}_${n}
    [ -f "$o/results.json" ] && continue
    $PY -I src/s1/train_value_v2.py "$F" "$o" --arch "$a" --max-train-runs "$n" --epochs "$ep" || echo "FAILED $o"
  done
done
echo ALL_DONE
