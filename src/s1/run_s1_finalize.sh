#!/bin/bash
# After the ensemble: calibration, dominance checks, T_f builder tests and the G1 report.
# env: F (built dataset dir, default runs/floors_v2), M (macro run dir, default runs/macro), PY
set -e
W=$(cd "$(dirname "$0")/../.." && pwd); cd "$W"
PY=${PY:-./.venv/bin/python}; F=${F:-runs/floors_v2}; M=${M:-runs/macro}
$PY -I src/s1/s1_finalize.py "$F" "$M/s1_final.json" "$M/noattn_light" "$M/noattn_light_seed1" \
  "$M/noattn_light_seed2" "$M/noattn_light_seed3" "$M/noattn_light_seed4"
echo FINALIZE_DONE
