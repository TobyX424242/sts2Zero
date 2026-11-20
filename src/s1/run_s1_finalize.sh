#!/bin/bash
# After the ensemble: calibration, dominance checks, T_f builder tests and the G1 report.
. /root/sts2/env.sh
while pgrep -f "value_v2/run_macro_ensemble.sh" > /dev/null; do sleep 30; done
M=runs/macro
$PY src/s1/s1_finalize.py runs/floors_v2 $M/s1_final.json $M/noattn_light $M/noattn_light_seed1 $M/noattn_light_seed2 $M/noattn_light_seed3 $M/noattn_light_seed4 || echo FAILED
echo FINALIZE_DONE
