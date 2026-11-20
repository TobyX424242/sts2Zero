#!/bin/bash
# ic113 combat warm-up, final evaluation (GPU, one job after another): fight harness on all held-out elite / boss starts,
# step-0 net vs the warmed net (paired by start, search 64), then the day-0 Ironclad full-run arms with the warmed net.
# usage: src/ic113/warmup_final.sh WARM_WEIGHTS OUTDIR [N_ACTORS 32]
CW=$1; O=$2; N=${3:-32}; L=/root/exp/ic113; D=/root/exp/ic113data/warm; cd $L; mkdir -p $O
log() { echo "$(date -u +%FT%TZ) final: $*" >> $O/progress.log; }
log "harness start ($N actors)"
SIMS=64 CONC=32 src/ic113/warmup_eval.sh $O/harness $D/eval_eb_all.jsonl $D/tables_eval.jsonl $N "step0=$L/w/combat_s3v3_s005_v113.pt warm=$CW" > $O/harness.log 2>&1
python3 -I src/ic113/warmup_report.py harness $O/harness $D/eval_eb_all.jsonl step0 warm > $O/harness_report.txt 2>&1
log "harness done: $(grep '^all ' $O/harness_report.txt)"
src/ic113/warmup_fullruns.sh $CW $O/runs ic10 ic0
for a in ic10 ic0; do
  python3 -I src/ic113/warmup_report.py runs $L/out/day0/$a/$a.jsonl $O/runs/$a/$a.jsonl $L/out/day0/$a/${a}_fights.jsonl $O/runs/$a/${a}_fights.jsonl > $O/runs_${a}_report.txt 2>&1
done
log "all done"
