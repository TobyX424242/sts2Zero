#!/bin/bash
# S3 server test 1 (2026-11-12): throughput of the full training loop with 28 actors (6 min, ascension 0), then the
# baseline of gate G3: S1 macro policy (greedy) + S2 phase-1 battle network with deployment search (64 simulations on
# every move), fights scored by T_f vs by end HP, 210 fixed seeds, ascension 0; then T_f at ascension 10.
cd /root/sts2
log() { echo "[$(date +%T)] $*"; }
log "bench: 28 actors, 360 s, A0"
ASC=0 CONC=16 BACKUP=1 MTRAIN_ARGS="--runs-per-iter 512" src/s3/run_s3.sh bench28 28 360
log "eval A0: tf64 vs hp64"
COUNT=210 ASC=0 CONC=12 src/s3/eval_s3.sh src/s3/weights/combat_s2p1.pt src/s3/data/macro_s1.pt src/s3/out/eval_base_a0 30 "tf64:64 hp64:64:scorer=hp"
log "eval A10: tf64"
COUNT=210 ASC=10 CONC=12 src/s3/eval_s3.sh src/s3/weights/combat_s2p1.pt src/s3/data/macro_s1.pt src/s3/out/eval_base_a10 30 "tf64:64"
log "S3_PIPELINE1_DONE"
