#!/bin/bash
# ic113 combat warm-up (PLAN step 6, src/ic113/WARMUP.md): S2-style fight self-play on human Ironclad battle starts
# with the current combat network (layout v5), the S3 combat trainer settings and coverage mutation.
# usage: src/ic113/warmup_run.sh NAME N_ACTORS SECONDS
#   trainer: src/s2/py/train_combat.py from INIT (converted s3v3_s005 combat net), lr 5e-5, wd 0.05, value dropout
#            0.3, surprise 0.5, EMA 0.999 (the live s3v3 combat trainer) and the S2 replay window of 1M decisions
#   servers: CSERVERS (2) combat processes of src/s3/py/s3_server.py (CUDA graphs, overlap), MPS on
#   actors:  N_ACTORS S2 self-play actors on the training starts (search 64 / 16, 25% full, 4 root candidates; mutate
#            MUTATE (0.2) of the loaded starts) + HOLD (1) actor on the held-out starts writing chunks no trainer reads
# env: INIT, STARTS / TABLES (training pool), HSTARTS / HTABLES (held-out pool), CONC (fights in flight per actor, 48),
#      MUTATE (0.2), POOL (mutation pool JSON), CTRAIN_ARGS (extra trainer flags), HOLD (1),
#      SHARDS (dir with starts_<i>.jsonl / tables_<i>.jsonl per actor, i < N: an actor reading the whole 1.3 GB table file
#      keeps ~2 GB of heap, 53 of them exhausted the 93 GB of RAM on 2026-11-18; pre-sharded: ~0.4 GB each)
NAME=$1; N=$2; S=$3
W=${TREE:-/root/exp/ic113}; cd $W
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1
D=/root/exp/ic113data/warm
INIT=${INIT:-$W/w/combat_s3v3_s005_v113.pt}
STARTS=${STARTS:-$D/starts_train.jsonl}; TABLES=${TABLES:-$D/tables_train.jsonl}
HSTARTS=${HSTARTS:-$D/starts_eval.jsonl}; HTABLES=${HTABLES:-$D/tables_eval.jsonl}
POOL=${POOL:-$D/mutate_pool.json}; HOLD=${HOLD:-1}; CONC=${CONC:-48}
R=$W/warm/$NAME; mkdir -p $R/data $R/heldout $R/fights $R/ckpt $R/logs $R/cards; rm -f $R/.done
P=/dev/shm/icw_${NAME}_
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/logs/run.txt; }
log "run $NAME: $N actors + $HOLD held-out, $S s, init $INIT, starts $STARTS, mutate ${MUTATE:-0.2}"
if src/s3/mps.sh start /tmp/s3_mps >> $R/logs/mps.txt 2>&1; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/s3_mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/s3_mps/log; log "MPS on"
else log "MPS not usable: $(tail -n1 $R/logs/mps.txt)"; fi
./.venv/bin/python -I src/s2/py/train_combat.py --data $R/data --weights $R/weights.pt --ckpt-dir $R/ckpt --init $INIT --obs 5 \
  --lr 5e-5 --wd 0.05 --config vdrop=0.3 --surprise 0.5 --ema 0.999 --window 1000000 --keep-every 25 \
  --log $R/train_log.jsonl $CTRAIN_ARGS > $R/logs/train.txt 2>&1 &
TP=$!; echo $TP > $R/trainer.pid; echo -800 > /proc/$TP/oom_score_adj   # the OOM killer takes actors first
for i in $(seq 1 120); do [ -f $R/weights.pt ] && break; sleep 1; done
[ -f $R/weights.pt ] || { log "FAILED: trainer wrote no weights"; kill $TP; exit 1; }
NT=$((N + HOLD)); CS=${CSERVERS:-2}; PERS=$(( (NT + CS - 1) / CS )); SP=""
for k in $(seq 0 $((CS-1))); do
  F=$((k * PERS)); NA=$(( NT - F < PERS ? NT - F : PERS )); [ $NA -le 0 ] && continue
  ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first $F --actors $NA \
    --capacity $CONC --weights $R/weights.pt --macro $W/w/macro_s3v3_s005_v113.pt --stats-every 60 --idle-exit 900 \
    --graphs 1 --overlap 1 --overlap-min 48 > $R/logs/server$k.txt 2>&1 &
  SP="$SP $!"
done
WAITS=""
for i in $(seq 0 $((N-1))); do
  dotnet src/s2/S2/bin/Release/net9.0/S2.dll play mode=selfplay bridge=${P}c$i seconds=$S concurrent=$CONC tag=a$i \
    seed=$((7000 + i)) $(if [ -n "$SHARDS" ]; then echo "starts=$SHARDS/starts_$i.jsonl tables=$SHARDS/tables_$i.jsonl"; else echo "shard=$i/$N starts=$STARTS tables=$TABLES"; fi) data=$R/data out=$R/fights/a$i.jsonl obs=5 \
    sims_full=64 sims_fast=16 pfull=0.25 candidates=4 mutate=${MUTATE:-0.2} mutate_pool=$POOL card_counts=$R/cards/a$i.json \
    shutdown=1 > $R/logs/a$i.txt 2>&1 &
  WAITS="$WAITS $!"
done
for j in $(seq 0 $((HOLD-1))); do
  i=$((N + j))
  dotnet src/s2/S2/bin/Release/net9.0/S2.dll play mode=selfplay bridge=${P}c$i seconds=$S concurrent=$CONC tag=h$j \
    seed=$((9000 + j)) shard=$j/$HOLD starts=$HSTARTS tables=$HTABLES data=$R/heldout out=$R/fights/h$j.jsonl obs=5 \
    sims_full=64 sims_fast=16 pfull=0.25 candidates=4 card_counts=$R/cards/h$j.json shutdown=1 > $R/logs/h$j.txt 2>&1 &
  WAITS="$WAITS $!"
done
echo "$WAITS" > $R/actors.pid
wait $WAITS
log "actors done; stopping the trainer (it writes its last weights on SIGTERM)"
kill $TP; wait $TP 2>/dev/null
kill $SP 2>/dev/null; wait $SP 2>/dev/null
touch $R/.done
log "run $NAME done: $(cat $R/fights/a*.jsonl | wc -l) training fights, $(tail -n1 $R/train_log.jsonl | cut -c1-200)"
