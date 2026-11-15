#!/bin/bash
# S2 self-play training run: trainer + inference server + N C# actors, for S seconds (PROPOSAL.md §6.4 phase 1).
# usage: src/s2/run_selfplay.sh NAME N_ACTORS SECONDS [trainer args...]
# env: STARTS, TABLES (battle-start pool + T_f tables), CONC (fights in flight per actor, 64), SIMS_FULL (64),
#      SIMS_FAST (16), PFULL (0.25), FIGHTS (fight cap per actor), SERVER_CPUS (taskset list for the inference server, "0,1"),
#      KEEP (1 = keep the training chunks in runs/<name>/archive; 0 = the trainer deletes them after reading)
NAME=$1; N=$2; S=$3; shift 3
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
STARTS=${STARTS:-src/s2/data/starts_v0.111.0_defect_a10_train.jsonl}
TABLES=${TABLES:-src/s2/data/tables_v0.111.0_defect_a10_train.jsonl}
R=src/s2/runs/$NAME; mkdir -p $R/data $R/fights $R/ckpt $R/logs; rm -f $R/.done
ARCH=""; [ "${KEEP:-0}" = "1" ] && { mkdir -p $R/archive; ARCH="--archive-dir $R/archive"; }
P=/dev/shm/s2_${NAME}_
echo "[$(date +%T)] run $NAME: $N actors, $S s, starts $STARTS" | tee -a $R/logs/run.txt
./.venv/bin/python -I src/s2/py/train_combat.py --data $R/data --weights $R/weights.pt --ckpt-dir $R/ckpt $ARCH "$@" > $R/logs/train.txt 2>&1 &
TP=$!
until [ -f $R/weights.pt ]; do sleep 1; done
taskset -c ${SERVER_CPUS:-0,1} ./.venv/bin/python -I src/s2/py/infer_server.py --prefix $P --actors $N --capacity ${CONC:-64} \
  --weights $R/weights.pt --stats-every 60 > $R/logs/infer.txt 2>&1 &
WAITS="$!"
for i in $(seq 0 $((N-1))); do
  dotnet src/s2/S2/bin/Release/net9.0/S2.dll play mode=selfplay bridge=$P$i seconds=$S concurrent=${CONC:-64} tag=a$i \
    seed=$((1000 + i)) shard=$i/$N starts=$STARTS tables=$TABLES data=$R/data out=$R/fights/a$i.jsonl ${FIGHTS:+fights=$FIGHTS} \
    sims_full=${SIMS_FULL:-64} sims_fast=${SIMS_FAST:-16} pfull=${PFULL:-0.25} shutdown=1 > $R/logs/a$i.txt 2>&1 &
  WAITS="$WAITS $!"
done
wait $WAITS
kill $TP; wait $TP 2>/dev/null
echo "[$(date +%T)] run $NAME done" | tee -a $R/logs/run.txt
