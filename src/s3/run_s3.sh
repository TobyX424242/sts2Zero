#!/bin/bash
# S3 full-game self-play run (PROPOSAL.md §6.5): macro trainer (PPO + V_run) + combat trainer + one inference server for
# both networks + N actors playing whole runs, for S seconds.
# usage: src/s3/run_s3.sh NAME N_ACTORS SECONDS
# env: ASC (ascension, 10), CONC (runs in flight per actor, 32), RUNS (run cap per actor), SIMS_FULL (64), SIMS_FAST (16),
#      PFULL (0.25), PERTURB (share of runs with perturbed fight starts, 0.1), COMBAT_INIT (combat checkpoint to start from),
#      COMBAT_WINDOW (combat replay window in decisions, 500000; 4.1 kB of RAM each),
#      MACRO_INIT (macro ensemble to start from, default the S1 ensemble), MTRAIN_ARGS / CTRAIN_ARGS (extra trainer
#      arguments), ACTOR_ARGS (extra actor key=value), FREEZE_COMBAT=1 (no combat trainer), FREEZE_MACRO=1 (no macro
#      trainer), CSERVERS (combat server processes, 1), MACRO_PERIOD (seconds between macro batches, 0.03),
#      ASC_MIX (share of runs at ASC, the rest uniform over the levels below; 1), KEEP (1 = keep the training chunks in
#      runs/<name>/archive; 0 = the trainers delete what they have read)
# Per-chain extra options: src/s3/runs/<chain>/extra.env (chain = NAME without its _sNNN suffix) is read at every
# segment start and may set EXTRA_ACTOR_ARGS (actor key=value), EXTRA_CTRAIN_ARGS (combat trainer flags) and
# EXTRA_MTRAIN_ARGS (macro trainer flags, after the chain settings; the last value of a flag wins), so methods can
# be switched on or off between segments without restarting the chain; delete the file to go back.
# extra.env may also set EXTRA_SERVER_ARGS (combat inference-server flags, e.g. "--graphs 1 --overlap 1", see
# src/s3/py/s3_server.py), CSERVERS (combat server processes) and MPS=1 (below). SERVER_ARGS (environment) is added too.
# MPS=1 (environment or extra.env; default 0): the segment's trainers and inference servers run as clients of one CUDA MPS
# server, so their kernels overlap on the GPU instead of time-slicing. src/s3/mps.sh starts the MPS control daemon
# (pipe and logs under MPS_DIR, default /tmp/s3_mps) unless one is already running, checks it with a CUDA probe and leaves
# it running for the next segment; if MPS is not usable the segment runs without it (logs/mps.txt says why). Only the
# processes of a segment started with MPS=1 use the daemon (eval_s3.sh and others are unaffected).
# Stop it while no segment runs: src/s3/mps.sh stop
# CARRY=1 (environment or extra.env) keeps the trainers' replay windows and optimizer states across segments: they save
# them on exit to $R/carry_combat.bin and $R/carry_macro.pt and load the files in the directory of COMBAT_INIT / MACRO_INIT
# (the previous good segment). Files are complete or absent (temp file + rename; a broken file is rejected by the trainer);
# at a segment start the carry files of the chain's other segments are deleted, so the one being loaded stays until the
# next start (a failed segment's successor falls back to it).
# SHADOW_ARGS (environment or extra.env; IMPROVEMENT_PLAN.md change C2): a second combat trainer with these flags (e.g.
# "--d 256 --layers 4 --lr 3e-4 --surprise 0.5 --ema 0.999 --window 4000000") learns from the same chunks (the main
# trainer hard-links them into $R/shadow_data) and writes $R/shadow.pt, which actors do not use; the next segment
# continues it from there (optimizer state in carry_shadow.bin). Not watched by the segment guard.
NAME=$1; N=$2; S=$3
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
R=src/s3/runs/$NAME; mkdir -p $R/cdata $R/mdata $R/runs $R/fights $R/ckpt $R/mckpt $R/logs; rm -f $R/.done
COMBAT_INIT=${COMBAT_INIT:-src/s3/weights/combat_s2p1.pt}
MACRO_INIT=${MACRO_INIT:-src/s3/data/macro_s1.pt}
ARCH=""; MARCH=""; [ "${KEEP:-0}" = "1" ] && { mkdir -p $R/archive; ARCH="--archive-dir $R/archive"; MARCH="--archive-dir $R/archive"; }
P=/dev/shm/s3_${NAME}_
CHAIN=${NAME%_s[0-9][0-9][0-9]}; XE=src/s3/runs/$CHAIN/extra.env; EXTRA_ACTOR_ARGS=""; EXTRA_CTRAIN_ARGS=""; EXTRA_MTRAIN_ARGS=""; SHADOW_ARGS=${SHADOW_ARGS:-}; EXTRA_N=""; EXTRA_SERVER_ARGS=""
[ "$CHAIN" != "$NAME" ] && [ -f $XE ] && . $XE
# extra.env may also set EXTRA_N: the number of actors for this segment (overrides the chain's N)
[ -n "$EXTRA_N" ] && [ "$EXTRA_N" -gt 0 ] 2>/dev/null && N=$EXTRA_N
# Observation layout of the checkpoint (config "obs", default 4 = pre-v5). OBS overrides; the actors and the combat
# trainer(s) must agree, so it is detected once here and passed to all of them (train_combat.py --obs only records it in
# the checkpoint config; the ext head needs --config ext=1, e.g. in SHADOW_ARGS; evaluation detects obs the same way).
OBS=${OBS:-}
if [ -z "$OBS" ]; then
  OBS=$(./.venv/bin/python -I -c "import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('config',{});print(int(c.get('obs',4)))" "$COMBAT_INIT" 2>/dev/null || echo 4)
fi
echo "[$(date +%T)] observation layout obs=$OBS (from COMBAT_INIT or OBS)" | tee -a $R/logs/run.txt
echo "[$(date +%T)] run $NAME: $N actors, $S s, ascension ${ASC:-10}, combat init $COMBAT_INIT, macro init $MACRO_INIT" | tee -a $R/logs/run.txt
[ -n "$EXTRA_ACTOR_ARGS$EXTRA_CTRAIN_ARGS$EXTRA_MTRAIN_ARGS" ] && echo "[$(date +%T)] extra options from $XE: actor '$EXTRA_ACTOR_ARGS', combat trainer '$EXTRA_CTRAIN_ARGS', macro trainer '$EXTRA_MTRAIN_ARGS'" | tee -a $R/logs/run.txt
[ -n "$SERVER_ARGS$EXTRA_SERVER_ARGS" ] && echo "[$(date +%T)] combat server options: '$SERVER_ARGS $EXTRA_SERVER_ARGS'" | tee -a $R/logs/run.txt
if [ "${MPS:-0}" = "1" ]; then
  MPS_DIR=${MPS_DIR:-/tmp/s3_mps}
  if src/s3/mps.sh start $MPS_DIR >> $R/logs/mps.txt 2>&1; then
    export CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPS_DIR/log
    echo "[$(date +%T)] CUDA MPS on: $(tail -n 1 $R/logs/mps.txt)" | tee -a $R/logs/run.txt
  else
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    echo "[$(date +%T)] CUDA MPS requested but not usable, running without it: $(tail -n 1 $R/logs/mps.txt)" | tee -a $R/logs/run.txt
  fi
fi
CCARRY=""; MCARRY=""
if [ "${CARRY:-0}" = "1" ]; then
  CSRC=$(dirname $COMBAT_INIT); MSRC=$(dirname $MACRO_INIT)
  if [ "$CHAIN" != "$NAME" ]; then
    for d in src/s3/runs/${CHAIN}_s[0-9][0-9][0-9]; do
      [ "$d" -ef "$CSRC" ] || [ "$d" -ef "$MSRC" ] || rm -f $d/carry_combat.* $d/carry_macro.* $d/carry_shadow.*
    done
  fi
  rm -f $CSRC/carry_combat.bin.tmp $MSRC/carry_macro.pt.tmp
  CCARRY="--save-carry $R/carry_combat.bin --load-carry $CSRC/carry_combat.bin"
  MCARRY="--save-carry $R/carry_macro.pt --load-carry $MSRC/carry_macro.pt"
  echo "[$(date +%T)] carry: loading from $CSRC, $MSRC; saving to $R" | tee -a $R/logs/run.txt
fi
PIDS=""; SHP=""; TEE=""
if [ -n "$SHADOW_ARGS" ] && [ "${FREEZE_COMBAT:-0}" != "1" ]; then
  SD=$(dirname $COMBAT_INIT); SINIT=""; [ -f $SD/shadow.pt ] && SINIT="--init $SD/shadow.pt"
  SCARRY=""; [ "${CARRY:-0}" = "1" ] && SCARRY="--save-carry $R/carry_shadow.bin --load-carry $SD/carry_shadow.bin --carry-keep 0"
  # A shadow trained before the layout-v5 columns (2026-11-15) has a 10-wide ext_feat and cannot load into the 12-wide
  # one. Rather than crash the segment, start that shadow from the playing network (the recommended v5 restart: fresh
  # ext head, one or two segments of shadow training).
  if [ -n "$SINIT" ]; then
    SW=$(./.venv/bin/python -I -c "import torch,sys;sd=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('state_dict',{});print(sd['ext_feat.0.weight'].shape[1] if 'ext_feat.0.weight' in sd else -1)" "$SD/shadow.pt" 2>/dev/null || echo -1)
    EW=$(./.venv/bin/python -I -c "import sys;sys.path.insert(0,'src/s0/py');import combat_obs as co;print(len(co.EXT_COLS))" 2>/dev/null || echo -1)
    if [ "$SW" != "$EW" ]; then
      echo "[$(date +%T)] shadow ext input $SW != current $EW: starting the shadow from COMBAT_INIT" | tee -a $R/logs/run.txt
      SINIT="--init $COMBAT_INIT"
      [ "${CARRY:-0}" = "1" ] && SCARRY="--save-carry $R/carry_shadow.bin --load-carry $SD/carry_combat.bin --carry-keep 0"
    fi
  fi
  # SHADOW_FROM_PLAYING=1 (2026-11-15, layout-v4 "ext" shadow): with no shadow to continue, the shadow starts as a copy of
  # the playing network (COMBAT_INIT; SHADOW_ARGS adds --config ext=1) and loads the playing trainer's replay carry
  if [ -z "$SINIT" ] && [ "${SHADOW_FROM_PLAYING:-0}" = "1" ]; then
    SINIT="--init $COMBAT_INIT"
    [ "${CARRY:-0}" = "1" ] && SCARRY="--save-carry $R/carry_shadow.bin --load-carry $SD/carry_combat.bin --carry-keep 0"
  fi
  mkdir -p $R/shadow_data $R/shadow_ckpt; TEE="--tee-dir $R/shadow_data"
  ./.venv/bin/python -I src/s2/py/train_combat.py --data $R/shadow_data --weights $R/shadow.pt --ckpt-dir $R/shadow_ckpt \
    --log $R/shadow_train_log.jsonl --obs $OBS $SINIT $SHADOW_ARGS $SCARRY > $R/logs/shadow.txt 2>&1 &
  SHP=$!
  echo "[$(date +%T)] shadow combat trainer: ${SINIT:-from scratch} $SHADOW_ARGS" | tee -a $R/logs/run.txt
fi
if [ "${FREEZE_COMBAT:-0}" = "1" ]; then cp $COMBAT_INIT $R/weights.pt
else
  ./.venv/bin/python -I src/s2/py/train_combat.py --data $R/cdata $TEE --weights $R/weights.pt --ckpt-dir $R/ckpt --init $COMBAT_INIT \
    --lr 2e-4 --window ${COMBAT_WINDOW:-500000} --obs $OBS $ARCH $CCARRY $CTRAIN_ARGS $EXTRA_CTRAIN_ARGS > $R/logs/ctrain.txt 2>&1 &
  PIDS="$PIDS $!"
fi
if [ "${FREEZE_MACRO:-0}" = "1" ]; then cp $MACRO_INIT $R/macro.pt
else
  ./.venv/bin/python -I src/s3/py/train_macro_s3.py --mdata $R/mdata --macro $R/macro.pt --init $MACRO_INIT --ckpt-dir $R/mckpt \
    --log $R/macro_train_log.jsonl --target-asc ${ASC:-10} $MARCH $MCARRY $MTRAIN_ARGS $EXTRA_MTRAIN_ARGS > $R/logs/mtrain.txt 2>&1 &
  PIDS="$PIDS $!"
fi
for i in $(seq 1 120); do [ -f $R/weights.pt ] && [ -f $R/macro.pt ] && break; sleep 1; done
[ -f $R/weights.pt ] && [ -f $R/macro.pt ] || { echo "[$(date +%T)] FAILED: trainers wrote no weights" | tee -a $R/logs/run.txt; kill $PIDS 2>/dev/null; exit 1; }
# server processes: CSERVERS combat processes share the actors (one Python loop saturates at about 27k states/s), and one
# macro process, so that a macro batch never delays a combat batch
CS=${CSERVERS:-1}; PERS=$(( (N + CS - 1) / CS )); WAITS=""
for k in $(seq 0 $((CS-1))); do
  F=$((k * PERS)); NA=$(( N - F < PERS ? N - F : PERS )); [ $NA -le 0 ] && continue
  ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first $F --actors $NA \
    --capacity ${CONC:-32} --weights $R/weights.pt --macro $R/macro.pt --stats-every 60 $SERVER_ARGS $EXTRA_SERVER_ARGS > $R/logs/server$k.txt 2>&1 &
  WAITS="$WAITS $!"
done
./.venv/bin/python -I src/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --actors $N --min-period ${MACRO_PERIOD:-0.03} \
  --capacity ${CONC:-32} --weights $R/weights.pt --macro $R/macro.pt --stats-every 60 > $R/logs/mserver.txt 2>&1 &
WAITS="$WAITS $!"
echo "$WAITS" > $R/logs/server_pids.txt; echo "$PIDS" > $R/logs/trainer_pids.txt
for i in $(seq 0 $((N-1))); do
  dotnet src/s3/S3/bin/Release/net9.0/S3.dll play mode=selfplay bridge=${P}c$i mbridge=${P}m$i seconds=$S concurrent=${CONC:-32} tag=a$i \
    seed=$((${SEED:-1000} + i)) ascension=${ASC:-10} ${RUNS:+runs=$RUNS} sims_full=${SIMS_FULL:-64} sims_fast=${SIMS_FAST:-16} pfull=${PFULL:-0.25} \
    perturb=${PERTURB:-0.1} asc_mix=${ASC_MIX:-1} data=$R/cdata mdata=$R/mdata out=$R/runs/a$i.jsonl fights=$R/fights/a$i.jsonl shutdown=1 obs=$OBS $ACTOR_ARGS $EXTRA_ACTOR_ARGS > $R/logs/a$i.txt 2>&1 &
  WAITS="$WAITS $!"
done
wait $WAITS
sleep 8   # let the trainers read the last chunks
[ -n "$PIDS$SHP" ] && { kill $PIDS $SHP; wait $PIDS $SHP 2>/dev/null; }
rm -rf $R/shadow_data
if [ "${KEEP:-0}" = "1" ]; then
  mv $R/cdata/*.s2d $R/mdata/*.s3r $R/archive/ 2>/dev/null
fi
echo "[$(date +%T)] run $NAME done: $(cat $R/runs/*.jsonl | wc -l) runs" | tee -a $R/logs/run.txt
