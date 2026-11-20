#!/bin/bash
# S3 full-game self-play run (PROPOSAL.md §6.5): macro trainer (PPO + V_run) + combat trainer + one inference server for
# both networks + N actors playing whole runs, for S seconds.
# usage: src/s3/run_s3.sh NAME N_ACTORS SECONDS
# env: ASC (ascension, 10), CONC (runs in flight per actor, 32), RUNS (run cap per actor), SIMS_FULL (64), SIMS_FAST (16),
#      PFULL (0.25), PERTURB (share of runs with perturbed fight starts, 0.1), COMBAT_INIT (combat checkpoint to start from),
#      COMBAT_WINDOW (combat replay window in decisions, 500000; 4.1 kB of RAM each),
#      MACRO_INIT (macro ensemble to start from, default the S1 ensemble), MTRAIN_ARGS / CTRAIN_ARGS (extra trainer
#      arguments), ACTOR_ARGS (extra actor key=value), FREEZE_COMBAT=1 (no combat trainer), FREEZE_MACRO=1 (no macro
#      trainer), CSERVERS (combat server processes, 1), MSERVERS (macro server processes, 1; also from extra.env), MACRO_PERIOD (seconds between macro batches, 0.03),
#      ASC_MIX (share of runs at ASC, the rest uniform over the levels below; 1), BACKUP_WAIT (0 = return before the last upload ends), BACKUP (1 = archive chunks and upload them with
#      src/s3/backup_loop.sh; 0 = trainers delete what they have read)
# Per-chain extra options: src/s3/runs/<chain>/extra.env (chain = NAME without its _sNNN suffix) is read at every
# segment start and may set EXTRA_ACTOR_ARGS (actor key=value), EXTRA_CTRAIN_ARGS (combat trainer flags) and
# EXTRA_MTRAIN_ARGS (macro trainer flags, after the supervisor's; the last value of a flag wins), so methods can
# be switched on or off between segments without restarting the supervisor; delete the file to go back.
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
# CARRY_EVERY=S (environment, config.env or extra.env; 2026-11-19, interruptible servers): with CARRY=1 the trainers also
# write their carry files every S seconds (default 1800; 0 = only at the end / on SIGTERM), so a segment killed without
# a signal keeps carries at most S seconds old and s3_supervise.sh can adopt it at its restart. Missing --load-carry
# file = the trainers start with empty replay windows and fresh optimizer states (a log line, no error).
# SHADOW_ARGS (environment or extra.env; IMPROVEMENT_PLAN.md change C2): a second combat trainer with these flags (e.g.
# "--d 256 --layers 4 --lr 3e-4 --surprise 0.5 --ema 0.999 --window 4000000") learns from the same chunks (the main
# trainer hard-links them into $R/shadow_data) and writes $R/shadow.pt, which actors do not use; the next segment
# continues it from there (optimizer state in carry_shadow.bin). Not watched by the supervisor's guard.
NAME=$1; N=$2; S=$3
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
R=src/s3/runs/$NAME; mkdir -p $R/cdata $R/mdata $R/runs $R/fights $R/ckpt $R/mckpt $R/logs $R/archive; rm -f $R/.done
COMBAT_INIT=${COMBAT_INIT:-src/s3/weights/combat_s2p1.pt}
MACRO_INIT=${MACRO_INIT:-src/s3/data/macro_s1.pt}
ARCH=""; MARCH=""; [ "${BACKUP:-0}" = "1" ] && { ARCH="--archive-dir $R/archive"; MARCH="--archive-dir $R/archive"; }
P=/dev/shm/s3_${NAME}_
CHAIN=${NAME%_s[0-9][0-9][0-9]}; XE=src/s3/runs/$CHAIN/extra.env; EXTRA_ACTOR_ARGS=""; EXTRA_CTRAIN_ARGS=""; EXTRA_MTRAIN_ARGS=""; SHADOW_ARGS=${SHADOW_ARGS:-}; EXTRA_N=""; EXTRA_SERVER_ARGS=""; EXTRA_MSERVER_ARGS=""
[ "$CHAIN" != "$NAME" ] && [ -f $XE ] && . $XE
# extra.env may also set EXTRA_N: the number of actors for this segment (overrides the supervisor's N)
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
# 2026-11-19: the trainers run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (less memory reserved but unused;
# the macro trainer's --milestones update peaked at ~19 GB on a 32 GB GPU shared with the inference servers);
# TRAIN_ALLOC_CONF="" (environment or extra.env) turns it off. The inference servers (CUDA graphs) are unchanged.
# Two GPUs (2026-11-19): SERVE_GPU / TRAIN_GPU (environment or extra.env; unset = all visible GPUs, as before) put the
# segment's inference servers / trainers on that GPU index (CUDA_VISIBLE_DEVICES); the evaluation's: EVAL_GPU
# (config.env, s3_supervise.sh).
CCARRY=""; MCARRY=""
if [ "${CARRY:-0}" = "1" ]; then
  CSRC=$(dirname $COMBAT_INIT); MSRC=$(dirname $MACRO_INIT)
  if [ "$CHAIN" != "$NAME" ]; then
    for d in src/s3/runs/${CHAIN}_s[0-9][0-9][0-9]; do
      [ "$d" -ef "$CSRC" ] || [ "$d" -ef "$MSRC" ] || rm -f $d/carry_combat.* $d/carry_macro.* $d/carry_shadow.*
    done
  fi
  rm -f $CSRC/carry_combat.bin.tmp $CSRC/carry_shadow.bin.tmp $MSRC/carry_macro.pt.tmp
  CEV=${CARRY_EVERY:-1800}
  CCARRY_IN=$CSRC/carry_combat.bin
  # When switching to a shadow network (COMBAT_INIT=.../shadow.pt), load carry_shadow.bin if present so both its replay
  # window and warm Adam state carry over into the playing trainer.
  if [ "$(basename "$COMBAT_INIT")" = "shadow.pt" ] && [ -f "$CSRC/carry_shadow.bin" ]; then
    CCARRY_IN=$CSRC/carry_shadow.bin
  fi
  CCARRY="--save-carry $R/carry_combat.bin --load-carry $CCARRY_IN --carry-every $CEV"
  MCARRY="--save-carry $R/carry_macro.pt --load-carry $MSRC/carry_macro.pt --carry-every $CEV"
  echo "[$(date +%T)] carry: loading from $CSRC ($([ -f $CCARRY_IN ] && echo $(basename $CCARRY_IN) || echo NO combat carry)), $MSRC ($([ -f $MSRC/carry_macro.pt ] && echo present || echo NO macro carry)); saving to $R (also every $CEV s)" | tee -a $R/logs/run.txt
fi
PIDS=""; SHP=""; TEE=""
if [ -n "$SHADOW_ARGS" ] && [ "${FREEZE_COMBAT:-0}" != "1" ]; then
  SD=$(dirname $COMBAT_INIT); SINIT=""; [ -f $SD/shadow.pt ] && SINIT="--init $SD/shadow.pt"
  SKEEP=${SHADOW_CARRY_KEEP:-0}; SCEV=${CARRY_EVERY:-1800}
  SCARRY=""; [ "${CARRY:-0}" = "1" ] && SCARRY="--save-carry $R/carry_shadow.bin --load-carry $SD/carry_shadow.bin --carry-keep $SKEEP --carry-every $SCEV"
  # A shadow trained before the layout-v5 columns (2026-11-15) has a 10-wide ext_feat and cannot load into the 12-wide
  # one. Rather than crash the segment, start that shadow from the playing network (the recommended v5 restart: fresh
  # ext head, one or two segments of shadow training). Non-ext shadows have SW=-1 and are left alone.
  if [ -n "$SINIT" ]; then
    SW=$(./.venv/bin/python -I -c "import torch,sys;sd=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('state_dict',{});print(sd['ext_feat.0.weight'].shape[1] if 'ext_feat.0.weight' in sd else -1)" "$SD/shadow.pt" 2>/dev/null || echo -1)
    EW=$(./.venv/bin/python -I -c "import sys;sys.path.insert(0,'src/s0/py');import combat_obs as co;print(len(co.EXT_COLS))" 2>/dev/null || echo -1)
    if [ "$SW" != "-1" ] && [ "$SW" != "$EW" ]; then
      echo "[$(date +%T)] shadow ext input $SW != current $EW: starting the shadow from COMBAT_INIT" | tee -a $R/logs/run.txt
      SINIT="--init $COMBAT_INIT"
      [ "${CARRY:-0}" = "1" ] && SCARRY="--save-carry $R/carry_shadow.bin --load-carry $SD/carry_combat.bin --carry-keep $SKEEP --carry-every $SCEV"
    fi
  fi
  # SHADOW_FROM_PLAYING=1 (2026-11-15, layout-v4 "ext" shadow): with no shadow to continue, the shadow starts as a copy of
  # the playing network (COMBAT_INIT; SHADOW_ARGS adds --config ext=1) and loads the playing trainer's replay carry.
  # SHADOW_PRELOAD_CARRY=1 (2026-11-20, clean-init larger shadow): starts from random weights (no --init), but preloads
  # the playing trainer's replay entries from carry_combat.bin (train_combat.py keeps the replay buffer and starts a fresh
  # optimizer with steps_before=0 when parameter shapes differ).
  if [ -z "$SINIT" ] && [ "${SHADOW_FROM_PLAYING:-0}" = "1" ]; then
    SINIT="--init $COMBAT_INIT"
    [ "${CARRY:-0}" = "1" ] && SCARRY="--save-carry $R/carry_shadow.bin --load-carry $SD/carry_combat.bin --carry-keep $SKEEP --carry-every $SCEV"
  elif [ -z "$SINIT" ] && [ "${SHADOW_PRELOAD_CARRY:-0}" = "1" ] && [ "${CARRY:-0}" = "1" ] && [ -f "$SD/carry_combat.bin" ]; then
    SCARRY="--save-carry $R/carry_shadow.bin --load-carry $SD/carry_combat.bin --carry-keep $SKEEP --carry-every $SCEV"
  fi
  mkdir -p $R/shadow_data $R/shadow_ckpt; TEE="--tee-dir $R/shadow_data"
  SGPU=${SHADOW_GPU:-$TRAIN_GPU}
  env ${SGPU:+CUDA_VISIBLE_DEVICES=$SGPU} PYTORCH_CUDA_ALLOC_CONF=${TRAIN_ALLOC_CONF-expandable_segments:True} ./.venv/bin/python -I src/s2/py/train_combat.py --data $R/shadow_data --weights $R/shadow.pt --ckpt-dir $R/shadow_ckpt \
    --log $R/shadow_train_log.jsonl --obs $OBS $SINIT $SHADOW_ARGS $SCARRY > $R/logs/shadow.txt 2>&1 &
  SHP=$!
  echo "$SHP" > $R/logs/shadow_pid.txt
  echo "[$(date +%T)] shadow combat trainer (pid $SHP${SGPU:+, gpu $SGPU}): ${SINIT:-from scratch} $SHADOW_ARGS $SCARRY" | tee -a $R/logs/run.txt
fi
if [ "${FREEZE_COMBAT:-0}" = "1" ]; then cp $COMBAT_INIT $R/weights.pt
else
  env ${TRAIN_GPU:+CUDA_VISIBLE_DEVICES=$TRAIN_GPU} PYTORCH_CUDA_ALLOC_CONF=${TRAIN_ALLOC_CONF-expandable_segments:True} ./.venv/bin/python -I src/s2/py/train_combat.py --data $R/cdata $TEE --weights $R/weights.pt --ckpt-dir $R/ckpt --init $COMBAT_INIT \
    --lr 2e-4 --window ${COMBAT_WINDOW:-500000} --obs $OBS $ARCH $CCARRY $CTRAIN_ARGS $EXTRA_CTRAIN_ARGS > $R/logs/ctrain.txt 2>&1 &
  PIDS="$PIDS $!"
fi
if [ "${FREEZE_MACRO:-0}" = "1" ]; then cp $MACRO_INIT $R/macro.pt
else
  env ${TRAIN_GPU:+CUDA_VISIBLE_DEVICES=$TRAIN_GPU} PYTORCH_CUDA_ALLOC_CONF=${TRAIN_ALLOC_CONF-expandable_segments:True} ./.venv/bin/python -I src/s3/py/train_macro_s3.py --mdata $R/mdata --macro $R/macro.pt --init $MACRO_INIT --ckpt-dir $R/mckpt \
    --log $R/macro_train_log.jsonl --target-asc ${ASC:-10} $MARCH $MCARRY $MTRAIN_ARGS $EXTRA_MTRAIN_ARGS > $R/logs/mtrain.txt 2>&1 &
  PIDS="$PIDS $!"
fi
BP=""; [ "${BACKUP:-0}" = "1" ] && { src/s3/backup_loop.sh $R $NAME & BP=$!; }
for i in $(seq 1 120); do [ -f $R/weights.pt ] && [ -f $R/macro.pt ] && break; sleep 1; done
[ -f $R/weights.pt ] && [ -f $R/macro.pt ] || { echo "[$(date +%T)] FAILED: trainers wrote no weights" | tee -a $R/logs/run.txt; kill $PIDS $BP 2>/dev/null; exit 1; }
# server processes: CSERVERS combat processes share the actors (one Python loop saturates at about 27k states/s), and one
# macro process, so that a macro batch never delays a combat batch
CS=${CSERVERS:-1}; PERS=$(( (N + CS - 1) / CS )); WAITS=""
for k in $(seq 0 $((CS-1))); do
  F=$((k * PERS)); NA=$(( N - F < PERS ? N - F : PERS )); [ $NA -le 0 ] && continue
  env ${SERVE_GPU:+CUDA_VISIBLE_DEVICES=$SERVE_GPU} ./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first $F --actors $NA \
    --capacity ${CONC:-32} --weights $R/weights.pt --macro $R/macro.pt --stats-every 60 $SERVER_ARGS $EXTRA_SERVER_ARGS > $R/logs/server$k.txt 2>&1 &
  WAITS="$WAITS $!"
done
# MSERVERS macro processes (default 1) share the actors the same way: a layout v3 macro process is ~91% busy at 52 actors
# (33 ms per batch of ~18 rows vs 11 ms for v2), so actors wait on it (bridge wait 20% vs 10%, 15% fewer runs; 2026-11-16).
# Logs: mserver.txt for the first process, mserver<k>.txt for the others.
# extra.env may set EXTRA_MSERVER_ARGS (macro inference-server flags, e.g. "--mgraphs 1": CUDA graphs, branch mgraphs).
MSV=${MSERVERS:-1}; MPERS=$(( (N + MSV - 1) / MSV ))
for k in $(seq 0 $((MSV-1))); do
  F=$((k * MPERS)); NA=$(( N - F < MPERS ? N - F : MPERS )); [ $NA -le 0 ] && continue
  ML=$R/logs/mserver.txt; [ $k -gt 0 ] && ML=$R/logs/mserver$k.txt
  env ${SERVE_GPU:+CUDA_VISIBLE_DEVICES=$SERVE_GPU} ./.venv/bin/python -I src/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --first $F --actors $NA --min-period ${MACRO_PERIOD:-0.03} \
    --capacity ${CONC:-32} --weights $R/weights.pt --macro $R/macro.pt --stats-every 60 $EXTRA_MSERVER_ARGS > $ML 2>&1 &
  WAITS="$WAITS $!"
done
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
if [ -n "$BP" ]; then
  mv $R/cdata/*.s2d $R/mdata/*.s3r $R/archive/ 2>/dev/null
  touch $R/.done
  # BACKUP_WAIT=0: do not wait for the last upload (several minutes over WebDAV); the backup loop finishes on its own
  if [ "${BACKUP_WAIT:-1}" = "1" ]; then wait $BP; echo "[$(date +%T)] backup: $(tail -1 $R/logs/backup.txt)" | tee -a $R/logs/run.txt; fi
fi
echo "[$(date +%T)] run $NAME done: $(cat $R/runs/*.jsonl | wc -l) runs" | tee -a $R/logs/run.txt
