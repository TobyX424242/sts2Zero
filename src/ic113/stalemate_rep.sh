#!/bin/bash
# src/ic113/STALEMATE.md: replay stalemate seeds (stale_seeds.txt lines "SEED ASC") to floor 2 with the chain's self-play
# settings, CPU inference on 4 cores (CORES), one actor; per-decision traces of SLIMES_WEAK fights (ENC) in out/NAME/trace.txt.
# Server tree /root/exp/ic113/stalemate (src/ic113/s3ic_setup.sh UPLOAD.tgz /root/exp/ic113/stalemate); weights in w/.
# usage: rep.sh NAME SEEDFILE [extra S3 keys...]  (stalemate diagnosis; CPU inference, one actor)
T=/root/exp/ic113/stalemate; cd $T
export PATH=$T/dotnet:$PATH DOTNET_ROOT=$T/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
N=$1; S=$2; shift 2; O=$T/out/$N; mkdir -p $O; P=/dev/shm/stale_${N}_
nice -n 5 taskset -c ${CORES:-56-59} ./.venv/bin/python -I src/s3/py/s3_server.py --serve both --prefix ${P}c --mprefix ${P}m --actors 1 --capacity 16 \
  --device cpu --dtype fp32 --weights ${W:-$T/w/weights.pt} --macro ${M:-$T/w/macro.pt} --idle-exit 60 > $O/server.txt 2>&1 &
nice -n 5 dotnet src/s3/S3/bin/Release/net9.0/S3.dll play mode=selfplay bridge=${P}c0 mbridge=${P}m0 concurrent=${CONC:-8} tag=r seed=1 \
  seed_list=$S max_floors=${MF:-2} sims_full=64 sims_fast=16 pfull=0.25 perturb=0 obs=5 character=ironclad era=6 tf_heal=1 hall_sims=0.5 \
  boss_pfull=1 candidates=4 look_beta=20 potion_mix=$T/w/vbar.json potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 \
  stale_trace=$O/trace.txt stale_trace_enc=${ENC:-ENCOUNTER.SLIMES_WEAK} out=$O/runs.jsonl fights=$O/fights.jsonl shutdown=1 "$@" > $O/actor.txt 2>&1
wait
