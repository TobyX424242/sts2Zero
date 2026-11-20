#!/bin/bash
# usage: act.sh NAME BUILD_DIR CORES [extra keys...]  -- one S3 actor, uniform evaluators (no inference server, CPU only)
T=/root/exp/ic113/chain; D=/root/exp/ic113/drtest; cd $T
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 CUDA_VISIBLE_DEVICES=
N=$1; B=$2; C=$3; shift 3; O=$D/out/$N; rm -rf $O; mkdir -p $O
nice -n 19 taskset -c $C dotnet $B/S3.dll play mode=selfplay bridge=uniform mbridge=uniform concurrent=${CONC:-8} tag=t seed=${SEED:-7} \
  runs=${RUNS:-24} ascension=${ASC:-10} sims_full=${SF:-16} sims_fast=${SS:-4} pfull=0.25 perturb=0.1 asc_mix=1 max_floors=${MF:-80} \
  data=$O/cdata mdata=$O/mdata out=$O/runs.jsonl fights=$O/fights.jsonl shutdown=1 obs=5 \
  character=ironclad era=6 tf_heal=1 hall_sims=0.5 boss_pfull=1 practice_boss=2 practice_min_act=2 practice_pfull=0.25 candidates=4 \
  look_beta=20 potion_mix=$T/src/s3/runs/s3ic/vmix/vbar_s007.json potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 \
  "$@" > $O/actor.txt 2>&1
echo "$N rc=$? $(wc -l < $O/runs.jsonl) runs"
