#!/bin/bash
# Speed test of one S3 segment on a new server with the live chain's settings (runs/s3a/config.env + extra.env copied
# from the training server), started from a segment's weights and carry files, without backup. Compare runs per hour,
# trainer steps per second and inference states per second with the same minutes of the live segment.
#   setsid nohup src/s3/speed_test.sh INIT_DIR N CSERVERS SECONDS TAG > /root/sts2/speed_TAG.log 2>&1 < /dev/null &
# INIT_DIR holds weights.pt, macro.pt, carry_combat.bin, carry_macro.pt (e.g. src/s3/runs/s3a_s036).
D=$1; N=$2; CS=$3; SECS=$4; TAG=$5
cd /root/sts2; export PATH=$PWD/dotnet:$PATH DOTNET_ROOT=$PWD/dotnet
CH=src/s3/runs/sp_$TAG; NAME=sp_${TAG}_s001; R=src/s3/runs/$NAME
rm -rf $CH $R; mkdir -p $CH
sed -e "s/^EXTRA_N=.*/EXTRA_N=$N/" -e "s/^CSERVERS=.*/CSERVERS=$CS/" src/s3/runs/s3a/extra.env > $CH/extra.env
source src/s3/runs/s3a/config.env
echo "[$(date -u +%T)] speed test $TAG: $N actors, $CS combat servers, $SECS s, from $D"
CTRAIN_ARGS="$CTRAIN_ARGS" ASC=$ASC ASC_MIX=$ASC_MIX CONC=$CONC CSERVERS=$CS BACKUP=0 COMBAT_INIT=$D/weights.pt MACRO_INIT=$D/macro.pt SEED=7000 \
  MTRAIN_ARGS="--runs-per-iter $RUNS_PER_ITER --kl-runs $KL_RUNS --runs-offset 0 --bc src/s3/data/macro_s1.pt --epochs ${PPO_EPOCHS:-10} --target-kl ${PPO_TARGET_KL:-0.012}" \
  ACTOR_ARGS="chunk=${CHUNK:-16384} mchunk=${MCHUNK:-32}" src/s3/run_s3.sh $NAME $N $SECS
echo "[$(date -u +%T)] done: $(cat $R/runs/*.jsonl 2>/dev/null | wc -l) runs"
rm -f $R/carry_* ; rm -rf $R/cdata $R/mdata $R/archive
