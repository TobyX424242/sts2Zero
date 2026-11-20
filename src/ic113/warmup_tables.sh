#!/bin/bash
# ic113 combat warm-up (src/ic113/WARMUP.md): T_f tables of human battle starts, priced by a macro file through
# `S3 tables` (src/s3/S3/HumanTables.cs) in N shards served by one macro server; shards concatenated.
# usage: src/ic113/warmup_tables.sh STARTS OUT_TABLES N_SHARDS [MACRO_FILE]
# env: TF_HEAL (1), CHECK (1: also start each fight to the first decision; failures are dropped), MCAP (rows per bridge, 512)
STARTS=$1; OUT=$2; N=$3; MW=${4:-/root/exp/ic113/w/macro_s3v3_s005_v113.pt}
W=${TREE:-/root/exp/ic113}; cd $W
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1
D=$OUT.parts; mkdir -p $D; P=/dev/shm/ictab_$$_
./.venv/bin/python -I src/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --actors $N --min-period 0 \
  --capacity 1 --mcapacity ${MCAP:-512} --weights $W/w/combat_s3v3_s005_v113.pt --macro $MW --mlayout 3 > $D/mserver.txt 2>&1 &
SP=$!
for i in $(seq 0 $((N-1))); do
  dotnet src/s3/S3/bin/Release/net9.0/S3.dll tables starts=$STARTS out=$D/part$i.jsonl mbridge=${P}m$i shard=$i/$N \
    tf_heal=${TF_HEAL:-1} check=${CHECK:-1} concurrent=64 shutdown=1 > $D/part$i.txt 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $SP 2>/dev/null; wait $SP 2>/dev/null
cat $D/part*.jsonl > $OUT
tail -qn1 $D/part*.txt > $D/summary.txt
echo "tables: $(wc -l < $OUT) of $(wc -l < $STARTS) starts -> $OUT"
