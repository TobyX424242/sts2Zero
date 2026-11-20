#!/bin/bash
# ic113 potion valuation phase 2 (PLAN step 8; src/ic113/CHAIN.md): the paired final-boss probe. Written, NOT run yet.
# GPU job (one combat server): about BUDGET final-boss fights at search 64 on every move (16 root candidates, the S2 eval
# default), each sampled entry played with an empty belt and with each single potion; T(k) -> OUT/potion_table.json.
# usage: src/ic113/potion_probe.sh COMBAT_WEIGHTS MACRO_FILE "STARTS_GLOB" OUT [N_ACTORS (16)]
#   STARTS_GLOB: starts_dump files of the chain's actors, e.g. add "starts_dump=src/s3/runs/s3ic/starts/{tag}.jsonl" to
#   EVAL_BASE_ARGS (config.env) or to EXTRA_ACTOR_ARGS (extra.env) for a segment; final-boss lines are used.
# env: BUDGET (2000), ASC (10), KINDS (all | a potion_mix Vbar file | comma list), CONC (32), SIMS (64), SEED (0)
# then: potion_mix.py vbar ... --table OUT/potion_table.json (or VMIX_TABLE in config.env) and the actors' potion_mix_phase=2.
CW=$1; MW=$2; STARTS=$3; O=$4; N=${5:-16}
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
mkdir -p $O
./.venv/bin/python -I src/ic113/potion_probe.py prep "$STARTS" $O --budget ${BUDGET:-2000} --entries ${ENTRIES:-0} --asc ${ASC:-10} \
  --kinds ${KINDS:-all} --seed ${SEED:-0} | tee $O/prep.txt || exit 1
P=/dev/shm/icpp_$$_
./.venv/bin/python -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --actors $N --capacity ${CONC:-32} \
  --weights $CW --macro $MW ${CSERVER_ARGS:-} > $O/server.txt 2>&1 &
SPID=$!; PIDS=""
for i in $(seq 0 $((N-1))); do
  dotnet ${S2_DLL:-src/s2/S2/bin/Release/net9.0/S2.dll} play mode=eval bridge=${P}c$i concurrent=${CONC:-32} tag=pp$i shard=$i/$N obs=5 \
    starts=$O/probe.jsonl tables=$O/probe_tables.jsonl sims_full=${SIMS:-64} ${WIN_BOUND:+win_bound=$WIN_BOUND} out=$O/probe_$i.jsonl shutdown=1 > $O/probe_$i.txt 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; rm -f ${P}*
cat $O/probe_[0-9]*.jsonl > $O/probe_fights.jsonl; rm -f $O/probe_[0-9]*.jsonl
echo "load failures $(cat $O/probe_[0-9]*.txt | grep -c 'load failed')" | tee -a $O/prep.txt
./.venv/bin/python -I src/ic113/potion_probe.py report $O/probe_fights.jsonl $O/potion_table.json | tee $O/report.txt
