#!/bin/bash
# Throughput test: one inference server + N self-play actors for S seconds (random-init network, no training).
# usage: src/s2/bench_actors.sh N S [extra server args]
N=${1:-16}; S=${2:-60}; shift 2
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
[ -x $W/dotnet/dotnet ] && export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
O=src/s2/out/bench_n$N; rm -rf $O; mkdir -p $O
taskset -c ${SERVER_CPUS:-0,1} ./.venv/bin/python -I src/s2/py/infer_server.py --prefix /dev/shm/s2b_ --actors $N --capacity 128 --stats-every 15 "$@" > $O/infer.txt 2>&1 &
for i in $(seq 0 $((N-1))); do
  dotnet src/s2/S2/bin/Release/net9.0/S2.dll play mode=selfplay bridge=/dev/shm/s2b_$i seconds=$S concurrent=64 tag=b$i seed=$((i+1)) \
    out=$O/b$i.jsonl shutdown=1 > $O/b$i.txt 2>&1 &
done
wait
python3 - "$O" "$N" <<'PY'
import sys, re, glob
O, N = sys.argv[1], int(sys.argv[2])
tot = 0; fights = 0
for f in glob.glob(f"{O}/b*.txt"):
    last = [l for l in open(f) if "sims " in l][-1]
    tot += int(re.search(r"sims (\d+) =", last).group(1)); fights += int(re.search(r"fights (\d+)", last).group(1))
    secs = float(re.search(r"\] (\d+)s", last).group(1))
print(f"actors {N}: fights {fights}, simulations {tot}, ~{tot/secs:.0f} sims/s total")
PY
grep infer $O/infer.txt | tail -2
