#!/bin/bash
# ic113 battle-start import test (CPU, server). Human Ironclad battle starts exported by src/s2/export_starts.py
# (/root/exp/ic113data/starts) loaded into the v0.113 simulator (S0 BattleStartLoader, character from the start) and played
# to the first decision (S0Tests importfirst); results aggregated by src/ic113/import_report.py.
#   a10s : a random 20,000 of starts_v0.111.0_ironclad_a10_train.jsonl (Python random.Random(113).sample of line indices)
#   v113 : every v0.113.0 / v0.113.1 start (A8-A10, train and eval)
# usage: src/ic113/import_starts.sh [TREE (/root/exp/ic113)] [SHARDS (16)]   output: /root/exp/ic113data/import/
T=${1:-/root/exp/ic113}; K=${2:-16}; D=/root/exp/ic113data; O=$D/import; mkdir -p $O/in $O/res; cd $T
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1
python3 -I - $D/starts $O/in $K <<'PY'
import glob, os, random, sys
src, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = open(os.path.join(src, "starts_v0.111.0_ironclad_a10_train.jsonl")).read().splitlines()
sample = [lines[i] for i in sorted(random.Random(113).sample(range(len(lines)), 20000))]
v113 = [l for p in sorted(glob.glob(os.path.join(src, "starts_v0.113.*_ironclad_a*.jsonl"))) for l in open(p).read().splitlines()]
for name, rows in (("a10s", sample), ("v113", v113)):
    for j in range(k):
        open(os.path.join(out, f"{name}_{j:02d}.jsonl"), "w").write("".join(l + "\n" for l in rows[j::k]))
    print(name, len(rows))
PY
X=src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll
for f in $O/in/*.jsonl; do b=$(basename $f .jsonl); nice -n 10 dotnet $X importfirst $f $O/res/$b.jsonl IF-$b > $O/res/$b.txt 2>&1 & done
wait
cat $O/res/*.txt
./.venv/bin/python -I src/ic113/import_report.py $O/res/a10s_*.jsonl $O/res/v113_*.jsonl | tee $O/report.txt
