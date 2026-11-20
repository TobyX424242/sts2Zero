#!/bin/bash
# ic113 port checks (CPU only; server tree /root/exp/ic113, outputs in out/checks/). Usage: src/ic113/checks.sh [part]
# parts: s0 (existing S0 tests + leak113), s3 (prompttest, tftest, healtest), root (portcheck runs, 8 processes x 2 runs),
#        cards (portcheck cards), events (portcheck events, 4 processes x 2 runs); default: all, in the background.
set -u
cd /root/exp/ic113
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
O=out/checks; mkdir -p $O
T=src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll
X=src/s3/S3/bin/Release/net9.0/S3.dll
part=${1:-all}
run() { local name=$1; shift; ( nice -n 5 "$@" > $O/$name.txt 2>&1; echo "EXIT $?" >> $O/$name.txt ) & }
if [ $part = all ] || [ $part = s0 ]; then
  run leak113_ironclad dotnet $T leak113 states=300 prefix=L113I character=IRONCLAD
  run leak113_defect dotnet $T leak113 states=200 prefix=L113D character=DEFECT outer_runs=3
  run resample dotnet $T resample 60 10000 L 3000
  run choices dotnet $T choices 10 10 4 B 3000
  run actions dotnet $T actions 6 2000 10 4 P 3000
  run obsv4 dotnet $T obsv4
  run obsv5 dotnet $T obsv5
fi
if [ $part = all ] || [ $part = s3 ]; then
  run prompttest dotnet $X prompttest
  run tftest dotnet $X tftest
  run healtest dotnet $X healtest
fi
if [ $part = all ] || [ $part = root ]; then
  for k in 0 1 2 3 4 5 6 7; do run root_$k dotnet $X portcheck what=runs runs=2 first=$((2 * k)) sims=64 tokens=150 ptok=0.25; done
fi
if [ $part = all ] || [ $part = cards ]; then
  run cards dotnet $X portcheck what=cards per=6 sims=64
fi
if [ $part = all ] || [ $part = events ]; then
  for k in 0 1 2 3; do run events_$k dotnet $X portcheck what=events runs=2 first=$((2 * k)) sims=8 hp=600; done
fi
wait
echo "CHECKS DONE $part"
