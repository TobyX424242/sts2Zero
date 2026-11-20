#!/bin/bash
# Live play: the S3 agent plays the real game through the AgentBridge mod (src/live/README.md).
# usage: src/live/play_live.sh [WEIGHTS_DIR] [key=value ...]
#   WEIGHTS_DIR (default src/live/weights/s3ic_s029): weights.pt (combat network), macro.pt (macro ensemble) and
#     the chain's potion_mix file vbar_*.json (the newest one is used; potion_mix=FILE overrides)
#   key=value: passed to `LiveAgent play` (src/live/agent/Program.cs), e.g.
#     asc=7            start a fresh Ironclad run at A7 when the game is at the main menu (omit: play the run in progress
#                      or the one started from the in-game panel); seed=ABC; loop=1 (next run after a game over)
#     dry=1            decide and log only (no actions sent); delay_ms=600 (mod's delay before each action)
#     any S3 agent key (defaults = the s3ic chain evaluation: greedy, 64 simulations per move, ...)
# env: UNIFORM=1      no inference servers, uniform networks (wiring test: the agent plays at random but legally)
#      DEVICE=cuda    device of the inference servers (cuda, or cpu); CAPACITY (16)
#      PY=...         python of the venv (default: <tree>/.venv/bin/python, else the main checkout's)
# Logs: src/live/logs/ (controller JSONL, server outputs). Ctrl-C stops the controller and the servers.
set -euo pipefail
L=$(cd "$(dirname "$0")" && pwd)
W=$(cd "$L/../.." && pwd)
MAIN=$(dirname "$(git -C "$L" rev-parse --path-format=absolute --git-common-dir)")
cd "$W"
WEIGHTS=src/live/weights/s3ic_s029
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then WEIGHTS=$1; shift; fi
LOGS=src/live/logs; mkdir -p $LOGS
TS=$(date +%Y%m%d_%H%M%S)
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
export NUGET_PACKAGES=${NUGET_PACKAGES:-$MAIN/.cache/nuget} DOTNET_CLI_HOME=${DOTNET_CLI_HOME:-$MAIN/.cache/dotnet}
# also set by LiveAgent.runtimeconfig.json (no quick JIT for loops): see the comment in agent/LiveAgent.csproj
export DOTNET_TC_OnStackReplacement=0

# ---- build when needed
DLL=$L/agent/artifacts/bin/LiveAgent/release/LiveAgent.dll
if [ ! -f "$DLL" ] || [ -n "$(find $L/agent src/s3/S3 src/s2/S2 src/s0/S0.Sim -name '*.cs' -newer "$DLL" -not -path '*/artifacts/*' -not -path '*/obj/*' -not -path '*/bin/*' 2>/dev/null | head -n 1)" ]; then
  echo "[play_live] building LiveAgent"
  "$L/agent/build.sh" > $LOGS/build_$TS.txt 2>&1 || { tail -n 30 $LOGS/build_$TS.txt; exit 1; }
fi

ARGS=()
PIDS=()
VBAR=$(ls -t $WEIGHTS/vbar_*.json 2>/dev/null | head -n 1 || true)
[ -n "$VBAR" ] && ARGS+=(potion_mix=$VBAR)
P=/dev/shm/live_$$_
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  sleep 0.5
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
  rm -f ${P}* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "${UNIFORM:-0}" = "1" ]; then
  echo "[play_live] UNIFORM=1: no inference servers, uniform networks"
  ARGS+=(bridge=uniform mbridge=uniform)
else
  CW=$WEIGHTS/weights.pt; MW=$WEIGHTS/macro.pt
  for f in $CW $MW; do [ -f $f ] || { echo "[play_live] missing $f"; exit 1; }; done
  PY=${PY:-}
  if [ -z "$PY" ]; then for c in "$W/.venv/bin/python" "$MAIN/.venv/bin/python"; do [ -x "$c" ] && { PY=$c; break; }; done; fi
  [ -n "$PY" ] || { echo "[play_live] no python venv found (.venv/bin/python); set PY="; exit 1; }
  # the observation layout of the combat checkpoint (config "obs"), as eval_s3.sh reads it
  OBS=$($PY -I -c "import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('config',{});print(int(c.get('obs',4)))" "$CW" 2>/dev/null || echo 5)
  DEV=${DEVICE:-cuda}; CAP=${CAPACITY:-16}
  # one actor: combat and macro servers as in src/s3/eval_s3.sh / src/ic113/eval_ic113.sh, without the idle exit
  $PY -I src/s3/py/s3_server.py --serve combat --prefix ${P}c --mprefix ${P}m --first 0 --actors 1 --capacity $CAP \
    --weights $CW --macro $MW --device $DEV --idle-exit 1e9 > $LOGS/server_combat_$TS.txt 2>&1 &
  PIDS+=($!)
  $PY -I src/s3/py/s3_server.py --serve macro --prefix ${P}c --mprefix ${P}m --actors 1 --min-period 0.02 --capacity $CAP \
    --weights $CW --macro $MW --mlayout 3 --device $DEV --idle-exit 1e9 > $LOGS/server_macro_$TS.txt 2>&1 &
  PIDS+=($!)
  echo "[play_live] servers starting (weights $WEIGHTS, device $DEV, obs $OBS); logs $LOGS/server_*_$TS.txt"
  for i in $(seq 1 600); do
    n=$(cat $LOGS/server_combat_$TS.txt $LOGS/server_macro_$TS.txt 2>/dev/null | grep -c "s3 server ready" || true)
    [ "$n" -ge 2 ] && break
    for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null || { echo "[play_live] a server died:"; tail -n 20 $LOGS/server_*_$TS.txt; exit 1; }; done
    sleep 0.5
  done
  [ "$n" -ge 2 ] || { echo "[play_live] servers not ready after 300 s"; exit 1; }
  echo "[play_live] servers ready"
  ARGS+=(bridge=${P}c0 mbridge=${P}m0 obs=$OBS)
fi

dotnet "$DLL" play "${ARGS[@]}" log=$LOGS "$@" 2>&1 | tee $LOGS/controller_$TS.txt
