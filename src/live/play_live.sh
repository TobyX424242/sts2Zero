#!/usr/bin/env bash
# Live play: the agent plays the real game through the AgentBridge mod (doc/live-play.md, src/live/README.md).
# Windows: play_live.ps1.
#
# usage: src/live/play_live.sh [WEIGHTS_DIR] [key=value ...]
#   WEIGHTS_DIR (default weights/ic113_s029, from scripts/fetch_weights.sh): the combat network (weights.pt or
#     *_weights.pt), the macro network (macro.pt or *_macro.pt) and the potion_mix table (vbar.json, vbar_*.json or
#     *_vbar.json; potion_mix=FILE overrides)
#   key=value: passed to `LiveAgent play` (src/live/agent/Program.cs), e.g.
#     asc=7            start a fresh Ironclad run at A7 when the game is at the main menu (omit: play the run in progress
#                      or the one started from the in-game panel); seed=ABC; loop=1 (next run after a game over)
#     dry=1            decide and log only (no actions sent); delay_ms=600 (mod's delay before each action)
#     any S3 agent key (defaults = the s3ic chain evaluation: greedy, 64 simulations per move, ...)
# env: UNIFORM=1       no inference servers, uniform networks (wiring test: the agent plays at random but legally)
#      DEVICE=cuda|cpu device of the inference servers (default: cuda when available); CAPACITY (16);
#                      CPU_THREADS (torch threads per server on the CPU, default 4)
#      PY=...          Python to use (default: <repo>/.venv/bin/python, created by scripts/setup.sh)
#      LIVEAGENT=source|prebuilt  which controller to run (default: bin/LiveAgent if present, else built from source)
# Logs: src/live/logs/ (controller JSONL, server outputs). Ctrl-C stops the controller and the servers.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
WARG=""
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then WARG=$1; shift; fi
LOGS=src/live/logs; mkdir -p "$LOGS"
TS=$(date +%Y%m%d_%H%M%S)
live_dotnet_env
live_controller "$LOGS" "$TS"

ARGS=()
trap live_stop_servers EXIT
trap 'exit 130' INT TERM
if [ "${UNIFORM:-0}" = "1" ]; then
  echo "[live] UNIFORM=1: no inference servers, uniform networks"
  ARGS+=(bridge=uniform mbridge=uniform)
else
  live_weights "$WARG"
  [ -n "$VBAR" ] && ARGS+=("potion_mix=$VBAR")
  live_python
  live_bridge_prefix
  live_start_servers "$LOGS" "$TS"
  echo "[live] servers ready"
  ARGS+=("bridge=${P}c0" "mbridge=${P}m0" "obs=$OBS")
fi

dotnet "$DLL" play "${ARGS[@]}" "log=$LOGS" "$@" 2>&1 | tee "$LOGS/controller_$TS.txt"
