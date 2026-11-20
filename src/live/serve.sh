#!/usr/bin/env bash
# Starts only the two inference servers (combat + macro, src/s3/py/s3_server.py) and keeps them running until Ctrl-C.
# play_live.sh does this itself; use serve.sh to keep the networks loaded while you start and stop the controller
# by hand. Windows: serve.ps1.
#
# usage: src/live/serve.sh [WEIGHTS_DIR]        (default weights/ic113_s029)
# env:   DEVICE=cuda|cpu (default: cuda when available), CAPACITY (16), CPU_THREADS (4, CPU only),
#        PY (default <repo>/.venv/bin/python)
# Prints the controller keys to pass on, e.g.
#   dotnet bin/LiveAgent/LiveAgent.dll play bridge=/dev/shm/live_123_c0 mbridge=/dev/shm/live_123_m0 obs=5 potion_mix=...
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
LOGS=src/live/logs; mkdir -p "$LOGS"
TS=$(date +%Y%m%d_%H%M%S)
trap live_stop_servers EXIT
trap 'exit 130' INT TERM
live_weights "${1:-}"
live_python
live_bridge_prefix
live_start_servers "$LOGS" "$TS"
KEYS="bridge=${P}c0 mbridge=${P}m0 obs=$OBS"
[ -n "$VBAR" ] && KEYS="$KEYS potion_mix=$VBAR"
echo "[live] servers ready. Controller keys (run from $ROOT):"
echo "  $KEYS"
echo "[live] Ctrl-C stops the servers"
while :; do
  for p in "${SERVER_PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null || { echo "[live] an inference server stopped:"; tail -n 20 "$LOGS"/server_*_"$TS".txt; exit 1; }
  done
  sleep 2
done
