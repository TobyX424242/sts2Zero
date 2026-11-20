# Shared by play_live.sh and serve.sh (sourced, not run). Windows: common.ps1.
# Sets LIVE (this folder) and ROOT (the repository root); defines the functions below.

LIVE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$LIVE/../.." && pwd)
DEFAULT_WEIGHTS=weights/ic113_s029            # laid out by scripts/fetch_weights.sh

# live_weights [DIR]: sets WEIGHTS, CW (combat network), MW (macro ensemble), VBAR (potion_mix table, may be empty).
# Accepts both layouts: weights.pt / macro.pt / vbar.json (or vbar_*.json, the newest), and the release names
# *_weights.pt / *_macro.pt / *_vbar.json.
live_weights() {
  WEIGHTS=${1:-}
  if [ -z "$WEIGHTS" ]; then
    WEIGHTS=$DEFAULT_WEIGHTS
    [ -d "$WEIGHTS" ] || [ ! -d src/live/weights/s3ic_s029 ] || WEIGHTS=src/live/weights/s3ic_s029
  fi
  if [ ! -d "$WEIGHTS" ]; then
    echo "[live] weights folder not found: $WEIGHTS"
    echo "       download the released weights first: scripts/fetch_weights.sh"
    return 1
  fi
  CW=""; MW=""; VBAR=""
  if [ -f "$WEIGHTS/weights.pt" ]; then CW=$WEIGHTS/weights.pt; else CW=$(ls -t "$WEIGHTS"/*_weights.pt 2>/dev/null | head -n 1 || true); fi
  if [ -f "$WEIGHTS/macro.pt" ]; then MW=$WEIGHTS/macro.pt; else MW=$(ls -t "$WEIGHTS"/*_macro.pt 2>/dev/null | head -n 1 || true); fi
  if [ -f "$WEIGHTS/vbar.json" ]; then VBAR=$WEIGHTS/vbar.json
  else VBAR=$(ls -t "$WEIGHTS"/vbar_*.json "$WEIGHTS"/*_vbar.json 2>/dev/null | head -n 1 || true); fi
  [ -n "$CW" ] || { echo "[live] no combat network in $WEIGHTS (weights.pt or *_weights.pt)"; return 1; }
  [ -n "$MW" ] || { echo "[live] no macro network in $WEIGHTS (macro.pt or *_macro.pt)"; return 1; }
  return 0
}

# live_python: sets PY (env PY, else <root>/.venv/bin/python).
live_python() {
  PY=${PY:-}
  if [ -z "$PY" ] && [ -x "$ROOT/.venv/bin/python" ]; then PY=$ROOT/.venv/bin/python; fi
  [ -n "$PY" ] || { echo "[live] no Python environment (.venv): run scripts/setup.sh first, or set PY=/path/to/python"; return 1; }
  return 0
}

# live_bridge_prefix: sets P, the path prefix of the bridge files (shared memory on Linux, a temp file elsewhere).
live_bridge_prefix() {
  if [ -d /dev/shm ] && [ -w /dev/shm ]; then P=/dev/shm/live_$$_; else P=${TMPDIR:-/tmp}/live_$$_; P=${P//\/\//\/}; fi
}

SERVER_PIDS=()

# live_start_servers LOGDIR TS: starts the combat and the macro inference server (src/s3/py/s3_server.py) for one
# actor and waits until both are ready. Needs CW, MW, PY, P. Sets OBS (the combat network's observation layout).
# env: DEVICE (default: cuda when available, else cpu), CAPACITY (16), CPU_THREADS (torch threads per server on the
#      CPU, default 4: the networks run small batches, which are slower with many threads)
live_start_servers() {
  local logs=$1 ts=$2 dev=() n=0 i p d
  # the observation layout of the combat checkpoint (config "obs"), as src/s3/eval_s3.sh reads it
  OBS=$("$PY" -I -c "import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('config',{});print(int(c.get('obs',4)))" "$CW" 2>/dev/null || echo 5)
  d=${DEVICE:-}
  [ -n "$d" ] || d=$("$PY" -I -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || true)
  [ -n "$d" ] && dev=(--device "$d")
  [ "$d" = cpu ] && dev+=(--threads "${CPU_THREADS:-4}")
  DEVICE_USED=${d:-auto}
  local cap=${CAPACITY:-16}
  # one actor: combat and macro servers as in src/s3/eval_s3.sh / src/ic113/eval_ic113.sh, without the idle exit
  "$PY" -I src/s3/py/s3_server.py --serve combat --prefix "${P}c" --mprefix "${P}m" --first 0 --actors 1 --capacity "$cap" \
    --weights "$CW" --macro "$MW" ${dev[@]+"${dev[@]}"} --idle-exit 1e9 > "$logs/server_combat_$ts.txt" 2>&1 &
  SERVER_PIDS+=($!)
  "$PY" -I src/s3/py/s3_server.py --serve macro --prefix "${P}c" --mprefix "${P}m" --actors 1 --min-period 0.02 --capacity "$cap" \
    --weights "$CW" --macro "$MW" --mlayout 3 ${dev[@]+"${dev[@]}"} --idle-exit 1e9 > "$logs/server_macro_$ts.txt" 2>&1 &
  SERVER_PIDS+=($!)
  echo "[live] inference servers starting (weights $WEIGHTS, device $DEVICE_USED, obs $OBS); logs $logs/server_*_$ts.txt"
  for i in $(seq 1 600); do
    n=$(cat "$logs/server_combat_$ts.txt" "$logs/server_macro_$ts.txt" 2>/dev/null | grep -c "s3 server ready" || true)
    [ "$n" -ge 2 ] && break
    for p in "${SERVER_PIDS[@]}"; do
      kill -0 "$p" 2>/dev/null || { echo "[live] an inference server died:"; tail -n 20 "$logs"/server_*_"$ts".txt; return 1; }
    done
    sleep 0.5
  done
  [ "$n" -ge 2 ] || { echo "[live] inference servers not ready after 300 s"; return 1; }
  grep -h "s3 server ready" "$logs/server_combat_$ts.txt" "$logs/server_macro_$ts.txt" | sed 's/^/[live] /'
  return 0
}

# live_stop_servers: stops the servers started above and removes the bridge files.
live_stop_servers() {
  local p
  for p in "${SERVER_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  sleep 0.5
  for p in "${SERVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true; done
  SERVER_PIDS=()
  [ -n "${P:-}" ] && rm -f "${P}"* 2>/dev/null || true
}

# live_dotnet_env: .NET settings shared by the build and the controller (caches inside the repository).
live_dotnet_env() {
  export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
  export NUGET_PACKAGES=${NUGET_PACKAGES:-$ROOT/.cache/nuget} DOTNET_CLI_HOME=${DOTNET_CLI_HOME:-$ROOT/.cache/dotnet}
  # also set by LiveAgent.runtimeconfig.json (no quick JIT for loops): see the comment in agent/LiveAgent.csproj
  export DOTNET_TC_OnStackReplacement=0
}

# live_controller LOGDIR TS: sets DLL, the controller to run. A prebuilt one in <root>/bin/LiveAgent (from the
# release, scripts/fetch_weights.sh --binaries) is used when present; otherwise it is built from source (needs the
# .NET 9 SDK) and rebuilt when a source file is newer. env LIVEAGENT=source forces the source build,
# LIVEAGENT=prebuilt refuses to build.
live_controller() {
  local logs=$1 ts=$2 mode=${LIVEAGENT:-auto}
  DLL=$ROOT/bin/LiveAgent/LiveAgent.dll
  if [ "$mode" != "source" ] && [ -f "$DLL" ]; then echo "[live] controller: prebuilt $DLL"; return 0; fi
  if [ "$mode" = "prebuilt" ]; then echo "[live] no prebuilt controller at $DLL (scripts/fetch_weights.sh --binaries)"; return 1; fi
  DLL=$LIVE/agent/artifacts/bin/LiveAgent/release/LiveAgent.dll
  if [ ! -f "$DLL" ] || [ -n "$(find "$LIVE/agent" src/s3/S3 src/s2/S2 src/s0/S0.Sim src/sims/iRyougi-sts2-sim-v113/src/Sts2Sim.Core \
       -name '*.cs' -newer "$DLL" -not -path '*/artifacts/*' -not -path '*/obj/*' -not -path '*/bin/*' 2>/dev/null | head -n 1)" ]; then
    command -v dotnet >/dev/null 2>&1 || { echo "[live] dotnet not found: install the .NET 9 SDK, or use the prebuilt controller (scripts/fetch_weights.sh --binaries)"; return 1; }
    echo "[live] building LiveAgent from source (log $logs/build_$ts.txt)"
    "$LIVE/agent/build.sh" > "$logs/build_$ts.txt" 2>&1 || { tail -n 30 "$logs/build_$ts.txt"; return 1; }
  fi
  echo "[live] controller: $DLL"
  return 0
}
