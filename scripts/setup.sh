#!/usr/bin/env bash
# Sets up the Python environment of sts2Zero (Linux / macOS).
#
#   scripts/setup.sh
#
# 1. Finds uv on PATH, or installs it into .tools/uv with the official installer in its "unmanaged" mode: no
#    change to your shell profile or PATH, nothing written outside that folder (UV_INSTALL_DIR overrides it).
# 2. Runs `uv sync --locked`: creates .venv in the repository root with Python 3.12, torch, numpy and orjson as
#    pinned in uv.lock (torch with CUDA 12.8 on Linux: about 3 GB to download).
# 3. Checks for the .NET 9 SDK / runtime (needed for the live-play controller and the game mod).
#
# Unless you set them yourself, uv's download cache and any Python it downloads stay inside the repository
# (.cache/uv, .tools/python), so deleting the folder removes everything.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.tools/python}"

# ---- uv
UV=""
if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
else
  DIR="${UV_INSTALL_DIR:-$ROOT/.tools/uv}"
  for c in "$DIR/uv" "$DIR/bin/uv"; do [ -x "$c" ] && { UV=$c; break; }; done
  if [ -z "$UV" ]; then
    echo "[setup] uv not found: installing it into $DIR"
    mkdir -p "$DIR"
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$DIR" sh
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$DIR" sh
    else
      echo "[setup] neither curl nor wget found: install uv yourself (https://docs.astral.sh/uv/) and run this again"
      exit 1
    fi
    for c in "$DIR/uv" "$DIR/bin/uv"; do [ -x "$c" ] && { UV=$c; break; }; done
    [ -n "$UV" ] || { echo "[setup] the uv installer finished but no uv binary is in $DIR"; exit 1; }
  fi
fi
echo "[setup] uv: $UV ($("$UV" --version))"

# ---- Python environment
echo "[setup] creating .venv (uv sync --locked)"
"$UV" sync --locked
PY="$ROOT/.venv/bin/python"
"$PY" -c "import sys, torch, numpy, orjson; print(f'[setup] Python {sys.version.split()[0]}, torch {torch.__version__}, numpy {numpy.__version__}, orjson {orjson.__version__}')"
if [ "${SETUP_SKIP_GPU_CHECK:-0}" != "1" ]; then
  "$PY" -c "import torch; ok = torch.cuda.is_available(); print('[setup] CUDA GPU: ' + (torch.cuda.get_device_name(0) if ok else 'none found (the inference servers will run on the CPU: slower)'))"
fi

# ---- .NET 9
if command -v dotnet >/dev/null 2>&1; then
  sdk=$(dotnet --list-sdks 2>/dev/null | grep -E '^9\.' | tail -n 1 || true)
  rt=$(dotnet --list-runtimes 2>/dev/null | grep -E '^Microsoft\.NETCore\.App 9\.' | tail -n 1 || true)
  if [ -n "$sdk" ]; then echo "[setup] .NET SDK: ${sdk%% *} (builds the mod and the controller from source)"
  elif [ -n "$rt" ]; then echo "[setup] .NET runtime: $rt (no SDK: use the prebuilt binaries, scripts/fetch_weights.sh --binaries)"
  else echo "[setup] dotnet found, but no .NET 9 runtime or SDK: install the .NET 9 SDK (https://dotnet.microsoft.com/download/dotnet/9.0)"; fi
else
  echo "[setup] .NET not found. Install the .NET 9 SDK (to build from source) or at least the .NET 9 runtime"
  echo "        (to run the prebuilt controller): https://dotnet.microsoft.com/download/dotnet/9.0"
  echo "        Ubuntu: sudo apt install dotnet-sdk-9.0"
fi
echo "[setup] done. Next: scripts/fetch_weights.sh"
