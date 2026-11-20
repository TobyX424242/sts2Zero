#!/usr/bin/env bash
# Builds the two prebuilt release binaries into release-assets/ (Linux, needs the .NET 9 SDK and the game):
#
#   AgentBridge-mod.zip   AgentBridge/AgentBridge.dll + AgentBridge/AgentBridge.json, ready to unzip into <game>/mods/.
#                         Built against the installed game's sts2.dll (managed, AnyCPU).
#   LiveAgent.zip         LiveAgent/ : framework-dependent, portable `dotnet publish -c Release` output of the
#                         controller (run with `dotnet LiveAgent.dll`, needs the .NET 9 runtime; same files on Linux
#                         and Windows).
#
#   scripts/build_release_binaries.sh            game folder: $STS2_GAME_DIR, else auto-detected (src/live/install_mod.sh)
#
# Every run starts from clean build folders, so the result depends only on the sources and the game's assemblies.
# Also writes release-assets/SHA256SUMS.binaries (append those lines to SHA256SUMS before publishing, so that
# scripts/fetch_weights.sh --binaries can verify the downloads). Caches stay inside the repository (.cache/).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export NUGET_PACKAGES=$ROOT/.cache/nuget DOTNET_CLI_HOME=$ROOT/.cache/dotnet
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
OUT=$ROOT/release-assets
STAGE=$ROOT/.cache/release-build
mkdir -p "$OUT"
rm -rf "$STAGE"; mkdir -p "$STAGE"

zipdir() {   # zipdir ZIP BASEDIR ENTRY: zip BASEDIR/ENTRY (recursively) as ENTRY/...
  local zip=$1 base=$2 entry=$3
  rm -f "$zip"
  if command -v zip >/dev/null 2>&1; then
    (cd "$base" && zip -q -X -r "$zip" "$entry")
  else
    python3 - "$zip" "$base" "$entry" <<'PY'
import os, sys, zipfile
zip_path, base, entry = sys.argv[1:]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for d, _, files in os.walk(os.path.join(base, entry)):
        for f in sorted(files):
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, base))
PY
  fi
}

# ---- AgentBridge mod (clean build against the game's assemblies, through install_mod.sh build)
echo "== AgentBridge mod"
rm -rf src/live/mod/bin src/live/mod/obj
src/live/install_mod.sh build
mkdir -p "$STAGE/mod/AgentBridge"
cp src/live/mod/bin/Release/AgentBridge.dll src/live/mod/AgentBridge.json "$STAGE/mod/AgentBridge/"
zipdir "$OUT/AgentBridge-mod.zip" "$STAGE/mod" AgentBridge

# ---- LiveAgent controller (framework-dependent, no apphost: the same files run on Linux and Windows)
echo "== LiveAgent controller"
dotnet publish src/live/agent/LiveAgent.csproj -c Release --artifacts-path "$STAGE/agent-artifacts" \
  -p:UseAppHost=false -o "$STAGE/agent/LiveAgent" -nologo -v:q
[ -f "$STAGE/agent/LiveAgent/LiveAgent.dll" ] || { echo "publish output missing"; exit 1; }
zipdir "$OUT/LiveAgent.zip" "$STAGE/agent" LiveAgent

(cd "$OUT" && sha256sum AgentBridge-mod.zip LiveAgent.zip > SHA256SUMS.binaries)
echo "== done:"
ls -la "$OUT/AgentBridge-mod.zip" "$OUT/LiveAgent.zip"
cat "$OUT/SHA256SUMS.binaries"
echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo ' (with uncommitted changes)')"
unzip -l "$OUT/AgentBridge-mod.zip" 2>/dev/null || true
