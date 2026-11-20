#!/usr/bin/env bash
# Build the AgentBridge mod and install it into the game's mods folder (run by the owner).
#
#   src/live/install_mod.sh             build + install
#   src/live/install_mod.sh build       build only (output: src/live/mod/bin/Release/AgentBridge.dll)
#   src/live/install_mod.sh uninstall   remove <game>/mods/AgentBridge
#   src/live/install_mod.sh status      show what is installed
#
# Game folder: $STS2_GAME_DIR, default the Steam library path below.
# All .NET / NuGet caches stay inside the project (.cache/), never in ~/.cache or /tmp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/mnt/4dbcffbd-bd6e-43c4-bc27-9274605ea5e4/AI/Slay_the_Spire_2_agent"
GAME_DIR="${STS2_GAME_DIR:-/mnt/4dbcffbd-bd6e-43c4-bc27-9274605ea5e4/SteamLibrary/steamapps/common/Slay the Spire 2}"
MOD_SRC="$HERE/mod"
MOD_ID="AgentBridge"
DEST="$GAME_DIR/mods/$MOD_ID"

export NUGET_PACKAGES="$PROJECT_ROOT/.cache/nuget"
export DOTNET_CLI_HOME="$PROJECT_ROOT/.cache/dotnet"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

cmd="${1:-install}"

build() {
    local data_dir=""
    for cand in "$GAME_DIR/data_sts2_linuxbsd_x86_64" "$GAME_DIR/data_sts2_windows_x86_64"; do
        if [ -f "$cand/sts2.dll" ]; then data_dir="$cand"; break; fi
    done
    [ -n "$data_dir" ] || { echo "game assemblies not found under: $GAME_DIR (set STS2_GAME_DIR)"; exit 1; }
    echo "== building $MOD_ID against: $data_dir"
    dotnet build "$MOD_SRC/AgentBridge.csproj" -c Release -p:STS2GameDir="$GAME_DIR" -p:STS2DataDir="$data_dir" -nologo -v minimal
    [ -f "$MOD_SRC/bin/Release/$MOD_ID.dll" ] || { echo "build output missing"; exit 1; }
}

install() {
    build
    mkdir -p "$DEST"
    cp -f "$MOD_SRC/bin/Release/$MOD_ID.dll" "$DEST/$MOD_ID.dll"
    cp -f "$MOD_SRC/$MOD_ID.json" "$DEST/$MOD_ID.json"
    echo "== installed into $DEST:"
    ls -la "$DEST"
    cat <<EOF

Next (first time only): start the game. A popup asks whether to load mods: choose "Load mods" (the game
then closes itself; start it again). If no popup appears, open the Mods screen from the main menu and enable
mod loading there. Mod saves live in a separate "modded" profile folder (the game
copies your normal saves there the first time), so agent runs do not touch your unmodded progress.
Then switch "Agent mode" on in the small panel at the top right and start src/live/play_live.sh.
EOF
}

uninstall() {
    if [ -d "$DEST" ]; then
        rm -f "$DEST/$MOD_ID.dll" "$DEST/$MOD_ID.json"
        rmdir "$DEST" 2>/dev/null || echo "note: $DEST not empty, left in place"
        rmdir "$GAME_DIR/mods" 2>/dev/null || true
        echo "== removed $DEST"
    else
        echo "== not installed ($DEST does not exist)"
    fi
}

status() {
    if [ -d "$DEST" ]; then ls -la "$DEST"; else echo "not installed ($DEST)"; fi
}

case "$cmd" in
    install) install ;;
    build) build ;;
    uninstall) uninstall ;;
    status) status ;;
    *) echo "usage: $0 [install|build|uninstall|status]"; exit 2 ;;
esac
