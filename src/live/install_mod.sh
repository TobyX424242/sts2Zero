#!/usr/bin/env bash
# Installs the AgentBridge mod into the game's mods folder (Linux). Windows: install_mod.ps1.
#
#   src/live/install_mod.sh             build from source (.NET 9 SDK) + install
#   src/live/install_mod.sh prebuilt    install the prebuilt mod from bin/AgentBridge (scripts/fetch_weights.sh --binaries)
#   src/live/install_mod.sh build       build only (output: src/live/mod/bin/Release/AgentBridge.dll)
#   src/live/install_mod.sh uninstall   remove <game>/mods/AgentBridge
#   src/live/install_mod.sh status      show what is installed
#
# Game folder: $STS2_GAME_DIR, else the first "Slay the Spire 2" found in Steam's libraries
# (~/.local/share/Steam, ~/.steam/steam, Flatpak Steam, and every library listed in their libraryfolders.vdf).
# NuGet / .NET caches stay inside the repository (.cache/).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MOD_SRC="$HERE/mod"
MOD_ID="AgentBridge"
PREBUILT="$ROOT/bin/$MOD_ID"

export NUGET_PACKAGES="${NUGET_PACKAGES:-$ROOT/.cache/nuget}"
export DOTNET_CLI_HOME="${DOTNET_CLI_HOME:-$ROOT/.cache/dotnet}"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

find_game() {
    local steam lib vdf
    local roots=("$HOME/.local/share/Steam" "$HOME/.steam/steam" "$HOME/.var/app/com.valvesoftware.Steam/.local/share/Steam")
    local libs=()
    for steam in "${roots[@]}"; do
        libs+=("$steam")
        vdf="$steam/steamapps/libraryfolders.vdf"
        if [ -f "$vdf" ]; then
            while IFS= read -r lib; do libs+=("$lib"); done < <(sed -n 's/^[[:space:]]*"path"[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "$vdf")
        fi
    done
    for lib in "${libs[@]}"; do
        if [ -d "$lib/steamapps/common/Slay the Spire 2" ]; then echo "$lib/steamapps/common/Slay the Spire 2"; return 0; fi
    done
    echo "$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2"
}

GAME_DIR="${STS2_GAME_DIR:-$(find_game)}"
DEST="$GAME_DIR/mods/$MOD_ID"
cmd="${1:-install}"

check_game() {
    [ -d "$GAME_DIR" ] || { echo "game folder not found: $GAME_DIR"; echo "set STS2_GAME_DIR=/path/to/Slay the Spire 2"; exit 1; }
}

build() {
    check_game
    local data_dir=""
    for cand in "$GAME_DIR/data_sts2_linuxbsd_x86_64" "$GAME_DIR/data_sts2_windows_x86_64"; do
        if [ -f "$cand/sts2.dll" ]; then data_dir="$cand"; break; fi
    done
    [ -n "$data_dir" ] || { echo "game assemblies (sts2.dll) not found under: $GAME_DIR (set STS2_GAME_DIR)"; exit 1; }
    command -v dotnet >/dev/null 2>&1 || { echo "dotnet not found: install the .NET 9 SDK, or use: $0 prebuilt"; exit 1; }
    echo "== building $MOD_ID against: $data_dir"
    dotnet build "$MOD_SRC/AgentBridge.csproj" -c Release -p:STS2GameDir="$GAME_DIR" -p:STS2DataDir="$data_dir" -nologo -v minimal
    [ -f "$MOD_SRC/bin/Release/$MOD_ID.dll" ] || { echo "build output missing"; exit 1; }
}

copy_in() {   # $1 = folder holding AgentBridge.dll and AgentBridge.json
    mkdir -p "$DEST"
    cp -f "$1/$MOD_ID.dll" "$DEST/$MOD_ID.dll"
    cp -f "$1/$MOD_ID.json" "$DEST/$MOD_ID.json"
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

install() {
    build
    local stage="$ROOT/.cache/mod-stage"
    rm -rf "$stage"; mkdir -p "$stage"
    cp -f "$MOD_SRC/bin/Release/$MOD_ID.dll" "$MOD_SRC/$MOD_ID.json" "$stage/"
    copy_in "$stage"
    rm -rf "$stage"
}

install_prebuilt() {
    check_game
    [ -f "$PREBUILT/$MOD_ID.dll" ] && [ -f "$PREBUILT/$MOD_ID.json" ] || {
        echo "no prebuilt mod in $PREBUILT: run scripts/fetch_weights.sh --binaries (or unzip AgentBridge-mod.zip into bin/)"; exit 1; }
    copy_in "$PREBUILT"
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
    echo "game folder: $GAME_DIR"
    if [ -d "$DEST" ]; then ls -la "$DEST"; else echo "not installed ($DEST)"; fi
}

case "$cmd" in
    install) install ;;
    prebuilt) install_prebuilt ;;
    build) build ;;
    uninstall) uninstall ;;
    status) status ;;
    *) echo "usage: $0 [install|prebuilt|build|uninstall|status]"; exit 2 ;;
esac
