#!/usr/bin/env bash
# Downloads the network weights of a GitHub release, checks them against the release's SHA256SUMS and lays them
# out as weights/<name>/{weights.pt, macro.pt, vbar.json, manifest.json} (the folder play_live.sh uses).
# Windows: fetch_weights.ps1.
#
#   scripts/fetch_weights.sh [--tag TAG] [--name NAME] [--binaries] [--from DIR] [--repo OWNER/REPO]
#
#   --tag TAG       release tag (default v0.2.0-ic113-s029)
#   --name NAME     folder under weights/ (default: from the tag, v0.2.0-ic113-s029 -> ic113_s029)
#   --binaries      also fetch the prebuilt controller and mod: LiveAgent.zip -> bin/LiveAgent/,
#                   AgentBridge-mod.zip -> bin/AgentBridge/ (then: src/live/install_mod.sh prebuilt)
#   --from DIR      take the files from a local folder instead of downloading (same names as the release assets)
#   --repo R        GitHub repository (default TobyX424242/sts2Zero)
#
# Downloads with the GitHub CLI when it is installed and logged in, else with curl (or wget).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
REPO=TobyX424242/sts2Zero
TAG=v0.2.0-ic113-s029
NAME=""
FROM=""
BINARIES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --from) FROM=$(cd "$2" && pwd); shift 2 ;;
    --binaries) BINARIES=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
done
if [ -z "$NAME" ]; then NAME=${TAG#v}; NAME=${NAME#*-}; NAME=${NAME//-/_}; fi
DL=$ROOT/weights/.download/$TAG
DEST=$ROOT/weights/$NAME
mkdir -p "$DL"

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

USE_GH=0
if [ -z "$FROM" ] && command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then USE_GH=1; fi

fetch() {   # fetch NAME: the release asset NAME into $DL/NAME
  local f=$1
  rm -f "$DL/$f"
  if [ -n "$FROM" ]; then
    [ -f "$FROM/$f" ] || { echo "[fetch] $FROM/$f not found"; return 1; }
    cp "$FROM/$f" "$DL/$f"
  elif [ $USE_GH = 1 ]; then
    gh release download "$TAG" -R "$REPO" -p "$f" -D "$DL" --clobber
  else
    local url="https://github.com/$REPO/releases/download/$TAG/$f"
    if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 --progress-bar -o "$DL/$f" "$url"
    else wget -q --show-progress -O "$DL/$f" "$url"; fi
  fi
}

expected() { awk -v f="$1" '$2 == f || $2 == "*" f { print $1 }' "$DL/SHA256SUMS"; }

verify() {  # verify NAME [optional]: checks $DL/NAME against SHA256SUMS
  local f=$1 want got
  want=$(expected "$f")
  if [ -z "$want" ]; then
    if [ "${2:-}" = optional ]; then echo "[fetch] warning: $f is not listed in SHA256SUMS (not verified)"; return 0; fi
    echo "[fetch] $f is not listed in SHA256SUMS"; return 1
  fi
  got=$(sha256 "$DL/$f")
  [ "$got" = "$want" ] || { echo "[fetch] CHECKSUM MISMATCH for $f: expected $want, got $got"; rm -f "$DL/$f"; return 1; }
  echo "[fetch] ok  $f"
}

echo "[fetch] release $TAG of $REPO -> weights/$NAME$( [ -n "$FROM" ] && echo " (from $FROM)" )"
fetch SHA256SUMS
fetch manifest.json
verify manifest.json optional
# the weight files of the release: <prefix>_weights.pt, <prefix>_macro.pt, <prefix>_vbar.json
CW=$(awk '{ sub(/^\*/, "", $2) } $2 ~ /_weights\.pt$/ { print $2 }' "$DL/SHA256SUMS" | head -n 1)
MW=$(awk '{ sub(/^\*/, "", $2) } $2 ~ /_macro\.pt$/ { print $2 }' "$DL/SHA256SUMS" | head -n 1)
VB=$(awk '{ sub(/^\*/, "", $2) } $2 ~ /_vbar\.json$/ { print $2 }' "$DL/SHA256SUMS" | head -n 1)
[ -n "$CW" ] && [ -n "$MW" ] || { echo "[fetch] SHA256SUMS of $TAG lists no *_weights.pt / *_macro.pt"; exit 1; }

mkdir -p "$DEST"
place() {   # place ASSET TARGET: fetch + verify unless TARGET already has the right checksum
  local f=$1 target=$2
  if [ -f "$DEST/$target" ] && [ "$(sha256 "$DEST/$target")" = "$(expected "$f")" ]; then
    echo "[fetch] ok  $f (already in weights/$NAME/$target)"; return 0
  fi
  fetch "$f"; verify "$f"
  mv -f "$DL/$f" "$DEST/$target"
}
place "$CW" weights.pt
place "$MW" macro.pt
[ -n "$VB" ] && place "$VB" vbar.json
mv -f "$DL/manifest.json" "$DEST/manifest.json"
cp -f "$DL/SHA256SUMS" "$DEST/SHA256SUMS"

if [ $BINARIES = 1 ]; then
  for z in LiveAgent.zip AgentBridge-mod.zip; do
    fetch "$z"; verify "$z" optional
  done
  mkdir -p "$ROOT/bin"
  rm -rf "$ROOT/bin/LiveAgent" "$ROOT/bin/AgentBridge"
  for z in LiveAgent.zip AgentBridge-mod.zip; do
    if command -v unzip >/dev/null 2>&1; then unzip -q -o "$DL/$z" -d "$ROOT/bin"
    else python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DL/$z" "$ROOT/bin"; fi
    rm -f "$DL/$z"
  done
  echo "[fetch] prebuilt controller: bin/LiveAgent/LiveAgent.dll; prebuilt mod: bin/AgentBridge/"
fi
rm -rf "$DL"
rmdir "$ROOT/weights/.download" 2>/dev/null || true
echo "[fetch] done: weights/$NAME"
ls -la "$DEST"
