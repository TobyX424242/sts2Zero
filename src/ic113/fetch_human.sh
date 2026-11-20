#!/bin/bash
# ic113 human data (server): Spire Codex per-version exports. v0.113.0 / v0.113.1 kept whole (all characters, small);
# v0.111.0 streamed and only Ironclad runs kept (disk). sha256 checked against the manifest. Then win-rate stats.
# usage: bash fetch_human.sh [OUT=/root/exp/ic113data]
set -euo pipefail
OUT=${1:-/root/exp/ic113data}
mkdir -p "$OUT"; cd "$OUT"
BASE=https://spire-codex.com
curl -sf "$BASE/api/exports/runs/manifest" -o manifest.json
sha() { python3 -c "import json,sys;print(json.load(open('manifest.json'))['versions'][sys.argv[1]]['sha256'])" "$1"; }
for v in v0.113.0 v0.113.1; do
  f=runs-$v.jsonl.gz
  if [ ! -s "$f" ]; then curl -sfL "$BASE/exports/$f" -o "$f.part"; mv -f "$f.part" "$f"; fi
  want=$(sha $v); got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "$f sha256 ok" || { echo "$f sha256 MISMATCH ($got vs $want)"; exit 1; }
done
f=runs-v0.111.0-ironclad.jsonl.gz
if [ ! -s "$f" ]; then
  # tee the raw stream into sha256sum while filtering, so the full 1.7 GB file never lands on disk
  curl -sfL "$BASE/exports/runs-v0.111.0.jsonl.gz" \
    | tee >(sha256sum | cut -d' ' -f1 > v0111.sha) \
    | gzip -dc | grep -F '"CHARACTER.IRONCLAD"' | gzip -6 > "$f.part"
  sleep 2
  want=$(sha v0.111.0); got=$(cat v0111.sha)
  [ "$want" = "$got" ] && echo "runs-v0.111.0 stream sha256 ok" || { echo "v0.111.0 sha256 MISMATCH ($got vs $want)"; exit 1; }
  mv -f "$f.part" "$f"
fi
ls -la "$OUT"
python3 - <<'EOF'
import gzip, json, re
MOD_RE = re.compile(rb'"(?:CARD|RELIC|POTION|CHARACTER|ENCOUNTER|MONSTER|EVENT|ENCHANTMENT|POWER|ORB|AFFLICTION|ACT)\.[A-Za-z0-9_]+-')
from collections import defaultdict
S = defaultdict(lambda: [0, 0, 0])
for fn in ["runs-v0.113.0.jsonl.gz", "runs-v0.113.1.jsonl.gz", "runs-v0.111.0-ironclad.jsonl.gz"]:
    bad = 0
    for line in gzip.open(fn, "rb"):
        if MOD_RE.search(line): continue
        try: r = json.loads(line)
        except Exception: bad += 1; continue
        pl = r.get("players") or []
        if len(pl) != 1 or r.get("game_mode", "standard") != "standard" or r.get("was_abandoned") or r.get("modifiers"):
            continue
        ch = (pl[0].get("character") or "").replace("CHARACTER.", "")
        if fn.startswith("runs-v0.111") and ch != "IRONCLAD": continue
        a = int(r.get("ascension") or 0); won = bool(r.get("win"))
        acts = len(r.get("map_point_history") or [])
        for k in [(r.get("build_id"), ch, "A10" if a == 10 else ("A8-9" if a >= 8 else "A0-7"))]:
            s = S[k]; s[0] += 1; s[1] += won; s[2] += acts >= 3
    print(fn, "unparsed", bad)
print("build      char         asc    n      win%   reach act3%")
for k in sorted(S, key=lambda k: (str(k[0]), k[1], k[2])):
    n, w, a3 = S[k]
    print(f"{str(k[0]):10s} {k[1]:12s} {k[2]:5s} {n:6d} {100*w/n:6.1f} {100*a3/n:6.1f}")
EOF
