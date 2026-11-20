#!/bin/bash
# Owner 2026-11-15: "after the shadow battle network is confirmed better, each segment becomes 2 hours". Waits for the
# supervisor's evaluations of segment OLD (last segment of the old playing network) and NEW (first segment played by
# the switched network), compares them paired on the same seeds, and if NEW won at least as many of the discordant
# seeds as OLD, sets SEG=SECONDS in config.env (the supervisor re-reads it at every segment start). Otherwise logs it
# and changes nothing.
#   setsid nohup src/s3/confirm_switch_seg.sh OLD NEW SECONDS > log 2>&1 < /dev/null &
OLD=$1; NEW=$2; SECS=${3:-7200}
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" >> $L; echo "$*"; }
for k in $OLD $NEW; do until grep -q "eval: seg $k " $L; do sleep 60; done; done
EO=$S/eval/seg_$(printf %03d $OLD)/tf64.jsonl; EN=$S/eval/seg_$(printf %03d $NEW)/tf64.jsonl
V=$(./.venv/bin/python -I - "$EO" "$EN" <<'PY'
import json, sys
def won(p):
    o = {}
    for l in open(p):
        try: r = json.loads(l)
        except ValueError: continue
        if not r.get("error"): o[r["seed"]] = bool(r["won"])
    return o
A, B = won(sys.argv[1]), won(sys.argv[2]); s = set(A) & set(B)
b = sum(A[x] and not B[x] for x in s); c = sum(B[x] and not A[x] for x in s)
print(f"{'CONFIRMED' if c >= b else 'NOT_CONFIRMED'} {len(s)} paired seeds: old {sum(A[x] for x in s)} won, new {sum(B[x] for x in s)} won; old-only {b}, new-only {c}")
PY
)
log "switch check (segment $OLD old network vs segment $NEW switched network): $V"
if [[ "$V" == CONFIRMED* ]]; then
  sed -i "s/^export SEG=.*/export SEG='$SECS'/" $S/config.env
  log "owner: switched network confirmed: segments of $SECS s from the next segment start ($(grep SEG $S/config.env))"
else
  log "switched network not confirmed: segment length unchanged ($(grep SEG $S/config.env)); owner to decide"
fi
