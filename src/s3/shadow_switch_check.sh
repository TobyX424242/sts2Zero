#!/bin/bash
# Shadow-network switch rule (owner, 2026-11-15: "replace the battle network when the updated one beats the existing one").
# For every finished segment K of the chain that trained a shadow (K >= FIRST), evaluates <seg K>/shadow.pt on the same
# fixed seeds and settings as the supervisor's evaluation of <seg K>/weights.pt (paired), and appends the comparison to
# runs/<chain>/shadow_switch.txt. Arms src/s3/switch_combat.sh (the next segment the supervisor starts plays with
# the shadow) when, pooled over the last two compared segments (EVAL_COUNT seeds each):
#   (1) the shadow won more runs than the playing network (paired, shadow-only wins > playing-only wins), and
#   (2) on the fresh chunks of the latest segment start (shadow_compare.txt, src/s3/shadow_watch.sh) the shadow's
#       end-HP and survival losses are not above the playing network's (x 1.02), when such a line exists.
# Exits after arming the switch. DRY=1: log the decision only.
# usage: setsid nohup src/s3/shadow_switch_check.sh CHAIN FIRST_SEGMENT > log 2>&1 < /dev/null &
C=${1:-s3a}; K=${2:?first segment that trains a shadow}
cd "$(dirname "$0")/../.."
S=src/s3/runs/$C; OUT=$S/shadow_switch.txt
. $S/config.env 2>/dev/null
EC=${EVAL_COUNT:-300}; EN=${EVAL_N:-30}; CS=${CSERVERS:-1}; ASCL=${ASC:-10}
log() { echo "[$(date -u +%F\ %T)] $*" >> $OUT; }
log "shadow switch check started: chain $C from segment $K, $EC seeds per segment, pooled over 2 segments"
while true; do
  P=src/s3/runs/${C}_s$(printf %03d $K)
  # wait for the supervisor's verdict on segment K
  until grep -qE "segment $K (ok|FAILED)" $S/supervisor.log 2>/dev/null; do sleep 60; done
  if grep -qE "segment $K FAILED" $S/supervisor.log || [ ! -f $P/shadow.pt ]; then
    log "segment $K: failed or no shadow.pt, skipped"; K=$((K + 1)); continue
  fi
  E=$S/eval/seg_$(printf %03d $K); ES=$S/eval/seg_$(printf %03d $K)_shadow
  COUNT=$EC ASC=$ASCL CONC=12 CSERVERS=$CS src/s3/eval_s3.sh $P/shadow.pt $P/macro.pt $ES $EN "tf64:64" > $S/eval_${K}_shadow.log 2>&1
  for i in $(seq 90); do [ -f $E/summary.txt ] && break; sleep 20; done   # the supervisor's evaluation of the same segment
  [ -f $E/summary.txt ] || { log "segment $K: no supervisor evaluation in $E, skipped"; K=$((K + 1)); continue; }
  D=$S/eval/cmp_$(printf %03d $K); mkdir -p $D
  cp $E/tf64.jsonl $D/playing.jsonl; cp $E/tf64_fights.jsonl $D/playing_fights.jsonl
  cp $ES/tf64.jsonl $D/shadow.jsonl; cp $ES/tf64_fights.jsonl $D/shadow_fights.jsonl
  ./.venv/bin/python -I src/s3/summarize_runs.py $D/playing.jsonl $D/shadow.jsonl > $D/summary.txt 2>&1
  { echo "== segment $K: playing $P/weights.pt vs shadow $P/shadow.pt"; grep -E "^== |^paired" $D/summary.txt; } >> $OUT
  # decision over the last two compared segments
  V=$(./.venv/bin/python -I - "$S" "$K" <<'EOF'
import json, os, re, sys, glob
S, K = sys.argv[1], int(sys.argv[2])
def won(p):
    out = {}
    for l in open(p):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if not r.get("error") and not r.get("perturbed"):
            out[r["seed"]] = bool(r["won"])
    return out
b = c = n = 0; segs = []
for k in (K - 1, K):
    d = os.path.join(S, "eval", f"cmp_{k:03d}")
    if not os.path.exists(os.path.join(d, "shadow.jsonl")):
        continue
    A, B = won(os.path.join(d, "playing.jsonl")), won(os.path.join(d, "shadow.jsonl"))
    both = set(A) & set(B); segs.append(k); n += len(both)
    b += sum(A[s] and not B[s] for s in both); c += sum(B[s] and not A[s] for s in both)
loss_ok, loss_txt = True, "no fresh-chunk comparison"
p = os.path.join(S, "shadow_compare.txt")
if os.path.exists(p):
    # the block written at the start of segment K + 1 compares segment K's networks (older blocks may be another shadow)
    blocks = [x for x in open(p).read().split("== segment")[1:] if re.match(r"\s*(\d+)", x) and int(re.match(r"\s*(\d+)", x).group(1)) == K + 1]
    last = blocks[-1] if blocks else ""
    rows = {m.group(1): m.groups() for m in re.finditer(r"^(playing|shadow)\s+\d+\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", last, re.M)}
    if "playing" in rows and "shadow" in rows:
        hp_p, sv_p, hp_s, sv_s = float(rows["playing"][3]), float(rows["playing"][4]), float(rows["shadow"][3]), float(rows["shadow"][4])
        loss_ok = hp_s <= 1.02 * hp_p and sv_s <= 1.02 * sv_p
        loss_txt = f"fresh chunks: hp loss {hp_s:.4f} vs {hp_p:.4f}, survival {sv_s:.4f} vs {sv_p:.4f}"
ok = len(segs) >= 2 and c > b and loss_ok
print(f"{'SWITCH' if ok else 'KEEP'} segments {segs}: {n} paired seeds, shadow-only wins {c}, playing-only wins {b} "
      f"({100 * (c - b) / max(1, n):+.1f} points); {loss_txt}")
EOF
)
  log "decision after segment $K: $V"
  if [[ "$V" == SWITCH* ]]; then
    if [ "${DRY:-0}" = "1" ]; then log "DRY=1: switch not armed"
    else src/s3/switch_combat.sh "shadow (ext network, current layout) beat the playing network: $V"; log "switch armed"; exit 0; fi
  fi
  K=$((K + 1))
done
