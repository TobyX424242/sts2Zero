#!/bin/bash
# Makes the next segment of chain CHAIN (default s3a) play with the shadow combat network: removes SHADOW_* lines from
# extra.env (no shadow trainer any more) and writes runs/<CHAIN>/combat_switch, which s3_supervise.sh reads before it
# starts a segment (it then uses <last good segment>/shadow.pt as COMBAT_INIT, and run_s3.sh loads carry_shadow.bin if
# present).
# usage:
#   src/s3/switch_combat.sh "reason"             # CHAIN=${CHAIN:-s3a} (backward compatible)
#   src/s3/switch_combat.sh CHAIN "reason"       # e.g. src/s3/switch_combat.sh s3ic "shadow won +3.5 pp"
cd "$(dirname "$0")/../.."
if [ $# -ge 2 ]; then
  C=$1; REASON=$2
else
  C=${CHAIN:-s3a}; REASON=$1
fi
S=src/s3/runs/$C
[ -d "$S" ] || { echo "chain dir $S not found"; exit 1; }
grep -v -E "^(SHADOW_ARGS|SHADOW_CARRY_KEEP|SHADOW_PRELOAD_CARRY|SHADOW_FROM_PLAYING|SHADOW_GPU)=" $S/extra.env > $S/extra.env.tmp && mv $S/extra.env.tmp $S/extra.env
echo "$REASON" > $S/combat_switch_reason; touch $S/combat_switch
echo "[$(date -u +%F\ %T)] combat switch armed for $C (next segment): $REASON" | tee -a $S/supervisor.log
