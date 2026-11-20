#!/bin/bash
# Larger combat network: value-overfitting fixes, offline on the laptop (owner, 2026-11-14, this time only).
# Every arm starts from the segment-26 networks and replays the same stream (every 4th chunk of segments 27-28, in
# time order; window and warm-up scaled by 1/4 like the stream) with src/s3/py/shadow_fix.py; the held-out set
# is the first chunks of segment 29 (fresh data, as src/s3/runs/s3a/shadow_compare.txt). Arms run one after
# another (one GPU job at a time on the laptop).
# usage: src/s3/shadow_fix.sh ARM...   (arm names below; default: all)
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
O=src/s3/out/shadow_fix; C=$O/ckpt
key() { sed -E 's/.*\/a([0-9]+)_([0-9]+)\.s2d\.zst/\2 \1 &/' | sort -n -k1,1 -k2,2 | cut -d" " -f3; }
[ -f $O/stream.txt ] || { for s in 027 028; do sed "s|^|$O/data/s$s/|" $O/want_s$s.txt | key; done > $O/stream.txt; }
[ -f $O/heldout.txt ] || sed "s|^|$O/data/s029/|" $O/want_s029.txt > $O/heldout.txt
BIG="--init $C/s026_shadow.pt --lr 1e-4 --window 1000000 --warm 1000000"
HUM="--human $(ls $O/human/*.s2d 2>/dev/null | paste -sd,)"
declare -A A=(
  [refs]="--max-steps 0 --ref small26=$C/s026_weights.pt small28=$C/s028_weights.pt big26=$C/s026_shadow.pt big28=$C/s028_shadow.pt"
  [base]="$BIG"
  [vw25]="$BIG --vw 0.25"
  [fightw]="$BIG --fight-weight"
  [drop]="$BIG --dropout 0.1 --vdrop 0.3 --wd 0.05"
  [human]="$BIG --fight-weight $HUM"
  [small]="--init $C/s026_weights.pt --lr 5e-5 --window 1000000 --warm 1000000"
)
ARMS=${*:-refs base vw25 fightw drop human small}
for arm in $ARMS; do
  [ -n "${A[$arm]}" ] || { echo "unknown arm $arm"; exit 1; }
  echo "[$(date +%T)] arm $arm: ${A[$arm]}" | tee -a $O/arms.log
  ./.venv/bin/python -I src/s3/py/shadow_fix.py --out $O/arm_$arm --stream $O/stream.txt --heldout $O/heldout.txt \
    ${A[$arm]} > $O/arm_$arm.out 2>&1 || echo "[$(date +%T)] arm $arm FAILED" | tee -a $O/arms.log
  echo "[$(date +%T)] arm $arm done: $(tail -1 $O/arm_$arm/log.jsonl 2>/dev/null | cut -c1-300)" | tee -a $O/arms.log
done
