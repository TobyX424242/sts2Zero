#!/bin/bash
# Deletes outdated self-play data (owner 2026-11-18): Defect-era chain segments (layout2/3/4, looksp) and the ic113
# warm-up: fight logs, mid-segment checkpoints, replay carries, training chunks. Keeps weights.pt / macro.pt, run
# records, logs, evaluations. Never touches the live chain tree /root/exp/ic113/chain.
DRY=${DRY:-1}; tot=0
del() { for p in "$@"; do [ -e "$p" ] || continue; k=$(du -sk "$p" | cut -f1); tot=$((tot + k)); echo "$((k / 1024)) MB  $p"; [ "$DRY" = 0 ] && rm -rf "$p"; done; }
for R in /root/exp/layout2 /root/exp/layout3 /root/exp/layout4 /root/exp/looksp; do
  for d in $R/src/s3/runs/*_s[0-9][0-9][0-9] $R/src/s3/runs/*_x[0-9][0-9][0-9]; do
    [ -d "$d" ] || continue
    del $d/fights $d/ckpt $d/mckpt $d/archive $d/cdata $d/mdata $d/shadow_data $d/carry_*
  done
done
del /root/exp/out/l3gen/mdata /root/exp/ic113/warm/w2/fights /root/exp/ic113/warm/w2/ckpt /root/exp/ic113/warm/w1/fights /root/exp/ic113/warm/w1/ckpt
echo "total $((tot / 1024)) MB (DRY=$DRY)"
