#!/bin/bash
# Watch panel for the S3 training chains (local web page; Python standard library only, nothing is installed).
#   src/s3/panel/run_panel.sh [port]        default port 8733 -> open http://127.0.0.1:8733/
#   PANEL_RUN=s3v3 src/s3/panel/run_panel.sh  pin a chain (default: follow the newest chain, now s3ic)
# Listens on 127.0.0.1 only. Ctrl-C stops it (and a sync that is running).
# Chains come from two Nextcloud roots: sts2_backup/s3 (Defect chains s3a, s3c, s3v3, lsa/lsb -> src/server_backup/s3/)
# and sts2_backup/s3ic (Ironclad chain s3ic -> src/server_backup/s3ic/); PANEL_REMOTE overrides the list.
# The page shows the local copy of a chain's logs, a "Chain" selector, the A10 evaluation along the lineage
# (s3a -> s3c -> s3v3; s3ic stands alone), for s3ic also the ascension curriculum, results by ascension level, the
# mixed potion valuation (Vbar, potion_mix counters) and the hook's evaluation report, and buttons to pull the latest
# from Nextcloud with rclone (read-only there):
# "Sync light" (logs, evaluations, run records) and "Sync full" (light + fight records), for the chain shown.
# Never synced: weights, training chunks, <chain>/carry/ (the carry backup for a restore; local copies from syncs before
# 2026-11-19 under src/server_backup/s3ic/s3ic/carry/ are not used and can be deleted).
# Same from a terminal: python3 src/s3/panel/s3sync.py chains | preview|light|full [chain ...]
# Details: the header of src/s3/panel/panel.py.
cd "$(dirname "$0")/../../.." || exit 1
command -v rclone > /dev/null || echo "note: rclone is not on PATH: the page works on local data, sync will fail"
exec python3 src/s3/panel/panel.py "${1:-8733}"
