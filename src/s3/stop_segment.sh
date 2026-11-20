#!/bin/bash
# Stop one running segment (its run_s3.sh process group: actors, servers, trainers, backup loop) that runs without a
# supervisor, e.g. after a pause race (2026-11-16). TERM, 8 s, KILL; removes its shared-memory bridges.
# usage: src/s3/stop_segment.sh SEGMENT_NAME   (e.g. s3c_s001)
SN=$1
PG=$(ps -eo pgid,args | grep "^ *[0-9]* /bin/bash src/s3/run_s3.sh $SN " | awk '{print $1}' | head -1)
[ -z "$PG" ] && { echo "no run_s3.sh $SN running"; exit 1; }
echo "stopping process group $PG ($SN): $(ps -eo pgid | grep -c "^ *$PG$") processes"
kill -TERM -- -$PG 2>/dev/null; sleep 8; kill -KILL -- -$PG 2>/dev/null; sleep 1
rm -f /dev/shm/s3_${SN}_*
echo "left in group: $(ps -eo pgid | grep -c "^ *$PG$")"
