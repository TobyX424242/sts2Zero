#!/bin/bash
# ic113 warm-up: kills any S2 actor of run NAME whose resident memory exceeds LIMIT_GB (6), or whose log has not been
# written for STALE_MIN (10) minutes (actors report every 30 s), and logs it to the run's watch log. 2026-11-18: actors
# a3 and a43 of w2 stopped reporting (a synchronous call that never returned) and then grew to 33 / 6 GB; one runaway
# actor can exhaust the 93 GB box. Exits when the run is done.
NAME=$1; LIMIT=${2:-6}; STALE=${STALE_MIN:-10}; R=/root/exp/ic113/warm/$NAME
while [ ! -f $R/.done ]; do
  ps -eo pid,rss,args | grep "[S]2.dll play mode=selfplay bridge=/dev/shm/icw_${NAME}_c" | while read pid rss rest; do
    tag=$(echo "$rest" | grep -o 'tag=[a-z0-9]*'); t=${tag#tag=}
    age=$(( ( $(date +%s) - $(stat -c %Y $R/logs/$t.txt 2>/dev/null || date +%s) ) / 60 ))
    if [ $rss -gt $((LIMIT * 1048576)) ] || [ $age -ge $STALE ]; then
      kill $pid
      echo "[$(date -u +%FT%TZ)] ALERT memguard killed actor $pid ($tag): $((rss / 1048576)) GB, log $age min old" >> $R/logs/watch.txt
    fi
  done
  sleep 30
done
