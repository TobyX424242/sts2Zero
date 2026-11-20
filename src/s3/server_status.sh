#!/bin/bash
# One-line status of S3 jobs for the watchdog (run on the server): disk, load, GPU, process counts, newest run-log
# line, and error lines in S3 logs modified in the last 10 minutes.
cd /root/sts2
L=$(cut -d' ' -f1 /proc/loadavg); G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
A=$(pgrep -fc "S3.dll play"); I=$(pgrep -fc "py/s3_server.py"); T=$(pgrep -fc "py/train_combat.py"); M=$(pgrep -fc "py/train_macro_s3.py")
P=$(pgrep -fc "s3_pipeline|run_s3.sh|eval_s3.sh")
LAST=$(tail -1 $(ls -t s3_pipeline*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-140)
ERR=$(find src/s3/runs src/s3/out -name "*.txt" -mmin -10 2>/dev/null | xargs grep -lE "Traceback|Unhandled exception|TimeoutException|no progress for|FAILED" 2>/dev/null | head -3 | tr '\n' ' ')
RUNS=$(cat $(ls -td src/s3/runs/*/ 2>/dev/null | head -1)runs/*.jsonl 2>/dev/null | wc -l)
D=$(df / | awk 'NR==2{print $5}' | tr -d %)
echo "disk=$D load=$L gpu=$G actors=$A servers=$I ctrain=$T mtrain=$M pipeline=$P runs=$RUNS | last: $LAST | err: $ERR"
