#!/bin/bash
# CUDA MPS control daemon for S3 training segments (opt-in: MPS=1 for run_s3.sh, e.g. in the chain's extra.env).
# With MPS the inference servers and trainers of a segment share one MPS server, so their kernels run concurrently on
# the GPU instead of time-slicing between separate contexts.
# usage: src/s3/mps.sh start|status|stop [DIR]      DIR default /tmp/s3_mps (pipe DIR/pipe, logs DIR/log; keep it
#        short: the control socket path must stay below 108 characters, a long path makes the daemon fail silently)
#   start   start the daemon unless one already answers on DIR/pipe (under a lock: never two daemons), then check that
#           a CUDA client works through it (torch, 120 s timeout). Exit 0 = usable: clients must run with
#           CUDA_MPS_PIPE_DIRECTORY=DIR/pipe (run_s3.sh exports it for its own processes only). Exit 1 = not usable;
#           a daemon started by this call is stopped again. The daemon detaches (own session): it survives the segment
#           and its process-group kill, and the next segment reuses it.
#   status  exit 0 if a daemon answers; prints its server list
#   stop    ask the daemon to quit. Do it only while no S3 segment is running (between segments, or after the
#           supervisor ended): the CUDA contexts of running MPS clients live in the MPS server.
# Processes without CUDA_MPS_PIPE_DIRECTORY=DIR/pipe (eval_s3.sh, other runs) do not use this daemon and run as before
# (non-MPS contexts share the GPU with an MPS server by time-slicing, as with any other process).
CMD=$1; DIR=${2:-/tmp/s3_mps}
W=$(cd "$(dirname "$0")/../.." && pwd)
export CUDA_MPS_PIPE_DIRECTORY=$DIR/pipe CUDA_MPS_LOG_DIRECTORY=$DIR/log
say() { echo "[$(date +%T)] mps: $*"; }
answers() { echo get_server_list | timeout 10 nvidia-cuda-mps-control > /dev/null 2>&1; }
case $CMD in
  status)
    if answers; then say "daemon running on $DIR/pipe; servers: $(echo get_server_list | timeout 10 nvidia-cuda-mps-control | tr '\n' ' ')"; exit 0
    else say "no daemon on $DIR/pipe"; exit 1; fi ;;
  stop)
    if answers; then echo quit | timeout 30 nvidia-cuda-mps-control; say "quit sent to the daemon on $DIR/pipe"
    else say "no daemon on $DIR/pipe"; fi
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 start|status|stop [DIR]"; exit 2 ;;
esac
command -v nvidia-cuda-mps-control > /dev/null || { say "nvidia-cuda-mps-control not found"; exit 1; }
[ ${#DIR} -le 60 ] || { say "DIR $DIR too long for the control socket"; exit 1; }
mkdir -p $DIR/pipe $DIR/log || { say "cannot create $DIR"; exit 1; }
exec 9> $DIR/lock
flock -w 120 9 || { say "lock $DIR/lock busy for 120 s"; exit 1; }
probe() {   # a CUDA client through the daemon: creates (or reuses) the MPS server; a broken MPS setup fails or hangs here
  timeout 120 $W/.venv/bin/python -I -c "
import torch
x = torch.ones(256, 256, device='cuda')
assert float((x @ x).sum()) == 256.0 ** 3
" > $DIR/log/probe.txt 2>&1 && [ -n "$(echo get_server_list | timeout 10 nvidia-cuda-mps-control)" ]
}
start_daemon() {
  # no daemon answers here: pipe files left by one that died with the container (interruptible servers; from branch
  # resilient, 2026-11-18) are removed first
  find $DIR/pipe -mindepth 1 -delete 2>/dev/null
  nvidia-cuda-mps-control -d 9>&- || { say "daemon start failed (exit $?); see $DIR/log/control.log"; return 1; }
  for i in $(seq 50); do answers && break; sleep 0.1; done
  answers || { say "daemon started but does not answer; see $DIR/log/control.log"; return 1; }
  say "daemon started on $DIR/pipe"
}
stop_daemon() { echo quit | timeout 30 nvidia-cuda-mps-control > /dev/null 2>&1; for i in $(seq 100); do answers || return 0; sleep 0.1; done; }
started=0
if answers; then say "daemon already running on $DIR/pipe: reused"
else start_daemon || exit 1; started=1; fi
if probe; then say "probe ok: MPS server $(echo get_server_list | timeout 10 nvidia-cuda-mps-control | tr '\n' ' ')"; exit 0; fi
say "probe FAILED: $(tail -n 3 $DIR/log/probe.txt | tr '\n' ' ') $(tail -n 2 $DIR/log/control.log 2>/dev/null | tr '\n' ' ')"
if [ $started = 0 ]; then   # a daemon left by an earlier segment that no longer works: one restart
  stop_daemon; say "reused daemon stopped; starting a new one"
  if start_daemon; then
    started=1
    if probe; then say "probe ok after restart: MPS server $(echo get_server_list | timeout 10 nvidia-cuda-mps-control | tr '\n' ' ')"; exit 0; fi
    say "probe FAILED again: $(tail -n 3 $DIR/log/probe.txt | tr '\n' ' ')"
  fi
fi
[ $started = 1 ] && { stop_daemon; say "daemon started by this call stopped"; }
exit 1
