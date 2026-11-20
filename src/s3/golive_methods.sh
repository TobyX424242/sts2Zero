#!/bin/bash
# One-off (2026-11-13, owner): switch the running S3 chain to the new methods now. Installs the staged files
# (stage5/, already checksummed; the C# build is src/s3/S3/bin/Release/net9.0.new), writes the chain's extra.env,
# then stops the running segment's process group so the supervisor validates its weights, evaluates them and starts
# the next segment with the new code from those weights. The old build and sources are kept for rollback.
# usage: src/s3/golive_methods.sh NAME SEGMENT_NAME     e.g.  s3a s3a_s011
n=$1; SN=$2
cd "$(dirname "$0")/../.."
L=src/s3/runs/$n/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
B=src/s3/S3/bin/Release
[ -f $B/net9.0.new/S3.dll ] || { echo "no new build"; exit 1; }
for f in src/s2/py/train_combat.py src/s3/py/train_macro_s3.py src/s3/run_s3.sh src/s3/backup_loop.sh \
         src/s3/backup_leftovers.sh src/s3/s3_supervise.sh; do
  [ -f stage5/$f ] || { echo "missing stage5/$f"; exit 1; }
  cp -p stage5/$f $f.new && mv $f.new $f
done
rm -rf $B/net9.0.pre_methods; mv $B/net9.0 $B/net9.0.pre_methods && mv $B/net9.0.new $B/net9.0
cat > src/s3/runs/$n/extra.env.new <<'EOF'
EXTRA_ACTOR_ARGS="hall_sims=0.5 boss_pfull=1 practice_boss=1 practice_min_act=2 reuse=1"
EXTRA_CTRAIN_ARGS="--surprise 0.5 --ema 0.999"
CARRY=1
EOF
mv src/s3/runs/$n/extra.env.new src/s3/runs/$n/extra.env
log "new methods installed (owner): $(tr '\n' ' ' < src/s3/runs/$n/extra.env); old build $B/net9.0.pre_methods"
# the running segment: its run_s3.sh leads its own process group (started with setsid)
G=$(ps -eo pid=,args= | awk -v s="$SN" '$2 ~ /bash$/ && $3 ~ /run_s3\.sh$/ && $4 == s {print $1}')
if [ -n "$G" ] && [ $(echo "$G" | wc -w) -eq 1 ]; then
  log "segment $SN stopped by hand so that the next segment starts with the new methods (owner)"
  kill -TERM -- -$G 2>/dev/null; sleep 10; kill -KILL -- -$G 2>/dev/null
  rm -f /dev/shm/s3_${SN}_*
else
  log "segment $SN not found running (pids: '$G'): nothing stopped; the new methods start with the next segment"
fi
