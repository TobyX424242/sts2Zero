#!/bin/bash
# Discards the running segment K of chain s3a and starts it again at once from the last good segment, with new actor
# ascension weights (owner, 2026-11-14: "stop training on A5, increase A9. apply right now and give up what been
# trained on S30"). The supervisor and the segment's process group are stopped, the segment directory is renamed
# s3a_xKKK_discarded (its weights are not used; its chunks still go to Nextcloud under that name), the shared-memory
# bridges are removed, asc_weights in extra.env is replaced, and a new supervisor starts segment K from last_good.
#   setsid nohup src/s3/redo_seg.sh SUPERVISOR_PID K NEW_ASC_WEIGHTS >> /root/sts2/redo_seg.log 2>&1 < /dev/null &
SUP=$1; K=$2; AW=$3
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a; L=$S/supervisor.log
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a $L; }
N=$(printf %03d $K); R=src/s3/runs/s3a_s$N; X=src/s3/runs/s3a_x${N}_discarded
kill $SUP 2>/dev/null; sleep 2; kill -0 $SUP 2>/dev/null && kill -9 $SUP
G=$(ps -eo pid,args | awk -v n="s3a_s$N" '$2=="/bin/bash" && $3=="src/s3/run_s3.sh" && $4==n {print $1}')
if [ -n "$G" ]; then kill -TERM -- -$G 2>/dev/null; sleep 8; kill -KILL -- -$G 2>/dev/null; sleep 2; fi
left=$(ps -eo args | grep -c "[s]3a_s$N")
rm -f /dev/shm/s3_s3a_s${N}_*
mv $R $X
rm -f $X/carry_*
timeout 300 rclone moveto nextcloud:sts2_backup/s3/s3a_s$N nextcloud:sts2_backup/s3/s3a_x${N}_discarded 2>> $S/rclone.log
mkdir -p $X/archive; mv $X/cdata/*.s2d $X/mdata/*.s3r $X/archive/ 2>/dev/null; touch $X/.done
setsid nohup src/s3/backup_loop.sh $X s3a_x${N}_discarded > /dev/null 2>&1 < /dev/null &
old=$(grep -o 'asc_weights=[0-9.,]*' $S/extra.env)
sed "s/asc_weights=[0-9.,]*/asc_weights=$AW/" $S/extra.env > $S/extra.env.tmp && mv $S/extra.env.tmp $S/extra.env
log "owner: segment $K discarded (renamed $X, $left processes were left after the stop; its weights are not used) and restarted from $(cat $S/last_good); actor ascension weights $old -> asc_weights=$AW (no A5, more A9)"
setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
sleep 5
echo "[$(date -u +%T)] new supervisor started"
