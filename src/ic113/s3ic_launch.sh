#!/bin/bash
# Launches the Ironclad self-play chain s3ic (src/ic113/CHAIN.md) in the chain tree (src/ic113/s3ic_setup.sh).
# usage (server, from anywhere):
#   COMBAT0=/root/exp/ic113/warm/<RUN>/weights.pt /root/exp/ic113/chain/src/ic113/s3ic_launch.sh [HOURS (168)]
# COMBAT0 = the combat warm-up's output (required; src/ic113/WARMUP.md). MACRO0 = the converted s3v3_s005 macro
# (default /root/exp/ic113/w/macro_s3v3_s005_v113.pt), BC0 = the converted S1 ensemble (the macro trainer's --bc file;
# its KL weight is 0 from the first iteration, KL_RUNS=1). The three files are copied to src/s3/runs/s3ic_init/ (no
# carry files there: the chain starts with empty replay windows). Writes src/s3/runs/s3ic/config.env and extra.env
# (only if absent: a second call never changes a running chain's settings), starts the supervisor (setsid, nohup) and
# prints the cron line for s3_resume.sh, which this script does NOT install.
# DRY=1: write the files and print what would be started, start nothing.
HOURS=${1:-168}
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W
NAME=s3ic; S=src/s3/runs/$NAME; I=src/s3/runs/${NAME}_init
: ${COMBAT0:?set COMBAT0 to the combat warm-up output (weights.pt)}
MACRO0=${MACRO0:-/root/exp/ic113/w/macro_s3v3_s005_v113.pt}; BC0=${BC0:-/root/exp/ic113/w/macro_s1_v113.pt}
if ps -eo cmd | grep -q "[s]3_supervise.sh $NAME "; then echo "a supervisor of $NAME is running: nothing done"; exit 1; fi
if [ -f $S/end_time ] && ! grep -q "supervisor end" $S/supervisor.log 2>/dev/null; then
  echo "$S exists with an end time and no 'supervisor end': resume with src/s3/s3_resume.sh (or move $S away)"; exit 1
fi
for f in $COMBAT0 $MACRO0 $BC0; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
mkdir -p $I $S
if [ ! -f $I/weights.pt ]; then
  cp $COMBAT0 $I/weights.pt; cp $MACRO0 $I/macro.pt; cp $BC0 $I/macro_s1_v113.pt
  { date -u +%FT%TZ; for f in $COMBAT0 $MACRO0 $BC0; do echo "$(md5sum < $f | cut -c1-32)  $f"; done; } > $I/sources.txt
fi
ls $I | grep -q carry && { echo "carry files in $I: the chain must start without carries"; exit 1; }
./.venv/bin/python -I -c "
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False); m = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
assert int(c.get('config', {}).get('obs', 4)) == 5, 'combat net is not obs 5'
assert len(m['members']) == 5
print('init files ok: combat obs 5, macro 5 members')" $I/weights.pt $I/macro.pt || exit 1
[ -f $S/config.env ] || cat > $S/config.env <<EOF
# s3ic chain settings (src/ic113/CHAIN.md, written by s3ic_launch.sh $(date -u +%FT%TZ)); the supervisor reads this
# at every (re)start; SEG is re-read every segment. ASC_MIX is overridden every segment by the curriculum (segment.env).
export N='60'
export SEG='7200'
export ASC='10'
export ASC_MIX='0.20'
export CONC='32'
export CSERVERS='2'
export RUNS_PER_ITER='2560'
export KL_RUNS='1'
export PPO_TARGET_KL='0.03'
export COMBAT_WINDOW='6000000'
export EVAL_COUNT='300'
export EVAL_N='20'
export EVAL_SEEDS='Y'
export EVAL_FIRST='100000'
export STOP_INSTANCE='0'
export COMBAT0='$I/weights.pt'
export MACRO0='$I/macro.pt'
export NC_ROOT='nextcloud:sts2_backup/s3ic'
export CHAIN_HOOK='src/ic113/s3ic_hook.sh'
export KEEP_EVERY='4'
export FIGHT_KEEP_EVERY='1'
export PRUNE_KEEP='0'
export ACTOR_MEM_GB='8'
export EVAL_BASE_ARGS='character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20 look_smith=20 look_shop=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001'
export CURRIC_T1_A10='0.05'
export CURRIC_T1_LOW='0.50'
export CURRIC_T2_A10='0.12'
export CURRIC_MIN_RUNS='300'
export VMIX_ROWS='500'
export VMIX_RECENT='20000'
export VMIX_ASC='all'
export VMIX_DEVICE='cuda'
export VMIX_THREADS='16'
EOF
[ -f $S/extra.env ] || cat > $S/extra.env <<EOF
# s3ic per-segment options (read by run_s3.sh at every segment start; src/ic113/CHAIN.md). From the s3v3 chain's
# live extra.env (/root/exp/layout3/src/s3/runs/s3v3/extra.env, 2026-11-17) except: character / era / tf_heal (ic113),
# the potion table belt rule replaced by potion_mix (the file and asc_weights come from the chain hook every segment:
# segment.env SEG_ACTOR_ARGS; do NOT put potion_mix, asc_weights or ASC_MIX here), fake_merchant=1, combat weight decay on
# weight matrices only (--wd-groups matrices, owner 2026-11-18), 60 actors in N (config.env), the --bc
# file of the v0.113 vocabulary.
EXTRA_ACTOR_ARGS="character=ironclad era=6 tf_heal=1 hall_sims=0.5 boss_pfull=1 practice_boss=2 practice_min_act=2 practice_pfull=0.25 candidates=4 look_beta=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001"
COMBAT_WINDOW=6000000
EXTRA_CTRAIN_ARGS="--surprise 0.5 --ema 0.999 --carry-keep 1000000 --lr 5e-5 --wd 0.05 --wd-groups matrices --config vdrop=0.3"
EXTRA_MTRAIN_ARGS="--window-runs 50000 --value-row-share 0.16 --value-steps 150 --value-dropout --value-wd 0.1 --prob-avg --policy-warmup-s 600 --alpha 0.70 --bc $I/macro_s1_v113.pt"
CARRY=1
MPS=1
EXTRA_SERVER_ARGS="--graphs 1 --overlap 1 --overlap-min 48"
CSERVERS=2
OBS=5
EXTRA_MSERVER_ARGS="--mgraphs 1"
EOF
echo "settings: $S/config.env, $S/extra.env"
CRON="*/10 * * * * $W/src/s3/s3_resume.sh >> $W/s3_resume.log 2>&1"
if [ "${DRY:-0}" = "1" ]; then echo "DRY=1: would start: setsid nohup src/s3/s3_supervise.sh $NAME $HOURS > $W/s3_supervise_$NAME.log"; echo "cron line: $CRON"; exit 0; fi
setsid nohup src/s3/s3_supervise.sh $NAME $HOURS > $W/s3_supervise_$NAME.log 2>&1 < /dev/null &
sleep 5
echo "supervisor started (pid $!): $W/s3_supervise_$NAME.log, $S/supervisor.log, $S/status.txt"
echo "install the resume cron line yourself (crontab -e):"
echo "  $CRON"
