#!/bin/bash
# Puts the new design (IMPROVEMENT_PLAN.md, tests of 2026-11-13) into chain s3a while it is paused: extra.env from
# src/s3/new_design/extra.env, and in config.env 2-hour segments, PPO every 2,560 runs with target KL 0.03
# (E1, scaled with the 2,400 runs in flight) and 300 evaluation seeds. Old files are kept as *.pre_new_design.
# Then resume: setsid nohup src/s3/s3_supervise.sh s3a 0 >> /root/sts2/s3_supervise_s3a.log 2>&1 < /dev/null &
# Undo: mv src/s3/runs/s3a/extra.env.pre_new_design src/s3/runs/s3a/extra.env (same for config.env).
cd "$(dirname "$0")/../.."
S=src/s3/runs/s3a
pgrep -f "^/bin/bash src/s3/s3_supervise.sh s3a" > /dev/null && { echo "supervisor is running: pause it first"; exit 1; }
[ -f $S/extra.env.pre_new_design ] || cp $S/extra.env $S/extra.env.pre_new_design
[ -f $S/config.env.pre_new_design ] || cp $S/config.env $S/config.env.pre_new_design
cp src/s3/new_design/extra.env $S/extra.env.tmp && mv $S/extra.env.tmp $S/extra.env
cp $S/config.env $S/config.env.tmp
setv() { grep -q "^export $1=" $S/config.env.tmp && sed -i "s|^export $1=.*|export $1='$2'|" $S/config.env.tmp || echo "export $1='$2'" >> $S/config.env.tmp; }
setv SEG 7200; setv RUNS_PER_ITER 2560; setv PPO_TARGET_KL 0.03; setv EVAL_COUNT 300
mv $S/config.env.tmp $S/config.env
echo "[$(date -u +%F\ %T)] new design applied (IMPROVEMENT_PLAN.md; PROGRESS.md 2026-11-13): candidates=4, combat lr 5e-5 + window 6M, macro value fixes (window 50k x 16 rows, dropout, weight decay, prob-avg), policy warm-up 600 s, PPO every 2560 runs / target KL 0.03, 2-hour segments, 300 eval seeds" >> $S/supervisor.log
echo "--- extra.env"; cat $S/extra.env; echo "--- config.env"; cat $S/config.env; echo "end_time: $(date -u -d @$(cat $S/end_time))"
