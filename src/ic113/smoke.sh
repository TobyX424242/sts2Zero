#!/bin/bash
# ic113 CPU smoke (no GPU): the v0.111 v3 networks converted to the v0.113 vocabularies (src/ic113/convert_v113.py)
# served on CPU by s3_server.py; four actors with a small search budget: Ironclad A10 self-play (writes combat and macro
# chunks), Ironclad A10 evaluation, Defect A10 evaluation, Ironclad A0 evaluation (reaches later acts with these
# networks); deployment rules as in evaluation (map argmax, look_beta 20). Then the chunks are read back.
# usage: src/ic113/smoke.sh [RUNS per actor (3)] [SIMS (8)]   output: out/smoke
L=/root/exp/ic113; O=$L/out/smoke; W=$L/w; R=${1:-3}; SIMS=${2:-8}
rm -rf $O; mkdir -p $O/mdata $O/cdata; cd $L
export PATH=/root/sts2/dotnet:$PATH DOTNET_ROOT=/root/sts2/dotnet CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 DOTNET_CLI_TELEMETRY_OPTOUT=1
P=/dev/shm/ic113smoke_$$_
X=src/s3/S3/bin/Release/net9.0/S3.dll
taskset -c 0-11 nice -n 10 ./.venv/bin/python -I src/s3/py/s3_server.py --serve both --prefix ${P}c --mprefix ${P}m --actors 4 \
  --capacity 8 --weights $W/combat_s3v3_s005_v113.pt --macro $W/macro_s3v3_s005_v113.pt --device cpu --dtype fp32 --idle-exit 120 > $O/server.txt 2>&1 &
sleep 20
B="greedy_map=1 look_beta=20 obs=5"
nice -n 10 dotnet $X play mode=selfplay character=ironclad bridge=${P}c0 mbridge=${P}m0 runs=$R concurrent=$R tag=ic_sp \
  seed=7 ascension=10 sims_full=$SIMS sims_fast=4 pfull=0.25 perturb=0.5 practice_boss=1 practice_min_act=2 \
  data=$O/cdata mdata=$O/mdata mchunk=2 chunk=256 out=$O/ic_sp.jsonl fights=$O/ic_sp_f.jsonl shutdown=1 $B > $O/ic_sp.txt 2>&1 &
nice -n 10 dotnet $X play mode=eval character=ironclad bridge=${P}c1 mbridge=${P}m1 count=$R concurrent=$R tag=ic_ev \
  ascension=10 seeds=XIC first=0 sims_full=$SIMS out=$O/ic_ev.jsonl fights=$O/ic_ev_f.jsonl shutdown=1 $B > $O/ic_ev.txt 2>&1 &
nice -n 10 dotnet $X play mode=eval character=defect bridge=${P}c2 mbridge=${P}m2 count=$R concurrent=$R tag=de_ev \
  ascension=10 seeds=XDE first=0 sims_full=$SIMS out=$O/de_ev.jsonl fights=$O/de_ev_f.jsonl shutdown=1 $B > $O/de_ev.txt 2>&1 &
nice -n 10 dotnet $X play mode=eval character=ironclad bridge=${P}c3 mbridge=${P}m3 count=$R concurrent=$R tag=ic_a0 \
  ascension=0 seeds=XIC0 first=0 sims_full=$SIMS out=$O/ic_a0.jsonl fights=$O/ic_a0_f.jsonl shutdown=1 $B > $O/ic_a0.txt 2>&1 &
wait
grep -h "\] mode\|errors\|macro vocabulary\|T_f HP-axis\|heal_check" $O/ic_sp.txt $O/ic_ev.txt $O/de_ev.txt $O/ic_a0.txt | cut -c1-400
./.venv/bin/python -I - $O <<'PY'
import glob, json, os, sys
sys.path.insert(0, "src/s3/py"); sys.path.insert(0, "src/s2/py"); sys.path.insert(0, "src/s0/py"); sys.dont_write_bytecode = True
import numpy as np, torch
import macro_s3 as ms, train_macro_s3 as tr
O = sys.argv[1]; fails = []
runs = [json.loads(l) for f in ("ic_sp.jsonl", "ic_ev.jsonl", "de_ev.jsonl", "ic_a0.jsonl") for l in open(os.path.join(O, f))]
err = [r for r in runs if r.get("error")]
by = {}
for r in runs: by.setdefault((r["actor"], r["character"], r["build_id"]), []).append(r["floors"])
print(f"runs {len(runs)} (won {sum(r['won'] for r in runs)}, errors {len(err)}); floors by actor {by}")
if err or not runs: fails.append(f"run errors {[r.get('error') for r in err]}")
I, Fl = [], []
for p in sorted(glob.glob(os.path.join(O, "mdata", "*.s3r"))):
    _, i, f, m = tr.read_chunk(p); I.append(i); Fl.append(f)
if not I: fails.append("no macro chunks")
else:
    I, Fl = np.concatenate(I), np.concatenate(Fl)
    ctx = I[:, ms.OFF["ctx"][0]:ms.OFF["ctx"][1]]
    print(f"macro chunk rows {len(I)}: character tokens {sorted(set(ctx[:, 0].tolist()))}, era tokens {sorted(set(ctx[:, 2].tolist()))}, "
          f"largest card id {int(I[:, ms.OFF['d_ids'][0]:ms.OFF['d_ids'][1]].max())}")
    if set(ctx[:, 0].tolist()) != {0} or set(ctx[:, 2].tolist()) != {ms.ERA_V0113}: fails.append("ctx tokens")
    ens = ms.Ensemble.load(os.path.join("w", "macro_s3v3_s005_v113.pt"), "cpu")
    out = ens.serve(I[:256], Fl[:256], "cpu")
    if not torch.isfinite(out).all(): fails.append("non-finite macro outputs")
cd = sorted(glob.glob(os.path.join(O, "cdata", "*.s2d")))
print(f"combat chunks {len(cd)}")
if not cd: fails.append("no combat chunks")
print("SMOKE " + ("PASSED" if not fails else "FAILED: " + "; ".join(fails)))
PY
