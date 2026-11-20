#!/bin/bash
# ic113 (2026-11-18): the simulator is iRyougi tag v0.113.1.1 + src/sims/patches/iryougi-sts2-sim-v113.patch in src/sims/iRyougi-sts2-sim-v113 (the db844ad notes below are v0.111 history)
# S3 server setup for a fresh Vast.ai PyTorch box (first used on the Threadripper 9980X + RTX PRO 6000 box, 2026-11-13).
# Laptop side (project root):
#   tar -czf s3tr_code.tgz -T filelist   # src/{s0,s2,s3} sources + scripts, value_v1/v2 *.py, pipeline/sample_runs.py,
#                                        # the four iRyougi patches, s0/data/combat_vocab_v1.json, s3/data/{macro_vocab,boss2_w2}.json
#   rclone copyto FILE streamR2:video/NAME --bwlimit 9M; rclone link streamR2:video/NAME   (weights.pt, macro.pt, macro_s1.pt)
#   links.env: WEIGHTS_URL/_SHA, MACRO_URL/_SHA, MACRO_S1_URL/_SHA, WDIR (where weights.pt + macro.pt go)
#   ssh ... 'mkdir -p /root/sts2 && tar -xzf - -C /root/sts2' < s3tr_code.tgz; scp links.env to /root/sts2/
# Server:
#   cd /root/sts2 && setsid nohup bash src/s3/server_setup_tr.sh > setup.log 2>&1 < /dev/null &
# Installs: venv on the image's torch (--system-site-packages) + orjson, .NET SDK 9.0.118, iRyougi at db844ad + local +
# S0 + S3 + perf patches (perf: src/s3/iryougi_perf.patch, behaviour-identical speed-up of hook dispatch, 2026-11-13), weights from links.env (sha256 checked), S3 Release build,
# CUDA check (sm_120). No game files, no Nextcloud config.
set -e
W=/root/sts2; cd $W; mkdir -p .cache/tmp .cache/pip src/sims
export DOTNET_CLI_HOME=$W/.cache/dotnet NUGET_PACKAGES=$W/.cache/nuget DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$W/.cache/tmp PIP_CACHE_DIR=$W/.cache/pip
T0=$(date +%s)
echo "[$(date +%T)] weights (background)"
(
  set -e; . $W/links.env; mkdir -p $WDIR src/s3/data
  curl -sSfL -o $WDIR/weights.pt "$WEIGHTS_URL"; curl -sSfL -o $WDIR/macro.pt "$MACRO_URL"
  curl -sSfL -o src/s3/data/macro_s1.pt "$MACRO_S1_URL"
  echo "$WEIGHTS_SHA  $WDIR/weights.pt" | sha256sum -c; echo "$MACRO_SHA  $WDIR/macro.pt" | sha256sum -c
  echo "$MACRO_S1_SHA  src/s3/data/macro_s1.pt" | sha256sum -c
) > $W/weights_dl.log 2>&1 &
DL=$!
echo "[$(date +%T)] venv"
[ -d .venv ] || /venv/main/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install -q orjson==3.13.0
.venv/bin/python -c "
import torch, numpy, orjson
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list(), numpy.__version__, orjson.__version__)
x = torch.randn(512, 512, device='cuda'); print('cuda matmul ok', float((x @ x).float().abs().sum()) > 0)"
echo "[$(date +%T)] dotnet"
[ -x dotnet/dotnet ] || { curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh && bash dotnet-install.sh --version 9.0.118 --install-dir $W/dotnet > dotnet_install.log 2>&1; }
export PATH=$W/dotnet:$PATH DOTNET_ROOT=$W/dotnet
dotnet --version
echo "[$(date +%T)] iRyougi"
if [ ! -d src/sims/iRyougi-sts2-sim-v113 ]; then
  git clone -q https://github.com/iRyougi/sts2-sim.git src/sims/iRyougi-sts2-sim-v113
  (cd src/sims/iRyougi-sts2-sim-v113 && git checkout -q v0.113.1.1 \
    && git apply $W/src/sims/patches/iryougi-sts2-sim-v113.patch \
    && echo PATCHED && git status --short)
fi
echo "[$(date +%T)] build"
dotnet build src/s3/S3/S3.csproj -c Release -m:32 2>&1 | grep -E " error |Build succeeded" | sort -u
[ -f src/s3/S3/bin/Release/net9.0/S3.dll ] || { echo "BUILD FAILED"; exit 1; }
wait $DL || { echo "WEIGHTS DOWNLOAD FAILED"; cat $W/weights_dl.log; exit 1; }
cat $W/weights_dl.log
echo "[$(date +%T)] SETUP_DONE in $(( $(date +%s) - T0 )) s"
