#!/bin/bash
# Server tree of the s3ic chain (src/ic113/CHAIN.md): /root/exp/ic113/chain, separate from /root/exp/ic113 so that the
# chain's build never replaces binaries other jobs there run from. Copies /root/exp/ic113/research (sources, data files, the
# patched v0.113.1.1 simulator; no bin / obj / out / runs), overlays the branch's tracked files from UPLOAD (a tar.gz made on
# the laptop: src/{s0,s2,s3,ic113,value_v1,s1} as committed on branch ic113), links the venv and .NET of
# /root/sts2, builds S3 (and with it S2, S0) in the chain tree.
# usage: src/ic113/s3ic_setup.sh UPLOAD.tgz [TREE (/root/exp/ic113/chain)]
#   laptop: git -C .worktrees/ic113 archive --format=tar.gz -o /tmp/x.tgz HEAD src/s0 src/s2 src/s3 \
#           src/ic113 src/value_v1 src/s1
# A rebuild of an existing tree (code update between segments) is the same command; never while the chain's segment runs
# from it unless the build output is known to be unchanged.
UP=$1; T=${2:-/root/exp/ic113/chain}
set -e
mkdir -p $T/research
rsync -a --exclude bin/ --exclude obj/ --exclude /s3/out/ --exclude /s3/runs/ --exclude __pycache__/ /root/exp/ic113/src/ $T/src/
tar -xzf $UP -C $T
ln -sfn /root/sts2/.venv $T/.venv; ln -sfn /root/sts2/dotnet $T/dotnet
cd $T
export PATH=$T/dotnet:$PATH DOTNET_ROOT=$T/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
dotnet build src/s3/S3/S3.csproj -c Release -m:8 > $T/build.log 2>&1 || { tail -n 30 $T/build.log; exit 1; }
grep -E "Warn|Error|Build succeeded" $T/build.log | tail -n 3
chmod +x src/s3/*.sh src/ic113/*.sh
echo "tree $T ready: $(ls -la src/s3/S3/bin/Release/net9.0/S3.dll src/s2/S2/bin/Release/net9.0/S2.dll | awk '{print $6, $7, $8, $9}' | tr '\n' ';')"
