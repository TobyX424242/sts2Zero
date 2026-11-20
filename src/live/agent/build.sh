#!/bin/bash
# Builds LiveAgent (Release) with every bin/obj (its own and those of S3, S2, S0 and the simulator) under
# src/live/agent/artifacts: src/sims is shared with the training chain (symlink to the main checkout) and
# must not receive build outputs from here. Output: src/live/agent/artifacts/bin/LiveAgent/release/LiveAgent.dll
set -e
A=$(cd "$(dirname "$0")" && pwd)
# caches of the main checkout (a worktree shares them): <main>/.cache/nuget, <main>/.cache/dotnet
MAIN=$(dirname "$(git -C "$A" rev-parse --path-format=absolute --git-common-dir)")
export NUGET_PACKAGES=${NUGET_PACKAGES:-$MAIN/.cache/nuget}
export DOTNET_CLI_HOME=${DOTNET_CLI_HOME:-$MAIN/.cache/dotnet}
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
dotnet build "$A/LiveAgent.csproj" -c Release --artifacts-path "$A/artifacts" -nologo -v:q "$@"
echo "built: $A/artifacts/bin/LiveAgent/release/LiveAgent.dll"
