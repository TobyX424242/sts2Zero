#!/usr/bin/env bash
# Builds LiveAgent (Release) from source with every bin/obj (its own and those of S3, S2, S0 and the simulator)
# under src/live/agent/artifacts, so that no build output lands in the shared source folders (src/sims is the
# simulator used by the training chain too). Needs the .NET 9 SDK.
# Output: src/live/agent/artifacts/bin/LiveAgent/release/LiveAgent.dll
# NuGet / .NET caches stay inside the repository (.cache/) unless NUGET_PACKAGES / DOTNET_CLI_HOME are set.
set -euo pipefail
A=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$A/../../.." && pwd)
export NUGET_PACKAGES=${NUGET_PACKAGES:-$ROOT/.cache/nuget}
export DOTNET_CLI_HOME=${DOTNET_CLI_HOME:-$ROOT/.cache/dotnet}
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
dotnet build "$A/LiveAgent.csproj" -c Release --artifacts-path "$A/artifacts" -nologo -v:q "$@"
echo "built: $A/artifacts/bin/LiveAgent/release/LiveAgent.dll"
