# Live play on Windows: the agent plays the real game through the AgentBridge mod (doc/live-play.md).
# Linux: play_live.sh (same arguments).
#
# usage: powershell -ExecutionPolicy Bypass -File src\live\play_live.ps1 [WEIGHTS_DIR] [key=value ...]
#   WEIGHTS_DIR (default weights\ic113_s029, from scripts\fetch_weights.ps1): the combat network (weights.pt or
#     *_weights.pt), the macro network (macro.pt or *_macro.pt) and the potion_mix table (vbar.json, vbar_*.json or
#     *_vbar.json; potion_mix=FILE overrides)
#   key=value: passed to `LiveAgent play` (src\live\agent\Program.cs), e.g.
#     asc=7            start a fresh Ironclad run at A7 when the game is at the main menu (omit: play the run in progress
#                      or the one started from the in-game panel); seed=ABC; loop=1 (next run after a game over)
#     dry=1            decide and log only (no actions sent); delay_ms=600 (mod's delay before each action)
#     any S3 agent key (defaults = the s3ic chain evaluation: greedy, 64 simulations per move, ...)
#     Quote values that contain a comma: 'key=a,b'.
# env: $env:UNIFORM=1        no inference servers, uniform networks (wiring test: the agent plays at random but legally)
#      $env:DEVICE=cuda|cpu  device of the inference servers (default: cuda when available); $env:CAPACITY (16);
#                            $env:CPU_THREADS (torch threads per server on the CPU, default 4)
#      $env:PY               Python to use (default: <repo>\.venv\Scripts\python.exe, created by scripts\setup.ps1)
#      $env:LIVEAGENT=source|prebuilt  which controller to run (default: bin\LiveAgent if present, else built from source)
# Logs: src\live\logs\ (controller JSONL, server outputs). Ctrl-C stops the controller and the servers.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location -LiteralPath $RepoRoot

$rest = @($args)
$weightsArg = $null
if ($rest.Count -gt 0 -and "$($rest[0])" -notlike '*=*') {
    $weightsArg = "$($rest[0])"
    $rest = if ($rest.Count -gt 1) { @($rest[1..($rest.Count - 1)]) } else { @() }
}
$rest = @($rest | ForEach-Object { "$_" })
$logs = Join-Path $RepoRoot 'src\live\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
Set-LiveDotnetEnv
$dll = Get-LiveController $logs $ts

$agentArgs = @()
$servers = $null
$prefix = $null
$writer = $null
$code = 1
try {
    if ($env:UNIFORM -eq '1') {
        Write-Host '[live] UNIFORM=1: no inference servers, uniform networks'
        $agentArgs += @('bridge=uniform', 'mbridge=uniform')
    } else {
        $w = Get-LiveWeights $weightsArg
        if ($w.Vbar) { $agentArgs += "potion_mix=$($w.Vbar)" }
        $py = Get-LivePython
        $prefix = Get-BridgePrefix
        $servers = Start-LiveServers $w $py $prefix $logs $ts
        Write-Host '[live] servers ready'
        $agentArgs += @("bridge=$($prefix)c0", "mbridge=$($prefix)m0", "obs=$($servers.Obs)")
    }
    $ctlLog = Join-Path $logs "controller_$ts.txt"
    $writer = New-Object System.IO.StreamWriter($ctlLog, $true, (New-Object System.Text.UTF8Encoding($false)))
    & dotnet $dll play @agentArgs "log=$logs" @rest | ForEach-Object { Write-Host $_; $writer.WriteLine($_); $writer.Flush() }
    $code = $LASTEXITCODE
} finally {
    if ($writer) { $writer.Dispose() }
    Stop-LiveServers $servers $prefix
}
exit $code
