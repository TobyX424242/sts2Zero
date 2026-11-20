# Starts only the two inference servers (combat + macro, src\s3\py\s3_server.py) and keeps them running until Ctrl-C.
# play_live.ps1 does this itself; use serve.ps1 to keep the networks loaded while you start and stop the controller
# by hand. Linux: serve.sh.
#
# usage: powershell -ExecutionPolicy Bypass -File src\live\serve.ps1 [WEIGHTS_DIR]     (default weights\ic113_s029)
# env:   $env:DEVICE=cuda|cpu (default: cuda when available), $env:CAPACITY (16), $env:CPU_THREADS (4, CPU only), $env:PY
# Prints the controller keys to pass on, e.g.
#   dotnet bin\LiveAgent\LiveAgent.dll play bridge=...c0 mbridge=...m0 obs=5 potion_mix=...
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location -LiteralPath $RepoRoot

$logs = Join-Path $RepoRoot 'src\live\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$servers = $null
$prefix = $null
try {
    $w = Get-LiveWeights $(if ($args.Count -gt 0) { "$($args[0])" } else { $null })
    $py = Get-LivePython
    $prefix = Get-BridgePrefix
    $servers = Start-LiveServers $w $py $prefix $logs $ts
    $keys = @("bridge=$($prefix)c0", "mbridge=$($prefix)m0", "obs=$($servers.Obs)")
    if ($w.Vbar) { $keys += "potion_mix=$($w.Vbar)" }
    Write-Host "[live] servers ready. Controller keys (run from $RepoRoot; quote any key that contains a space):"
    Write-Host ('  ' + ($keys -join ' '))
    Write-Host '[live] Ctrl-C stops the servers'
    while ($true) {
        foreach ($proc in $servers.Processes) {
            if ($proc.HasExited) {
                Write-Host '[live] an inference server stopped:'
                foreach ($f in $servers.Logs + $servers.ErrLogs) { if (Test-Path -LiteralPath $f) { Get-Content -LiteralPath $f -Tail 20 | Write-Host } }
                exit 1
            }
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Stop-LiveServers $servers $prefix
}
