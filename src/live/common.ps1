# Shared by play_live.ps1 and serve.ps1 (dot-sourced, not run). Linux: common.sh.
# Works in Windows PowerShell 5.1 and PowerShell 7. Sets $LiveDir (this folder) and $RepoRoot.

$LiveDir = $PSScriptRoot
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$DefaultWeights = Join-Path $RepoRoot 'weights\ic113_s029'     # laid out by scripts\fetch_weights.ps1

# Quotes arguments for Start-Process (Windows PowerShell 5.1 joins -ArgumentList items with spaces, unquoted).
function ConvertTo-ArgString([string[]]$Items) {
    ($Items | ForEach-Object {
        if ($_ -match '[\s"]' -or $_ -eq '') { '"' + (($_ -replace '"', '\"') -replace '(\\+)$', '$1$1') + '"' } else { $_ }
    }) -join ' '
}

# The weights of a folder: weights.pt / macro.pt / vbar.json (or vbar_*.json), or the release names
# *_weights.pt / *_macro.pt / *_vbar.json. Returns an object with Dir, Combat, Macro, Vbar (Vbar may be $null).
function Get-LiveWeights([string]$Dir) {
    if (-not $Dir) {
        $Dir = $DefaultWeights
        $legacy = Join-Path $RepoRoot 'src\live\weights\s3ic_s029'
        if (-not (Test-Path -LiteralPath $Dir) -and (Test-Path -LiteralPath $legacy)) { $Dir = $legacy }
    }
    if (-not (Test-Path -LiteralPath $Dir -PathType Container)) {
        throw "weights folder not found: $Dir`n  download the released weights first: scripts\fetch_weights.ps1"
    }
    $Dir = (Resolve-Path -LiteralPath $Dir).Path
    function Pick([string]$Exact, [string[]]$Patterns) {
        $f = Join-Path $Dir $Exact
        if (Test-Path -LiteralPath $f -PathType Leaf) { return $f }
        $found = foreach ($pat in $Patterns) { Get-ChildItem -LiteralPath $Dir -Filter $pat -File -ErrorAction SilentlyContinue }
        $newest = $found | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($newest) { return $newest.FullName }
        return $null
    }
    $w = [pscustomobject]@{
        Dir    = $Dir
        Combat = Pick 'weights.pt' @('*_weights.pt')
        Macro  = Pick 'macro.pt' @('*_macro.pt')
        Vbar   = Pick 'vbar.json' @('vbar_*.json', '*_vbar.json')
    }
    if (-not $w.Combat) { throw "no combat network in $Dir (weights.pt or *_weights.pt)" }
    if (-not $w.Macro) { throw "no macro network in $Dir (macro.pt or *_macro.pt)" }
    return $w
}

# The Python of the environment: $env:PY, else <repo>\.venv\Scripts\python.exe (scripts\setup.ps1 creates it).
function Get-LivePython {
    if ($env:PY) { return $env:PY }
    $py = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $py) { return $py }
    throw "no Python environment (.venv): run scripts\setup.ps1 first, or set `$env:PY to a python.exe"
}

# Path prefix of the bridge files: regular files in the temp folder (Windows has no /dev/shm).
function Get-BridgePrefix {
    return (Join-Path ([System.IO.Path]::GetTempPath()) "sts2zero_live_$($PID)_")
}

# .NET settings shared by the build and the controller (caches inside the repository).
function Set-LiveDotnetEnv {
    $env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'; $env:DOTNET_NOLOGO = '1'; $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = '1'
    if (-not $env:NUGET_PACKAGES) { $env:NUGET_PACKAGES = Join-Path $RepoRoot '.cache\nuget' }
    if (-not $env:DOTNET_CLI_HOME) { $env:DOTNET_CLI_HOME = Join-Path $RepoRoot '.cache\dotnet' }
    # also set by LiveAgent.runtimeconfig.json (no quick JIT for loops): see the comment in agent\LiveAgent.csproj
    $env:DOTNET_TC_OnStackReplacement = '0'
}

# The controller to run: the prebuilt one in <repo>\bin\LiveAgent (scripts\fetch_weights.ps1 -Binaries) when present,
# else built from source (needs the .NET 9 SDK) and rebuilt when a source file is newer.
# $env:LIVEAGENT = 'source' forces the source build, 'prebuilt' refuses to build.
function Get-LiveController([string]$Logs, [string]$Ts) {
    $mode = if ($env:LIVEAGENT) { $env:LIVEAGENT } else { 'auto' }
    $dll = Join-Path $RepoRoot 'bin\LiveAgent\LiveAgent.dll'
    if ($mode -ne 'source' -and (Test-Path -LiteralPath $dll)) { Write-Host "[live] controller: prebuilt $dll"; return $dll }
    if ($mode -eq 'prebuilt') { throw "no prebuilt controller at $dll (scripts\fetch_weights.ps1 -Binaries)" }
    $dll = Join-Path $LiveDir 'agent\artifacts\bin\LiveAgent\release\LiveAgent.dll'
    $stale = -not (Test-Path -LiteralPath $dll)
    if (-not $stale) {
        $built = (Get-Item -LiteralPath $dll).LastWriteTime
        $dirs = @('src\live\agent', 'src\s3\S3', 'src\s2\S2', 'src\s0\S0.Sim', 'src\sims\iRyougi-sts2-sim-v113\src\Sts2Sim.Core') |
            ForEach-Object { Join-Path $RepoRoot $_ }
        $newer = Get-ChildItem -LiteralPath $dirs -Recurse -Filter '*.cs' -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '\\(artifacts|obj|bin)\\' -and $_.LastWriteTime -gt $built } | Select-Object -First 1
        $stale = [bool]$newer
    }
    if ($stale) {
        if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
            throw 'dotnet not found: install the .NET 9 SDK, or use the prebuilt controller (scripts\fetch_weights.ps1 -Binaries)'
        }
        $buildLog = Join-Path $Logs "build_$Ts.txt"
        Write-Host "[live] building LiveAgent from source (log $buildLog)"
        & dotnet build (Join-Path $LiveDir 'agent\LiveAgent.csproj') -c Release --artifacts-path (Join-Path $LiveDir 'agent\artifacts') -nologo -v:q |
            Out-File -FilePath $buildLog -Encoding utf8
        if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $buildLog -Tail 30 | Write-Host; throw 'LiveAgent build failed' }
    }
    Write-Host "[live] controller: $dll"
    return $dll
}

# Starts the combat and the macro inference server (src\s3\py\s3_server.py) for one actor and waits until both are
# ready. Returns an object with Processes, Obs (the combat network's observation layout) and the log files.
# env: DEVICE (cuda or cpu; default: cuda when available), CAPACITY (16), CPU_THREADS (torch threads per server on the
#      CPU, default 4: the networks run small batches, which are slower with many threads)
function Start-LiveServers($Weights, [string]$Py, [string]$Prefix, [string]$Logs, [string]$Ts) {
    # the observation layout of the combat checkpoint (config "obs"), as src/s3/eval_s3.sh reads it
    $code = "import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('config',{});print(int(c.get('obs',4)))"
    $obs = 5
    $out = & $Py -I -c $code $Weights.Combat
    if ($LASTEXITCODE -eq 0 -and "$out" -match '^\s*(\d+)\s*$') { $obs = [int]$Matches[1] }
    $device = $env:DEVICE
    if (-not $device) {
        $device = & $Py -I -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')"
        if ($LASTEXITCODE -ne 0 -or "$device" -notmatch '^(cuda|cpu)$') { $device = $null }
    }
    $dev = @()
    if ($device) { $dev = @('--device', "$device") }
    if ("$device" -eq 'cpu') { $dev += @('--threads', $(if ($env:CPU_THREADS) { $env:CPU_THREADS } else { '4' })) }
    $cap = if ($env:CAPACITY) { $env:CAPACITY } else { '16' }
    $server = Join-Path $RepoRoot 'src\s3\py\s3_server.py'
    $common = @('--prefix', "$($Prefix)c", '--mprefix', "$($Prefix)m", '--actors', '1', '--capacity', $cap,
                '--weights', $Weights.Combat, '--macro', $Weights.Macro) + $dev + @('--idle-exit', '1e9')
    # one actor: combat and macro servers as in src/s3/eval_s3.sh / src/ic113/eval_ic113.sh, without the idle exit
    $argC = @('-I', $server, '--serve', 'combat', '--first', '0') + $common
    $argM = @('-I', $server, '--serve', 'macro', '--min-period', '0.02', '--mlayout', '3') + $common
    $s = [pscustomobject]@{
        Processes = @(); Obs = $obs
        Logs = @((Join-Path $Logs "server_combat_$Ts.txt"), (Join-Path $Logs "server_macro_$Ts.txt"))
        ErrLogs = @((Join-Path $Logs "server_combat_$Ts.err.txt"), (Join-Path $Logs "server_macro_$Ts.err.txt"))
    }
    $s.Processes += Start-Process -FilePath $Py -ArgumentList (ConvertTo-ArgString $argC) -WorkingDirectory $RepoRoot `
        -NoNewWindow -PassThru -RedirectStandardOutput $s.Logs[0] -RedirectStandardError $s.ErrLogs[0]
    $s.Processes += Start-Process -FilePath $Py -ArgumentList (ConvertTo-ArgString $argM) -WorkingDirectory $RepoRoot `
        -NoNewWindow -PassThru -RedirectStandardOutput $s.Logs[1] -RedirectStandardError $s.ErrLogs[1]
    $dshow = if ($device) { $device } else { 'auto' }
    Write-Host "[live] inference servers starting (weights $($Weights.Dir), device $dshow, obs $obs); logs $Logs\server_*_$Ts*.txt"
    $ready = 0
    for ($i = 0; $i -lt 600; $i++) {
        $ready = @(Select-String -LiteralPath $s.Logs -SimpleMatch 's3 server ready' -ErrorAction SilentlyContinue).Count
        if ($ready -ge 2) { break }
        foreach ($proc in $s.Processes) {
            if ($proc.HasExited) {
                Write-Host '[live] an inference server died:'
                foreach ($f in $s.Logs + $s.ErrLogs) { if (Test-Path -LiteralPath $f) { Get-Content -LiteralPath $f -Tail 20 | Write-Host } }
                throw 'inference server failed to start'
            }
        }
        Start-Sleep -Milliseconds 500
    }
    if ($ready -lt 2) { throw 'inference servers not ready after 300 s' }
    Select-String -LiteralPath $s.Logs -SimpleMatch 's3 server ready' | ForEach-Object { Write-Host "[live] $($_.Line)" }
    return $s
}

# Stops the servers and removes the bridge files.
function Stop-LiveServers($Servers, [string]$Prefix) {
    if ($Servers) {
        foreach ($proc in $Servers.Processes) {
            if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
        }
        Start-Sleep -Milliseconds 500
    }
    if ($Prefix) { Remove-Item -Path "$($Prefix)*" -Force -ErrorAction SilentlyContinue }
}
