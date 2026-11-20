# Sets up the Python environment of sts2Zero (Windows). Linux: setup.sh.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# 1. Finds uv on PATH, or installs it into .tools\uv with the official installer in its "unmanaged" mode: no change
#    to your PATH or profile, nothing written outside that folder ($env:UV_INSTALL_DIR overrides it).
# 2. Runs `uv sync --locked`: creates .venv in the repository root with Python 3.12, torch (CUDA 12.8 build, about
#    3 GB to download), numpy and orjson as pinned in uv.lock.
# 3. Checks for the .NET 9 SDK / runtime (needed for the live-play controller and the game mod).
#
# Unless you set them yourself, uv's download cache and any Python it downloads stay inside the repository
# (.cache\uv, .tools\python), so deleting the folder removes everything.
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Root
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $Root '.cache\uv' }
if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root '.tools\python' }

# ---- uv
$uv = $null
$cmd = Get-Command uv -ErrorAction SilentlyContinue
if ($cmd) { $uv = $cmd.Source }
else {
    $dir = if ($env:UV_INSTALL_DIR) { $env:UV_INSTALL_DIR } else { Join-Path $Root '.tools\uv' }
    $cands = @((Join-Path $dir 'uv.exe'), (Join-Path $dir 'bin\uv.exe'))
    $uv = $cands | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $uv) {
        Write-Host "[setup] uv not found: installing it into $dir"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        # the official installer, run in a child PowerShell; UV_UNMANAGED_INSTALL = install into $dir only
        $env:UV_UNMANAGED_INSTALL = $dir
        try {
            $ps = (Get-Process -Id $PID).Path
            & $ps -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
        } finally {
            Remove-Item Env:UV_UNMANAGED_INSTALL -ErrorAction SilentlyContinue
        }
        $uv = $cands | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if (-not $uv) { throw "the uv installer finished but no uv.exe is in $dir (see https://docs.astral.sh/uv/)" }
    }
}
Write-Host "[setup] uv: $uv ($(& $uv --version))"

# ---- Python environment
Write-Host '[setup] creating .venv (uv sync --locked)'
& $uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
$py = Join-Path $Root '.venv\Scripts\python.exe'
& $py -c "import sys, torch, numpy, orjson; print('[setup] Python', sys.version.split()[0] + ', torch', torch.__version__ + ', numpy', numpy.__version__ + ', orjson', orjson.__version__)"
if ($env:SETUP_SKIP_GPU_CHECK -ne '1') {
    & $py -c "import torch; ok = torch.cuda.is_available(); print('[setup] CUDA GPU: ' + (torch.cuda.get_device_name(0) if ok else 'none found (the inference servers will run on the CPU: slower)'))"
}

# ---- .NET 9
if (Get-Command dotnet -ErrorAction SilentlyContinue) {
    $sdk = @(& dotnet --list-sdks | Where-Object { $_ -match '^9\.' }) | Select-Object -Last 1
    $rt = @(& dotnet --list-runtimes | Where-Object { $_ -match '^Microsoft\.NETCore\.App 9\.' }) | Select-Object -Last 1
    if ($sdk) { Write-Host "[setup] .NET SDK: $(($sdk -split ' ')[0]) (builds the mod and the controller from source)" }
    elseif ($rt) { Write-Host "[setup] .NET runtime: $rt (no SDK: use the prebuilt binaries, scripts\fetch_weights.ps1 -Binaries)" }
    else { Write-Host '[setup] dotnet found, but no .NET 9 runtime or SDK: install the .NET 9 SDK (https://dotnet.microsoft.com/download/dotnet/9.0)' }
} else {
    Write-Host '[setup] .NET not found. Install the .NET 9 SDK (to build from source) or at least the .NET 9 runtime'
    Write-Host '        (to run the prebuilt controller): https://dotnet.microsoft.com/download/dotnet/9.0'
    Write-Host '        or: winget install Microsoft.DotNet.SDK.9'
}
Write-Host '[setup] done. Next: scripts\fetch_weights.ps1'
