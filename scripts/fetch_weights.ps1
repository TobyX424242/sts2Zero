# Downloads the network weights of a GitHub release, checks them against the release's SHA256SUMS and lays them
# out as weights\<name>\{weights.pt, macro.pt, vbar.json, manifest.json} (the folder play_live.ps1 uses).
# Linux: fetch_weights.sh.
#
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_weights.ps1 [-Tag TAG] [-Name NAME] [-Binaries] [-From DIR] [-Repo OWNER/REPO]
#
#   -Tag TAG      release tag (default v0.2.0-ic113-s029)
#   -Name NAME    folder under weights\ (default: from the tag, v0.2.0-ic113-s029 -> ic113_s029)
#   -Binaries     also fetch the prebuilt controller and mod: LiveAgent.zip -> bin\LiveAgent\,
#                 AgentBridge-mod.zip -> bin\AgentBridge\ (then: src\live\install_mod.ps1 prebuilt)
#   -From DIR     take the files from a local folder instead of downloading (same names as the release assets)
#   -Repo R       GitHub repository (default TobyX424242/sts2Zero)
#
# Downloads with the GitHub CLI when it is installed and logged in, else with Invoke-WebRequest.
param(
    [string]$Tag = 'v0.2.0-ic113-s029',
    [string]$Name = '',
    [switch]$Binaries,
    [string]$From = '',
    [string]$Repo = 'TobyX424242/sts2Zero'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'     # the progress bar slows Invoke-WebRequest down a lot in PowerShell 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Root
if (-not $Name) {
    $n = $Tag.TrimStart('v')
    $i = $n.IndexOf('-')
    if ($i -ge 0) { $n = $n.Substring($i + 1) }
    $Name = $n -replace '-', '_'
}
if ($From) { $From = (Resolve-Path -LiteralPath $From).Path }
$Dl = Join-Path $Root "weights\.download\$Tag"
$Dest = Join-Path $Root "weights\$Name"
New-Item -ItemType Directory -Force -Path $Dl, $Dest | Out-Null

$useGh = $false
if (-not $From -and (Get-Command gh -ErrorAction SilentlyContinue)) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & gh auth status *> $null
    $useGh = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $old
}

function Get-Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

function Get-Asset([string]$File) {
    $target = Join-Path $Dl $File
    Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
    if ($From) {
        $src = Join-Path $From $File
        if (-not (Test-Path -LiteralPath $src)) { throw "$src not found" }
        Copy-Item -LiteralPath $src -Destination $target
    } elseif ($useGh) {
        & gh release download $Tag -R $Repo -p $File -D $Dl --clobber
        if ($LASTEXITCODE -ne 0) { throw "gh release download failed for $File" }
    } else {
        $url = "https://github.com/$Repo/releases/download/$Tag/$File"
        Write-Host "[fetch] downloading $url"
        Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing
    }
}

$sums = @{}
function Read-Sums {
    foreach ($line in Get-Content -LiteralPath (Join-Path $Dl 'SHA256SUMS')) {
        if ($line -match '^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$') { $script:sums[$Matches[2]] = $Matches[1].ToLowerInvariant() }
    }
}

function Test-Asset([string]$File, [switch]$Optional) {
    if (-not $sums.ContainsKey($File)) {
        if ($Optional) { Write-Host "[fetch] warning: $File is not listed in SHA256SUMS (not verified)"; return }
        throw "$File is not listed in SHA256SUMS"
    }
    $got = Get-Sha256 (Join-Path $Dl $File)
    if ($got -ne $sums[$File]) {
        Remove-Item -LiteralPath (Join-Path $Dl $File) -Force
        throw "CHECKSUM MISMATCH for ${File}: expected $($sums[$File]), got $got"
    }
    Write-Host "[fetch] ok  $File"
}

function Install-Asset([string]$File, [string]$Target) {
    $t = Join-Path $Dest $Target
    if ((Test-Path -LiteralPath $t) -and (Get-Sha256 $t) -eq $sums[$File]) {
        Write-Host "[fetch] ok  $File (already in weights\$Name\$Target)"; return
    }
    Get-Asset $File
    Test-Asset $File
    Move-Item -LiteralPath (Join-Path $Dl $File) -Destination $t -Force
}

$suffix = if ($From) { " (from $From)" } else { '' }
Write-Host "[fetch] release $Tag of $Repo -> weights\$Name$suffix"
Get-Asset 'SHA256SUMS'
Read-Sums
Get-Asset 'manifest.json'
Test-Asset 'manifest.json' -Optional
$cw = $sums.Keys | Where-Object { $_ -like '*_weights.pt' } | Select-Object -First 1
$mw = $sums.Keys | Where-Object { $_ -like '*_macro.pt' } | Select-Object -First 1
$vb = $sums.Keys | Where-Object { $_ -like '*_vbar.json' } | Select-Object -First 1
if (-not $cw -or -not $mw) { throw "SHA256SUMS of $Tag lists no *_weights.pt / *_macro.pt" }
Install-Asset $cw 'weights.pt'
Install-Asset $mw 'macro.pt'
if ($vb) { Install-Asset $vb 'vbar.json' }
Move-Item -LiteralPath (Join-Path $Dl 'manifest.json') -Destination (Join-Path $Dest 'manifest.json') -Force
Copy-Item -LiteralPath (Join-Path $Dl 'SHA256SUMS') -Destination (Join-Path $Dest 'SHA256SUMS') -Force

if ($Binaries) {
    $bin = Join-Path $Root 'bin'
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    foreach ($z in @('LiveAgent.zip', 'AgentBridge-mod.zip')) { Get-Asset $z; Test-Asset $z -Optional }
    foreach ($d in @('LiveAgent', 'AgentBridge')) {
        $p = Join-Path $bin $d
        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Recurse -Force }
    }
    foreach ($z in @('LiveAgent.zip', 'AgentBridge-mod.zip')) {
        Expand-Archive -LiteralPath (Join-Path $Dl $z) -DestinationPath $bin -Force
        Remove-Item -LiteralPath (Join-Path $Dl $z) -Force
    }
    Write-Host '[fetch] prebuilt controller: bin\LiveAgent\LiveAgent.dll; prebuilt mod: bin\AgentBridge\'
}
Remove-Item -LiteralPath $Dl -Recurse -Force
$dlRoot = Join-Path $Root 'weights\.download'
if (@(Get-ChildItem -LiteralPath $dlRoot -Force).Count -eq 0) { Remove-Item -LiteralPath $dlRoot -Force }
Write-Host "[fetch] done: weights\$Name"
Get-ChildItem -LiteralPath $Dest | Format-Table Name, Length -AutoSize | Out-String | Write-Host
