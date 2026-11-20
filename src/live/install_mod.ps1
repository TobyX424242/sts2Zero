# Installs the AgentBridge mod into the game's mods folder (Windows). Linux: install_mod.sh.
#
#   powershell -ExecutionPolicy Bypass -File src\live\install_mod.ps1 [install|prebuilt|build|uninstall|status]
#
#   install     build from source (.NET 9 SDK) + install (default)
#   prebuilt    install the prebuilt mod from bin\AgentBridge (scripts\fetch_weights.ps1 -Binaries)
#   build       build only (output: src\live\mod\bin\Release\AgentBridge.dll)
#   uninstall   remove <game>\mods\AgentBridge
#   status      show what is installed
#
# Game folder: $env:STS2_GAME_DIR, else the first "Slay the Spire 2" found in Steam's libraries (Steam's folder from
# the registry and every library listed in its libraryfolders.vdf), else
# C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2.
# NuGet / .NET caches stay inside the repository (.cache\).
param([ValidateSet('install', 'prebuilt', 'build', 'uninstall', 'status')][string]$Command = 'install')
$ErrorActionPreference = 'Stop'

$Here = $PSScriptRoot
$Root = (Resolve-Path -LiteralPath (Join-Path $Here '..\..')).Path
$ModSrc = Join-Path $Here 'mod'
$ModId = 'AgentBridge'
$Prebuilt = Join-Path $Root "bin\$ModId"
$DefaultGame = 'C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2'

if (-not $env:NUGET_PACKAGES) { $env:NUGET_PACKAGES = Join-Path $Root '.cache\nuget' }
if (-not $env:DOTNET_CLI_HOME) { $env:DOTNET_CLI_HOME = Join-Path $Root '.cache\dotnet' }
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'; $env:DOTNET_NOLOGO = '1'; $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = '1'

function Find-Game {
    if ($env:STS2_GAME_DIR) { return $env:STS2_GAME_DIR.TrimEnd('\', '/') }
    $steams = @()
    try {
        $sp = (Get-ItemProperty -Path 'HKCU:\Software\Valve\Steam' -Name SteamPath -ErrorAction Stop).SteamPath
        if ($sp) { $steams += ($sp -replace '/', '\') }
    } catch { }
    if (${env:ProgramFiles(x86)}) { $steams += (Join-Path ${env:ProgramFiles(x86)} 'Steam') }
    $libs = @()
    foreach ($s in $steams) {
        $libs += $s
        $vdf = Join-Path $s 'steamapps\libraryfolders.vdf'
        if (Test-Path -LiteralPath $vdf) {
            foreach ($line in Get-Content -LiteralPath $vdf) {
                if ($line -match '^\s*"path"\s+"(.+)"\s*$') { $libs += ($Matches[1] -replace '\\\\', '\') }
            }
        }
    }
    foreach ($lib in $libs) {
        $g = Join-Path $lib 'steamapps\common\Slay the Spire 2'
        if (Test-Path -LiteralPath $g -PathType Container) { return $g }
    }
    return $DefaultGame
}

$GameDir = Find-Game
$Dest = Join-Path $GameDir "mods\$ModId"

function Assert-Game {
    if (-not (Test-Path -LiteralPath $GameDir -PathType Container)) {
        throw "game folder not found: $GameDir`nset it first, e.g.  `$env:STS2_GAME_DIR = 'D:\SteamLibrary\steamapps\common\Slay the Spire 2'"
    }
}

function Invoke-Build {
    Assert-Game
    $dataDir = $null
    foreach ($cand in @('data_sts2_windows_x86_64', 'data_sts2_linuxbsd_x86_64')) {
        $d = Join-Path $GameDir $cand
        if (Test-Path -LiteralPath (Join-Path $d 'sts2.dll')) { $dataDir = $d; break }
    }
    if (-not $dataDir) { throw "game assemblies (sts2.dll) not found under: $GameDir (set `$env:STS2_GAME_DIR)" }
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        throw "dotnet not found: install the .NET 9 SDK, or use: install_mod.ps1 prebuilt"
    }
    Write-Host "== building $ModId against: $dataDir"
    & dotnet build (Join-Path $ModSrc 'AgentBridge.csproj') -c Release "-p:STS2GameDir=$GameDir" "-p:STS2DataDir=$dataDir" -nologo -v minimal
    if ($LASTEXITCODE -ne 0) { throw 'build failed' }
    $dll = Join-Path $ModSrc "bin\Release\$ModId.dll"
    if (-not (Test-Path -LiteralPath $dll)) { throw 'build output missing' }
}

function Copy-Mod([string]$Dll, [string]$Json) {
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    Copy-Item -LiteralPath $Dll -Destination (Join-Path $Dest "$ModId.dll") -Force
    Copy-Item -LiteralPath $Json -Destination (Join-Path $Dest "$ModId.json") -Force
    Write-Host "== installed into ${Dest}:"
    Get-ChildItem -LiteralPath $Dest | Format-Table Name, Length, LastWriteTime -AutoSize | Out-String | Write-Host
    Write-Host @'
Next (first time only): start the game. A popup asks whether to load mods: choose "Load mods" (the game
then closes itself; start it again). If no popup appears, open the Mods screen from the main menu and enable
mod loading there. Mod saves live in a separate "modded" profile folder (the game copies your normal saves
there the first time), so agent runs do not touch your unmodded progress.
Then switch "Agent mode" on in the small panel at the top right and start src\live\play_live.ps1.
'@
}

switch ($Command) {
    'install' {
        Invoke-Build
        Copy-Mod (Join-Path $ModSrc "bin\Release\$ModId.dll") (Join-Path $ModSrc "$ModId.json")
    }
    'prebuilt' {
        Assert-Game
        $dll = Join-Path $Prebuilt "$ModId.dll"; $json = Join-Path $Prebuilt "$ModId.json"
        if (-not ((Test-Path -LiteralPath $dll) -and (Test-Path -LiteralPath $json))) {
            throw "no prebuilt mod in ${Prebuilt}: run scripts\fetch_weights.ps1 -Binaries (or unzip AgentBridge-mod.zip into bin\)"
        }
        Copy-Mod $dll $json
    }
    'build' { Invoke-Build }
    'uninstall' {
        if (Test-Path -LiteralPath $Dest) {
            Remove-Item -LiteralPath (Join-Path $Dest "$ModId.dll"), (Join-Path $Dest "$ModId.json") -Force -ErrorAction SilentlyContinue
            if (@(Get-ChildItem -LiteralPath $Dest -Force).Count -eq 0) { Remove-Item -LiteralPath $Dest -Force }
            else { Write-Host "note: $Dest not empty, left in place" }
            $mods = Join-Path $GameDir 'mods'
            if ((Test-Path -LiteralPath $mods) -and @(Get-ChildItem -LiteralPath $mods -Force).Count -eq 0) { Remove-Item -LiteralPath $mods -Force }
            Write-Host "== removed $Dest"
        } else {
            Write-Host "== not installed ($Dest does not exist)"
        }
    }
    'status' {
        Write-Host "game folder: $GameDir"
        if (Test-Path -LiteralPath $Dest) { Get-ChildItem -LiteralPath $Dest | Format-Table Name, Length, LastWriteTime -AutoSize | Out-String | Write-Host }
        else { Write-Host "not installed ($Dest)" }
    }
}
