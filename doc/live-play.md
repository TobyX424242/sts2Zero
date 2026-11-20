# Watch the agent play Slay the Spire 2

This guide sets up the trained Ironclad agent so that it plays the real game on your computer. You watch it on
screen. You can pause it at any time and take over.

Three pieces work together:

- **AgentBridge**, a small game mod. It reads the game state and clicks for the agent, through the same code
  paths the game's own buttons use. It adds a small panel to the top right of the screen.
- **LiveAgent**, the controller. A .NET program that talks to the mod, rebuilds the situation in our simulator,
  and asks the agent what to do.
- **Two inference servers** (Python, PyTorch). They run the agent's two neural networks: one for fights, one for
  everything else (map, rewards, events, shops, rest sites).

The scripts in this repository start the servers and the controller for you.

## What you need

| | |
|---|---|
| Game | Slay the Spire 2 **v0.113.1** on Steam (Windows or Linux). The mod and the agent are built for exactly this version. |
| OS | Linux (tested) or Windows 10/11 (not tested yet, see [Status](#status)). |
| GPU | An NVIDIA GPU is recommended (RTX 20xx to RTX 50xx, driver 570 or newer). The agent also runs on the CPU (see below). |
| .NET 9 | The [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0) to build the mod and the controller from source. With the prebuilt binaries the .NET 9 runtime is enough. |
| Python | Nothing to install by hand. The setup script installs [uv](https://docs.astral.sh/uv/) if needed, and uv installs Python 3.12 and PyTorch. |
| Disk | About 6 GB for the Python environment (PyTorch with CUDA). |
| Character | Ironclad only. Any ascension from A0 to A10. |

**CPU only?** It works. On a laptop CPU (Intel Core i9, 32 threads, no GPU), a whole simulated A10 run with the
full search took 55 seconds (600 decisions), so the agent's thinking adds well under a second per action. In the real game the animations
and the mod's action delay take longer than that. The launch scripts give each inference server 4 CPU threads;
change it with `CPU_THREADS`.

## 1. Get the code

```bash
git clone https://github.com/TobyX424242/sts2Zero.git
cd sts2Zero
```

On Windows you can also download the repository as a ZIP from GitHub and unpack it. Run every command below from
the repository folder.

## 2. Set up Python

Linux:

```bash
scripts/setup.sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

The script:

1. uses `uv` if it is on your PATH, or installs it into `.tools/uv` inside the repository (your PATH and shell
   profile are not changed);
2. creates `.venv` with Python 3.12, PyTorch 2.7.1 (CUDA 12.8 build), NumPy and orjson, exactly as pinned in
   `uv.lock`;
3. tells you whether a CUDA GPU and the .NET 9 SDK or runtime were found.

uv's download cache and any Python it downloads stay inside the repository (`.cache/uv`, `.tools/python`), unless
you set `UV_CACHE_DIR` or `UV_PYTHON_INSTALL_DIR` yourself. Deleting the folder removes everything.

The `-ExecutionPolicy Bypass` part lets Windows run the script once without changing your system settings. Use it
for every `.ps1` script in this guide.

## 3. Download the weights

Linux:

```bash
scripts/fetch_weights.sh --binaries
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_weights.ps1 -Binaries
```

This downloads the release `v0.2.0-ic113-s029`, checks every file against the release's `SHA256SUMS`, and puts
the networks into `weights/ic113_s029/`:

| File | Content |
|---|---|
| `weights.pt` | the fight network |
| `macro.pt` | the network for everything outside fights (5-member ensemble) |
| `vbar.json` | the potion value table |
| `manifest.json` | what the release contains and how it was evaluated |

`--binaries` (`-Binaries`) also downloads the prebuilt controller into `bin/LiveAgent/` and the prebuilt mod into
`bin/AgentBridge/`. Leave it out if you want to build both from source with the .NET 9 SDK.

The script uses the GitHub CLI (`gh`) when it is installed and logged in, and plain HTTPS downloads otherwise.
Other options: `--tag` (another release), `--name` (another folder name), `--from DIR` (copy from a local folder
instead of downloading).

## 4. Install the mod

The installer finds the game in your Steam libraries. If it picks the wrong folder, or none, set the game folder
first:

```bash
export STS2_GAME_DIR="/path/to/steamapps/common/Slay the Spire 2"          # Linux
```
```powershell
$env:STS2_GAME_DIR = 'D:\SteamLibrary\steamapps\common\Slay the Spire 2'  # Windows
```

Then install either the prebuilt mod (from step 3) or a build from source:

| | Linux | Windows |
|---|---|---|
| prebuilt mod | `src/live/install_mod.sh prebuilt` | `powershell -ExecutionPolicy Bypass -File src\live\install_mod.ps1 prebuilt` |
| build from source (.NET 9 SDK) | `src/live/install_mod.sh` | `powershell -ExecutionPolicy Bypass -File src\live\install_mod.ps1` |
| show what is installed | `src/live/install_mod.sh status` | `... install_mod.ps1 status` |
| remove the mod | `src/live/install_mod.sh uninstall` | `... install_mod.ps1 uninstall` |

The mod ends up in `<game>/mods/AgentBridge/` (`AgentBridge.dll` and `AgentBridge.json`). Building from source
compiles it against the `sts2.dll` of your own game. The prebuilt mod was compiled against the Linux v0.113.1
game files; the game's code is the same .NET assembly on both systems, so it should load on Windows too, as long
as your game is v0.113.1. If it does not load, build it from source.

You can also do it by hand: unzip `AgentBridge-mod.zip` from the release page into `<game>/mods/`.

## 5. Enable mods in the game (first time only)

1. Start the game. On the first start with a mod present, a popup asks whether to load mods. Choose
   **Load mods**. The game closes itself; start it again.
   If no popup appears, open the **Mods** screen from the main menu and enable mod loading there.
2. With mods enabled, the game keeps its saves in a separate `modded` profile folder. The first time, it copies
   your normal saves there. Agent runs never touch your normal progress.
3. Recommended: switch off tutorials in the game settings. Tutorial popups block the agent until you click them
   away.

You should now see a small **Agent mode** panel at the top right of the main menu.

## 6. Start the agent

Start the game first and leave it at the main menu. Then, in a terminal:

```bash
src/live/play_live.sh                     # Linux
```
```powershell
powershell -ExecutionPolicy Bypass -File src\live\play_live.ps1    # Windows
```

The script starts the two inference servers (this takes a few seconds), starts the controller and connects to
the game. In the game, switch **Agent mode** on in the panel.

Then start a run in one of two ways:

- **From the panel:** pick the ascension (A0 to A10), optionally type a seed, and click **Start agent run**.
- **From the command line:** pass `asc=`, for example `src/live/play_live.sh asc=10`. The controller starts a
  fresh Ironclad run at that ascension as soon as the game is at the main menu.

Any ascension from A0 to A10 works, even if your profile has not unlocked it yet. A run can only start from the
main menu with no saved run in progress: finish or abandon a saved run first. If a run is already in progress
when the controller connects, it plays that run.

Useful options (add them after the script name, as `key=value`):

| Option | Effect |
|---|---|
| `asc=7` | start a fresh run at A7 |
| `seed=ABC123` | use this seed for the run started with `asc=` |
| `loop=1` | after a game over, start the next run (at the same `asc=`, or wait for the panel) |
| `delay_ms=800` | wait this long before each action, so you can follow (default 400 ms; also in the panel) |
| `dry=1` | decide and log only; the agent does not click anything |
| `potion_mix=FILE` | use another potion value table |

The first argument can also be another weights folder, for example `src/live/play_live.sh weights/my_weights
asc=5`. The folder may use the names `weights.pt`, `macro.pt`, `vbar.json`, or the release names
(`*_weights.pt`, `*_macro.pt`, `*_vbar.json`).

Environment variables:

| Variable | Effect |
|---|---|
| `DEVICE=cpu` | run the networks on the CPU even when a GPU is present (default: the GPU when available) |
| `CPU_THREADS=8` | threads per inference server on the CPU (default 4) |
| `LIVEAGENT=source` | build the controller from source even when `bin/LiveAgent/` exists |
| `UNIFORM=1` | no networks: the agent plays legal but random moves (to test the connection) |

On Windows set them in PowerShell first, for example `$env:DEVICE = 'cpu'`.

Press **Ctrl-C** in the terminal to stop the controller and the servers. The game keeps running.

## 7. While it plays

- **Pause:** press **F9** in the game, or click **Pause** in the panel. The agent stops before its next action.
  You can play by hand for as long as you like. Click **Resume** (or press F9 again) to hand control back.
- **When the agent is unsure:** if the controller meets a screen it cannot handle, it pauses by itself and shows
  the reason in the panel. Play that step by hand, then click **Resume**.
- **Speed:** the action delay in the panel (default 400 ms) sets how long the mod waits before each action.
- **Turning it off:** switch **Agent mode** off. The mod then does nothing at all to the game. Agent mode is off
  at every game start.

The panel also shows whether the controller is connected, what the game is waiting for, the last action, and the
last error. The **-** button folds it.

## 8. Logs

| What | Where |
|---|---|
| controller decisions (JSONL) and terminal output | `src/live/logs/` in the repository |
| inference server output | `src/live/logs/server_*` |
| mod log (JSONL: states, actions, pauses) | Linux: `~/.local/share/SlayTheSpire2/AgentBridge/logs/` |
| | Windows: expected in `%APPDATA%\SlayTheSpire2\AgentBridge\logs\` (not verified) |

The mod also writes lines starting with `[AgentBridge]` into the game's own log. When you report a problem,
include the controller log of that session; it can be replayed against the controller without the game
(see `src/live/agent/README.md`).

## Troubleshooting

| Problem | What to do |
|---|---|
| `waiting for the game on 127.0.0.1:47213` | The game is not running, mods are not enabled, or the mod did not load. Check `install_mod status`, then the Mods screen in the game. |
| The panel does not appear | Mods are not enabled (step 5), or the game is not v0.113.1. |
| `start_run` is refused | A saved run exists. Continue it (the agent plays it) or abandon it, then start again. |
| The agent waits on a popup (screen `OTHER`, `modal:...`) | Click the popup away. Switch off tutorials. |
| `no Python environment (.venv)` | Run the setup script (step 2). |
| `weights folder not found` | Run the fetch script (step 3). |
| `dotnet not found` | Install .NET 9 (SDK or runtime), or open a new terminal after installing it. |
| `game folder not found` | Set `STS2_GAME_DIR` (step 4). |
| Windows: `running scripts is disabled on this system` | Start the script with `powershell -ExecutionPolicy Bypass -File ...` as shown. |
| Windows: access denied when installing the mod | Your Steam library is in a protected folder. Run PowerShell as administrator for the install step only. |
| A server died at start | Read `src/live/logs/server_*`. Without a working GPU driver, try `DEVICE=cpu`. |
| Slow on the CPU | Try `CPU_THREADS=2` or `CPU_THREADS=8`. More threads is often slower for these small networks. |

## Status

What has been checked, and what has not:

- **Linux, real game:** the agent won a full A10 run on 2026-11-20 with the released segment-29 weights. Earlier,
  with segment-21 weights, it won an A0 run and an A6 run. On fixed simulated seeds the segment-29 weights win
  about half of all A10 runs (152 of 300), so expect losses too.
- **Linux, scripts:** setup, weight download and checksum check, both inference servers on the CPU, the prebuilt
  controller against them (a full simulated A10 run through the same import and decision path the live game uses),
  and building the mod against the game files.
- **Windows: not tested yet.** The PowerShell scripts were written for Windows PowerShell 5.1 and PowerShell 7
  and reviewed, but never run. The servers and the controller were made portable (the bridge files live in the
  temp folder on Windows). Please report problems with the full terminal output.
- **macOS:** not tested.
