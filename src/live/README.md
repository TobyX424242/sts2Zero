# Live play: the agent plays the real game (v0.113.1)

Setup and use for players: [doc/live-play.md](../../doc/live-play.md). This file describes the components and the
protocol between them.

## Components

| Path | What |
|---|---|
| `mod/` | **AgentBridge**, the game mod (C#, loaded by the game's own mod loader from `<game>/mods/AgentBridge/{AgentBridge.json, AgentBridge.dll}`; Harmony ships with the game). It serves the state at each decision point and performs the actions it is sent, through the same code paths the UI uses. In-game panel: agent mode on/off, ascension A0-A10, seed, "Start agent run", pause/resume (F9), action delay, status line. |
| `agent/` | **LiveAgent**, the controller (C#, .NET 9; references `src/s3/S3`, `src/s2/S2`, `src/s0/S0.Sim` and the v0.113 simulator). It connects to the mod, rebuilds the simulator state from the game state at every decision, asks the S3 agent (`RunAgent`: macro network + network-guided fight search) for its choice and sends it back. See [agent/README.md](agent/README.md). |
| `play_live.sh`, `play_live.ps1` | Start the two inference servers and the controller (Linux / Windows). Same arguments: `[WEIGHTS_DIR] [key=value ...]`. |
| `serve.sh`, `serve.ps1` | Start only the two inference servers and print the controller keys. |
| `common.sh`, `common.ps1` | Shared by the scripts above: weights lookup, Python, bridge paths, server start/stop, controller build. |
| `install_mod.sh`, `install_mod.ps1` | Build the mod and copy it into the game's `mods/` folder; `prebuilt`, `build`, `uninstall`, `status`. |

The simulator (`src/sims/iRyougi-sts2-sim-v113`) is shared with the training chain. The controller never modifies
it: anything beyond its public API is done in `agent/` (reflection where unavoidable).

### Processes and data paths

```
 game + AgentBridge  <-- TCP 127.0.0.1:47213, JSON lines -->  LiveAgent (dotnet)
                                                                 |  shared-memory bridge files
                                                                 v
                                     s3_server.py --serve combat      s3_server.py --serve macro
```

The controller and the two servers talk through file-backed memory maps (protocol in `src/s0/S0.Sim/Bridge.cs`
and `src/s3/S3/MacroBridge.cs`). The scripts create the files under `/dev/shm/live_<pid>_{c0,m0}` on Linux and
`%TEMP%\sts2zero_live_<pid>_{c0,m0}` on Windows, and remove them when they stop. `s3_server.py` uses `/dev/shm`
when it exists and the temp folder otherwise.

### Weights

`play_live` takes a weights folder (default `weights/ic113_s029`, laid out by `scripts/fetch_weights.sh`). It
accepts `weights.pt` / `macro.pt` / `vbar.json` (or the newest `vbar_*.json`) and the release names
`*_weights.pt` / `*_macro.pt` / `*_vbar.json`. The observation layout (`obs=`) is read from the combat
checkpoint's config.

### Controller binary

`play_live` runs `bin/LiveAgent/LiveAgent.dll` (the prebuilt release binary, `scripts/fetch_weights.sh
--binaries`) when it exists. Otherwise it builds `agent/` with `agent/build.sh` (outputs under
`agent/artifacts/`) and rebuilds when a source file is newer. `LIVEAGENT=source` forces the source build.
`scripts/build_release_binaries.sh` builds both release zips (`AgentBridge-mod.zip`, `LiveAgent.zip`).

## Flow

1. The game runs with the mod enabled; `play_live` starts the servers and the controller, which connects and sends
   `hello`.
2. A run starts from the in-game panel (pick A0-A10, "Start agent run") or from `play_live asc=N`, which sends
   `start_run`. The mod starts a fresh **Ironclad** run at that ascension directly, without the game's ascension
   unlock check, so any of A0-A10 works on any profile. Optional seed.
3. Loop: the controller asks for `state`. When `stable` (no animation or action queue running, the game waits
   for input), it decides and sends `act`. The mod waits `delay_ms` (panel setting, default 400 ms) before each
   action.
4. Anything the controller cannot map (an unknown screen, an import failure, an action the mod rejects) makes it
   send `pause` with a reason. The mod shows the reason on the panel; the player does that step by hand and clicks
   Resume. The player can also pause at any time (F9) and take over.
5. Both sides append every state and action to a JSONL log (mod: `<game user dir>/AgentBridge/logs/`, controller:
   `src/live/logs/`), so a failure can be replayed offline against the controller.

## Protocol v1

Transport: TCP, `127.0.0.1:47213`, newline-delimited JSON, UTF-8, one request then one reply. The controller is
the only client (a second connection replaces the first). The mod runs the socket on a background thread and does
all game access on the main (Godot) thread: a request is queued and answered once the main thread has run it.

Requests (`cmd`):

| cmd | fields | reply |
|---|---|---|
| `hello` | `client`, `protocol` (1) | `ok`, `protocol`, `game_version` (release_info), `mod_version`, `agent_mode`, `paused` |
| `state` | – | the state object below |
| `act` | `state_id`, `option` (index into `options`), optional `target` (creature index), optional `cards` (list of indices into the option's `cards`) | `ok`, `error` (text; e.g. stale `state_id`, illegal) |
| `start_run` | `character` ("IRONCLAD"), `ascension` (0-10), optional `seed` (string) | `ok`, `error` |
| `pause` / `resume` | `reason` (pause) | `ok` |
| `config` | optional `delay_ms`, `fast_mode` (bool) | `ok` |

State object:

```
{ "state_id": int,            // increases whenever anything changes; act must quote the current one
  "stable": bool,             // game is waiting for input; options are valid only when true
  "agent_mode": bool, "paused": bool, "pause_reason": str|null,
  "requested_run": {"character": "IRONCLAD", "ascension": 7, "seed": null} | null,   // set by the panel button until a run starts
  "screen": "MAIN_MENU" | "MAP" | "COMBAT" | "COMBAT_SELECT" | "CARD_REWARD" | "REWARDS" | "EVENT" | "ANCIENT"
           | "REST" | "SHOP" | "CARD_SELECT" | "TREASURE" | "CRYSTAL_SPHERE" | "GAME_OVER" | "OTHER",
  "screen_detail": str,       // e.g. the node / screen class name, the event id, the selection prompt purpose
  "run": null | {
     "character", "ascension", "seed", "act_index", "act_ids", "floor", "hp", "max_hp", "gold",
     "deck": [{"id","upgrade","enchantment","enchantment_amount","floor_added"}],
     "relics": [{"id","counter"}], "potions": [id|null per slot],
     "map": {"current": {"col","row"}|null, "visited": [{"col","row"}],
             "points": [{"col","row","type","children":[{"col","row"}]}]},
     "save": { ... }          // the game's own SerializableRun of the live run, serialized with the game's own
                              // JSON context (same JSON as a save file): RNG counters, odds, relic bags, room
                              // progress, saved properties. The controller builds the simulator RunState from this
                              // through the simulator's transplant importer (Runs/Transplant).
  },
  "combat": null | {
     "round": int, "turn_side": "PLAYER"|"ENEMY", "encounter_id": str, "room_type": "MONSTER"|"ELITE"|"BOSS",
     "player": {"hp","max_hp","block","energy","max_energy","stars",
                "powers":[{"id","amount"}],
                "hand":[CARD], "draw":[CARD], "discard":[CARD], "exhaust":[CARD],   // draw pile in game order (the
                "orbs":[{"id","value"}]},                                        // controller treats it as hidden)
     "monsters": [{"index","id","slot","hp","max_hp","block","alive","powers":[{"id","amount"}],
                   "move_id": str,                       // the monster's next move (state machine id)
                   "intents":[{"type","damage","hits"}]}]
  },                          // CARD = {"id","upgrade","enchantment","enchantment_amount","cost","cost_this_turn",
                              //         "playable","needs_target","instance" (stable per fight)}
  "options": [ OPTION ... ]   // every legal action now; empty when not stable
}
```

Options (`kind` + payload). The controller answers with the option's index; `target` / `cards` only where the
option says so.

| screen | kinds |
|---|---|
| COMBAT | `play` {`hand_index`, `card` CARD, `targets`: [creature index] or []}; `potion` {`slot`, `id`, `targets`}; `end_turn` |
| COMBAT_SELECT (hand/pile choice inside a fight) | `select` {`purpose`, `min`, `max`, `cards`: [CARD], `source`} → `act` with `cards` |
| MAP | `map` {`col`, `row`, `type`} |
| CARD_REWARD | `card` {`id`, `upgrade`, ...}; `skip` |
| REWARDS | `reward` {`index`, `type`: GOLD/POTION/RELIC/CARD/..., `id`, `amount`}; `proceed` |
| EVENT, ANCIENT | `event_option` {`index`, `key` (option text key), `title_key`, `text`, `locked`}; ANCIENT options carry `relic` id |
| REST | `rest_option` {`id` (HEAL, SMITH, ...)} |
| SHOP | `buy` {`item`: card/relic/potion/remove, `id`, `price`}; `leave` |
| CARD_SELECT (deck selection: upgrade, remove, transform, ...) | `select` as above, plus `skip` when cancelling is allowed |
| TREASURE | `open`, `relic` {`id`}, `proceed` |
| CRYSTAL_SPHERE | `sphere_cell` {`x`, `y`, `divinations`} per clickable hidden cell while divinations are left; `proceed` |
| GAME_OVER | `to_menu` |
| MAIN_MENU | (none; runs start through `start_run`) |

Additional option kinds:

| screen | kinds |
|---|---|
| REST | `proceed` (leave once the rest action is done) |
| CARD_REWARD | `alternative` {`id`} for non-skip alternatives the game offers (e.g. `REROLL`) |
| REWARDS | after a fight with no rewards overlay, a single `proceed` (the combat room's proceed button) |
| COMBAT_SELECT | `skip` when the selection screen can be closed |
| OTHER, `screen_detail` = `NChooseARelicSelection` | `relic` {`id`, `index`}; `skip` |

### Semantics

- **Indices.** `target` = the monster's `index` (position in `combat.monsters`, the game's `CombatState.Enemies`
  order at the time of that state; the game removes most dead monsters from it, so indices can shift between
  states: match monsters by `id` + `slot`). `cards` = indices into the chosen option's `cards`. Model ids are the
  game's `ModelId.Entry` everywhere (e.g. `STRIKE_IRONCLAD`, `BURNING_BLOOD`, `JAW_WORM`). Map `type` = the game's
  `MapPointType` name (`Monster`, `Elite`, `RestSite`, `Shop`, `Treasure`, `Unknown`, `Boss`, `Ancient`).
  `room_type` = MONSTER/ELITE/BOSS. Intent `type` = the game's `IntentType` name; attack intents carry `damage`
  (per hit, as displayed, after the game's damage hooks) and `hits`.
- **`state_id`** changes whenever the state's content (screen, run summary, combat, options) changes, and after
  every action the mod performs. `run.save` is not part of that comparison (it carries wall-clock fields).
- **`stable`** = the game waits for input: the screen's own readiness signals (combat: player phase, actions
  enabled, action executor idle, queue empty, hand in play mode, end-turn button enabled; other screens: their
  buttons enabled), held for 250 ms, the screen at least 600 ms old, no mod action pending or running.
  `options` and `run.save` are filled only when `stable` (empty / null otherwise).
- **`act` is asynchronous.** The reply `ok` means "validated and scheduled". The mod waits `delay_ms`, checks that
  the state did not change meanwhile (else the action is dropped and `last_error` says so), then performs it
  through the UI path. Multi-step options (card selections, shop purchases, `to_menu`) run several clicks. The
  result shows up in later `state` replies. If performing an action fails, the mod pauses with `pause_reason`.
  `act` is refused while agent mode is off, while paused, while busy, when not stable, or on a stale `state_id`.
- **`start_run`** works only on the main menu with no saved run (the game would otherwise overwrite it). It replies
  after validation; the run then loads (seconds). Seeds go through the game's `SeedHelper.CanonicalizeSeed`.
  `requested_run` is set from the moment a run start is requested (panel button or `start_run`) until the run is
  in progress. The panel button starts the run itself (it also clears a pause).
- **Ancient dialogue** lines are clicked through by the mod itself in agent mode (no decision involved); the
  ANCIENT screen becomes stable when the options are clickable.
- **Event option keys.** `key` is the game's option text key; `title_key` (optional) is the loc key of the
  option's title, e.g. `MR_STRUGGLES.title`. The controller normalises live keys to the simulator's
  `EventOption.Key`: DOLL_ROOM relic options take the relic id from `title_key`; TABLET_OF_TRUTH `DECIPHER_1` →
  `DECIPHER`; COLOSSAL_FLOWER `EXTRACT_CURRENT_PRIZE_N` / `REACH_DEEPER_N` → without `_N`; TRIAL verdicts
  `GUILTY` / `INNOCENT` → `<DEFENDANT>_GUILTY` / `<DEFENDANT>_INNOCENT`.
- **CRYSTAL_SPHERE.** The minigame is pure chance: the controller clicks a uniformly random hidden cell (the game's
  own AutoSlay plays it the same way). The rewards it gives open the normal REWARDS / CARD_REWARD screens.
- **`config.fast_mode`** switches the game's own animation-speed setting to Fast (restored when switched off or
  when agent mode is turned off). `delay_ms` is persisted in `user://AgentBridge/config.json`.
- Potion options are offered in COMBAT only (no out-of-combat potion use, no potion discard); potion rewards and
  shop potions are not offered while the potion belt is full. Cards and potions that target allies are not
  offered.

Extra fields (ignore them if not needed):

- every reply has `ok`; `hello` adds `log_file` (the mod's JSONL log) and `warning` on a protocol mismatch; `act`
  adds `state_id`, `scheduled_in_ms`.
- state: `wait_reason` (why not stable), `last_action`, `last_error`; `run.act_floor`, `run.save_error`.
- CARD: `affliction`, `x_cost` (then `cost` = -1), `target_type`. Monster: `hittable`. Orb: `evoke`.
- `select`: `prompt_key` (the game's loc key, e.g. `TO_EXHAUST`), `prompt` (text). `purpose` is one of upgrade,
  transform, enchant, remove, exhaust, discard, absorb, choose, or the lower-cased prompt key. `source` = `hand`,
  `pile` (in-fight pile grid), `deck`, `generated` (choose one of N new cards).
- `reward`: `text`, `linked` (part of a choose-one linked set). `event_option`: `description`, `proceed`.
  `rest_option`: `index`, `enabled`. `buy`: `entry_index`; card items also `upgrade`, `on_sale`, `card` (CARD).
  `card` (CARD_REWARD): `index`, `enchantment`, `enchantment_amount`, `card` (CARD). TREASURE `relic`: `index`.

`screen_detail`: event / encounter id on EVENT, ANCIENT, COMBAT; the selection screen class (or `hand:<mode>`) on
COMBAT_SELECT / CARD_SELECT; on OTHER the blocking node, e.g. `modal:NSomeFtue` (tutorial popup: click it away),
`capstone:NDeckViewScreen`, `event:<id> (custom event screen)`.

Whoever changes the protocol updates this section and adds a line to the change list.

### Changes

- v1 (2026-11-20): first version.
- v1 (2026-11-20): clarified indices, `state_id` / `stable` / asynchronous `act` semantics, `run.save` only when
  stable; added the extra fields and the option kinds REST `proceed`, CARD_REWARD `alternative`, REWARDS
  `proceed`-only, COMBAT_SELECT `skip`, OTHER relic selection. All additive.
- v1 (2026-11-20): screen `CRYSTAL_SPHERE` (was OTHER, played by hand) with options `sphere_cell` and `proceed`.
- v1 (2026-11-20): `event_option` carries the optional `title_key`; the controller normalises event option keys
  (DOLL_ROOM, TABLET_OF_TRUTH, COLOSSAL_FLOWER, TRIAL). Additive.

## Mod notes (AgentBridge)

Files: `mod/AgentBridge.csproj` (net9.0, the game's runtime; references `sts2.dll` and `GodotSharp.dll` in the
game's data folder in place, nothing copied), `mod/AgentBridge.json` (manifest: id `AgentBridge`, `has_dll`, no
PCK, `min_game_version` 0.113.1), `mod/src/*.cs`.

| file | role |
|---|---|
| `Bridge.cs` | `[ModInitializer("Init")]` entry; main-thread pump (SceneTree.ProcessFrame): requests, stability, delayed actions, auto ancient dialogue, F9, panel refresh |
| `BridgeServer.cs` | TCP 127.0.0.1:47213 on a background thread; requests are queued to the main thread |
| `Screens.cs` | screen detection + raw readiness (read-only) |
| `StateBuilder.cs` | run / combat / options JSON; every option carries its validator and executor |
| `Actions.cs` | multi-step UI sequences (grid / hand selection + confirm, shop, game over) |
| `RunStarter.cs` | new Ironclad run at A0-A10 via the game's own setup path |
| `Panel.cs` | the overlay (plain Godot controls) |
| `Util.cs` | JSON helpers, the JSONL log |

**Build.** `install_mod.sh build` / `install_mod.ps1 build`, or `dotnet build mod/AgentBridge.csproj -c Release`.
The game folder comes from `-p:STS2GameDir=...`, else the environment variable `STS2_GAME_DIR`, else Steam's
default library (Windows `C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2`, Linux
`~/.local/share/Steam/steamapps/common/Slay the Spire 2`). The install scripts also search every Steam library
listed in `libraryfolders.vdf`. The data folder is `data_sts2_linuxbsd_x86_64` (Linux) or
`data_sts2_windows_x86_64` (Windows), whichever holds `sts2.dll`. NuGet / .NET caches stay in the repository's
`.cache/`.

**Enable mods (first time).** The game loads mods only after the player agreed: on the first start with a mod
present the main menu shows a popup; choose "Load mods" (the game closes; start it again). Otherwise open the Mods
screen in the main menu. While modded, the game keeps saves in a separate `modded/` profile folder (it copies the
normal saves there once), so agent runs do not touch unmodded progress; a copied in-progress run must be continued
or abandoned before `start_run` works. Tutorial popups block the agent (state shows `OTHER`, `modal:...`) until
clicked away.

**Panel.** Top right, below the game's top bar. "Agent mode" toggle (off at every game start; when off, the panel
is just the toggle and the mod does nothing to the game: `act` / `start_run` are refused, `state` still answers).
When on: ascension A0-A10, optional seed, "Start agent run" (main menu, no saved run), Pause/Resume (also F9),
action delay (ms, default 400), status line (controller connected, ready / busy, screen), last action, pause
reason or last error; "-" folds the panel.

**Logs.** Game log lines are prefixed `[AgentBridge]`. JSONL: `user://AgentBridge/logs/bridge_<date>.jsonl`. The
game sets `custom_user_dir_name = SlayTheSpire2`, so `user://` is `~/.local/share/SlayTheSpire2/` on Linux and is
expected to be `%APPDATA%\SlayTheSpire2\` on Windows (not verified). The exact path is printed in the game log and
returned by `hello` (`log_file`). Logged: every new stable state served (the save JSON only for the first state
of each screen), every `act` request, each executed / dropped / finished action, start_run, pause / resume,
config, agent-mode changes and the mod's automatic steps.

**Manual check** (Linux, game running):
`printf '{"cmd":"hello","client":"nc","protocol":1}\n{"cmd":"state"}\n' | nc -q1 127.0.0.1 47213` shows
`screen`, `stable` and `wait_reason`.
