# Live play: the agent controls the real game (v0.113.1)

Owner, 2026-11-20: "connect our agent to the real game, maybe by a mode, so I can see the agent actually control the game
and play. Make sure I can select any level from A0 to A10."

## Pieces

| Path | What |
|---|---|
| `mod/` | **AgentBridge**, a game mod (C#, the game's own mod loader: `<game>/mods/AgentBridge/{AgentBridge.json, AgentBridge.dll}`, Harmony is bundled with the game). Runs inside the game: serves the state at each decision point and performs the actions it is sent, through the same code paths the UI uses, so the owner watches it happen on screen. In-game panel: agent mode on/off, ascension A0-A10, "Start agent run", pause/resume (hotkey), status line. |
| `agent/` | **LiveAgent**, the controller (C#, references `src/s3/S3` + `src/s0` + the v0.113 simulator). Connects to the mod, rebuilds the simulator state from the game state at every decision, asks the S3 agent (`RunAgent`: macro network + network-guided fight search) for its choice and sends it back. |
| `play_live.sh` | Starts the two inference servers (`src/s3/py/s3_server.py --serve combat/macro`) and the controller, with the weights of a chain segment. |
| `install_mod.sh` | Builds the mod and copies it into the game's `mods/` folder (run by the owner). |

Simulator code (`src/sims/iRyougi-sts2-sim-v113`) is shared with the training chain: **never modify it**. Anything the
controller needs beyond its public API is done in `agent/` (reflection if unavoidable), or written as a separate patch file
under `agent/` for review.

## Flow

1. Owner starts the game with the mod enabled (Settings → mods, first time only), then `src/live/play_live.sh`.
2. Run start, either way: the in-game panel (pick A0-A10, click "Start agent run"), or `play_live.sh asc=7`, which
   sends `start_run`. The mod starts a fresh **Ironclad** run at that ascension directly, without the game's ascension
   unlock check (so any of A0-A10 works on any profile). Optional seed.
3. Loop: controller asks for `state`; when `stable` (no animation or action queue running, the game waits for input),
   it decides and sends `act`. The mod waits `delay_ms` (panel setting, default 400 ms) before each action so the owner
   can follow.
4. Anything the controller cannot map (an unknown screen, an import failure, an action the mod rejects) → it sends
   `pause` with a reason; the mod shows the reason on the panel; the owner plays that step by hand and clicks Resume.
   The owner can also pause at any time (hotkey F9) and take over.
5. Every state and action is appended to a JSONL log by both sides (`mod`: `<game user dir>/AgentBridge/logs/`,
   `agent`: `src/live/logs/`), so a failure can be replayed offline against the controller.

## Protocol v1

Transport: TCP, `127.0.0.1:47213`, newline-delimited JSON, one request then one reply, UTF-8. The controller is the only
client (a second connection replaces the first). The mod runs the socket on a background thread and does all game
access on the main (Godot) thread (queue the request, answer when the main thread has run it).

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
           | "REST" | "SHOP" | "CARD_SELECT" | "TREASURE" | "GAME_OVER" | "OTHER",
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

Options (`kind` + payload). The controller answers with the option's index; `target`/`cards` only where the option says so.

| screen | kinds |
|---|---|
| COMBAT | `play` {`hand_index`, `card` CARD, `targets`: [creature index] or []}; `potion` {`slot`, `id`, `targets`}; `end_turn` |
| COMBAT_SELECT (hand/pile choice inside a fight) | `select` {`purpose`, `min`, `max`, `cards`: [CARD], `source`} → `act` with `cards` |
| MAP | `map` {`col`, `row`, `type`} |
| CARD_REWARD | `card` {`id`, `upgrade`, ...}; `skip` |
| REWARDS | `reward` {`index`, `type`: GOLD/POTION/RELIC/CARD/..., `id`, `amount`}; `proceed` |
| EVENT, ANCIENT | `event_option` {`index`, `key` (option text key), `text`, `locked`}; ANCIENT options carry `relic` id |
| REST | `rest_option` {`id` (HEAL, SMITH, ...)} |
| SHOP | `buy` {`item`: card/relic/potion/remove, `id`, `price`}; `leave` |
| CARD_SELECT (deck selection: upgrade, remove, transform, ...) | `select` as above, plus `skip` when cancelling is allowed |
| TREASURE | `open`, `relic` {`id`}, `proceed` |
| GAME_OVER | `to_menu` |
| MAIN_MENU | (none; runs start through `start_run`) |

#### v1 as implemented by the mod (clarifications and additive extensions; nothing above is removed)

Semantics:
- **Indices.** `target` = the monster's `index` (position in `combat.monsters`, the game's `CombatState.Enemies`
  order at the time of that state: the game removes most dead monsters from it, so indices can shift between
  states; match monsters by `id` + `slot`). `cards` = indices into the chosen option's `cards`.
  Model ids are the game's `ModelId.Entry` everywhere (e.g. `STRIKE_IRONCLAD`, `BURNING_BLOOD`, `JAW_WORM`).
  Map `type` = the game's `MapPointType` name (`Monster`, `Elite`, `RestSite`, `Shop`, `Treasure`, `Unknown`,
  `Boss`, `Ancient`). `room_type` = MONSTER/ELITE/BOSS. Intent `type` = the game's `IntentType` name; attack
  intents carry `damage` (per hit, as displayed, after the game's damage hooks) and `hits`.
- **`state_id`** changes whenever the state's content (screen, run summary, combat, options) changes, and after
  every action the mod performs. `run.save` is not part of that comparison (it carries wall-clock fields).
- **`stable`** = the game waits for input: the screen's own readiness signals (combat: player phase, actions
  enabled, action executor idle, queue empty, hand in play mode, end-turn button enabled; other screens: their
  buttons enabled), held for 250 ms, the screen at least 600 ms old, no mod action pending or running.
  `options` and `run.save` are filled only when `stable` (empty / null otherwise).
- **`act` is asynchronous**: the reply `ok` means "validated and scheduled"; the mod waits `delay_ms`, checks that
  the state did not change meanwhile (else the action is dropped and `last_error` says so), then performs it
  through the UI path. Multi-step options (card selections, shop purchases, `to_menu`) run several clicks. The
  result shows up in later `state` replies. If performing an action fails, the mod pauses with `pause_reason`.
  `act` is refused while agent mode is off, while paused, while busy, when not stable or on a stale `state_id`.
- **`start_run`** works only on the main menu with no saved run (the game would otherwise overwrite it): abandon or
  finish a saved run first. It replies after validation; the run then loads (seconds). Seeds go through the game's
  `SeedHelper.CanonicalizeSeed`. `requested_run` is set from the moment a run start is requested (panel button or
  `start_run`) until the run is in progress. The panel button starts the run itself (it also clears a pause).
- **Ancient dialogue** lines are clicked through by the mod itself in agent mode (no decision involved); the
  ANCIENT screen becomes stable when the options are clickable.
- **`config.fast_mode`** switches the game's own animation-speed setting to Fast (restored when switched off or
  when agent mode is turned off). `delay_ms` is persisted in `user://AgentBridge/config.json`.
- Potion options are offered in COMBAT only (no out-of-combat potion use, no potion discard); potion rewards and
  shop potions are not offered while the potion belt is full. Cards/potions that target allies are not offered.

Extra fields (ignore them if not needed):
- every reply has `ok`; `hello` adds `log_file` (the mod's JSONL log) and `warning` on a protocol mismatch;
  `act` adds `state_id`, `scheduled_in_ms`.
- state: `wait_reason` (why not stable), `last_action`, `last_error`; `run.act_floor`, `run.save_error`.
- CARD: `affliction`, `x_cost` (then `cost` = -1), `target_type`. Monster: `hittable`. Orb: `evoke`.
- `select`: `prompt_key` (the game's loc key, e.g. `TO_EXHAUST`), `prompt` (text). `purpose` is one of upgrade,
  transform, enchant, remove, exhaust, discard, absorb, choose, or the lower-cased prompt key.
  `source` = `hand`, `pile` (in-fight pile grid), `deck`, `generated` (choose one of N new cards).
- `reward`: `text`, `linked` (part of a choose-one linked set). `event_option`: `description`, `proceed`.
  `rest_option`: `index`, `enabled`. `buy`: `entry_index`; card items also `upgrade`, `on_sale`, `card` (CARD).
  `card` (CARD_REWARD): `index`, `enchantment`, `enchantment_amount`, `card` (CARD). TREASURE `relic`: `index`.

Additional option kinds:
| screen | kinds |
|---|---|
| REST | `proceed` (leave once the rest action is done) |
| CARD_REWARD | `alternative` {`id`} for non-skip alternatives the game offers (e.g. `REROLL`) |
| REWARDS | after a fight with no rewards overlay, a single `proceed` (the combat room's proceed button) |
| COMBAT_SELECT | `skip` when the selection screen can be closed |
| OTHER, `screen_detail` = `NChooseARelicSelection` | `relic` {`id`, `index`}; `skip` |

`screen_detail`: event / encounter id on EVENT, ANCIENT, COMBAT; the selection screen class (or `hand:<mode>`) on
COMBAT_SELECT / CARD_SELECT; on OTHER the blocking node, e.g. `modal:NSomeFtue` (tutorial popup: click it away),
`capstone:NDeckViewScreen`, `NCrystalSphereScreen`, `event:<id> (custom event screen)`.

The protocol may grow while it is built: whoever changes it updates this section and adds a line to the change list.

### Changes
- v1 (2026-11-20): first version.
- v1 (2026-11-20, mod): clarified indices, `state_id` / `stable` / asynchronous `act` semantics, `run.save` only
  when stable; added extra fields and the option kinds REST `proceed`, CARD_REWARD `alternative`, REWARDS
  `proceed`-only, COMBAT_SELECT `skip`, OTHER relic selection. All additive: a v1 controller keeps working.
- v1 (2026-11-20, owner): screen `CRYSTAL_SPHERE` (was OTHER, played by hand) with options `sphere_cell` {`x`, `y`,
  `divinations`} per clickable hidden cell while divinations are left, and `proceed`. The controller clicks a uniformly
  random cell (the minigame is pure chance; the game's own AutoSlay plays it the same way); the rewards it gives open
  the normal REWARDS / CARD_REWARD screens, decided by the agent.

## Mod notes (AgentBridge)

Files: `mod/AgentBridge.csproj` (net9.0, the game's runtime; references `sts2.dll` and `GodotSharp.dll` from the game
folder in place, nothing copied), `mod/AgentBridge.json` (manifest: id `AgentBridge`, `has_dll`, no PCK),
`mod/src/*.cs`, `install_mod.sh`.

| file | role |
|---|---|
| `Bridge.cs` | `[ModInitializer("Init")]` entry; main-thread pump (SceneTree.ProcessFrame): requests, stability, delayed actions, auto ancient dialogue, F9, panel refresh |
| `BridgeServer.cs` | TCP 127.0.0.1:47213 on a background thread; requests are queued to the main thread |
| `Screens.cs` | screen detection + raw readiness (read-only) |
| `StateBuilder.cs` | run / combat / options JSON; every option carries its validator and executor |
| `Actions.cs` | multi-step UI sequences (grid / hand selection + confirm, shop, game over) |
| `RunStarter.cs` | new Ironclad run at A0-A10 via the game's own setup path |
| `Panel.cs` | the overlay (plain Godot controls) |

**Build / install** (owner): `src/live/install_mod.sh` (build + copy `AgentBridge.dll` and `AgentBridge.json`
into `<game>/mods/AgentBridge/`), `install_mod.sh build` (build only), `install_mod.sh uninstall`, `install_mod.sh
status`. Other game folder: `STS2_GAME_DIR=... install_mod.sh`. .NET/NuGet caches stay in the project's `.cache/`.

**Enable mods (first time):** the game loads mods only after the player agreed: on the first start with a mod
present the main menu shows a popup; choose "Load mods" (the game closes; start it again). Otherwise open the
Mods screen in the main menu. While modded, the game keeps saves in a separate `modded/` profile folder (it copies the
normal saves there once), so agent runs do not touch unmodded progress; a copied in-progress run must be continued
or abandoned before `start_run` works. Tip: switch off tutorials in the game settings; tutorial popups block the
agent (state shows `OTHER`, `modal:...`) until clicked away.

**Use:** panel at the top right, below the game's top bar: "Agent mode" toggle (off at every game start; when off
the panel is just the toggle and the mod does nothing to the game: `act` / `start_run` are refused, `state` still
answers). When on: ascension A0-A10, optional seed, "Start agent run" (main menu, no saved run), Pause/Resume (also
F9), action delay (ms, default 400), status line (controller connected, ready / busy, screen), last action,
pause reason or last error, "-" folds the panel.

**Logs:** game log lines are prefixed `[AgentBridge]`. JSONL: `user://AgentBridge/logs/bridge_<date>.jsonl`; on
Linux `user://` is `~/.local/share/SlayTheSpire2/` (the game sets `custom_user_dir_name = SlayTheSpire2`), so
`~/.local/share/SlayTheSpire2/AgentBridge/logs/`. The exact path is printed in the game log and returned by
`hello` (`log_file`). Logged: every new stable state served (the save JSON only for the first state of each
screen), every `act` request, each executed / dropped / finished action, start_run, pause / resume, config,
agent-mode changes and the mod's auto steps.

**What to check first when testing** (nothing was run against the live game yet):
1. The mod loads: game log shows `[AgentBridge] listening on 127.0.0.1:47213` and the panel appears.
2. `printf '{"cmd":"hello","client":"nc","protocol":1}\n{"cmd":"state"}\n' | nc -q1 127.0.0.1 47213` on the main
   menu and in a fight: `screen`, `stable`, `wait_reason`.
3. Start a run from the panel at an ascension above your unlocked one; Neow's dialogue should click through by itself.
4. Per screen, that `stable` becomes true (else `wait_reason` names the blocking signal) and one act of each kind
   works: play (with target), potion, end_turn, map, reward, card reward / skip, event option, rest + smith grid,
   shop buy / remove / leave, treasure open / relic / proceed, hand selection (e.g. Armaments, True Grit),
   game over `to_menu`.
