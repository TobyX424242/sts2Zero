# LiveAgent: the live-play controller

LiveAgent lets the S3 agent (`src/s3/S3/RunAgent.cs`: macro network + network-guided fight search) play the real
game through the AgentBridge mod (protocol v1, [../README.md](../README.md)). Normally you start it with
`src/live/play_live.sh` or `play_live.ps1`, which also start the inference servers.

## Build and run

| | |
|---|---|
| prebuilt | `bin/LiveAgent/LiveAgent.dll` from the release (`scripts/fetch_weights.sh --binaries`); needs the .NET 9 runtime |
| from source | `src/live/agent/build.sh` (Linux) or `play_live.ps1` (Windows, builds when needed); output `src/live/agent/artifacts/bin/LiveAgent/release/LiveAgent.dll`; needs the .NET 9 SDK |
| release zip | `scripts/build_release_binaries.sh` (framework-dependent `dotnet publish`, the same files on Linux and Windows) |

The build keeps every bin/obj (also those of S3, S2, S0 and the simulator) under `artifacts/`, so nothing is
written into the shared source folders. The project disables quick JIT for loops (see the comment in
`LiveAgent.csproj`).

Commands (`dotnet LiveAgent.dll <command> key=value ...`, run from the repository root or anywhere below it):

| Command | What |
|---|---|
| `play` | connect to the mod and play |
| `replay file=LOG.jsonl` | feed a controller log's states to the decision code again (no game needed) |
| `selfcheck runs=N` | play simulated runs, export every state as the mod would, and run it through import, decision and mapping |

Keys: those of `S3 play` (`src/s3/S3/Program.cs`) with the same meaning. The defaults are the s3ic chain
evaluation (`AgentSetup.ChainEval`: greedy macro, 64 simulations on every move, ...; `preset=none` drops them).
`bridge` / `mbridge` are the bridge files of the inference servers, or `uniform` (default: uniform networks, no
servers). Play keys: `asc` (0-10), `seed`, `force_start`, `loop`, `host` (127.0.0.1), `port` (47213), `log`
(`src/live/logs`), `dry`, `poll_ms` (150), `delay_ms`, `root` (repository root; found automatically).

## Files

| File | What |
|---|---|
| `Program.cs` | the three commands and the play loop |
| `AgentSetup.cs` | agent keys exactly as `S3 play` parses them; the live defaults |
| `RunImport.cs` | game state (`run.save` = the game's SerializableRun JSON) → simulator RunState + player + RunAgent's FloorFacts |
| `Controller.cs` | one state → RunAgent's decision → option index (every screen); event option key normalisation |
| `Shadow.cs` | the simulator copy of the game's fight, synced at every decision |
| `SelfCheck.cs` | simulator runs exported as protocol states and run through import + decision + mapping |
| `GameClient.cs` | protocol client, JSONL log |
| `Priv.cs` | reflection helpers (non-public simulator / RunAgent members; nothing in S3 or the simulator is modified) |
| `build.sh` | the source build described above |

## How a decision is made

RunAgent is used unchanged. There is one RunAgent per game run. Before every out-of-combat decision the run is
imported again and bound to it (its private `_run`, `_p`, `_facts`). Then its own public decision method is called
on simulator objects built from the game's options, and the object it returns is mapped back to the option:

| Screen | RunAgent method | Mapping back |
|---|---|---|
| MAP | ChooseMapPointAsync (points of the imported map) | map coordinates |
| CARD_REWARD | ChooseRewardActionAsync (a RewardsSet holding a CardReward with the offered cards) | card object → option; skip |
| REWARDS | RewardDecisionClassifier's order (gold, potion, relic, card, extras), RunAgent's belt rule on a full belt | reward type + id |
| EVENT / ANCIENT | ChooseEventOptionAsync (EventRoom of the event on the imported run; options keyed by the simulator's EventOption.Key, normalised from the game's text key; Ancient relics as offered content) | option object |
| REST | ChooseRestSiteActionAsync (Heal, Smith per upgradable deck card, other options by id) | option id; the Smith card is picked in the selection screen that follows |
| SHOP | ChooseShopActionAsync (a MerchantInventory of the offered entries at the shown prices) | entry; a removal's card is picked in the selection screen |
| CARD_SELECT | ChooseCardsAsync (deck cards of the import, or new cards for "generated"), with the deck purpose (upgrade / remove / transform) | card objects → indices |
| COMBAT | ChooseCombatActionAsync on the synced shadow fight (T_f table built at the fight's first decision, as in self-play) | hand position / potion slot, target = the game's monster index |
| COMBAT_SELECT | the answer RunAgent gave to the same prompt in the shadow fight (or its search action's planned answer); else ChooseCardsAsync on the offered cards | card keys → indices |
| OTHER (relic choice) | the Ancient head through RunAgent's option sampler (no such screen in the simulator) | relic id |
| CRYSTAL_SPHERE | a uniformly random hidden cell (pure chance) | cell |
| TREASURE, GAME_OVER | open / take / proceed; to_menu | – |

Anything that cannot be mapped (an unknown screen, a card or event the simulator lacks, an option missing on
screen, an exception) sends `pause` with the reason; the player does that step and resumes.

Event option keys: the game's keys differ from the simulator's for a few events. The controller maps them:
DOLL_ROOM relic options use the relic id from `title_key`; TABLET_OF_TRUTH `DECIPHER_1` → `DECIPHER`;
COLOSSAL_FLOWER `EXTRACT_CURRENT_PRIZE_N` / `REACH_DEEPER_N` → without `_N`; TRIAL verdicts → `<DEFENDANT>_GUILTY` /
`<DEFENDANT>_INNOCENT`.

## Approximations

Run import (`RunImport.cs`): RNG streams, odds, relic grab bags and room sequences are the simulator's for the
seed, not the save's. No decision reads them: the encoder does not, and fights resample every stream. The fight
history for the encoder's fight tokens comes from the map point history: end HP = current_hp - hp_healed, start
HP = end HP + damage_taken. Saved properties the simulator cannot read are dropped (noted in the log); unknown ids
are dropped.

Shadow fight (`Shadow.cs`), synced at every decision: round; the player's HP, max HP, block, energy, stars and
powers (amount, order); the four piles (by the game's card instance ids: id, upgrade, enchantment, this-turn cost,
affliction); draw order (keeping the copy's known top / bottom when they agree); the potion belt; monsters (id +
slot matching, HP, block, powers, next move by state-machine id, spawned and removed monsters, revivable dead
ones). Not synced: hidden power counters (e.g. Inferno's next damage), relic counters inside a fight, cards'
in-fight growth (Rampage, Thrash) when the copy had to create or rebuild the card, monsters' move history beyond
the next move, afflictions' amounts. The copy applies every action the agent sends. When the copy's fight ends
before the game's (its own random targets or rolls), it is rebuilt and RunAgent keeps its table and search.

Not available from the mod (v1): potion rewards with a full belt and out-of-combat potion discard (the potion_mix
belt rule cannot swap; such potions are left), some custom event screens (e.g. the Fake Merchant: the mod waits for
the player), and the source model of a deck selection (deck_route uses the current event when there is one).

## Self-check

```bash
dotnet bin/LiveAgent/LiveAgent.dll selfcheck runs=8 asc=10 tank=1 sims=4
```

Uniform networks, CPU, a few seconds per run; see `SelfCheck.cs` for what is compared. With `bridge=` /
`mbridge=` (from `src/live/serve.sh`) and `obs=5` it uses the real networks. `dump=FILE` writes the exported
states (JSONL), which `replay file=FILE` reads.

The import reuses one simulator RunState per game run: thousands of RunState constructions per process (one per
decision) crashed the process in the simulator's map generation (MapPathPruning) with errors that code cannot
throw, the same failure that quick JIT for loops caused.
