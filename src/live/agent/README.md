# LiveAgent: the controller of live play

The S3 agent (`src/s3/S3/RunAgent.cs`: macro network + network-guided fight search) plays the real game through the
AgentBridge mod (`../README.md`, protocol v1). Start it with `src/live/play_live.sh` (servers + controller).

| File | What |
|---|---|
| `Program.cs` | `LiveAgent play` (the loop), `replay file=LOG` (a controller log fed back to the decision code, no game), `selfcheck` |
| `AgentSetup.cs` | agent keys exactly as `S3 play` parses them; live defaults = the s3ic chain evaluation (`ChainEval`); `preset=none` drops them |
| `RunImport.cs` | game state (run.save = the game's SerializableRun JSON) -> simulator RunState + player + RunAgent's FloorFacts |
| `Controller.cs` | one state -> RunAgent's decision -> option index (every screen) |
| `Shadow.cs` | the simulator copy of the game's fight, synced at every decision |
| `SelfCheck.cs` | simulator runs exported as protocol states (as the mod writes them) and run through import + decision + mapping |
| `GameClient.cs` | protocol client, JSONL log |
| `Priv.cs` | reflection helpers (non-public simulator / RunAgent members; nothing in S3 or the simulator is modified) |
| `build.sh` | build with every bin/obj under `artifacts/` (src/sims is the main checkout's: no outputs there) |

## How a decision is made

RunAgent is not changed. One RunAgent per game run; before every out-of-combat decision the run is imported again and
bound to it (its private `_run`, `_p`, `_facts`), then its own public decision method is called on simulator objects
built from the game's options, and the object it returns is mapped back to the option:

| Screen | RunAgent method | Mapping back |
|---|---|---|
| MAP | ChooseMapPointAsync (points of the imported map) | map coordinates |
| CARD_REWARD | ChooseRewardActionAsync (a RewardsSet holding a CardReward with the offered cards) | card object -> option; skip |
| REWARDS | RewardDecisionClassifier's order (gold, potion, relic, card, extras), RunAgent's belt rule on a full belt | reward type + id |
| EVENT / ANCIENT | ChooseEventOptionAsync (EventRoom of the event on the imported run, options keyed by the last segment of the game's text key, Ancient relics as offered content) | option object |
| REST | ChooseRestSiteActionAsync (Heal, Smith per upgradable deck card, other options by id) | option id; the Smith card is picked in the selection screen that follows |
| SHOP | ChooseShopActionAsync (a MerchantInventory of the offered entries at the shown prices) | entry; a removal's card is picked in the selection screen |
| CARD_SELECT | ChooseCardsAsync (deck cards of the import, or new cards for "generated"), with the deck purpose (upgrade / remove / transform) | card objects -> indices |
| COMBAT | ChooseCombatActionAsync on the synced shadow fight (T_f table built at the fight's first decision, as in self-play) | hand position / potion slot, target = the game's monster index |
| COMBAT_SELECT | the answer RunAgent gave to the same prompt in the shadow fight (or its search action's planned answer); else ChooseCardsAsync on the offered cards | card keys -> indices |
| OTHER (relic choice) | the Ancient head through RunAgent's option sampler (no such screen in the simulator) | relic id |
| TREASURE, GAME_OVER | open / take / proceed; to_menu | - |

Anything that cannot be mapped (an unknown screen, a card or event the simulator lacks, an option missing on screen, an
exception) sends `pause` with the reason; the owner plays that step and resumes.

## Approximations

Run import (`RunImport.cs`): RNG streams, odds, relic grab bags and room sequences are the simulator's for the seed,
not the save's (no decision reads them: the encoder does not, fights resample every stream). Fight history for the
encoder's fight tokens comes from the map point history: end HP = current_hp - hp_healed, start HP = end HP +
damage_taken. Saved properties the simulator cannot read are dropped (noted in the log); unknown ids are dropped.

Shadow fight (`Shadow.cs`): synced every decision: round, HP / max HP / block / energy / stars / powers (amount, order)
of the player, the four piles (by the game's card instance ids; id, upgrade, enchantment, this-turn cost, affliction),
draw order (with the copy's known top / bottom kept when they agree), potion belt, monsters (id + slot matching, HP,
block, powers, next move by state-machine id, spawned and removed monsters, revivable dead ones). Not synced: hidden
power counters (e.g. Inferno's next damage), relic counters inside a fight, cards' in-fight growth (Rampage, Thrash)
when the copy had to create or rebuild the card, monsters' move history beyond the next move, afflictions' amounts.
The copy applies every action the agent sends; when the copy's fight ends before the game's (its own random targets
or rolls) it is rebuilt and RunAgent keeps its table and search.

Not available from the mod (v1): potion rewards with a full belt and out-of-combat potion discard (the chain's
potion_mix belt rule cannot swap; such potions are left), custom event screens (Crystal Sphere, Fake Merchant: the mod
waits for the owner), the source model of a deck selection (deck_route uses the current event when there is one).

## Self-check

`dotnet artifacts/bin/LiveAgent/release/LiveAgent.dll selfcheck runs=8 asc=10 tank=1 sims=4` (uniform networks, CPU,
seconds per run): see `SelfCheck.cs` for what is compared. `dump=FILE` writes the exported states (JSONL), which
`replay file=FILE` reads.

Note: the build disables quick JIT for loops (`LiveAgent.csproj`); the import also reuses one simulator RunState per
game run. Thousands of RunState constructions per process (one per decision) crashed the process in the simulator's
map generation (MapPathPruning) with AccessViolation / IndexOutOfRange / NullReference in code that cannot throw them.
