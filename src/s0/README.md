# S0: simulator extensions for combat search

Code for PROPOSAL.md section 6.1, items 3-6, on top of iRyougi/sts2-sim at commit `db844ad`
(+ `src/sims/patches/iryougi_local.patch`, + `src/sims/patches/iryougi_s0.patch` from this folder).

| Path | Content |
|---|---|
| `S0.Sim/` | Library `Sts2Sim.S0` (references `Sts2Sim.Core`) |
| `S0.Sim/ResamplingClone.cs` | `ResamplingClone.Exact(s)`, `ResamplingClone.Resampled(s, seed[, out map])`, `ResampleInPlace(clone, seed)` |
| `S0.Sim/SearchCardSelection.cs` | `SearchCardSelectionSource` (card prompts inside clones), `CardSelectionOptions.Enumerate` |
| `S0.Sim/BattleStart.cs` | `BattleStartSpec` (JSON), `BattleStartLoader.LoadAsync / LoadMatchedAsync / PlayLiveAsync` |
| `S0.Sim/CombatActions.cs` | positional legal actions (`CombatActions`), observable-state fingerprint (`ObservableState`) |
| `S0.Sim/CombatObservation.cs` | draft combat-network token encoder (deliverable 4) |
| `S0Tests/Program.cs` | console test runner: `resample`, `choices`, `import` |
| `export_battle_starts.py` | Spire Codex runs -> battle-start JSON lines (uses `value_v1/build_floor_dataset.reconstruct`, read-only) |
| `data/` | `combat_vocab_v1.json` (ids used by the network); exported battle starts are run-derived and supplied separately. |
| `out/` | test outputs quoted below |
| `src/sims/patches/iryougi_s0.patch` | changes to the iRyougi clone (`git diff` against db844ad + the local patch) |

## Changes to iRyougi (`src/sims/patches/iryougi_s0.patch`)

1. `Sts2Sim.Core.csproj`: `InternalsVisibleTo` for `Sts2Sim.S0` and `S0Tests` (needed for the transplant
   importer pieces and the new internal pile methods).
2. `CardPile`: draw-pile position knowledge. Two counters, `KnownTopCount` / `KnownBottomCount`: the number of
   cards at the top / bottom whose position the player knows. A deterministic insert at or inside the known top run
   extends it (Headbutt, Thinking Ahead, Innate at combat start, ...); an append extends the bottom run; a random
   reorder or a random-position insert forgets everything; an insert into the unknown middle is not tracked;
   removals shrink the run they come from. New internal `ShuffleUnknownRegionInternal(rng)` and
   `CopyPositionKnowledgeFrom`. Game behaviour is unchanged (only counters are added; `Remove` keeps its semantics).
3. `CardPileCmd`: the three insert sites pass `randomPosition: position == Random`; `Shuffle` resets the knowledge.
4. `Player.PopulateCombatState`: resets the knowledge after the initial shuffle (before the Innate moves).
5. `PlayerCombatState.CloneForCombat`: clones copy the counters.

Apply on a fresh clone: `git checkout db844ad && git apply src/sims/patches/iryougi_local.patch && git apply src/sims/patches/iryougi_s0.patch`.

## 1. Resampling clone

`ResamplingClone.Resampled(state, seed)` = `state.Clone()` followed by, in place on the clone's own generator
objects (counters kept, so every reference in the clone graph sees the new stream):
every `RunRngType` stream, every `PlayerRngType` stream of every player, each monster's private `Rng`, and a
Fisher-Yates reshuffle of the draw-pile cards between the known top and known bottom runs. Refuses a non-clone
state and keyed (semantic-key) RNG mode.

Supported / not supported:
- Known positions: deterministic top/bottom placement since the last shuffle (tracked by the patch). Not tracked:
  cards inserted deterministically into the unknown middle (shuffled, conservative), knowledge after a random insert
  (forgotten, conservative), information revealed by "first card of type X" draws or by enchantments that reorder
  a reshuffle (e.g. PerfectFit at index 0). None of these leak; they only lose information.
- Not resampled: the relic grab bags (setter private; out of combat only); the seed that iRyougi derives for
  monsters spawned later in the fight (`RunState.Rng.Seed` + location + combat id; the monster private Rng is not
  read by any combat code at this commit, so this is not a leak path today).

## 2. Card choices inside clones

`SearchCardSelectionSource.InstallOn(clone, mode[, policy])`. Prompts are answered from `Script` first, then by mode:
- `Policy`: a callback picks one of `CardSelectionOptions.Enumerate(request)` (rollouts; `RandomPolicy(rng)`).
- `Suspend`: throws `SelectionPendingException` (a `DecisionSuspendedException`) carrying the request and the option
  list; discard that clone and replay the action on a fresh clone with the chosen option queued in `Script`. This
  makes (card, target, option) a compound search action.
- `Pause`: the engine call returns an unfinished `Task`; `Pending`/`PendingOptions` expose the prompt; `Resolve(option)`
  continues the same clone synchronously.
Options are index sets over the candidates with interchangeable cards (same id, upgrade, cost, enchantment,
affliction) merged, sizes MinCount..MaxCount, plus "nothing" when cancelable; capped at `MaxOptions` (256).
Run-level prompts inside a clone (deck transform/upgrade/remove, bundle screens) still use iRyougi's deterministic
default; none occurred in the tests.

## 3. Battle-start import

`export_battle_starts.py` writes one JSON object per fight: deck (id, upgrade, enchantment, amount), relics,
potions, HP / max HP / gold after floor g-1, ascension, act ids and index, room type, encounter id and logged
monster ids of floor g. `BattleStartLoader` builds an injected-combat `RunTransplantSnapshot` from it (RNG states,
odds and relic bags from a fresh run with the simulator seed), imports it with iRyougi's `RunTransplantImporter`,
runs `RunDriver.RunOneInjectedCombatAsync`, and returns a clone of the state at the player's first decision
(`LoadAsync`). `LoadMatchedAsync` retries simulator seeds until the starting enemies equal the logged monster ids.
Not imported: relic counters (no data in the logs), the real RNG state (fights are not the logged fights draw for
draw), potion-slot count (taken as max(default for the ascension, potions held)).

## 4. Observation encoder (draft)

`CombatObservation.Encode(state, maxTokens)`: kind-tagged tokens (global, hand, draw/discard/exhaust multisets with
counts, enemies with intent type / damage / hits, enemy powers, player powers, orbs, potions, relics), each with a
primary and secondary id string and 8 integer features; layout in the class comment. Not yet tied to a Python
model or vocabulary; no round-trip test.

## Rebuild and rerun

```bash
W=$(pwd)  # repository root; cd $W
export NUGET_PACKAGES=$W/.cache/nuget DOTNET_CLI_HOME=$W/.cache/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$W/.cache/tmp
dotnet build src/s0/S0Tests/S0Tests.csproj -c Release -m:8
T=src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll
dotnet $T resample 60 10000 L 3000        # deliverable 1 (a)-(d) + encoder check, ~1 min
dotnet $T choices 10 10 4 B 3000          # deliverable 2, ~1.5 min
./.venv/bin/python -I src/s0/export_battle_starts.py --max 1000 --per-run 2 \
    --out src/s0/data/battle_starts_v0.111.0_defect_1000.jsonl
dotnet $T import src/s0/data/battle_starts_v0.111.0_defect_1000.jsonl 1000   # deliverable 3
```

## Results (laptop, 2026-11-11; full outputs in `out/`)

- **(a)** exact clone vs original, one real action later (real card choices replayed): 10,548 comparisons, 0 full-digest
  mismatches. Two exact clones under the same 5 random actions: 46,791 steps, 0 mismatches.
- **(b)** 10,000 states: observable fingerprint and encoder output identical for original and resampled clones in all;
  draw order differs between two seeds in 8,767/9,180 states with >= 2 unknown cards (rest: identical cards);
  next hand differs after end turn in 8,884/9,620. Known positions (Headbutt in a clone): top card kept in 300/300
  cases x 10 seeds and drawn 3,000/3,000; a random insert afterwards cleared the knowledge 296/296.
- **(c)** next drawn card, n = 9,963: resampled == exact 20.57% vs posterior expectation 21.12% (z = -1.63),
  resampled == resampled 21.43%. Negative control (RNG reseeded, draw pile not reshuffled): 96.87%.
  End-turn variant (whole next hand): resampled == exact 7.91%, resampled == resampled 7.65%.
- **(d)** single thread, 50,000 clone+action pairs interleaved: clone 25.6 us, resample 1.0 us, action ~110 us;
  7.2k/s exact vs 7.4k/s resampled (no measurable cost; difference is noise).
- **Choices** (10 runs, K=10, D=4, 67,530 rollouts per mode): 3.16% errors without the source (all "non-forced combat
  card selection"), 0 with random choices (2,174 prompts). Suspend+replay and Pause/Resolve reproduce the Policy result
  exactly in 2,145/2,145 prompting actions.
- **Import** (1,000 Defect v0.111.0 battle starts from 500 runs): 998 imported and played to the end (random moves,
  live RunDriver path also 998/998); 2 failures = modded relic `RELIC.ARCHIPELAGO_RELIC_RELIC_COUP`. Starting enemies
  equal the logged monster ids in 924, subset in 74 (all summoner encounters, the log lists summons). 58 HP and 9 potion
  differences at the first decision come from combat-start relics (Blood Vial, Petrified Toad, ...).
- iRyougi's own tests touching piles, clones, shuffles, draws, innate, transplant and determinism (filter in the report):
  1,774 passed, 0 failed with the patch applied.

# S0 part 2 (2026-11-11)

New files: `S0.Sim/ActionSpace.cs`, `S0.Sim/Search.cs`, `S0.Sim/ObsLayout.cs`, `S0.Sim/Bridge.cs`,
`S0Tests/{ActionTests,M1,EncodeExport,BridgeBench}.cs`, `py/{combat_obs.py,test_roundtrip.py,bridge_server.py}`,
`data/combat_vocab_v1.json`. No further iRyougi changes (`src/sims/patches/iryougi_s0.patch` unchanged).

## Action space (`ActionSpace`)
`ActionSpace.LegalAsync(state, includePotions, expandSelections)` / `Base(state)` return `SearchAction`s: card plays
(hand card by equivalence key x legal targets, RunDriver's rule), potions (every slot x {none} ∪ all creatures where
`PotionCmd.CanUseManually`), end turn, and with expansion one action per prompt option (Suspend probe on an exact
clone, nested up to 2 prompts). Actions are identity-addressed (`Key`: `C:<card key>>t<combatId>|S:<chosen keys>`,
`P:<potion id>>...`, `E`; `Hash` = FNV-64) so they apply in any resampled world; `ApplyAsync` resolves them and feeds
the prompt answers through `SearchCardSelectionSource.KeyScript`. Transpositions: key nodes by `ObservableState.Hash`.

## M1 search (`GumbelSearch`)
Gumbel-top-k root (k = 8, uniform prior), sequential halving with sigma(q) = (50 + max N) q; a fresh
`ResamplingClone.Resampled` world per simulation; open-loop tree over action keys (SO-ISMCTS UCB with availability
counts, c = 0.7, one node expanded per simulation); random rollout (cards + end turn, p_end = 0.1, no potions) to the
end of the fight, capped at 50 turns from the root (capped rollouts scored on their state). Prompt options are root
actions; below the root prompts are random. Scorers: `HpScorer` (end HP / max HP, 0 on death) and `TfScorer`
(table file; subset = belt slots still holding their starting potion; T at end HP; 0 on death).
Runner: `S0Tests m1 <battle_starts> <first> <count> <sims> <hp|tf> <tf_tables> <compare 0|1> <out.jsonl> [seed]`.

## Layout v1, vocabulary, Python side
`ObsLayout` freezes the int32 layout (128 tokens x [kind, id, id2, f0..f7]; 64 actions x [kind+1, source token,
target token, prompt answers]); `combat_vocab_v1.json` lists every card/enchantment/monster/intent/power/orb/potion/
relic id of the simulator with `<pad>`=0 and `<unk>`=1. `py/combat_obs.py`: decoder, `CombatObsDataset`, `CombatNet`
(677k parameters: token transformer, pointer-style policy over the action rows, P(survive) and relative-T_f heads).

## Bridge (`Bridge.cs` + `py/bridge_server.py`)
Protocol and layout in the `BridgeClient` comment: one shared-memory file per actor in /dev/shm, header with magic,
protocol/layout versions, sizes and model version; SPSC ping-pong on request/response sequence numbers (release /
acquire in C#, x86 store order on the Python side); the server echoes the model version that answered; weights are
re-loaded between batches when the checkpoint file's mtime changes (write + rename).

## Commands
```bash
T=src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll
dotnet $T actions 6 2000 10 4 P 3000                     # action space, ~40 s
dotnet $T encode 20 1000 src/s0/data/obs_v1_1000     # 1,000 encoded states (vocab is reused if present)
./.venv/bin/python -I src/s0/py/test_roundtrip.py   # decode + compare + CombatNet forward (CPU)
./.venv/bin/python -I src/s0/py/bridge_server.py --path /dev/shm/s0_bridge &   # GPU
dotnet $T bridge /dev/shm/s0_bridge src/s0/data/obs_v1_1000 2 64 256 1024
dotnet $T m1 src/s0/data/battle_starts_v0.111.0_defect_1000.jsonl 0 6 200 tf \
    src/s0/data/tf_tables_v0.111.0_defect_1000.jsonl 1 src/s0/out/m1_tf200_compare_6.jsonl
```
