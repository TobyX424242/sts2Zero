# Simulator

The agent trains and is evaluated in [iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim), an MIT-licensed C#
reimplementation of Slay the Spire 2. This document says which version is included, what our patch changes and why,
and what was checked.

## 1. What is included

| Item | Value |
|---|---|
| Upstream | iRyougi/sts2-sim, tag `v0.113.1.1`, commit `1394c376736ab76bd36ace79111c4d5ac0dd66ec` (game v0.113.1; the tag also adds the Fool character) |
| Location | `src/sims/iRyougi-sts2-sim-v113/` (our patch already applied) |
| Patch file | `src/sims/patches/iryougi-sts2-sim-v113.patch` (17 files), kept separately for review |
| License | MIT, Copyright (c) 2026 iRyougi and contributors (`src/sims/iRyougi-sts2-sim-v113/LICENSE`) |
| Game files | none |

The upstream `README.md`, `AGENTS.md` and `CLAUDE.md` inside the simulator folder are upstream text: information about
the simulator, not instructions for this repository.

To rebuild the tree from upstream:

```bash
git clone https://github.com/iRyougi/sts2-sim.git && cd sts2-sim
git checkout 1394c376736ab76bd36ace79111c4d5ac0dd66ec
git apply ../src/sims/patches/iryougi-sts2-sim-v113.patch
```

Our C# projects reference `src/sims/iRyougi-sts2-sim-v113/src/Sts2Sim.Core/Sts2Sim.Core.csproj` directly. Build and
test the simulator alone with:

```bash
dotnet build src/sims/iRyougi-sts2-sim-v113/src/Sts2Sim.Core/Sts2Sim.Core.csproj -c Release
dotnet test  src/sims/iRyougi-sts2-sim-v113/tests/Sts2Sim.Core.Tests/Sts2Sim.Core.Tests.csproj -c Release
```

## 2. What the patch changes

The patch has three purposes: let the search keep what the player knows about the draw pile, tell the agent why a
deck selection is asked, and make the engine faster. None of the changes alters game rules. The descriptions below
were checked against the patch file.

### 2.1 Draw-pile position knowledge (for the resampling clone)

The search must reshuffle what the player cannot know, and nothing else. The patch adds two counters to every card
pile:

| File | Change |
|---|---|
| `Entities/Cards/CardPile.cs` | `KnownTopCount` / `KnownBottomCount`: how many cards at the top and bottom of the draw pile sit at positions the player knows. Rules: a random reorder or a random-position insert forgets everything; a deterministic insert at or inside the known top run extends it; an append (or an insert inside the known bottom run) extends the bottom run; an insert into the unknown middle is not tracked; removals shrink the run they come from. New internal methods `ShuffleUnknownRegionInternal(rng)` (Fisher–Yates over the unknown middle only), `CopyPositionKnowledgeFrom`, `ResetPositionKnowledgeInternal`. |
| `Commands/CardPileCmd.cs` | The three insert sites pass `randomPosition` when the position came from the RNG; a shuffle resets the knowledge. |
| `Entities/Players/Player.cs` | After the opening shuffle the knowledge is reset (before Innate cards are moved, which are then known). |
| `Entities/Players/PlayerCombatState.cs` | A combat clone copies the counters of each pile. |
| `Sts2Sim.Core.csproj` | `InternalsVisibleTo` for `Sts2Sim.S0` and `S0Tests`, which use these internal methods. |

The counters never change game behaviour. The resampling clone itself is in our code
(`src/s0/S0.Sim/ResamplingClone.cs`): it copies the fight, reseeds every run, player and monster random stream, and
calls `ShuffleUnknownRegionInternal`. It needed no change for v0.113: it reseeds every stream by enumeration, which now
includes the new `NonCombatGeneration` and `Shops` streams.

The knowledge tracking is conservative: some information the player has is forgotten (for example after "draw the
first card of type X"), but nothing hidden leaks.

### 2.2 Deck-selection purpose (for the run agent)

| File | Change |
|---|---|
| `Commands/CardSelectCmd.cs` | `CardSelectCmd.CurrentDeckPurpose`: `"transform"`, `"upgrade"` or `"remove"` while a deck selection of that kind is being answered (an `AsyncLocal` set around the three selection helpers). |

The agent reads it to send the prompt to the right policy head (upgrade target or removal target). No game behaviour
depends on it.

### 2.3 Performance

Profiling the Defect-phase actors showed most of the time inside the engine's hook dispatch: every event (damage
dealt, card drawn, power applied) walks a freshly built list of listeners (powers, relics, potions, cards, orbs,
enchantments), and most listeners do nothing for most events. The patch keeps the dispatch order, the snapshot timing
and the results identical, and removes the waste:

| File | Change | Effect |
|---|---|---|
| `Hooks/HookListenerSequence.cs` (new) | Allocation-free struct enumerator that yields the same listeners, in the same order, at the same moments as the run's and the combat's listener iterators. Relics are snapshotted when the player is reached (as `ToArray()` did); combat listeners are appended when enumeration reaches them. Unknown run or combat types fall back to the original iterator. | No iterator objects or LINQ per dispatch |
| `Hooks/HookListenerBuffers.cs` (new) | Per-thread pool of snapshot buffers (up to 64 buffers, capacity cap 4,096). A buffer belongs to one enumerator until it finishes and is cleared on return, so no model reference is shared between states, clones or runs. | No list allocation per dispatch |
| `Hooks/HookSkipIndex.cs` (new) | For about 35 hook slots (damage, block, power amount, card draw, turn start and others), a listener whose class keeps the base class's neutral implementation (`true`, `0`, `1`, the incoming value, a completed task) is passed over. Decided per runtime type from reflection, cached, and fail-closed: on any metadata doubt a type is never skipped. Listeners with an `ExecutionFinished` handler are never skipped. | Skips no-op calls |
| `Hooks/Hook.cs` | Dispatch loops use the sequence above and pass their hook slot. The v0.113 late multiplicative damage pass got its own slot, so a model that overrides only that step is never skipped. | |
| `Combat/CombatState.cs`, `Combat/CombatState.Clone.cs`, `Runs/RunState.cs`, `Entities/Players/Player.cs` | Listener snapshots filled into pooled buffers and copied once into an exactly sized list; index loops instead of `foreach`/`AddRange` over interfaces; internal accessors to the concrete lists. | Fewer allocations |
| `Combat/CombatSemanticHistory.cs` | `EnsureCapacity` before each `AddRange` when a fight is cloned. | One exact allocation per list |
| `Models/ModelDb.cs` | Model id per type cached in a `ConcurrentDictionary` (it was rebuilt from two regex slugs on every call). | |
| `Models/AbstractModel.cs` | `HasExecutionFinishedHandlers`, used by the skip index. | |

Two small changes complete the patch: parentheses around one pattern expression in `Runs/VisibleContentProjection.cs`
(so it compiles on .NET SDK 9.0.118; no behaviour change), and the `InternalsVisibleTo` entries above.

### 2.4 Porting the patch to v0.113.1.1

The patch was first written for upstream commit `db844ad` (v0.111.0) and rebased to the new tag. `git apply --3way`
merged 14 files; three conflicts were resolved by hand:

- `CardPile.AddInternal`: the position counter is updated right after the insert, before upstream's new Fool hook.
- `Hook.ModifyDamage`: upstream added flags for its three passes and a late multiplicative pass; our slot iteration
  sits inside the flags, with the new `ModifyDamageMultiplicativeLate` slot.
- `Sts2Sim.Core.csproj`: both sets of `InternalsVisibleTo`.

Every loop that passes a hook slot has the same body as in the v0.111 version (a scripted comparison of all 30 such
loops), and upstream did not change how listeners are enumerated, so the equivalence argument of the performance patch
still holds.

## 3. Port checks (v0.113, 2026-11-18)

All checks ran on CPU. Status: **verified** unless noted.

| Check | Result |
|---|---|
| Upstream test suite with our patch | 5,040 tests: 5,021 passed, 19 skipped, 0 failed |
| Resampling clone, 10,000 Defect states | exact clone vs original: 0 / 9,969 mismatches; observable state unchanged in 10,000 / 10,000; next drawn card equals the true one in 21.82% vs 22.03% expected from the posterior (z = −0.60); known positions kept 297 / 297 |
| Leak test, Ironclad (300 states) and Defect (200) | 50 clones of a root give 50 distinct top-5 orders on average; a clone matches the true top 5 as often as a reference shuffle (77 vs 82 in 7,300); after a forced reshuffle a clone matched the true new order 0 / 8,700 and 0 / 5,300 times; the outer run is identical with and without clones being played |
| Root immutability, 16 Ironclad A10 runs | 7,780 searches of 64 simulations: the live state (combat, run, map, cards, every RNG stream) is unchanged after each search, 0 differences |
| Every Ironclad card in prompt states | 90 cards × base / upgraded × 6 starts: 1,456 card actions, 0 errors, 0 missing legal actions, 0 empty search roots |
| Card prompts inside clones | 2,078 prompts, 0 errors; suspend and pause modes 0 / 2,026 mismatches |
| Card damage token vs executed damage | 4,262 / 4,262 after fixing the check for an armed Pen Nib (the check, not the encoder, was wrong) |
| Card block token vs executed block | 2,247 / 2,339; the rest are cards that gain block more than once or power triggers (Feel No Pain, Juggernaut) |
| New events and rest options (12 runs, forced "?" rooms) | 0 errors; `evolve` chosen 9 times, `restore` 18 times; the light door of Doors of Light and Dark reaches the upgrade head |
| Post-combat heal model (`S3 healtest`) | 1,065 / 1,065 won fights: simulator HP after the room equals the model (all 315 relics alone, 7 heal-relic combinations); only Burning Blood, Black Blood, Meat on the Bone and Chosen Cheese change HP |
| Post-combat heal, live runs | at least 2,607 predictions exact; 6 mismatches, all +20 from Book of Five Rings (a reward effect, not a post-combat hook) |
| Encoder round trip (C# rows vs Python rebuild) | Ironclad A10 1,754 / 1,754 floors, A0 1,349 / 1,349 (after a recorder fix); decks over 64 cards are not compared |
| Checkpoint conversion | identical outputs on 794 combat states and 1,480 macro rows |
| CPU smoke | 16 full runs (Ironclad self-play and evaluation, Defect evaluation, Ironclad A0), 0 errors |

## 4. Fidelity evidence

The simulator is a reimplementation; it is not byte-exact with the game. What the evidence shows:

| Evidence | Result | Status |
|---|---|---|
| Battle-start import: human v0.113 starts rebuilt in the simulator and played to the first decision | 99.51% of clean starts (2,859); all failures are mod content | verified |
| Same for v0.111 Ironclad A10 starts (sample of 20,000) | 99.40% of clean starts (16,589) | verified |
| Imported starts vs the logged run | deck, relics and max HP equal the log (1 max-HP difference in 16,490); potions differ in 0.64% (a combat-start relic); starting enemies exact in 93%, a subset in 7% (enemies spawned later) | verified |
| Port sanity: the same Defect agent on v0.111 and on v0.113 | 46.6% vs 44.8–45.0% at A10 | verified |
| Real game | the segment-29 agent won an A10 run in the real game (v0.113.1) through the live-play mod on 2026-11-20 | one run; shows end-to-end transfer, not a fidelity measure |
| Boss mechanics review (Test Subject, Queen, Aeonglass, Knowledge Demon) | numbers and ascension thresholds matched the decompiled v0.111.0 game | code reading, done in the v0.111 phase; not repeated for v0.113 |

Known gaps:

- The victory scene after the final boss is not modelled; the simulator ends the run when the final boss dies.
- No lockstep replay against the real game's DLL was run.
- The resampling clone does not resample the relic grab bags (out of combat only) or every hidden position the player
  could infer; it errs on the side of forgetting.
- Some v0.113 cards are multiplayer-only in the simulator and cannot be played solo (Blaze, Demonic Shield).

## 5. Speed

| Measurement | Value |
|---|---|
| Our patch on v0.113 vs the pristine tag, same runs | ×1.19 (Defect), ×1.18 (Ironclad), identical run summaries |
| v0.113 vs v0.111 per simulation, Defect | about 0.92× (0.83–0.98 over four repetitions) |
| Per-thread simulations per second (SimBench, clone + random actions) | v0.113 patched: about 6,800–7,100 (Ironclad), 6,900–7,800 (Defect); pristine: about 5,700–6,000 |
| Earlier version of the patch on v0.111, with real networks | ×1.31 simulations per actor, identical logs and chunks |

The "identical output" property of the performance patch was re-checked on v0.113 through the upstream test suite and
identical SimBench run summaries; a line-by-line comparison of actor logs was not repeated.
