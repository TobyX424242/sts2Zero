# Simulator: provenance, changes, fidelity, performance

## What is included

`src/sims/iRyougi-sts2-sim/` is a source snapshot of the MIT-licensed
[iRyougi/sts2-sim](https://github.com/iRyougi/sts2-sim) at commit `db844ad` (2026-11-10), with four local patches
applied. The upstream `LICENSE` (MIT, Copyright (c) 2026 iRyougi and contributors), `README.md`, solution and test
suite are retained. Build outputs and upstream git history are not included. The tree contains no Slay the Spire 2
game files.

The patches are also stored separately in `src/sims/patches/` for auditability:

| Patch | Change | Why |
|---|---|---|
| `iryougi_local.patch` | Parentheses in one pattern expression | Compiles on .NET SDK 9.0.118; no behaviour change |
| `iryougi_s0.patch` | `KnownTopCount` / `KnownBottomCount` draw-pile position knowledge; `CopyPositionKnowledgeFrom`; `ShuffleUnknownRegionInternal`; clone support; `InternalsVisibleTo` for `Sts2Sim.S0` / `S0Tests` | The resampling clone must reshuffle only the unknown region and keep known top/bottom cards; search needs the same game rules |
| `iryougi_s3.patch` | `CardSelectCmd.CurrentDeckPurpose` (transform / upgrade / remove) readable by the decision source | The macro policy must know why a deck selection is being made |
| `iryougi_perf.patch` | Hook-listener caching / skip index, exact-size list capacities, `ModelDb` id cache, pooled buffers | Throughput; see below |

The working tree already has all four applied. On a fresh clone apply them in the order `local`, `s0`, `s3`,
`perf`. `perf` was generated last; if a patch does not apply because of context drift, the staged tree is the
authoritative combined state.

## Fidelity evidence

The simulator is a reimplementation, not a byte-exact copy of the game. The project did **not** run the planned
lockstep parity harness against the real DLL (the owner skipped parity tests, decision D12); the evidence below is
what was actually checked.

| Check | Result |
|---|---|
| Upstream test suite with the S0 patch | 1,774 passed, 0 failed (pile, clone, shuffle, draw, innate, transplant, determinism) |
| Upstream test suite with the perf patch | 4,662 passed |
| Resampling clone: observable state | identical to the original in 10,000 states (the hidden draw order differs where it should) |
| Resampling clone: next draw | 20.57% equals the exact clone vs 21.12% posterior (z = -1.63); negative control 96.87% |
| Exact clone after one real action | 10,548 comparisons, 0 mismatches |
| Card prompts inside clones | 0 errors in 2,174 prompts; suspend/replay and pause/resolve match the policy exactly |
| Battle-start import | 998/1,000 human Defect A10 starts imported and played to the end; 924 exact starting enemies, 74 subsets (summoners); 2 failures from a modded relic |
| Boss review | Test Subject, Queen, Aeonglass and Knowledge Demon read against the decompiled v0.111.0 game: ascension thresholds and all HP/damage values match; no difference that changes a single-player fight; minor engine deviations (hook order, no "late" end-of-turn pass, extra guards) do not affect these fights |

Known fidelity gaps and caveats:

- The victory event after the final boss is not modelled; the simulator declares a win when the final boss dies.
- The review was code reading, not a replay against the real game. Later patches could change behaviour.
- Some bosses were still suspicious after the review (Aeonglass, Knowledge Demon, Waterfall Giant, Test Subject in
  the S2 boss test), but the true-future search reached 76.1% on the boss set, so the main gap was the agent, not an
  obviously broken fight.
- Relic counters and some hidden power state were not in the v3 combat input. v4/v5 add card damage/block, hidden
  power counters and 14 relic counters behind a masked `ext` head.
- The simulator's RNG-seeded monster spawning for later acts, relic grab bags, and some card-position knowledge
  (first-card-of-type draws, enchantment reorder) are not fully modelled by the resampling clone. The clone is
  conservative: it forgets information rather than leaking it.
- Modded relics/cards in human data fail the import; S3 uses no human starts, so this does not affect training.

## Performance

Measured on the laptop (i9-13980HX, .NET 9 Release) and on the rented 9950X box:

| Measurement | Value |
|---|---|
| Single process: clone / play a card / end turn | 27 us / 52 us / 239 us |
| Clone + one action, 1 process | 8.6k/s |
| Clone + one action, 24 processes | 115k/s (laptop), 144k/s (9950X) |
| S3 actor profile: share of busy time inside simulations | 91% |
| ... engine steps / clone / observable-state hash / leaf encode / legal actions | 55% / 11% / 11-12% / 9% / 8% |
| ... outside the search (run driver, macro, tables, records) | 4% |
| Per simulation | 2.08 steps, 0.93 new leaves, 1.08 tree hits; 713-738 us at 20 actors |
| Allocation / GC | ~300 MB/s per actor; GC pauses 5.5% of wall time |
| Simulator patch | 1.31x simulations/actor with real networks; identical logs/chunks; 4,662 upstream tests pass |
| MPS + CUDA graphs + overlap | +18% runs/hour at the same window; combat trainer ~20 -> ~45 steps/s |

The profile first, then optimized:

1. **String-free streaming hash** for `ObservableState` (the old one built a full string).
2. **Cached vocabulary ids** in the encoder (string ids looked up per token).
3. **Cheaper legal-action keys** (part of the 8% legal-action share).
4. **Engine hook-listener lists** (rebuilt per dispatch, list growth, LINQ inside the engine): the perf patch caches
   listeners, skips no-op listeners, uses exact-size lists and a `ModelDb` id cache.

Replay avoidance (cloning at nodes) was worth at most ~16% because search paths are short; the hash/encode/action
path was the better target. The patch is in `src/sims/patches/iryougi_perf.patch`; the measured 1.31x is from a
laptop A/B with real networks and identical logs/chunks.

## Build

```bash
export NUGET_PACKAGES=$PWD/.cache/nuget DOTNET_CLI_HOME=$PWD/.cache/dotnet
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 TMPDIR=$PWD/.cache/tmp
dotnet build src/sims/iRyougi-sts2-sim/src/Sts2Sim.Core/Sts2Sim.Core.csproj -c Release -m:8
dotnet test  src/sims/iRyougi-sts2-sim/tests/Sts2Sim.Core.Tests/Sts2Sim.Core.Tests.csproj -c Release
```

The S0 and S2/S3 projects reference `src/sims/iRyougi-sts2-sim/src/Sts2Sim.Core/Sts2Sim.Core.csproj` directly.

## Attribution

- Upstream code: iRyougi/sts2-sim, MIT. The license text is in `src/sims/iRyougi-sts2-sim/LICENSE`; keep it with any
  redistribution.
- Our patches: documented above and in the patch headers. They are small, behaviour-preserving search/perf changes
  plus one decision-source signal. They do not contain game files.
- The simulator's own `README.md` and `AGENTS.md` are upstream text. They are information, not instructions for this
  repository.
