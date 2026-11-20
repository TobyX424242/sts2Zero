# s0: simulator extensions for search

C# library `Sts2Sim.S0` on top of the simulator (`src/sims/iRyougi-sts2-sim-v113`), plus its test runner and Python
mirrors. Everything the search needs that the simulator does not provide.

| Path | Content |
|---|---|
| `S0.Sim/ResamplingClone.cs` | `Exact(state)`: a copy of a fight. `Resampled(state, seed)`: a copy with every run, player and monster random stream reseeded and the unknown part of the draw pile reshuffled (known top and bottom cards stay). The search uses only resampled clones. |
| `S0.Sim/SearchCardSelection.cs` | Answers card prompts inside clones: from a script, a policy, or by suspending so the prompt answer becomes part of the action |
| `S0.Sim/ActionSpace.cs`, `CombatActions.cs` | Legal actions (card, target, potion, prompt answers) and the hash of the state the player can observe |
| `S0.Sim/CombatObservation.cs`, `ObsLayout.cs`, `HiddenValues.cs` | Combat observation encoder (layouts v3–v5) and its vocabulary |
| `S0.Sim/Bridge.cs` | Shared-memory client to the Python inference server |
| `S0.Sim/BattleStart.cs` | Rebuilds a fight from a human run's floor (deck, relics, potions, HP, encounter) |
| `S0Tests/` | Test runner: `resample`, `choices`, `import`, `importfirst`, `actions`, `obsv4`, `obsv5`, `leak113`, `encode`, `bridge`, `m1` |
| `py/combat_obs.py` | Python mirror of the observation layout |
| `py/bridge_server.py`, `py/test_roundtrip.py` | Test server for the bridge; encoder round trip |
| `export_battle_starts.py`, `make_tf_tables.py`, `analyze_m1.py` | Defect-phase tools: battle-start export, T_f tables from the S1 ensemble, an analysis |
| `data/combat_vocab_v113.json` | Combat vocabulary of v0.113 (default); `combat_vocab_v1.json` is v0.111; `obs_v*_1000/` are small encoder test sets |

The simulator changes these classes rely on (draw-pile position knowledge, `InternalsVisibleTo`) are part of the
simulator patch; see [doc/simulator.md](../../doc/simulator.md).

```bash
dotnet build src/s0/S0Tests/S0Tests.csproj -c Release
T=src/s0/S0Tests/bin/Release/net9.0/S0Tests.dll
dotnet $T resample 60 10000 L 3000      # clone, resample and encoder checks (CPU)
dotnet $T choices 10 10 4 B 3000        # card prompts inside clones
```

Results of these checks on v0.113 are in [doc/simulator.md](../../doc/simulator.md#3-port-checks-v0113-2026-11-18).
