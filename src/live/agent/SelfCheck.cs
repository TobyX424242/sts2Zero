using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Merchant;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Events;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Rewards;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S3;

namespace Sts2Sim.Live;

/// <summary>
/// Test without the game: simulator runs played by an ordinary RunAgent ("true" agent, as in self-play), where every
/// decision is first exported as a protocol state (<see cref="FakeGame"/>: run.save in the game's JSON names, combat
/// snapshot, options) and run through the live controller (import, RunAgent bound to the import, shadow fight sync,
/// mapping back to an option). The true agent's choice is what the run then plays. Checked per decision:
/// <list type="bullet">
/// <item>macro decisions: the macro row RunAgent encodes from the imported run equals the row of the true run (every
/// differing slot is reported by its layout name); the live answer is an option of the right kind; the live choice
/// equals the true agent's where both are deterministic;</item>
/// <item>fights: the shadow fight after sync shows the same observable state as the true fight (S0 ObservableState
/// fingerprint: piles, energy, HP, block, powers with hidden counters, intents with damage, potions, relic counters;
/// creature ids normalised); the agent's action maps to a protocol option (and its target);</item>
/// <item>combat and deck card prompts: the answer maps to the offered cards.</item>
/// </list>
/// keys: runs (2), asc (10), seeds (prefix, LIVECHECK), max_floors (60), sims (search budget in this test, 8), tank (1: the
/// player starts with 5000 HP so that uniform-network runs reach acts 2-3, bosses, shops and events), dump (FILE: the exported
/// states as JSONL, readable by `replay file=FILE`).
/// </summary>
public static class SelfCheck
{
    public sealed class Stats
    {
        public readonly Dictionary<string, long> N = new(), Bad = new();
        public readonly Dictionary<string, long> RowDiff = new(), FpDiff = new(), Notes = new();
        public readonly List<string> Examples = new();
        public void Add(Dictionary<string, long> d, string k, long v = 1) => d[k] = d.GetValueOrDefault(k) + v;
        public void Example(string s) { if (Examples.Count < 60) Examples.Add(s); }
    }

    public static async Task<int> RunAsync(AgentSetup s)
    {
        int runs = s.GetI("runs", 2), asc = s.GetI("asc", 10), maxFloors = s.GetI("max_floors", 60);
        string prefix = s.Get("seeds", "LIVECHECK");
        // a small search budget: the test is about import, sync and mapping, not about play strength
        s.Cfg.SimsFull = s.Cfg.SimsFast = s.GetI("sims", 8);
        var st = new Stats();
        CheckEventKeyMapping(st);
        FakeGame.DiscardOption = s.GetI("discard_option", 0) == 1;
        if (s.Kv.TryGetValue("dump", out string? dumpPath)) FakeGame.Dump = new StreamWriter(dumpPath) { AutoFlush = true };
        for (int i = 0; i < runs; i++)
        {
            string seed = $"{prefix}{i + s.GetI("first", 0):D4}";
            var acts = Sts2Sim.Core.Content.ActDefinition.GetRandomList(seed);
            var run = new RunState(seed, acts, asc);
            var player = Player.CreateForNewRun(s.Character, run);
            run.AddPlayer(player);
            if (s.GetI("tank", 0) == 1) { player.Creature.SetMaxHpInternal(5000); player.Creature.SetCurrentHpInternal(5000); }   // deep runs for the test
            var trueAgent = new RunAgent(s.Services, run, player, seed, false, asc);
            var ctl = new Controller(s, m => st.Add(st.Notes, "say: " + (m.Length > 60 ? m[..60] : m)));
            var check = new FakeGame(trueAgent, run, player, ctl, st);
            var driver = new RunDriver(run, check);
            driver.OnRoomResolved += trueAgent.OnRoomResolved;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            string result;
            try { var res = await driver.RunAsync(maxFloors); trueAgent.Finish(res); result = res.Outcome.ToString(); }
            catch (Exception e) { trueAgent.Fail(e); result = "error " + e.GetType().Name + ": " + e.Message; st.Add(st.Bad, "run_error"); st.Example($"run {seed}: {e}"); }
            Console.WriteLine($"[selfcheck] run {seed} A{asc}: {result}, floor {run.TotalFloor}, decisions checked {check.Checked}, {sw.Elapsed.TotalSeconds:F0} s");
        }
        Console.WriteLine("[selfcheck] decisions checked: " + string.Join(", ", st.N.OrderBy(x => x.Key).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("[selfcheck] problems: " + (st.Bad.Count == 0 ? "none" : string.Join(", ", st.Bad.OrderByDescending(x => x.Value).Select(x => $"{x.Key} {x.Value}"))));
        Console.WriteLine("[selfcheck] macro row slots that differ (decisions): " + (st.RowDiff.Count == 0 ? "none" : string.Join(", ", st.RowDiff.OrderByDescending(x => x.Value).Take(40).Select(x => $"{x.Key} {x.Value}"))));
        Console.WriteLine("[selfcheck] combat fingerprint lines that differ (decisions): " + (st.FpDiff.Count == 0 ? "none" : string.Join(", ", st.FpDiff.OrderByDescending(x => x.Value).Take(40).Select(x => $"{x.Key} {x.Value}"))));
        Console.WriteLine("[selfcheck] notes: " + string.Join(", ", st.Notes.OrderByDescending(x => x.Value).Take(60).Select(x => $"{x.Key} {x.Value}")));
        foreach (string e in st.Examples) Console.WriteLine("[selfcheck]   " + e);
        bool fail = st.Bad.Keys.Any(k => k.StartsWith("unmapped") || k.StartsWith("error") || k == "run_error");
        Console.WriteLine($"[selfcheck] {(fail ? "FAIL" : "PASS")} (mapping failures and errors fail the check; row and fingerprint differences are reported)");
        return fail ? 1 : 0;
    }

    private static void CheckEventKeyMapping(Stats st)
    {
        (string Ev, string Key, string? TitleKey, string Want)[] cases =
        {
            ("EVENT.DOLL_ROOM", "抱抱先生", "MR_STRUGGLES.title", "MR_STRUGGLES"),
            ("EVENT.DOLL_ROOM", "宾邦", "BING_BONG.title", "BING_BONG"),
            ("EVENT.DOLL_ROOM", "Mr. Struggles", "DAUGHTER_OF_THE_WIND.title", "DAUGHTER_OF_THE_WIND"),
            ("EVENT.DOLL_ROOM", "DOLL_ROOM.pages.INITIAL.options.TAKE_SOME_TIME", "DOLL_ROOM.pages.INITIAL.options.TAKE_SOME_TIME.title", "TAKE_SOME_TIME"),
            ("EVENT.DOLL_ROOM", "PROCEED", "PROCEED", "PROCEED"),
            ("EVENT.TABLET_OF_TRUTH", "TABLET_OF_TRUTH.pages.INITIAL.options.DECIPHER_1", null, "DECIPHER"),
            ("EVENT.TABLET_OF_TRUTH", "TABLET_OF_TRUTH.pages.DECIPHER_2.options.DECIPHER", null, "DECIPHER"),
            ("EVENT.TABLET_OF_TRUTH", "TABLET_OF_TRUTH.pages.INITIAL.options.SMASH", null, "SMASH"),
            ("EVENT.COLOSSAL_FLOWER", "COLOSSAL_FLOWER.pages.INITIAL.options.EXTRACT_CURRENT_PRIZE_1", null, "EXTRACT_CURRENT_PRIZE"),
            ("EVENT.COLOSSAL_FLOWER", "COLOSSAL_FLOWER.pages.INITIAL.options.REACH_DEEPER_1", null, "REACH_DEEPER"),
            ("EVENT.COLOSSAL_FLOWER", "COLOSSAL_FLOWER.pages.REACH_DEEPER_1.options.EXTRACT_CURRENT_PRIZE_2", null, "EXTRACT_CURRENT_PRIZE"),
            ("EVENT.COLOSSAL_FLOWER", "COLOSSAL_FLOWER.pages.REACH_DEEPER_1.options.REACH_DEEPER_2", null, "REACH_DEEPER"),
            ("EVENT.COLOSSAL_FLOWER", "COLOSSAL_FLOWER.pages.REACH_DEEPER_2.options.POLLINOUS_CORE", null, "POLLINOUS_CORE"),
            ("EVENT.TRIAL", "TRIAL.pages.MERCHANT.options.GUILTY", null, "MERCHANT_GUILTY"),
            ("EVENT.TRIAL", "TRIAL.pages.MERCHANT.options.INNOCENT", null, "MERCHANT_INNOCENT"),
            ("EVENT.TRIAL", "TRIAL.pages.NOBLE.options.GUILTY", null, "NOBLE_GUILTY"),
            ("EVENT.TRIAL", "TRIAL.pages.NOBLE.options.INNOCENT", null, "NOBLE_INNOCENT"),
            ("EVENT.TRIAL", "TRIAL.pages.NONDESCRIPT.options.GUILTY", null, "NONDESCRIPT_GUILTY"),
            ("EVENT.TRIAL", "TRIAL.pages.NONDESCRIPT.options.INNOCENT", null, "NONDESCRIPT_INNOCENT"),
            ("EVENT.TRIAL", "TRIAL.pages.INITIAL.options.ACCEPT", null, "ACCEPT"),
            ("EVENT.TRIAL", "TRIAL.pages.INITIAL.options.MERCHANT_GUILTY", null, "MERCHANT_GUILTY"),
            ("EVENT.SLIPPERY_BRIDGE", "SLIPPERY_BRIDGE.pages.INITIAL.options.HOLD_ON_0", null, "HOLD_ON_0"),
            ("EVENT.SELF_HELP_BOOK", "SELF_HELP_BOOK.pages.INITIAL.options.READ_THE_BACK_LOCKED", null, "READ_THE_BACK"),
        };
        foreach (var (ev, key, tk, want) in cases)
        {
            string got = Controller.EventKey(ev, key, tk);
            if (got != want)
            {
                st.Add(st.Bad, "error:event_key_map");
                st.Example($"event_key_map {ev} key={key} title_key={tk}: got {got}, want {want}");
            }
        }
    }

    // ------------------------------------------------------------------ row comparison

    private static readonly (string Name, int At)[] INames = typeof(ML).GetFields()
        .Where(f => f.IsLiteral && f.FieldType == typeof(int) && f.Name.StartsWith('O')).Select(f => (f.Name, (int)f.GetRawConstantValue()!)).OrderBy(x => x.Item2).ToArray();
    private static readonly (string Name, int At)[] FNames = typeof(ML).GetFields()
        .Where(f => f.IsLiteral && f.FieldType == typeof(int) && (f.Name.StartsWith('F') || f.Name.StartsWith('S')) && !f.Name.StartsWith("Fight"))
        .Select(f => (f.Name, (int)f.GetRawConstantValue()!)).OrderBy(x => x.Item2).ToArray();

    private static string Name((string Name, int At)[] t, int i)
    {
        var b = t.LastOrDefault(x => x.At <= i);
        return b.Name is null ? $"#{i}" : b.At == i ? b.Name : $"{b.Name}+{i - b.At}";
    }

    public static List<string> RowDiff(MacroRow a, MacroRow b)
    {
        var d = new List<string>();
        for (int i = 0; i < a.I.Length; i++) if (a.I[i] != b.I[i]) d.Add("I:" + Name(INames, i));
        for (int i = 0; i < a.F.Length; i++) if (Math.Abs(a.F[i] - b.F[i]) > 1e-5) d.Add("F:" + Name(FNames, i));
        return d;
    }
}

/// <summary>The decision source of a self-check run: exports each decision as a protocol state, runs the live
/// controller on it, checks, then lets the true RunAgent decide for the run.</summary>
public sealed class FakeGame : IRunDecisionSource, ICombatCompletionValidator
{
    private readonly RunAgent _a;
    private readonly RunState _run;
    private readonly Player _p;
    private readonly Controller _ctl;
    private readonly SelfCheck.Stats _st;
    private readonly List<List<JsonObject>> _hist = new();
    private int _stateId;
    public int Checked;
    private static readonly ConditionalWeakTable<CardModel, object> Inst = new();
    private static int _nextInst;

    public FakeGame(RunAgent a, RunState run, Player p, Controller ctl, SelfCheck.Stats st) { _a = a; _run = run; _p = p; _ctl = ctl; _st = st; }

    private FloorFacts Facts => Priv.Get<FloorFacts>(_a, "_facts");

    // ------------------------------------------------------------------ history (map_point_history of the save)

    private static string RoomTypeName(RoomType t) => JsonNamingPolicy.SnakeCaseLower.ConvertName(t.ToString());

    private void Track()
    {
        Priv.Call(_a, "Pre", "selfcheck");   // the true agent's bookkeeping for this callback (idempotent)
        while (_hist.Count <= _run.CurrentActIndex) _hist.Add(new List<JsonObject>());
        int have = _hist.Sum(h => h.Count);
        while (have < _run.TotalFloor)
        {
            var e = new JsonObject
            {
                ["map_point_type"] = RoomTypeName(RoomType.Unassigned),
                ["rooms"] = new JsonArray(),
                ["player_stats"] = new JsonArray(new JsonObject { ["current_hp"] = _p.Creature.CurrentHp, ["max_hp"] = _p.Creature.MaxHp }),
            };
            _hist[_run.CurrentActIndex].Add(e);
            have++;
        }
        if (_hist[_run.CurrentActIndex].LastOrDefault() is { } cur && have == _run.TotalFloor)
        {
            if (_run.CurrentMapPoint is { } mp) cur["map_point_type"] = JsonNamingPolicy.SnakeCaseLower.ConvertName(mp.PointType.ToString());
            var rooms = (JsonArray)cur["rooms"]!;
            if (_run.BaseRoom is { } br && rooms.Count == 0) rooms.Add(RoomJson(br));
            if (_run.CurrentRoom is { } cr && !ReferenceEquals(cr, _run.BaseRoom) && rooms.Count < 2) rooms.Add(RoomJson(cr));
            var stats = (JsonObject)((JsonArray)cur["player_stats"]!)[0]!;
            if (!stats.ContainsKey("damage_taken")) { stats["current_hp"] = _p.Creature.CurrentHp; stats["max_hp"] = _p.Creature.MaxHp; }
        }
        // completed fights (RunAgent.EndFight): the floor's stats such that the import gives back the same FightNote
        int start = 0;
        var offsets = _hist.Select(h => { int o = start; start += h.Count; return o; }).ToList();
        foreach (FightNote f in Facts.Fights)
        {
            if (f.Act >= _hist.Count) continue;
            int k = f.Floor - offsets[f.Act] - 1;
            if (k < 0 || k >= _hist[f.Act].Count) continue;
            JsonObject e = _hist[f.Act][k];
            var stats = (JsonObject)((JsonArray)e["player_stats"]!)[0]!;
            stats["current_hp"] = f.EndHp; stats["hp_healed"] = 0; stats["damage_taken"] = f.StartHp - f.EndHp; stats["max_hp"] = f.MaxHpStart;
            stats["potion_used"] = new JsonArray(Enumerable.Range(0, f.Potions).Select(_ => (JsonNode)"POTION.X").ToArray());
            var rooms = (JsonArray)e["rooms"]!;
            JsonObject? cr = rooms.OfType<JsonObject>().LastOrDefault(r => r.Str("model_id") == f.Encounter)
                ?? rooms.OfType<JsonObject>().LastOrDefault(r => r.Str("model_id") is null && r.Str("room_type") is "monster" or "elite" or "boss");
            if (cr is not null) cr["model_id"] = f.Encounter;
            if (cr is null) rooms.Add(cr = new JsonObject { ["room_type"] = RoomTypeName(f.Room == ML.FightBoss ? RoomType.Boss : f.Room == ML.FightElite ? RoomType.Elite : RoomType.Monster), ["model_id"] = f.Encounter });
            cr["turns_taken"] = f.Turns;
        }
    }

    private static JsonObject RoomJson(AbstractRoom r) => new()
    {
        ["room_type"] = RoomTypeName(r.RoomType), ["model_id"] = FloorFacts.RoomModelId(r) is { Length: > 0 } m ? m : null,
    };

    // ------------------------------------------------------------------ export

    private static int InstOf(CardModel c) => (int)Inst.GetValue(c, _ => Interlocked.Increment(ref _nextInst));

    /// <summary>A protocol CARD as the mod writes it (StateBuilder.Card: ModelId.Entry ids, affliction id only).</summary>
    public static JsonObject CardJson(CardModel c, bool inHand = false)
    {
        var e = c.Enchantments.Count > 0 ? c.Enchantments[0] : null;
        bool x = c.CostsXEnergy;
        return new JsonObject
        {
            ["id"] = c.Id.Entry, ["upgrade"] = c.CurrentUpgradeLevel, ["enchantment"] = e?.Id.Entry,
            ["enchantment_amount"] = e is null ? null : (int)e.Magnitude, ["affliction"] = c.Affliction?.Id.Entry,
            ["cost"] = x ? -1 : c.CanonicalEnergyCostValue, ["cost_this_turn"] = x ? -1 : c.EnergyCost, ["x_cost"] = x,
            ["playable"] = inHand && c.CanPlay(out _), ["needs_target"] = c.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly,
            ["target_type"] = c.TargetType.ToString(), ["instance"] = InstOf(c),
        };
    }

    private static JsonNode? PropsJson(Func<IReadOnlyDictionary<string, JsonElement>> export)
    {
        try { return export().TryGetValue("props", out JsonElement pr) ? JsonNode.Parse(pr.GetRawText()) : null; }
        catch { return null; }
    }

    private static readonly Type ExporterType = typeof(Sts2Sim.Core.Runs.Transplant.RunTransplantExporter);
    private static readonly Type SavedPropsType = Priv.SimType("Sts2Sim.Core.Runs.Transplant.TransplantSavedProperties");

    private static JsonObject Coord(MapCoord c) => new() { ["col"] = c.col, ["row"] = c.row };

    private static JsonObject PointJson(MapPoint p) => new()
    {
        ["coord"] = Coord(p.coord), ["type"] = JsonNamingPolicy.SnakeCaseLower.ConvertName(p.PointType.ToString()),
        ["can_modify"] = p.CanBeModified, ["children"] = new JsonArray(p.Children.Select(c => (JsonNode)Coord(c.coord)).ToArray()),
    };

    /// <summary>The run as the mod sends it: protocol run fields plus run.save in the game's SerializableRun JSON names.</summary>
    public JsonObject RunJson()
    {
        ActMap map = _run.Map;
        var pts = map.GetAllMapPoints().ToList();
        var savedMap = new JsonObject
        {
            ["points"] = new JsonArray(pts.Select(p => (JsonNode)PointJson(p)).ToArray()),
            ["boss"] = PointJson(map.BossMapPoint), ["start"] = PointJson(map.StartingMapPoint),
            ["start_coords"] = new JsonArray(map.startMapPoints.Select(p => (JsonNode)Coord(p.coord)).ToArray()),
            ["width"] = map.GetColumnCount(), ["height"] = map.GetRowCount(),
        };
        if (map.SecondBossMapPoint is { } b2) savedMap["second_boss"] = PointJson(b2);
        var acts = new JsonArray();
        for (int i = 0; i < _run.Acts.Count; i++)
        {
            var a = new JsonObject { ["id"] = RunImport.ActId(_run.Acts[i]) };
            if (i == _run.CurrentActIndex)
            {
                a["rooms"] = new JsonObject
                {
                    ["boss_id"] = MacroEncoder.EncounterId(_run.CurrentActBossEncounter),
                    ["second_boss_id"] = _run.CurrentActSecondBossEncounter is { } sb ? MacroEncoder.EncounterId(sb) : null,
                };
                a["saved_map"] = savedMap;
            }
            acts.Add(a);
        }
        var facts = Facts;
        var deck = new JsonArray();
        for (int i = 0; i < _p.Deck.Cards.Count; i++)
        {
            CardModel c = _p.Deck.Cards[i];
            var e = c.Enchantments.Count > 0 ? c.Enchantments[0] : null;
            deck.Add(new JsonObject
            {
                ["id"] = c.Id.ToString(), ["current_upgrade_level"] = c.CurrentUpgradeLevel,
                ["enchantment"] = e is null ? null : new JsonObject { ["id"] = e.Id.ToString(), ["amount"] = (int)e.Magnitude },
                ["props"] = PropsJson(() => ((Sts2Sim.Core.Runs.Transplant.CardSnapshot)Priv.CallStatic(ExporterType, "ExportCardForTransplant", c, "x")!).Props),
                ["floor_added_to_deck"] = c.FloorAddedToDeck,
            });
        }
        var relics = new JsonArray(_p.Relics.Select(r => (JsonNode)new JsonObject
        {
            ["id"] = r.Id.ToString(), ["floor_added_to_deck"] = facts.RelicFloor.TryGetValue(r, out int f) ? f : null,
            ["props"] = PropsJson(() => (IReadOnlyDictionary<string, JsonElement>)Priv.CallStatic(SavedPropsType, "Export", r, "x")!),
        }).ToArray());
        var potions = new JsonArray();
        for (int k = 0; k < _p.PotionSlots.Count; k++)
            if (_p.PotionSlots[k] is { } pm) potions.Add(new JsonObject { ["id"] = pm.Id.ToString(), ["slot_index"] = k });
        var save = new JsonObject
        {
            ["schema_version"] = 0, ["acts"] = acts, ["current_act_index"] = _run.CurrentActIndex, ["ascension"] = (int)Priv.Call(_run, "GetAscensionLevelForTransplant")!,
            ["rng"] = new JsonObject { ["seed"] = _run.Rng.StringSeed }, ["start_time"] = 1,
            ["visited_map_coords"] = new JsonArray(_run.VisitedMapCoords.Select(c => (JsonNode)Coord(c)).ToArray()),
            ["map_point_history"] = new JsonArray(_hist.Select(h => (JsonNode)new JsonArray(h.Select(e => (JsonNode)e.DeepClone()).ToArray())).ToArray()),
            ["players"] = new JsonArray(new JsonObject
            {
                ["character_id"] = _p.Character.Id.ToString(), ["current_hp"] = _p.Creature.CurrentHp, ["max_hp"] = _p.Creature.MaxHp,
                ["max_energy"] = _p.MaxEnergy, ["max_potion_slot_count"] = _p.PotionSlots.Count, ["gold"] = _p.Gold,
                ["deck"] = deck, ["relics"] = relics, ["potions"] = potions,
                ["extra_fields"] = new JsonObject { ["card_shop_removals_used"] = _p.CardRemovalsUsed },
            }),
        };
        return new JsonObject
        {
            ["character"] = _p.Character.Id.Entry, ["ascension"] = (int)Priv.Call(_run, "GetAscensionLevelForTransplant")!, ["seed"] = _run.Rng.StringSeed,
            ["act_index"] = _run.CurrentActIndex, ["floor"] = _run.TotalFloor, ["hp"] = _p.Creature.CurrentHp, ["max_hp"] = _p.Creature.MaxHp,
            ["gold"] = _p.Gold, ["potions"] = new JsonArray(_p.PotionSlots.Select(x => (JsonNode?)(x?.Id.Entry)).ToArray()),
            ["act_ids"] = new JsonArray(_run.Acts.Select(a => (JsonNode)RunImport.ActId(a)[4..]).ToArray()),
            ["deck"] = new JsonArray(_p.Deck.Cards.Select(c => (JsonNode)new JsonObject
            {
                ["id"] = c.Id.Entry, ["upgrade"] = c.CurrentUpgradeLevel, ["enchantment"] = c.Enchantments.FirstOrDefault()?.Id.Entry,
                ["enchantment_amount"] = c.Enchantments.FirstOrDefault() is { } en ? (int)en.Magnitude : null, ["floor_added"] = c.FloorAddedToDeck,
            }).ToArray()),
            ["relics"] = new JsonArray(_p.Relics.Select(r => (JsonNode)new JsonObject { ["id"] = r.Id.Entry }).ToArray()),
            ["save"] = save,
        };
    }

    private static JsonArray Powers(Creature c) => new(c.Powers.Select(p => (JsonNode)new JsonObject { ["id"] = p.Id.Entry, ["amount"] = p.Amount }).ToArray());

    public static JsonObject CombatJson(CombatState s, RunState run)
    {
        Player p = s.Players[0];
        var pcs = p.PlayerCombatState!;
        var room = run.CurrentRoom as CombatRoom;
        JsonArray Pile(IEnumerable<CardModel> cs, bool hand = false) => new(cs.Select(c => (JsonNode)CardJson(c, hand)).ToArray());
        return new JsonObject
        {
            ["round"] = s.RoundNumber, ["turn_side"] = "PLAYER", ["encounter_id"] = room?.Encounter is { } e ? e.IdEntry : room?.EncounterName ?? "",
            ["room_type"] = (room?.RoomType ?? RoomType.Monster).ToString().ToUpperInvariant(),
            ["player"] = new JsonObject
            {
                ["hp"] = p.Creature.CurrentHp, ["max_hp"] = p.Creature.MaxHp, ["block"] = p.Creature.Block, ["energy"] = pcs.Energy, ["stars"] = pcs.Stars,
                ["max_energy"] = pcs.MaxEnergy, ["powers"] = Powers(p.Creature), ["hand"] = Pile(pcs.Hand.Cards, true), ["draw"] = Pile(pcs.DrawPile.Cards),
                ["discard"] = Pile(pcs.DiscardPile.Cards), ["exhaust"] = Pile(pcs.ExhaustPile.Cards),
            },
            // the mod's index: the position in the Enemies list (it shifts when a dead monster is removed)
            ["monsters"] = new JsonArray(s.Enemies.Select((c, i) => (JsonNode)new JsonObject
            {
                ["index"] = i, ["id"] = c.Monster?.Id.Entry, ["slot"] = c.SlotName, ["hp"] = c.CurrentHp, ["max_hp"] = c.MaxHp,
                ["block"] = c.Block, ["alive"] = c.IsAlive, ["powers"] = Powers(c), ["move_id"] = c.Monster?.NextMove?.Id,
            }).ToArray()),
        };
    }

    private JsonObject State(string screen, string? detail, JsonArray options, JsonObject? combat = null) => new()
    {
        ["state_id"] = ++_stateId, ["stable"] = true, ["agent_mode"] = true, ["paused"] = false, ["screen"] = screen,
        ["screen_detail"] = detail, ["run"] = RunJson(), ["combat"] = combat, ["options"] = options,
    };

    // ------------------------------------------------------------------ checks

    /// <summary>dump=FILE: every exported state as one JSON line (input for a mock game or `replay`).</summary>
    public static StreamWriter? Dump;
    /// <summary>discard_option=1: potion rewards with a full belt are shown with discard_potion options (the mod hides them).</summary>
    public static bool DiscardOption;

    private async Task<Decision?> LiveAsync(string type, JsonObject state, params string[] kinds)
    {
        Checked++;
        Dump?.WriteLine(new JsonObject { ["type"] = "decide", ["state"] = state.DeepClone() }.ToJsonString());
        _st.Add(_st.N, type);
        MacroRow? trueRow = type.StartsWith("combat") ? null : (MacroRow)Priv.Call(_a, "StateRow")!;
        Decision d;
        try { d = await _ctl.DecideAsync(state); }
        catch (Exception e)
        {
            _st.Add(_st.Bad, "error:" + type + ":" + e.GetType().Name);
            _st.Example($"{type} floor {_run.TotalFloor}: {e.GetType().Name}: {e.Message}\n{e.StackTrace?.Split('\n').FirstOrDefault()}");
            return null;
        }
        if (_ctl.Cur is { } imp) foreach (string n in imp.Notes) _st.Add(_st.Notes, "import: " + System.Text.RegularExpressions.Regex.Replace(n, @"\d+", "N"));
        if (d.Pause is not null)
        {
            _st.Add(_st.Bad, "unmapped:" + type);
            _st.Example($"{type} floor {_run.TotalFloor}: PAUSE {d.Pause}");
            return d;
        }
        var opts = state.Arr("options");
        if (!d.Wait && (d.Option < 0 || d.Option >= opts.Count || !kinds.Contains(opts[d.Option]!.Str("kind"))))
        {
            _st.Add(_st.Bad, "unmapped:" + type + ":bad_option");
            _st.Example($"{type} floor {_run.TotalFloor}: option {d.Option} ({d.Why})");
        }
        if (trueRow is not null && _ctl.Agent is not null)
        {
            var liveRow = (MacroRow)Priv.Call(_ctl.Agent, "StateRow")!;
            var diff = SelfCheck.RowDiff(trueRow, liveRow);
            if (diff.Count > 0)
            {
                _st.Add(_st.N, "row_mismatch");
                foreach (string k in diff.Distinct()) _st.Add(_st.RowDiff, k);
                if (_st.Examples.Count(x => x.StartsWith("row")) < 5)
                {
                    string F(FloorFacts f) => string.Join(" ", f.Fights.Select(x => $"{x.Encounter}/{x.Room}/{x.StartHp}-{x.EndHp}/{x.MaxHpStart}/p{x.Potions}/t{x.Turns}/f{x.Floor}/a{x.Act}"));
                    _st.Example($"row {type} floor {_run.TotalFloor}: {string.Join(" ", diff.Take(12))}" +
                        (diff.Any(x => x.Contains("Fight")) ? $"\n      true fights: {F(Facts)}\n      live fights: {F(_ctl.Cur!.Facts)}" : ""));
                }
            }
            else _st.Add(_st.N, "row_equal");
        }
        return d;
    }

    private void Agree(string type, bool same) => _st.Add(_st.N, $"{type}_{(same ? "same_choice" : "other_choice")}");

    public async Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options)
    {
        Track();
        var opts = new JsonArray(options.Select(p => (JsonNode)new JsonObject
            { ["kind"] = "map", ["col"] = p.coord.col, ["row"] = p.coord.row, ["type"] = p.PointType.ToString() }).ToArray());
        Decision? d = await LiveAsync("map", State("MAP", null, opts), "map");
        MapPoint pick = await _a.ChooseMapPointAsync(options);
        if (d is { Pause: null } && options.Count > 1) Agree("map", d.Option == options.ToList().IndexOf(pick));
        return pick;
    }

    public async Task<RewardDecision> ChooseRewardActionAsync(RewardsSet rewards)
    {
        Track();
        var cls = RewardDecisionClassifier.Classify(rewards);
        if (cls is RewardDecisionClassification.Choice ch && ch.Candidates.Any(c => c is RewardDecision.TakeCard))
        {
            var cards = ch.Candidates.OfType<RewardDecision.TakeCard>().Select(t => t.Card).ToList();
            var opts = new JsonArray(cards.Select((c, i) => (JsonNode)new JsonObject
            { ["kind"] = "card", ["index"] = i, ["id"] = c.Id.Entry, ["upgrade"] = c.CurrentUpgradeLevel, ["card"] = CardJson(c) }).Append(new JsonObject { ["kind"] = "skip" }).ToArray());
            Decision? dc = await LiveAsync("card_reward", State("CARD_REWARD", null, opts), "card", "skip");
            RewardDecision pc = await _a.ChooseRewardActionAsync(rewards);
            if (dc is { Pause: null }) Agree("card_reward", pc is RewardDecision.TakeCard t ? dc.Option == cards.IndexOf(t.Card) : dc.Option == cards.Count);
            return pc;
        }
        // the reward list as the game shows it
        var list = new JsonArray();
        void R(string t, string? id, int amount = 0) => list.Add(new JsonObject { ["kind"] = "reward", ["index"] = list.Count, ["type"] = t, ["id"] = id, ["amount"] = amount });
        if (!rewards.Gold.IsResolved) R("GOLD", null);
        // the mod hides potion rewards while the belt is full (discard_option=1: shows them with discard_potion options)
        bool full = !_p.PotionSlots.Contains(null);
        if (rewards.Potion is { IsResolved: false } pr && (!full || DiscardOption)) R("POTION", pr.Potion?.Id.Entry);
        if (rewards.Relic is { IsResolved: false } rr) R("RELIC", rr.Relic?.Id.Entry);
        if (!rewards.Card.IsResolved) R("CARD", null);
        list.Add(new JsonObject { ["kind"] = "proceed" });
        if (full && DiscardOption)   // a discard option (requested from the mod): one per held potion
            for (int k = 0; k < _p.PotionSlots.Count; k++)
                list.Add(new JsonObject { ["kind"] = "discard_potion", ["slot"] = k, ["id"] = _p.PotionSlots[k]?.Id.Entry });
        Decision? d = list.Count > 1 ? await LiveAsync(full ? "rewards_full_belt" : "rewards", State("REWARDS", null, list), "reward", "proceed", "discard_potion") : null;
        var before = _p.PotionSlots.Select(x => x?.Id.ToString()).ToList();
        RewardDecision pick = await _a.ChooseRewardActionAsync(rewards);
        int dropped = Enumerable.Range(0, before.Count).FirstOrDefault(k => before[k] is not null && _p.PotionSlots[k] is null, -1);
        if (d is { Pause: null } && d.Option >= 0)
        {
            JsonObject o = list[d.Option]!.AsObject();
            string want = dropped >= 0 ? "discard" : pick switch
            {
                RewardDecision.TakeGold => "GOLD", RewardDecision.TakePotion => full ? "skip_potion" : "POTION", RewardDecision.TakeRelic => "RELIC", _ => "",
            };
            if (want == "discard") Agree("rewards_discard", o.Str("kind") == "discard_potion" && o.Int("slot") == dropped);
            else if (want == "skip_potion") Agree("rewards_potion_left", o.Str("type") != "POTION");
            else if (want != "")
            {
                Agree("rewards", o.Str("type") == want);
                if (o.Str("type") != want) _st.Add(_st.Notes, $"rewards: true {want} ({pick.GetType().Name}), live {o.Str("kind")} {o.Str("type")}, belt full {full}");
            }
        }
        return pick;
    }

    public async Task<ShopDecision> ChooseShopActionAsync(MerchantInventory inv, Player player)
    {
        Track();
        var opts = new JsonArray();
        foreach (var e in inv.Cards.Where(e => !e.Purchased))
            opts.Add(new JsonObject { ["kind"] = "buy", ["item"] = "card", ["id"] = e.Card.Id.Entry, ["upgrade"] = e.Card.CurrentUpgradeLevel, ["price"] = e.Price, ["card"] = CardJson(e.Card) });
        foreach (var e in inv.Relics.Where(e => !e.Purchased)) opts.Add(new JsonObject { ["kind"] = "buy", ["item"] = "relic", ["id"] = e.Relic.Id.Entry, ["price"] = e.Price });
        foreach (var e in inv.Potions.Where(e => !e.Purchased)) opts.Add(new JsonObject { ["kind"] = "buy", ["item"] = "potion", ["id"] = e.Potion.Id.Entry, ["price"] = e.Price });
        if (!inv.CardRemoval.Purchased) opts.Add(new JsonObject { ["kind"] = "buy", ["item"] = "remove", ["price"] = inv.CardRemoval.Price });
        opts.Add(new JsonObject { ["kind"] = "leave" });
        await LiveAsync("shop", State("SHOP", null, opts), "buy", "leave");
        return await _a.ChooseShopActionAsync(inv, player);
    }

    public async Task<RestSiteDecision> ChooseRestSiteActionAsync(Player player, IReadOnlyList<RestSiteDecision> candidates)
    {
        Track();
        var ids = candidates.GroupBy(c => c.OptionId).Select(g => (Id: g.Key, Enabled: g.Any(x => x.IsEnabled))).ToList();
        var opts = new JsonArray(ids.Select(x => (JsonNode)new JsonObject { ["kind"] = "rest_option", ["id"] = x.Id.ToUpperInvariant(), ["enabled"] = x.Enabled }).ToArray());
        Decision? d = await LiveAsync("rest", State("REST", null, opts), "rest_option");
        RestSiteDecision pick = await _a.ChooseRestSiteActionAsync(player, candidates);
        if (d is { Pause: null }) Agree("rest", opts[d.Option]!.Str("id")!.ToLowerInvariant() == pick.OptionId);
        return pick;
    }

    public async Task<EventOption> ChooseEventOptionAsync(IReadOnlyList<EventOption> options)
    {
        Track();
        string evId = (_run.BaseRoom as EventRoom)?.Event.Id.ToString() ?? "";
        string evEntry = (_run.BaseRoom as EventRoom)?.Event.Id.Entry ?? "";
        bool ancient = _run.CurrentMapPoint?.PointType == MapPointType.Ancient;
        var opts = new JsonArray(options.Select((o, i) => (JsonNode)new JsonObject
        {
            ["kind"] = "event_option", ["index"] = i, ["key"] = $"{evId[6..]}.pages.INITIAL.options.{o.Key}{(o.IsLocked ? "_LOCKED" : "")}",
            ["locked"] = o.IsLocked, ["relic"] = o.OfferedContent is { ContentId.Length: > 0 } c && ancient ? c.ContentId : null,
        }).ToArray());
        Decision? d = await LiveAsync(ancient ? "ancient" : "event", State(ancient ? "ANCIENT" : "EVENT", evEntry, opts), "event_option");
        EventOption pick = await _a.ChooseEventOptionAsync(options);
        if (d is { Pause: null } && ancient) Agree("ancient", d.Option == options.ToList().IndexOf(pick));
        return pick;
    }

    public Task<CustomEventDecision> ChooseCustomEventActionAsync(EventModel ev)
    {
        Track();
        _st.Add(_st.N, "custom_event_not_checked:" + ev.Id.Entry);
        return _a.ChooseCustomEventActionAsync(ev);
    }

    public async Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request)
    {
        if (request.IsLiveCombatSelection)
        {
            var cs = (CombatState)request.Player.Creature.CombatState!;
            var sel = new JsonObject
            {
                ["kind"] = "select", ["purpose"] = "choose", ["source"] = "hand", ["min"] = request.MinCount, ["max"] = request.MaxCount,
                ["cards"] = new JsonArray(request.Candidates.Select(c => (JsonNode)CardJson(c)).ToArray()),
            };
            var opts = new JsonArray(sel);
            if (request.Cancelable) opts.Add(new JsonObject { ["kind"] = "skip" });
            if (!request.IsBundleSelection)
            {
                Decision? d = await LiveAsync("combat_select", State("COMBAT_SELECT", null, opts, CombatJson(cs, _run)), "select", "skip");
                if (d is { Pause: null, Cards: { } k } && (k.Any(i => i < 0 || i >= request.Candidates.Count) || k.Count > request.MaxCount))
                    _st.Add(_st.Bad, "unmapped:combat_select:cards");
            }
            return await _a.ChooseCardsAsync(request);
        }
        Track();
        if (request.Candidates.Count > 0 && !request.IsBundleSelection)
        {
            var deck = _p.Deck.Cards.ToList();
            bool inDeck = request.Candidates.All(c => deck.Any(x => ReferenceEquals(x, c)));
            var sel = new JsonObject
            {
                // as the mod: source deck (grid of deck cards) or generated (choose one of N new cards); no source model
                ["kind"] = "select", ["purpose"] = Sts2Sim.Core.Commands.CardSelectCmd.CurrentDeckPurpose ?? (inDeck ? "select" : "choose"),
                ["source"] = inDeck ? "deck" : "generated", ["min"] = request.MinCount, ["max"] = request.MaxCount,
                ["cards"] = new JsonArray(request.Candidates.Select(c => (JsonNode)CardJson(c)).ToArray()),
            };
            var opts = new JsonArray(sel);
            if (request.Cancelable || request.MinCount == 0) opts.Add(new JsonObject { ["kind"] = "skip" });
            Decision? d = await LiveAsync("card_select", State("CARD_SELECT", null, opts), "select", "skip");
            if (d is { Pause: null, Cards: { } k } && (k.Any(i => i < 0 || i >= request.Candidates.Count) || k.Count > request.MaxCount || k.Count < request.MinCount))
                _st.Add(_st.Bad, "unmapped:card_select:cards");
        }
        return await _a.ChooseCardsAsync(request);
    }

    // ------------------------------------------------------------------ combat

    private static string Norm(string line) => System.Text.RegularExpressions.Regex.Replace(line, @"#\d+", "#");

    public async Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
    {
        Track();
        // as the mod: targets are positions in the Enemies list; an action on the player itself has no target
        int Pos(uint? id) => state.Enemies.ToList().FindIndex(e => e.CombatId == id);
        var acts = ActionSpace.Base(state, includePotions: true).ToList();   // self-target potions: no target in the option
        var opts = new JsonArray();
        foreach (var g in acts.Where(a => a.Kind == ActionKind.PlayCard).GroupBy(a => a.SourceIndex))
        {
            var card = state.Players[0].PlayerCombatState!.Hand.Cards[g.Key];
            opts.Add(new JsonObject
            {
                ["kind"] = "play", ["hand_index"] = g.Key, ["card"] = CardJson(card),
                ["targets"] = new JsonArray(g.Where(a => a.TargetId is not null && Pos(a.TargetId) >= 0).Select(a => (JsonNode)Pos(a.TargetId)).Distinct().ToArray()),
            });
        }
        foreach (var g in acts.Where(a => a.Kind == ActionKind.UsePotion).GroupBy(a => a.SourceIndex))
            opts.Add(new JsonObject
            {
                ["kind"] = "potion", ["slot"] = g.Key, ["id"] = g.First().Source,
                ["targets"] = new JsonArray(g.Where(a => a.TargetId is not null && Pos(a.TargetId) >= 0).Select(a => (JsonNode)Pos(a.TargetId)).Distinct().ToArray()),
            });
        opts.Add(new JsonObject { ["kind"] = "end_turn" });
        JsonObject st = State("COMBAT", null, opts, CombatJson(state, _run));
        Decision? d = await LiveAsync("combat", st, "play", "potion", "end_turn");
        if (d is { Pause: null } && _ctl.Fight?.State is { } sh)
        {
            var a = ObservableState.Fingerprint(state).Split('\n').Select(Norm).ToList();
            var b = ObservableState.Fingerprint(sh).Split('\n').Select(Norm).ToList();
            var diffs = a.Except(b).Concat(b.Except(a)).Select(l => l.Split(new[] { '=', ' ' }, 2)[0]).Distinct().ToList();
            if (diffs.Count == 0) _st.Add(_st.N, "combat_fingerprint_equal");
            else
            {
                _st.Add(_st.N, "combat_fingerprint_differs");
                foreach (string k in diffs) _st.Add(_st.FpDiff, k);
                string cat = string.Join(",", diffs);
                if (_st.Examples.Count(x => x.StartsWith("fp")) < 12 && !_st.Examples.Any(x => x.StartsWith("fp [" + cat + "]")))
                    _st.Example("fp [" + cat + "] floor " + _run.TotalFloor + " round " + state.RoundNumber + ":\n      true:   " + string.Join("\n      true:   ", a.Except(b)) +
                        "\n      shadow: " + string.Join("\n      shadow: ", b.Except(a)));
            }
            foreach (string n in _ctl.Fight.SyncNotes) _st.Add(_st.Notes, "sync: " + System.Text.RegularExpressions.Regex.Replace(n, @"\d+", "N"));
            if (_ctl.Fight.Restarts > 0) _st.Add(_st.Notes, "shadow restarts (cumulative per fight)", 0);
            if (d.Target is not null) _st.Add(_st.N, "combat_targeted");
        }
        CombatDecision pick = await _a.ChooseCombatActionAsync(state);
        // the copy follows what the run plays (the true agent's action), as the shadow follows the game in live play
        if (d is { Pause: null } && _ctl.Fight is { State: { } shs } f)
        {
            var hand = state.Players[0].PlayerCombatState!.Hand.Cards.ToList();
            Creature? T(Creature? t) => t is null ? null : t.Monster is null ? shs.Players[0].Creature : f.CreatureAt(state.Enemies.ToList().IndexOf(t));
            CombatDecision? mapped = pick switch
            {
                CombatDecision.PlayCard pc when hand.IndexOf(pc.Card) is var hi and >= 0 && hi < shs.Players[0].PlayerCombatState!.Hand.Cards.Count
                    => new CombatDecision.PlayCard(shs.Players[0].PlayerCombatState!.Hand.Cards[hi], T(pc.Target)),
                CombatDecision.UsePotion up when state.Players[0].PotionSlots.ToList().IndexOf(up.Potion) is var k and >= 0 && shs.Players[0].PotionSlots[k] is { } sp
                    => new CombatDecision.UsePotion(sp, T(up.Target)),
                CombatDecision.EndTurn => new CombatDecision.EndTurn(),
                _ => null,
            };
            if (mapped is null) _st.Add(_st.Notes, "true action not mappable to the copy");
            else await f.ApplyAsync(mapped);
            if (f.Broken) _st.Add(_st.Notes, "shadow broken: " + (f.BrokenWhy is { } w ? (w.Length > 60 ? w[..60] : w) : ""));
        }
        return pick;
    }

    public void ValidateCompletedCombat(CombatState state) => _a.ValidateCompletedCombat(state);
}
