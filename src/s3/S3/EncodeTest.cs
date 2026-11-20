using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Entities.Merchant;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Events;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Characters;
using Sts2Sim.Core.Rewards;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S2;

namespace Sts2Sim.S3;

/// <summary>
/// Round-trip test of the macro encoder (S3_HANDOVER §9.1). Plays runs with the S3 agent (uniform networks; enemies are
/// set to 1 HP so runs go deep) and writes, per run,
/// <list type="bullet">
/// <item>the rows the C# encoder produced for the state after every floor (<c>rows.jsonl</c>), and</item>
/// <item>a run record in the Spire Codex / game run-history form (<c>runs.jsonl</c>), derived independently from
/// snapshots of the deck, relics and potions at every floor end (cards gained, removed, upgraded, enchanted; potions
/// gained and used; final deck and relics).</item>
/// </list>
/// src/s3/py/roundtrip_macro.py rebuilds the per-floor examples from the run records with the S1 dataset builder
/// (src/value_v1/build_floor_dataset.py) and compares them with the C# rows.
/// </summary>
public static class EncodeTest
{
    private sealed record CardSnap(CardModel Ref, string Id, int Upg, string Ench, int Eamt, int Floor);

    private sealed class Recorder(RunAgent agent, RunState run, Player p, MacroVocab mvocab) : IRunDecisionSource, ICombatCompletionValidator
    {
        public readonly List<List<Dictionary<string, object?>>> Acts = new();
        private List<CardSnap> _deck = new();
        private List<PotionModel> _pots = new();
        private List<RelicModel> _relics = new();
        private int _lastFloor;
        private object? _engine;
        private string _modelId = "", _roomType = "";
        private int _modelFloor = -1;

        private List<CardSnap> Deck() => p.Deck.Cards.Select(c =>
        {
            var e = c.Enchantments.Count > 0 ? c.Enchantments[0] : null;
            return new CardSnap(c, c.Id.ToString(), c.CurrentUpgradeLevel, e?.Id.ToString() ?? "", e is null ? 0 : (int)e.Magnitude, c.FloorAddedToDeck ?? 0);
        }).ToList();

        private static Dictionary<string, object?> CardJson(CardSnap c)
        {
            var d = new Dictionary<string, object?> { ["id"] = c.Id, ["floor_added_to_deck"] = c.Floor };
            if (c.Upg > 0) d["current_upgrade_level"] = c.Upg;
            if (c.Ench != "") d["enchantment"] = new Dictionary<string, object?> { ["id"] = c.Ench, ["amount"] = c.Eamt };
            return d;
        }

        private void NoteRoom()
        {
            int g = run.TotalFloor;
            if (_modelFloor == g || run.BaseRoom is not { } room) return;
            _modelFloor = g; _modelId = FloorFacts.RoomModelId(room);
            _roomType = room switch { CombatRoom c => c.RoomType.ToString().ToLowerInvariant(), EventRoom => "event", MerchantRoom => "shop", RestSiteRoom => "rest_site", _ => "treasure" };
        }

        private static string PointName(MapPointType t) => t switch
        {
            MapPointType.Ancient => "ancient", MapPointType.Monster => "monster", MapPointType.Elite => "elite", MapPointType.Boss => "boss",
            MapPointType.Unknown => "unknown", MapPointType.Shop => "shop", MapPointType.RestSite => "rest_site", MapPointType.Treasure => "treasure", _ => "",
        };

        /// <summary>Writes the entry of the current floor from the difference to the previous floor's snapshot.</summary>
        public void EndFloor()
        {
            int g = run.TotalFloor;
            if (g <= _lastFloor) return;
            _lastFloor = g;
            while (Acts.Count <= run.CurrentActIndex) Acts.Add(new());
            var now = Deck();
            var prev = _deck.ToDictionary(c => c.Ref, c => c, ReferenceEqualityComparer.Instance);
            var cur = now.ToDictionary(c => c.Ref, c => c, ReferenceEqualityComparer.Instance);
            // a card that left the deck and came back as the same object within the floor (Thieving Hopper's Swipe: the
            // stolen deck card returns as a SpecialCardReward and CardPileCmd sets FloorAddedToDeck to this floor) is
            // logged as removed (its old entry) and gained (its new one), so the log-based rebuild keeps it on the floors
            // between its first gain and its return instead of dropping it as a future card (ic113 round trip)
            var readded = now.Where(c => prev.TryGetValue(c.Ref, out CardSnap? b) && b.Floor != c.Floor).ToList();
            var gained = now.Where(c => !prev.ContainsKey(c.Ref)).Concat(readded).Select(CardJson).ToList();
            var removed = _deck.Where(c => !cur.ContainsKey(c.Ref) || readded.Any(x => ReferenceEquals(x.Ref, c.Ref))).Select(CardJson).ToList();
            var upgraded = new List<string>(); var downgraded = new List<string>(); var enchanted = new List<object>();
            foreach (CardSnap c in now)
            {
                if (!prev.TryGetValue(c.Ref, out CardSnap? b) || b.Floor != c.Floor) continue;   // re-added: logged above in full
                for (int i = b.Upg; i < c.Upg; i++) upgraded.Add(c.Id);
                for (int i = c.Upg; i < b.Upg; i++) downgraded.Add(c.Id);
                if (c.Ench != "" && b.Ench == "")
                    enchanted.Add(new Dictionary<string, object?> { ["card"] = new Dictionary<string, object?> { ["id"] = c.Id }, ["enchantment"] = c.Ench });
            }
            var pots = p.PotionSlots.Where(x => x is not null).Select(x => x!).ToList();
            var gainedP = pots.Where(x => !_pots.Any(y => ReferenceEquals(x, y))).Select(x => (object)new Dictionary<string, object?> { ["choice"] = x.Id.ToString(), ["was_picked"] = true }).ToList();
            var usedP = _pots.Where(x => !pots.Any(y => ReferenceEquals(x, y))).Select(x => x.Id.ToString()).ToList();
            var relics = p.Relics.ToList();
            var removedR = _relics.Where(x => !relics.Any(y => ReferenceEquals(x, y))).Select(x => x.Id.ToString()).ToList();
            _deck = now; _pots = pots; _relics = relics;
            bool here = _modelFloor == g;
            var room = new Dictionary<string, object?> { ["room_type"] = here ? _roomType : "treasure", ["turns_taken"] = 0 };
            if (here && _modelId != "") room["model_id"] = _modelId;
            Acts[run.CurrentActIndex].Add(new Dictionary<string, object?>
            {
                ["map_point_type"] = PointName(run.CurrentMapPoint?.PointType ?? MapPointType.Ancient),
                ["rooms"] = new[] { room },
                ["player_stats"] = new[]
                {
                    new Dictionary<string, object?>
                    {
                        ["current_hp"] = Math.Max(0, p.Creature.CurrentHp), ["max_hp"] = p.Creature.MaxHp, ["current_gold"] = p.Gold,
                        ["cards_gained"] = gained, ["cards_removed"] = removed, ["upgraded_cards"] = upgraded, ["downgraded_cards"] = downgraded,
                        ["cards_enchanted"] = enchanted, ["potion_choices"] = gainedP, ["potion_used"] = usedP,
                        ["relics_removed"] = removedR,
                    },
                },
            });
        }

        public Dictionary<string, object?> RunJson(string seed, bool won, int ascension, Dictionary<RelicModel, int> relicFloor) => new()
        {
            ["run_hash"] = seed, ["seed"] = seed, ["build_id"] = ML.BuildId, ["era"] = ML.Era, ["ascension"] = ascension, ["win"] = won,
            ["acts"] = run.Acts.Select(a => "ACT." + a.GetType().Name.ToUpperInvariant()).ToList(),
            ["map_point_history"] = Acts,
            ["players"] = new[]
            {
                new Dictionary<string, object?>
                {
                    ["character"] = p.Character.Id.ToString(), ["deck"] = Deck().Select(CardJson).ToList(),
                    ["relics"] = p.Relics.Select(r => new Dictionary<string, object?> { ["id"] = r.Id.ToString(), ["floor_added_to_deck"] = relicFloor.GetValueOrDefault(r, 1) }).ToList(),
                },
            },
        };

        public void OnRoomResolved(MapPoint pt, RoomType rt) { agent.OnRoomResolved(pt, rt); EndFloor(); }

        public async Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options)
        {
            EndFloor();
            int before = agent.Rec.Steps.Count;
            MapPoint chosen = await agent.ChooseMapPointAsync(options);
            NoteMapDecision(options, before);
            return chosen;
        }

        // ---- layout v2 printout: the recorded map decisions (as text) and the act maps they were taken on

        /// <summary>Act index and text of every recorded map decision of the run, in play order.</summary>
        public readonly List<(int Act, string Text)> MapNotes = new();
        public readonly Dictionary<int, string> MapTexts = new();

        private static char Letter(MapPointType t) => t switch
        {
            MapPointType.Monster => 'M', MapPointType.Elite => 'E', MapPointType.RestSite => 'R', MapPointType.Shop => '$',
            MapPointType.Treasure => 'T', MapPointType.Unknown => '?', MapPointType.Ancient => 'A', MapPointType.Boss => 'B', _ => '.',
        };

        private string MapText()
        {
            ActMap map = run.Map;
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"map of act {run.CurrentActIndex + 1}: boss at row {map.BossMapPoint.coord.row}" +
                (map.SecondBossMapPoint is { } s ? $", second boss at row {s.coord.row}" : "") +
                "; per point c<column><type>><columns of its children> (M monster, E elite, R rest site, $ shop, T treasure, ? unknown, B boss)");
            for (int row = map.GetRowCount() - 1; row >= 0; row--)
            {
                var pts = map.GetPointsInRow(row).Where(q => q.Children.Count > 0 || q.parents.Count > 0).ToList();
                if (pts.Count == 0) continue;
                sb.AppendLine($"  row {row,2}: " + string.Join("  ", pts.Select(q => $"c{q.coord.col}{Letter(q.PointType)}>" +
                    string.Join(",", q.Children.OrderBy(c => c.coord.col).Select(c => c.PointType == MapPointType.Boss ? "B" : c.coord.col.ToString())))));
            }
            return sb.ToString().TrimEnd();
        }

        /// <summary>Text of the map decision the agent just recorded (none when there was a single option): the layout v2
        /// inputs read back from the recorded row, next to the options in the agent's order (column, then row).</summary>
        private void NoteMapDecision(IReadOnlyList<MapPoint> options, int stepsBefore)
        {
            MacroStep? st = null;
            for (int i = stepsBefore; i < agent.Rec.Steps.Count; i++)
                if (agent.Rec.Steps[i] is { Kind: 1, Dtype: ML.DMap } x) st = x;
            if (st is null) return;
            int act = run.CurrentActIndex;
            if (!MapTexts.ContainsKey(act)) MapTexts[act] = MapText();
            MacroRow r = st.Row;
            MacroVocab mv = mvocab;
            ReadOnlySpan<float> x2 = r.F.AsSpan(ML.FScal2, ML.NS2);
            var ci = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder();
            MapPoint? here = run.CurrentMapPoint;
            sb.AppendLine($"map decision {agent.Rec.Seed} floor {st.Floor} act {act + 1} at " +
                (here is null ? "the act start" : $"row {here.coord.row} c{here.coord.col} {here.PointType}") +
                $", hp {p.Creature.CurrentHp}/{p.Creature.MaxHp}: boss {r.I[ML.OBoss]} '{mv.Name("encounter", r.I[ML.OBoss])}', boss2 {r.I[ML.OBoss2]} '{mv.Name("encounter", r.I[ML.OBoss2])}'" +
                $" (simulator: '{MacroEncoder.EncounterId(run.CurrentActBossEncounter)}', '{MacroEncoder.EncounterId(run.CurrentActSecondBossEncounter)}')");
            sb.AppendLine($"    scalars: elites {x2[ML.XElites] * 4:0.##} rests {x2[ML.XRests] * 4:0.##} shops {x2[ML.XShops] * 3:0.##} (this act), " +
                $"free potion slots {x2[ML.XPotFree] * 3:0.##} of {x2[ML.XPotSlots] * 5:0.##}, map row {x2[ML.XMapRow] * 16:0.##}; raw [" +
                string.Join(" ", x2.ToArray().Select(v => v.ToString("0.###", ci))) + "]");
            var sorted = options.OrderBy(o => o.coord.col).ThenBy(o => o.coord.row).Take(ML.K).ToList();
            for (int j = 0; j < Math.Min(sorted.Count, r.NOptions); j++)
            {
                float[] f = r.F.AsSpan(ML.FOptFeat2 + j * ML.MF2, ML.MF2).ToArray();
                sb.AppendLine($"    {(j == st.Chosen ? "*" : " ")} option {j}: row {sorted[j].coord.row} c{sorted[j].coord.col} {sorted[j].PointType,-8} " +
                    $"rest in {f[0] * 8:0.##}, shop in {(f[1] >= 1f ? "8+/none" : (f[1] * 8).ToString("0.##", ci))}, to the first rest: elites >= {f[2] * 3:0.##}, fights >= {f[3] * 5:0.##}; " +
                    $"elites in 3 rows <= {f[4] * 3:0.##}; elite>rest {f[5]:0}, rest>elite {f[6]:0}; boss in {f[7] * 16:0.##}; raw [" +
                    string.Join(" ", f.Select(v => v.ToString("0.###", ci))) + "]");
            }
            MapNotes.Add((act, sb.ToString().TrimEnd()));
        }

        public Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
        {
            NoteRoom();
            if (!ReferenceEquals(_engine, state.Engine))
            {
                _engine = state.Engine;
                foreach (var e in state.Enemies) if (e.CurrentHp > 1) e.SetCurrentHpInternal(1);   // test only: fights end at once
            }
            return agent.ChooseCombatActionAsync(state);
        }

        public void ValidateCompletedCombat(CombatState state) => agent.ValidateCompletedCombat(state);
        public Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request) => agent.ChooseCardsAsync(request);
        public Task<RewardDecision> ChooseRewardActionAsync(RewardsSet rewards) { NoteRoom(); return agent.ChooseRewardActionAsync(rewards); }
        public Task<ShopDecision> ChooseShopActionAsync(MerchantInventory inventory, Player player) { NoteRoom(); return agent.ChooseShopActionAsync(inventory, player); }
        public Task<CustomEventDecision> ChooseCustomEventActionAsync(EventModel @event) { NoteRoom(); return agent.ChooseCustomEventActionAsync(@event); }
        public Task<RestSiteDecision> ChooseRestSiteActionAsync(Player player, IReadOnlyList<RestSiteDecision> candidates) { NoteRoom(); return agent.ChooseRestSiteActionAsync(player, candidates); }
        public Task<EventOption> ChooseEventOptionAsync(IReadOnlyList<EventOption> options) { NoteRoom(); return agent.ChooseEventOptionAsync(options); }
    }

    /// <param name="maps">Layout v2 printout: this many map decisions of the first run that finished are printed
    /// (spread evenly over the run, with the act maps they were taken on); 0 = none.</param>
    public static async Task<int> Run(string outDir, int runs, int ascension, string mvocabPath, int maps = 8, string character = "defect")
    {
        string charId = character.StartsWith("CHARACTER.", StringComparison.Ordinal) ? character : "CHARACTER." + character.ToUpperInvariant();
        var characterModel = ModelDb.GetById<CharacterModel>(ModelId.Deserialize(charId));
        Directory.CreateDirectory(outDir);
        var mvocab = new MacroVocab(mvocabPath);
        var services = new ActorServices
        {
            Net = new UniformEvaluator(), Macro = new UniformMacroEvaluator(), MVocab = mvocab,
            Vocab = new ObsLayout.Vocab(ObsLayout.BuildVocab()),
            Cfg = new AgentConfig { Ascension = ascension, SimsFull = 0, SimsFast = 0, RecordCombat = false },
        };
        using var rowsOut = new StreamWriter(Path.Combine(outDir, "rows.jsonl"));
        // layout v3: every recorded row (floor states and decisions) with the map nodes and fights the encoder meant to
        // pack, for src/s3/py/layout3_check.py --encode DIR (it unpacks the rows and compares)
        using var rows3Out = new StreamWriter(Path.Combine(outDir, "rows3.jsonl"));
        MacroEncoder.Debug = true;
        long rows3 = 0;
        using var runsOut = new StreamWriter(Path.Combine(outDir, "runs.jsonl"));
        int won = 0, errors = 0; long floors = 0, rows = 0;
        var kinds = new Dictionary<string, int>();
        bool mapsPrinted = maps <= 0;
        for (int i = 0; i < runs; i++)
        {
            string seed = $"ENC{i:D5}";
            var run = new RunState(seed, ActDefinition.GetRandomList(seed), ascensionLevel: ascension);
            var player = Player.CreateForNewRun(characterModel, run);
            run.AddPlayer(player);
            var agent = new RunAgent(services, run, player, seed, perturbed: false);
            var rec = new Recorder(agent, run, player, mvocab);
            var driver = new RunDriver(run, rec);
            driver.OnRoomResolved += rec.OnRoomResolved;
            bool win = false;
            try { var res = await driver.RunAsync(80); agent.Finish(res); win = res.Won; rec.EndFloor(); }
            catch (Exception e) { agent.Fail(e); errors++; Console.WriteLine($"run {seed} error at floor {run.TotalFloor}: {e.GetType().Name}: {e.Message}"); continue; }
            if (win) won++;
            floors += run.TotalFloor;
            if (!mapsPrinted && rec.MapNotes.Count > 0)
            {
                mapsPrinted = true;
                int n = rec.MapNotes.Count, k = Math.Min(maps, n), lastAct = -1;
                Console.WriteLine($"layout v2 (ints {ML.IW}, floats {ML.FW}; boss ids at {ML.OBoss}, option features from {ML.FOptFeat2}, scalars from {ML.FScal2}): " +
                    $"{k} of the {n} map decisions of run {seed} (ascension {ascension}, {run.TotalFloor} floors, {(win ? "won" : "lost")})");
                for (int j = 0; j < k; j++)
                {
                    var (act, text) = rec.MapNotes[k == 1 ? 0 : (int)Math.Round(j * (n - 1) / (double)(k - 1))];
                    if (act != lastAct) { lastAct = act; Console.WriteLine(rec.MapTexts[act]); }
                    Console.WriteLine(text);
                }
            }
            var relicFloor = new Dictionary<RelicModel, int>(ReferenceEqualityComparer.Instance);
            foreach (MacroStep s in agent.Rec.Steps)
            {
                rows3Out.WriteLine(JsonSerializer.Serialize(new
                {
                    run = seed, floor = s.Floor, act = s.Act, kind = s.Kind, dtype = s.Dtype, chosen = s.Chosen, nopt = s.Row.NOptions,
                    ints = s.Row.I, floats = s.Row.F,
                    map = (s.Row.DbgMap ?? Array.Empty<MapNode>()).Select(m => new[] { (int)m.Type + 1, m.Row, m.Col, m.Visited ? 1 : 0, m.Current ? 1 : 0, m.Reachable ? 1 : 0, m.Mask }),
                    fights = (s.Row.DbgFights ?? Array.Empty<FightNote>()).Select(f => new object[] { f.EncTok, f.Encounter, f.Room, f.StartHp, f.EndHp, f.MaxHpStart, f.Potions, f.Turns, f.Floor, f.Act }),
                }));
                rows3++;
                if (s.Kind == 1) { string n = ML.DtNames[s.Dtype]; kinds[n] = kinds.GetValueOrDefault(n) + 1; continue; }
                rowsOut.WriteLine(JsonSerializer.Serialize(new { run = seed, floor = s.Floor, ints = s.Row.I, floats = s.Row.F }));
                rows++;
            }
            runsOut.WriteLine(JsonSerializer.Serialize(rec.RunJson(seed, win, ascension, agent.RelicFloors)));
        }
        Console.WriteLine($"encode test: {runs} runs, {won} won, {errors} errors, mean floors {floors / (double)Math.Max(1, runs - errors):F1}, {rows} floor rows; " +
            "decisions " + string.Join(", ", kinds.OrderBy(k => k.Key).Select(k => $"{k.Key} {k.Value}")));
        Console.WriteLine($"layout v3 (ints {ML.IW}, floats {ML.FW}; fight ints from {ML.OFightEnc}, fight floats from {ML.FFights}, map nodes from {ML.FMap}): " +
            $"{rows3} rows in rows3.jsonl; map rows {MacroEncoder.MapRows}, most nodes in an act map {MacroEncoder.MapNodesMax}, rows truncated to {ML.NMap} nodes {MacroEncoder.MapTruncated}");
        MacroEncoder.Debug = false;
        Console.WriteLine("macro vocabulary: " + string.Join(", ", mvocab.Seen.Where(x => x.Value > 0).Select(x => $"{x.Key} unknown {mvocab.Unknown[x.Key]}/{x.Value}")));
        Console.WriteLine("run-level card selections: " + string.Join(", ", services.SelectionKinds.OrderByDescending(x => x.Value).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("notes: " + string.Join(", ", services.Notes.OrderByDescending(x => x.Value).Take(30).Select(x => $"{x.Key} {x.Value}")));
        return 0;
    }
}

/// <summary><c>S3 tftest</c>: checks the death value of a T_f table (TfScorer) and its use in the search's outcome value
/// (FightSignals.OutcomeValue), with and without a death value. Exit code 0 when every check holds.</summary>
public static class TfDeathTest
{
    public static int Run()
    {
        int fails = 0;
        void Check(string what, double got, double want)
        {
            bool ok = Math.Abs(got - want) < 1e-12;
            if (!ok) fails++;
            Console.WriteLine($"{(ok ? "ok  " : "FAIL")} {what}: {got:F12} (expected {want:F12})");
        }
        const int M = 40;
        var subsets = new List<int[]> { Array.Empty<int>(), new[] { 0 }, new[] { 1 }, new[] { 0, 1 } };
        double[] Tab(double baseV, double slope) { var t = new double[M + 1]; for (int h = 1; h <= M; h++) t[h] = baseV + slope * h / M; return t; }
        foreach (double death in new[] { 0.0, 0.6234 })
        {
            var tables = new List<double[]> { Tab(0.63, 0.05), Tab(0.64, 0.05), Tab(0.635, 0.05), Tab(0.645, 0.05) };
            foreach (var t in tables) t[0] = death;
            var tf = new TfScorer();
            tf.Set("x", M, new[] { "A", "B" }, subsets, tables, false, death);
            double tMax = tf.MaxValue("x");
            Check($"[d={death}] DeathValue", tf.DeathValue("x"), death);
            Check($"[d={death}] Value(hp 0)", tf.Value("x", new[] { 0, 1 }, 0), death);
            Check($"[d={death}] Value(hp 20, both potions)", tf.Value("x", new[] { 0, 1 }, 20), 0.645 + 0.05 * 20 / M);
            Check($"[d={death}] T_max", tMax, 0.645 + 0.05);
            var spec = new BattleStartSpec { Id = "x", RoomType = "monster", ActIndex = 2, Floor = 40, Hp = 20, MaxHp = M, Potions = new() { "A", "B" } };
            var sig = new FightSignals(tf, new BattleContext("x", M, new[] { "A", "B" }), spec, tMax);
            var pHp = new double[Outcome.Bins]; var keep = new double[Outcome.Slots];
            keep[0] = 1; keep[1] = 0;
            pHp[0] = 1;
            Check($"[d={death}] OutcomeValue(certain death)", sig.OutcomeValue(pHp, keep, 0b11), death / tMax);
            pHp[0] = 0.25; pHp[Outcome.Bins - 1] = 0.75;   // bin 40 = HP 40 (one HP per bin here)
            Check($"[d={death}] OutcomeValue(1/4 death, 3/4 full HP keeping potion A)", sig.OutcomeValue(pHp, keep, 0b11),
                (0.25 * death + 0.75 * (0.64 + 0.05)) / tMax);
        }
        Console.WriteLine(fails == 0 ? "all checks passed" : $"{fails} checks FAILED");
        return fails == 0 ? 0 : 1;
    }
}
