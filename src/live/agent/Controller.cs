using System.Collections.Immutable;
using System.Text.Json.Nodes;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Entities.Merchant;
using Sts2Sim.Core.Events;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Rewards;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S2;
using Sts2Sim.S3;

namespace Sts2Sim.Live;

/// <summary>What the controller does with one stable state.</summary>
public sealed class Decision
{
    public int Option = -1;
    public int? Target;
    public List<int>? Cards;
    /// <summary>Send `pause` with this reason (the player plays the step by hand and resumes).</summary>
    public string? Pause;
    /// <summary>Nothing to do (main menu, waiting).</summary>
    public bool Wait;
    public string Why = "";
    /// <summary>A combat action: applied to the shadow fight once the game accepted it.</summary>
    public CombatDecision? Combat;
    public readonly Dictionary<string, object?> Info = new();

    public static Decision Act(int option, string why) => new() { Option = option, Why = why };
    public static Decision Paused(string reason) => new() { Pause = reason, Why = "pause" };
    public static Decision Idle(string why) => new() { Wait = true, Why = why };
}

/// <summary>A rest-site option the simulator has no record type for (LIFT, DIG, RECALL, ...): RunAgent only reads
/// OptionId (its macro token is SPECIAL "REST:ID" or the hashed event bucket, exactly as for the simulator's own).</summary>
public sealed record LiveRestOption(string Id, bool Enabled) : RestSiteDecision
{
    public override string OptionId => Id;
    public override bool IsEnabled => Enabled;
    public override int Priority => 9;
    public override Task ExecuteAsync(Sts2Sim.Core.Entities.Players.Player player) => throw new NotSupportedException("live option");
}

/// <summary>
/// Maps one game state to the S3 agent's decision. Every macro decision goes through RunAgent's own public decision
/// method (ChooseMapPointAsync, ChooseRewardActionAsync, ChooseEventOptionAsync, ChooseRestSiteActionAsync,
/// ChooseShopActionAsync, ChooseCardsAsync) on simulator objects built from the game's options, with RunAgent bound to
/// the run imported from the game (<see cref="RunImport"/>); fights go through RunAgent.ChooseCombatActionAsync on the
/// synced shadow fight (<see cref="Shadow"/>). The simulator objects RunAgent returns are mapped back to option indices
/// (map coordinates, card / relic / potion ids and upgrades, event option keys, deck positions, hand positions).
/// Anything unmappable gives a pause with the reason.
/// </summary>
public sealed class Controller
{
    private readonly AgentSetup _s;
    public RunAgent? Agent;
    private string? _agentKey;
    public Imported? Cur;
    public Shadow? Fight;
    /// <summary>A deck card chosen with a rest-site Smith or a shop removal, to pick in the selection screen that follows.</summary>
    private (string Purpose, string Key, int DeckIndex, int Floor)? _pending;
    private readonly HashSet<string> _handled = new();
    private int _handledFloor = -1;
    public int Decisions;
    public readonly Action<string> Say;

    public Controller(AgentSetup s, Action<string>? say = null) { _s = s; Say = say ?? Console.WriteLine; }

    private static List<(int I, JsonObject O)> Opts(JsonObject st, params string[] kinds) =>
        st.Arr("options").Select((n, i) => (i, n as JsonObject)).Where(x => x.Item2 is not null && kinds.Contains(x.Item2.Str("kind")))
            .Select(x => (x.i, x.Item2!)).ToList();

    // ------------------------------------------------------------------ binding RunAgent to an imported run

    private static string RunKey(JsonObject st)
    {
        JsonObject? run = st.Obj("run");
        JsonObject? save = run?.Obj("save");
        return $"{save?.Obj("rng")?.Str("seed") ?? run?.Str("seed")}:{save?.Str("start_time")}:{save?.IntOrNull("ascension") ?? run?.IntOrNull("ascension")}";
    }

    /// <summary>Imports the run and binds RunAgent to it (a new RunAgent for a new game run).</summary>
    public Imported Bind(JsonObject st, bool inCombat)
    {
        Imported imp = RunImport.Build(st, _s.MVocab, inCombat);
        string key = RunKey(st);
        if (Agent is null || key != _agentKey)
        {
            EndFight();
            Agent = new RunAgent(_s.Services, imp.Run, imp.P, "LIVE-" + imp.Seed, false, imp.Ascension);
            _agentKey = key;
            _pending = null; _handled.Clear();
            Say($"new run: seed {imp.Seed}, ascension {imp.Ascension}");
        }
        Priv.Set(Agent, "_run", imp.Run);
        Priv.Set(Agent, "_p", imp.P);
        Priv.Set(Agent, "_facts", imp.Facts);
        Cur = imp;
        return imp;
    }

    /// <summary>Drops RunAgent's live fight (as RunAgent.Fail does) and the shadow.</summary>
    public void EndFight()
    {
        if (Agent is not null && Priv.Get(Agent, "_fight") is { } lf)
        {
            ((NetSearch)Priv.Get(lf, "Search")!).Clear();
            _s.Services.Tf.Remove((string)Priv.Get(lf, "Id")!);
            Priv.Set(Agent, "_fight", null);
        }
        if (Agent is not null) ((Queue<string[]>)Priv.Get(Agent, "_selScript")!).Clear();
        Fight = null;
    }

    // ------------------------------------------------------------------ dispatch

    public async Task<Decision> DecideAsync(JsonObject st)
    {
        Decisions++;
        string screen = st.Str("screen") ?? "OTHER";
        if (screen is not ("COMBAT" or "COMBAT_SELECT") && Fight is not null) EndFight();
        if (st.Obj("run") is { } run)
        {
            int floor = run.Int("floor", -1);
            if (floor != _handledFloor) { _handledFloor = floor; _handled.Clear(); }
        }
        switch (screen)
        {
            case "MAIN_MENU": return Decision.Idle("main menu");
            case "GAME_OVER": return GameOver(st);
        }
        if (st.Obj("run") is null) return Decision.Paused($"screen {screen} without a run object");
        return screen switch
        {
            "MAP" => await MapAsync(st),
            "CARD_REWARD" => await CardRewardAsync(st),
            "REWARDS" => await RewardsAsync(st),
            "EVENT" or "ANCIENT" => await EventAsync(st, screen),
            "REST" => await RestAsync(st),
            "SHOP" => await ShopAsync(st),
            "CARD_SELECT" => await CardSelectAsync(st),
            "COMBAT" => await CombatAsync(st),
            "COMBAT_SELECT" => await CombatSelectAsync(st),
            "TREASURE" => Treasure(st),
            "CRYSTAL_SPHERE" => Sphere(st),
            "OTHER" when st.Str("screen_detail") == "NChooseARelicSelection" => await RelicChoiceAsync(st),
            "OTHER" when (st.Str("screen_detail") ?? "").StartsWith("modal:") =>
                Decision.Paused($"a popup blocks the game ({st.Str("screen_detail")}): close it (tip: switch tutorials off), then resume"),
            _ => Decision.Paused($"screen {screen} ({st.Str("screen_detail")}) is not handled by the controller"),
        };
    }

    private Decision GameOver(JsonObject st)
    {
        var o = Opts(st, "to_menu");
        var d = o.Count > 0 ? Decision.Act(o[0].I, "game over") : Decision.Idle("game over (no to_menu option)");
        d.Info["game_over"] = true;
        return d;
    }

    private readonly Random _sphereRng = new();

    /// <summary>Crystal Sphere minigame: a uniformly random hidden cell while divinations are left (2026-11-20:
    /// the minigame is pure chance; the game's own AutoSlay plays it the same way), then proceed. The rewards it
    /// gives open the normal rewards / card-reward screens, decided by the agent.</summary>
    private Decision Sphere(JsonObject st)
    {
        var cells = Opts(st, "sphere_cell");
        if (cells.Count > 0)
        {
            var (i, o) = cells[_sphereRng.Next(cells.Count)];
            return Decision.Act(i, $"crystal sphere: random cell ({o.Int("x", -1)},{o.Int("y", -1)}), {o.Int("divinations", -1)} divinations left");
        }
        if (Opts(st, "proceed") is { Count: > 0 } p) return Decision.Act(p[0].I, "crystal sphere: proceed");
        return Decision.Paused("crystal sphere without cells or proceed");
    }

    private static Decision Treasure(JsonObject st)
    {
        foreach (string k in new[] { "open", "relic", "proceed" })
            if (Opts(st, k) is { Count: > 0 } o) return Decision.Act(o[0].I, "treasure: " + k + " (the simulator takes the chest's relic)");
        return Decision.Paused("treasure screen without open / relic / proceed");
    }

    // ------------------------------------------------------------------ map

    private async Task<Decision> MapAsync(JsonObject st)
    {
        Imported imp = Bind(st, false);
        var o = Opts(st, "map");
        if (o.Count == 0) return Decision.Paused("map screen without map options");
        var pts = new List<MapPoint>();
        foreach (var (_, x) in o)
        {
            MapPoint? pt = imp.Run.Map.GetPoint(new MapCoord(x.Int("col"), x.Int("row")));
            if (pt is null) return Decision.Paused($"map option {x.Int("col")},{x.Int("row")} is not on the imported map");
            pts.Add(pt);
        }
        MapPoint pick = await Agent!.ChooseMapPointAsync(pts);
        int k = pts.FindIndex(p => ReferenceEquals(p, pick));
        var d = Decision.Act(o[k].I, $"map {pick.coord.col},{pick.coord.row} {pick.PointType}");
        d.Info["notes"] = imp.Notes;
        return d;
    }

    // ------------------------------------------------------------------ rewards

    private static CardSpec SpecOf(JsonObject opt) => CardSpec.Read(opt.Obj("card") ?? opt);

    private static void Resolved(Reward r) => Priv.SetProp(r, "IsResolved", true);

    private static RewardsSet MakeRewards(Sts2Sim.Core.Entities.Players.Player p, CardReward card, PotionReward? potion)
    {
        var gold = new GoldReward(0, p);
        Resolved(gold);
        return (RewardsSet)Priv.New(typeof(RewardsSet), gold, potion, card, null, (IReadOnlyList<Reward>)Array.Empty<Reward>());
    }

    private async Task<Decision> CardRewardAsync(JsonObject st)
    {
        Imported imp = Bind(st, false);
        var o = Opts(st, "card");
        var skip = Opts(st, "skip", "proceed");
        if (o.Count == 0) return skip.Count > 0 ? Decision.Act(skip[0].I, "card reward without cards") : Decision.Paused("card reward without card options");
        var cards = new List<CardModel>();
        foreach (var (_, x) in o)
        {
            try { cards.Add(RunImport.CreateCard(imp.P, SpecOf(x))); }
            catch (Exception e) { return Decision.Paused($"card reward: card {SpecOf(x).Id} not in the simulator ({e.Message})"); }
        }
        var reward = new CardReward(imp.P, cards);
        Priv.SetProp(reward, "Options", (IReadOnlyList<CardModel>)cards);
        RewardDecision dec = await Agent!.ChooseRewardActionAsync(MakeRewards(imp.P, reward, null));
        switch (dec)
        {
            case RewardDecision.TakeCard t:
                int k = cards.FindIndex(c => ReferenceEquals(c, t.Card));
                if (k >= 0) return Decision.Act(o[k].I, $"take card {t.Card.Id}+{t.Card.CurrentUpgradeLevel}");
                break;
            case RewardDecision.SkipCard:
                _handled.Add("card-skipped");
                return skip.Count > 0 ? Decision.Act(skip[0].I, "skip card reward") : Decision.Paused("the agent skips this card reward but the screen offers no skip");
        }
        return Decision.Paused($"card reward: unmapped decision {dec}");
    }

    private async Task<Decision> RewardsAsync(JsonObject st)
    {
        var o = Opts(st, "reward");
        var proceed = Opts(st, "proceed", "skip");
        Imported imp = Bind(st, false);
        // RewardDecisionClassifier's order: gold, potion, relic, the card reward, then extras (taken, as RunAgent does)
        int Rank(string t) => t switch { "GOLD" => 0, "POTION" => 1, "RELIC" => 2, "CARD" => 3, _ => 4 };
        var seen = new Dictionary<string, int>();
        var rows = new List<(int I, JsonObject O, string Type, string Key)>();
        foreach (var (i, x) in o)
        {
            string t = (x.Str("type") ?? "").ToUpperInvariant();
            string baseKey = $"{t}:{x.Str("id")}";
            int n = seen[baseKey] = seen.GetValueOrDefault(baseKey) + 1;
            rows.Add((i, x, t, $"{baseKey}#{n}"));
        }
        // a handled reward that is no longer listed is gone: forget it (a later reward screen may reuse its key)
        _handled.RemoveWhere(k => k.Contains(':') && !rows.Any(r => r.Key == k));
        foreach (var (i, x, t, key) in rows.OrderBy(r => Rank(r.Type)))
        {
            if (_handled.Contains(key)) continue;
            if (t == "POTION")
            {
                string pid = J.ModelId(x.Str("id"), "POTION");
                if (imp.P.PotionSlots.Contains(null)) return Decision.Act(i, "take potion " + pid);
                // full belt: RunAgent's belt rule (potion_mix / potion_values) may drop a held potion first
                PotionModel? canon = ModelDb.GetByIdOrNull<PotionModel>(ModelId.Deserialize(pid));
                if (canon is null) { _handled.Add(key); continue; }
                var before = imp.P.PotionSlots.Select(q => q?.Id.ToString()).ToList();
                var empty = new CardReward(imp.P, Array.Empty<CardModel>());
                Resolved(empty);
                await Agent!.ChooseRewardActionAsync(MakeRewards(imp.P, empty, new PotionReward((PotionModel)canon.MutableClone(), imp.P)));
                var after = imp.P.PotionSlots.Select(q => q?.Id.ToString()).ToList();
                int drop = Enumerable.Range(0, before.Count).FirstOrDefault(k => before[k] is not null && after[k] is null, -1);
                if (drop < 0) { _handled.Add(key); Say($"potion {pid} left (belt full; the agent keeps its potions)"); continue; }
                var disc = Opts(st, "discard_potion").Where(d => d.O.Int("slot", -1) == drop).ToList();
                if (disc.Count > 0) return Decision.Act(disc[0].I, $"discard {before[drop]} (slot {drop}) to take {pid}");
                _handled.Add(key);
                return Decision.Paused($"the agent drops {before[drop]} (potion slot {drop}) to take {pid}: discard it, take the potion, then resume");
            }
            if (t == "CARD") _handled.Add(key);   // opened once; skipped cards stay on the list
            else if (t is not ("GOLD" or "RELIC")) _handled.Add(key);
            return Decision.Act(i, $"take reward {t} {x.Str("id")}");
        }
        return proceed.Count > 0 ? Decision.Act(proceed[0].I, "rewards done") : Decision.Paused("rewards: nothing left to take and no proceed option");
    }

    // ------------------------------------------------------------------ events

    /// <summary>The simulator's option key: the last segment of the game's text key ("X.pages.P.options.KEY" -> KEY).</summary>
    public static string EventKey(string? key)
    {
        if (string.IsNullOrEmpty(key)) return "";
        int i = key.LastIndexOf('.');
        string k = i >= 0 ? key[(i + 1)..] : key;
        return k.EndsWith("_LOCKED", StringComparison.Ordinal) ? k[..^7] : k;
    }

    /// <summary>
    /// Normalises a live game event option's key to the exact <see cref="EventOption.Key"/> used by the simulator
    /// (and hashed by <see cref="ML.EventBucket"/>):
    /// <list type="bullet">
    /// <item><c>DOLL_ROOM</c> page 2: the game puts the localised relic title in <c>TextKey</c>, while the simulator
    /// uses <c>relic.Id.Entry</c> (<c>MR_STRUGGLES</c>, <c>BING_BONG</c>, <c>DAUGHTER_OF_THE_WIND</c>); resolved
    /// language-independently from <c>title_key</c> (<c>&lt;RELIC_ID&gt;.title</c>).</item>
    /// <item><c>TABLET_OF_TRUTH</c> page 1: <c>DECIPHER_1</c> -> <c>DECIPHER</c>.</item>
    /// <item><c>COLOSSAL_FLOWER</c> pages 1-2: <c>EXTRACT_CURRENT_PRIZE_N</c> -> <c>EXTRACT_CURRENT_PRIZE</c>,
    /// <c>REACH_DEEPER_N</c> -> <c>REACH_DEEPER</c>.</item>
    /// <item><c>TRIAL</c> verdict pages: <c>TRIAL.pages.&lt;DEFENDANT&gt;.options.GUILTY|INNOCENT</c> ->
    /// <c>&lt;DEFENDANT&gt;_GUILTY|INNOCENT</c>.</item>
    /// </list>
    /// </summary>
    public static string EventKey(string? eventId, string? key, string? titleKey = null, string? relic = null)
    {
        string k = EventKey(key);
        if (string.IsNullOrEmpty(eventId)) return k;
        string ev = J.ModelId(eventId, "EVENT");
        if (ev == "EVENT.DOLL_ROOM" && (key is null || !key.Contains(".options.", StringComparison.Ordinal)) && k != "PROCEED")
        {
            if (!string.IsNullOrEmpty(relic)) return J.ModelId(relic, "RELIC")[6..];
            if (!string.IsNullOrEmpty(titleKey))
            {
                int dot = titleKey.IndexOf('.');
                return dot > 0 ? titleKey[..dot] : titleKey;
            }
        }
        if (ev == "EVENT.TABLET_OF_TRUTH" && k.StartsWith("DECIPHER_", StringComparison.Ordinal))
            return "DECIPHER";
        if (ev == "EVENT.COLOSSAL_FLOWER")
        {
            if (k.StartsWith("EXTRACT_CURRENT_PRIZE_", StringComparison.Ordinal)) return "EXTRACT_CURRENT_PRIZE";
            if (k.StartsWith("REACH_DEEPER_", StringComparison.Ordinal)) return "REACH_DEEPER";
        }
        if (ev == "EVENT.TRIAL" && k is "GUILTY" or "INNOCENT" && !string.IsNullOrEmpty(key))
        {
            string[] parts = key.Split('.');
            int pi = Array.IndexOf(parts, "pages");
            if (pi >= 0 && pi + 1 < parts.Length && !string.IsNullOrEmpty(parts[pi + 1]))
                return parts[pi + 1] + "_" + k;
        }
        return k;
    }

    /// <summary>The event of an EVENT / ANCIENT screen: screen_detail (the game's ModelId.Entry, "EVENT.X" accepted), else
    /// the last event room of the run save's map point history.</summary>
    private static string? CurrentEventId(JsonObject st)
    {
        string? det = st.Str("screen_detail");
        if (st.Str("screen") is "EVENT" or "ANCIENT" && !string.IsNullOrEmpty(det) && !det.Contains(' ') && !det.Contains(':'))
            return J.ModelId(det, "EVENT");
        var hist = st.Obj("run")?.Obj("save")?.Arr("map_point_history");
        var last = (hist?.LastOrDefault() as JsonArray)?.LastOrDefault() as JsonObject;
        var room = last?.Objs("rooms").LastOrDefault(r => J.TryEnum(r.At("room_type"), out RoomType t) && t == RoomType.Event);
        return room?.Str("model_id") is { } m ? J.ModelId(m, "EVENT") : null;
    }

    private async Task<Decision> EventAsync(JsonObject st, string screen)
    {
        var o = Opts(st, "event_option");
        if (o.Count == 0) return Decision.Paused($"{screen.ToLowerInvariant()} without event options");
        string? evId = CurrentEventId(st);
        // The Architect: the ending scene after the final boss (the run is won); its lines and replies change nothing the
        // agent values, so the first open option is taken until the victory screen (first live run, 2026-11-20: paused here)
        if (evId is not null && evId.EndsWith("THE_ARCHITECT"))
        {
            var open = o.FirstOrDefault(x => !x.O.Bool("locked"));
            return Decision.Act(open.O is null ? o[0].I : open.I, "the Architect (run won): " + ((open.O ?? o[0].O).Str("key") ?? "option"));
        }
        Imported imp = Bind(st, false);
        EventModel? ev = evId is null ? null : ModelDb.GetByIdOrNull<EventModel>(ModelId.Deserialize(evId));
        if (ev is null) return Decision.Paused($"event {evId ?? "(unknown id)"} is not in the simulator");
        if (screen == "ANCIENT" && imp.Run.CurrentMapPoint?.PointType != MapPointType.Ancient && imp.Run.VisitedMapCoords.Count == 0)
        {
            imp.Run.AddVisitedMapCoord(imp.Run.Map.StartingMapPoint.coord);
            Priv.Set(imp.Run, "_completedActFloors", Math.Max(0, imp.Run.TotalFloor - 1 - imp.Run.VisitedMapCoords.Count));
            imp.Notes.Add("ancient: start point added to the visited coordinates");
        }
        var room = new EventRoom(() => ev);
        Priv.SetProp(room, "Event", ev);
        imp.Run.PushRoom(room);
        var sim = new List<EventOption>();
        foreach (var (_, x) in o)
        {
            string? relic = x.Str("relic");
            VisibleContent? content = string.IsNullOrEmpty(relic) ? null
                : new VisibleContent(J.ModelId(relic, "RELIC")[6..], ImmutableArray<VisibleValue>.Empty, ImmutableArray<VisibleContent>.Empty);
            bool locked = x.Bool("locked") || (x.Str("key")?.EndsWith("_LOCKED", StringComparison.Ordinal) ?? false);
            sim.Add(new EventOption(EventKey(evId, x.Str("key"), x.Str("title_key"), relic), locked ? null : () => Task.CompletedTask, content));
        }
        EventOption pick = await Agent!.ChooseEventOptionAsync(sim);
        int k = sim.FindIndex(e => ReferenceEquals(e, pick));
        if (k < 0) return Decision.Paused($"event {evId}: the agent's option {pick.Key} is not on screen");
        if (o[k].O.Bool("locked")) return Decision.Paused($"event {evId}: no open option");
        return Decision.Act(o[k].I, $"event {evId}: {pick.Key}" + (pick.OfferedContent is { } c ? " (relic " + c.ContentId + ")" : ""));
    }

    /// <summary>"Choose a relic" screen (OTHER / NChooseARelicSelection): RunAgent has no such callback (the simulator
    /// resolves relic rewards automatically); the relics go to the macro network's Ancient head, the S1 head for a choice
    /// among relics, through RunAgent's own option sampler (greedy in evaluation). Skip is never chosen.</summary>
    private async Task<Decision> RelicChoiceAsync(JsonObject st)
    {
        Bind(st, false);
        var o = Opts(st, "relic");
        if (o.Count == 0) return Opts(st, "skip") is { Count: > 0 } sk ? Decision.Act(sk[0].I, "relic choice without relics") : Decision.Paused("relic choice without relic options");
        if (o.Count == 1) return Decision.Act(o[0].I, "relic " + o[0].O.Str("id"));
        var row = (MacroRow)Priv.Call(Agent!, "StateRow")!;
        foreach (var (_, x) in o) row.AddOption(ML.KRelic, _s.MVocab.Tok("relic", J.ModelId(x.Str("id"), "RELIC")));
        int k = await (Task<int>)Priv.Call(Agent!, "DecideOptionsAsync", row, ML.DAncient, null, null, null)!;
        return Decision.Act(o[k].I, $"relic choice (Ancient head): {o[k].O.Str("id")}");
    }

    // ------------------------------------------------------------------ rest site

    private async Task<Decision> RestAsync(JsonObject st)
    {
        Imported imp = Bind(st, false);
        var o = Opts(st, "rest_option");
        if (o.Count == 0) return Opts(st, "proceed") is { Count: > 0 } pr ? Decision.Act(pr[0].I, "rest site done") : Decision.Paused("rest site without options");
        var cands = new List<RestSiteDecision>();
        var owner = new List<int>();
        for (int j = 0; j < o.Count; j++)
        {
            string id = (o[j].O.Str("id") ?? "").ToLowerInvariant();
            bool enabled = o[j].O.Bool("enabled", !o[j].O.Bool("locked"));
            if (id == "heal" && enabled) { cands.Add(new RestSiteDecision.Heal()); owner.Add(j); }
            else if (id == "smith" && enabled)
                foreach (CardModel c in imp.P.Deck.Cards.Where(c => c.IsUpgradable)) { cands.Add(new RestSiteDecision.Smith(c)); owner.Add(j); }
            else { cands.Add(new LiveRestOption(id, enabled)); owner.Add(j); }
        }
        RestSiteDecision pick = await Agent!.ChooseRestSiteActionAsync(imp.P, cands);
        string cap = Agent.LastKindleCap is { } kc ? $" [{kc}]" : "";
        int k = cands.FindIndex(c => ReferenceEquals(c, pick));
        if (k < 0) return Decision.Paused($"rest: unmapped decision {pick.OptionId}");
        if (pick is RestSiteDecision.Smith sm)
        {
            int di = imp.Deck.ToList().FindIndex(c => ReferenceEquals(c, sm.Card));
            _pending = ("upgrade", CardSpec.KeyOf(sm.Card), di, imp.Run.TotalFloor);
            return Decision.Act(o[owner[k]].I, $"rest: smith {sm.Card.Id}+{sm.Card.CurrentUpgradeLevel} (deck {di}){cap}");
        }
        return Decision.Act(o[owner[k]].I, "rest: " + pick.OptionId + cap);
    }

    // ------------------------------------------------------------------ shop

    private async Task<Decision> ShopAsync(JsonObject st)
    {
        Imported imp = Bind(st, false);
        var buys = Opts(st, "buy");
        var leave = Opts(st, "leave", "proceed");
        var cardE = new List<MerchantCardEntry>(); var relicE = new List<MerchantRelicEntry>(); var potE = new List<MerchantPotionEntry>();
        var of = new Dictionary<object, int>(ReferenceEqualityComparer.Instance);
        MerchantCardRemovalEntry? removal = null;
        foreach (var (i, x) in buys)
        {
            string item = (x.Str("item") ?? "").ToLowerInvariant();
            int price = x.Int("price");
            try
            {
                switch (item)
                {
                    case "card": { var e = new MerchantCardEntry(RunImport.CreateCard(imp.P, SpecOf(x)), price); cardE.Add(e); of[e] = i; break; }
                    case "relic":
                    {
                        var canon = ModelDb.GetByIdOrNull<RelicModel>(ModelId.Deserialize(J.ModelId(x.Str("id"), "RELIC")));
                        if (canon is null) { Say("shop: unknown relic " + x.Str("id") + " not offered to the agent"); break; }
                        var e = new MerchantRelicEntry((RelicModel)canon.MutableClone(), price); relicE.Add(e); of[e] = i; break;
                    }
                    case "potion":
                    {
                        var canon = ModelDb.GetByIdOrNull<PotionModel>(ModelId.Deserialize(J.ModelId(x.Str("id"), "POTION")));
                        if (canon is null) { Say("shop: unknown potion " + x.Str("id") + " not offered to the agent"); break; }
                        var e = new MerchantPotionEntry((PotionModel)canon.MutableClone(), price); potE.Add(e); of[e] = i; break;
                    }
                    case "remove": case "removal": case "card_removal": { removal = new MerchantCardRemovalEntry(price); of[removal] = i; break; }
                }
            }
            catch (Exception e) { Say($"shop: {item} {x.Str("id")} not offered to the agent ({e.Message})"); }
        }
        if (removal is null) { removal = new MerchantCardRemovalEntry(0); removal.MarkPurchased(); }
        var inv = (MerchantInventory)Priv.New(typeof(MerchantInventory), imp.P, cardE, relicE, potE, removal);
        ShopDecision dec = await Agent!.ChooseShopActionAsync(inv, imp.P);
        int L() => leave.Count > 0 ? leave[0].I : -1;
        switch (dec)
        {
            case ShopDecision.BuyCard b: return Decision.Act(of[b.Entry], $"shop: buy card {b.Entry.Card.Id}+{b.Entry.Card.CurrentUpgradeLevel} ({b.Entry.Price})");
            case ShopDecision.BuyRelic b: return Decision.Act(of[b.Entry], $"shop: buy relic {b.Entry.Relic.Id} ({b.Entry.Price})");
            case ShopDecision.BuyPotion b: return Decision.Act(of[b.Entry], $"shop: buy potion {b.Entry.Potion.Id} ({b.Entry.Price})");
            case ShopDecision.BuyCardRemoval r:
                int di = imp.Deck.ToList().FindIndex(c => ReferenceEquals(c, r.CardToRemove));
                _pending = ("remove", CardSpec.KeyOf(r.CardToRemove), di, imp.Run.TotalFloor);
                return Decision.Act(of[removal], $"shop: card removal of {r.CardToRemove.Id}+{r.CardToRemove.CurrentUpgradeLevel} (deck {di})");
            default:
                return L() >= 0 ? Decision.Act(L(), "shop: leave") : Decision.Paused("the agent leaves the shop but there is no leave option");
        }
    }

    // ------------------------------------------------------------------ deck selections

    private static readonly System.Threading.AsyncLocal<string?> PurposeSlot =
        (System.Threading.AsyncLocal<string?>)Priv.GetStatic(typeof(CardSelectCmd), "ScopedDeckPurpose")!;

    /// <summary>The simulator's deck purpose ("upgrade", "remove", "transform") of a protocol purpose text; null = none.</summary>
    public static string? Purpose(string? p)
    {
        p = (p ?? "").ToLowerInvariant();
        if (p.Contains("upgrade") || p.Contains("smith")) return "upgrade";
        if (p.Contains("transform")) return "transform";
        if (p.Contains("remove") || p.Contains("purge")) return "remove";
        return null;
    }

    private static bool LastRoomIsEvent(JsonObject st)
    {
        var hist = st.Obj("run")?.Obj("save")?.Arr("map_point_history");
        var last = (hist?.LastOrDefault() as JsonArray)?.LastOrDefault() as JsonObject;
        var room = last?.Objs("rooms").LastOrDefault();
        return room is not null && J.TryEnum(room.At("room_type"), out RoomType t) && t == RoomType.Event;
    }

    private static AbstractModel? SourceModel(string? id)
    {
        if (string.IsNullOrEmpty(id) || !id.Contains('.')) return null;
        try
        {
            ModelId mid = ModelId.Deserialize(id);
            return (AbstractModel?)ModelDb.GetByIdOrNull<RelicModel>(mid) ?? (AbstractModel?)ModelDb.GetByIdOrNull<EventModel>(mid)
                ?? (AbstractModel?)ModelDb.GetByIdOrNull<CardModel>(mid) ?? ModelDb.GetByIdOrNull<PotionModel>(mid);
        }
        catch { return null; }
    }

    private async Task<Decision> CardSelectAsync(JsonObject st)
    {
        Imported imp = Bind(st, false);
        var sel = Opts(st, "select");
        var skip = Opts(st, "skip", "cancel");
        if (sel.Count == 0) return skip.Count > 0 ? Decision.Act(skip[0].I, "selection without select option") : Decision.Paused("card selection without a select option");
        var (si, so) = sel[0];
        var specs = so.Objs("cards").Select(CardSpec.Read).ToList();
        int min = so.Int("min", 1), max = so.Int("max", 1);
        string? purpose = Purpose(so.Str("purpose"));
        // a card already chosen by the agent (rest-site smith, shop removal)
        if (_pending is { } pd && pd.Floor == imp.Run.TotalFloor && (purpose is null || purpose == pd.Purpose))
        {
            _pending = null;
            int k = specs.FindIndex(c => c.Key == pd.Key);
            var withIdx = so.Objs("cards").Select((c, j) => (j, c.IntOrNull("deck_index"))).Where(x => x.Item2 == pd.DeckIndex).Select(x => x.j).ToList();
            if (withIdx.Count > 0 && specs[withIdx[0]].Key == pd.Key) k = withIdx[0];
            if (k >= 0) return new Decision { Option = si, Cards = new() { k }, Why = $"{pd.Purpose}: {specs[k].Id}+{specs[k].Upgrade} (chosen before)" };
            Say($"pending {pd.Purpose} card {pd.Key} not on the selection screen: asking the agent again");
        }
        // candidates: the imported deck's cards when every offered card is in the deck, else new cards
        var deck = imp.Deck.ToList();
        var usedDeck = new HashSet<CardModel>(ReferenceEqualityComparer.Instance);
        var cands = new List<CardModel>();
        bool inDeck = so.Str("source") != "generated";   // "choose one of N new cards": never deck cards
        var nodes = so.Objs("cards").ToList();
        for (int j = 0; j < specs.Count && inDeck; j++)   // deck cards, matched in order by id, upgrade and enchantment
        {
            int? di = nodes[j].IntOrNull("deck_index");
            CardModel? m = di is { } d && d >= 0 && d < deck.Count && CardSpec.KeyOf(deck[d]) == specs[j].Key && !usedDeck.Contains(deck[d]) ? deck[d]
                : deck.FirstOrDefault(c => !usedDeck.Contains(c) && CardSpec.KeyOf(c) == specs[j].Key);
            if (m is null) inDeck = false; else { usedDeck.Add(m); cands.Add(m); }
        }
        if (!inDeck)
        {
            cands.Clear();
            foreach (CardSpec c in specs)
            {
                try { cands.Add(RunImport.CreateCard(imp.P, c)); }
                catch (Exception e) { return Decision.Paused($"selection: card {c.Id} not in the simulator ({e.Message})"); }
            }
        }
        // the selection's source model (deck_route reads its type): `source_model` if the mod sends it, else the event of
        // the room the selection opens in (the mod's `source` is hand / pile / deck / generated)
        AbstractModel? srcModel = SourceModel(J.ModelId(so.Str("source_model"), "RELIC"))
            ?? (CurrentEventId(st) is { } ev && imp.Run.CurrentMapPoint is not null && LastRoomIsEvent(st) ? SourceModel(ev) : null);
        var request = new CardSelectionRequest(imp.P, cands, Math.Min(min, cands.Count), Math.Min(max, cands.Count), srcModel,
            Cancelable: skip.Count > 0);
        string? prev = PurposeSlot.Value;
        PurposeSlot.Value = purpose;
        IReadOnlyList<CardModel> picked;
        long randomBefore = _s.Services.SelectionKinds.Values.Sum();
        try { picked = await Agent!.ChooseCardsAsync(request); }
        finally { PurposeSlot.Value = prev; }
        var idx = picked.Select(c => cands.FindIndex(x => ReferenceEquals(x, c))).ToList();
        if (idx.Any(i => i < 0)) return Decision.Paused("selection: the agent's answer is not among the offered cards");
        bool random = purpose is null && !(!inDeck && max == 1) && !(_s.Cfg.DeckRoute);
        string why = $"select ({so.Str("purpose")}{(purpose is null ? "" : " -> " + purpose)}, source {so.Str("source")}): " +
            (idx.Count == 0 ? "none" : string.Join(", ", idx.Select(i => $"{specs[i].Id}+{specs[i].Upgrade}"))) + (random ? " [RunAgent answers this prompt at random]" : "");
        if (idx.Count == 0 && skip.Count > 0) return Decision.Act(skip[0].I, why);
        if (idx.Count < min) return Decision.Paused($"selection: the agent picked {idx.Count} cards, the screen needs at least {min}");
        return new Decision { Option = si, Cards = idx, Why = why };
    }

    // ------------------------------------------------------------------ combat

    private async Task<Shadow> EnsureFightAsync(JsonObject st)
    {
        JsonObject combat = st.Obj("combat")!;
        string enc = J.ModelId(combat.Str("encounter_id"), "ENCOUNTER");
        string key = $"{st.Obj("run")!.Int("floor")}:{enc}";
        J.TryEnum(combat.At("room_type"), out RoomType rt);
        if (rt is not (RoomType.Monster or RoomType.Elite or RoomType.Boss)) rt = RoomType.Monster;
        if (Fight is not null && Fight.Key == key && Fight.Alive) return Fight;
        bool restart = Fight is not null && Fight.Key == key;
        object? keepFight = restart && Agent is not null ? Priv.Get(Agent, "_fight") : null;
        if (!restart) EndFight();
        Imported imp = Bind(st, inCombat: true);
        var sh = new Shadow { Imp = imp, Agent = Agent!, Key = key, Room = rt, Encounter = enc, Restarts = (Fight?.Restarts ?? -1) + 1 };
        await sh.StartAsync(st.Obj("combat")!);
        if (keepFight is not null)
        {
            Priv.Set(keepFight, "Engine", sh.State!.Engine);   // RunAgent keeps its T_f table and search for this fight
            Say($"shadow fight rebuilt ({Fight!.BrokenWhy ?? "the copy's fight ended"})");
        }
        Fight = sh;
        return sh;
    }

    private async Task<Decision> CombatAsync(JsonObject st)
    {
        JsonObject? combat = st.Obj("combat");
        if (combat is null) return Decision.Paused("combat screen without a combat object");
        if (combat.Str("turn_side") is "ENEMY") return Decision.Idle("enemy turn");
        Shadow sh = await EnsureFightAsync(st);
        List<string> notes = sh.Sync(combat, st.Obj("run"));
        CombatDecision d = await Agent!.ChooseCombatActionAsync(sh.State!);
        sh.PlannedKeys.Clear();
        sh.PlannedKeys.AddRange((Queue<string[]>)Priv.Get(Agent, "_selScript")!);
        Decision res = MapCombat(st, combat, sh, d);
        res.Combat = d;
        res.Info["sync"] = notes;
        if (sh.Restarts > 0) res.Info["shadow_restarts"] = sh.Restarts;
        return res;
    }

    private Decision MapCombat(JsonObject st, JsonObject combat, Shadow sh, CombatDecision d)
    {
        var p = sh.State!.Players[0];
        switch (d)
        {
            case CombatDecision.PlayCard pc:
            {
                int hi = p.PlayerCombatState!.Hand.Cards.ToList().FindIndex(c => ReferenceEquals(c, pc.Card));
                var plays = Opts(st, "play");
                var hit = plays.Where(x => x.O.Int("hand_index", -1) == hi).ToList();
                if (hit.Count == 0 || CardSpec.Read(hit[0].O.Obj("card") ?? hit[0].O).Id != pc.Card.Id.ToString())
                    hit = plays.Where(x => CardSpec.Read(x.O.Obj("card") ?? x.O).Key == CardSpec.KeyOf(pc.Card)).ToList();
                if (hit.Count == 0) return Decision.Paused($"combat: the agent plays {pc.Card.Id} (hand {hi}) but the game offers no such play");
                var dec = Decision.Act(hit[0].I, $"play {pc.Card.Id}+{pc.Card.CurrentUpgradeLevel}" + (pc.Target is { } t ? $" -> {t.Monster?.Id.Entry ?? "player"}" : ""));
                return WithTarget(dec, hit[0].O, pc.Target, combat);
            }
            case CombatDecision.UsePotion up:
            {
                int slot = p.PotionSlots.ToList().FindIndex(x => ReferenceEquals(x, up.Potion));
                var hit = Opts(st, "potion").Where(x => x.O.Int("slot", -1) == slot).ToList();
                if (hit.Count == 0) hit = Opts(st, "potion").Where(x => J.ModelId(x.O.Str("id"), "POTION") == up.Potion.Id.ToString()).ToList();
                if (hit.Count == 0) return Decision.Paused($"combat: the agent uses {up.Potion.Id} (slot {slot}) but the game offers no such potion use");
                var dec = Decision.Act(hit[0].I, $"potion {up.Potion.Id}" + (up.Target is { } t ? $" -> {t.Monster?.Id.Entry ?? "player"}" : ""));
                return WithTarget(dec, hit[0].O, up.Target, combat);
            }
            case CombatDecision.EndTurn:
            {
                var e = Opts(st, "end_turn");
                return e.Count > 0 ? Decision.Act(e[0].I, "end turn") : Decision.Paused("combat: the agent ends the turn but there is no end_turn option");
            }
        }
        return Decision.Paused($"combat: unmapped decision {d}");
    }

    private Decision WithTarget(Decision d, JsonObject opt, Sts2Sim.Core.Entities.Creatures.Creature? target, JsonObject combat)
    {
        var targets = opt.Arr("targets").Select(x => x is JsonValue v && v.TryGetValue(out int k) ? k : int.MinValue).Where(k => k != int.MinValue).ToList();
        if (target is null || targets.Count == 0) return d;
        int? ti = Fight!.IndexOf(target, combat);
        if (ti is { } t && targets.Contains(t)) { d.Target = t; return d; }
        if (target.Monster is null && targets.Count == 1) { d.Target = targets[0]; return d; }
        return Decision.Paused($"combat: target {target.Monster?.Id.ToString() ?? "player"} of the agent's action is not among the option's targets");
    }

    private async Task<Decision> CombatSelectAsync(JsonObject st)
    {
        var sel = Opts(st, "select");
        var skip = Opts(st, "skip", "cancel");
        if (sel.Count == 0) return skip.Count > 0 ? Decision.Act(skip[0].I, "combat selection without select option") : Decision.Paused("combat selection without a select option");
        var (si, so) = sel[0];
        var specs = so.Objs("cards").Select(CardSpec.Read).ToList();
        int min = so.Int("min", 1), max = so.Int("max", specs.Count);
        // 1. the answer RunAgent gave to the same prompt in the shadow fight, or the prompt answers of its search action
        foreach (var keys in (Fight?.SimAnswers ?? new()).Concat(Fight?.PlannedKeys ?? new()).ToList())
        {
            var used = new HashSet<int>();
            var idx = new List<int>();
            foreach (string k in keys)
            {
                string[] parts = k.Split('#');
                int j = Enumerable.Range(0, specs.Count).FirstOrDefault(i => !used.Contains(i) && specs[i].Id == parts[0] && specs[i].Upgrade.ToString() == parts.ElementAtOrDefault(1), -1);
                if (j < 0) { idx = null!; break; }
                used.Add(j); idx.Add(j);
            }
            if (idx is null || idx.Count < Math.Min(min, specs.Count) || idx.Count > max) continue;
            Fight!.SimAnswers.Remove(keys); Fight.PlannedKeys.Remove(keys);
            if (idx.Count == 0 && skip.Count > 0) return Decision.Act(skip[0].I, "combat selection: none (agent)");
            return new Decision { Option = si, Cards = idx, Why = "combat selection (agent): " + string.Join(", ", idx.Select(i => specs[i].Id)) };
        }
        // 2. RunAgent's own answer to the prompt built from the game's cards (live combat: its script or at random)
        if (Fight?.State is null || Agent is null) return Decision.Paused("combat selection outside a known fight");
        var p = Fight.State.Players[0];
        var cands = new List<CardModel>();
        foreach (CardSpec c in specs)
        {
            try { cands.Add(RunImport.CreateCard(p, c)); }
            catch (Exception e) { return Decision.Paused($"combat selection: card {c.Id} not in the simulator ({e.Message})"); }
        }
        var req = new CardSelectionRequest(p, cands, Math.Min(min, cands.Count), Math.Min(max, cands.Count), null, Cancelable: skip.Count > 0);
        IReadOnlyList<CardModel> picked = await Agent.ChooseCardsAsync(req);
        var ix = picked.Select(c => cands.FindIndex(x => ReferenceEquals(x, c))).ToList();
        if (ix.Count == 0 && skip.Count > 0) return Decision.Act(skip[0].I, "combat selection: none");
        return new Decision { Option = si, Cards = ix, Why = "combat selection (RunAgent, no matching search answer: its script or at random): " + string.Join(", ", ix.Select(i => specs[i].Id)) };
    }

    /// <summary>After the game accepted an action: the shadow fight follows it.</summary>
    public async Task AfterSentAsync(Decision d)
    {
        if (d.Combat is not null && Fight is not null) await Fight.ApplyAsync(d.Combat);
    }
}
