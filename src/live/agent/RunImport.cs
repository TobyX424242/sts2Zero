using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Content.Acts;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.Core.Runs.Transplant;
using Sts2Sim.S3;

namespace Sts2Sim.Live;

/// <summary>One card as the game reports it (run save or protocol CARD).</summary>
public sealed record CardSpec(string Id, int Upgrade, string? Ench, int EnchAmount, int? Floor, JsonNode? Props)
{
    /// <summary>A card of the run save (SerializableCard) or of the protocol (CARD / deck entry).</summary>
    public static CardSpec Read(JsonObject c)
    {
        string id = J.ModelId(c.Str("id"), "CARD");
        int upg = c.IntOrNull("current_upgrade_level") ?? c.Int("upgrade");
        string? ench = null; int amt = 0;
        if (c.At("enchantment") is JsonObject e) { ench = J.ModelId(e.Str("id"), "ENCHANTMENT"); amt = e.Int("amount"); }
        else if (c.Str("enchantment") is { Length: > 0 } es) { ench = J.ModelId(es, "ENCHANTMENT"); amt = c.Int("enchantment_amount"); }
        return new CardSpec(id, upg, string.IsNullOrEmpty(ench) ? null : ench, amt, c.IntOrNull("floor_added_to_deck") ?? c.IntOrNull("floor_added"), c.At("props"));
    }

    /// <summary>Identity used to match game cards with simulator cards: id, upgrade, enchantment.</summary>
    public string Key => $"{Id}+{Upgrade}~{Ench}:{(Ench is null ? 0 : EnchAmount)}";

    public static string KeyOf(CardModel c)
    {
        var e = c.Enchantments.Count > 0 ? c.Enchantments[0] : null;
        return $"{c.Id}+{c.CurrentUpgradeLevel}~{e?.Id.ToString()}:{(e is null ? 0 : (int)e.Magnitude)}";
    }
}

/// <summary>A simulator run rebuilt from the game (see <see cref="RunImport"/>).</summary>
public sealed class Imported
{
    public required RunState Run;
    public required Player P;
    public required FloorFacts Facts;
    public required int Ascension;
    public required string Seed;
    /// <summary>Approximations and dropped content of this import (logged with each decision).</summary>
    public readonly List<string> Notes = new();
    /// <summary>Deck cards of the import in game deck order (index = the game's deck index).</summary>
    public IReadOnlyList<CardModel> Deck => P.Deck.Cards;
}

/// <summary>
/// Game state -> simulator RunState for the macro encoder and RunAgent (rebuilt at every out-of-combat decision and at
/// every fight start). Source: the protocol state's run.save (the game's own SerializableRun JSON); the protocol's own
/// run fields are the fallback when the save is missing.
/// <para>What is imported: seed, acts, ascension, act index; the current act's map exactly as saved (points, types,
/// children, start / boss / second boss; a data-only map through the simulator's own ActEntryMapState); the act's boss
/// encounter(s) from the saved room set; visited coordinates; total floor (sum of the map point history); deck with
/// upgrades, enchantments, saved properties and the floor each card was added; relics with saved properties; the potion
/// belt by slot (belt size = max_potion_slot_count); HP, max HP, gold, max energy, shop removals used. RunAgent's own
/// bookkeeping (FloorFacts) is rebuilt from the map point history: relic floors (relic floor_added_to_deck), the
/// floor's first room model id, the act's elite / rest / shop counts, and the completed fights (encounter, room type,
/// turns, potions used, HP at start and end).</para>
/// <para>Approximated: the fights' start / end HP come from the floor's player stats (end HP = current_hp - hp_healed,
/// start HP = end HP + damage_taken; heals during a fight shift both); RNG streams, odds, relic grab bags and the room
/// sequences of the run are the simulator's for the same seed, not the save's (no decision reads them: the macro
/// encoder does not, and fights resample every random stream). A card or relic whose saved properties the simulator
/// cannot read is imported without them; an id the simulator lacks is dropped (both noted).</para>
/// </summary>
public static class RunImport
{
    private static readonly Type ActEntryMapStateType = Priv.SimType("Sts2Sim.Core.Runs.RunState+ActEntryMapState");
    private static readonly Type SavedPropsType = Priv.SimType("Sts2Sim.Core.Runs.Transplant.TransplantSavedProperties");
    private static readonly Dictionary<string, JsonElement> NoProps = new(StringComparer.Ordinal);
    private static (string Key, RunState R, Player P)? _cache;

    public static ActDefinition ActOf(string id, int index) => (id.StartsWith("ACT.", StringComparison.Ordinal) ? id[4..] : id).ToUpperInvariant() switch
    {
        "OVERGROWTH" => new Overgrowth(),
        "UNDERDOCKS" => new Underdocks(),
        "HIVE" => new Hive(),
        "GLORY" => new Glory(),
        _ => throw new InvalidDataException($"unknown act id {id} (index {index})"),
    };

    public static string ActId(ActDefinition a) => "ACT." + a.GetType().Name.ToUpperInvariant();

    /// <param name="inCombat">The current floor's fight is in progress: it is not a completed fight of the facts.</param>
    public static Imported Build(JsonObject state, MacroVocab v, bool inCombat)
    {
        JsonObject run = state.Obj("run") ?? throw new InvalidDataException("state has no run");
        JsonObject? save = run.Obj("save");
        JsonObject? sp = save?.Arr("players").OfType<JsonObject>().FirstOrDefault();
        var notes = new List<string>();
        if (save is null) notes.Add("no run.save: imported from the protocol's run fields only");

        string seed = save?.Obj("rng")?.Str("seed") ?? run.Str("seed") ?? "LIVE";
        int asc = save?.IntOrNull("ascension") ?? run.Int("ascension");
        int actIndex = save?.IntOrNull("current_act_index") ?? run.Int("act_index");
        var actIds = save is not null ? save.Objs("acts").Select(a => a.Str("id") ?? "").ToList()
            : run.Arr("act_ids").Select(x => x?.GetValue<string>() ?? "").ToList();
        if (actIds.Count == 0) throw new InvalidDataException("no act ids");
        var acts = actIds.Select((id, i) => ActOf(id, i)).ToList();
        string charId = J.ModelId(sp?.Str("character_id") ?? run.Str("character") ?? "IRONCLAD", "CHARACTER");
        var character = ModelDb.GetById<CharacterModel>(ModelId.Deserialize(charId));

        // One simulator RunState per game run, reused by every import (cleared and refilled): constructing a RunState
        // generates the act-1 map, and the simulator's map generation (MapPathPruning) crashed the process with an
        // access violation / IndexOutOfRange / NullReference after some thousands of constructions (self-check runs,
        // with and without tiered PGO and OSR). A reused run generates one map per controller session.
        string key = $"{seed}|{asc}|{string.Join(",", actIds)}|{charId}";
        RunState r; Player p;
        if (_cache is { } cached && cached.Key == key)
        {
            (r, p) = (cached.R, cached.P);
            while (r.CurrentRoom is not null) r.PopCurrentRoom();
            if (p.PlayerCombatState is not null) Priv.Call(p, "ClearCombatForVictory");   // an abandoned shadow fight
            Priv.Call(p, "DetachCombatState");
            Priv.Get<List<MapCoord>>(r, "_visitedMapCoords").Clear();
        }
        else
        {
            r = new RunState(seed, acts, asc);
            p = Player.CreateForNewRun(character, r);
            r.AddPlayer(p);
            Priv.Call(r, "EnsureRoomsGenerated");
            _cache = (key, r, p);
        }
        // the act's rooms (boss, encounter sequences) without generating the act's map (the saved map replaces it)
        if (r.CurrentActIndex != actIndex)
        {
            Priv.SetProp(r, "CurrentActIndex", actIndex);
            Priv.Call(r, "InitializeCurrentAct");
        }

        // ---- player (the simulator's own transplant restore, after a per-item check so one bad item does not fail all)
        var deckSpecs = sp is not null ? sp.Objs("deck").Select(CardSpec.Read).ToList() : run.Objs("deck").Select(CardSpec.Read).ToList();
        var deck = new List<CardSnapshot>();
        foreach (CardSpec c in deckSpecs)
        {
            CardSnapshot? snap = CardSnap(p, c, notes, "deck");
            if (snap is not null) deck.Add(snap);
        }
        var relicNodes = sp is not null ? sp.Objs("relics").ToList() : run.Objs("relics").ToList();
        var relics = new List<RelicSnapshot>();
        var relicFloors = new List<int?>();
        foreach (JsonObject rn in relicNodes)
        {
            string id = J.ModelId(rn.Str("id"), "RELIC");
            RelicSnapshot? snap = RelicSnap(p, id, rn.At("props"), notes);
            if (snap is null) continue;
            relics.Add(snap); relicFloors.Add(rn.IntOrNull("floor_added_to_deck"));
        }
        int slots = sp?.IntOrNull("max_potion_slot_count") ?? Math.Max(3, run.Arr("potions").Count);
        var belt = new string?[Math.Max(0, slots)];
        if (sp is not null)
            foreach (JsonObject pn in sp.Objs("potions"))
            {
                int k = pn.Int("slot_index", -1);
                if (k >= 0 && k < belt.Length) belt[k] = J.ModelId(pn.Str("id"), "POTION");
            }
        else
        {
            var pr = run.Arr("potions");
            for (int k = 0; k < Math.Min(pr.Count, belt.Length); k++) belt[k] = pr[k] is JsonValue jv && jv.TryGetValue(out string? s) ? J.ModelId(s, "POTION") : null;
        }
        for (int k = 0; k < belt.Length; k++)
            if (belt[k] is { } pid && ModelDb.GetByIdOrNull<PotionModel>(ModelId.Deserialize(pid)) is null) { notes.Add("unknown potion dropped: " + pid); belt[k] = null; }
        int maxHp = Math.Max(1, sp?.IntOrNull("max_hp") ?? run.Int("max_hp", 1));
        int hp = Math.Clamp(sp?.IntOrNull("current_hp") ?? run.Int("hp", 1), 1, maxHp);
        var snapshot = new PlayerSnapshot(hp, maxHp, Math.Max(0, sp?.IntOrNull("gold") ?? run.Int("gold")),
            sp?.Obj("extra_fields")?.Int("card_shop_removals_used") ?? 0, deck, relics, belt,
            p.PlayerRng.ToSerializable(), p.Odds.ToSerializable(), (RelicBagSnapshot)Priv.Call(p.RelicGrabBag, "ExportForTransplant")!)
        {
            MaxEnergy = sp?.IntOrNull("max_energy") ?? p.MaxEnergy,
            BaseOrbSlotCount = character.BaseOrbSlotCount,
        };
        Priv.Call(p, "RestoreForTransplant", snapshot);

        // ---- map, boss, visited coordinates, floor
        JsonObject? actSave = save?.Arr("acts").ElementAtOrDefault(actIndex) as JsonObject;
        if (actSave?.Obj("saved_map") is { } sm) SetMap(r, p, MapFromSave(sm));
        else if (run.Obj("map") is { } pm && pm.Arr("points").Count > 0) SetMap(r, p, MapFromProtocol(pm, notes));
        else { Priv.Call(r, "GenerateCurrentMap"); notes.Add("no saved map: the simulator's own map for this seed and act is used"); }
        if (actSave?.Obj("rooms") is { } rooms)
        {
            SetBoss(r, "_bossEncounter", rooms.Str("boss_id"), notes);
            if (rooms.Str("second_boss_id") is { Length: > 0 } b2) SetBoss(r, "_secondBossEncounter", b2, notes);
        }
        else notes.Add("no saved room set: the act's boss is the simulator's for this seed");
        var visited = save is not null ? save.Objs("visited_map_coords").ToList() : run.Obj("map").Objs("visited").ToList();
        foreach (JsonObject c in visited)
        {
            var coord = new MapCoord(c.Int("col"), c.Int("row"));
            if (r.Map.GetPoint(coord) is null) { notes.Add($"visited coordinate {coord.col},{coord.row} not on the map"); continue; }
            r.AddVisitedMapCoord(coord);
        }
        var history = save?.Arr("map_point_history").Select(a => (a as JsonArray ?? new JsonArray()).OfType<JsonObject>().ToList()).ToList()
            ?? new List<List<JsonObject>>();
        int total = history.Count > 0 ? history.Sum(h => h.Count) : run.Int("floor", r.VisitedMapCoords.Count);
        Priv.Set(r, "_completedActFloors", Math.Max(0, total - r.VisitedMapCoords.Count));
        if (r.TotalFloor != total) notes.Add($"total floor {r.TotalFloor} != history {total}");

        // ---- RunAgent's facts
        int g = r.TotalFloor;
        var facts = new FloorFacts { Ascension = asc, CountAct = actIndex, FloorModelIdFloor = g };
        for (int i = 0; i < p.Relics.Count; i++)
            facts.RelicFloor[p.Relics[i]] = Math.Max(1, i < relicFloors.Count && relicFloors[i] is { } f ? f : g);
        int floor = 0;
        for (int a = 0; a < history.Count; a++)
            for (int k = 0; k < history[a].Count; k++)
            {
                floor++;
                JsonObject e = history[a][k];
                var roomsH = e.Objs("rooms").ToList();
                bool current = a == history.Count - 1 && k == history[a].Count - 1;
                if (a == actIndex && roomsH.Count > 0)
                {
                    J.TryEnum(roomsH[0].At("room_type"), out RoomType rt0);
                    if (rt0 == RoomType.Elite) facts.ActElites++;
                    else if (rt0 == RoomType.RestSite) facts.ActRests++;
                    else if (rt0 == RoomType.Shop) facts.ActShops++;
                }
                if (current && roomsH.Count > 0)
                {
                    J.TryEnum(roomsH[0].At("room_type"), out RoomType rt0);
                    facts.FloorModelId = rt0 is RoomType.Monster or RoomType.Elite or RoomType.Boss or RoomType.Event
                        ? J.ModelId(roomsH[0].Str("model_id"), rt0 == RoomType.Event ? "EVENT" : "ENCOUNTER") : "";
                }
                JsonObject? st = e.Objs("player_stats").FirstOrDefault();
                for (int j = 0; j < roomsH.Count; j++)
                {
                    J.TryEnum(roomsH[j].At("room_type"), out RoomType rt);
                    if (rt is not (RoomType.Monster or RoomType.Elite or RoomType.Boss)) continue;
                    if (current && inCombat && j == roomsH.Count - 1) continue;   // the fight in progress
                    string enc = J.ModelId(roomsH[j].Str("model_id"), "ENCOUNTER");
                    int mx = Math.Max(1, st.Int("max_hp", maxHp));
                    int end = Math.Max(0, st.Int("current_hp", mx) - st.Int("hp_healed"));
                    int start = Math.Min(mx, end + st.Int("damage_taken"));
                    facts.Fights.Add(new FightNote(v.Tok("encounter", enc), enc,
                        rt == RoomType.Boss ? ML.FightBoss : rt == RoomType.Elite ? ML.FightElite : ML.FightHall,
                        start, end, mx, st.Arr("potion_used").Count, roomsH[j].Int("turns_taken"), floor, a));
                }
            }
        if (history.Count == 0) notes.Add("no map point history: no fight tokens, act counters from the map only");
        var imp = new Imported { Run = r, P = p, Facts = facts, Ascension = asc, Seed = seed };
        imp.Notes.AddRange(notes);
        return imp;
    }

    private static CardSnapshot? CardSnap(Player p, CardSpec c, List<string> notes, string where)
    {
        if (ModelDb.GetByIdOrNull<CardModel>(ModelId.Deserialize(c.Id)) is null) { notes.Add($"{where}: unknown card dropped: {c.Id}"); return null; }
        IReadOnlyDictionary<string, JsonElement> props = NoProps;
        if (c.Props is JsonObject po && po.Count > 0)
            props = new Dictionary<string, JsonElement>(StringComparer.Ordinal) { ["props"] = JsonSerializer.SerializeToElement(po) };
        if (c.Id == "CARD.SPOILS_MAP")
            props = new Dictionary<string, JsonElement>(props, StringComparer.Ordinal) { ["spoils_coord"] = JsonDocument.Parse("null").RootElement };
        string? ench = c.Ench is not null && ModelDb.GetByIdOrNull<EnchantmentModel>(ModelId.Deserialize(c.Ench)) is null ? null : c.Ench;
        if (ench is null && c.Ench is not null) notes.Add($"{where}: unknown enchantment dropped: {c.Ench} on {c.Id}");
        var snap = new CardSnapshot(c.Id, c.Upgrade, ench, ench is null ? null : c.EnchAmount, c.Floor, props);
        try { CreateCard(p, snap); return snap; }
        catch (Exception e) when (props.Count > 0)
        {
            notes.Add($"{where}: {c.Id} imported without saved properties ({e.Message})");
            snap = snap with { Props = c.Id == "CARD.SPOILS_MAP" ? new Dictionary<string, JsonElement> { ["spoils_coord"] = JsonDocument.Parse("null").RootElement } : NoProps };
        }
        try { CreateCard(p, snap); return snap; }
        catch (Exception e) { notes.Add($"{where}: card dropped: {c.Id} ({e.Message})"); return null; }
    }

    private static RelicSnapshot? RelicSnap(Player p, string id, JsonNode? propsNode, List<string> notes)
    {
        RelicModel? canonical = ModelDb.GetByIdOrNull<RelicModel>(ModelId.Deserialize(id));
        if (canonical is null) { notes.Add("unknown relic dropped: " + id); return null; }
        IReadOnlyDictionary<string, JsonElement> props = propsNode is JsonObject po && po.Count > 0
            ? new Dictionary<string, JsonElement>(StringComparer.Ordinal) { ["props"] = JsonSerializer.SerializeToElement(po) } : NoProps;
        bool Try(IReadOnlyDictionary<string, JsonElement> pr, out string err)
        {
            err = "";
            try
            {
                var relic = (RelicModel)canonical.MutableClone();
                relic.AssignOwner(p);
                Priv.CallStatic(SavedPropsType, "Apply", relic, pr, "relic", true);
                return true;
            }
            catch (Exception e) { err = e.Message; return false; }
        }
        if (Try(props, out string e1)) return new RelicSnapshot(id, props);
        if (props.Count > 0 && Try(NoProps, out _)) { notes.Add($"relic {id} imported without saved properties ({e1})"); return new RelicSnapshot(id, NoProps); }
        notes.Add($"relic dropped: {id} ({e1})");
        return null;
    }

    /// <summary>A card of the player through the simulator's transplant path (enchantment, upgrades, saved properties).</summary>
    public static CardModel CreateCard(Player p, CardSnapshot snap) => (CardModel)Priv.Call(p, "CreateCardForTransplant", snap, "live")!;

    /// <summary>A card from a protocol CARD (no saved properties), falling back to the bare card if the enchantment fails.</summary>
    public static CardModel CreateCard(Player p, CardSpec c)
    {
        try { return CreateCard(p, new CardSnapshot(c.Id, c.Upgrade, c.Ench, c.Ench is null ? null : c.EnchAmount, c.Floor, NoProps)); }
        catch when (c.Ench is not null) { return CreateCard(p, new CardSnapshot(c.Id, c.Upgrade, null, null, c.Floor, NoProps)); }
    }

    private static void SetBoss(RunState r, string field, string? id, List<string> notes)
    {
        if (string.IsNullOrEmpty(id)) return;
        string entry = id.StartsWith("ENCOUNTER.", StringComparison.Ordinal) ? id[10..] : id;
        EncounterDefinition? e = r.Act.BossEncounterCandidates.FirstOrDefault(x => x.IdEntry == entry);
        if (e is null) { notes.Add($"boss {id} not in the simulator's act pool"); return; }
        Priv.Set(r, field, e);
    }

    private static void SetMap(RunState r, Player p, ActEntryMap data)
    {
        object map = Priv.New(ActEntryMapStateType, data, p);
        Priv.SetProp(r, "Map", map);
    }

    private static MapCoordDto Coord(JsonNode? c) => new(c.Int("col"), c.Int("row"));

    private static MapPointType PointType(JsonNode? v) => J.TryEnum(v, out MapPointType t) ? t : MapPointType.Unknown;

    /// <summary>SerializableActMap (points, boss, second_boss, start, start_coords, width, height) -> the simulator's
    /// data-only map record (parents derived from children).</summary>
    public static ActEntryMap MapFromSave(JsonObject sm)
    {
        var nodes = new Dictionary<(int, int), (MapPointType Type, bool CanModify, bool InGrid, List<MapCoordDto> Children)>();
        int w = sm.Int("width"), h = sm.Int("height");
        void Add(JsonObject? pt, bool grid)
        {
            if (pt is null) return;
            var c = Coord(pt.At("coord"));
            var key = (c.Col, c.Row);
            if (nodes.ContainsKey(key)) return;
            bool inGrid = grid && c.Col >= 0 && c.Col < w && c.Row >= 0 && c.Row < h;
            nodes[key] = (PointType(pt.At("type")), pt.Bool("can_modify", true), inGrid, pt.Arr("children").Select(Coord).ToList());
        }
        foreach (JsonObject pt in sm.Objs("points")) Add(pt, true);
        Add(sm.Obj("start"), false); Add(sm.Obj("boss"), false); Add(sm.Obj("second_boss"), false);
        var start = Coord(sm.Obj("start")?.At("coord"));
        var boss = Coord(sm.Obj("boss")?.At("coord"));
        MapCoordDto? boss2 = sm.Obj("second_boss") is { } b2 ? Coord(b2.At("coord")) : null;
        var starts = sm.Arr("start_coords").Count > 0 ? sm.Arr("start_coords").Select(Coord).ToList()
            : nodes.Where(n => n.Value.InGrid && n.Key.Item2 == nodes.Where(m => m.Value.InGrid).Min(m => m.Key.Item2)).Select(n => new MapCoordDto(n.Key.Item1, n.Key.Item2)).ToList();
        return Assemble(nodes, Math.Max(1, w), Math.Max(1, h), start, boss, boss2, starts);
    }

    /// <summary>The protocol's run.map (points with type and children): start = the Ancient point (or the lowest row),
    /// boss = the boss point(s) without boss children.</summary>
    public static ActEntryMap MapFromProtocol(JsonObject pm, List<string> notes)
    {
        notes.Add("map from the protocol's run.map (no saved map)");
        var nodes = new Dictionary<(int, int), (MapPointType Type, bool CanModify, bool InGrid, List<MapCoordDto> Children)>();
        foreach (JsonObject pt in pm.Objs("points"))
            nodes[(pt.Int("col"), pt.Int("row"))] = (PointType(pt.At("type")), true, true, pt.Arr("children").Select(Coord).ToList());
        var startKey = nodes.Where(n => n.Value.Type == MapPointType.Ancient).Select(n => n.Key).DefaultIfEmpty(nodes.Keys.OrderBy(k => k.Item2).First()).First();
        var bosses = nodes.Where(n => n.Value.Type == MapPointType.Boss).Select(n => n.Key).OrderBy(k => k.Item2).ToList();
        if (bosses.Count == 0) bosses.Add(nodes.Keys.OrderByDescending(k => k.Item2).First());
        int w = nodes.Keys.Max(k => k.Item1) + 1, h = nodes.Keys.Max(k => k.Item2) + 1;
        foreach (var k in new[] { startKey }.Concat(bosses)) nodes[k] = nodes[k] with { InGrid = false };
        int firstRow = nodes.Where(n => n.Value.InGrid).Select(n => n.Key.Item2).DefaultIfEmpty(0).Min();
        var starts = nodes.Where(n => n.Value.InGrid && n.Key.Item2 == firstRow).Select(n => new MapCoordDto(n.Key.Item1, n.Key.Item2)).ToList();
        return Assemble(nodes, w, h, new(startKey.Item1, startKey.Item2), new(bosses[0].Item1, bosses[0].Item2),
            bosses.Count > 1 ? new(bosses[1].Item1, bosses[1].Item2) : null, starts);
    }

    private static ActEntryMap Assemble(Dictionary<(int, int), (MapPointType Type, bool CanModify, bool InGrid, List<MapCoordDto> Children)> nodes,
        int w, int h, MapCoordDto start, MapCoordDto boss, MapCoordDto? boss2, List<MapCoordDto> starts)
    {
        var parents = nodes.Keys.ToDictionary(k => k, _ => new List<MapCoordDto>());
        foreach (var (k, n) in nodes)
            foreach (MapCoordDto c in n.Children)
                if (parents.TryGetValue((c.Col, c.Row), out var ps) && !ps.Contains(new MapCoordDto(k.Item1, k.Item2))) ps.Add(new MapCoordDto(k.Item1, k.Item2));
        var points = nodes.Select(n => new ActEntryMapPoint(n.Key.Item1, n.Key.Item2, n.Value.Type.ToString(), n.Value.CanModify, n.Value.InGrid,
            n.Value.Children.Where(c => nodes.ContainsKey((c.Col, c.Row))).Distinct().ToList(), parents[n.Key], Array.Empty<ActEntryModelRef>())).ToList();
        return new ActEntryMap("StandardActMap", w, h, start, boss, boss2, starts, points);
    }
}
