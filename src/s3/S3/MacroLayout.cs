using System.Text.Json;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;

namespace Sts2Sim.S3;

/// <summary>
/// Macro-network row layout, shared with src/s3/py/macro_s3.py (see its header) and the run records.
/// A row is the state after a floor in the conventions of the S1 dataset (src/value_v1/build_floor_dataset.py:
/// floor g counted from the Ancient of act 1 = 1, room type = the map point's type, encounter = the model id of the
/// floor's first room, card age = g - floor added, damage/heal/turn scalars zero), optionally followed by one decision
/// (type + up to <see cref="K"/> options, or a choice among deck slots).
/// </summary>
public static class ML
{
    public const int MaxDeck = 64, MaxRel = 40, MaxPot = 5, K = 16, MF = 12, NRamps = 26;
    // Layout v2 (2026-11-16, append only: every v1 offset is unchanged; v1 rows are IW1 ints and FW1 floats wide).
    // ints: the act boss and the act's second boss (0 = none), in the id space of the context encounter (OCtx + 5).
    // floats: MF2 more features per option (map options only: RunAgent.MapFeatures2), then NS2 more state scalars.
    public const int MF2 = 8, NS2 = 8, IW1 = 460, FW1 = 16 + K * MF;
    // Layout v3 (2026-11-16, append only again; v2 rows are IW2 ints and FW2 floats wide):
    //   ints  from OFightEnc: the last NFight completed live fights of the run, most recent first: encounter token (the
    //         id space of OCtx + 5) [NFight], then room kind (1 hallway, 2 elite, 3 boss; 0 = no fight) [NFight];
    //   floats from FFights: NFight x FightW per fight (XfHpLost ...), then from FMap NMap map nodes of the current act,
    //         each ONE integer (< 2^24, exact in a float32) packed as in MapNode (0 = no node).
    public const int IW2 = 462, FW2 = 16 + K * MF + K * MF2 + NS2, NFight = 8, FightW = 6, NMap = 80;
    public const int ODIds = 0, ODUpg = 64, ODEnch = 128, ODEamt = 192, ODAge = 256, ORIds = 320, ORAge = 360, OPIds = 400,
        OCtx = 405, ODtype = 411, OOptKind = 412, OOptId = 428, OOptUpg = 444, OBoss = 460, OBoss2 = 461,
        OFightEnc = 462, OFightRoom = OFightEnc + NFight, IW = OFightRoom + NFight;
    public const int FScal = 0, FOptFeat = 16, FOptFeat2 = 16 + K * MF, FScal2 = FOptFeat2 + K * MF2, FFights = FScal2 + NS2,
        FMap = FFights + NFight * FightW, FW = FMap + NMap;
    // extra state scalars from FScal2: elites fought / 4, rest sites visited / 4, shops visited / 3 (all in this act),
    // free potion slots / 3, potion slots / 5, current map row / 16; 6 (layout v3) the price of the next card removal / 200;
    // 7 reserved (0). None depends on HP.
    public const int XElites = 0, XRests = 1, XShops = 2, XPotFree = 3, XPotSlots = 4, XMapRow = 5, XRemovePrice = 6;
    // per fight (FFights + i * FightW): HP lost in the fight / max HP at its start (negative = healed; -1..1), HP share at
    // its start, potions used / 3, turns / 20, floors ago / 20 (0 = this floor), 1 if in the current act
    public const int XfHpLost = 0, XfHpStart = 1, XfPotions = 2, XfTurns = 3, XfFloorsAgo = 4, XfSameAct = 5;
    public const int FightHall = 1, FightElite = 2, FightBoss = 3;
    // map node packing (MapNode.Pack): bits 0-3 map point type + 1 (0 = no node), 4-8 row, 9-11 column, 12 visited
    // (this act), 13 current, 14 reachable from the current point (a strict descendant; every node when there is none),
    // 15-23 option-reach mask: bit 15 + j set if the node is reachable (inclusive) from map option slot j (j < MapSlots).
    // On a map decision row the slots are the row's options; on every other row they are the points the run could
    // travel to next (MapTravel.GetTravelablePointsFrom), sorted as RunAgent sorts map options (column, then row).
    public const int MapRowShift = 4, MapColShift = 9, MapVisitedBit = 12, MapCurrentBit = 13, MapReachBit = 14, MapMaskShift = 15, MapSlots = 9;
    public const int OutA = 0, OutC = 1, OutV = 27, OutStd = 28, OutVPol = 29, OutOpt = 30, OutSlot = 46;
    // milestone heads (macro_s3.py header): 4 x [a, 26 c] from OutMs, then alpha, boss-1 share, 1 if the heads are in use
    public const int NMs = 4, OutMs = 110, OutAlpha = 218, OutW1 = 219, OutHasMs = 220, OW = 221;
    public const int DoubleBossAsc = 10;
    // scalars (train_value_v2.S)
    public const int SFloor = 0, SAct = 1, SHp = 2, SMaxHp = 3, SHpFrac = 4, SGold = 5, SDeck = 8, SUpg = 9, SEnch = 10, SRelics = 11,
        SPotions = 12, SIsCombat = 14, SFloorInAct = 15;
    // decision types (macro_s3.DT)
    public const int DCard = 0, DRelic = 1, DAncient = 2, DRest = 3, DSmith = 4, DRemove = 5, DShop = 6, DMap = 7, DEvent = 8, DNone = -1;
    // option kinds
    public const int KPad = 0, KCard = 1, KRelic = 2, KPotion = 3, KSpecial = 4, KMap = 5, KEvent = 6, KZero = 7;
    public const int EventBuckets = 4096, CharDefect = 2, EraV0111 = 4;
    // ic113 (2026-11-18): version era of the simulator content. S1 eras 0-4 come from the human data (4 = v0.111),
    // 5 is the era-dropout "unknown" token, 6 = v0.113 (the macro network's era embedding is grown to 7 rows, row 6
    // initialised as a copy of row 4: src/ic113/convert_v113.py, macro_s3.S3MacroNet). Set by `S3 play era=`.
    public const int EraV0113 = 6;
    /// <summary>Game build of the simulator content (iRyougi tag v0.113.1.1), written to run records and battle starts.</summary>
    public const string BuildId = "v0.113.1";
    public static int Era = EraV0113;
    /// <summary>S1 character index (build_floor_dataset.CHARS: IRONCLAD 0, SILENT 1, DEFECT 2, NECROBINDER 3,
    /// REGENT 4) of a character model id ("CHARACTER.IRONCLAD"); characters S1 never saw (the Fool) are refused.</summary>
    public static int CharIndex(string characterId) => characterId switch
    {
        "CHARACTER.IRONCLAD" => 0, "CHARACTER.SILENT" => 1, "CHARACTER.DEFECT" => 2, "CHARACTER.NECROBINDER" => 3,
        "CHARACTER.REGENT" => 4,
        _ => throw new NotSupportedException($"character {characterId} has no S1 character token"),
    };
    public static readonly string[] DtNames = { "CARD_REWARD", "RELIC_PICK", "ANCIENT", "REST", "SMITH_TARGET", "REMOVE_TARGET", "SHOP", "MAP", "EVENT" };

    /// <summary>ROOM_TYPES index of build_floor_dataset.py for a map point type.</summary>
    public static int RoomIndex(MapPointType t) => t switch
    {
        MapPointType.Ancient => 2, MapPointType.Monster => 3, MapPointType.Elite => 4, MapPointType.Boss => 5,
        MapPointType.Unknown => 6, MapPointType.Shop => 7, MapPointType.RestSite => 8, MapPointType.Treasure => 9,
        _ => 1,
    };

    public static bool IsCombatPoint(MapPointType t) => t is MapPointType.Monster or MapPointType.Elite or MapPointType.Boss;

    /// <summary>Value logit at HP <paramref name="hp"/> from a served row: a + sum_k c_k phi_k(hp, max HP)
    /// (train_value_v2.hp_features: 16 absolute ramps of width 10 from 0, 10 fraction ramps of width 0.1).</summary>
    public static double LogitAt(ReadOnlySpan<float> outRow, int hp, int maxHp, int at = OutA)
    {
        double z = outRow[at];
        for (int k = 0; k < 16; k++) z += outRow[at + 1 + k] * Math.Clamp((hp - 10.0 * k) / 10.0, 0, 1);
        double frac = Math.Clamp(hp / (double)Math.Max(1, maxHp), 0, 1);
        for (int k = 0; k < 10; k++) z += outRow[at + 17 + k] * Math.Clamp((frac - 0.1 * k) / 0.1, 0, 1);
        return z;
    }

    public static bool HasMilestones(ReadOnlySpan<float> outRow) => outRow[OutHasMs] > 0.5f;

    /// <summary>Run value at HP <paramref name="hp"/> from the milestone heads of a served row (macro_s3.blend): reach
    /// probabilities from the conditional heads for the stages after <paramref name="stage"/>, then
    /// V = alpha R4 + (1 - alpha) (R1 + R2 + w R3 + (1 - w) R4) / 3. Below the double-boss level head 3 is 1.</summary>
    public static double MilestoneValue(ReadOnlySpan<float> outRow, int hp, int maxHp, int stage, bool doubleBoss)
    {
        double alpha = outRow[OutAlpha], w = outRow[OutW1], run = 1;
        Span<double> r = stackalloc double[NMs];
        for (int m = 0; m < NMs; m++)
        {
            double q = m == 3 && !doubleBoss ? 1.0 : Sigmoid(LogitAt(outRow, hp, maxHp, OutMs + m * (1 + NRamps)));
            run = stage > m ? 1.0 : run * q;
            r[m] = run;
        }
        return alpha * r[3] + (1 - alpha) * (r[0] + r[1] + w * r[2] + (1 - w) * r[3]) / 3;
    }

    /// <summary>Stage of a row (macro_s3.stage_of): act index, plus 1 on a boss floor.</summary>
    public static int Stage(int actIndex, bool bossFloor) => actIndex + (bossFloor ? 1 : 0);

    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public static int EventBucket(string key) => (int)(Sts2Sim.S0.ActionSpace.Fnv(key) % EventBuckets);
}

public sealed class MacroVocab
{
    private readonly Dictionary<string, Dictionary<string, int>> _m;
    public readonly Dictionary<string, long> Unknown = new(), Seen = new();
    /// <summary>ic113: the first unknown ids (kind:id), for the actor's end-of-run report.</summary>
    public readonly SortedSet<string> UnknownIds = new(StringComparer.Ordinal);
    public IReadOnlyList<string> Special { get; }

    public MacroVocab(string path)
    {
        var lists = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(File.ReadAllText(path))!;
        _m = lists.ToDictionary(kv => kv.Key, kv => kv.Value.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i, StringComparer.Ordinal));
        Special = lists["special"];
        foreach (string k in lists.Keys) { Unknown[k] = 0; Seen[k] = 0; }
    }

    /// <summary>0 = none, 1 = unknown (as build_floor_dataset.tok).</summary>
    public int Tok(string kind, string? id)
    {
        if (string.IsNullOrEmpty(id)) return 0;
        Seen[kind]++;
        if (_m[kind].TryGetValue(id, out int i)) return i;
        Unknown[kind]++;
        if (UnknownIds.Count < 64) lock (UnknownIds) UnknownIds.Add(kind + ":" + id);
        return 1;
    }

    public bool Has(string kind, string id) => _m[kind].ContainsKey(id);

    /// <summary>The id behind a token (test output only; "" for 0 or a token outside the vocabulary).</summary>
    public string Name(string kind, int tok)
    {
        if (tok <= 0) return "";
        foreach (var kv in _m[kind]) if (kv.Value == tok) return kv.Key;
        return "";
    }
    public int SpecialId(string name) => _m["special"].GetValueOrDefault(name, -1);
}

/// <summary>One macro row (state + optional decision) as sent to the network and stored in the run records.</summary>
public sealed class MacroRow
{
    public readonly int[] I = new int[ML.IW];
    public readonly float[] F = new float[ML.FW];
    public int NOptions;
    /// <summary>Encode test only (<see cref="MacroEncoder.Debug"/>): the map nodes and fights as the encoder meant them.</summary>
    public MapNode[]? DbgMap;
    public FightNote[]? DbgFights;

    public MacroRow Clone()
    {
        var r = new MacroRow { NOptions = NOptions, DbgMap = DbgMap, DbgFights = DbgFights };
        I.CopyTo(r.I, 0); F.CopyTo(r.F, 0);
        return r;
    }

    public void ClearDecision()
    {
        I[ML.ODtype] = 0;
        Array.Clear(I, ML.OOptKind, 3 * ML.K);
        Array.Clear(F, ML.FOptFeat, ML.K * ML.MF);
        Array.Clear(F, ML.FOptFeat2, ML.K * ML.MF2);
        NOptions = 0;
    }

    /// <param name="feat2">Layout v2: the option's extra features (map options; zeros otherwise).</param>
    public int AddOption(int kind, int id, int upg = 0, ReadOnlySpan<float> feat = default, ReadOnlySpan<float> feat2 = default)
    {
        int j = NOptions;
        if (j >= ML.K) return -1;
        I[ML.OOptKind + j] = kind; I[ML.OOptId + j] = id; I[ML.OOptUpg + j] = Math.Clamp(upg, 0, 3);
        for (int f = 0; f < Math.Min(feat.Length, ML.MF); f++) F[ML.FOptFeat + j * ML.MF + f] = feat[f];
        for (int f = 0; f < Math.Min(feat2.Length, ML.MF2); f++) F[ML.FOptFeat2 + j * ML.MF2 + f] = feat2[f];
        NOptions++;
        return j;
    }

    public int Hp => (int)Math.Round(F[ML.SHp] * 100);
    public int MaxHp => (int)Math.Round(F[ML.SMaxHp] * 100);

    public void SetHp(int hp, int maxHp)
    {
        F[ML.SHp] = hp / 100f; F[ML.SMaxHp] = maxHp / 100f; F[ML.SHpFrac] = hp / (float)Math.Max(1, maxHp);
    }

    /// <summary>Replaces the potion belt (macro potion ids, at most <see cref="ML.MaxPot"/>). The free-slot scalar of
    /// layout v2 follows from the row's potion slots (set by the encoder before this call).</summary>
    public void SetPotions(IReadOnlyList<int> toks)
    {
        for (int j = 0; j < ML.MaxPot; j++) I[ML.OPIds + j] = j < toks.Count ? toks[j] : 0;
        F[ML.SPotions] = Math.Min(toks.Count, ML.MaxPot) / 5f;
        int slots = (int)Math.Round(F[ML.FScal2 + ML.XPotSlots] * 5);
        F[ML.FScal2 + ML.XPotFree] = Math.Max(0, slots - toks.Count) / 3f;
    }
}

/// <summary>Layout v3: one completed live fight (RunAgent.EndFight). Room: ML.FightHall / FightElite / FightBoss.</summary>
public readonly record struct FightNote(int EncTok, string Encounter, int Room, int StartHp, int EndHp, int MaxHpStart,
    int Potions, int Turns, int Floor, int Act);

/// <summary>Layout v3: one map node as the encoder means it (packed into one integer by <see cref="Pack"/>).</summary>
public readonly record struct MapNode(MapPointType Type, int Row, int Col, bool Visited, bool Current, bool Reachable, int Mask)
{
    public int Pack() => (((int)Type + 1) & 15) | (Math.Clamp(Row, 0, 31) << ML.MapRowShift) | (Math.Clamp(Col, 0, 7) << ML.MapColShift)
        | (Visited ? 1 << ML.MapVisitedBit : 0) | (Current ? 1 << ML.MapCurrentBit : 0) | (Reachable ? 1 << ML.MapReachBit : 0)
        | ((Mask & ((1 << ML.MapSlots) - 1)) << ML.MapMaskShift);
}

/// <summary>What the encoder needs beyond <see cref="RunState"/>: facts the run agent tracks.</summary>
public sealed class FloorFacts
{
    public int Ascension;
    /// <summary>Floor on which each relic was obtained (starter relic: 1).</summary>
    public readonly Dictionary<RelicModel, int> RelicFloor = new(ReferenceEqualityComparer.Instance);
    /// <summary>Model id of the first room of the current floor ("ENCOUNTER.X" / "EVENT.X"; "" for shop, rest, treasure).</summary>
    public string FloorModelId = "";
    public int FloorModelIdFloor = -1;
    /// <summary>Layout v2: rooms entered in the act <see cref="CountAct"/> so far, the current floor included, by the
    /// floor's first room (an Unknown map point that turned out to be a shop counts as a shop).</summary>
    public int CountAct = -1, ActElites, ActRests, ActShops;
    /// <summary>Layout v3: the run's completed live fights in play order (RunAgent.EndFight).</summary>
    public readonly List<FightNote> Fights = new();

    public void NoteRelics(Player p, int floor)
    {
        foreach (RelicModel r in p.Relics) RelicFloor.TryAdd(r, Math.Max(1, floor));
    }

    public void NoteRoom(RunState run)
    {
        if (CountAct != run.CurrentActIndex) { CountAct = run.CurrentActIndex; ActElites = ActRests = ActShops = 0; }
        int g = run.TotalFloor;
        if (FloorModelIdFloor == g) return;
        AbstractRoom? room = run.BaseRoom;
        if (room is null) return;
        FloorModelIdFloor = g;
        FloorModelId = RoomModelId(room);
        switch (room)
        {
            case CombatRoom { RoomType: RoomType.Elite }: ActElites++; break;
            case RestSiteRoom: ActRests++; break;
            case MerchantRoom: ActShops++; break;
        }
    }

    public static string RoomModelId(AbstractRoom room) => room switch
    {
        CombatRoom c => c.Encounter is { } e ? "ENCOUNTER." + e.IdEntry : "",
        EventRoom e => e.Event.Id.ToString(),
        _ => "",
    };
}

public static class MacroEncoder
{
    public static string CardId(CardModel c) => c.Id.ToString();

    /// <summary>The state after the current floor (or, during a floor, the state now, which is what the S1 policy saw
    /// before a choice: the after-floor state with the choice undone).</summary>
    public static MacroRow Encode(RunState run, Player p, FloorFacts facts, MacroVocab v)
    {
        var r = new MacroRow();
        int g = run.TotalFloor;
        MapPointType pt = run.CurrentMapPoint?.PointType ?? MapPointType.Ancient;
        string modelId = facts.FloorModelIdFloor == g ? facts.FloorModelId : "";
        FillEntities(r, p, facts, v, g);
        r.I[ML.OCtx + 0] = ML.CharIndex(p.Character.Id.ToString()); r.I[ML.OCtx + 1] = facts.Ascension; r.I[ML.OCtx + 2] = ML.Era;
        r.I[ML.OCtx + 3] = run.CurrentActIndex; r.I[ML.OCtx + 4] = ML.RoomIndex(pt); r.I[ML.OCtx + 5] = v.Tok("encounter", modelId);
        var c = p.Creature;
        r.F[ML.SFloor] = g / 50f; r.F[ML.SAct] = run.CurrentActIndex / 3f;
        r.SetHp(Math.Max(0, c.CurrentHp), c.MaxHp);
        r.F[ML.SGold] = p.Gold / 500f;
        r.F[ML.SIsCombat] = ML.IsCombatPoint(pt) ? 1f : 0f;
        r.F[ML.SFloorInAct] = run.VisitedMapCoords.Count / 20f;
        FillV2(r, run, facts, v);
        r.F[ML.FScal2 + ML.XRemovePrice] = Sts2Sim.Core.Entities.Merchant.MerchantCardRemovalEntry.PriceFor(run.Ascension, p.CardRemovalsUsed) / 200f;
        FillFights(r, run, facts);
        FillMap(r, run, null);
        return r;
    }

    /// <summary>Encode test: rows keep the map nodes and fights they were built from (<see cref="MacroRow.DbgMap"/>).</summary>
    public static bool Debug;
    /// <summary>Rows whose act map had more than <see cref="ML.NMap"/> nodes (nodes behind the current row dropped first).</summary>
    public static long MapTruncated, MapRows, MapNodesMax;

    /// <summary>Layout v3: the last <see cref="ML.NFight"/> completed live fights, most recent first (HP-free with respect
    /// to the row's own HP: everything here is about fights already over).</summary>
    public static void FillFights(MacroRow r, RunState run, FloorFacts facts)
    {
        Array.Clear(r.I, ML.OFightEnc, 2 * ML.NFight);
        Array.Clear(r.F, ML.FFights, ML.NFight * ML.FightW);
        var fs = facts.Fights;
        int n = Math.Min(ML.NFight, fs.Count), g = run.TotalFloor;
        for (int i = 0; i < n; i++)
        {
            FightNote f = fs[fs.Count - 1 - i];
            r.I[ML.OFightEnc + i] = f.EncTok; r.I[ML.OFightRoom + i] = f.Room;
            int o = ML.FFights + i * ML.FightW;
            float mx = Math.Max(1, f.MaxHpStart);
            r.F[o + ML.XfHpLost] = Math.Clamp((f.StartHp - f.EndHp) / mx, -1f, 1f);
            r.F[o + ML.XfHpStart] = Math.Clamp(f.StartHp / mx, 0f, 1f);
            r.F[o + ML.XfPotions] = f.Potions / 3f;
            r.F[o + ML.XfTurns] = Math.Min(f.Turns, 100) / 20f;
            r.F[o + ML.XfFloorsAgo] = Math.Clamp(g - f.Floor, 0, 60) / 20f;
            r.F[o + ML.XfSameAct] = f.Act == run.CurrentActIndex ? 1f : 0f;
        }
        if (Debug) r.DbgFights = Enumerable.Range(0, n).Select(i => fs[fs.Count - 1 - i]).ToArray();
    }

    /// <summary>Layout v3: the nodes of the current act's map, sorted by (row, column), packed (<see cref="MapNode"/>).
    /// <paramref name="options"/>: the map options of a decision row in their slot order; null = the points the run could
    /// travel to next, in the order RunAgent gives map options.</summary>
    public static void FillMap(MacroRow r, RunState run, IReadOnlyList<MapPoint>? options)
    {
        Array.Clear(r.F, ML.FMap, ML.NMap);
        var nodes = MapNodes(run, options);
        for (int i = 0; i < nodes.Count; i++) r.F[ML.FMap + i] = nodes[i].Pack();
        if (Debug) r.DbgMap = nodes.ToArray();
    }

    public static List<MapNode> MapNodes(RunState run, IReadOnlyList<MapPoint>? options)
    {
        var res = new List<MapNode>();
        if (run.Map is not { } map) return res;
        var pts = new HashSet<MapPoint>(ReferenceEqualityComparer.Instance);
        foreach (MapPoint q in map.GetAllMapPoints()) if (q.Children.Count > 0 || q.parents.Count > 0) pts.Add(q);
        pts.Add(map.StartingMapPoint); pts.Add(map.BossMapPoint);
        if (map.SecondBossMapPoint is { } sb) pts.Add(sb);
        MapPoint? cur = run.CurrentMapCoord is { } cc ? map.GetPoint(cc) : null;
        if (cur is not null && !pts.Contains(cur)) cur = null;
        IReadOnlyList<MapPoint> opts = options ?? (cur is null ? Array.Empty<MapPoint>()
            : Sts2Sim.Core.Map.MapTravel.GetTravelablePointsFrom(run, cur).OrderBy(o => o.coord.col).ThenBy(o => o.coord.row).Take(ML.K).ToList());
        var visited = new HashSet<MapPoint>(ReferenceEqualityComparer.Instance);
        foreach (MapCoord c in run.VisitedMapCoords) if (map.GetPoint(c) is { } vp) visited.Add(vp);
        static void Below(MapPoint p, HashSet<MapPoint> acc) { foreach (MapPoint ch in p.Children) if (acc.Add(ch)) Below(ch, acc); }
        HashSet<MapPoint>? ahead = null;
        if (cur is not null) { ahead = new(ReferenceEqualityComparer.Instance); Below(cur, ahead); }
        var mask = new Dictionary<MapPoint, int>(ReferenceEqualityComparer.Instance);
        for (int j = 0; j < Math.Min(opts.Count, ML.MapSlots); j++)
        {
            var sub = new HashSet<MapPoint>(ReferenceEqualityComparer.Instance) { opts[j] };
            Below(opts[j], sub);
            foreach (MapPoint q in sub) mask[q] = mask.GetValueOrDefault(q) | (1 << j);
        }
        int curRow = cur?.coord.row ?? -1;
        // more nodes than slots (an act map has up to ~75 nodes; NMap = 80): the nodes behind the current row that were
        // not visited go first (they cannot matter), then those the run can no longer reach, then the highest rows
        int Rank(MapPoint q) => q.coord.row < curRow && !visited.Contains(q) ? 2
            : ahead is not null && !ahead.Contains(q) && !visited.Contains(q) ? 1 : 0;
        var order = pts.OrderBy(Rank).ThenBy(q => q.coord.row).ThenBy(q => q.coord.col).ToList();
        MapRows++; MapNodesMax = Math.Max(MapNodesMax, order.Count);
        if (order.Count > ML.NMap) { MapTruncated++; order = order.Take(ML.NMap).ToList(); }
        foreach (MapPoint q in order.OrderBy(q => q.coord.row).ThenBy(q => q.coord.col))
            res.Add(new MapNode(q.PointType, q.coord.row, q.coord.col, visited.Contains(q), ReferenceEquals(q, cur),
                ahead?.Contains(q) ?? true, mask.GetValueOrDefault(q)));
        return res;
    }

    public static string EncounterId(Sts2Sim.Core.Content.EncounterDefinition? e) => e is null ? "" : "ENCOUNTER." + e.IdEntry;

    /// <summary>Layout v2 state inputs: the act's boss(es) and the per-act counters. The counters are the larger of the
    /// agent's bookkeeping (<see cref="FloorFacts.NoteRoom"/>: actual rooms) and the types of the map points visited
    /// in this act (which misses rooms behind Unknown points, but needs no bookkeeping). The potion-slot scalars are
    /// set in <see cref="FillEntities"/>.</summary>
    public static void FillV2(MacroRow r, RunState run, FloorFacts facts, MacroVocab v)
    {
        r.I[ML.OBoss] = v.Tok("encounter", EncounterId(run.CurrentActBossEncounter));
        r.I[ML.OBoss2] = v.Tok("encounter", EncounterId(run.CurrentActSecondBossEncounter));
        int elites = 0, rests = 0, shops = 0;
        if (run.Map is { } map)
            foreach (MapCoord c in run.VisitedMapCoords)
                switch (map.GetPoint(c)?.PointType)
                {
                    case MapPointType.Elite: elites++; break;
                    case MapPointType.RestSite: rests++; break;
                    case MapPointType.Shop: shops++; break;
                }
        if (facts.CountAct == run.CurrentActIndex)
        {
            elites = Math.Max(elites, facts.ActElites); rests = Math.Max(rests, facts.ActRests); shops = Math.Max(shops, facts.ActShops);
        }
        int at = ML.FScal2;
        r.F[at + ML.XElites] = elites / 4f; r.F[at + ML.XRests] = rests / 4f; r.F[at + ML.XShops] = shops / 3f;
        r.F[at + ML.XMapRow] = (run.CurrentMapCoord?.row ?? 0) / 16f;
    }

    public static void FillEntities(MacroRow r, Player p, FloorFacts facts, MacroVocab v, int g)
    {
        var deck = p.Deck.Cards;
        for (int j = 0; j < Math.Min(deck.Count, ML.MaxDeck); j++) CardSlot(r, j, deck[j], v, g);
        var relics = p.Relics;
        for (int j = 0; j < Math.Min(relics.Count, ML.MaxRel); j++)
        {
            r.I[ML.ORIds + j] = v.Tok("relic", relics[j].Id.ToString());
            r.I[ML.ORAge + j] = Math.Clamp(g - facts.RelicFloor.GetValueOrDefault(relics[j], g), 0, 63);
        }
        r.F[ML.FScal2 + ML.XPotSlots] = p.PotionSlots.Count / 5f;   // before SetPotions: it derives the free slots
        r.SetPotions(PotionToks(p.PotionSlots, v));
        DeckScalars(r, deck.Count, Upgraded(deck), Enchanted(deck));
        r.F[ML.SRelics] = relics.Count / 20f;
    }

    /// <summary>Deck slot <paramref name="j"/>: card id, upgrade level, first enchantment and its amount, age (floors since added).</summary>
    public static void CardSlot(MacroRow r, int j, CardModel c, MacroVocab v, int g)
    {
        var e = c.Enchantments.Count > 0 ? c.Enchantments[0] : null;
        r.I[ML.ODIds + j] = v.Tok("card", CardId(c));
        r.I[ML.ODUpg + j] = Math.Clamp(c.CurrentUpgradeLevel, 0, 3);
        r.I[ML.ODEnch + j] = v.Tok("ench", e?.Id.ToString());
        r.I[ML.ODEamt + j] = e is null ? 0 : Math.Clamp((int)e.Magnitude, 0, 20);
        r.I[ML.ODAge + j] = Math.Clamp(g - (c.FloorAddedToDeck ?? g), 0, 63);
    }

    public static int Upgraded(IEnumerable<CardModel> deck) => deck.Count(c => c.CurrentUpgradeLevel > 0);
    public static int Enchanted(IEnumerable<CardModel> deck) => deck.Count(c => c.Enchantments.Count > 0);

    /// <summary>Deck scalars of a deck of <paramref name="n"/> cards, <paramref name="upg"/> of them upgraded and <paramref name="ench"/> enchanted.</summary>
    public static void DeckScalars(MacroRow r, int n, int upg, int ench)
    {
        r.F[ML.SDeck] = n / 40f; r.F[ML.SUpg] = upg / 20f; r.F[ML.SEnch] = ench / 10f;
    }

    /// <summary>Potion tokens of a belt in slot order (empty slots skipped; the dataset keeps the last 5).</summary>
    public static List<int> PotionToks(IEnumerable<PotionModel?> slots, MacroVocab v)
    {
        var pots = new List<int>();
        foreach (PotionModel? pm in slots) if (pm is not null) pots.Add(v.Tok("potion", pm.Id.ToString()));
        return pots.Count > ML.MaxPot ? pots.GetRange(pots.Count - ML.MaxPot, ML.MaxPot) : pots;
    }

    // ---- lookahead post-states (RunAgent look_*): the state row b = Encode(run, p, ...) of the floor now with one change,
    // encoded as Encode would encode the player after that change (deck stays contiguous; counts recomputed).

    /// <summary>Deck slot <paramref name="slot"/> removed: the later slots move down one (a 65th card, if any, enters slot 63).</summary>
    public static MacroRow WithoutCard(MacroRow b, Player p, int slot, MacroVocab v, int g)
    {
        MacroRow r = b.Clone();
        var deck = p.Deck.Cards;
        foreach (int o in new[] { ML.ODIds, ML.ODUpg, ML.ODEnch, ML.ODEamt, ML.ODAge })
        {
            Array.Copy(r.I, o + slot + 1, r.I, o + slot, ML.MaxDeck - slot - 1);
            r.I[o + ML.MaxDeck - 1] = 0;
        }
        if (deck.Count > ML.MaxDeck) CardSlot(r, ML.MaxDeck - 1, deck[ML.MaxDeck], v, g);
        CardModel c = deck[slot];
        DeckScalars(r, deck.Count - 1, Upgraded(deck) - (c.CurrentUpgradeLevel > 0 ? 1 : 0), Enchanted(deck) - (c.Enchantments.Count > 0 ? 1 : 0));
        return r;
    }

    /// <summary>The card in deck slot <paramref name="slot"/> upgraded once.</summary>
    public static MacroRow WithUpgrade(MacroRow b, Player p, int slot)
    {
        MacroRow r = b.Clone();
        var deck = p.Deck.Cards;
        CardModel c = deck[slot];
        r.I[ML.ODUpg + slot] = Math.Clamp(c.CurrentUpgradeLevel + 1, 0, 3);
        DeckScalars(r, deck.Count, Upgraded(deck) + (c.CurrentUpgradeLevel == 0 ? 1 : 0), Enchanted(deck));
        return r;
    }

    /// <summary>Card <paramref name="c"/> added at the end of the deck (age 0).</summary>
    public static MacroRow WithCard(MacroRow b, Player p, CardModel c, MacroVocab v, int g)
    {
        MacroRow r = b.Clone();
        var deck = p.Deck.Cards;
        if (deck.Count < ML.MaxDeck) CardSlot(r, deck.Count, c, v, g);
        DeckScalars(r, deck.Count + 1, Upgraded(deck) + (c.CurrentUpgradeLevel > 0 ? 1 : 0), Enchanted(deck) + (c.Enchantments.Count > 0 ? 1 : 0));
        return r;
    }

    /// <summary>Relic token <paramref name="tok"/> added after the relics held (age 0).</summary>
    public static MacroRow WithRelic(MacroRow b, Player p, int tok)
    {
        MacroRow r = b.Clone();
        int n = p.Relics.Count;
        if (n < ML.MaxRel) { r.I[ML.ORIds + n] = tok; r.I[ML.ORAge + n] = 0; }
        r.F[ML.SRelics] = (n + 1) / 20f;
        return r;
    }

    /// <summary>Potion <paramref name="pm"/> put into the first free belt slot (as Player.AddPotionInternal); unchanged without one.</summary>
    public static MacroRow WithPotion(MacroRow b, Player p, PotionModel pm, MacroVocab v)
    {
        MacroRow r = b.Clone();
        var slots = p.PotionSlots.ToList();
        int free = slots.IndexOf(null);
        if (free < 0) return r;
        slots[free] = pm;
        r.SetPotions(PotionToks(slots, v));
        return r;
    }

    public static MacroRow WithGold(MacroRow r, int gold)
    {
        r.F[ML.SGold] = gold / 500f;
        return r;
    }

    /// <summary>Median gold reward by room type measured on human fights (src/s0/make_tf_tables.py GOLD).</summary>
    public static float ExpectedGold(RoomType t) => t switch { RoomType.Elite => 32f, RoomType.Boss => 75f, _ => 13f };

    /// <summary>
    /// Post-fight state of the fight in progress (PROPOSAL.md §4.2): the state after this floor if the fight is won
    /// now and no reward card is taken; deck and relics as they are, gold + the expected gold reward. HP and potions
    /// are set per table row by the caller.
    /// </summary>
    public static MacroRow PostFight(RunState run, Player p, FloorFacts facts, MacroVocab v, RoomType roomType)
    {
        MacroRow r = Encode(run, p, facts, v);
        r.F[ML.SGold] = (p.Gold + ExpectedGold(roomType)) / 500f;
        return r;
    }
}
