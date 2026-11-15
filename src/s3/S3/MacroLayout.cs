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
    public const int ODIds = 0, ODUpg = 64, ODEnch = 128, ODEamt = 192, ODAge = 256, ORIds = 320, ORAge = 360, OPIds = 400,
        OCtx = 405, ODtype = 411, OOptKind = 412, OOptId = 428, OOptUpg = 444, IW = 460;
    public const int FScal = 0, FOptFeat = 16, FW = 16 + K * MF;
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
        return 1;
    }

    public bool Has(string kind, string id) => _m[kind].ContainsKey(id);
    public int SpecialId(string name) => _m["special"].GetValueOrDefault(name, -1);
}

/// <summary>One macro row (state + optional decision) as sent to the network and stored in the run records.</summary>
public sealed class MacroRow
{
    public readonly int[] I = new int[ML.IW];
    public readonly float[] F = new float[ML.FW];
    public int NOptions;

    public MacroRow Clone()
    {
        var r = new MacroRow { NOptions = NOptions };
        I.CopyTo(r.I, 0); F.CopyTo(r.F, 0);
        return r;
    }

    public void ClearDecision()
    {
        I[ML.ODtype] = 0;
        Array.Clear(I, ML.OOptKind, 3 * ML.K);
        Array.Clear(F, ML.FOptFeat, ML.K * ML.MF);
        NOptions = 0;
    }

    public int AddOption(int kind, int id, int upg = 0, ReadOnlySpan<float> feat = default)
    {
        int j = NOptions;
        if (j >= ML.K) return -1;
        I[ML.OOptKind + j] = kind; I[ML.OOptId + j] = id; I[ML.OOptUpg + j] = Math.Clamp(upg, 0, 3);
        for (int f = 0; f < Math.Min(feat.Length, ML.MF); f++) F[ML.FOptFeat + j * ML.MF + f] = feat[f];
        NOptions++;
        return j;
    }

    public int Hp => (int)Math.Round(F[ML.SHp] * 100);
    public int MaxHp => (int)Math.Round(F[ML.SMaxHp] * 100);

    public void SetHp(int hp, int maxHp)
    {
        F[ML.SHp] = hp / 100f; F[ML.SMaxHp] = maxHp / 100f; F[ML.SHpFrac] = hp / (float)Math.Max(1, maxHp);
    }

    /// <summary>Replaces the potion belt (macro potion ids, at most <see cref="ML.MaxPot"/>).</summary>
    public void SetPotions(IReadOnlyList<int> toks)
    {
        for (int j = 0; j < ML.MaxPot; j++) I[ML.OPIds + j] = j < toks.Count ? toks[j] : 0;
        F[ML.SPotions] = Math.Min(toks.Count, ML.MaxPot) / 5f;
    }
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

    public void NoteRelics(Player p, int floor)
    {
        foreach (RelicModel r in p.Relics) RelicFloor.TryAdd(r, Math.Max(1, floor));
    }

    public void NoteRoom(RunState run)
    {
        int g = run.TotalFloor;
        if (FloorModelIdFloor == g) return;
        AbstractRoom? room = run.BaseRoom;
        if (room is null) return;
        FloorModelIdFloor = g;
        FloorModelId = RoomModelId(room);
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
        r.I[ML.OCtx + 0] = ML.CharDefect; r.I[ML.OCtx + 1] = facts.Ascension; r.I[ML.OCtx + 2] = ML.EraV0111;
        r.I[ML.OCtx + 3] = run.CurrentActIndex; r.I[ML.OCtx + 4] = ML.RoomIndex(pt); r.I[ML.OCtx + 5] = v.Tok("encounter", modelId);
        var c = p.Creature;
        r.F[ML.SFloor] = g / 50f; r.F[ML.SAct] = run.CurrentActIndex / 3f;
        r.SetHp(Math.Max(0, c.CurrentHp), c.MaxHp);
        r.F[ML.SGold] = p.Gold / 500f;
        r.F[ML.SIsCombat] = ML.IsCombatPoint(pt) ? 1f : 0f;
        r.F[ML.SFloorInAct] = run.VisitedMapCoords.Count / 20f;
        return r;
    }

    public static void FillEntities(MacroRow r, Player p, FloorFacts facts, MacroVocab v, int g)
    {
        var deck = p.Deck.Cards;
        int nUpg = 0, nEnch = 0;
        for (int j = 0; j < deck.Count; j++)
        {
            CardModel c = deck[j];
            var e = c.Enchantments.Count > 0 ? c.Enchantments[0] : null;
            if (c.CurrentUpgradeLevel > 0) nUpg++;
            if (e is not null) nEnch++;
            if (j >= ML.MaxDeck) continue;
            r.I[ML.ODIds + j] = v.Tok("card", CardId(c));
            r.I[ML.ODUpg + j] = Math.Clamp(c.CurrentUpgradeLevel, 0, 3);
            r.I[ML.ODEnch + j] = v.Tok("ench", e?.Id.ToString());
            r.I[ML.ODEamt + j] = e is null ? 0 : Math.Clamp((int)e.Magnitude, 0, 20);
            r.I[ML.ODAge + j] = Math.Clamp(g - (c.FloorAddedToDeck ?? g), 0, 63);
        }
        var relics = p.Relics;
        for (int j = 0; j < Math.Min(relics.Count, ML.MaxRel); j++)
        {
            r.I[ML.ORIds + j] = v.Tok("relic", relics[j].Id.ToString());
            r.I[ML.ORAge + j] = Math.Clamp(g - facts.RelicFloor.GetValueOrDefault(relics[j], g), 0, 63);
        }
        var pots = new List<int>();
        foreach (PotionModel? pm in p.PotionSlots) if (pm is not null) pots.Add(v.Tok("potion", pm.Id.ToString()));
        if (pots.Count > ML.MaxPot) pots = pots.GetRange(pots.Count - ML.MaxPot, ML.MaxPot);   // the dataset keeps the last 5
        r.SetPotions(pots);
        r.F[ML.SDeck] = deck.Count / 40f; r.F[ML.SUpg] = nUpg / 20f; r.F[ML.SEnch] = nEnch / 10f; r.F[ML.SRelics] = relics.Count / 20f;
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
