using System.Text.Json;
using System.Text.Json.Serialization;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Content.Acts;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.Core.Runs.Transplant;

namespace Sts2Sim.S0;

/// <summary>One card of a logged deck (Spire Codex / game .run ids, e.g. CARD.ZAP, ENCHANTMENT.SHARP).</summary>
public sealed record BattleCard(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("upgrade")] int Upgrade,
    [property: JsonPropertyName("enchantment")] string? Enchantment,
    [property: JsonPropertyName("enchantment_amount")] int EnchantmentAmount);

/// <summary>A fight start as written by src/s0/export_battle_starts.py (state after floor g-1, fight of floor g).</summary>
public sealed record BattleStartSpec
{
    [JsonPropertyName("id")] public string Id { get; init; } = "";
    [JsonPropertyName("build_id")] public string BuildId { get; init; } = "";
    [JsonPropertyName("character")] public string Character { get; init; } = "CHARACTER.DEFECT";
    [JsonPropertyName("ascension")] public int Ascension { get; init; }
    [JsonPropertyName("act_ids")] public List<string> ActIds { get; init; } = new();
    [JsonPropertyName("act_index")] public int ActIndex { get; init; }
    [JsonPropertyName("floor")] public int Floor { get; init; }
    [JsonPropertyName("room_type")] public string RoomType { get; init; } = "monster";
    [JsonPropertyName("encounter_id")] public string EncounterId { get; init; } = "";
    [JsonPropertyName("monster_ids")] public List<string> MonsterIds { get; init; } = new();
    [JsonPropertyName("hp")] public int Hp { get; init; }
    [JsonPropertyName("max_hp")] public int MaxHp { get; init; }
    [JsonPropertyName("gold")] public int Gold { get; init; }
    [JsonPropertyName("deck")] public List<BattleCard> Deck { get; init; } = new();
    [JsonPropertyName("relics")] public List<string> Relics { get; init; } = new();
    [JsonPropertyName("potions")] public List<string> Potions { get; init; } = new();
    [JsonPropertyName("potion_slots")] public int PotionSlots { get; init; }

    public static IEnumerable<BattleStartSpec> ReadJsonl(string path)
    {
        foreach (string line in File.ReadLines(path))
            if (!string.IsNullOrWhiteSpace(line))
                yield return JsonSerializer.Deserialize<BattleStartSpec>(line)!;
    }
}

/// <summary>Why a battle start could not be imported (unknown id, content the simulator lacks, ...).</summary>
public sealed class BattleImportException(string cause, string detail, Exception? inner = null)
    : Exception($"{cause}: {detail}", inner)
{
    public string Cause { get; } = cause;
}

/// <summary>
/// Builds a fight from a logged battle start through iRyougi's own transplant importer (injected-combat mode:
/// <see cref="RunTransplantImporter.Import"/> + <see cref="RunDriver.RunOneInjectedCombatAsync"/>). Deck,
/// upgrades, enchantments, relics, potions, HP, max HP, gold, ascension, act and encounter come from the log.
/// Everything the log does not contain (RNG states, relic bags, card-reward odds, relic counters) comes from a
/// fresh run with <c>simSeed</c>, so the fight's draws and enemy rolls are not the logged ones.
/// </summary>
public static class BattleStartLoader
{
    public static RunState BuildRun(BattleStartSpec spec, string simSeed)
    {
        if (spec.ActIndex < 0 || spec.ActIndex > 2)
            throw new BattleImportException("unsupported_act", $"act index {spec.ActIndex}");
        var acts = spec.ActIds.Take(3).Select((id, i) => id switch
        {
            "ACT.OVERGROWTH" => (ActDefinition)new Overgrowth(),
            "ACT.UNDERDOCKS" => new Underdocks(),
            "ACT.HIVE" => new Hive(),
            "ACT.GLORY" => new Glory(),
            _ => throw new BattleImportException("unknown_act", id),
        }).ToList();
        CharacterModel character;
        try { character = ModelDb.GetById<CharacterModel>(ModelId.Deserialize(spec.Character)); }
        catch (Exception e) { throw new BattleImportException("unknown_character", spec.Character, e); }

        var fresh = new RunState(simSeed, acts, spec.Ascension);
        Player freshPlayer = Player.CreateForNewRun(character, fresh);
        fresh.AddPlayer(freshPlayer);

        var empty = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        var deck = spec.Deck.Select(c => new CardSnapshot(c.Id, c.Upgrade,
            string.IsNullOrEmpty(c.Enchantment) ? null : c.Enchantment,
            string.IsNullOrEmpty(c.Enchantment) ? null : c.EnchantmentAmount,
            spec.Floor - 1, c.Id == "CARD.SPOILS_MAP" ? SpoilsMapProps : empty)).ToList();
        var relics = spec.Relics.Select(id => new RelicSnapshot(id, empty)).ToList();
        int slots = Math.Max(Math.Max(spec.PotionSlots, freshPlayer.PotionSlots.Count), spec.Potions.Count);
        var potions = spec.Potions.Cast<string?>().Concat(Enumerable.Repeat<string?>(null, slots - spec.Potions.Count)).ToList();
        var player = new PlayerSnapshot(Math.Clamp(spec.Hp, 1, spec.MaxHp), spec.MaxHp, Math.Max(0, spec.Gold), 0,
            deck, relics, potions, freshPlayer.PlayerRng.ToSerializable(), freshPlayer.Odds.ToSerializable(),
            freshPlayer.RelicGrabBag.ExportForTransplant())
        {
            MaxEnergy = freshPlayer.MaxEnergy,
            BaseOrbSlotCount = character.BaseOrbSlotCount,
        };
        var progress = new RoomProgressSnapshot(0, 0, 0, 0, [], [], [], [], null)
        {
            NextRoomId = 0, CurrentMapPointHasShop = false, PreviousMapPointHasShop = false,
        };
        var snapshot = new RunTransplantSnapshot(RunTransplantSnapshot.CurrentSchemaVersion, simSeed,
            character.Id.ToString(), spec.Ascension, spec.ActIndex, acts.Select(a => a.GetType().Name).ToArray(),
            [], spec.Floor, null, ParseRoomType(spec.RoomType).ToString(), player, fresh.Rng.ToSerializable(),
            fresh.Odds.ToSerializable(), progress, fresh.SharedRelicGrabBag!.ExportForTransplant(), empty)
        { IsInjected = true };
        try { return RunTransplantImporter.Import(snapshot); }
        catch (TransplantMappingException e) { throw new BattleImportException(Classify(e), e.Message, e); }
    }

    // SpoilsMap (a quest card) must carry its cross-room map coordinate; the log has none, so it is imported unset.
    private static readonly Dictionary<string, JsonElement> SpoilsMapProps = new(StringComparer.Ordinal)
    {
        ["spoils_coord"] = JsonDocument.Parse("null").RootElement,
    };

    public static RoomType ParseRoomType(string roomType) => roomType switch
    {
        "monster" => RoomType.Monster,
        "elite" => RoomType.Elite,
        "boss" => RoomType.Boss,
        _ => throw new BattleImportException("unsupported_room_type", roomType),
    };

    public static string EncounterEntry(BattleStartSpec spec) =>
        spec.EncounterId.StartsWith("ENCOUNTER.", StringComparison.Ordinal) ? spec.EncounterId["ENCOUNTER.".Length..] : spec.EncounterId;

    /// <summary>Starts the fight and returns a search clone of the state at the player's first decision
    /// (combat-start hooks and the first draw done). The live combat is abandoned.</summary>
    public static async Task<CombatState> LoadAsync(BattleStartSpec spec, string simSeed)
    {
        RunState run = BuildRun(spec, simSeed);
        var capture = new CaptureSource();
        var driver = new RunDriver(run, capture);
        try
        {
            await driver.RunOneInjectedCombatAsync(ParseRoomType(spec.RoomType), EncounterEntry(spec));
        }
        catch (CaptureDone) { return capture.Captured!; }
        catch (TransplantMappingException e) { throw new BattleImportException(Classify(e), e.Message, e); }
        throw new BattleImportException("ended_before_first_decision", spec.Id);
    }

    public enum MonsterMatch { NotChecked, Exact, Subset, Different }

    /// <summary>
    /// Like <see cref="LoadAsync"/>, but encounters whose monster set is random (e.g. BOWLBUGS_NORMAL, SLIMES_WEAK)
    /// are retried with simulator seeds <c>simSeed#0, #1, ...</c> until the starting enemies equal the logged
    /// <c>monster_ids</c> (Exact). If no try matches, the first try whose enemies are a subset of the log is used
    /// (Subset: the log also lists monsters spawned during the fight), else the first try (Different).
    /// </summary>
    public static async Task<(CombatState State, string SimSeed, MonsterMatch Match)> LoadMatchedAsync(
        BattleStartSpec spec, string simSeed, int maxTries = 64)
    {
        if (spec.MonsterIds.Count == 0)
            return (await LoadAsync(spec, simSeed), simSeed, MonsterMatch.NotChecked);
        var want = spec.MonsterIds.Order(StringComparer.Ordinal).ToList();
        (CombatState, string)? subset = null, first = null;
        for (int t = 0; t < maxTries; t++)
        {
            string seed = $"{simSeed}#{t}";
            CombatState s = await LoadAsync(spec, seed);
            var have = s.Enemies.Select(e => e.Monster!.Id.ToString()).Order(StringComparer.Ordinal).ToList();
            if (have.SequenceEqual(want)) return (s, seed, MonsterMatch.Exact);
            first ??= (s, seed);
            var left = new List<string>(want);
            if (subset is null && have.All(left.Remove)) subset = (s, seed);
        }
        if (subset is { } sub) return (sub.Item1, sub.Item2, MonsterMatch.Subset);
        return (first!.Value.Item1, first.Value.Item2, MonsterMatch.Different);
    }

    /// <summary>Plays the live (non-clone) fight to the end with <paramref name="policy"/>.</summary>
    public static async Task<CombatRoom> PlayLiveAsync(BattleStartSpec spec, string simSeed, IRunDecisionSource policy)
    {
        RunState run = BuildRun(spec, simSeed);
        var driver = new RunDriver(run, policy);
        try { return await driver.RunOneInjectedCombatAsync(ParseRoomType(spec.RoomType), EncounterEntry(spec)); }
        catch (TransplantMappingException e) { throw new BattleImportException(Classify(e), e.Message, e); }
    }

    private static string Classify(TransplantMappingException e)
    {
        string m = System.Text.RegularExpressions.Regex.Replace(e.Message, @"\[\d+\]", "[i]");
        if (m.Contains("EncounterId")) return "encounter_not_in_act_pool";
        if (m.Contains(".Id") || m.Contains("EnchantmentId")) return "unknown_or_unsupported_id";
        return "transplant_mapping";
    }

    private sealed class CaptureDone : Exception;

    private sealed class CaptureSource : IRunDecisionSource
    {
        public CombatState? Captured;
        public Task<Sts2Sim.Core.Map.MapPoint> ChooseMapPointAsync(IReadOnlyList<Sts2Sim.Core.Map.MapPoint> options) =>
            throw new InvalidOperationException("No map in an injected combat.");
        public Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
        {
            Captured = state.Clone();
            throw new CaptureDone();
        }
    }
}
