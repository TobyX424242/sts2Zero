using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Cards;
using Sts2Sim.S0;

namespace Sts2Sim.S3;

/// <summary>
/// Layout v3 (2026-11-16): static content properties of every card / relic / potion id of the macro vocabulary, read
/// from the SIMULATOR's model classes (iRyougi sts2-sim, MIT): <c>dotnet S3.dll props out=FILE mvocab=FILE</c> writes
/// src/s3/data/content_props.json, which src/s3/py/macro_s3.py turns into feature vectors (a zero-initialised
/// linear map added to the token embeddings) and into the deck counts.
/// <list type="bullet">
/// <item>cards, per upgrade level 0 and 1 (level 1 only when the card is upgradable): type, rarity, target, energy cost
/// (X cost flagged; unplayable flagged), keywords, tags, the simulator's own effect probe (<see cref="CardEffectProbe"/>:
/// damage, block, draw, energy, created cards, powers applied with their amounts ...; Silent sandbox), and orbs channelled
/// by type, observed by playing the card once in a Defect sandbox (the probe's Silent has no orb slots);</item>
/// <item>relics and potions: rarity (potions: usage and target) and <see cref="ItemEffectProbe"/>'s observed effects.</item>
/// </list>
/// Ids the simulator lacks are listed with "missing": true. Nothing here reads game files.
/// </summary>
public static class ContentProps
{
    private static string Entry(string id) => id.Contains('.') ? id[(id.IndexOf('.') + 1)..] : id;

    private static Dictionary<string, object?> Probe(IReadOnlyDictionary<string, float> values, object? unsupported) => new()
    {
        ["values"] = values.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToDictionary(kv => kv.Key, kv => (object)Math.Round(kv.Value, 4)),
        ["unsupported"] = unsupported,
    };

    public static async Task<int> Run(string outPath, string mvocabPath)
    {
        var lists = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(File.ReadAllText(mvocabPath))!;
        var orbBase = await OrbSandbox.CreateAsync();
        Console.WriteLine(orbBase is null ? "orb sandbox: not available (orb counts left out)" : $"orb sandbox: {orbBase.Encounter}");
        var cards = new Dictionary<string, object?>();
        int nCard = 0, nMissing = 0, nOrb = 0;
        foreach (string id in lists["card"].Where(s => s.StartsWith("CARD.", StringComparison.Ordinal)))
        {
            CardModel? canon = ModelDb.GetByIdOrNull<CardModel>(new ModelId("CARD", Entry(id)));
            if (canon is null) { cards[id] = new Dictionary<string, object?> { ["missing"] = true }; nMissing++; continue; }
            var levels = new List<object?>();
            for (int upg = 0; upg <= Math.Min(1, canon.MaxUpgradeLevel); upg++)
            {
                var c = (CardModel)canon.MutableClone();
                for (int i = 0; i < upg; i++) c.Upgrade();
                CardEffectFacts facts;
                try { facts = CardEffectProbe.Probe(Entry(id), upg); }
                catch (Exception e) { facts = new(System.Collections.Immutable.ImmutableDictionary<string, float>.Empty, "probe:" + e.GetType().Name); }
                Dictionary<string, int>? orbs = null;
                if (orbBase is not null)
                {
                    try { orbs = await orbBase.ChannelledAsync(Entry(id), upg); if (orbs.Count > 0) nOrb++; }
                    catch (Exception e) { orbs = new() { ["error:" + e.GetType().Name] = 1 }; }
                }
                levels.Add(new Dictionary<string, object?>
                {
                    ["upgrade"] = upg, ["cost"] = c.EnergyCost, ["x_cost"] = c.CostsXEnergy,
                    ["unplayable"] = c.Keywords.Contains(CardKeyword.Unplayable),
                    ["keywords"] = c.Keywords.Select(k => k.ToString()).Order(StringComparer.Ordinal).ToList(),
                    ["probe"] = Probe(facts.Values, facts.UnsupportedReason), ["orbs"] = orbs,
                });
            }
            cards[id] = new Dictionary<string, object?>
            {
                ["type"] = canon.Type.ToString(), ["rarity"] = canon.Rarity.ToString(), ["target"] = canon.TargetType.ToString(),
                ["tags"] = canon.Tags.Select(t => t.ToString()).Order(StringComparer.Ordinal).ToList(), ["colorless"] = canon.IsColorless,
                ["max_upgrade"] = canon.MaxUpgradeLevel, ["levels"] = levels,
            };
            nCard++;
        }
        var relics = new Dictionary<string, object?>();
        foreach (string id in lists["relic"].Where(s => s.StartsWith("RELIC.", StringComparison.Ordinal)))
        {
            RelicModel? m = ModelDb.GetByIdOrNull<RelicModel>(new ModelId("RELIC", Entry(id)));
            if (m is null) { relics[id] = new Dictionary<string, object?> { ["missing"] = true }; nMissing++; continue; }
            ItemEffectFacts f;
            try { f = ItemEffectProbe.ProbeRelic(Entry(id)); }
            catch (Exception e) { f = new(System.Collections.Immutable.ImmutableDictionary<string, float>.Empty, ["probe:" + e.GetType().Name]); }
            relics[id] = new Dictionary<string, object?>
            {
                ["rarity"] = m.Rarity.ToString(), ["probe"] = Probe(f.Values, f.UnsupportedReasons.IsEmpty ? null : f.UnsupportedReasons.ToList()),
            };
        }
        var potions = new Dictionary<string, object?>();
        foreach (string id in lists["potion"].Where(s => s.StartsWith("POTION.", StringComparison.Ordinal)))
        {
            PotionModel? m = ModelDb.GetByIdOrNull<PotionModel>(new ModelId("POTION", Entry(id)));
            if (m is null) { potions[id] = new Dictionary<string, object?> { ["missing"] = true }; nMissing++; continue; }
            ItemEffectFacts f;
            try { f = ItemEffectProbe.ProbePotion(Entry(id)); }
            catch (Exception e) { f = new(System.Collections.Immutable.ImmutableDictionary<string, float>.Empty, ["probe:" + e.GetType().Name]); }
            potions[id] = new Dictionary<string, object?>
            {
                ["rarity"] = m.Rarity.ToString(), ["usage"] = m.Usage.ToString(), ["target"] = m.TargetType.ToString(),
                ["probe"] = Probe(f.Values, f.UnsupportedReasons.IsEmpty ? null : f.UnsupportedReasons.ToList()),
            };
        }
        var doc = new Dictionary<string, object?>
        {
            ["version"] = 1, ["source"] = "iRyougi sts2-sim model classes (CardEffectProbe, ItemEffectProbe, Defect orb sandbox)",
            ["orb_sandbox"] = orbBase?.Encounter, ["card"] = cards, ["relic"] = relics, ["potion"] = potions,
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath + ".tmp", JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = false }));
        File.Move(outPath + ".tmp", outPath, overwrite: true);
        Console.WriteLine($"content props: {nCard} cards ({nOrb} card levels channel orbs), {relics.Count} relics, {potions.Count} potions, " +
            $"{nMissing} ids missing in the simulator -> {outPath}");
        return 0;
    }

    /// <summary>ic113: the ids of this simulator in the macro vocabulary's categories (src/ic113/convert_v113.py adds
    /// the ones the S1 vocabulary lacks): card, ench, relic, potion, encounter (ENCOUNTER.X of every act's monster / elite /
    /// boss pool, as FloorFacts.RoomModelId writes them, and EVENT.X of every event model).</summary>
    public static int SimIds(string outPath)
    {
        var enc = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var a in new Sts2Sim.Core.Content.ActDefinition[] { new Sts2Sim.Core.Content.Acts.Overgrowth(), new Sts2Sim.Core.Content.Acts.Underdocks(),
                     new Sts2Sim.Core.Content.Acts.Hive(), new Sts2Sim.Core.Content.Acts.Glory() })
            foreach (var e in a.MonsterEncounterCandidates.Concat(a.EliteEncounterCandidates).Concat(a.BossEncounterCandidates))
                enc.Add("ENCOUNTER." + e.IdEntry);
        foreach (var ev in ModelDb.All<Sts2Sim.Core.Models.EventModel>()) enc.Add(ev.Id.ToString());
        List<string> Ids<T>() where T : AbstractModel => ModelDb.All<T>().Select(x => x.Id.ToString()).Distinct().Order(StringComparer.Ordinal).ToList();
        var doc = new Dictionary<string, object?>
        {
            ["build_id"] = ML.BuildId, ["card"] = Ids<CardModel>(), ["ench"] = Ids<EnchantmentModel>(), ["relic"] = Ids<RelicModel>(),
            ["potion"] = Ids<PotionModel>(), ["encounter"] = enc.ToList(),
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, JsonSerializer.Serialize(doc));
        Console.WriteLine($"simulator ids -> {outPath}: " + string.Join(", ", doc.Where(kv => kv.Value is List<string>).Select(kv => $"{kv.Key} {((List<string>)kv.Value!).Count}")));
        return 0;
    }

    /// <summary>ic113: `S3 cardinfo out=FILE`: every card of the simulator (ModelDb) with its type, rarity, colorless flag and
    /// card pools (character pools and the colorless / curse / status pools that list it; "" = in no pool: event, token or
    /// generated cards), for src/ic113/convert_v113.py (type-mean initialisation of new card embeddings).</summary>
    public static int CardInfo(string outPath)
    {
        var pools = typeof(Sts2Sim.Core.Models.CardPools.CardPoolModel).Assembly.GetTypes()
            .Where(t => !t.IsAbstract && typeof(Sts2Sim.Core.Models.CardPools.CardPoolModel).IsAssignableFrom(t) && t.GetConstructor(Type.EmptyTypes) is not null)
            .Select(t => (Name: t.Name.Replace("CardPool", "").ToUpperInvariant(), Pool: (Sts2Sim.Core.Models.CardPools.CardPoolModel)Activator.CreateInstance(t)!))
            .Where(x => x.Name != "EMPTY").ToList();
        var member = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var (name, pool) in pools)
        {
            IReadOnlyList<CardModel> all;
            try { all = pool.AllCards; } catch (Exception) { continue; }
            foreach (CardModel c in all)
            {
                string id = c.Id.ToString();
                if (!member.TryGetValue(id, out var l)) member[id] = l = new();
                if (!l.Contains(name)) l.Add(name);
            }
        }
        var cards = new SortedDictionary<string, object?>(StringComparer.Ordinal);
        foreach (CardModel c in ModelDb.All<CardModel>())
            cards[c.Id.ToString()] = new Dictionary<string, object?>
            {
                ["type"] = c.Type.ToString(), ["rarity"] = c.Rarity.ToString(), ["colorless"] = c.IsColorless, ["multiplayer_only"] = c.IsMultiplayerOnly,
                ["pools"] = member.TryGetValue(c.Id.ToString(), out var l) ? l.Order(StringComparer.Ordinal).ToList() : new List<string>(),
            };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["source"] = "iRyougi sts2-sim ModelDb + CardPoolModel subclasses", ["build_id"] = ML.BuildId,
            ["pools"] = pools.Select(p => p.Name).Order(StringComparer.Ordinal).ToList(), ["card"] = cards,
        }));
        Console.WriteLine($"card info: {cards.Count} cards, pools {string.Join(", ", pools.Select(p => $"{p.Name} {member.Values.Count(l => l.Contains(p.Name))}"))} -> {outPath}");
        return 0;
    }

    /// <summary>A Defect fight with no relics and an empty orb queue (built by the S0 battle loader): each card is
    /// generated into the hand of an exact clone, with 10 energy, and played once; the orbs in the queue afterwards,
    /// by type, are what it channelled (the queue holds 3, so at most 3 are seen).</summary>
    private sealed class OrbSandbox
    {
        public required CombatState Base;
        public required string Encounter;

        public static async Task<OrbSandbox?> CreateAsync()
        {
            string[] encs = { "BATTLEWORN_DUMMY_EVENT_ENCOUNTER", "SHRINKER_BEETLE_WEAK", "AXEBOTS_NORMAL", "NIBBITS_WEAK" };
            var deck = Enumerable.Repeat(new BattleCard("CARD.DEFEND_DEFECT", 0, null, 0), 10).ToList();
            foreach (string e in encs)
                foreach (string act in new[] { "ACT.OVERGROWTH", "ACT.UNDERDOCKS", "ACT.HIVE", "ACT.GLORY" })
                {
                    try
                    {
                        var spec = new BattleStartSpec
                        {
                            Id = "orb-sandbox", Character = "CHARACTER.DEFECT", Ascension = 0, ActIds = new() { act, "ACT.HIVE", "ACT.GLORY" },
                            ActIndex = 0, Floor = 2, RoomType = "monster", EncounterId = "ENCOUNTER." + e, Hp = 70, MaxHp = 70,
                            Deck = deck, PotionSlots = 3,
                        };
                        CombatState s = await BattleStartLoader.LoadAsync(spec, "ORB-SANDBOX");
                        if (s.Players[0].PlayerCombatState!.OrbQueue.Orbs.Count != 0 || s.Enemies.Count == 0) continue;
                        foreach (var en in s.Enemies) { en.SetMaxHpInternal(999); en.SetCurrentHpInternal(999); }
                        return new OrbSandbox { Base = s, Encounter = act + "/" + e };
                    }
                    catch (Exception) { }
                }
            return null;
        }

        public async Task<Dictionary<string, int>> ChannelledAsync(string entry, int upg)
        {
            CombatState s = ResamplingClone.Exact(Base);
            SearchCardSelectionSource.InstallOn(s, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(1)));
            Player p = s.Players[0];
            var card = (CardModel)ModelDb.GetById<CardModel>(new ModelId("CARD", entry)).MutableClone();
            for (int i = 0; i < upg; i++) card.Upgrade();
            card.AssignOwner(p);
            await CardPileCmd.Generate(s, card, PileType.Hand);
            p.PlayerCombatState!.Energy = 10;
            var res = new Dictionary<string, int>();
            if (!card.CanPlay(out _)) return res;
            var target = card.TargetType is TargetType.AnyEnemy ? s.Enemies.FirstOrDefault(e => e.CurrentHp > 0) : null;
            await s.Engine!.PlayCardAsync(p, card, target);
            foreach (var g in p.PlayerCombatState.OrbQueue.Orbs.GroupBy(o => o.GetType().Name)) res[g.Key] = g.Count();
            return res;
        }
    }
}
