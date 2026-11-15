using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Entities.Creatures;

using Sts2Sim.Core.Models.Monsters;
using Sts2Sim.Core.Models.Powers;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Relics;
using Sts2Sim.Core.MonsterMoves.Intents;
using Sts2Sim.Core.ValueProps;
using Sts2Sim.S0;
using System.Reflection;

/// <summary>
/// obsv5 [seed=OBSV5]: observation layout v5/v6 checks on a Knowledge Demon fight (starter Defect deck + Pen Nib, A0).
/// Ends the first turn (the curse prompt is answered by the random fallback), which brings Slap (a single attack) to
/// the enemy's intent, then adds 3 Strength to the demon and 2 Vulnerable to the player. Expectations are computed
/// independently of the encoder:
///  - obs 4 (default): enemy damage = raw 17, relic features 0/0, Version 4;
///  - obs 5 (relic counters in the masked v5 slots): damage stays raw, Pen Nib 9/1, Version 5;
///  - obs 6 (also corrected intent damage): damage floor((17+3)*1.5) = 30, Pen Nib 9/1, Version 6;
///  - the token list (kinds/ids/other features) is otherwise unchanged in every mode, so no flag can reorder or drop
///    tokens, and the live-network slots stay untouched in obs 4 and 5.
/// Prints the tokens and PASS/FAIL; returns 1 on failure.
/// </summary>
static class ObsV5Test
{
    private const BindingFlags Inst = BindingFlags.Instance | BindingFlags.NonPublic;

    public static async Task<int> Run(string[] args)
    {
        string seed = args.Length > 1 ? args[1] : "OBSV5";
        int fails = 0;
        void Check(bool ok, string what)
        {
            if (!ok) fails++;
            Console.WriteLine($"{(ok ? "PASS" : "FAIL")} {what}");
        }

        var deck = new List<BattleCard>();
        for (int i = 0; i < 4; i++) deck.Add(new BattleCard("CARD.STRIKE_DEFECT", 0, null, 0));
        for (int i = 0; i < 4; i++) deck.Add(new BattleCard("CARD.DEFEND_DEFECT", 0, null, 0));
        deck.Add(new BattleCard("CARD.ZAP", 0, null, 0));
        deck.Add(new BattleCard("CARD.DUALCAST", 0, null, 0));
        var spec = new BattleStartSpec
        {
            Id = "obsv5", Ascension = 0, ActIds = ["ACT.UNDERDOCKS", "ACT.HIVE", "ACT.GLORY"], ActIndex = 1, Floor = 33,
            RoomType = "boss", EncounterId = "ENCOUNTER.KNOWLEDGE_DEMON_BOSS", MonsterIds = ["MONSTER.KNOWLEDGE_DEMON"],
            Hp = 70, MaxHp = 75, Gold = 0, Deck = deck, Relics = ["RELIC.CRACKED_CORE", "RELIC.PEN_NIB"], PotionSlots = 3,
        };
        CombatState s = await BattleStartLoader.LoadAsync(spec, seed);
        Creature player = s.Players[0].Creature;
        Creature demon = s.Enemies[0];

        Check(demon.Monster is KnowledgeDemon, $"enemy is KnowledgeDemon ({demon.Monster?.Id})");
        Check(demon.Monster?.NextMove?.StateId == "CURSE_OF_KNOWLEDGE_MOVE", $"first move {demon.Monster?.NextMove?.StateId}");
        bool ended = await ActionSpace.ApplyAsync(s, SearchAction.End, SearchCardSelectionSource.RandomPolicy(new Random(3)));
        Check(ended && s.Engine is { IsInProgress: true }, $"end turn 1 applied (round {s.RoundNumber})");

        AttackIntent? atk = demon.Monster?.NextMove?.Intents.OfType<AttackIntent>().FirstOrDefault();
        Check(atk is not null && demon.Monster?.NextMove?.StateId == "SLAP_MOVE", $"second move {demon.Monster?.NextMove?.StateId}, attack {atk is not null}");
        if (atk?.DamageCalc is null) { Console.WriteLine("cannot test without an attack intent"); return 1; }

        await PowerCmd.Apply<StrengthPower>(s, demon, 3m, demon, null);
        await PowerCmd.Apply<VulnerablePower>(s, player, 2m, demon, null);
        int raw = atk.GetSingleDamage(s.Allies, demon);
        int expect = Math.Max(0, (int)((raw + 3) * 1.5m));

        // Pen Nib: 9 attacks played, so the counter (and the game's badge) is 9/10.
        PenNib nib = s.Players[0].Relics.OfType<PenNib>().First();
        typeof(PenNib).GetField("_attacksPlayed", Inst)!.SetValue(nib, 9);

        bool savedDamage = CombatObservation.ModifiedIntentDamage, savedCounters = CombatObservation.RelicCounters;
        try
        {
            List<ObsToken> Encode(bool counters, bool damage, out int version)
            {
                CombatObservation.RelicCounters = counters;
                CombatObservation.ModifiedIntentDamage = damage;
                version = CombatObservation.Version;
                return CombatObservation.Encode(s, ObsLayout.MaxTokens);
            }

            List<ObsToken> off = Encode(false, false, out int v4);
            Check(v4 == 4, $"obs 4 Version {v4}");
            List<ObsToken> legacy = LegacyEncode(s, ObsLayout.MaxTokens);
            Check(SameTokens(off, legacy), "obs 4 tokens are identical to the pre-change encoder");
            string fpOff = ObservableState.Fingerprint(s);
            Check(!fpOff.Contains("RELIC.PEN_NIB#9"), "obs 4 fingerprint has no relic counter");
            Check(off.First(t => t.Kind == TokenKind.Enemy).F[5] == raw, $"obs 4 intent damage {off.First(t => t.Kind == TokenKind.Enemy).F[5]} == raw {raw}");
            Check(off.First(t => t.Kind == TokenKind.Relic && t.Id == "RELIC.PEN_NIB").F[0] == 0, "obs 4 relic features 0");

            List<ObsToken> v5 = Encode(true, false, out int v5v);
            Check(v5v == 5, $"obs 5 Version {v5v}");
            Check(v5.First(t => t.Kind == TokenKind.Enemy).F[5] == raw, $"obs 5 intent damage stays raw {raw}");
            ObsToken nib5 = v5.First(t => t.Kind == TokenKind.Relic && t.Id == "RELIC.PEN_NIB");
            Check(nib5.F[0] == 9 && nib5.F[1] == 1, $"obs 5 Pen Nib features [{nib5.F[0]},{nib5.F[1]}]");
            Check(v5.Count == off.Count, $"obs 5 token count unchanged ({v5.Count} vs {off.Count})");
            Check(SameTokensExcept(off, v5, enemy: false, relic: true), "obs 5 changed only the relic features");
            string fp5 = ObservableState.Fingerprint(s);
            Check(fp5.Contains("RELIC.PEN_NIB#9"), "obs 5 fingerprint carries the relic counter");

            List<ObsToken> v6 = Encode(true, true, out int v6v);
            Check(v6v == 6, $"obs 6 Version {v6v}");
            int dmg6 = v6.First(t => t.Kind == TokenKind.Enemy).F[5];
            Check(dmg6 == expect, $"obs 6 intent damage {dmg6} == ({raw}+3)*1.5 = {expect}");
            ObsToken nib6 = v6.First(t => t.Kind == TokenKind.Relic && t.Id == "RELIC.PEN_NIB");
            Check(nib6.F[0] == 9 && nib6.F[1] == 1, $"obs 6 Pen Nib features [{nib6.F[0]},{nib6.F[1]}]");
            Check(v6.Count == off.Count, $"obs 6 token count unchanged ({v6.Count} vs {off.Count})");
            Check(SameTokensExcept(off, v6, enemy: true, relic: true), "obs 6 changed only the enemy damage and relic features");

            Console.WriteLine($"  intent damage: raw {raw}, obs 6 {expect}; Pen Nib 9/10; tokens {v6.Count}");
            foreach (ObsToken t in v6.Where(t => t.Kind is TokenKind.Enemy or TokenKind.Relic))
                Console.WriteLine($"  {t.Kind} {t.Id} [{string.Join(",", t.F)}]");
        }
        finally
        {
            CombatObservation.ModifiedIntentDamage = savedDamage;
            CombatObservation.RelicCounters = savedCounters;
        }

        Console.WriteLine(fails == 0 ? "ALL PASS" : $"{fails} checks FAILED");
        return fails == 0 ? 0 : 1;

        static bool SameTokensExcept(List<ObsToken> a, List<ObsToken> b, bool enemy, bool relic)
        {
            if (a.Count != b.Count) return false;
            for (int i = 0; i < a.Count; i++)
            {
                if (a[i].Kind != b[i].Kind || a[i].Id != b[i].Id || a[i].Id2 != b[i].Id2) return false;
                for (int f = 0; f < CombatObservation.FeatureCount; f++)
                {
                    bool allowed = (enemy && a[i].Kind == TokenKind.Enemy && f == 5) ||
                                   (relic && a[i].Kind == TokenKind.Relic && f <= 1);
                    if (!allowed && a[i].F[f] != b[i].F[f]) return false;
                }
            }
            return true;
        }

        static bool SameTokens(List<ObsToken> a, List<ObsToken> b)
        {
            if (a.Count != b.Count) return false;
            for (int i = 0; i < a.Count; i++)
            {
                if (a[i].Kind != b[i].Kind || a[i].Id != b[i].Id || a[i].Id2 != b[i].Id2) return false;
                for (int f = 0; f < CombatObservation.FeatureCount; f++)
                    if (a[i].F[f] != b[i].F[f]) return false;
            }
            return true;
        }
    }

    /// <summary>The pre-2026-11-15 encoder body (layout v4, no opt-in flags), kept here as the off-switch oracle.</summary>
    private static List<ObsToken> LegacyEncode(CombatState state, int maxTokens)
    {
        var t = new List<ObsToken>(maxTokens);
        Player p = state.Players[0];
        PlayerCombatState pcs = p.PlayerCombatState!;
        Creature me = p.Creature;
        t.Add(new(TokenKind.Global, "", "", F(state.RoundNumber, pcs.Energy, pcs.MaxEnergy, pcs.Stars,
            me.CurrentHp, me.MaxHp, me.Block, pcs.OrbQueue.Capacity)));
        for (int i = 0; i < pcs.Hand.Cards.Count; i++)
        {
            CardModel c = pcs.Hand.Cards[i];
            var e = c.Enchantments.FirstOrDefault();
            t.Add(new(TokenKind.Hand, c.Id.ToString(), e?.Id.ToString() ?? "",
                F(i, c.CurrentUpgradeLevel, c.EnergyCost, c.CanPlay(out _) ? 1 : 0, (int)(e?.Magnitude ?? 0),
                    CardValues.Damage(c), CardValues.Block(c))));
        }
        var draw = pcs.DrawPile;
        var knownTop = new Dictionary<CardModel, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < draw.KnownTopCount; i++) knownTop[draw.Cards[i]] = i;
        AddPile(t, TokenKind.Draw, draw.Cards, c => knownTop.TryGetValue(c, out int pos) ? pos : -1);
        AddPile(t, TokenKind.Discard, pcs.DiscardPile.Cards, null);
        AddPile(t, TokenKind.Exhaust, pcs.ExhaustPile.Cards, null);
        for (int i = 0; i < state.Enemies.Count; i++)
        {
            Creature en = state.Enemies[i];
            var move = en.Monster?.NextMove;
            int type = -1, dmg = 0, hits = 0;
            if (move?.Intents.FirstOrDefault() is { } intent)
            {
                type = (int)intent.IntentType;
                if (move.Intents.OfType<AttackIntent>().FirstOrDefault() is { } atk && !en.IsDead)
                {
                    hits = atk.Repeats;
                    dmg = atk.GetSingleDamage(state.Allies, en);
                }
            }
            t.Add(new(TokenKind.Enemy, en.Monster?.Id.ToString() ?? "", type < 0 ? "" : "INTENT." + (IntentType)type,
                F(i, en.CurrentHp, en.MaxHp, en.Block, type, dmg, hits, en.IsDead ? 0 : 1)));
            foreach (PowerModel pw in en.Powers.OrderBy(x => x.Id.ToString(), StringComparer.Ordinal))
                t.Add(new(TokenKind.EnemyPower, pw.Id.ToString(), "", F(i, pw.Amount, PowerCounters.Get(pw))));
        }
        foreach (PowerModel pw in me.Powers.OrderBy(x => x.Id.ToString(), StringComparer.Ordinal))
            t.Add(new(TokenKind.PlayerPower, pw.Id.ToString(), "", F(pw.Amount, PowerCounters.Get(pw))));
        for (int i = 0; i < pcs.OrbQueue.Orbs.Count; i++)
        {
            var o = pcs.OrbQueue.Orbs[i];
            t.Add(new(TokenKind.Orb, o.Id.ToString(), "", F(i, (int)o.PassiveVal, (int)o.EvokeVal)));
        }
        for (int i = 0; i < p.PotionSlots.Count; i++)
            if (p.PotionSlots[i] is { } pot) t.Add(new(TokenKind.Potion, pot.Id.ToString(), "", F(i)));
        int room = maxTokens - t.Count;
        foreach (var r in p.Relics.Take(Math.Max(0, room)))
            t.Add(new(TokenKind.Relic, r.Id.ToString(), "", F()));
        if (t.Count > maxTokens) t.RemoveRange(maxTokens, t.Count - maxTokens);
        while (t.Count < maxTokens) t.Add(new(TokenKind.Pad, "", "", F()));
        return t;
    }

    private static void AddPile(List<ObsToken> t, TokenKind kind, IReadOnlyList<CardModel> cards, Func<CardModel, int>? topPos)
    {
        var groups = cards.GroupBy(c =>
        {
            var e = c.Enchantments.FirstOrDefault();
            return (Id: c.Id.ToString(), Up: c.CurrentUpgradeLevel, E: e?.Id.ToString() ?? "", Ea: (int)(e?.Magnitude ?? 0),
                Pos: topPos?.Invoke(c) ?? -1);
        }).OrderBy(g => g.Key.Pos < 0 ? int.MaxValue : g.Key.Pos).ThenBy(g => g.Key.Id, StringComparer.Ordinal)
          .ThenBy(g => g.Key.Up).ThenBy(g => g.Key.E, StringComparer.Ordinal);
        foreach (var g in groups)
        {
            int dmg = g.Max(CardValues.Damage), blk = g.Max(CardValues.Block);
            t.Add(new(kind, g.Key.Id, g.Key.E, kind == TokenKind.Draw
                ? F(g.Count(), g.Key.Up, g.Key.Ea, g.Key.Pos, dmg, blk)
                : F(g.Count(), g.Key.Up, g.Key.Ea, dmg, blk)));
        }
    }

    private static int[] F(params int[] xs)
    {
        var f = new int[CombatObservation.FeatureCount];
        Array.Copy(xs, f, Math.Min(xs.Length, CombatObservation.FeatureCount));
        return f;
    }
}
