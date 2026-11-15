using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Hooks;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.MonsterMoves.Intents;
using Sts2Sim.Core.ValueProps;

namespace Sts2Sim.S0;

/// <summary>Token kinds of the combat-network input (PROPOSAL.md section 6.4). Values are part of the layout.</summary>
public enum TokenKind : byte
{
    Pad = 0, Global = 1, Hand = 2, Draw = 3, Discard = 4, Exhaust = 5,
    Enemy = 6, EnemyPower = 7, PlayerPower = 8, Orb = 9, Potion = 10, Relic = 11,
    /// <summary>Fight context (layout v3): what HP and potions are worth in this fight, from its T_f table.</summary>
    Context = 12,
}

/// <summary>One input token: a kind, a primary content id and a secondary id in game/Spire Codex form
/// ("CARD.ZAP", "ENCHANTMENT.SHARP", "" for none; the Python side maps them through vocab.json, unknown ids to
/// &lt;unk&gt;), and <see cref="CombatObservation.FeatureCount"/> integer features whose meaning depends on the kind
/// (see <see cref="CombatObservation"/>).</summary>
public readonly record struct ObsToken(TokenKind Kind, string Id, string Id2, int[] F);

/// <summary>
/// C# encoder for the combat network (layout v5, <see cref="ObsLayout"/>; mirrored in research/s0/py/combat_obs.py).
/// Only player-observable information is used: the draw pile is a multiset (plus the known top run tracked by the
/// S0 CardPile patch), never its order.
/// <code>
/// kind         Id / Id2                   F[0..7]
/// Global       "" / ""                    round, energy, max energy, stars, hp, max hp, block, orb capacity
/// Hand         card / enchantment         hand pos, upgrade, cost (-1 = X/unplayable), playable, ench amount, damage*, block*, 0
/// Draw         card / enchantment         count, upgrade, ench amount, known top position (-1 if none), damage*, block*, 0...
/// Discard      card / enchantment         count, upgrade, ench amount, damage*, block*, 0...
/// Exhaust      card / enchantment         count, upgrade, ench amount, damage*, block*, 0...
/// Enemy        monster / INTENT.<type>    slot, hp, max hp, block, intent type, damage per hit (+*), hits, alive
/// EnemyPower   power / ""                 owner enemy slot, amount, hidden counter*, 0...
/// PlayerPower  power / ""                 amount, hidden counter*, 0...
/// Orb          orb / ""                   position, passive value, evoke value, 0...
/// Potion       potion / ""                slot, marginal T_f in thousandths of T_max (layout v3, set by S2 FightSignals; 0 here), 0...
/// Relic        relic / ""                 counter*, shows-counter*, 0...
/// </code>
/// Tokens are emitted in this kind order, multisets sorted by id, and the list is truncated/padded to
/// <c>maxTokens</c> (relics are dropped first when truncating).
/// <para>* = layout v4 (2026-11-15), in slots that were always 0 in v3 (networks built without "ext" mask them out,
/// research/s0/py/combat_obs.py V4_SLOTS): damage / block = the card's current own values (<see cref="CardValues"/>:
/// upgrades and permanent or in-fight changes such as Claw or Wither growth; no Strength/Dexterity/Vulnerable);
/// hidden counter = <see cref="PowerCounters"/> (e.g. WitheringPresence: cards left before the next Wither), 0 for
/// powers without one. Pile multisets: by default the grouping is the v3 one (id, upgrade, enchantment[, known
/// position]) and a group's damage / block are the largest of its cards, so the token list is the v3 list exactly;
/// with <see cref="SplitPilesByValue"/> copies with different damage / block become separate tokens (more tokens:
/// the v3 network would then see a different list).</para>
/// <para>+ = layout v5 (2026-11-15, opt-in, <see cref="RelicCounters"/>): relic tokens carry the counters the game
/// shows (Pen Nib 9/10, Nunchaku 9/10, ...) in slots that were always 0. The Python side registers them in
/// <c>V4_SLOTS</c>, so the v3/live network masks them and stays bit-exact; the shadow/ext network reads them. The
/// buffer shape is unchanged.</para>
/// <para>Layout v5.1 / obs 6 (<see cref="ModifiedIntentDamage"/>, 2026-11-15): Enemy f5 carries the intent damage after
/// Strength / Weak / Vulnerable (the game's displayed number). This changes an existing slot that every network reads,
/// so it is a separate opt-in and needs a retrained network; obs 5 alone leaves the number raw.</para>
/// </summary>
public static class CombatObservation
{
    public const int FeatureCount = 8;

    /// <summary>Layout v4 pile grouping (see the class comment). Off by default: the token list then equals the v3 list
    /// for every state, so networks that mask the v4 slots see exactly their v3 input.</summary>
    public static bool SplitPilesByValue = false;

    /// <summary>Layout v5 (2026-11-15, opt-in): encode the enemy intent's damage <b>after</b> <see cref="Hook.ModifyDamage"/>
    /// (Strength, Weak, Vulnerable, ...), which is the number the game shows the player; the default false keeps the
    /// simulator's raw <see cref="AttackIntent.GetSingleDamage"/> (the pre-v5 behaviour, bit-identical).</summary>
    public static bool ModifiedIntentDamage = false;

    /// <summary>Layout v5 (2026-11-15, opt-in): encode the counter of the relics that show one (Pen Nib, Nunchaku,
    /// Happy Flower, Letter Opener, Pendulum, Kunai, Shuriken, Ornamental Fan, Pocketwatch, Velvet Choker, Girya,
    /// Book of Five Rings, Pael's Wing, Toy Box) into Relic F[0] (+1 in F[1] = "this relic has a counter"). The default
    /// false writes zero features, as before. The Python side masks these slots for v3 networks (<c>V5_SLOTS</c>), so a
    /// v3 checkpoint sees the same input as before whether the flag is on or off.</summary>
    public static bool RelicCounters = false;

    /// <summary>Observation version this process encodes: 4 = pre-v5; 5 = relic counters (new, masked slots, the live
    /// network is bit-exact); 6 = also the corrected intent damage (an existing slot, so a network must be retrained).</summary>
    public static int Version => ModifiedIntentDamage ? 6 : RelicCounters ? 5 : 4;

    public static List<ObsToken> Encode(CombatState state, int maxTokens = 96)
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
                    if (ModifiedIntentDamage && atk.DamageCalc is { } raw)
                        dmg = Math.Max(0, (int)Hook.ModifyDamage(state, state.Players[0].Creature, en, raw(),
                            ValueProp.Move, null, null, out _));
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
        {
            int[] f = RelicCounters && RelicCounterValues.TryGet(r, out int counter) ? F(counter, 1) : F();
            t.Add(new(TokenKind.Relic, r.Id.ToString(), "", f));
        }
        if (t.Count > maxTokens) t.RemoveRange(maxTokens, t.Count - maxTokens);
        while (t.Count < maxTokens) t.Add(new(TokenKind.Pad, "", "", F()));
        return t;
    }

    private static void AddPile(List<ObsToken> t, TokenKind kind, IReadOnlyList<CardModel> cards, Func<CardModel, int>? topPos)
    {
        if (SplitPilesByValue) { AddPileSplit(t, kind, cards, topPos); return; }
        // multiset: one token per (card, upgrade, enchantment[, known position]); count in F[0]; v4: the group's
        // largest damage / block (equal for all copies except cards that change one copy only, e.g. Genetic Algorithm)
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

    private static void AddPileSplit(List<ObsToken> t, TokenKind kind, IReadOnlyList<CardModel> cards, Func<CardModel, int>? topPos)
    {
        // multiset: one token per (card, upgrade, enchantment, damage, block[, known position]); count in F[0]
        var groups = cards.GroupBy(c =>
        {
            var e = c.Enchantments.FirstOrDefault();
            return (Id: c.Id.ToString(), Up: c.CurrentUpgradeLevel, E: e?.Id.ToString() ?? "", Ea: (int)(e?.Magnitude ?? 0),
                Pos: topPos?.Invoke(c) ?? -1, Dmg: CardValues.Damage(c), Blk: CardValues.Block(c));
        }).OrderBy(g => g.Key.Pos < 0 ? int.MaxValue : g.Key.Pos).ThenBy(g => g.Key.Id, StringComparer.Ordinal)
          .ThenBy(g => g.Key.Up).ThenBy(g => g.Key.E, StringComparer.Ordinal).ThenBy(g => g.Key.Dmg).ThenBy(g => g.Key.Blk);
        foreach (var g in groups)
            t.Add(new(kind, g.Key.Id, g.Key.E, kind == TokenKind.Draw
                ? F(g.Count(), g.Key.Up, g.Key.Ea, g.Key.Pos, g.Key.Dmg, g.Key.Blk)
                : F(g.Count(), g.Key.Up, g.Key.Ea, g.Key.Dmg, g.Key.Blk)));
    }

    private static int[] F(params int[] xs)
    {
        var f = new int[FeatureCount];
        Array.Copy(xs, f, Math.Min(xs.Length, FeatureCount));
        return f;
    }
}
