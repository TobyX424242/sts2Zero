using System.Text;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.MonsterMoves.Intents;

namespace Sts2Sim.S0;

/// <summary>A combat action addressed by position, so the same action can be applied to any clone of a state.</summary>
public readonly record struct CombatAction(bool EndTurn, int HandIndex, int TargetIndex)
{
    public static CombatAction End => new(true, -1, -1);
    public override string ToString() => EndTurn ? "end" : $"play{HandIndex}->{TargetIndex}";
}

public static class CombatActions
{
    /// <summary>Card plays (hand index, target index into <see cref="CombatTargetCandidates.ForCard"/>; -1 = none)
    /// plus end turn. Potions are not enumerated.</summary>
    public static List<CombatAction> Legal(CombatState state, Player player)
    {
        var list = new List<CombatAction>();
        IReadOnlyList<CardModel> hand = player.PlayerCombatState!.Hand.Cards;
        for (int i = 0; i < hand.Count; i++)
        {
            CardModel card = hand[i];
            if (!card.CanPlay(out _)) continue;
            if (card.TargetType is not (TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer))
            {
                list.Add(new CombatAction(false, i, -1));
                continue;
            }
            int n = CombatTargetCandidates.ForCard(state, player, card.TargetType).Count();
            for (int t = 0; t < n; t++) list.Add(new CombatAction(false, i, t));
        }
        list.Add(CombatAction.End);
        return list;
    }

    public static async Task ApplyAsync(CombatState state, CombatAction action)
    {
        CombatEngine engine = state.Engine!;
        Player player = state.Players[0];
        if (action.EndTurn)
        {
            await engine.EndPlayerTurnAsync();
            return;
        }
        CardModel card = player.PlayerCombatState!.Hand.Cards[action.HandIndex];
        Creature? target = action.TargetIndex < 0 ? null
            : CombatTargetCandidates.ForCard(state, player, card.TargetType).ElementAt(action.TargetIndex);
        await engine.PlayCardAsync(player, card, target);
        engine.CheckWinCondition();
    }

    public static bool IsOver(CombatState state) => state.Engine is not { IsInProgress: true };
}

/// <summary>
/// Text fingerprint of what the player can observe: hand (in order), discard and exhaust piles (in order), the
/// draw pile as a sorted multiset plus the known top/bottom runs, energy, stars, orbs, potions, relics, and per
/// creature HP, max HP, block, powers and the shown intent. Two states with the same fingerprint are
/// indistinguishable to the player as far as these fields go.
/// <para>With <see cref="Extended"/> (default, 2026-11-15) every card also carries its current damage / block
/// (<see cref="CardValues"/>: Claw, Wither, Genetic Algorithm, ... change during a fight) and every power its hidden
/// counter (<see cref="PowerCounters"/>: e.g. WitheringPresence's cards left before the next Wither); without it two
/// states differing only there were merged as one search node. Extended = false gives the old fingerprint exactly.</para>
/// </summary>
public static class ObservableState
{
    /// <summary>Card values and hidden power counters in the fingerprint (see the class comment). Set once at
    /// start-up, before any search; false = the fingerprint (and hash) of the old build.</summary>
    public static bool Extended = true;

    /// <summary>64-bit hash of <see cref="Fingerprint"/>: the transposition key of a search node.</summary>
    public static ulong Hash(CombatState state) => ActionSpace.Fnv(Fingerprint(state));

    public static string Fingerprint(CombatState state)
    {
        var sb = new StringBuilder();
        sb.Append("round=").Append(state.RoundNumber).Append(" side=").Append(state.CurrentSide)
          .Append(" over=").Append(CombatActions.IsOver(state)).Append('\n');
        foreach (Player p in state.Players)
        {
            PlayerCombatState pcs = p.PlayerCombatState!;
            sb.Append("energy=").Append(pcs.Energy).Append(" stars=").Append(pcs.Stars).Append('\n');
            sb.Append("hand=").Append(Cards(pcs.Hand.Cards)).Append('\n');
            sb.Append("discard=").Append(Cards(pcs.DiscardPile.Cards)).Append('\n');
            sb.Append("exhaust=").Append(Cards(pcs.ExhaustPile.Cards)).Append('\n');
            CardPile draw = pcs.DrawPile;
            sb.Append("draw=").Append(string.Join(",", draw.Cards.Select(CardFp).Order(StringComparer.Ordinal))).Append('\n');
            sb.Append("draw_top=").Append(Cards(draw.Cards.Take(draw.KnownTopCount)))
              .Append(" draw_bottom=").Append(Cards(draw.Cards.Skip(draw.Cards.Count - draw.KnownBottomCount))).Append('\n');
            sb.Append("orbs=").Append(string.Join(",", pcs.OrbQueue.Orbs.Select(o => o.Id.ToString())))
              .Append(" cap=").Append(pcs.OrbQueue.Capacity).Append('\n');
            sb.Append("potions=").Append(string.Join(",", p.PotionSlots.Select(x => x?.Id.ToString() ?? "-"))).Append('\n');
            sb.Append("relics=").Append(string.Join(",", p.Relics.Select(r =>
                CombatObservation.RelicCounters && RelicCounterValues.TryGet(r, out int counter) ? $"{r.Id}#{counter}" : r.Id.ToString()))).Append('\n');
        }
        foreach (Creature c in state.Allies.Concat(state.Enemies))
            sb.Append(Creature(state, c)).Append('\n');
        return sb.ToString();
    }

    public static string Card(CardModel c) =>
        $"{c.Id.Entry}+{c.CurrentUpgradeLevel}${c.EnergyCost}" +
        (c.Enchantments.Count > 0 ? "~" + string.Join("/", c.Enchantments.Select(e => $"{e.Id.Entry}:{e.Magnitude}")) : "") +
        (c.Affliction is { } a ? "!" + a.Id.Entry : "");

    /// <summary>Card identity without in-combat cost (which some effects re-roll on draw, e.g. Confused).</summary>
    public static string CardIdentity(CardModel c) =>
        $"{c.Id.Entry}+{c.CurrentUpgradeLevel}" +
        (c.Enchantments.Count > 0 ? "~" + string.Join("/", c.Enchantments.Select(e => $"{e.Id.Entry}:{e.Magnitude}")) : "");

    private static string Cards(IEnumerable<CardModel> cards) => string.Join(",", cards.Select(CardFp));

    /// <summary><see cref="Card"/>, plus "=damage/block" with <see cref="Extended"/>.</summary>
    private static string CardFp(CardModel c) =>
        Extended ? $"{Card(c)}={CardValues.Damage(c)}/{CardValues.Block(c)}" : Card(c);

    private static string Power(PowerModel pw) =>
        Extended && PowerCounters.TryGet(pw, out int k) ? $"{pw.Id.Entry}:{pw.Amount}^{k}" : $"{pw.Id.Entry}:{pw.Amount}";

    private static string Creature(CombatState state, Creature c)
    {
        var sb = new StringBuilder();
        sb.Append(c.Monster?.Id.Entry ?? "player").Append('#').Append(c.CombatId)
          .Append(" hp=").Append(c.CurrentHp).Append('/').Append(c.MaxHp).Append(" block=").Append(c.Block)
          .Append(" powers=").Append(string.Join(",", c.Powers.Select(Power)));
        if (c.Monster?.NextMove is { } move)
        {
            sb.Append(" intent=").Append(move.Id).Append(':');
            foreach (AbstractIntent intent in move.Intents)
            {
                sb.Append(intent.IntentType);
                if (intent is AttackIntent attack && !c.IsDead)
                    sb.Append('=').Append(attack.GetTotalDamage(state.Allies, c));
                sb.Append(';');
            }
        }
        return sb.ToString();
    }
}
