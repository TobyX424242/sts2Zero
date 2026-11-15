using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;

namespace Sts2Sim.S0;

public enum ActionKind : byte { EndTurn = 0, PlayCard = 1, UsePotion = 2 }

/// <summary>
/// A combat action addressed by identity, not by position, so it applies to any world that shares the observable
/// state (exact clones, resampled clones, the real fight):
/// <list type="bullet">
/// <item><see cref="Source"/>: the card's equivalence key (id, upgrade, cost, enchantment, affliction) or the potion id;</item>
/// <item><see cref="TargetId"/>: the target's <c>CombatId</c> (null = no target);</item>
/// <item><see cref="Selections"/>: the answers to the card prompts the action opens, in order, each as a sorted
/// multiset of candidate keys (empty = prompts, if any, are left to the clone's selection policy). End turn carries
/// the answer to the first prompt of the end turn (enemy turn, next turn start) when the root expanded it.</item>
/// </list>
/// <see cref="Key"/> is the canonical string; <see cref="Hash"/> its FNV-1a 64 hash. Two actions with the same key
/// are the same move (two identical Strikes are one action), which is what a search node keys its children by.
/// To merge transpositions inside a turn, key nodes by <see cref="ObservableState.Hash"/> of the resulting state.
/// </summary>
public sealed record SearchAction(ActionKind Kind, string Source, uint? TargetId, IReadOnlyList<string[]> Selections)
{
    public static readonly SearchAction End = new(ActionKind.EndTurn, "", null, Array.Empty<string[]>());

    public string Key { get; } = MakeKey(Kind, Source, TargetId, Selections);
    public ulong Hash => ActionSpace.Fnv(Key);

    /// <summary>Hand index / potion slot in the state the action was enumerated from (a hint for the encoder and
    /// for the network's pointer head; <see cref="ActionSpace.ApplyAsync"/> resolves by identity).</summary>
    public int SourceIndex { get; init; } = -1;

    // (not `with`: Key is computed in the constructor)
    public SearchAction WithSelection(string[] keys) =>
        new(Kind, Source, TargetId, Selections.Append(keys).ToArray()) { SourceIndex = SourceIndex };

    public SearchAction WithoutSelections() => new(Kind, Source, TargetId, Array.Empty<string[]>()) { SourceIndex = SourceIndex };

    private static string MakeKey(ActionKind kind, string source, uint? target, IReadOnlyList<string[]> sel)
    {
        // End turn: "E", plus the answers of an end-turn prompt variant (root only, see ActionSpace.LegalAsync);
        // a plain end turn keeps the key "E".
        string k = kind == ActionKind.EndTurn ? "E"
            : (kind == ActionKind.PlayCard ? "C:" : "P:") + source + ">" + (target is { } t ? "t" + t : "-");
        foreach (string[] s in sel) k += "|S:" + string.Join("&", s);
        return k;
    }

    public override string ToString() => Key;
    public bool Equals(SearchAction? other) => other is not null && other.Key == Key;
    public override int GetHashCode() => Key.GetHashCode();
}

/// <summary>Legal actions of a combat state: card plays with targets, potion uses with targets, end turn, and
/// (optionally) the options of the card prompts these open.</summary>
public static class ActionSpace
{
    /// <summary>
    /// Card plays: every playable hand card (deduplicated by key) with each legal target (the rule of
    /// <c>RunDriver.ValidatePlayCard</c>: AnyEnemy -> hittable enemies, AnyAlly/AnyPlayer -> legal living players,
    /// otherwise no target). Potions: every potion slot (deduplicated by potion id) with every target among
    /// {none} ∪ all creatures for which <see cref="PotionCmd.CanUseManually"/> holds. End turn is always last.
    /// With <paramref name="expandSelections"/>, every card/potion action is probed on an exact clone in
    /// <see cref="SelectionMode.Suspend"/> mode; an action that opens a prompt is replaced by one action per option
    /// (nested prompts up to <paramref name="maxPromptDepth"/>; deeper prompts are left to the selection policy).
    /// <para>With <paramref name="endTurnPrompts"/> (and <paramref name="expandSelections"/>), end turn is probed too:
    /// once, on a <see cref="ResamplingClone.Resampled"/> clone (draw order, enemy and card rolls drawn afresh) seeded
    /// by <see cref="EndTurnProbeSeed"/>, a function of the observable state only. What happens during the enemy turn
    /// and the next turn start is hidden information, so an exact clone would leak it. If that end turn opens a prompt
    /// (e.g. Knowledge Demon's Curse of Knowledge, start-of-turn powers such as Tools of the Trade, end-of-turn retain
    /// choices), end turn is replaced by one end-turn action per answer of that first prompt (at most
    /// <paramref name="maxEndTurnOptions"/>; later prompts of the same end turn are left to the selection policy,
    /// so one probe suffices); otherwise the plain end turn is kept. The end-turn actions stay last in the list. The
    /// answers are world-independent keys: in a world where the prompt does not come or the keys do not match the
    /// candidates (e.g. a hand-dependent prompt), it falls back to the selection policy (KeyMisses).
    /// Off (the default): the list is exactly the one without this option.</para>
    /// </summary>
    public static async Task<List<SearchAction>> LegalAsync(CombatState state, bool includePotions = true,
        bool expandSelections = true, int maxPromptDepth = 2, int maxOptionsPerPrompt = 64,
        bool endTurnPrompts = false, int maxEndTurnOptions = 16)
    {
        var baseActions = Base(state, includePotions);
        if (!expandSelections) return baseActions;
        var result = new List<SearchAction>(baseActions.Count + 4);
        foreach (SearchAction a in baseActions)
        {
            if (a.Kind == ActionKind.EndTurn)
            {
                if (endTurnPrompts) await ExpandEndTurnAsync(state, Math.Min(maxOptionsPerPrompt, maxEndTurnOptions), result);
                else result.Add(a);
                continue;
            }
            await ExpandAsync(state, a, maxPromptDepth, maxOptionsPerPrompt, result);
        }
        return result;
    }

    /// <summary>End-turn probes run (<see cref="LegalAsync"/> with endTurnPrompts), probes that opened a prompt,
    /// probes that failed (exception other than the prompt; the plain end turn is kept). Process-wide counters.</summary>
    public static long EndTurnProbes, EndTurnProbesWithPrompt, EndTurnProbeErrors;

    /// <summary>Seed of the end-turn probe world: a hash of the observable state (the same information set gives the
    /// same probe world, hence the same action list; nothing hidden enters the seed).</summary>
    public static ulong EndTurnProbeSeed(CombatState state) => ResamplingClone.Mix(ObservableState.Hash(state), 0xE7D0E7D0UL);

    private static async Task ExpandEndTurnAsync(CombatState state, int maxOptions, List<SearchAction> into)
    {
        EndTurnProbes++;
        try
        {
            // never an exact clone: the probe must not see the true draw order or enemy rolls
            CombatState probe = ResamplingClone.Resampled(state, EndTurnProbeSeed(state));
            var src = SearchCardSelectionSource.InstallOn(probe, SelectionMode.Suspend);
            src.MaxOptions = Math.Max(1, maxOptions);
            await ApplyCoreAsync(probe, SearchAction.End);
            into.Add(SearchAction.End);
        }
        catch (SelectionPendingException e)
        {
            EndTurnProbesWithPrompt++;
            foreach (SelectionOption o in e.Options)
                into.Add(SearchAction.End.WithSelection(CardSelectionOptions.Keys(e.Request, o)));
        }
        catch (Exception)
        {
            // a probe that fails (unsupported RNG mode, a simulator error inside the enemy turn) must not fail the
            // decision: keep the plain end turn, as without the option
            EndTurnProbeErrors++;
            into.Add(SearchAction.End);
        }
    }

    public static List<SearchAction> Base(CombatState state, bool includePotions = true)
    {
        var list = new List<SearchAction>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        Player p = state.Players[0];
        if (CombatActions.IsOver(state)) return list;
        IReadOnlyList<CardModel> hand = p.PlayerCombatState!.Hand.Cards;
        for (int i = 0; i < hand.Count; i++)
        {
            CardModel card = hand[i];
            if (!card.CanPlay(out _)) continue;
            string key = CardSelectionOptions.EquivalenceKey(card);
            if (card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer)
            {
                foreach (Creature t in CombatTargetCandidates.ForCard(state, p, card.TargetType))
                    Add(new SearchAction(ActionKind.PlayCard, key, t.CombatId, Array.Empty<string[]>()) { SourceIndex = i });
            }
            else Add(new SearchAction(ActionKind.PlayCard, key, null, Array.Empty<string[]>()) { SourceIndex = i });
        }
        if (includePotions)
        {
            for (int slot = 0; slot < p.PotionSlots.Count; slot++)
            {
                if (p.PotionSlots[slot] is not { } potion) continue;
                foreach (Creature? t in PotionTargetCandidates(state))
                    if (PotionCmd.CanUseManually(state, p, potion, t))
                        Add(new SearchAction(ActionKind.UsePotion, potion.Id.ToString(), t?.CombatId, Array.Empty<string[]>()) { SourceIndex = slot });
            }
        }
        list.Add(SearchAction.End);
        return list;

        void Add(SearchAction a) { if (seen.Add(a.Key)) list.Add(a); }
    }

    public static IEnumerable<Creature?> PotionTargetCandidates(CombatState state)
    {
        yield return null;
        foreach (Creature c in state.Allies) yield return c;
        foreach (Creature c in state.Enemies) yield return c;
    }

    private static async Task ExpandAsync(CombatState state, SearchAction a, int depth, int maxOptions, List<SearchAction> into)
    {
        CombatState probe = state.Clone();
        var src = SearchCardSelectionSource.InstallOn(probe, SelectionMode.Suspend);
        src.MaxOptions = maxOptions;
        foreach (string[] k in a.Selections) src.KeyScript.Enqueue(k);
        try
        {
            await ApplyCoreAsync(probe, a);
            into.Add(a);
        }
        catch (SelectionPendingException e)
        {
            foreach (SelectionOption o in e.Options)
            {
                var child = a.WithSelection(CardSelectionOptions.Keys(e.Request, o));
                if (a.Selections.Count + 1 >= depth) into.Add(child);
                else await ExpandAsync(state, child, depth, maxOptions, into);
            }
        }
    }

    /// <summary>
    /// Applies <paramref name="action"/> to <paramref name="state"/> by identity. The action's selection answers are
    /// queued on the state's <see cref="SearchCardSelectionSource"/> (installed in Policy mode with
    /// <paramref name="fallback"/> if absent); prompts without an answer go to that source's mode.
    /// Returns false if the action is not applicable in this world (card/potion/target absent or not legal).
    /// </summary>
    public static async Task<bool> ApplyAsync(CombatState state, SearchAction action,
        Func<CardSelectionRequest, IReadOnlyList<SelectionOption>, SelectionOption>? fallback = null)
    {
        if (state.CardSelectionSource is not SearchCardSelectionSource src)
            src = SearchCardSelectionSource.InstallOn(state, SelectionMode.Policy,
                fallback ?? SearchCardSelectionSource.RandomPolicy(new System.Random(1)));
        src.KeyScript.Clear();
        foreach (string[] k in action.Selections) src.KeyScript.Enqueue(k);
        bool ok = await ApplyCoreAsync(state, action);
        src.KeyScript.Clear();
        return ok;
    }

    private static async Task<bool> ApplyCoreAsync(CombatState state, SearchAction action)
    {
        CombatEngine engine = state.Engine!;
        Player p = state.Players[0];
        if (!engine.IsInProgress) return false;
        switch (action.Kind)
        {
            case ActionKind.EndTurn:
                await engine.EndPlayerTurnAsync();
                return true;
            case ActionKind.PlayCard:
            {
                CardModel? card = FindCard(p, action);
                if (card is null) return false;
                Creature? target = null;
                if (card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer)
                {
                    target = CombatTargetCandidates.ForCard(state, p, card.TargetType).FirstOrDefault(c => c.CombatId == action.TargetId);
                    if (target is null) return false;
                }
                await engine.PlayCardAsync(p, card, target);
                engine.CheckWinCondition();
                return true;
            }
            default:
            {
                PotionModel? potion = action.SourceIndex >= 0 && action.SourceIndex < p.PotionSlots.Count &&
                    p.PotionSlots[action.SourceIndex]?.Id.ToString() == action.Source
                    ? p.PotionSlots[action.SourceIndex]
                    : p.PotionSlots.FirstOrDefault(x => x?.Id.ToString() == action.Source);
                if (potion is null) return false;
                Creature? target = action.TargetId is { } id ? PotionTargetCandidates(state).FirstOrDefault(c => c?.CombatId == id) : null;
                if (action.TargetId is not null && target is null) return false;
                if (!PotionCmd.CanUseManually(state, p, potion, target)) return false;
                await PotionCmd.Use(potion, p, target);
                engine.CheckWinCondition();
                return true;
            }
        }
    }

    private static CardModel? FindCard(Player p, SearchAction a)
    {
        var hand = p.PlayerCombatState!.Hand.Cards;
        if (a.SourceIndex >= 0 && a.SourceIndex < hand.Count && hand[a.SourceIndex].CanPlay(out _) &&
            CardSelectionOptions.EquivalenceKey(hand[a.SourceIndex]) == a.Source)
            return hand[a.SourceIndex];
        foreach (CardModel c in hand)
            if (c.CanPlay(out _) && CardSelectionOptions.EquivalenceKey(c) == a.Source) return c;
        return null;
    }

    public static ulong Fnv(string s)
    {
        ulong h = 14695981039346656037UL;
        foreach (char ch in s) { h ^= ch; h *= 1099511628211UL; }
        return h;
    }
}
