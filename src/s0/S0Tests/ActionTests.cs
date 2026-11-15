using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;

/// <summary>Real-run policy over <see cref="ActionSpace"/>: uses a potion with p=0.15 when one is legal, ends the turn
/// with p=0.15, else a random card play. Card prompts: random option.</summary>
class ActionRunPolicy(Random rng) : IRunDecisionSource
{
    public Func<CombatState, Task>? OnDecision;
    public long PotionDecisions;
    public Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options) => Task.FromResult(options[rng.Next(options.Count)]);

    public Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request)
    {
        var options = CardSelectionOptions.Enumerate(request);
        return Task.FromResult(options[rng.Next(options.Count)].Apply(request));
    }

    public async Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
    {
        if (OnDecision is not null) await OnDecision(state);
        var legal = ActionSpace.Base(state, includePotions: true);
        var potions = legal.Where(a => a.Kind == ActionKind.UsePotion).ToList();
        var cards = legal.Where(a => a.Kind == ActionKind.PlayCard).ToList();
        SearchAction a;
        if (potions.Count > 0 && rng.NextDouble() < 0.15) a = potions[rng.Next(potions.Count)];
        else if (cards.Count == 0 || rng.NextDouble() < 0.15) a = SearchAction.End;
        else a = cards[rng.Next(cards.Count)];
        Player p = state.Players[0];
        switch (a.Kind)
        {
            case ActionKind.EndTurn: return new CombatDecision.EndTurn();
            case ActionKind.UsePotion:
                PotionDecisions++;
                return new CombatDecision.UsePotion(p.PotionSlots[a.SourceIndex]!,
                    a.TargetId is { } id ? ActionSpace.PotionTargetCandidates(state).First(c => c?.CombatId == id) : null);
            default:
                var card = p.PlayerCombatState!.Hand.Cards[a.SourceIndex];
                return new CombatDecision.PlayCard(card, a.TargetId is { } t ? state.Allies.Concat(state.Enemies).First(c => c.CombatId == t) : null);
        }
    }
}

static class ActionTests
{
    public static async Task<int> Run(int runs, int maxStates, int k, int d, string prefix, int hp)
    {
        long states = 0, enumerated = 0, withSelections = 0, potionActions = 0, maxActions = 0;
        long execErr = 0, execNotApplicable = 0, keyMisses = 0;
        long oracleCardAccepted = 0, oracleCardMissing = 0, oracleCardLax = 0;
        long oraclePotionLegal = 0, oraclePotionMissing = 0, enumPotionNotLegal = 0;
        long rollouts = 0, rolloutErr = 0, rolloutPotionUses = 0, rolloutSteps = 0;
        var errors = new Dictionary<string, int>();
        var rng = new Random(4242);
        var policy = new ActionRunPolicy(new Random(1));
        policy.OnDecision = async state =>
        {
            Player rp = state.Players[0];
            bool check = states < maxStates;
            if (check)
            {
                states++;
                var expanded = await ActionSpace.LegalAsync(state);
                enumerated += expanded.Count; maxActions = Math.Max(maxActions, expanded.Count);
                var baseSet = new HashSet<string>(ActionSpace.Base(state).Select(a => a.Key));
                // (i) every expanded action executes on a fresh exact clone
                foreach (var a in expanded)
                {
                    if (a.Selections.Count > 0) withSelections++;
                    if (a.Kind == ActionKind.UsePotion) potionActions++;
                    var c = state.Clone();
                    var src = SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(3)));
                    try { if (!await ActionSpace.ApplyAsync(c, a)) execNotApplicable++; }
                    catch (Exception e) { execErr++; Util.Count(errors, "exec " + a.Kind + ": " + Util.Key(e)); }
                    keyMisses += src.KeyMisses;
                }
                // (ii) card oracle: every (hand card, target in {none} ∪ all creatures) the engine plays must be enumerated
                var hand = rp.PlayerCombatState!.Hand.Cards;
                var creatures = ActionSpace.PotionTargetCandidates(state).ToList();
                for (int i = 0; i < hand.Count; i++)
                    for (int ti = 0; ti < creatures.Count; ti++)
                    {
                        var c = state.Clone(out var map);
                        SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(3)));
                        var card = c.Players[0].PlayerCombatState!.Hand.Cards[i];
                        Creature? t = creatures[ti] is { } ct ? map.Creature(ct) : null;
                        CardPlay? played;
                        try { played = await c.Engine!.PlayCardWithResultAsync(c.Players[0], card, t); }
                        catch { played = null; }
                        if (played is null) continue;
                        oracleCardAccepted++;
                        bool targeted = card.TargetType is Sts2Sim.Core.Entities.Cards.TargetType.AnyEnemy
                            or Sts2Sim.Core.Entities.Cards.TargetType.AnyAlly or Sts2Sim.Core.Entities.Cards.TargetType.AnyPlayer;
                        string key = new SearchAction(ActionKind.PlayCard, CardSelectionOptions.EquivalenceKey(hand[i]),
                            targeted ? creatures[ti]?.CombatId : null, Array.Empty<string[]>()).Key;
                        if (baseSet.Contains(key)) continue;
                        bool driverLegal = targeted && creatures[ti] is { } tt && CombatTargetCandidates.IsLegalCardTarget(state, rp, hand[i], tt);
                        if (driverLegal) { oracleCardMissing++; Util.Count(errors, "missing card action " + key); }
                        else { oracleCardLax++; Util.Count(errors, $"engine-lax (driver rejects): {hand[i].Id.Entry} {hand[i].TargetType} -> {creatures[ti]?.Monster?.Id.Entry ?? creatures[ti]?.Player?.Character.Id.Entry ?? "none"} alive={creatures[ti]?.IsAlive}"); }
                    }
                // (iii) potion oracle: CanUseManually over all targets equals the enumerated potion keys
                var legalPot = new HashSet<string>();
                foreach (var pot in rp.PotionSlots)
                    if (pot is not null)
                        foreach (var t in creatures)
                            if (PotionCmd.CanUseManually(state, rp, pot, t))
                                legalPot.Add(new SearchAction(ActionKind.UsePotion, pot.Id.ToString(), t?.CombatId, Array.Empty<string[]>()).Key);
                var enumPot = baseSet.Where(x => x.StartsWith("P:")).ToHashSet();
                oraclePotionLegal += legalPot.Count;
                oraclePotionMissing += legalPot.Count(x => !enumPot.Contains(x));
                enumPotionNotLegal += enumPot.Count(x => !legalPot.Contains(x));
            }
            // (iv) random rollouts over the full action space (potions included, prompts answered at random)
            for (int r = 0; r < k; r++)
            {
                var c = state.Clone();
                var ar = new Random(rng.Next());
                SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(ar.Next())));
                rollouts++;
                try
                {
                    for (int step = 0; step < d && !CombatActions.IsOver(c); step++)
                    {
                        var legal = ActionSpace.Base(c);
                        var pots = legal.Where(x => x.Kind == ActionKind.UsePotion).ToList();
                        SearchAction a = pots.Count > 0 && ar.NextDouble() < 0.3 ? pots[ar.Next(pots.Count)]
                            : legal.Count == 1 || ar.NextDouble() < 0.25 ? SearchAction.End : legal[ar.Next(legal.Count - 1)];
                        if (a.Kind == ActionKind.UsePotion) rolloutPotionUses++;
                        if (!await ActionSpace.ApplyAsync(c, a)) Util.Count(errors, "rollout action not applicable " + a.Kind);
                        rolloutSteps++;
                    }
                }
                catch (Exception e) { rolloutErr++; Util.Count(errors, "rollout: " + Util.Key(e)); }
            }
        };
        int failed = 0;
        for (int i = 0; i < runs; i++)
        {
            try { await Util.PlayRun($"{prefix}{i:D7}", hp, policy); }
            catch (Exception e) { failed++; Console.Error.WriteLine($"run {i} failed: {Util.Key(e)}"); }
        }
        Console.WriteLine($"runs {runs} (failed {failed}); real potion decisions {policy.PotionDecisions}");
        Console.WriteLine($"enumeration on {states} states: {enumerated} actions (max {maxActions}), {withSelections} with prompt answers, {potionActions} potion actions");
        Console.WriteLine($"  executing every enumerated action on a clone: errors {execErr}, not applicable {execNotApplicable}, key-script misses {keyMisses}");
        Console.WriteLine($"  card oracle (all hand cards x {{none}} ∪ all creatures): engine played {oracleCardAccepted}; not enumerated but driver-legal {oracleCardMissing}; " +
            $"engine-only (target the driver rejects) {oracleCardLax}");
        Console.WriteLine($"  potion oracle (CanUseManually): legal {oraclePotionLegal}, not enumerated {oraclePotionMissing}, enumerated but not legal {enumPotionNotLegal}");
        Console.WriteLine($"rollouts {rollouts} x up to {d} actions (steps {rolloutSteps}, potion uses {rolloutPotionUses}): errors {rolloutErr}");
        foreach (var kv in errors.OrderByDescending(kv => kv.Value).Take(12)) Console.WriteLine($"    x{kv.Value}: {kv.Key}");
        return 0;
    }
}
