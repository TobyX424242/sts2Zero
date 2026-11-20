using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Entities.Rngs;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Random;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;

/// <summary>
/// ic113 (2026-11-18): the resampling-clone leak test on the v0.113.1 simulator (child RNG per shuffle, the
/// NonCombatGeneration and Shops player streams), extending `resample`.
/// <code>
/// S0Tests leak113 [states=300] [prefix=L113] [character=IRONCLAD] [clones=50] [perms=20] [outer_runs=6]
/// </code>
/// States: combat decisions of random-policy A10 runs (HP 3000) of the character.
/// (1) diversity: from one root with >= 10 unknown draw-pile cards, `clones` resampled clones; distinct top-5 orders
///     (card identities) of their draw piles (target >= 45 / 50); exact clones for contrast (always 1 order).
/// (2) child shuffle: the same after a forced reshuffle (CardPileCmd.Shuffle: discard + draw shuffled with the child of
///     the Shuffle stream) in each clone, plus how often a resampled clone's reshuffled top 5 equals the exact clone's
///     (the true future) against the rate between two resampled clones.
/// (3) the true order is not recoverable: `perms` copies of the root whose hidden draw order is permuted (and whose
///     Shuffle / NonCombatGeneration / Shops / monster streams hold other generator states), `clones` resampled clones of
///     each (independent seeds): the clone's top card identity is distributed alike over the copies (chi-square test of
///     homogeneity per state; the share of states with p &lt; 0.05 must be near 5%), and the post-reshuffle draw order of a
///     clone does not depend on the root's hidden RNG state (identical for the same seed).
/// (4) the outer run: `outer_runs` runs played twice by the same deterministic policy, once plain and once with 5
///     resampled clones per combat decision, each stepped through random actions (end turns, reshuffles, card plays) and
///     its NonCombatGeneration / Shops streams drawn from; at every decision of the real run the serialised run and player
///     RNG sets, the map options, rest / event options and the deck must be identical between the two plays.
/// </summary>
static class Leak113
{
    public static async Task<int> Run(string[] args)
    {
        var kv = args.Skip(1).Select(a => a.Split('=', 2)).ToDictionary(p => p[0], p => p.Length > 1 ? p[1] : "");
        int I(string k, int d) => kv.TryGetValue(k, out var v) ? int.Parse(v) : d;
        string prefix = kv.GetValueOrDefault("prefix", "L113");
        string character = "CHARACTER." + kv.GetValueOrDefault("character", "IRONCLAD").ToUpperInvariant();
        int maxStates = I("states", 300), clones = I("clones", 50), perms = I("perms", 20), outerRuns = I("outer_runs", 6);
        var charModel = ModelDb.GetById<CharacterModel>(ModelId.Deserialize(character));

        // ---- collect states
        var states = new List<CombatState>();
        var rng = new Random(113);
        var policy = new RandomRunPolicy(rng);
        policy.OnDecision = (state, _) => { if (states.Count < maxStates && rng.NextDouble() < 0.3) states.Add(state.Clone()); return Task.CompletedTask; };
        for (int i = 0; i < 400 && states.Count < maxStates; i++)
        {
            try { await PlayRun($"{prefix}{i:D6}", charModel, policy); }
            catch (Exception e) { Console.Error.WriteLine($"run {i}: {Util.Key(e)}"); }
        }
        Console.WriteLine($"leak113: {states.Count} combat states from {character} A10 random-policy runs (HP 3000)");

        // ---- (1) and (2)
        int d1N = 0, d1Pass = 0; double d1Sum = 0, d1IdSum = 0, d1RefSum = 0; long d1True = 0, d1RefTrue = 0, d1Trials = 0; int d1Min = int.MaxValue; int exactDistinctMax = 0;
        int d2N = 0, d2Pass = 0; double d2Sum = 0; int d2Min = int.MaxValue; long hitTrue = 0, hitRR = 0, pairs = 0;
        var errors = new Dictionary<string, int>();
        for (int si = 0; si < states.Count; si++)
        {
            CombatState s = states[si];
            if (CombatActions.IsOver(s)) continue;
            Player sp = s.Players[0];
            var know = ResamplingClone.DrawPileKnowledge(sp);
            var pcs = sp.PlayerCombatState!;
            try
            {
                if (know.Unknown >= 10 && pcs.DrawPile.Cards.Count >= 10)
                {
                    // instance level: each clone's top 5 as positions of the root's draw pile (CombatCloneMap)
                    var index = new Dictionary<CardModel, int>(ReferenceEqualityComparer.Instance);
                    var tops = new HashSet<string>(); var topsId = new HashSet<string>();
                    string truthId = Top5(s);
                    for (int c = 0; c < clones; c++)
                    {
                        var r = ResamplingClone.Resampled(s, ResamplingClone.Mix(0x1113, (ulong)(si * 1000 + c)), out CombatCloneMap map);
                        var back = new Dictionary<CardModel, int>(ReferenceEqualityComparer.Instance);
                        for (int j = 0; j < pcs.DrawPile.Cards.Count; j++) if (map.TryCard(pcs.DrawPile.Cards[j], out var cc)) back[cc] = j;
                        tops.Add(string.Join(",", r.Players[0].PlayerCombatState!.DrawPile.Cards.Take(5).Select(x => back.GetValueOrDefault(x, -1))));
                        string id5 = Top5(r); topsId.Add(id5); if (id5 == truthId) d1True++;
                    }
                    // the same multiset shuffled by a reference generator: the identity-level diversity to expect
                    var ids = pcs.DrawPile.Cards.Skip(know.Top).Take(know.Unknown).Select(ObservableState.CardIdentity).ToList();
                    var refRng = new Random(si); var refTops = new HashSet<string>();
                    for (int c = 0; c < clones; c++)
                    {
                        var l = ids.ToList();
                        for (int i2 = l.Count - 1; i2 > 0; i2--) { int j2 = refRng.Next(i2 + 1); (l[i2], l[j2]) = (l[j2], l[i2]); }
                        string rt = string.Join("|", pcs.DrawPile.Cards.Take(know.Top).Select(ObservableState.CardIdentity).Concat(l).Take(5));
                        refTops.Add(rt); if (rt == truthId) d1RefTrue++;
                    }
                    d1IdSum += topsId.Count; d1RefSum += refTops.Count; d1Trials += clones;
                    var ex = new HashSet<string>();
                    for (int c = 0; c < 5; c++) ex.Add(Top5(s.Clone()));
                    exactDistinctMax = Math.Max(exactDistinctMax, ex.Count);
                    d1N++; d1Sum += tops.Count; d1Min = Math.Min(d1Min, tops.Count); if (tops.Count >= 45) d1Pass++;
                }
                if (pcs.DrawPile.Cards.Count + pcs.DiscardPile.Cards.Count >= 10 && pcs.DiscardPile.Cards.Count >= 5)
                {
                    var all = pcs.DrawPile.Cards.Concat(pcs.DiscardPile.Cards).ToList();
                    string Inst(CombatState x, CombatCloneMap m)
                    {
                        var back = new Dictionary<CardModel, int>(ReferenceEqualityComparer.Instance);
                        for (int j = 0; j < all.Count; j++) if (m.TryCard(all[j], out var cc)) back[cc] = j;
                        return string.Join(",", x.Players[0].PlayerCombatState!.DrawPile.Cards.Take(5).Select(y => back.GetValueOrDefault(y, -1)));
                    }
                    var exact = s.Clone(out CombatCloneMap em);
                    await CardPileCmd.Shuffle(exact, exact.Players[0]);
                    string truth = Inst(exact, em);
                    var tops = new List<string>();
                    for (int c = 0; c < clones; c++)
                    {
                        var r = ResamplingClone.Resampled(s, ResamplingClone.Mix(0x2113, (ulong)(si * 1000 + c)), out CombatCloneMap rm);
                        await CardPileCmd.Shuffle(r, r.Players[0]);
                        tops.Add(Inst(r, rm));
                    }
                    int distinct = tops.Distinct().Count();
                    d2N++; d2Sum += distinct; d2Min = Math.Min(d2Min, distinct); if (distinct >= 45) d2Pass++;
                    hitTrue += tops.Count(t => t == truth);
                    for (int c = 1; c < tops.Count; c++) { pairs++; if (tops[c] == tops[c - 1]) hitRR++; }
                }
            }
            catch (Exception e) { Util.Count(errors, Util.Key(e)); }
        }
        Console.WriteLine($"(1) diversity: {d1N} roots with >= 10 unknown draw-pile cards; distinct top-5 orders (card instances) of {clones} resampled clones: " +
            $"mean {d1Sum / Math.Max(1, d1N):F1}, min {(d1N > 0 ? d1Min : 0)}, roots with >= 45: {d1Pass}/{d1N}; exact clones: at most {exactDistinctMax} order(s); " +
            $"by card identity (duplicates such as 5 Strikes merge): mean {d1IdSum / Math.Max(1, d1N):F1} vs {d1RefSum / Math.Max(1, d1N):F1} for a reference shuffle of the same cards; " +
            $"clone's top 5 == the true top 5 (identities) in {d1True}/{d1Trials} vs {d1RefTrue}/{d1Trials} for the reference shuffle (chance)");
        Console.WriteLine($"(2) child shuffle: {d2N} roots, distinct top-5 orders (card instances) after a forced reshuffle in {clones} resampled clones: mean {d2Sum / Math.Max(1, d2N):F1}, " +
            $"min {(d2N > 0 ? d2Min : 0)}, roots with >= 45: {d2Pass}/{d2N}; resampled == the true reshuffle {hitTrue}/{d2N * clones} " +
            $"({hitTrue / Math.Max(1.0, d2N * clones):P2}) vs resampled == resampled {hitRR}/{pairs} ({hitRR / Math.Max(1.0, pairs):P2})");

        // ---- (3)
        int t3N = 0, t3Sig = 0, t3RngSame = 0, t3RngN = 0;
        var pvals = new List<double>();
        for (int si = 0; si < states.Count && t3N < 150; si++)
        {
            CombatState s = states[si];
            if (CombatActions.IsOver(s)) continue;
            var know = ResamplingClone.DrawPileKnowledge(s.Players[0]);
            if (know.Unknown < 10) continue;
            try
            {
                var table = new List<Dictionary<string, int>>();
                string? reshuffleRef = null; bool rngSame = true;
                for (int pi = 0; pi < perms; pi++)
                {
                    CombatState root = s.Clone();
                    if (pi > 0)
                    {
                        root.Players[0].PlayerCombatState!.DrawPile.ShuffleUnknownRegionInternal(new Rng((ulong)(7919 * pi + si)));
                        // different hidden RNG states (generator state; counters as in the original)
                        Perturb(root.RunState.Rng.GetRng(RunRngType.Shuffle), pi);
                        Perturb(root.Players[0].PlayerRng.GetRng(PlayerRngType.NonCombatGeneration), pi + 1);
                        Perturb(root.Players[0].PlayerRng.GetRng(PlayerRngType.Shops), pi + 2);
                        foreach (var e in root.Enemies) if (e.Monster is { Rng: { } mr }) Perturb(mr, pi + 3);
                    }
                    var counts = new Dictionary<string, int>();
                    for (int c = 0; c < clones; c++)
                    {
                        // independent seeds per copy: each copy's clones are i.i.d. (the same seed would apply one fixed
                        // permutation to every copy's order, so its samples would not be independent draws)
                        var r = ResamplingClone.Resampled(root, ResamplingClone.Mix(0x3113, (ulong)((si * 100 + pi) * 1000 + c)));
                        string top = ObservableState.CardIdentity(r.Players[0].PlayerCombatState!.DrawPile.Cards[know.Top]);
                        counts[top] = counts.GetValueOrDefault(top) + 1;
                        if (c == 0)
                        {
                            // the reshuffle of all cards (discard + draw) of the clone, by identity multiset order: depends
                            // only on the seed, not on the root's hidden generator states
                            var rr = ResamplingClone.Resampled(root, ResamplingClone.Mix(0x4113, (ulong)si));
                            await CardPileCmd.Shuffle(rr, rr.Players[0]);
                            string after = string.Join(",", rr.Players[0].PlayerCombatState!.DrawPile.Cards.Select(ObservableState.CardIdentity));
                            reshuffleRef ??= after;
                            if (after != reshuffleRef) rngSame = false;
                        }
                    }
                    table.Add(counts);
                }
                t3RngN++; if (rngSame) t3RngSame++;
                double p = ChiSquareP(table);
                if (!double.IsNaN(p)) { t3N++; pvals.Add(p); if (p < 0.05) t3Sig++; }
            }
            catch (Exception e) { Util.Count(errors, "3 " + Util.Key(e)); }
        }
        pvals.Sort();
        Console.WriteLine($"(3) not recoverable: {t3N} roots x {perms} permuted / reseeded copies x {clones} clones: homogeneity of the clones' top card, " +
            $"p < 0.05 in {t3Sig}/{t3N} ({t3Sig / Math.Max(1.0, t3N):P1}; 5% expected under no leak), median p {(pvals.Count > 0 ? pvals[pvals.Count / 2] : double.NaN):F2}; " +
            $"reshuffle after resampling identical over the copies' hidden RNG states in {t3RngSame}/{t3RngN}");
        // per seed, the Fisher-Yates over the unknown region maps the input order through the seed's permutation, so a
        // clone's order depends on the root's order seed by seed while its distribution is uniform for every input; (3)
        // therefore tests the distribution (and the reshuffle stream, which does not depend on the root at all).

        // ---- (4) outer run unchanged
        int same = 0, diff = 0; long cloneSteps = 0; var diffs = new List<string>();
        for (int i = 0; i < outerRuns; i++)
        {
            string seed = $"{prefix}OUT{i:D4}";
            var a = new TracePolicy(new Random(i), null); var b = new TracePolicy(new Random(i), new Random(1000 + i));
            try { await PlayRun(seed, charModel, a, 300); } catch (Exception e) { a.Trace.Add("error " + Util.Key(e)); }
            try { await PlayRun(seed, charModel, b, 300); } catch (Exception e) { b.Trace.Add("error " + Util.Key(e)); }
            cloneSteps += b.CloneSteps;
            int n = Math.Min(a.Trace.Count, b.Trace.Count), k = 0;
            while (k < n && a.Trace[k] == b.Trace[k]) k++;
            if (k == n && a.Trace.Count == b.Trace.Count) same++;
            else { diff++; diffs.Add($"{seed}: first difference at trace entry {k}/{a.Trace.Count}/{b.Trace.Count}: {(k < a.Trace.Count ? a.Trace[k][..Math.Min(200, a.Trace[k].Length)] : "end")} | {(k < b.Trace.Count ? b.Trace[k][..Math.Min(200, b.Trace[k].Length)] : "end")}"); }
            Console.WriteLine($"  outer run {seed}: trace entries {a.Trace.Count} (combat decisions {a.Combat}, map {a.Maps}, rewards {a.Rewards}, shops {a.Shops}, rests {a.Rests}, events {a.Events}); " +
                $"with clones: {b.Trace.Count}, clone steps {b.CloneSteps}");
        }
        Console.WriteLine($"(4) outer run: {same}/{outerRuns} runs identical with and without mid-fight resampling ({cloneSteps} clone steps, every RNG set compared at every decision)");
        foreach (string d in diffs) Console.WriteLine("    " + d);
        foreach (var kv2 in errors.OrderByDescending(x => x.Value).Take(8)) Console.WriteLine($"    error x{kv2.Value}: {kv2.Key}");
        bool ok = d1N > 0 && d1Pass >= 0.95 * d1N && d2N > 0 && d2Pass >= 0.95 * d2N && t3N > 0 && t3Sig <= Math.Max(3, 0.12 * t3N)
            && t3RngSame == t3RngN && same == outerRuns && errors.Count == 0;
        Console.WriteLine(ok ? "LEAK113 PASSED" : "LEAK113 FAILED");
        return ok ? 0 : 1;
    }

    static Task<RunDriver.Result> PlayRun(string seed, CharacterModel character, IRunDecisionSource policy, int hp = 3000)
    {
        var acts = ActDefinition.GetRandomList(seed);
        var runState = new RunState(seed, acts, ascensionLevel: 10);
        var player = Player.CreateForNewRun(character, runState);
        runState.AddPlayer(player);
        if (policy is TracePolicy tp) { tp.Run = runState; tp.P = player; }
        player.Creature.SetMaxHpInternal(hp); player.Creature.SetCurrentHpInternal(hp);
        return new RunDriver(runState, policy).RunAsync(maxFloors: 60);
    }

    static string Top5(CombatState s) =>
        string.Join("|", s.Players[0].PlayerCombatState!.DrawPile.Cards.Take(5).Select(ObservableState.CardIdentity));

    /// <summary>Replaces the generator state with another one (counter kept), like ResamplingClone.Reseed.</summary>
    static void Perturb(Rng r, int k)
    {
        var fresh = new Rng(ResamplingClone.Mix(0xBEEF, (ulong)k)).ToSerializable();
        fresh.counter = r.Counter;
        r.LoadFromSerializable(fresh);
    }

    /// <summary>p-value of the chi-square test of homogeneity of the rows' category counts (NaN when there is a single
    /// category). Upper tail of the chi-square distribution by the regularized gamma function.</summary>
    static double ChiSquareP(List<Dictionary<string, int>> rows)
    {
        var cats = rows.SelectMany(r => r.Keys).Distinct().ToList();
        if (cats.Count < 2) return double.NaN;
        double total = rows.Sum(r => r.Values.Sum());
        var colTot = cats.ToDictionary(c => c, c => (double)rows.Sum(r => r.GetValueOrDefault(c)));
        double chi = 0;
        foreach (var r in rows)
        {
            double rt = r.Values.Sum();
            foreach (string c in cats)
            {
                double e = rt * colTot[c] / total;
                if (e > 0) chi += Math.Pow(r.GetValueOrDefault(c) - e, 2) / e;
            }
        }
        int df = (rows.Count - 1) * (cats.Count - 1);
        return 1 - RegGammaP(df / 2.0, chi / 2.0);
    }

    static double RegGammaP(double a, double x)
    {
        if (x <= 0) return 0;
        double lg = LogGamma(a);
        if (x < a + 1)
        {
            double sum = 1 / a, del = sum, ap = a;
            for (int n = 0; n < 1000; n++) { ap++; del *= x / ap; sum += del; if (Math.Abs(del) < Math.Abs(sum) * 1e-12) break; }
            return sum * Math.Exp(-x + a * Math.Log(x) - lg);
        }
        double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
        for (int i = 1; i < 1000; i++)
        {
            double an = -i * (i - a); b += 2;
            d = an * d + b; if (Math.Abs(d) < 1e-300) d = 1e-300;
            c = b + an / c; if (Math.Abs(c) < 1e-300) c = 1e-300;
            d = 1 / d; double del = d * c; h *= del;
            if (Math.Abs(del - 1) < 1e-12) break;
        }
        return 1 - Math.Exp(-x + a * Math.Log(x) - lg) * h;
    }

    static double LogGamma(double x)
    {
        double[] c = { 76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };
        double y = x, tmp = x + 5.5; tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        foreach (double v in c) ser += v / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    /// <summary>A deterministic random policy (its own Random only) that writes a trace line at every decision: the kind,
    /// floor, the serialised run and player RNG sets, the options and the deck. With <c>cloneRng</c>, every combat
    /// decision first makes 5 resampled clones of the live state and steps each with up to 6 random actions (end turns
    /// included), a forced reshuffle and draws from its NonCombatGeneration and Shops streams.</summary>
    sealed class TracePolicy(Random rng, Random? cloneRng) : IRunDecisionSource
    {
        public readonly List<string> Trace = new();
        public RunState? Run;
        public Player? P;
        public int Combat, Maps, Rewards, Shops, Rests, Events;
        public long CloneSteps;
        private static readonly JsonSerializerOptions Opt = new() { IncludeFields = true };

        private void Note(string kind, RunState? run, Player? p, string extra)
        {
            string rngs = run is null ? "" : JsonSerializer.Serialize(run.Rng.ToSerializable(), Opt);
            string prng = p is null ? "" : JsonSerializer.Serialize(p.PlayerRng.ToSerializable(), Opt);
            string deck = p is null ? "" : string.Join(",", p.Deck.Cards.Select(c => c.Id.Entry + c.CurrentUpgradeLevel));
            Trace.Add($"{kind} floor {run?.TotalFloor} hp {p?.Creature.CurrentHp} gold {p?.Gold} {extra} | rng {rngs} | prng {prng} | deck {deck}");
        }

        public Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options)
        {
            Maps++;
            Note("map", Run, P, string.Join(";", options.Select(o => $"{o.coord.row}/{o.coord.col}/{o.PointType}")));
            return Task.FromResult(options[rng.Next(options.Count)]);
        }

        public Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request)
        {
            var options = CardSelectionOptions.Enumerate(request);
            return Task.FromResult(options[rng.Next(options.Count)].Apply(request));
        }

        public Task<RewardDecision> ChooseRewardActionAsync(Sts2Sim.Core.Rewards.RewardsSet rewards)
        {
            Rewards++;
            Note("reward", Run, P, "");
            return Task.FromResult(RewardDecisionClassifier.ChooseDefault(rewards));
        }

        public Task<ShopDecision> ChooseShopActionAsync(Sts2Sim.Core.Entities.Merchant.MerchantInventory inventory, Player player)
        {
            Shops++;
            Note("shop", Run, P, $"cards {inventory.Cards.Count} relics {inventory.Relics.Count} potions {inventory.Potions.Count}");
            return Task.FromResult<ShopDecision>(new ShopDecision.Leave());
        }

        public Task<RestSiteDecision> ChooseRestSiteActionAsync(Player player, IReadOnlyList<RestSiteDecision> candidates)
        {
            Rests++;
            Note("rest", Run, P, string.Join(";", candidates.Select(c => c.OptionId + (c.IsEnabled ? "" : "(off)"))));
            var en = candidates.Where(c => c.IsEnabled).ToList();
            return Task.FromResult(en.Count > 0 ? en[rng.Next(en.Count)] : RestSiteDecisionPolicy.ChooseDefault(candidates));
        }

        public Task<Sts2Sim.Core.Events.EventOption> ChooseEventOptionAsync(IReadOnlyList<Sts2Sim.Core.Events.EventOption> options)
        {
            Events++;
            Note("event", Run, P, string.Join(";", options.Select(o => o.Key + (o.IsLocked ? "(locked)" : ""))));
            var open = options.Where(o => !o.IsLocked).ToList();
            return Task.FromResult(open.Count > 0 ? open[rng.Next(open.Count)] : EventOptionDecisionPolicy.ChooseDefault(options));
        }

        public async Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
        {
            Combat++;
            var p = state.Players[0];
            Note("combat", Run, P, $"round {state.RoundNumber} hand {string.Join(",", p.PlayerCombatState!.Hand.Cards.Select(c => c.Id.Entry))}");
            if (cloneRng is not null)
            {
                for (int k = 0; k < 5; k++)
                {
                    var c = ResamplingClone.Resampled(state, (ulong)cloneRng.NextInt64());
                    SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(cloneRng.Next())));
                    var cp = c.Players[0];
                    cp.PlayerRng.GetRng(PlayerRngType.NonCombatGeneration).NextInt(100);
                    cp.PlayerRng.GetRng(PlayerRngType.Shops).NextInt(100);
                    try
                    {
                        await CardPileCmd.Shuffle(c, cp);
                        for (int step = 0; step < 6 && !CombatActions.IsOver(c); step++)
                        {
                            var legal = CombatActions.Legal(c, cp);
                            var a = cloneRng.NextDouble() < 0.3 ? CombatAction.End : legal[cloneRng.Next(legal.Count)];
                            await CombatActions.ApplyAsync(c, a);
                            CloneSteps++;
                        }
                    }
                    catch (Exception) { }
                }
            }
            var legalReal = CombatActions.Legal(state, p);
            CombatAction act = legalReal.Count == 1 || rng.NextDouble() < 0.15 ? CombatAction.End : legalReal[rng.Next(legalReal.Count - 1)];
            if (act.EndTurn) return new CombatDecision.EndTurn();
            var card = p.PlayerCombatState!.Hand.Cards[act.HandIndex];
            var target = act.TargetIndex < 0 ? null : CombatTargetCandidates.ForCard(state, p, card.TargetType).ElementAt(act.TargetIndex);
            return new CombatDecision.PlayCard(card, target);
        }
    }
}
