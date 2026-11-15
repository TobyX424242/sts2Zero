// S0 tests (src/s0/README.md). Single-threaded console runner, one subcommand per test:
//   resample <runs> <maxStates> <seedPrefix> [hpBoost=3000]    deliverable 1: tests (a)-(d)
//   choices  <runs> <K> <D> <seedPrefix> [hpBoost=3000]        deliverable 2: SimBench rollouts with/without the source
//   import   <battle_starts.jsonl> <maxSpecs> [simSeed=S0]      deliverable 3: import + random play to the end
//   obsv4    [seed] [maxActions] [verbose 0/1]                  layout v4 values on an Aeonglass fight (ObsV4Test.cs)
//   obsv5    [seed]                                              layout v5 values: modified intent damage, relic counters (ObsV5Test.cs)
using System.Diagnostics;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Combat.StateDescription;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Characters;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;

ModelDb.Init(ContentRegistry.AllTypes);
string cmd = args.Length > 0 ? args[0] : "help";
return cmd switch
{
    "resample" => await ResampleTests.Run(int.Parse(args[1]), int.Parse(args[2]), args[3], args.Length > 4 ? int.Parse(args[4]) : 3000),
    "choices" => await ChoiceTests.Run(int.Parse(args[1]), int.Parse(args[2]), int.Parse(args[3]), args[4], args.Length > 5 ? int.Parse(args[5]) : 3000),
    "import" => await ImportTests.Run(args[1], int.Parse(args[2]), args.Length > 3 ? args[3] : "S0"),
    "m1" => await M1Runner.Run(args),
    "bridge" => BridgeBench.Run(args),
    "encode" => await EncodeExport.Run(int.Parse(args[1]), int.Parse(args[2]), args[3], args.Length > 4 ? args[4] : "src/s0/data/combat_vocab_v1.json"),
    "obsv4" => await ObsV4Test.Run(args),
    "obsv5" => await ObsV5Test.Run(args),
    "actions" => await ActionTests.Run(int.Parse(args[1]), int.Parse(args[2]), int.Parse(args[3]), int.Parse(args[4]), args[5], args.Length > 6 ? int.Parse(args[6]) : 3000),
    _ => Usage(),
};

static int Usage()
{
    Console.WriteLine("usage: S0Tests resample <runs> <maxStates> <prefix> [hp] | choices <runs> <K> <D> <prefix> [hp] | import <jsonl> <n> [seed]");
    return 2;
}

static class Util
{
    public static CombatStateDescriptionDigest Digest(CombatState s)
    {
        var b = new CombatStateDescriptionBuilder();
        CombatStateDescription.AppendExactState(ref b, s);
        return b.Build();
    }

    public static Task<RunDriver.Result> PlayRun(string seed, int hpBoost, IRunDecisionSource policy)
    {
        var acts = ActDefinition.GetRandomList(seed);
        var runState = new RunState(seed, acts, ascensionLevel: 10);
        var player = Player.CreateForNewRun(ModelDb.Character<Defect>(), runState);
        runState.AddPlayer(player);
        if (hpBoost > 0) { player.Creature.SetMaxHpInternal(hpBoost); player.Creature.SetCurrentHpInternal(hpBoost); }
        return new RunDriver(runState, policy).RunAsync(maxFloors: 60);
    }

    public static string Key(Exception e) => e.GetType().Name + ": " + (e.Message.Length > 90 ? e.Message[..90] : e.Message);

    public static void Count(Dictionary<string, int> d, string k) => d[k] = d.GetValueOrDefault(k) + 1;
}

/// <summary>Random policy for the real run. Combat: a random legal play, or end turn with p=0.15. Card prompts:
/// a random option from <see cref="CardSelectionOptions.Enumerate"/>, recorded so a clone can replay them.</summary>
class RandomRunPolicy(Random rng) : IRunDecisionSource
{
    public readonly List<SelectionOption> SelectionLog = new();
    public Func<CombatState, CombatAction, Task>? OnDecision; // called with the chosen action before it is applied
    public Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options) => Task.FromResult(options[rng.Next(options.Count)]);

    public Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request)
    {
        var options = CardSelectionOptions.Enumerate(request);
        var o = options[rng.Next(options.Count)];
        SelectionLog.Add(o);
        return Task.FromResult(o.Apply(request));
    }

    public async Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
    {
        var legal = CombatActions.Legal(state, state.Players[0]);
        CombatAction a = legal.Count == 1 || rng.NextDouble() < 0.15 ? CombatAction.End : legal[rng.Next(legal.Count - 1)];
        if (OnDecision is not null) await OnDecision(state, a);
        if (a.EndTurn) return new CombatDecision.EndTurn();
        var p = state.Players[0];
        var card = p.PlayerCombatState!.Hand.Cards[a.HandIndex];
        var target = a.TargetIndex < 0 ? null : CombatTargetCandidates.ForCard(state, p, card.TargetType).ElementAt(a.TargetIndex);
        return new CombatDecision.PlayCard(card, target);
    }
}

static class ResampleTests
{
    public static async Task<int> Run(int runs, int maxStates, string prefix, int hp)
    {
        var states = new List<CombatState>();
        // (a1) original vs exact clone: clone at decision k, apply the real decision (with the real card choices),
        // compare with the real state at decision k+1 of the same fight.
        long a1Compared = 0, a1Mismatch = 0;
        CombatState? pendingClone = null; CombatAction pendingAction = default; int logStart = 0; object? pendingFight = null;
        var rng = new Random(12345);
        var policy = new RandomRunPolicy(rng);
        policy.OnDecision = async (state, action) =>
        {
            if (pendingClone is not null && ReferenceEquals(pendingFight, state.Engine))
            {
                var src = SearchCardSelectionSource.InstallOn(pendingClone, SelectionMode.Suspend);
                foreach (var o in policy.SelectionLog.Skip(logStart)) src.Script.Enqueue(o);
                try
                {
                    await CombatActions.ApplyAsync(pendingClone, pendingAction);
                    a1Compared++;
                    if (Util.Digest(pendingClone) != Util.Digest(state)) a1Mismatch++;
                }
                catch (Exception e) { a1Mismatch++; Console.Error.WriteLine("a1 replay error: " + Util.Key(e)); }
            }
            pendingClone = state.Clone(); pendingAction = action; logStart = policy.SelectionLog.Count; pendingFight = state.Engine;
            if (states.Count < maxStates) states.Add(state.Clone());
        };
        for (int i = 0; i < runs && states.Count < maxStates; i++)
        {
            try { await Util.PlayRun($"{prefix}{i:D7}", hp, policy); }
            catch (Exception e) { Console.Error.WriteLine($"run {i} failed: {Util.Key(e)}"); }
        }
        Console.WriteLine($"collected {states.Count} combat decision states from real runs (Defect A10, HP {hp})");
        Console.WriteLine($"(a1) original vs exact clone, one real action later: compared {a1Compared}, digest mismatches {a1Mismatch}");

        // (a2) two exact clones under the same random action sequence (5 actions) stay identical (full digest).
        long a2Steps = 0, a2Mismatch = 0, a2Err = 0;
        // (b) resampled clones: observable state equal; hidden order differs.
        long bObsMismatch = 0, bDrawOrderComparable = 0, bDrawOrderDiffers = 0, bTurnComparable = 0, bHandDiffers = 0;
        long kTopTotal = 0, kBottomTotal = 0, kStatesWithKnown = 0;
        // (c) leak: next drawn card.
        long cHitCtrl = 0;
        long cN = 0, cHitRE = 0, cHitRR = 0, cInformative = 0, cHitREinf = 0, cHitRRinf = 0; double cExpect = 0, cVar = 0, cExpectInf = 0, cVarInf = 0;
        long eN = 0, eHitRE = 0, eHitRR = 0; // end-turn variant: whole next hand as a multiset
        long encTicks = 0, encN = 0, encTokens = 0, encMax = 0, encMismatch = 0; // deliverable 4 encoder
        var errors = new Dictionary<string, int>();
        for (int si = 0; si < states.Count; si++)
        {
            CombatState s = states[si];
            if (CombatActions.IsOver(s)) continue;
            Player sp = s.Players[0];
            string obs = ObservableState.Fingerprint(s);
            var know = ResamplingClone.DrawPileKnowledge(sp);
            if (know.Top + know.Bottom > 0) { kStatesWithKnown++; kTopTotal += know.Top; kBottomTotal += know.Bottom; }
            try
            {
                // (a2)
                var e1 = s.Clone(); var e2 = s.Clone();
                var r1 = new Random(si); var r2 = new Random(si);
                SearchCardSelectionSource.InstallOn(e1, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(si + 7)));
                SearchCardSelectionSource.InstallOn(e2, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(si + 7)));
                for (int step = 0; step < 5 && !CombatActions.IsOver(e1); step++)
                {
                    var l1 = CombatActions.Legal(e1, e1.Players[0]); var l2 = CombatActions.Legal(e2, e2.Players[0]);
                    var x1 = l1[r1.Next(l1.Count)]; var x2 = l2[r2.Next(l2.Count)];
                    await CombatActions.ApplyAsync(e1, x1); await CombatActions.ApplyAsync(e2, x2);
                    a2Steps++;
                    if (x1 != x2 || Util.Digest(e1) != Util.Digest(e2)) { a2Mismatch++; break; }
                }
            }
            catch (Exception e) { a2Err++; Util.Count(errors, "a2 " + Util.Key(e)); }

            try
            {
                // (b)
                ulong seedA = ResamplingClone.Mix(0xA11CE, (ulong)si), seedB = ResamplingClone.Mix(0xB0B, (ulong)si);
                var ra = ResamplingClone.Resampled(s, seedA); var rb = ResamplingClone.Resampled(s, seedB);
                if (ObservableState.Fingerprint(ra) != obs || ObservableState.Fingerprint(rb) != obs) bObsMismatch++;
                long te = Stopwatch.GetTimestamp();
                var enc = CombatObservation.Encode(s, 128);
                encTicks += Stopwatch.GetTimestamp() - te; encN++;
                int used = enc.Count(x => x.Kind != TokenKind.Pad); encTokens += used; encMax = Math.Max(encMax, used);
                if (EncString(enc) != EncString(CombatObservation.Encode(ra, 128))) encMismatch++;
                if (know.Unknown >= 2)
                {
                    bDrawOrderComparable++;
                    var da = ra.Players[0].PlayerCombatState!.DrawPile.Cards.Select(ObservableState.Card);
                    var db = rb.Players[0].PlayerCombatState!.DrawPile.Cards.Select(ObservableState.Card);
                    if (!da.SequenceEqual(db)) bDrawOrderDiffers++;
                }
                // end turn on E, Ra, Rb: compare the next hand (multiset) -> (b) differs, (c-endturn) leak check
                var ex = s.Clone();
                foreach (var c in new[] { ex, ra, rb })
                    SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(si)));
                await CombatActions.ApplyAsync(ex, CombatAction.End);
                await CombatActions.ApplyAsync(ra, CombatAction.End);
                await CombatActions.ApplyAsync(rb, CombatAction.End);
                if (!CombatActions.IsOver(ex) && !CombatActions.IsOver(ra) && !CombatActions.IsOver(rb))
                {
                    string he = Hand(ex), ha = Hand(ra), hb = Hand(rb);
                    bTurnComparable++; if (ha != hb) bHandDiffers++;
                    eN++; if (ha == he) eHitRE++; if (ha == hb) eHitRR++;
                }
            }
            catch (Exception e) { Util.Count(errors, "b " + Util.Key(e)); }

            try
            {
                // (c) draw exactly one card in E, Ra, Rb through the engine's draw (reshuffles if needed).
                var ex = s.Clone();
                var ra = ResamplingClone.Resampled(s, ResamplingClone.Mix(0xC0FFEE, (ulong)si));
                var rb = ResamplingClone.Resampled(s, ResamplingClone.Mix(0xDECAF, (ulong)si));
                var rc = s.Clone(); ResamplingClone.ResampleInPlace(rc, ResamplingClone.Mix(0xBAD, (ulong)si), shuffleDrawPile: false);
                var pcs = sp.PlayerCombatState!;
                if (pcs.Hand.Cards.Count >= 10) continue;
                // posterior probability that a resampled clone draws the card key the exact clone draws
                var drawn = new List<string>();
                foreach (var c in new[] { ex, ra, rb, rc })
                {
                    SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(si)));
                    var d = await CardPileCmd.Draw(c, 1, c.Players[0], fromHandDraw: false);
                    drawn.Add(d.Count == 1 ? ObservableState.CardIdentity(d[0]) : "");
                }
                if (drawn.Any(x => x == "")) continue;
                var draw = pcs.DrawPile.Cards;
                List<string> pool;
                bool informative;
                if (draw.Count > 0)
                {
                    if (know.Top > 0 || know.Unknown == 0) { pool = new() { ObservableState.CardIdentity(draw[0]) }; informative = false; }
                    else { pool = draw.Skip(know.Top).Take(know.Unknown).Select(ObservableState.CardIdentity).ToList(); informative = pool.Distinct().Count() > 1; }
                }
                else { pool = pcs.DiscardPile.Cards.Select(ObservableState.CardIdentity).ToList(); informative = pool.Distinct().Count() > 1; }
                double p = pool.Count(x => x == drawn[0]) / (double)pool.Count;
                cN++; cExpect += p; cVar += p * (1 - p);
                if (drawn[1] == drawn[0]) cHitRE++;
                if (drawn[1] == drawn[2]) cHitRR++;
                if (drawn[3] == drawn[0]) cHitCtrl++;
                if (informative)
                {
                    cInformative++; cExpectInf += p; cVarInf += p * (1 - p);
                    if (drawn[1] == drawn[0]) cHitREinf++;
                    if (drawn[1] == drawn[2]) cHitRRinf++;
                }
            }
            catch (Exception e) { Util.Count(errors, "c " + Util.Key(e)); }
        }
        Console.WriteLine($"(a2) exact clone vs exact clone, same 5 random actions: steps {a2Steps}, digest mismatches {a2Mismatch}, errors {a2Err}");
        Console.WriteLine($"(b) resampled clones: observable-fingerprint mismatches {bObsMismatch} of {states.Count}; " +
            $"draw order differs between two seeds in {bDrawOrderDiffers}/{bDrawOrderComparable} states with >=2 unknown cards; " +
            $"next hand differs after end turn in {bHandDiffers}/{bTurnComparable}");
        Console.WriteLine($"    states with known draw-pile positions: {kStatesWithKnown} (known top cards {kTopTotal}, known bottom cards {kBottomTotal})");
        Console.WriteLine($"(c) leak, next drawn card: n={cN}; resampled==exact {cHitRE} ({(double)cHitRE / cN:P2}); " +
            $"posterior expectation {cExpect:F1} ({cExpect / cN:P2}), z={(cHitRE - cExpect) / Math.Sqrt(cVar):F2}; resampled==resampled {cHitRR} ({(double)cHitRR / cN:P2})");
        Console.WriteLine($"    negative control (RNG reseeded but draw pile NOT reshuffled): control==exact {cHitCtrl} ({(double)cHitCtrl / cN:P2})");
        Console.WriteLine($"    informative subset (>=2 distinct candidate cards): n={cInformative}; resampled==exact {cHitREinf} ({(double)cHitREinf / Math.Max(1, cInformative):P2}); " +
            $"expectation {cExpectInf:F1} ({cExpectInf / Math.Max(1, cInformative):P2}), z={(cHitREinf - cExpectInf) / Math.Sqrt(Math.Max(1e-9, cVarInf)):F2}; resampled==resampled {cHitRRinf} ({(double)cHitRRinf / Math.Max(1, cInformative):P2})");
        Console.WriteLine($"    end-turn variant (whole next hand as multiset): n={eN}; resampled==exact {eHitRE} ({(double)eHitRE / Math.Max(1, eN):P2}); resampled==resampled {eHitRR} ({(double)eHitRR / Math.Max(1, eN):P2})");
        foreach (var kv in errors.OrderByDescending(kv => kv.Value).Take(8)) Console.WriteLine($"    error x{kv.Value}: {kv.Key}");
        Console.WriteLine($"(4) observation encoder: {encN} states, mean {(double)encTokens / Math.Max(1, encN):F1} tokens (max {encMax}), " +
            $"{encTicks * 1e6 / Stopwatch.Frequency / Math.Max(1, encN):F1} us/state; encoding differs between original and resampled clone in {encMismatch}");

        await KnownPositionTest(states);

        // (d) throughput, single thread: clone (+resample) + one random action (20% end turn). Exact and resampled
        // are interleaved state by state (same states, same action choice rule) so JIT/GC drift hits both equally.
        await Throughput(states, reps: 1, report: false);
        await Throughput(states, reps: 5, report: true);
        return 0;
    }

    // Known positions: play Headbutt (puts a chosen discard card on top of the draw pile) in a clone, then check
    // that resampled clones keep that card on top and draw it, and that a random-position insert forgets it.
    static async Task KnownPositionTest(List<CombatState> states)
    {
        int randomInsertTried = 0;
        int cases = 0, knownAfterPlay = 0, keptTop = 0, seedsTried = 0, drewKnown = 0, randomInsertForgets = 0, shuffledOthers = 0, othersComparable = 0;
        var errors = new Dictionary<string, int>();
        foreach (var s in states)
        {
            if (cases >= 300) break;
            var pcs0 = s.Players[0].PlayerCombatState!;
            if (CombatActions.IsOver(s) || pcs0.DiscardPile.Cards.Count == 0 || pcs0.Hand.Cards.Count >= 9 || pcs0.DrawPile.Cards.Count < 3) continue;
            try
            {
                var k = s.Clone();
                SearchCardSelectionSource.InstallOn(k, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(cases)));
                Player p = k.Players[0];
                var pcs = p.PlayerCombatState!;
                var hb = (CardModel)ModelDb.Card<Sts2Sim.Core.Models.Cards.Headbutt>().MutableClone();
                hb.AssignOwner(p);
                await CardPileCmd.EnterCombat(k, hb, Sts2Sim.Core.Entities.Cards.PileType.Hand);
                pcs.Energy = Math.Max(pcs.Energy, 3);
                var target = CombatTargetCandidates.ForCard(k, p, hb.TargetType).First();
                int drawBefore = pcs.DrawPile.Cards.Count;
                await k.Engine!.PlayCardAsync(p, hb, target);
                k.Engine.CheckWinCondition();
                if (CombatActions.IsOver(k)) continue;
                cases++;
                if (pcs.DrawPile.KnownTopCount != 1)
                {
                    if (Environment.GetEnvironmentVariable("S0_DEBUG") == "1")
                        Console.Error.WriteLine($"known={pcs.DrawPile.KnownTopCount}/{pcs.DrawPile.KnownBottomCount} drawBefore={drawBefore} enemies={string.Join(",", k.Enemies.Select(x => x.Monster?.Id.Entry + ":" + string.Join("/", x.Powers.Select(y => y.Id.Entry))))} draw={pcs.DrawPile.Cards.Count} discard={pcs.DiscardPile.Cards.Count} hand={pcs.Hand.Cards.Count} powers={string.Join(",", p.Creature.Powers.Select(x => x.Id.Entry))} relics={string.Join(",", p.Relics.Select(x => x.Id.Entry))}");
                    continue;
                }
                knownAfterPlay++;
                string top = ObservableState.CardIdentity(pcs.DrawPile.Cards[0]);
                bool allKept = true;
                for (int j = 0; j < 10; j++)
                {
                    var r = ResamplingClone.Resampled(k, ResamplingClone.Mix(0x7E57, (ulong)(cases * 100 + j)));
                    seedsTried++;
                    var rp = r.Players[0].PlayerCombatState!;
                    if (ObservableState.CardIdentity(rp.DrawPile.Cards[0]) != top) allKept = false;
                    if (rp.DrawPile.Cards.Count - 1 >= 2)
                    {
                        othersComparable++;
                        if (!rp.DrawPile.Cards.Skip(1).Select(ObservableState.CardIdentity).SequenceEqual(pcs.DrawPile.Cards.Skip(1).Select(ObservableState.CardIdentity))) shuffledOthers++;
                    }
                    SearchCardSelectionSource.InstallOn(r, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(j)));
                    var d = await CardPileCmd.Draw(r, 1, r.Players[0], fromHandDraw: false);
                    if (d.Count == 1 && ObservableState.CardIdentity(d[0]) == top) drewKnown++;
                    else if (Environment.GetEnvironmentVariable("S0_DEBUG") == "1")
                        Console.Error.WriteLine($"drew {(d.Count == 1 ? ObservableState.CardIdentity(d[0]) : "none")} expected {top}; hand={rp.Hand.Cards.Count} powers={string.Join(",", r.Players[0].Creature.Powers.Select(x => x.Id.Entry))}");
                }
                if (allKept) keptTop++;
                // random insert forgets the knowledge
                var k2 = k.Clone();
                var any = k2.Players[0].PlayerCombatState!.Hand.Cards.FirstOrDefault();
                if (any is not null)
                {
                    randomInsertTried++;
                    CardPileCmd.Add(any, Sts2Sim.Core.Entities.Cards.PileType.Draw, Sts2Sim.Core.Entities.Cards.CardPilePosition.Random);
                    if (k2.Players[0].PlayerCombatState!.DrawPile.KnownTopCount == 0) randomInsertForgets++;
                }
            }
            catch (Exception e) { Util.Count(errors, Util.Key(e)); }
        }
        Console.WriteLine($"(b2) known positions (Headbutt puts a discard card on top): cases {cases}, top marked known {knownAfterPlay}, " +
            $"top kept in all 10 resampled clones {keptTop}/{knownAfterPlay}, resampled clones drew the known card {drewKnown}/{seedsTried}; " +
            $"rest of pile reordered {shuffledOthers}/{othersComparable}; a following random insert cleared the knowledge {randomInsertForgets}/{randomInsertTried}");
        foreach (var kv in errors.OrderByDescending(kv => kv.Value).Take(5)) Console.WriteLine($"    error x{kv.Value}: {kv.Key}");
    }

    static string EncString(List<ObsToken> enc) => string.Join(";", enc.Select(x => $"{x.Kind}|{x.Id}|{x.Id2}|{string.Join(",", x.F)}"));

    static string Hand(CombatState s) => string.Join(",", s.Players[0].PlayerCombatState!.Hand.Cards.Select(ObservableState.Card).Order(StringComparer.Ordinal));

    static async Task Throughput(List<CombatState> states, int reps, bool report)
    {
        var rng = new Random(99);
        var policy = SearchCardSelectionSource.RandomPolicy(new Random(5));
        long[] cloneT = new long[2], resT = new long[2], actT = new long[2], endN = new long[2]; long n = 0;
        for (int rep = 0; rep < reps; rep++)
            foreach (var s in states)
            {
                if (CombatActions.IsOver(s)) continue;
                bool endTurn = rng.NextDouble() < 0.2; int pick = rng.Next(1 << 20);
                for (int m = 0; m < 2; m++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    var c = s.Clone();
                    long t1 = Stopwatch.GetTimestamp();
                    if (m == 1) ResamplingClone.ResampleInPlace(c, (ulong)n * 7919UL + 1);
                    long t2 = Stopwatch.GetTimestamp();
                    SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, policy);
                    var legal = CombatActions.Legal(c, c.Players[0]);
                    var a = legal.Count == 1 || endTurn ? CombatAction.End : legal[pick % (legal.Count - 1)];
                    long t3 = Stopwatch.GetTimestamp();
                    try { await CombatActions.ApplyAsync(c, a); } catch { }
                    long t4 = Stopwatch.GetTimestamp();
                    cloneT[m] += t1 - t0; resT[m] += t2 - t1; actT[m] += t4 - t3; if (a.EndTurn) endN[m]++;
                }
                n++;
            }
        if (!report) return;
        double us(long t) => t * 1e6 / Stopwatch.Frequency / n;
        double rate(int m) => 1e6 / (us(cloneT[m]) + us(resT[m]) + us(actT[m]));
        for (int m = 0; m < 2; m++)
            Console.WriteLine($"    {(m == 1 ? "resampled" : "exact    ")}: n={n} (end turns {endN[m]}) clone {us(cloneT[m]):F1} us, resample {us(resT[m]):F2} us, action {us(actT[m]):F1} us -> {rate(m):F0}/s");
        Console.WriteLine($"(d) clone + one action, single thread: exact {rate(0):F0}/s, resampled {rate(1):F0}/s ({rate(1) / rate(0):P0} of exact)");
    }
}

static class ChoiceTests
{
    public static async Task<int> Run(int runs, int k, int d, string prefix, int hp)
    {
        var stats = new Dictionary<string, long>();
        var errOff = new Dictionary<string, int>(); var errOn = new Dictionary<string, int>();
        long probes = 0, probeMismatch = 0, pauseChecked = 0, pauseMismatch = 0, promptsOn = 0;
        var maxOptions = 0;
        var rng = new Random(777);
        var policy = new RandomRunPolicy(new Random(1));
        policy.OnDecision = async (state, _) =>
        {
            stats["decisions"] = stats.GetValueOrDefault("decisions") + 1;
            for (int c = 0; c < k; c++)
            {
                int seed = rng.Next();
                // the same random action sequence with and without the search selection source
                foreach (bool on in new[] { false, true })
                {
                    var clone = state.Clone();
                    var ar = new Random(seed);
                    SearchCardSelectionSource? src = on ? SearchCardSelectionSource.InstallOn(clone, SelectionMode.Policy,
                        SearchCardSelectionSource.RandomPolicy(new Random(seed + 1))) : null;
                    var actions = new List<CombatAction>();
                    stats[on ? "rollouts_on" : "rollouts_off"] = stats.GetValueOrDefault(on ? "rollouts_on" : "rollouts_off") + 1;
                    try
                    {
                        for (int step = 0; step < d && !CombatActions.IsOver(clone); step++)
                        {
                            var legal = CombatActions.Legal(clone, clone.Players[0]);
                            var a = legal.Count == 1 || ar.NextDouble() < 0.25 ? CombatAction.End : legal[ar.Next(legal.Count - 1)];
                            int before = src?.PromptsSeen ?? 0;
                            await CombatActions.ApplyAsync(clone, a);
                            actions.Add(a);
                            if (src is not null && src.PromptsSeen > before)
                            {
                                promptsOn += src.PromptsSeen - before;
                                // consistency of the other two modes on this action: Suspend probe + scripted replay,
                                // and Pause/Resolve, must both reproduce the Policy-mode result exactly.
                                if (probes < 3000)
                                {
                                    var answers = src.History.Skip(before).ToList();
                                    var (ok1, ok2, nopt) = await CheckModes(state, actions, src.History.Take(before).ToList(), answers, clone);
                                    maxOptions = Math.Max(maxOptions, nopt);
                                    probes++; if (!ok1) probeMismatch++;
                                    pauseChecked++; if (!ok2) pauseMismatch++;
                                }
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        stats[on ? "errors_on" : "errors_off"] = stats.GetValueOrDefault(on ? "errors_on" : "errors_off") + 1;
                        Util.Count(on ? errOn : errOff, Util.Key(e));
                    }
                }
            }
        };
        int failed = 0;
        for (int i = 0; i < runs; i++)
        {
            try { await Util.PlayRun($"{prefix}{i:D7}", hp, policy); }
            catch (Exception e) { failed++; Console.Error.WriteLine($"run {i} failed: {Util.Key(e)}"); }
        }
        long offN = stats.GetValueOrDefault("rollouts_off"), onN = stats.GetValueOrDefault("rollouts_on");
        long offE = stats.GetValueOrDefault("errors_off"), onE = stats.GetValueOrDefault("errors_on");
        Console.WriteLine($"runs {runs} (failed {failed}), real decisions {stats.GetValueOrDefault("decisions")}, rollouts per mode {offN} (K={k}, D={d})");
        Console.WriteLine($"without source (iRyougi default): errors {offE} ({(double)offE / Math.Max(1, offN):P2})");
        foreach (var kv in errOff.OrderByDescending(kv => kv.Value).Take(5)) Console.WriteLine($"    x{kv.Value}: {kv.Key}");
        Console.WriteLine($"with SearchCardSelectionSource (random choices): errors {onE} ({(double)onE / Math.Max(1, onN):P2}); prompts answered {promptsOn}");
        foreach (var kv in errOn.OrderByDescending(kv => kv.Value).Take(5)) Console.WriteLine($"    x{kv.Value}: {kv.Key}");
        Console.WriteLine($"mode consistency on prompting actions: Suspend probe + scripted replay mismatches {probeMismatch}/{probes}; " +
            $"Pause/Resolve mismatches {pauseMismatch}/{pauseChecked}; largest option list {maxOptions}");
        return 0;
    }

    // Replays `actions` on fresh clones of `root`: earlier prompts scripted, the prompting (last) action handled
    // by (1) Suspend -> probe options -> new clone with the full script, (2) Pause -> Resolve. Compare with `reference`.
    static async Task<(bool, bool, int)> CheckModes(CombatState root, List<CombatAction> actions,
        List<SelectionOption> earlier, List<SelectionOption> answers, CombatState reference)
    {
        var refDigest = Util.Digest(reference);
        // (1) Suspend probe
        var probe = root.Clone();
        var ps = SearchCardSelectionSource.InstallOn(probe, SelectionMode.Suspend);
        foreach (var o in earlier) ps.Script.Enqueue(o);
        int nopt = 0;
        try
        {
            foreach (var a in actions) await CombatActions.ApplyAsync(probe, a);
        }
        catch (SelectionPendingException e)
        {
            nopt = e.Options.Count;
            if (!e.Options.Any(o => o.Indices.SequenceEqual(answers[0].Indices))) return (false, false, nopt);
        }
        var replay = root.Clone();
        var rs = SearchCardSelectionSource.InstallOn(replay, SelectionMode.Suspend);
        foreach (var o in earlier.Concat(answers)) rs.Script.Enqueue(o);
        bool ok1;
        try { foreach (var a in actions) await CombatActions.ApplyAsync(replay, a); ok1 = Util.Digest(replay) == refDigest; }
        catch { ok1 = false; }
        // (2) Pause / Resolve
        var paused = root.Clone();
        var qs = SearchCardSelectionSource.InstallOn(paused, SelectionMode.Pause);
        foreach (var o in earlier) qs.Script.Enqueue(o);
        bool ok2;
        try
        {
            for (int i = 0; i < actions.Count; i++)
            {
                Task t = CombatActions.ApplyAsync(paused, actions[i]);
                int next = 0;
                while (!t.IsCompleted && qs.Pending is not null && next < answers.Count) qs.Resolve(answers[next++]);
                await t;
            }
            ok2 = Util.Digest(paused) == refDigest;
        }
        catch { ok2 = false; }
        return (ok1, ok2, nopt);
    }
}

static class ImportTests
{
    public static async Task<int> Run(string path, int max, string simSeed)
    {
        var specs = BattleStartSpec.ReadJsonl(path).Take(max).ToList();
        var causes = new Dictionary<string, int>();
        int imported = 0, finished = 0, won = 0, lost = 0, capped = 0, liveOk = 0, hpChanged = 0;
        long actions = 0;
        var sw = Stopwatch.StartNew();
        var byRoom = new Dictionary<string, (int n, int ok)>();
        var matches = new Dictionary<string, int>();
        for (int i = 0; i < specs.Count; i++)
        {
            var spec = specs[i];
            string stage = "import";
            try
            {
                var (s, usedSeed, match) = await BattleStartLoader.LoadMatchedAsync(spec, $"{simSeed}-{i}");
                Util.Count(matches, match.ToString());
                imported++;
                // fidelity of the import: run deck, relics, potions, max HP, monsters vs. the logged battle start
                var pl = s.Players[0];
                string Want(IEnumerable<string> xs) => string.Join(",", xs.Order(StringComparer.Ordinal));
                string deckWant = Want(spec.Deck.Select(c => $"{c.Id}+{c.Upgrade}~{c.Enchantment}"));
                string deckHave = Want(pl.Deck.Cards.Select(c => $"{c.Id}+{c.CurrentUpgradeLevel}~{c.Enchantments.SingleOrDefault()?.Id.ToString()}"));
                if (deckWant != deckHave) Util.Count(causes, "check: run deck differs from log");
                if (Want(spec.Relics) != Want(pl.Relics.Select(r => r.Id.ToString()))) Util.Count(causes, "check: relics differ from log");
                if (Want(spec.Potions) != Want(pl.PotionSlots.Where(x => x is not null).Select(x => x!.Id.ToString())))
                {
                    Util.Count(causes, "check: potions differ from log");
                    Console.Error.WriteLine($"potions {spec.Id}: log {Want(spec.Potions)} sim {Want(pl.PotionSlots.Select(x => x?.Id.ToString() ?? "-"))} relics {Want(spec.Relics)}");
                }
                if (pl.Creature.MaxHp != spec.MaxHp) Util.Count(causes, "check: max HP differs from log");
                if (pl.Creature.CurrentHp != Math.Min(spec.Hp, pl.Creature.MaxHp))
                {
                    hpChanged++;
                    if (hpChanged <= 6) Console.Error.WriteLine($"hp {spec.Id}: log {spec.Hp}/{spec.MaxHp} sim {pl.Creature.CurrentHp}/{pl.Creature.MaxHp} relics {Want(spec.Relics)}");
                }
                string monWant = Want(spec.MonsterIds), monHave = Want(s.Enemies.Select(e => e.Monster!.Id.ToString()));
                if (match is BattleStartLoader.MonsterMatch.Different or BattleStartLoader.MonsterMatch.Subset)
                    Console.Error.WriteLine($"monsters ({match}) {spec.Id} {spec.EncounterId}: log {monWant} sim {monHave}");
                stage = "play";
                var rng = new Random(i);
                SearchCardSelectionSource.InstallOn(s, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(i + 1)));
                int steps = 0;
                while (!CombatActions.IsOver(s) && steps < 400)
                {
                    var legal = CombatActions.Legal(s, s.Players[0]);
                    var a = legal.Count == 1 || rng.NextDouble() < 0.2 ? CombatAction.End : legal[rng.Next(legal.Count - 1)];
                    await CombatActions.ApplyAsync(s, a);
                    steps++;
                }
                actions += steps;
                if (!CombatActions.IsOver(s)) { capped++; Util.Count(causes, "play: not finished within 400 actions"); continue; }
                finished++;
                if (s.Engine!.Won) won++; else lost++;
                var r = byRoom.GetValueOrDefault(spec.RoomType); byRoom[spec.RoomType] = (r.n + 1, r.ok + 1);
            }
            catch (BattleImportException e) { Util.Count(causes, $"{stage}: {e.Cause}: {Short(e.Message)}"); Room(byRoom, spec); }
            catch (Exception e) { Util.Count(causes, $"{stage}: {Util.Key(e)}"); Room(byRoom, spec); }
            // the live (non-clone) path through RunDriver with a random policy, for comparison
            try
            {
                var room = await BattleStartLoader.PlayLiveAsync(spec, $"{simSeed}-live-{i}", new RandomRunPolicy(new Random(i)));
                liveOk++;
            }
            catch (Exception e) { Util.Count(causes, "live: " + (e is BattleImportException b ? b.Cause + ": " + Short(b.Message) : Util.Key(e))); }
        }
        Console.WriteLine($"battle starts {specs.Count}: imported {imported}, played to the end {finished} (won {won}, lost {lost}), " +
            $"capped {capped}; actions {actions}; live RunDriver path finished {liveOk}; wall {sw.Elapsed.TotalSeconds:F1} s");
        Console.WriteLine($"    fidelity: HP at first decision differs from logged HP in {hpChanged} (combat-start effects); starting enemies vs logged monster_ids: {string.Join(", ", matches.OrderBy(kv => kv.Key).Select(kv => kv.Key + " " + kv.Value))}");
        foreach (var kv in byRoom.OrderBy(kv => kv.Key)) Console.WriteLine($"    {kv.Key}: {kv.Value.ok}/{kv.Value.n} finished");
        foreach (var kv in causes.OrderByDescending(kv => kv.Value)) Console.WriteLine($"    x{kv.Value}: {kv.Key}");
        return 0;
    }

    static void Room(Dictionary<string, (int n, int ok)> d, BattleStartSpec s) { var r = d.GetValueOrDefault(s.RoomType); d[s.RoomType] = (r.n + 1, r.ok); }
    static string Short(string m) { m = System.Text.RegularExpressions.Regex.Replace(m, @"\[\d+\]", "[i]"); return m.Length > 140 ? m[..140] : m; }
}
