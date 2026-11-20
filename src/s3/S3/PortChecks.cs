using System.Collections;
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Content.Acts;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Events;
using Sts2Sim.Core.Hooks;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.CardPools;
using Sts2Sim.Core.Models.Powers;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.Core.ValueProps;
using Sts2Sim.S0;
using Sts2Sim.S2;

namespace Sts2Sim.S3;

/// <summary>
/// ic113 (2026-11-18): checks of the v0.113.1 simulator port with Ironclad (src/ic113/PORT.md). CPU only, uniform
/// networks (no inference server).
/// <code>
/// S3 portcheck what=runs [runs=16] [first=0] [sims=64] [hp=400] [fpp=1] [tokens=500] [ptok=0.2]
///    full Ironclad A10 runs through RunAgent (search with `sims` simulations on every move); every run's deck holds the 8
///    Ironclad cards reworked in v0.113 plus 6 of the other 80 in rotation (16 runs cover all 88, base and upgraded).
///    (1) root immutability: a fingerprint of the whole object graph reachable from the live root (combat, run state,
///    players, cards, RNG streams; runtime locks / caches / event handlers excluded as in iRyougi's own
///    RunCloneGraphSnapshot) before and after each search, with probability fpp per move; plus a fingerprint of every
///    canonical model before and after all runs; (2) empty root action lists; (3) card-token damage / block against the
///    executed effect (TokenCheck) on up to `tokens` root states.
/// S3 portcheck what=cards [per=6] [sims=64] [seed=1]
///    every Ironclad card, base and upgraded, generated into the hand of `per` battle starts (encounters of every act in
///    rotation, the v0.113-changed monsters included) with non-empty draw / discard / exhaust piles: legal actions with
///    prompt expansion contain the card when it is playable, every action of the card applies on a clone without an
///    error, a `sims` search returns a non-empty root and leaves the root unchanged; TokenCheck on each state.
/// S3 portcheck what=events [runs=8] [sims=8] [hp=400] [events=Name,...] [gold=N] [foul=1] [fake_merchant=1] [look_shop=B]
///    full Ironclad runs whose "?" rooms are forced to the v0.113 new / changed events in rotation (Waterfall of Obvious
///    Secrets, Museum of Torture, Shrine of the Warlord, Thermae Nonupeipe, Wuppy in Trouble, Doors of Light and Dark (the
///    "light" door: a hand-picked upgrade since v0.113), Zen Weaver, Waterlogged Scriptorium, the Lantern Key), starting with
///    Wuppy (damaged) and an evolvable Byrdpip so rest sites offer "restore" and "evolve"; the uniform macro policy is
///    sampled so every option is taken: counts of events, options, rest options offered / chosen, deck selections; errors.
/// </code>
/// </summary>
public static class PortChecks
{
    public static readonly string[] Reworked =
        { "CARD.DISMANTLE", "CARD.EVIL_EYE", "CARD.EXPECT_A_FIGHT", "CARD.FORGOTTEN_RITUAL", "CARD.PACTS_END", "CARD.RAMPAGE", "CARD.SPITE", "CARD.THRASH" };

    public static List<string> IroncladCards() =>
        new IroncladCardPool().AllCards.Select(c => c.Id.ToString()).Distinct().Order(StringComparer.Ordinal).ToList();

    public static async Task<int> Run(Dictionary<string, string> kv)
    {
        string Get(string k, string d) => kv.TryGetValue(k, out var v) ? v : d;
        int I(string k, int d) => int.Parse(Get(k, d.ToString()));
        double D(string k, double d) => double.Parse(Get(k, d.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
        string what = Get("what", "runs");
        var mvocab = new MacroVocab(Get("mvocab", "src/s3/data/macro_vocab_v113.json"));
        return what switch
        {
            "runs" => await RootRuns(mvocab, I("runs", 16), I("first", 0), I("sims", 64), I("hp", 400), D("fpp", 1.0), I("tokens", 500), D("ptok", 0.2), I("diag", 0) == 1),
            "cards" => await CardStates(I("per", 6), I("sims", 64), I("seed", 1)),
            "events" => await EventRuns(mvocab, I("runs", 8), I("first", 0), I("sims", 8), I("hp", 400), Get("events", ""), I("gold", -1),
                I("foul", 0) == 1, I("fake_merchant", 0) == 1, D("look_shop", 0)),
            "graph" => GraphSelfTest(),
            _ => throw new ArgumentException("what=runs|cards|events|graph"),
        };
    }

    private static ActorServices Services(MacroVocab mvocab, int sims) => new()
    {
        Net = new UniformEvaluator(), Macro = new UniformMacroEvaluator(), MVocab = mvocab,
        Vocab = new ObsLayout.Vocab(ObsLayout.BuildVocab()),
        Cfg = new AgentConfig
        {
            Ascension = 10, SimsFull = sims, SimsFast = sims, PFull = 1.0, Greedy = true, RecordCombat = false, Context = true,
            Search = new NetSearchConfig { GumbelNoise = false, RootCandidates = 16 },
        },
    };

    private static CardModel NewCard(string id, int upgrade, Player owner)
    {
        var c = (CardModel)ModelDb.GetById<CardModel>(ModelId.Deserialize(id)).MutableClone();
        for (int u = 0; u < upgrade && c.IsUpgradable; u++) c.Upgrade();
        c.FloorAddedToDeck = 1;
        c.AssignOwner(owner);
        return c;
    }

    private static string Variant(CardModel c) => $"{c.Id}+{c.CurrentUpgradeLevel}";

    // ------------------------------------------------------------------ runs: root immutability, empty roots, tokens

    private static async Task<int> RootRuns(MacroVocab mvocab, int runs, int first, int sims, int hp, double fpp, int tokens, double ptok, bool diag)
    {
        var ironclad = ModelDb.GetById<CharacterModel>(ModelId.Deserialize("CHARACTER.IRONCLAD"));
        var all = IroncladCards();
        var others = all.Where(id => !Reworked.Contains(id)).ToList();
        string canon0 = CanonicalHash();
        var services = Services(mvocab, sims);
        var rng = new Random(20261118 + first);
        long decisions = 0, checkedRoots = 0, mismatches = 0, emptyRoots = 0, fights0 = 0;
        double hashSeconds = 0; long hashObjects = 0;
        var played = new Dictionary<string, int>(); var inHand = new Dictionary<string, int>();
        var mismatchInfo = new List<string>();
        var tok = new TokenStats();
        string? pending = null;
        RunAgent.SearchHook = async (state, r) =>
        {
            if (r is null)
            {
                decisions++;
                if (decisions == 1) await Sensitivity(state);
                foreach (CardModel c in state.Players[0].PlayerCombatState!.Hand.Cards) inHand[Variant(c)] = inHand.GetValueOrDefault(Variant(c)) + 1;
                pending = null;
                if (rng.NextDouble() < fpp)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    (pending, long n) = StateGraph.Hash(state, keepText: diag);
                    hashSeconds += sw.Elapsed.TotalSeconds; hashObjects += n;
                }
                return;
            }
            if (pending is not null)
            {
                checkedRoots++;
                var (h1, _) = StateGraph.Hash(state);
                if (h1 != pending)
                {
                    mismatches++;
                    if (mismatchInfo.Count < 10)
                        mismatchInfo.Add($"round {state.RoundNumber}, hand {string.Join(",", state.Players[0].PlayerCombatState!.Hand.Cards.Select(Variant))}, " +
                            $"enemies {string.Join(",", state.Enemies.Select(e => e.Monster?.Id.Entry))}, action {r.Action.Key}; diff {StateGraph.FirstDiff(state, pending)}");
                }
            }
            if (r.RootActions.Count == 0 && !CombatActions.IsOver(state)) emptyRoots++;
            if (r.Action.Kind == ActionKind.PlayCard)
            {
                var hand = state.Players[0].PlayerCombatState!.Hand.Cards;
                var card = r.Action.SourceIndex >= 0 && r.Action.SourceIndex < hand.Count ? hand[r.Action.SourceIndex] : null;
                if (card is not null) played[Variant(card)] = played.GetValueOrDefault(Variant(card)) + 1;
            }
            if (tok.States < tokens && rng.NextDouble() < ptok) await TokenCheck(state, rng, tok);
        };
        int errors = 0, won = 0; long fights = 0; var errText = new List<string>();
        for (int i = first; i < first + runs; i++)
        {
            string seed = $"IC113ROOT{i:D4}";
            var run = new RunState(seed, ActDefinition.GetRandomList(seed), ascensionLevel: 10);
            var player = Player.CreateForNewRun(ironclad, run);
            run.AddPlayer(player);
            player.Creature.SetMaxHpInternal(hp); player.Creature.SetCurrentHpInternal(hp);
            foreach (string id in Reworked) player.Deck.AddInternal(NewCard(id, i % 2, player));
            for (int j = 0; j < 6; j++) player.Deck.AddInternal(NewCard(others[(i * 6 + j) % others.Count], (i / 14 + j) % 2, player));
            var agent = new RunAgent(services, run, player, seed, perturbed: false, ascension: 10);
            var driver = new RunDriver(run, agent);
            driver.OnRoomResolved += agent.OnRoomResolved;
            try { agent.Finish(await driver.RunAsync(80)); }
            catch (Exception e) { agent.Fail(e); }
            var rec = agent.Rec;
            fights += rec.Fights; if (rec.Won) won++;
            if (rec.Error) { errors++; errText.Add($"{seed}: {rec.ErrorText}"); }
            Console.WriteLine($"run {seed}: floors {rec.Floors}, fights {rec.Fights} (won {rec.FightsWon}), outcome {rec.Outcome}{(rec.Error ? " ERROR " + rec.ErrorText : "")}; " +
                $"roots checked {checkedRoots}, mismatches {mismatches}, empty roots {emptyRoots}, token states {tok.States}");
        }
        RunAgent.SearchHook = null;
        string canon1 = CanonicalHash();
        var variants = all.SelectMany(id => new[] { id + "+0", id + "+1" }).ToList();
        int upgradable(string v) => v.EndsWith("+1") ? 1 : 0;
        var neverHand = variants.Where(v => !inHand.ContainsKey(v)).ToList();
        var neverPlayed = variants.Where(v => !played.ContainsKey(v)).ToList();
        Console.WriteLine($"ROOT: runs {runs} (won {won}, errors {errors}), fights {fights}, search decisions {decisions}, roots fingerprinted {checkedRoots}, " +
            $"MISMATCHES {mismatches}; empty root action lists {emptyRoots}; canonical models unchanged: {canon0 == canon1}; " +
            $"fingerprint {hashSeconds / Math.Max(1, checkedRoots):F3} s / {hashObjects / Math.Max(1, checkedRoots)} objects per root");
        Console.WriteLine($"coverage: Ironclad card variants (88 x base/upgraded) seen in hand {variants.Count - neverHand.Count}/{variants.Count}, " +
            $"played {variants.Count - neverPlayed.Count}/{variants.Count}; reworked cards played: " +
            string.Join(", ", Reworked.Select(id => $"{id[5..]} {played.GetValueOrDefault(id + "+0")}/{played.GetValueOrDefault(id + "+1")}")));
        if (neverHand.Count > 0) Console.WriteLine("  never in hand: " + string.Join(" ", neverHand));
        if (neverPlayed.Count > 0) Console.WriteLine("  never played: " + string.Join(" ", neverPlayed.Take(60)));
        foreach (string m in mismatchInfo) Console.WriteLine("  mismatch: " + m);
        foreach (string e in errText.Take(10)) Console.WriteLine("  run error: " + e);
        Console.WriteLine("notes: " + string.Join(", ", services.Notes.Where(kv => kv.Key.StartsWith("heal_check") || kv.Key.StartsWith("combat_") || kv.Key.StartsWith("tf_heal"))
            .OrderBy(kv => kv.Key).Select(kv => $"{kv.Key} {kv.Value}")));
        tok.Report();
        bool ok = mismatches == 0 && emptyRoots == 0 && errors == 0 && canon0 == canon1 && checkedRoots > 0;
        Console.WriteLine(ok ? "ROOTCHECK PASSED" : "ROOTCHECK FAILED");
        return ok ? 0 : 1;
    }

    /// <summary>The fingerprint must see what a search could change: on exact clones of the live root, one draw of the
    /// shuffle stream, one draw of the player's NonCombatGeneration stream, one draw of a monster's own stream, a card
    /// upgraded in the draw pile, a power amount, the draw order (two cards swapped) must each change it; and the live
    /// root's graph must reach the run state (its type names are listed).</summary>
    private static async Task Sensitivity(CombatState root)
    {
        var (h, n) = StateGraph.Hash(root, keepText: true);
        string text = StateGraph.Text(h);
        string[] want = { "Sts2Sim.Core.Runs.RunState", "Sts2Sim.Core.Runs.RunRngSet", "Sts2Sim.Core.Random.PlayerRngSet", "Sts2Sim.Core.Map" };
        Console.WriteLine($"sensitivity: live root graph {n} objects; reaches " + string.Join(", ", want.Select(w => $"{w} {text.Contains(w)}")));
        var probes = new List<(string, Action<CombatState>)>
        {
            ("shuffle stream", c => c.RunState.Rng.GetRng(Sts2Sim.Core.Entities.Rngs.RunRngType.Shuffle).NextInt(10)),
            ("NonCombatGeneration stream", c => c.Players[0].PlayerRng.GetRng(Sts2Sim.Core.Entities.Rngs.PlayerRngType.NonCombatGeneration).NextInt(10)),
            ("Shops stream", c => c.Players[0].PlayerRng.GetRng(Sts2Sim.Core.Entities.Rngs.PlayerRngType.Shops).NextInt(10)),
            ("monster stream", c => c.Enemies.First().Monster!.Rng.NextInt(10)),
            ("card upgraded in draw pile", c => c.Players[0].PlayerCombatState!.DrawPile.Cards.FirstOrDefault(x => x.IsUpgradable)?.Upgrade()),
            ("draw order", c => { var d = c.Players[0].PlayerCombatState!.DrawPile; if (d.Cards.Count >= 2) { var x = d.Cards[0]; d.MoveToBottomInternal(x); } }),
            ("enemy HP", c => c.Enemies.First().SetCurrentHpInternal(Math.Max(1, c.Enemies.First().CurrentHp - 1))),
        };
        foreach (var (name, f) in probes)
        {
            CombatState c = root.Clone();
            var (h0, _) = StateGraph.Hash(c);
            try { f(c); } catch (Exception e) { Console.WriteLine($"  probe {name}: {e.GetType().Name}"); continue; }
            var (h1, _) = StateGraph.Hash(c);
            Console.WriteLine($"  probe {name}: fingerprint {(h0 != h1 ? "changes (ok)" : "UNCHANGED")}");
        }
        await Task.CompletedTask;
    }

    // ------------------------------------------------------------------ token check

    public sealed class TokenStats
    {
        public long States, Errors, NibPlays, NibMatch;   // attack plays with Pen Nib armed (counter 9 mod 10) and how many of them match
        public readonly Dictionary<string, long[]> Dmg = new(), Blk = new();   // card -> [n, matchModified, nNeutral, matchNeutral, redirected]
        public readonly List<string> Examples = new();
        public readonly Dictionary<string, int> ErrorKinds = new();
        public readonly HashSet<string> Monsters = new();

        public void Report()
        {
            long dn = Dmg.Values.Sum(x => x[0]), dm = Dmg.Values.Sum(x => x[1]), dnn = Dmg.Values.Sum(x => x[2]), dmn = Dmg.Values.Sum(x => x[3]), red = Dmg.Values.Sum(x => x[4]);
            long bn = Blk.Values.Sum(x => x[0]), bm = Blk.Values.Sum(x => x[1]), bnn = Blk.Values.Sum(x => x[2]), bmn = Blk.Values.Sum(x => x[3]), bmul = Blk.Values.Sum(x => x[4]);
            Console.WriteLine($"TOKENS: {States} states ({Errors} play errors), monsters seen {Monsters.Count}: damage plays {dn}, executed first hit == Hook.ModifyDamage(token) in {dm} " +
                $"(MISMATCH {dn - dm}); neutral states (no Strength/Weak/Vulnerable/Vigor...: modifier = token) {dnn}, executed == token in {dmn} (MISMATCH {dnn - dmn}); " +
                $"redirected hits {red}; block plays {bn}, block gained == Hook.ModifyBlock(token) in {bm} (MISMATCH {bn - bm}, of which {bmul} an exact multiple: the card gains block more than once), neutral {bnn} match {bmn} (MISMATCH {bnn - bmn})");
            Console.WriteLine($"  Pen Nib armed (the play is the 10th attack: x2 in the hook's multiplicative pass, scored with it): {NibPlays} damage plays, {NibMatch} match");
            Console.WriteLine("  v0.113-changed monsters seen: " + string.Join(", ", Monsters.Where(m => m is "AXEBOT" or "INFESTED_PRISM" or "TUNNELER" or "CEREMONIAL_BEAST"
                or "THIEVING_HOPPER" or "AEONGLASS" or "ROCKET").Order()));
            foreach (var (name, d) in new[] { ("damage", Dmg), ("block", Blk) })
            {
                var bad = d.Where(kv => kv.Value[0] != kv.Value[1]).OrderByDescending(kv => kv.Value[0] - kv.Value[1]).ToList();
                Console.WriteLine($"  {name} mismatches by card ({bad.Count} cards): " + string.Join(", ", bad.Select(kv => $"{kv.Key} {kv.Value[0] - kv.Value[1]}/{kv.Value[0]}")));
            }
            foreach (string e in Examples.Take(25)) Console.WriteLine("  e.g. " + e);
            foreach (var kv in ErrorKinds.OrderByDescending(kv => kv.Value).Take(8)) Console.WriteLine($"  play error x{kv.Value}: {kv.Key}");
        }
    }

    /// <summary>On exact clones of <paramref name="root"/> (never the root itself): random Strength / Dexterity / Weak /
    /// Frail on the player and Vulnerable on an enemy (half the states neutral), enemies at 999 HP; each playable hand card
    /// with a token damage or block (CardValues: what the combat observation's hand token carries) is played on a fresh
    /// clone. Damage: the first hit of the card in the damage history against Hook.ModifyDamage(token) computed before the
    /// play (target, dealer, Move props, card); block: the player's block gain against Hook.ModifyBlock(token).</summary>
    public static async Task TokenCheck(CombatState root, Random rng, TokenStats st)
    {
        if (CombatActions.IsOver(root) || root.CurrentSide != CombatSide.Player) return;
        CombatState c0 = root.Clone();
        SearchCardSelectionSource.InstallOn(c0, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(rng.Next())));
        Player p0 = c0.Players[0];
        bool neutral = rng.NextDouble() < 0.5;
        try
        {
            if (!neutral)
            {
                Creature me = p0.Creature;
                if (rng.NextDouble() < 0.6) await PowerCmd.Apply<StrengthPower>(c0, me, rng.Next(-2, 6), me, null);
                if (rng.NextDouble() < 0.4) await PowerCmd.Apply<DexterityPower>(c0, me, rng.Next(-2, 4), me, null);
                if (rng.NextDouble() < 0.4) await PowerCmd.Apply<WeakPower>(c0, me, rng.Next(1, 3), null, null);
                if (rng.NextDouble() < 0.3) await PowerCmd.Apply<FrailPower>(c0, me, rng.Next(1, 3), null, null);
                var en = c0.Enemies.Where(e => e.IsAlive).ToList();
                if (en.Count > 0 && rng.NextDouble() < 0.6) await PowerCmd.Apply<VulnerablePower>(c0, en[rng.Next(en.Count)], rng.Next(1, 3), me, null);
            }
        }
        catch (Exception e) { st.ErrorKinds["apply powers: " + e.GetType().Name] = st.ErrorKinds.GetValueOrDefault("apply powers: " + e.GetType().Name) + 1; return; }
        if (CombatActions.IsOver(c0)) return;
        foreach (var e in c0.Enemies.Where(e => e.IsAlive)) { e.SetMaxHpInternal(999); e.SetCurrentHpInternal(999); st.Monsters.Add(e.Monster?.Id.Entry ?? "?"); }
        bool isNeutral = IsNeutral(c0);
        st.States++;
        var hand = p0.PlayerCombatState!.Hand.Cards;
        var seen = new HashSet<string>();
        for (int i = 0; i < hand.Count; i++)
        {
            CardModel card0 = hand[i];
            if (!seen.Add(CardSelectionOptions.EquivalenceKey(card0))) continue;
            int tokDmg = CardValues.Damage(card0), tokBlk = CardValues.Block(card0);
            if (tokDmg <= 0 && tokBlk <= 0) continue;
            CombatState c = c0.Clone();
            SearchCardSelectionSource.InstallOn(c, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(rng.Next())));
            Player p = c.Players[0];
            CardModel card = p.PlayerCombatState!.Hand.Cards[i];
            p.PlayerCombatState.Energy = Math.Max(p.PlayerCombatState.Energy, 10);
            if (!card.CanPlay(out _)) continue;
            Creature? target = null;
            if (card.TargetType is TargetType.AnyEnemy) target = CombatTargetCandidates.ForCard(c, p, card.TargetType).FirstOrDefault();
            else if (card.TargetType is TargetType.AnyAlly or TargetType.AnyPlayer) target = CombatTargetCandidates.ForCard(c, p, card.TargetType).FirstOrDefault();
            if (card.TargetType is TargetType.AnyEnemy && target is null) continue;
            // expected damage on every enemy (random / all-enemy cards hit others than the first)
            // Pen Nib doubles the 10th attack through its multiplicative hook, which is keyed to the CardPlay object and so
            // invisible to Hook.ModifyDamage called without one; the combat observation carries its counter (mod 10, relic
            // counters of obs v5), so an armed Nib is scored with its x2 (multiplicative: before the floor)
            bool nib = card.Type == CardType.Attack && p.Relics.OfType<Sts2Sim.Core.Models.Relics.PenNib>()
                .Any(r => RelicCounterValues.TryGet(r, out int n) && n == 9);
            var expBy = new Dictionary<Creature, decimal>(ReferenceEqualityComparer.Instance);
            foreach (Creature e in c.Enemies)
                expBy[e] = Math.Max(0m, Math.Floor(Hook.ModifyDamage(c, e, p.Creature, tokDmg, ValueProp.Move, card, null, out _) * (nib ? 2m : 1m)));
            decimal expBlk = Math.Max(0m, Math.Floor(Hook.ModifyBlock(c, p.Creature, tokBlk, ValueProp.Move, card, null, out _)));
            int hist0 = c.DamageHistory.Entries.Count, blk0 = p.Creature.Block;
            string name = Variant(card0);
            try
            {
                await c.Engine!.PlayCardAsync(p, card, target);
            }
            catch (Exception e)
            {
                st.Errors++;
                string k = name + " " + e.GetType().Name + ": " + (e.Message.Length > 60 ? e.Message[..60] : e.Message);
                st.ErrorKinds[k] = st.ErrorKinds.GetValueOrDefault(k) + 1;
                continue;
            }
            if (tokDmg > 0)
            {
                // the card's hits on enemies (self-damage such as Hemokinesis / Breakthrough excluded); damage = blocked +
                // HP loss before HP-loss reducers (what Hook.ModifyDamage predicts; reducers such as a monster's damage cap
                // act after it and are not the token's business)
                var hits = c.DamageHistory.Entries.Skip(hist0).Where(h => h.Dealer == p.Creature && ReferenceEquals(h.CardSource, card) && h.Receiver != p.Creature).ToList();
                if (hits.Count > 0)
                {
                    var a = st.Dmg.TryGetValue(name, out var x) ? x : st.Dmg[name] = new long[5];
                    int got = hits[0].Result.BlockedDamage + hits[0].Result.IntendedHpLoss;
                    Creature recv = hits[0].Result.OriginalTarget;
                    bool redirected = !ReferenceEquals(hits[0].Receiver, recv);
                    decimal expDmg = expBy.TryGetValue(recv, out var ev) ? ev : -1;
                    a[0]++; if (got == expDmg) a[1]++;
                    if (nib) { st.NibPlays++; if (got == expDmg) st.NibMatch++; }
                    if (isNeutral) { a[2]++; if (got == tokDmg) a[3]++; }
                    if (redirected) a[4]++;
                    if (got != expDmg && st.Examples.Count < 40)
                        st.Examples.Add($"{name} damage: token {tokDmg}, expected {expDmg}, executed first hit {got} on {hits[0].Receiver.Monster?.Id.Entry ?? "player"} (hits {hits.Count}{(isNeutral ? ", neutral" : "")}{(nib ? ", Pen Nib armed" : "")}; " +
                            $"player powers {string.Join("/", p.Creature.Powers.Select(w => w.Id.Entry + w.Amount))})");
                }
            }
            if (tokBlk > 0)
            {
                var a = st.Blk.TryGetValue(name, out var x) ? x : st.Blk[name] = new long[5];
                int got = p.Creature.Block - blk0;
                a[0]++; if (got == expBlk) a[1]++; else if (expBlk > 0 && got > expBlk && got % expBlk == 0) a[4]++;
                if (isNeutral) { a[2]++; if (got == tokBlk) a[3]++; }
                if (got != expBlk && st.Examples.Count < 40)
                    st.Examples.Add($"{name} block: token {tokBlk}, expected {expBlk}, gained {got}{(isNeutral ? " (neutral)" : "")}; player powers {string.Join("/", p.Creature.Powers.Select(w => w.Id.Entry + w.Amount))}");
            }
        }
    }

    /// <summary>No power on the player or any enemy and no relic that could modify a card's damage or block: the token value
    /// is then what the play does (the check of the token itself).</summary>
    private static bool IsNeutral(CombatState c) =>
        c.Players[0].Creature.Powers.Count == 0 && c.Enemies.All(e => e.Powers.Count == 0);

    // ------------------------------------------------------------------ cards in prompt states

    private static async Task<int> CardStates(int per, int sims, int seed)
    {
        var rng = new Random(seed);
        var all = IroncladCards();
        var encounters = Encounters();
        Console.WriteLine($"cards: {all.Count} Ironclad cards x base/upgraded x {per} states; {encounters.Count} encounters in rotation");
        long states = 0, loadFail = 0, legalMissing = 0, applyErrors = 0, emptyAfter = 0, emptyRoot = 0, searchErrors = 0, rootMismatch = 0, actions = 0, promptActions = 0, notPlayable = 0;
        var problems = new List<string>();
        var tok = new TokenStats();
        var loadErr = new Dictionary<string, int>();
        int ei = 0;
        foreach (string id in all)
            for (int up = 0; up <= 1; up++)
                for (int k = 0; k < per; k++)
                {
                    var (enc, act, room, actIds) = encounters[ei++ % encounters.Count];
                    var deck = new List<BattleCard>();
                    for (int j = 0; j < 4; j++) deck.Add(new BattleCard("CARD.STRIKE_IRONCLAD", 0, null, 0));
                    for (int j = 0; j < 4; j++) deck.Add(new BattleCard("CARD.DEFEND_IRONCLAD", 0, null, 0));
                    deck.Add(new BattleCard("CARD.BASH", 0, null, 0));
                    for (int j = 0; j < 9; j++) deck.Add(new BattleCard(all[rng.Next(all.Count)], rng.Next(2), null, 0));
                    var spec = new BattleStartSpec
                    {
                        Id = $"card-{id}-{up}-{k}", Character = "CHARACTER.IRONCLAD", Ascension = 10, ActIds = actIds, ActIndex = act, Floor = 5 + 17 * act,
                        RoomType = room, EncounterId = "ENCOUNTER." + enc, Hp = 60, MaxHp = 80, Deck = deck,
                        Relics = new() { "RELIC.BURNING_BLOOD" }, Potions = rng.NextDouble() < 0.5 ? new() { "POTION.FIRE_POTION" } : new(), PotionSlots = 3,
                    };
                    CombatState s;
                    try { s = await BattleStartLoader.LoadAsync(spec, $"IC113CARD{states}"); }
                    catch (Exception e)
                    {
                        loadFail++; string key = e is BattleImportException b ? b.Cause : e.GetType().Name;
                        loadErr[key + " " + enc] = loadErr.GetValueOrDefault(key + " " + enc) + 1; continue;
                    }
                    states++;
                    string tag = $"{id}+{up} vs {enc}";
                    try
                    {
                        SearchCardSelectionSource.InstallOn(s, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(rng.Next())));
                        Player p = s.Players[0];
                        var pcs = p.PlayerCombatState!;
                        // non-empty draw / discard / exhaust piles
                        foreach (var c in pcs.DrawPile.Cards.Take(3).ToList()) CardPileCmd.Add(c, PileType.Discard);
                        foreach (var c in pcs.DrawPile.Cards.Take(2).ToList()) CardPileCmd.Add(c, PileType.Exhaust);
                        var card = NewCard(id, up, p);
                        await CardPileCmd.Generate(s, card, PileType.Hand);
                        pcs.Energy = Math.Max(pcs.Energy, 5);
                        if (CombatActions.IsOver(s)) continue;
                        var legal = await ActionSpace.LegalAsync(s, includePotions: true, expandSelections: true, endTurnPrompts: true);
                        if (legal.Count == 0) { emptyRoot++; problems.Add($"{tag}: no legal action"); continue; }
                        string key = CardSelectionOptions.EquivalenceKey(card);
                        var mine = legal.Where(a => a.Kind == ActionKind.PlayCard && a.Source == key).ToList();
                        bool playable = card.CanPlay(out _);
                        if (!playable) notPlayable++;
                        if (playable && mine.Count == 0) { legalMissing++; if (problems.Count < 40) problems.Add($"{tag}: playable but not among {legal.Count} legal actions"); }
                        foreach (SearchAction a in mine)
                        {
                            actions++; if (a.Selections.Count > 0) promptActions++;
                            var c = ResamplingClone.Resampled(s, (ulong)rng.NextInt64());
                            try
                            {
                                if (!await ActionSpace.ApplyAsync(c, a)) { applyErrors++; if (problems.Count < 40) problems.Add($"{tag}: action {a.Key} not applicable"); continue; }
                                if (!CombatActions.IsOver(c) && (await ActionSpace.LegalAsync(c, true, true)).Count == 0) { emptyAfter++; if (problems.Count < 40) problems.Add($"{tag}: no legal action after {a.Key}"); }
                            }
                            catch (Exception e) { applyErrors++; if (problems.Count < 40) problems.Add($"{tag}: {a.Key}: {e.GetType().Name}: {e.Message[..Math.Min(80, e.Message.Length)]}"); }
                        }
                        // search on this root: non-empty, root unchanged
                        var tf = new TfScorer();
                        var subsets = new List<int[]> { Array.Empty<int>() };
                        if (spec.Potions.Count > 0) subsets.Add(new[] { 0 });
                        var tables = subsets.Select(_ => Enumerable.Range(0, spec.MaxHp + 1).Select(h => h / (double)spec.MaxHp).ToArray()).ToList();
                        tf.Set(spec.Id, spec.MaxHp, spec.Potions, subsets, tables, false);
                        var ctx = BattleContext.From(spec);
                        var search = new NetSearch(new NetSearchConfig { GumbelNoise = false }, new UniformEvaluator(), tf, ctx, 1.0, new FightSignals(tf, ctx, spec, 1.0));
                        var (h0, _) = StateGraph.Hash(s);
                        var r = await search.SearchAsync(s, sims, (ulong)rng.NextInt64());
                        var (h1, _) = StateGraph.Hash(s);
                        if (h0 != h1) { rootMismatch++; if (problems.Count < 40) problems.Add($"{tag}: root changed by the search ({StateGraph.FirstDiff(s, h0)})"); }
                        if (r.RootActions.Count == 0) { emptyRoot++; if (problems.Count < 40) problems.Add($"{tag}: empty search root"); }
                        await TokenCheck(s, rng, tok);
                    }
                    catch (Exception e)
                    {
                        searchErrors++;
                        if (problems.Count < 40) problems.Add($"{tag}: {e.GetType().Name}: {e.Message[..Math.Min(100, e.Message.Length)]}");
                    }
                }
        Console.WriteLine($"CARDS: states {states} (load failures {loadFail}), card actions applied {actions} ({promptActions} with prompt answers), " +
            $"playable card missing from legal actions {legalMissing}, not playable {notPlayable}, apply errors {applyErrors}, empty legal list after a play {emptyAfter}, " +
            $"empty roots {emptyRoot}, search/state errors {searchErrors}, roots changed by the search {rootMismatch}");
        foreach (var kv in loadErr.OrderByDescending(kv => kv.Value).Take(10)) Console.WriteLine($"  load failure x{kv.Value}: {kv.Key}");
        foreach (string pr in problems.Take(40)) Console.WriteLine("  " + pr);
        tok.Report();
        bool ok = states > 0 && legalMissing == 0 && applyErrors == 0 && emptyAfter == 0 && emptyRoot == 0 && searchErrors == 0 && rootMismatch == 0;
        Console.WriteLine(ok ? "CARDCHECK PASSED" : "CARDCHECK FAILED");
        return ok ? 0 : 1;
    }

    /// <summary>Every encounter of every act (monster / elite / boss pools), the act-1 pools of both act-1 variants.</summary>
    private static List<(string Enc, int Act, string Room, List<string> ActIds)> Encounters()
    {
        var res = new List<(string, int, string, List<string>)>();
        var acts = new (ActDefinition A, string Id, int Index)[] { (new Overgrowth(), "ACT.OVERGROWTH", 0), (new Underdocks(), "ACT.UNDERDOCKS", 0), (new Hive(), "ACT.HIVE", 1), (new Glory(), "ACT.GLORY", 2) };
        foreach (var (a, id, idx) in acts)
        {
            var ids = idx == 0 ? new List<string> { id, "ACT.HIVE", "ACT.GLORY" } : new List<string> { "ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY" };
            foreach (var e in a.MonsterEncounterCandidates) res.Add((e.IdEntry, idx, "monster", ids));
            foreach (var e in a.EliteEncounterCandidates) res.Add((e.IdEntry, idx, "elite", ids));
            foreach (var e in a.BossEncounterCandidates) res.Add((e.IdEntry, idx, "boss", ids));
        }
        // interleave so that consecutive states see different acts and rooms
        var rng = new Random(7);
        return res.OrderBy(_ => rng.Next()).ToList();
    }

    // ------------------------------------------------------------------ events and rest sites

    public static readonly string[] ForcedEvents =
        { "WaterfallOfObviousSecrets", "MuseumOfTorture", "ShrineOfTheWarlord", "ThermaeNonupeipe", "WuppyInTrouble", "DoorsOfLightAndDark", "ZenWeaver", "WaterloggedScriptorium", "TheLanternKey" };

    /// <param name="events">ic113 chain (2026-11-18): comma-separated event class names forced instead of ForcedEvents
    /// (e.g. FakeMerchant,CrystalSphere); gold: the run's gold at the start (-1 = unchanged); foul: a Foul Potion in the belt;
    /// at the start and again before each forced event when the belt has none and a free slot; fakeMerchant: AgentConfig.FakeMerchant.</param>
    private static async Task<int> EventRuns(MacroVocab mvocab, int runs, int first, int sims, int hp, string events = "", int gold = -1,
        bool foul = false, bool fakeMerchant = false, double lookShop = 0)
    {
        var ironclad = ModelDb.GetById<CharacterModel>(ModelId.Deserialize("CHARACTER.IRONCLAD"));
        var services = Services(mvocab, sims);
        services.Cfg.Greedy = false;   // sample the (uniform) macro policy so that every rest / event option gets taken
        services.Cfg.FakeMerchant = fakeMerchant; services.Cfg.LookShop = lookShop;
        var names = events == "" ? ForcedEvents : events.Split(',', StringSplitOptions.RemoveEmptyEntries);
        var eventTypes = names.Select(n => typeof(EventModel).Assembly.GetTypes().FirstOrDefault(t => t.Name == n && typeof(EventModel).IsAssignableFrom(t))
            ?? throw new InvalidOperationException("no event " + n)).ToList();
        int errors = 0, fakeRelics = 0; var errText = new List<string>(); int k = 0; var forced = new Dictionary<string, int>();
        for (int i = first; i < first + runs; i++)
        {
            string seed = $"IC113EVT{i:D4}";
            var run = new RunState(seed, ActDefinition.GetRandomList(seed), ascensionLevel: 10);
            var player = Player.CreateForNewRun(ironclad, run);
            run.AddPlayer(player);
            player.Creature.SetMaxHpInternal(hp); player.Creature.SetCurrentHpInternal(hp);
            await RelicCmd.Obtain(ModelDb.Relic<Sts2Sim.Core.Models.Relics.Wuppy>(), player);
            await RelicCmd.Obtain(ModelDb.Relic<Sts2Sim.Core.Models.Relics.Byrdpip>(), player);
            foreach (var w in player.Relics.OfType<Sts2Sim.Core.Models.Relics.Wuppy>()) w.CurrentHp = Math.Max(1, w.MaxHp - 7);
            foreach (var b in player.Relics.OfType<Sts2Sim.Core.Models.Relics.Byrdpip>())
                typeof(Sts2Sim.Core.Models.Relics.Byrdpip).GetProperty("SwoopsPlayed")!.SetValue(b, Sts2Sim.Core.Models.Relics.Byrdpip.PlaysToEvolve);
            if (gold >= 0) { if (player.Gold < gold) await PlayerCmd.GainGold(gold - player.Gold, player); }
            if (foul) player.AddPotionInternal(ModelDb.Potion<Sts2Sim.Core.Models.Potions.FoulPotion>());
            var agent = new RunAgent(services, run, player, seed, perturbed: false, ascension: 10);
            Func<RunState, AbstractRoom> create = _ =>
            {
                Type t = eventTypes[k++ % eventTypes.Count];
                // foul=1: a Foul Potion in the belt when the forced event starts (a free slot taken), so the Fake Merchant can be fought
                if (foul && !player.PotionSlots.Any(q => q is Sts2Sim.Core.Models.Potions.FoulPotion) && player.PotionSlots.Contains(null))
                    player.AddPotionInternal(ModelDb.Potion<Sts2Sim.Core.Models.Potions.FoulPotion>());
                forced[t.Name] = forced.GetValueOrDefault(t.Name) + 1;
                var ev = (EventModel)ModelDb.GetById<EventModel>(ModelDb.GetId(t)).MutableClone();
                return new EventRoom(() => ev);
            };
            var driver = new RunDriver(run, agent, null, null, false, create);
            driver.OnRoomResolved += agent.OnRoomResolved;
            try { agent.Finish(await driver.RunAsync(80)); }
            catch (Exception e) { agent.Fail(e); }
            var rec = agent.Rec;
            fakeRelics += player.Relics.Count(r => r.Id.Entry.StartsWith("FAKE_"));
            if (rec.Error) { errors++; errText.Add($"{seed}: {rec.ErrorText}"); }
            Console.WriteLine($"run {seed}: floors {rec.Floors}, events {rec.Events}, rests {rec.Rests}, outcome {rec.Outcome}{(rec.Error ? " ERROR " + rec.ErrorText : "")}");
        }
        Console.WriteLine($"EVENTS: runs {runs}, run errors {errors}; forced event rooms: " + string.Join(", ", forced.OrderBy(x => x.Key).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("  event decisions: " + string.Join(", ", services.Notes.Where(x => x.Key.StartsWith("event_")).OrderBy(x => x.Key).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("  custom events / fake merchant: " + string.Join(", ", services.Notes.Where(x => x.Key.StartsWith("custom_event") || x.Key.StartsWith("fake_merchant") || x.Key.Contains("fake_shop")).OrderBy(x => x.Key).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("  FAKE_* relics held at the runs' ends: " + fakeRelics);
        Console.WriteLine("  rest sites: " + string.Join(", ", services.Notes.Where(x => x.Key.StartsWith("rest_")).OrderBy(x => x.Key).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("  deck selections: " + string.Join(", ", services.Notes.Where(x => x.Key.StartsWith("deck_select")).OrderBy(x => x.Key).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("  run-level card selections: " + string.Join(", ", services.SelectionKinds.OrderByDescending(x => x.Value).Select(x => $"{x.Key} {x.Value}")));
        Console.WriteLine("  macro vocabulary: " + string.Join(", ", mvocab.Seen.Where(x => x.Value > 0).Select(x => $"{x.Key} unknown {mvocab.Unknown[x.Key]}/{x.Value}")));
        Console.WriteLine("  heal checks: " + string.Join(", ", services.Notes.Where(x => x.Key.StartsWith("heal_check")).Select(x => $"{x.Key} {x.Value}")));
        foreach (string e in errText.Take(10)) Console.WriteLine("  run error: " + e);
        // pass = no run error; whether evolve / restore / the hand-picked upgrade were taken is counted above (sampled
        // policy: a single process may miss one; src/ic113/PORT.md adds the counts of all processes)
        bool ok = errors == 0;
        Console.WriteLine(ok ? "EVENTCHECK PASSED" : "EVENTCHECK FAILED");
        return ok ? 0 : 1;
    }

    // ------------------------------------------------------------------ graph fingerprint

    /// <summary>Hash of every canonical model's object graph (ModelDb): a search must never write to them.</summary>
    public static string CanonicalHash()
    {
        var roots = new List<object>();
        roots.AddRange(ModelDb.All<CardModel>()); roots.AddRange(ModelDb.All<RelicModel>()); roots.AddRange(ModelDb.All<PotionModel>());
        roots.AddRange(ModelDb.All<PowerModel>()); roots.AddRange(ModelDb.All<MonsterModel>()); roots.AddRange(ModelDb.All<EnchantmentModel>());
        roots.AddRange(ModelDb.All<OrbModel>());
        return StateGraph.Hash(roots, walkCanonical: true).Hash;
    }

    private static int GraphSelfTest()
    {
        var a = new List<int> { 1, 2 };
        var (h0, _) = StateGraph.Hash(a); a[1] = 3; var (h1, _) = StateGraph.Hash(a);
        Console.WriteLine(h0 != h1 ? "graph self-test PASSED" : "graph self-test FAILED");
        return h0 != h1 ? 0 : 1;
    }
}

/// <summary>Order-exact fingerprint of an object graph (the idea of iRyougi's tests/Shared/RunCloneGraphSnapshot, MIT):
/// every instance field of every reachable object, collections in enumeration order, shared references by identity
/// number. Canonical models are written by id (their own graphs: <see cref="PortChecks.CanonicalHash"/>). Excluded:
/// runtime control (locks, lifecycle tasks, the card-selection source the search installs on clones, the combat's hook
/// listener cache, AbstractModel.ExecutionFinished handlers, RelicModel.Flashed), as in iRyougi's snapshot.</summary>
public static class StateGraph
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, FieldInfo[]> Fields = new();
    private static readonly Dictionary<string, string> LastText = new();

    public static (string Hash, long Objects) Hash(object root, bool walkCanonical = false, bool keepText = false)
    {
        var text = new StringBuilder(1 << 16);
        var ids = new Dictionary<object, int>(ReferenceEqualityComparer.Instance);
        var pending = new Queue<object>();
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        StringBuilder? full = keepText ? new StringBuilder() : null;
        void Flush(bool force)
        {
            if (!force && text.Length < 1 << 20) return;
            full?.Append(text);
            foreach (ReadOnlyMemory<char> chunk in text.GetChunks()) sha.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(chunk.Span));
            text.Clear();
        }
        if (walkCanonical && root is IEnumerable rootList)
            foreach (object? o in rootList) { if (o is not null && !ids.ContainsKey(o)) { ids[o] = ids.Count; pending.Enqueue(o); } }
        else Append(root);
        while (pending.TryDequeue(out object? item))
        {
            Type type = item.GetType();
            text.Append('\n').Append(ids[item]).Append(':').Append(type.FullName).Append('{');
            if (item is Delegate d)
                foreach (Delegate inv in d.GetInvocationList()) { text.Append(inv.Method.DeclaringType?.FullName).Append('.').Append(inv.Method.Name).Append(';'); Append(inv.Target); }
            else if (item is IEnumerable seq && item is not string)
            {
                foreach (object? v in seq) Append(v);
            }
            else
                foreach (FieldInfo f in InstanceFields(type))
                {
                    if (Excluded(f)) continue;
                    text.Append(f.Name).Append('=');
                    Append(f.GetValue(item));
                }
            text.Append('}');
            Flush(false);
        }
        Flush(true);
        string h = Convert.ToHexString(sha.GetHashAndReset());
        if (full is not null) lock (LastText) { if (LastText.Count >= 4) LastText.Clear(); LastText[h] = full.ToString(); }
        return (h, ids.Count);

        void Append(object? v)
        {
            if (v is null) { text.Append("null;"); return; }
            Type t = v.GetType();
            if (t.IsPrimitive || t.IsEnum || v is decimal or Guid or DateTime or TimeSpan)
            { text.Append(Convert.ToString(v, CultureInfo.InvariantCulture)).Append(';'); return; }
            if (v is string s) { text.Append('"').Append(s).Append("\";"); return; }
            if (v is Type mt) { text.Append("type:").Append(mt.FullName).Append(';'); return; }
            if (v is MemberInfo mi) { text.Append("member:").Append(mi.Name).Append(';'); return; }
            if (!walkCanonical && v is AbstractModel { IsCanonical: true } cm) { text.Append("canonical:").Append(cm.Id).Append(';'); return; }
            if (v is Task || v is IAsyncStateMachine || v is Thread || v is SemaphoreSlim || v is CancellationTokenSource || v is System.Diagnostics.Stopwatch)
            { text.Append("runtime:").Append(t.Name).Append(';'); return; }
            if (t.IsValueType)
            {
                text.Append('(');
                foreach (FieldInfo f in InstanceFields(t)) { text.Append(f.Name).Append('='); Append(f.GetValue(v)); }
                text.Append(");");
                return;
            }
            if (!ids.TryGetValue(v, out int id)) { id = ids.Count; ids[v] = id; pending.Enqueue(v); }
            text.Append('@').Append(id).Append(';');
        }
    }

    public static string Text(string hash) { lock (LastText) return LastText.TryGetValue(hash, out var t) ? t : ""; }

    /// <summary>For a mismatch report: re-fingerprints with the text kept and names the first differing line, when the
    /// earlier text is available (it is only kept for the hash of the last call with keepText; otherwise "n/a").</summary>
    public static string FirstDiff(object root, string oldHash)
    {
        string? old;
        lock (LastText) LastText.TryGetValue(oldHash, out old);
        if (old is null) return "first difference n/a (rerun with keepText)";
        var (h, _) = Hash(root, keepText: true);
        string now; lock (LastText) now = LastText[h];
        var a = old.Split('\n'); var b = now.Split('\n');
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            if (a[i] != b[i]) return $"line {i}: before {Trim(a[i])} | after {Trim(b[i])}";
        return $"lengths {a.Length} vs {b.Length}";
        static string Trim(string s) => s.Length > 300 ? s[..300] : s;
    }

    private static bool Excluded(FieldInfo f) =>
        f.Name is "_stateLock" or "_lifecycleLock" or "_rewardLock" or "_takeLock" or "_resolutionLock"
            or "_purchaseGate" or "_revealGate" or "_interactionGate" or "_insideLifecycle" or "_observer"
            or "_cardSelectionSource" or "<CardSelectionSource>k__BackingField"
            or "_activeChoiceTask" or "_activeChoiceOption" or "_activeChoicePageVersion"
            or "_takeTask" or "_resolutionTask" or "_outcomeTask" or "_exitTask" or "_completionTask"
        || f.DeclaringType == typeof(CombatState) && f.Name == "_hookListenerSnapshotCache"
        || f.DeclaringType == typeof(AbstractModel) && f.Name == "ExecutionFinished"
        || f.DeclaringType == typeof(RelicModel) && f.Name == "Flashed";

    private static FieldInfo[] InstanceFields(Type type) => Fields.GetOrAdd(type, static root =>
    {
        var fields = new List<FieldInfo>();
        for (Type? cur = root; cur is not null && cur != typeof(object); cur = cur.BaseType)
            fields.AddRange(cur.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly));
        return fields.OrderBy(f => f.DeclaringType?.FullName, StringComparer.Ordinal).ThenBy(f => f.Name, StringComparer.Ordinal).ToArray();
    });
}
