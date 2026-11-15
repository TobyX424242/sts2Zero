using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Monsters;
using Sts2Sim.Core.Models.Powers;
using Sts2Sim.S0;
using WitherCard = Sts2Sim.Core.Models.Cards.Wither;
using ClawCard = Sts2Sim.Core.Models.Cards.Claw;

/// <summary>
/// obsv4 [seed=OBSV4] [maxActions=400] [verbose=1]: layout v4 check on an Aeonglass boss fight (A10, Defect with 8 Claws,
/// 4 Strikes, 4 Defends, 2 Zaps, 999 HP). Plays cards at random (end turn when nothing is playable) and checks, at every
/// decision, with expectations computed independently of the encoder:
///  - the WitheringPresence token (EnemyPower) f2 = cards left before the next Wither: 6 at the start, minus one per card
///    played, back to 6 when it reaches 0, and exactly then one more Wither exists; unchanged by the end of a turn;
///  - every Wither token's damage (hand f5, draw f4, discard/exhaust f3) = 3 + 3 x Aeonglass.WitherUpgradeCount;
///  - Claw tokens' damage = 3 + 2 x Claws played; Strike damage 6, Defend block 5 (hand f6, draw f5, discard f4);
///  - the extended fingerprint carries the counter and card values, the old one (Extended = false) does not.
/// Prints the Wither / WitheringPresence tokens, then PASS or FAIL (exit code 1).
/// </summary>
static class ObsV4Test
{
    public static async Task<int> Run(string[] args)
    {
        string seed = args.Length > 1 ? args[1] : "OBSV4";
        int maxActions = args.Length > 2 ? int.Parse(args[2]) : 400;
        bool verbose = args.Length <= 3 || args[3] != "0";
        var deck = new List<BattleCard>();
        void Add(string id, int n) { for (int i = 0; i < n; i++) deck.Add(new BattleCard(id, 0, null, 0)); }
        Add("CARD.CLAW", 8); Add("CARD.STRIKE_DEFECT", 4); Add("CARD.DEFEND_DEFECT", 4); Add("CARD.ZAP", 2);
        var spec = new BattleStartSpec
        {
            Id = "obsv4", Ascension = 10, ActIds = ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"], ActIndex = 2, Floor = 48,
            RoomType = "boss", EncounterId = "ENCOUNTER.AEONGLASS_BOSS", MonsterIds = ["MONSTER.AEONGLASS"],
            Hp = 999, MaxHp = 999, Gold = 0, Deck = deck, PotionSlots = 3,
        };
        CombatState s = await BattleStartLoader.LoadAsync(spec, seed);
        try { SearchCardSelectionSource.InstallOn(s, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(5))); }
        catch (InvalidOperationException) { }
        var rng = new Random(17);
        var fails = new List<string>();
        void Check(bool ok, string what) { if (!ok && fails.Count < 50) fails.Add(what); }

        Aeonglass? boss = s.Enemies.Select(e => e.Monster).OfType<Aeonglass>().FirstOrDefault();
        if (boss is null) { Console.WriteLine("FAIL: no Aeonglass in the fight"); return 1; }
        int bossSlot = s.Enemies.ToList().FindIndex(e => e.Monster is Aeonglass);
        int amount = PresencePower(s)?.Amount ?? -1;
        Check(amount == 6, $"WitheringPresence amount {amount}, expected 6");
        int expectLeft = amount, clawsPlayed = 0, plays = 0, wraps = 0, withersChecked = 0, upgradedWithersChecked = 0, endTurns = 0;
        int lastWithers = CountWithers(s);
        bool fpChecked = false;
        for (int step = 0; step < maxActions && !CombatActions.IsOver(s); step++)
        {
            var tokens = CombatObservation.Encode(s, ObsLayout.MaxTokens);
            // WitheringPresence token
            var presence = PresencePower(s);
            if (presence is null) { Check(false, $"step {step}: WitheringPresence missing"); break; }
            var pTok = tokens.Where(t => t.Kind == TokenKind.EnemyPower && t.Id == presence.Id.ToString() && t.F[0] == bossSlot).ToList();
            Check(pTok.Count == 1, $"step {step}: {pTok.Count} WitheringPresence tokens");
            if (pTok.Count == 1)
            {
                Check(pTok[0].F[1] == presence.Amount, $"step {step}: presence amount feature {pTok[0].F[1]} != {presence.Amount}");
                Check(pTok[0].F[2] == expectLeft, $"step {step}: presence counter feature {pTok[0].F[2]}, expected {expectLeft}");
            }
            // card values
            int witherDmg = 3 + 3 * boss.WitherUpgradeCount, clawDmg = 3 + 2 * clawsPlayed;
            var pcs = s.Players[0].PlayerCombatState!;
            foreach (var t in tokens)
            {
                int dSlot = t.Kind switch { TokenKind.Hand => 5, TokenKind.Draw => 4, TokenKind.Discard or TokenKind.Exhaust => 3, _ => -1 };
                if (dSlot < 0) continue;
                int dmg = t.F[dSlot], blk = t.F[dSlot + 1];
                if (t.Kind == TokenKind.Hand)
                    Check(pcs.Hand.Cards[t.F[0]].Id.ToString() == t.Id, $"step {step}: hand token {t.F[0]} is not {t.Id}");
                switch (t.Id)
                {
                    case "CARD.WITHER":
                        Check(dmg == witherDmg && blk == 0, $"step {step}: {t.Kind} Wither damage {dmg} block {blk}, expected {witherDmg} 0");
                        withersChecked++;
                        if (boss.WitherUpgradeCount > 0) upgradedWithersChecked++;
                        if (verbose && (t.Kind is TokenKind.Hand or TokenKind.Draw))
                            Console.WriteLine($"  step {step} round {s.RoundNumber}: {t.Kind} Wither [{string.Join(",", t.F)}] (upgrades {boss.WitherUpgradeCount})");
                        break;
                    case "CARD.CLAW": Check(dmg == clawDmg && blk == 0, $"step {step}: {t.Kind} Claw damage {dmg} block {blk}, expected {clawDmg} 0"); break;
                    case "CARD.STRIKE_DEFECT": Check(dmg == 6 && blk == 0, $"step {step}: {t.Kind} Strike damage {dmg} block {blk}"); break;
                    case "CARD.DEFEND_DEFECT": Check(dmg == 0 && blk == 5, $"step {step}: {t.Kind} Defend damage {dmg} block {blk}"); break;
                }
            }
            // fingerprint: extended carries the counter and card values; the old form does not
            if (!fpChecked && pcs.Hand.Cards.Count > 0)
            {
                bool saved = ObservableState.Extended;
                ObservableState.Extended = true; string fpNew = ObservableState.Fingerprint(s);
                ObservableState.Extended = false; string fpOld = ObservableState.Fingerprint(s);
                ObservableState.Extended = saved;
                Check(fpNew.Contains($"{presence.Id.Entry}:{presence.Amount}^{expectLeft}"), "extended fingerprint lacks the WitheringPresence counter");
                Check(!fpOld.Contains('^') && fpOld != fpNew, "old fingerprint carries extended fields");
                Check(fpOld.Contains($"{presence.Id.Entry}:{presence.Amount}") && !fpOld.Contains("=0/"), "old fingerprint form changed");
                fpChecked = true;
            }
            if (verbose)
                Console.WriteLine($"step {step} round {s.RoundNumber}: WitheringPresence token [{(pTok.Count > 0 ? string.Join(",", pTok[0].F) : "-")}]," +
                                  $" expected counter {expectLeft}; Withers {CountWithers(s)}");

            // act: a random playable card, else end turn
            var legal = CombatActions.Legal(s, s.Players[0]);
            var cardsLegal = legal.Where(a => !a.EndTurn).ToList();
            CombatAction act = cardsLegal.Count > 0 && rng.NextDouble() > 0.05 ? cardsLegal[rng.Next(cardsLegal.Count)] : CombatAction.End;
            bool isClaw = !act.EndTurn && pcs.Hand.Cards[act.HandIndex] is ClawCard;
            int before = CountWithers(s);
            await CombatActions.ApplyAsync(s, act);
            if (CombatActions.IsOver(s)) break;
            int after = CountWithers(s);
            if (act.EndTurn)
            {
                endTurns++;
                // enemy turn: Increasing Intensity adds Withers to the discard pile; the counter is not touched
            }
            else
            {
                plays++;
                if (isClaw) clawsPlayed++;
                expectLeft--;
                bool wrapped = expectLeft == 0;
                if (wrapped) { expectLeft = amount; wraps++; }
                Check(after - before == (wrapped ? 1 : 0), $"step {step}: Withers {before} -> {after} after a card play (counter wrapped {wrapped})");
            }
            lastWithers = after;
        }
        Console.WriteLine($"cards played {plays} (Claws {clawsPlayed}), end turns {endTurns}, Withers created by the counter {wraps}, " +
                          $"Wither tokens checked {withersChecked} (after an upgrade {upgradedWithersChecked}), Withers now {lastWithers}, " +
                          $"rounds {s.RoundNumber}, fight over {CombatActions.IsOver(s)}");
        Check(wraps >= 2, $"only {wraps} counter wraps: increase maxActions");
        Check(upgradedWithersChecked > 0, "no Wither token seen after Increasing Intensity: increase maxActions");
        foreach (string f in fails) Console.WriteLine("  " + f);
        Console.WriteLine(fails.Count == 0 ? "PASS" : $"FAIL ({fails.Count} problems)");
        return fails.Count == 0 ? 0 : 1;
    }

    private static WitheringPresencePower? PresencePower(CombatState s) =>
        s.Enemies.SelectMany(e => e.Powers).OfType<WitheringPresencePower>().FirstOrDefault();

    private static int CountWithers(CombatState s)
    {
        var pcs = s.Players[0].PlayerCombatState!;
        return pcs.Hand.Cards.Concat(pcs.DrawPile.Cards).Concat(pcs.DiscardPile.Cards).Concat(pcs.ExhaustPile.Cards).Count(c => c is WitherCard);
    }
}
