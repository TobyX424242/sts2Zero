using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Models.Monsters;
using Sts2Sim.Core.Models.Powers;
using Sts2Sim.S0;
using Sts2Sim.S2;

namespace Sts2Sim.S3;

/// <summary>
/// <c>S3 prompttest</c>: checks of the end-turn prompt expansion at the search root (<c>endturn_prompts</c>) on a
/// Knowledge Demon boss fight (starter Defect deck) whose first move is Curse of Knowledge (choose Disintegration or
/// Mind Rot during the enemy turn):
/// <list type="bullet">
/// <item>root list with the option: end turn split into the two answers, last in the list, the other actions and the
/// live state unchanged, reproducible; without it: the old list;</item>
/// <item>each end-turn answer applied to an exact and to a resampled clone gives the chosen card's power and not the
/// other's, with no key miss; an answer that matches nothing falls back to the selection policy;</item>
/// <item>the search (uniform network) runs with these root actions and gives them policy-target entries; the action
/// rows carry the answer as a card action's would;</item>
/// <item>a state whose end turn opens no prompt (round 2, next move Slap) keeps the plain end turn.</item>
/// </list>
/// Prints PASS/FAIL lines; returns 1 if any check failed.
/// </summary>
public static class PromptTest
{
    public static async Task<int> Run(string vocabPath)
    {
        int fails = 0;
        void Check(string what, bool ok, string detail = "")
        {
            if (!ok) fails++;
            Console.WriteLine($"{(ok ? "PASS" : "FAIL")} {what}{(detail.Length > 0 ? " (" + detail + ")" : "")}");
        }
        static string Keys(IEnumerable<SearchAction> l) => string.Join("  ", l.Select(a => a.Key));
        static string Norm(SearchAction a) => string.Join("&", a.Selections.SelectMany(s => s)).Replace("_", "").ToUpperInvariant();
        static bool IsEnd(SearchAction a) => a.Kind == ActionKind.EndTurn;

        BattleStartSpec spec = KnowledgeDemonSpec();
        CombatState s0 = await BattleStartLoader.LoadAsync(spec, "PROMPTTEST-1");
        Creature demon = s0.Enemies[0];
        string? move = demon.Monster?.NextMove?.StateId;
        Check("Knowledge Demon fight, next move CURSE_OF_KNOWLEDGE_MOVE", demon.Monster is KnowledgeDemon && move == "CURSE_OF_KNOWLEDGE_MOVE",
            $"{demon.Monster?.Id} {move}, round {s0.RoundNumber}");
        ulong h0 = ObservableState.Hash(s0);

        // ---- root lists
        List<SearchAction> old = await ActionSpace.LegalAsync(s0);
        List<SearchAction> off = await ActionSpace.LegalAsync(s0, endTurnPrompts: false);
        List<SearchAction> on = await ActionSpace.LegalAsync(s0, endTurnPrompts: true);
        List<SearchAction> on2 = await ActionSpace.LegalAsync(s0, endTurnPrompts: true);
        Console.WriteLine("old list:              " + Keys(old));
        Console.WriteLine("endturn_prompts=0:     " + Keys(off));
        Console.WriteLine("endturn_prompts=1:     " + Keys(on));
        Check("endturn_prompts=0 gives the old list", Keys(off) == Keys(old));
        Check("old list ends with the plain end turn, one end turn", old.Count > 0 && old[^1].Key == "E" && old.Count(IsEnd) == 1);
        var ends = on.Where(IsEnd).ToList();
        Check("end turn split into 2 answers of one prompt", ends.Count == 2 && ends.All(e => e.Selections.Count == 1) && ends.Select(e => e.Key).Distinct().Count() == 2,
            string.Join(", ", ends.Select(e => e.Key)));
        Check("end-turn answers are last", on.Skip(on.Count - ends.Count).All(IsEnd));
        Check("answers are Disintegration and Mind Rot", ends.Any(e => Norm(e).Contains("DISINTEGRATION")) && ends.Any(e => Norm(e).Contains("MINDROT")));
        Check("card and potion actions unchanged", Keys(on.Where(a => !IsEnd(a))) == Keys(old.Where(a => !IsEnd(a))));
        Check("expansion reproducible for the same state", Keys(on) == Keys(on2));
        Check("probing left the state unchanged", ObservableState.Hash(s0) == h0);

        // ---- applying each answer (exact clone = this world; resampled clone = another world of the same information set)
        CombatState? afterTurn = null;
        foreach (SearchAction e in ends)
        {
            bool wantDis = Norm(e).Contains("DISINTEGRATION");
            foreach (var (name, world) in new[] { ("exact clone", s0.Clone()), ("resampled clone", ResamplingClone.Resampled(s0, 0xC0FFEEUL)) })
            {
                bool ok = await ActionSpace.ApplyAsync(world, e);
                var src = world.CardSelectionSource as SearchCardSelectionSource;
                Creature pc = world.Players[0].Creature;
                bool dis = pc.HasPower<DisintegrationPower>(), rot = pc.HasPower<MindRotPower>();
                Check($"{e.Key} on {name}: chosen card applied, the other not, no key miss",
                    ok && dis == wantDis && rot == !wantDis && src is { KeyMisses: 0, PromptsSeen: >= 1 },
                    $"applied {ok}, DisintegrationPower {dis}, MindRotPower {rot}, key misses {src?.KeyMisses}, prompts {src?.PromptsSeen}, round {world.RoundNumber}");
                if (wantDis && name == "exact clone") afterTurn = world;
            }
        }
        {
            // graceful fallback: an answer that matches no candidate goes to the selection policy
            SearchAction bogus = SearchAction.End.WithSelection(new[] { "CARD.NO_SUCH_CARD#0#0##" });
            CombatState world = s0.Clone();
            bool ok = await ActionSpace.ApplyAsync(world, bogus);
            var src = world.CardSelectionSource as SearchCardSelectionSource;
            Creature pc = world.Players[0].Creature;
            Check("unmatched end-turn answer: key miss counted, prompt answered by the policy",
                ok && src is { KeyMisses: 1 } && pc.HasPower<DisintegrationPower>() != pc.HasPower<MindRotPower>(),
                $"applied {ok}, key misses {src?.KeyMisses}");
        }

        // ---- the search with these root actions (uniform network, terminal score = HP share)
        foreach (bool flag in new[] { true, false })
        {
            var cfg = new NetSearchConfig { EndTurnPrompts = flag };
            var search = new NetSearch(cfg, new UniformEvaluator(), new HpScorer(), BattleContext.From(spec), 1.0);
            NetSearchResult r = await search.SearchAsync(s0, 48, 7);
            string want = Keys(flag ? NetSearch.CapRoot(on) : Actions.Cap(old));
            Check($"search with endturn_prompts={(flag ? 1 : 0)}: root actions as listed, searched, policy target over them",
                Keys(r.RootActions) == want && r.Simulations > 0 && r.PolicyTarget?.Length == r.RootActions.Count,
                $"chosen {r.Action.Key}, sims {r.Simulations}, end-turn target mass {(r.PolicyTarget is null ? 0 : r.RootActions.Select((a, i) => IsEnd(a) ? r.PolicyTarget[i] : 0).Sum()):F3}");
        }

        // ---- action rows: an end-turn answer is encoded like a card action's (source token 0, prompt code, chosen card)
        if (File.Exists(vocabPath))
        {
            var vocab = ObsLayout.Vocab.Load(vocabPath);
            int aw = ObsLayout.MaxActions * ObsLayout.ActionWidth;
            var obs = new int[ObsLayout.MaxTokens * ObsLayout.TokenWidth]; var act = new int[aw];
            List<SearchAction> rows = NetSearch.CapRoot(on);
            ObsLayout.Write(Encoding.Tokens(s0, null), rows, s0, vocab, obs, act);
            bool rowsOk = true;
            for (int j = 0; j < rows.Count; j++)
            {
                if (!IsEnd(rows[j])) continue;
                int[] row = act.Skip(j * ObsLayout.ActionWidth).Take(ObsLayout.ActionWidth).ToArray();
                var (cid, up) = ObsLayout.ParseSelectionKey(rows[j].Selections[0][0]);
                bool ok = row[0] == 1 && row[1] == 0 && row[2] == -1 && row[3] == 2 && row[4] == vocab.Id("card", cid) && row[5] == up + 1;
                Console.WriteLine($"  row {j} {rows[j].Key}: [{string.Join(",", row.Take(6))}]");
                rowsOk &= ok;
            }
            Check("end-turn answer rows: kind end turn, source 0, no target, one chosen card", rowsOk);
        }
        else Console.WriteLine($"(action-row check skipped: no vocabulary at {vocabPath})");

        // ---- a state without end-turn prompt: round 2, next move Slap
        if (afterTurn is null) Check("round-2 state available", false);
        else
        {
            string? move2 = afterTurn.Enemies[0].Monster?.NextMove?.StateId;
            List<SearchAction> old2 = await ActionSpace.LegalAsync(afterTurn);
            List<SearchAction> on2b = await ActionSpace.LegalAsync(afterTurn, endTurnPrompts: true);
            Console.WriteLine($"round {afterTurn.RoundNumber}, next move {move2}");
            Console.WriteLine("old list:              " + Keys(old2));
            Console.WriteLine("endturn_prompts=1:     " + Keys(on2b));
            Check("no end-turn prompt: next move is not the curse", move2 is not null && move2 != "CURSE_OF_KNOWLEDGE_MOVE" && !CombatActions.IsOver(afterTurn), move2 ?? "none");
            Check("no end-turn prompt: plain end turn kept, list unchanged", Keys(on2b) == Keys(old2) && on2b.Count(IsEnd) == 1 && on2b[^1].Key == "E");
        }

        Console.WriteLine($"end-turn probes {ActionSpace.EndTurnProbes} (with prompt {ActionSpace.EndTurnProbesWithPrompt}, failed {ActionSpace.EndTurnProbeErrors})");
        Console.WriteLine(fails == 0 ? "ALL PASS" : $"{fails} checks FAILED");
        return fails == 0 ? 0 : 1;
    }

    /// <summary>Act-2 boss Knowledge Demon, starter Defect deck (4 Strike, 4 Defend, Zap, Dualcast; Cracked Core).</summary>
    private static BattleStartSpec KnowledgeDemonSpec()
    {
        var deck = new List<BattleCard>();
        for (int i = 0; i < 4; i++) deck.Add(new BattleCard("CARD.STRIKE_DEFECT", 0, null, 0));
        for (int i = 0; i < 4; i++) deck.Add(new BattleCard("CARD.DEFEND_DEFECT", 0, null, 0));
        deck.Add(new BattleCard("CARD.ZAP", 0, null, 0));
        deck.Add(new BattleCard("CARD.DUALCAST", 0, null, 0));
        return new BattleStartSpec
        {
            Id = "prompttest:kd", Ascension = 0, ActIds = new() { "ACT.UNDERDOCKS", "ACT.HIVE", "ACT.GLORY" }, ActIndex = 1, Floor = 33,
            RoomType = "boss", EncounterId = "ENCOUNTER.KNOWLEDGE_DEMON_BOSS", MonsterIds = new() { "MONSTER.KNOWLEDGE_DEMON" },
            Hp = 70, MaxHp = 75, Gold = 0, Deck = deck, Relics = new() { "RELIC.CRACKED_CORE" }, Potions = new(), PotionSlots = 3,
        };
    }
}
