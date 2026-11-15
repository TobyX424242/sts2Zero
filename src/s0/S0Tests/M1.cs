using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.S0;

/// <summary>
/// M1 runner: plays battle starts to the end with <see cref="GumbelSearch"/> choosing every move.
/// usage: m1 &lt;battle_starts.jsonl&gt; &lt;first&gt; &lt;count&gt; &lt;sims&gt; &lt;hp|tf&gt; &lt;tf_tables.jsonl&gt; &lt;compare 0|1&gt; &lt;out.jsonl&gt; [seed]
/// One JSON line per fight: result, end HP, potions used, turns, decisions, wall time, simulations/s; with
/// compare=1 also, per decision, whether the other scorer's search (same seed, same state) picks a different action.
/// Fights are independent, so a large set is split over processes with first/count.
/// </summary>
static class M1Runner
{
    public static async Task<int> Run(string[] a)
    {
        string path = a[1]; int first = int.Parse(a[2]), count = int.Parse(a[3]), sims = int.Parse(a[4]);
        string scorerName = a[5], tfPath = a[6]; bool compare = a[7] == "1"; string outPath = a[8];
        ulong seed = a.Length > 9 ? ulong.Parse(a[9]) : 1;
        var tf = new TfScorer(tfPath);
        var hp = new HpScorer();
        ITerminalScorer main = scorerName == "tf" ? tf : hp, other = scorerName == "tf" ? hp : tf;
        var cfg = new SearchConfig { Simulations = sims };
        var specs = BattleStartSpec.ReadJsonl(path).Skip(first).Take(count).ToList();
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        using var w = new StreamWriter(outPath);
        long totalSims = 0; double totalSearch = 0; int won = 0, n = 0;
        foreach (var spec in specs)
        {
            var wall = Stopwatch.StartNew();
            var ctx = BattleContext.From(spec);
            var (state, simSeed, match) = await BattleStartLoader.LoadMatchedAsync(spec, $"M1-{spec.Id}");
            SearchCardSelectionSource.InstallOn(state, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(7)));
            var search = new GumbelSearch(cfg, main, ctx);
            var otherSearch = new GumbelSearch(cfg, other, ctx);
            int decisions = 0, potionsUsed = 0, disagree = 0, fightSims = 0; double searchSec = 0;
            var perDecision = new List<object>();
            string? error = null;
            try
            {
                while (!CombatActions.IsOver(state) && state.RoundNumber <= cfg.MaxRolloutTurns)
                {
                    ulong ds = ResamplingClone.Mix(seed, (ulong)decisions);
                    var r = await search.SearchAsync(state, ds);
                    fightSims += r.Simulations; searchSec += r.Seconds;
                    if (compare && r.Simulations > 0)
                    {
                        var o = await otherSearch.SearchAsync(state, ds);
                        bool diff = o.Action.Key != r.Action.Key;
                        if (diff) disagree++;
                        perDecision.Add(new { d = decisions, round = state.RoundNumber, main = r.Action.Key, other = o.Action.Key, differ = diff });
                    }
                    if (r.Action.Kind == ActionKind.UsePotion) potionsUsed++;
                    if (!await ActionSpace.ApplyAsync(state, r.Action)) { error = "chosen action not applicable: " + r.Action.Key; break; }
                    decisions++;
                }
            }
            catch (Exception e) { error = Util.Key(e); }
            wall.Stop();
            var pc = state.Players[0].Creature;
            bool over = CombatActions.IsOver(state);
            string result = error is not null ? "error" : !over ? "capped" : state.Engine!.Won ? "won" : "lost";
            if (result == "won") won++;
            n++; totalSims += fightSims; totalSearch += searchSec;
            var rec = new Dictionary<string, object?>
            {
                ["id"] = spec.Id, ["room_type"] = spec.RoomType, ["encounter"] = spec.EncounterId, ["monster_match"] = match.ToString(),
                ["scorer"] = main.Name, ["sims_per_move"] = sims, ["result"] = result, ["error"] = error,
                ["start_hp"] = spec.Hp, ["end_hp"] = pc.CurrentHp, ["max_hp"] = pc.MaxHp,
                ["potions_start"] = spec.Potions.Count, ["potions_used"] = potionsUsed,
                ["belt_held_end"] = ctx.HeldBeltIndices(state.Players[0]).Count,
                ["turns"] = state.RoundNumber, ["decisions"] = decisions, ["simulations"] = fightSims,
                ["wall_s"] = Math.Round(wall.Elapsed.TotalSeconds, 3), ["search_s"] = Math.Round(searchSec, 3),
                ["sims_per_s"] = searchSec > 0 ? Math.Round(fightSims / searchSec, 1) : 0,
                ["score_hp"] = Math.Round(hp.Score(state, ctx), 5),
                ["score_tf"] = tf.Has(spec.Id) ? Math.Round(tf.Score(state, ctx), 5) : null,
                ["disagreements"] = compare ? disagree : null, ["per_decision"] = compare ? perDecision : null,
            };
            w.WriteLine(JsonSerializer.Serialize(rec));
            w.Flush();
            Console.WriteLine($"{spec.Id} {spec.RoomType} {spec.EncounterId}: {result} hp {pc.CurrentHp}/{pc.MaxHp} (start {spec.Hp}), " +
                $"potions used {potionsUsed}/{spec.Potions.Count}, turns {state.RoundNumber}, decisions {decisions}, " +
                $"sims {fightSims}, {wall.Elapsed.TotalSeconds:F1} s, {(searchSec > 0 ? fightSims / searchSec : 0):F0} sims/s" +
                (compare ? $", tf/hp disagree {disagree}/{perDecision.Count}" : "") + (error is null ? "" : " ERROR " + error));
        }
        Console.WriteLine($"fights {n}, won {won}; simulations {totalSims} in {totalSearch:F1} s search time = {totalSims / Math.Max(1e-9, totalSearch):F0} sims/s (one process); T_f subset fallbacks {tf.SubsetFallbacks}");
        return 0;
    }
}
