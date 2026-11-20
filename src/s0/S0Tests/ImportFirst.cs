using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.S0;

/// <summary>
/// ic113 (2026-11-18): human battle starts (src/s2/export_starts.py JSONL, any character / build) imported into this
/// simulator through <see cref="BattleStartLoader.LoadMatchedAsync"/> and played to the player's first decision (combat
/// start hooks and the first draw done), one JSON line per start: the failure cause if the import or the start failed,
/// otherwise the fidelity of the imported state against the log (deck, relics, potions, max HP as multisets, with the
/// ids that are missing or extra; starting enemies against monster_ids) and whether the first decision has legal actions.
/// src/ic113/import_report.py aggregates the lines.
/// usage: S0Tests importfirst &lt;starts.jsonl&gt; &lt;out.jsonl&gt; [sim seed prefix (IF)]
/// </summary>
static class ImportFirst
{
    public static async Task<int> Run(string[] args)
    {
        string path = args[1], outPath = args[2], seed = args.Length > 3 ? args[3] : "IF";
        var sw = Stopwatch.StartNew();
        int n = 0, ok = 0;
        using var w = new StreamWriter(outPath);
        int i = 0;
        foreach (var spec in BattleStartSpec.ReadJsonl(path))
        {
            var rec = new Dictionary<string, object?> { ["id"] = spec.Id, ["build"] = spec.BuildId, ["asc"] = spec.Ascension, ["act"] = spec.ActIndex,
                ["room"] = spec.RoomType, ["encounter"] = spec.EncounterId, ["character"] = spec.Character };
            n++;
            try
            {
                var (s, _, match) = await BattleStartLoader.LoadMatchedAsync(spec, $"{seed}-{i}");
                rec["match"] = match.ToString();
                var pl = s.Players[0];
                var checks = new List<string>();
                void Cmp(string what, IEnumerable<string> want, IEnumerable<string> have)
                {
                    var left = have.ToList(); var missing = new List<string>();
                    foreach (string x in want) if (!left.Remove(x)) missing.Add(x);
                    if (missing.Count == 0 && left.Count == 0) return;
                    checks.Add(what);
                    rec[what + "_missing"] = missing; rec[what + "_extra"] = left;
                }
                Cmp("deck", spec.Deck.Select(c => $"{c.Id}+{c.Upgrade}~{c.Enchantment}"),
                    pl.Deck.Cards.Select(c => $"{c.Id}+{c.CurrentUpgradeLevel}~{c.Enchantments.SingleOrDefault()?.Id.ToString()}"));
                Cmp("relics", spec.Relics, pl.Relics.Select(r => r.Id.ToString()));
                Cmp("potions", spec.Potions, pl.PotionSlots.Where(x => x is not null).Select(x => x!.Id.ToString()));
                if (pl.Creature.MaxHp != spec.MaxHp) { checks.Add("max_hp"); rec["max_hp_sim"] = pl.Creature.MaxHp; rec["max_hp_log"] = spec.MaxHp; }
                if (pl.Creature.CurrentHp != Math.Min(spec.Hp, pl.Creature.MaxHp)) checks.Add("hp_at_first_decision");
                rec["checks"] = checks;
                if (CombatActions.IsOver(s)) rec["cause"] = "first decision: combat already over";
                else if (s.CurrentSide != CombatSide.Player) rec["cause"] = "first decision: not the player's turn";
                else if (CombatActions.Legal(s, pl).Count == 0) rec["cause"] = "first decision: no legal action";
                else { rec["ok"] = true; ok++; }
            }
            catch (BattleImportException e)
            {
                rec["cause"] = "import: " + e.Cause;
                rec["detail"] = Short(e.Message + (e.InnerException is { } x ? " | " + x.GetType().Name + ": " + x.Message : ""));
            }
            catch (Exception e)
            {
                rec["cause"] = "exception: " + e.GetType().Name;
                rec["detail"] = Short(e.Message + " @ " + (e.StackTrace?.Split('\n').FirstOrDefault()?.Trim() ?? ""));
            }
            w.WriteLine(JsonSerializer.Serialize(rec));
            i++;
        }
        Console.WriteLine($"importfirst {path}: {n} starts, {ok} at a playable first decision ({100.0 * ok / Math.Max(1, n):F2}%), wall {sw.Elapsed.TotalSeconds:F0} s");
        return 0;
    }

    static string Short(string m) { m = System.Text.RegularExpressions.Regex.Replace(m, @"\[\d+\]", "[i]"); return m.Length > 300 ? m[..300] : m; }
}
