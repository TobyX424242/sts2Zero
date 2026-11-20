using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S2;

namespace Sts2Sim.S3;

/// <summary>
/// ic113 combat warm-up (2026-11-18, src/ic113/WARMUP.md): T_f tables of human battle starts priced by the S3 macro
/// network, in the S2 tables format, so the S2 fight harness and S2 self-play score these fights as S3 scores live ones.
///   dotnet S3.dll tables starts=F out=F mbridge=PATH [shard=i/N] [concurrent=64] [tf_heal=1] [era=6] [check=1]
/// Per start: the run is imported into the simulator (BattleStartLoader.BuildRun; with check=1 the fight is also started
/// to the player's first decision, and a start that fails either is dropped and counted by cause), the post-fight row is
/// built by the live code (MacroEncoder.PostFight + PostCombatHeal, as RunAgent.StartFightAsync), and the run-state inputs
/// an injected run lacks are taken from the start's "macro_ctx" (src/s2/export_starts.py --macro-ctx): room kind
/// (combat), floor in act, the act's elites / rest sites / shops, the act's boss(es) when the run reached them (else the
/// simulator's draw), the last 8 fights, card and relic ages. Not available: the map (no map tokens) and the map-derived
/// route inputs (map options only). Final boss (the second act-3 boss at A10, the act-3 boss below A10): table 1 at every
/// HP, as live. Death = 0 (death_value=zero, the s3v3 setting).
/// </summary>
public static class HumanTables
{
    /// <summary>Starts whose card or relic order could not be matched to the import (ages left at the import's); starts
    /// whose act boss was not reached in the human run (the simulator's boss of the act is encoded).</summary>
    public static int AgeMismatch, BossFromSim;

    public static int Run(Dictionary<string, string> kv)
    {
        string Get(string k, string d) => kv.TryGetValue(k, out var v) ? v : d;
        int GetI(string k, int d) => int.Parse(Get(k, d.ToString()));
        string[] shard = Get("shard", "0/1").Split('/');
        int si = int.Parse(shard[0]), sn = int.Parse(shard[1]);
        bool tfHeal = GetI("tf_heal", 1) == 1, check = GetI("check", 1) == 1;
        ML.Era = GetI("era", ML.EraV0113);
        var mvocab = new MacroVocab(Get("mvocab", "src/s3/data/macro_vocab_v113.json"));
        var lines = File.ReadLines(Get("starts", "")).Where(l => !string.IsNullOrWhiteSpace(l)).Where((l, i) => i % sn == si).ToList();
        string mb = Get("mbridge", "uniform");
        MacroBridgeEvaluator? bridge = mb == "uniform" ? null : new MacroBridgeEvaluator(mb);
        IMacroEvaluator macro = bridge is null ? new UniformMacroEvaluator() : bridge;
        string outPath = Get("out", "");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        using var w = new StreamWriter(outPath);
        var fails = new Dictionary<string, int>();
        int next = 0, written = 0, finals = 0, noCtx = 0, healing = 0;
        AgeMismatch = 0; BossFromSim = 0;
        long rows = 0;
        var sw = Stopwatch.StartNew();

        async Task Worker()
        {
            while (true)
            {
                int k = next++;
                if (k >= lines.Count) return;
                BattleStartSpec spec;
                JsonElement ctx;
                try
                {
                    using var doc = JsonDocument.Parse(lines[k]);
                    spec = JsonSerializer.Deserialize<BattleStartSpec>(lines[k])!;
                    if (!doc.RootElement.TryGetProperty("macro_ctx", out var c)) { noCtx++; continue; }
                    ctx = c.Clone();
                }
                catch (Exception e) { Fail("parse:" + e.GetType().Name); continue; }
                RunState run;
                try
                {
                    run = BattleStartLoader.BuildRun(spec, "TF-" + spec.Id);
                    if (check) await BattleStartLoader.LoadAsync(spec, "TFCHK-" + spec.Id);
                }
                catch (BattleImportException e) { Fail(e.Cause); continue; }
                catch (Exception e) { Fail("load:" + e.GetType().Name); continue; }
                try
                {
                    var p = run.Players[0];
                    int maxHp = Math.Max(1, p.Creature.MaxHp);
                    RoomType rt = BattleStartLoader.ParseRoomType(spec.RoomType);
                    bool final = ctx.GetProperty("final").GetBoolean();
                    var belt = spec.Potions.ToList();
                    int nHeld = Math.Min(belt.Count, Outcome.Slots);
                    var subsets = new List<int[]>();
                    for (int size = 0; size <= nHeld; size++)
                        for (int m = 0; m < 1 << nHeld; m++)
                            if (System.Numerics.BitOperations.PopCount((uint)m) == size)
                                subsets.Add(Enumerable.Range(0, nHeld).Where(j => (m >> j & 1) == 1).ToArray());
                    var tables = new double[subsets.Count][];
                    if (final)
                    {
                        for (int s = 0; s < subsets.Count; s++)
                        {
                            tables[s] = new double[maxHp + 1];
                            for (int h = 1; h <= maxHp; h++) tables[s][h] = 1.0;
                        }
                        finals++;
                    }
                    else
                    {
                        MacroRow post = PostRow(run, p, spec, ctx, mvocab, rt);
                        var heal = tfHeal ? PostCombatHeal.Table(p, maxHp) : PostCombatHeal.Identity(maxHp);
                        if (heal.Healing) healing++;
                        int postMax = heal.MaxHpAfter;
                        post.SetHp(postMax, postMax);
                        var tasks = new Task<MacroEval>[subsets.Count];
                        for (int s = 0; s < subsets.Count; s++)
                        {
                            MacroRow r = post.Clone();
                            r.SetPotions(subsets[s].Select(i => mvocab.Tok("potion", belt[i])).Take(ML.MaxPot).ToList());
                            tasks[s] = macro.EvaluateAsync(r);
                        }
                        MacroEval[] evs = await Task.WhenAll(tasks);
                        rows += subsets.Count;
                        for (int s = 0; s < subsets.Count; s++)
                        {
                            tables[s] = new double[maxHp + 1];
                            if (ML.HasMilestones(evs[s].Out)) throw new NotSupportedException("milestone heads are not supported here");
                            for (int h = 1; h <= maxHp; h++) tables[s][h] = ML.Sigmoid(ML.LogitAt(evs[s].Out, heal.Hp[h], postMax));
                        }
                    }
                    var d = new Dictionary<string, object?>
                    {
                        ["id"] = spec.Id, ["max_hp"] = maxHp, ["belt"] = belt.Take(nHeld).ToList(),
                        ["subsets"] = subsets, ["tables"] = tables.Select(t => t.Select(x => Math.Round(x, 6)).ToArray()).ToArray(),
                        ["final_boss"] = final, ["first_of_two"] = ctx.GetProperty("first_of_two").GetBoolean(),
                    };
                    w.WriteLine(JsonSerializer.Serialize(d));
                    written++;
                    if (written % 5000 == 0) Console.WriteLine($"[tables {si}/{sn}] {written} written, {rows} rows, {sw.Elapsed.TotalSeconds:F0}s");
                }
                catch (Exception e) { Fail("table:" + e.GetType().Name + ":" + e.Message[..Math.Min(80, e.Message.Length)]); }
            }
        }

        void Fail(string cause) => fails[cause] = fails.GetValueOrDefault(cause) + 1;

        bool Flush()
        {
            if (bridge is null) return false;
            bool did = false;
            if (bridge.InFlight && bridge.Ready) { bridge.Complete(); did = true; }
            if (!bridge.InFlight) bridge.Submit();
            if (!did && bridge.InFlight) { bridge.Complete(); did = true; }
            return did;
        }

        Pump.Run(() => Task.WhenAll(Enumerable.Range(0, GetI("concurrent", 64)).Select(_ => Worker())), Flush);
        w.Flush();
        Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["shard"] = $"{si}/{sn}", ["starts"] = lines.Count, ["written"] = written, ["final_boss"] = finals, ["rows"] = rows,
            ["no_ctx"] = noCtx, ["failed"] = fails, ["age_mismatch"] = AgeMismatch, ["boss_from_sim"] = BossFromSim,
            ["healing_tables"] = healing, ["seconds"] = Math.Round(sw.Elapsed.TotalSeconds),
            ["unknown_macro_ids"] = mvocab.UnknownIds.Take(40).ToList(),
        }));
        if (Get("shutdown", "0") == "1") bridge?.Shutdown();
        return 0;
    }

    /// <summary>The post-fight row of the start: the live encoder on the imported run, completed from macro_ctx.</summary>
    public static MacroRow PostRow(RunState run, Sts2Sim.Core.Entities.Players.Player p, BattleStartSpec spec, JsonElement ctx,
        MacroVocab v, RoomType rt)
    {
        int g = spec.Floor, act = spec.ActIndex;
        var facts = new FloorFacts { Ascension = spec.Ascension, FloorModelId = spec.EncounterId, FloorModelIdFloor = run.TotalFloor };
        // relic ages: relic floors in the start's relic order (the importer keeps it; checked by id)
        var rf = ctx.GetProperty("relic_floors").EnumerateArray().Select(x => x.GetInt32()).ToList();
        bool relOk = p.Relics.Count == spec.Relics.Count && rf.Count == spec.Relics.Count
            && p.Relics.Select(r => r.Id.ToString()).SequenceEqual(spec.Relics);
        if (relOk) for (int j = 0; j < p.Relics.Count; j++) facts.RelicFloor[p.Relics[j]] = Math.Max(1, rf[j]);
        facts.CountAct = run.CurrentActIndex;
        facts.ActElites = ctx.GetProperty("elites").GetInt32(); facts.ActRests = ctx.GetProperty("rests").GetInt32();
        facts.ActShops = ctx.GetProperty("shops").GetInt32();
        foreach (var f in ctx.GetProperty("fights").EnumerateArray())
        {
            var a = f.EnumerateArray().ToArray();
            string enc = a[0].GetString() ?? "";
            facts.Fights.Add(new FightNote(v.Tok("encounter", enc), enc, a[1].GetInt32(), a[2].GetInt32(), a[3].GetInt32(),
                a[4].GetInt32(), a[5].GetInt32(), a[6].GetInt32(), a[7].GetInt32(), a[8].GetInt32()));
        }
        MacroRow r = MacroEncoder.PostFight(run, p, facts, v, rt);
        // what the injected run cannot know: a combat map point of this room kind, the floor in the act, the floor's own
        // HP-free context as a live fight on floor g has it (FillFights / FillV2 read run.TotalFloor and the act index)
        var pt = rt switch { RoomType.Elite => MapPointType.Elite, RoomType.Boss => MapPointType.Boss, _ => MapPointType.Monster };
        r.I[ML.OCtx + 4] = ML.RoomIndex(pt);
        r.I[ML.OCtx + 5] = v.Tok("encounter", spec.EncounterId);
        r.F[ML.SIsCombat] = 1f;
        r.F[ML.SFloor] = g / 50f; r.F[ML.SAct] = act / 3f;
        r.F[ML.SFloorInAct] = ctx.GetProperty("floor_in_act").GetInt32() / 20f;
        string boss = ctx.GetProperty("boss").GetString() ?? "", boss2 = ctx.GetProperty("boss2").GetString() ?? "";
        if (boss != "") r.I[ML.OBoss] = v.Tok("encounter", boss); else BossFromSim++;
        if (boss2 != "") r.I[ML.OBoss2] = v.Tok("encounter", boss2);
        // card ages: the floor each card was added, in the start's deck order (checked by id)
        var cf = ctx.GetProperty("card_floors").EnumerateArray().Select(x => x.GetInt32()).ToList();
        var deck = p.Deck.Cards;
        if (cf.Count == spec.Deck.Count && deck.Count == spec.Deck.Count && deck.Select(c => c.Id.ToString()).SequenceEqual(spec.Deck.Select(c => c.Id)))
        {
            for (int j = 0; j < Math.Min(deck.Count, ML.MaxDeck); j++)
                r.I[ML.ODAge + j] = Math.Clamp(g - (cf[j] > 0 ? cf[j] : g), 0, 63);
        }
        else AgeMismatch++;
        if (!relOk) AgeMismatch++;
        return r;
    }
}
