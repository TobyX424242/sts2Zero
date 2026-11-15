// S3 actor: plays whole Defect runs; macro decisions by the macro network, fights by the network-guided search.
//   dotnet S3.dll play key=value ...
// keys: bridge (combat shm path or "uniform"), mbridge (macro shm path or "uniform"), vocab, mvocab,
//   out (run log .jsonl), fights (fight log .jsonl, optional), data (combat chunk dir), mdata (macro chunk dir),
//   tag, concurrent (runs in flight), runs (total), seconds (time limit: no new run starts after it),
//   mode (selfplay|eval), ascension, seed (actor seed), seeds (eval: seed prefix, runs use <prefix><index>),
//   first/count (eval: seed indices), sims_full, sims_fast, pfull, boss_sims (budget factor in boss fights), hall_sims (budget factor in hallway
//   fights), boss_pfull (share of boss moves searched with sims_full; default pfull), max_rounds, noise (0/1),
//   cskip (confident-skip threshold), candidates, cscale / cvisit (sigma of the completed Q: 0.1 / 50), qspan (floor of the
//   Q span in sigma, relative T_f; 0 = min-max as before), perturb (share of self-play runs with perturbed fight starts),
//   scorer (tf|hp), kappa (T_f pessimism), greedy (macro argmax; default eval 1), context (0/1), chunk, mchunk,
//   max_floors, shutdown (0/1), asc_mix (self-play: share of runs at `ascension`, the rest uniform over the levels below; 1),
//   asc_weights (self-play: weights of the levels below `ascension`, comma-separated; default uniform),
//   practice_boss / practice_elite (self-play: extra copies of each live boss / elite fight played on resampled clones
//   and recorded; 0), practice_min_act (1-based act from which practice copies are played; 1), practice_pfull (share of
//   practice moves at sims_full; default as in the live fight),
//   reuse (0/1: keep the search tree between moves of a fight; 0), reuse_cap (node cap of a kept tree; 5000),
//   death_value (zero|progress|progress_raw: T_f of a death = 0, or d = (1 - alpha) x acts cleared / 3 as in the macro value
//   target, capped at T_f(1 HP, no potions) (progress) or with table entries below d raised to d (progress_raw); zero),
//   death_alpha (alpha for death_value=progress; 0.05), death_scale (d multiplied by this; 1), death_alpha_log (macro_train_log.jsonl: alpha = its last "alpha"
//   entry when the file has one, else death_alpha), boss2_model (JSON of src/s3/fit_boss2.py: the first of two act-3
//   bosses scored by d + (1 - d) W2(HP share, potions); off), tf_dump (JSONL file, "{tag}" replaced by the actor tag: T_f
//   tables of the first tf_dump_k act-3 boss and hallway fights per act of this actor; off), tf_dump_k (20),
//   tf_stake (0/1: tables rescaled to the outcome not yet secured, (T - b) / (1 - b), b = (1 - alpha) x acts cleared / 3;
//   alpha from death_alpha / death_alpha_log; 0), stake_floor (floor of the rescaled table as a share of T; 0.02),
//   endturn_prompts (0/1: the search root probes end turn once on a resampled clone and splits it into one action per
//   answer of the prompt it opens, e.g. Knowledge Demon's curse; 0 = such prompts answered at random, the old build; 1)
//   fp_ext (0/1: the search's observable-state fingerprint includes card damage / block and hidden power counters
//   (S0 ObservableState.Extended); 0 = the old fingerprint; 1)
// dotnet S3.dll prompttest: checks of the end-turn prompt expansion on a Knowledge Demon fight (PASS/FAIL, exit 1 on failure)
using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Characters;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S2;
using Sts2Sim.S3;

ModelDb.Init(ContentRegistry.AllTypes);
if (args.Length == 0) { Console.WriteLine("usage: S3 play|encode key=value ..."); return 2; }
var kv = args.Skip(1).Select(a => a.Split('=', 2)).ToDictionary(p => p[0], p => p.Length > 1 ? p[1] : "");
string Get(string k, string d) => kv.TryGetValue(k, out var v) ? v : d;
int GetI(string k, int d) => int.Parse(Get(k, d.ToString()));
double GetD(string k, double d) => double.Parse(Get(k, d.ToString(System.Globalization.CultureInfo.InvariantCulture)), System.Globalization.CultureInfo.InvariantCulture);
if (args[0] == "tftest") return TfDeathTest.Run();
if (args[0] == "prompttest") return await PromptTest.Run(Get("vocab", "src/s0/data/combat_vocab_v1.json"));
if (args[0] == "encode") return await EncodeTest.Run(Get("out", "src/s3/out/encode"), GetI("runs", 20), GetI("ascension", 10), Get("mvocab", "src/s3/data/macro_vocab.json"));
if (args[0] != "play") { Console.WriteLine("usage: S3 play|encode key=value ..."); return 2; }

string mode = Get("mode", "selfplay");
bool eval = mode == "eval";
string tag = Get("tag", "a0");
var vocab = ObsLayout.Vocab.Load(Get("vocab", "src/s0/data/combat_vocab_v1.json"));
Sts2Sim.S0.ObservableState.Extended = GetI("fp_ext", 1) == 1;
// obs=5 (2026-11-15): relic counters in the masked v5 slots (the live v3 network is bit-exact); obs=6 also corrects the
// enemy intent damage (an existing slot, so only for a retrained network). Default 4 keeps the pre-v5 input bit for bit.
int obsVersion = GetI("obs", 4);
CombatObservation.RelicCounters = obsVersion >= 5;
CombatObservation.ModifiedIntentDamage = obsVersion >= 6 || GetI("intent_damage", 0) == 1;
var mvocab = new MacroVocab(Get("mvocab", "src/s3/data/macro_vocab.json"));
var cfg = new AgentConfig
{
    Ascension = GetI("ascension", 10), SimsFull = GetI("sims_full", 64), SimsFast = GetI("sims_fast", eval ? GetI("sims_full", 64) : 16),
    PFull = GetD("pfull", eval ? 1.0 : 0.25), BossSimsFactor = GetD("boss_sims", 1.0), RecordCombat = !eval && kv.ContainsKey("data"),
    Context = GetI("context", 1) == 1, Greedy = GetI("greedy", eval ? 1 : 0) == 1, Scorer = Get("scorer", "tf"), Kappa = GetD("kappa", 0),
    MaxRounds = GetI("max_rounds", 50), HallSimsFactor = GetD("hall_sims", 1.0), BossPFull = GetD("boss_pfull", -1.0),
    Search = new NetSearchConfig
    {
        GumbelNoise = GetI("noise", eval ? 0 : 1) == 1, ConfidentSkip = GetD("cskip", 2.0), RootCandidates = GetI("candidates", 16),
        CScale = GetD("cscale", 0.1), CVisit = GetD("cvisit", 50), QSpanMin = GetD("qspan", 0),
        Reuse = GetI("reuse", 0) == 1, ReuseCap = GetI("reuse_cap", 5000),
        EndTurnPrompts = GetI("endturn_prompts", 1) == 1,
    },
    // practice fights are training data: self-play only
    PracticeBoss = eval ? 0 : GetI("practice_boss", 0), PracticeElite = eval ? 0 : GetI("practice_elite", 0),
    PracticeMinAct = GetI("practice_min_act", 1), PracticePFull = GetD("practice_pfull", -1.0),
    DeathValue = Get("death_value", "zero"), DeathAlpha = GetD("death_alpha", 0.05), DeathScale = GetD("death_scale", 1.0),
    Boss2 = kv.ContainsKey("boss2_model") ? Boss2Model.Load(Get("boss2_model", "")) : null,
    TfStake = GetI("tf_stake", 0) == 1, StakeFloor = GetD("stake_floor", 0.02),
};
if (cfg.TfStake && (cfg.DeathValue != "zero" || cfg.Boss2 is not null)) { Console.WriteLine("tf_stake=1 replaces death_value and boss2_model: use it alone"); return 2; }
if (cfg.DeathValue is not ("zero" or "progress" or "progress_raw")) { Console.WriteLine($"unknown death_value {cfg.DeathValue} (zero|progress|progress_raw)"); return 2; }
string alphaFrom = $"death_alpha {cfg.DeathAlpha}";
if (kv.ContainsKey("death_alpha_log"))
{
    // the macro trainer's alpha at its last iteration (one JSON object per line; a line cut off by a crash is skipped)
    string lp = Get("death_alpha_log", "");
    double? found = null;
    try
    {
        foreach (string line in File.ReadLines(lp))
        {
            try
            {
                using var doc = JsonDocument.Parse(line);
                if (doc.RootElement.TryGetProperty("alpha", out var av) && av.ValueKind == JsonValueKind.Number) found = av.GetDouble();
            }
            catch (JsonException) { }
        }
    }
    catch (Exception e) { alphaFrom += $" (death_alpha_log {lp} unreadable: {e.GetType().Name})"; }
    if (found is { } fa && double.IsFinite(fa) && fa >= 0 && fa <= 1) { cfg.DeathAlpha = fa; alphaFrom = $"alpha {fa} from {lp}"; }
    else if (!alphaFrom.Contains("unreadable")) alphaFrom += $" (no alpha in {lp})";
}
if (eval && (GetI("practice_boss", 0) > 0 || GetI("practice_elite", 0) > 0)) Console.WriteLine($"[{tag}] practice fights ignored in eval mode");
double perturb = GetD("perturb", eval ? 0.0 : 0.1);
ulong seed = ulong.Parse(Get("seed", "1"));
int concurrent = GetI("concurrent", 32), maxFloors = GetI("max_floors", 80);
double seconds = GetD("seconds", 1e18);
string seedPrefix = Get("seeds", eval ? "EVAL" : $"S3-{tag}-{seed}-");
int first = GetI("first", 0);
int total = eval ? GetI("count", 100) : GetI("runs", int.MaxValue);

string bridgePath = Get("bridge", "uniform"), mbridgePath = Get("mbridge", "uniform");
BridgeEvaluator? bridge = bridgePath == "uniform" ? null : new BridgeEvaluator(bridgePath, vocab);
MacroBridgeEvaluator? mbridge = mbridgePath == "uniform" ? null : new MacroBridgeEvaluator(mbridgePath);
// a run waits for at most one combat evaluation at a time (the search awaits its simulations one by one), so the
// number of runs in flight is limited by the bridge capacity
if (bridge is not null && concurrent > bridge.Capacity) concurrent = bridge.Capacity;
var services = new ActorServices
{
    Net = bridge is null ? new UniformEvaluator() : bridge, Macro = mbridge is null ? new UniformMacroEvaluator() : mbridge,
    Vocab = vocab, MVocab = mvocab, Cfg = cfg,
    CombatWriter = cfg.RecordCombat ? new ChunkWriter(Get("data", ""), tag, GetI("chunk", 4096)) : null,
};
services.Trace = GetI("trace", 0) == 1;
if (kv.ContainsKey("tf_dump"))
{
    string dumpPath = Get("tf_dump", "").Replace("{tag}", tag);   // {tag}: one file per actor
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dumpPath))!);
    services.TfDump = new StreamWriter(dumpPath, append: true);
    services.TfDumpK = GetI("tf_dump_k", 20);
}
var runWriter = !eval && kv.ContainsKey("mdata") ? new RunChunkWriter(Get("mdata", ""), tag, GetI("mchunk", 16)) : null;
string outPath = Get("out", $"src/s3/out/{tag}.jsonl");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
using var log = new StreamWriter(outPath, append: true);
StreamWriter? fightLog = kv.ContainsKey("fights") ? new StreamWriter(Get("fights", ""), append: true) : null;
if (fightLog is not null)
    services.FightLog = (o, rec, floor) =>
    {
        var d = FightRunner.LogRecord(o, rec.Fights, tag);
        d["run_seed"] = rec.Seed; d["floor"] = floor;
        if (o.Practice) d["practice"] = true;
        if (cfg.Search.Reuse) d["reused_visits"] = o.ReusedVisits;
        if (cfg.DeathValue != "zero") d["death_value"] = Math.Round(o.DeathValue, 5);
        if (o.TfKind is not null) d["tf_kind"] = o.TfKind;
        fightLog.WriteLine(JsonSerializer.Serialize(d));
    };
var rng = new Random((int)(seed % int.MaxValue));
Console.WriteLine($"[{tag}] mode {mode}, ascension {cfg.Ascension} (share {(eval ? 1.0 : GetD("asc_mix", 1.0))}), concurrent {concurrent}, sims {cfg.SimsFull}/{cfg.SimsFast} pfull {cfg.PFull}, " +
    $"noise {cfg.Search.GumbelNoise}, greedy macro {cfg.Greedy}, scorer {cfg.Scorer}, perturb {perturb}, combat {bridgePath}, macro {mbridgePath}, " +
    $"runs {(total == int.MaxValue ? "unbounded" : total)}" +
    (cfg.PracticeBoss > 0 || cfg.PracticeElite > 0 ? $", practice boss {cfg.PracticeBoss} elite {cfg.PracticeElite} from act {cfg.PracticeMinAct}" : "") +
    (cfg.Search.Reuse ? $", tree reuse (cap {cfg.Search.ReuseCap})" : "") +
    (cfg.Search.EndTurnPrompts ? ", end-turn prompts at the root" : "") +
    (CombatObservation.Version >= 5 ? $", obs v{CombatObservation.Version} ({(CombatObservation.RelicCounters ? "relic counters" : "")}{(CombatObservation.ModifiedIntentDamage ? (CombatObservation.RelicCounters ? ", " : "") + "modified intent damage" : "")})" : "") +
    (cfg.DeathValue != "zero" ? $", death value {cfg.DeathValue} x {cfg.DeathScale} ({alphaFrom})" : "") +
    (cfg.Boss2 is not null ? $", double-boss model {cfg.Boss2}" : "") +
    (services.TfDump is not null ? $", T_f dump {Get("tf_dump", "").Replace("{tag}", tag)} (k {services.TfDumpK})" : ""));

var wall = Stopwatch.StartNew();
int next = 0, done = 0, won = 0, errors = 0;
long floors = 0;
var lastReport = Stopwatch.StartNew();

// fail fast if the single thread hangs inside the engine (a blocked wait cannot be seen by the pump's own check).
// Progress is real work: answered network requests and finished runs. Agent callbacks do not count (2026-11-12: an
// endless fight kept calling the agent, so a callback counter never stalled). Without bridges, callbacks are all there is.
long lastProgress = -1; var stall = Stopwatch.StartNew();
var watchdog = new Thread(() =>
{
    while (true)
    {
        Thread.Sleep(10_000);
        long p = bridge is null && mbridge is null ? Interlocked.Read(ref services.Progress)
            : (bridge?.Requests ?? 0) + (mbridge?.Requests ?? 0) + Volatile.Read(ref done);
        if (p != lastProgress) { lastProgress = p; stall.Restart(); }
        else if (stall.Elapsed.TotalSeconds > 300)
        {
            Console.WriteLine($"[{tag}] no progress for 300 s (last callback {services.WhereWho}, run {services.WhereSeed}, floor {services.WhereFloor}): exiting");
            Environment.Exit(3);
        }
    }
}) { IsBackground = true };
watchdog.Start();

// Self-play ascension: a share `asc_mix` of the runs at the target level (`ascension`), the rest uniformly at the
// levels below it. Evaluation always plays the target level.
double ascMix = GetD("asc_mix", 1.0);
// asc_weights=w0,w1,... (one weight per level below `ascension`): the levels below are drawn in proportion to these
// weights instead of uniformly; absent = uniform (the original draw, unchanged)
double[]? ascW = null;
if (kv.ContainsKey("asc_weights"))
{
    ascW = Get("asc_weights", "").Split(',').Select(x => double.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
    if (ascW.Length != cfg.Ascension || ascW.Any(w => w < 0) || ascW.Sum() <= 0)
        throw new ArgumentException($"asc_weights needs {cfg.Ascension} non-negative weights with a positive sum");
}
int Lower()
{
    if (ascW is null) return rng.Next(cfg.Ascension);
    double u = rng.NextDouble() * ascW.Sum();
    for (int k = 0; k < ascW.Length; k++) { u -= ascW[k]; if (u < 0) return k; }
    return Array.FindLastIndex(ascW, w => w > 0);
}
int SampleAscension() => eval || cfg.Ascension == 0 || rng.NextDouble() < ascMix ? cfg.Ascension : Lower();

async Task<RunRecord> PlayRunAsync(string runSeed, bool perturbed)
{
    int asc = SampleAscension();
    var acts = ActDefinition.GetRandomList(runSeed);
    var run = new RunState(runSeed, acts, ascensionLevel: asc);
    var player = Player.CreateForNewRun(ModelDb.Character<Defect>(), run);
    run.AddPlayer(player);
    var agent = new RunAgent(services, run, player, runSeed, perturbed, asc);
    var driver = new RunDriver(run, agent);
    driver.OnRoomResolved += agent.OnRoomResolved;
    try { agent.Finish(await driver.RunAsync(maxFloors)); }
    catch (Exception e) { agent.Fail(e); }
    return agent.Rec;
}

async Task Worker()
{
    while (true)
    {
        if (next >= total || wall.Elapsed.TotalSeconds > seconds) return;
        int i = next++;
        RunRecord rec = await PlayRunAsync($"{seedPrefix}{first + i:D6}", !eval && rng.NextDouble() < perturb);
        log.WriteLine(JsonSerializer.Serialize(rec.Log(tag, first + i)));
        runWriter?.Add(rec);
        done++; floors += rec.Floors;
        if (rec.Won) won++;
        if (rec.Error) { errors++; if (errors <= 20) Console.WriteLine($"[{tag}] run {rec.Seed} error at floor {rec.Floors}: {rec.ErrorText}"); }
        if (lastReport.Elapsed.TotalSeconds > 30) { lastReport.Restart(); log.Flush(); fightLog?.Flush(); services.TfDump?.Flush(); Report(); }
    }
}

void Report()
{
    double t = wall.Elapsed.TotalSeconds;
    Console.WriteLine($"[{tag}] {t:F0}s runs {done} (won {won}, errors {errors}, mean floors {floors / (double)Math.Max(1, done):F1}) = {3600 * done / t:F0}/h, " +
        $"fights {services.Fights} (won {services.FightsWon}), combat decisions {services.CombatDecisions}, sims {services.Simulations} = {services.Simulations / t:F0}/s, " +
        $"macro decisions {services.MacroDecisions}, T_f tables {services.TfBuilds} ({services.TfRows} rows)" +
        (bridge is null ? "" : $", combat evals {bridge.States} in {bridge.Requests} batches (mean {bridge.States / (double)Math.Max(1, bridge.Requests):F0}), bridge wait {100 * bridge.WaitSeconds / t:F0}%") +
        (mbridge is null ? "" : $", macro rows {mbridge.Rows} in {mbridge.Requests} requests") +
        (services.CombatWriter is null ? "" : $", combat records {services.CombatWriter.Written}") +
        (cfg.PracticeBoss > 0 || cfg.PracticeElite > 0 ? $", practice fights {services.PracticeFights} (won {services.PracticeWon}, abandoned {services.PracticeAbandoned}), " +
            $"practice decisions {services.PracticeDecisions}, practice sims {services.PracticeSimulations}, practice time {services.PracticeSeconds:F0}s" : "") +
        (cfg.DeathValue != "zero" || cfg.Boss2 is not null ? $", death values capped {services.TfDeathCapped}, T_f entries raised to the death value {services.TfDeathFloorRaised}, double-boss tables {services.TfBoss2Fights}" : "") +
        (services.TfMsFights > 0 ? $", milestone tables {services.TfMsFights}" : "") +
        (cfg.TfStake ? $", stake tables {services.TfStakeFights} (mean banked {services.TfStakeBankedSum / Math.Max(1, services.TfStakeFights):F3}, entries floored {services.TfStakeFloored}; {alphaFrom})" : "") +
        (cfg.Search.EndTurnPrompts ? $", end-turn probes {ActionSpace.EndTurnProbes} (prompt {ActionSpace.EndTurnProbesWithPrompt}, failed {ActionSpace.EndTurnProbeErrors})" : "") +
        (cfg.Search.Reuse ? $", reuse: {services.ReusedVisits / (double)Math.Max(1, services.ReuseSearches):F1} root visits reused per searched move over {services.ReuseSearches}, largest table {services.ReuseTableMax}" : "") +
        (runWriter is null ? "" : $", macro rows written {runWriter.RowsWritten}"));
    if (GetI("timing", 1) == 1) Console.WriteLine(TimingLine(t));
}

// Step 0 profiling (2026-11-13): where the actor's CPU goes. Process CPU covers every thread (GC, spin-waits); the
// listed parts are synchronous sections of the single actor thread; "outside search" is the actor thread's busy time
// (wall - bridge wait) minus the search parts listed (run driver, live steps, macro encoding, T_f tables, pump, ...).
string TimingLine(double t)
{
    double cpu = Process.GetCurrentProcess().TotalProcessorTime.TotalSeconds;
    double busy = t - (bridge?.WaitSeconds ?? 0);
    double inSearch = Timing.Sim + Timing.RootLegal + Timing.Decode;
    double rootEnc = Math.Max(0, Timing.Encode - Timing.SimEval);
    double simRest = Timing.Sim - Timing.Clone - Timing.Apply - Timing.Hash - Timing.Legal - Timing.SimEval;
    string P(double x) => $"{x:F1}s ({100 * x / Math.Max(1e-9, busy):F0}%)";
    long sims = Math.Max(1, Timing.Sims);
    return $"[{tag}] timing {t:F0}s: process CPU {cpu:F0}s = {cpu / t:F2} cores, actor busy {busy:F0}s; " +
        $"in simulations {P(Timing.Sim)} = clone {P(Timing.Clone)}, apply {P(Timing.Apply)} (replay {P(Timing.ApplyReplay)}, " +
        $"frontier {P(Timing.ApplyFront)}; end turns {P(Timing.ApplyEnd)}), hash {P(Timing.Hash)}, legal {P(Timing.Legal)}, " +
        $"leaf encode {P(Timing.SimEval)}, select/backup {P(simRest)}; root legal {P(Timing.RootLegal)}, root encode {P(rootEnc)}, " +
        $"decode {P(Timing.Decode)}, records {P(Timing.Record)}; outside search {P(busy - inSearch - rootEnc - Timing.Record)}; " +
        $"sims {Timing.Sims} ({Timing.Sims / t:F0}/s, {1e6 * Timing.Sim / sims:F0} us each), steps/sim {Timing.Steps / (double)sims:F2} " +
        $"(end turns {Timing.EndSteps / (double)sims:F2}), leaves {Timing.Leaves / (double)sims:F2}, terminals {Timing.Terminals / (double)sims:F2}, " +
        $"tree hits {Timing.Hits / (double)sims:F2} per sim; us/step {1e6 * Timing.Apply / Math.Max(1, Timing.Steps):F0} " +
        $"(end turn {1e6 * Timing.ApplyEnd / Math.Max(1, Timing.EndSteps):F0}), clone {1e6 * Timing.Clone / sims:F0}; " +
        $"GC alloc {GC.GetTotalAllocatedBytes() / 1e9:F1} GB ({GC.GetTotalAllocatedBytes() / 1e6 / t:F0} MB/s), " +
        $"collections {GC.CollectionCount(0)}/{GC.CollectionCount(1)}/{GC.CollectionCount(2)}, pause {GC.GetTotalPauseDuration().TotalSeconds:F1}s";
}

// Combat evaluations are answered within a flush. A macro request (few rows, slower per row) stays in flight while
// fights go on: it is completed when its answer is there, and waited for only when nothing else can run.
bool Flush()
{
    bool did = false;
    if (mbridge is not null)
    {
        if (mbridge.InFlight && mbridge.Ready) { mbridge.Complete(); did = true; }
        if (!mbridge.InFlight) mbridge.Submit();
    }
    if (bridge is not null && bridge.Submit()) { bridge.Complete(); did = true; }
    if (!did && mbridge is not null && mbridge.InFlight) { mbridge.Complete(); did = true; }
    return did;
}

Pump.Run(() => Task.WhenAll(Enumerable.Range(0, concurrent).Select(_ => Worker())), Flush);
services.CombatWriter?.Flush();
runWriter?.Flush();
log.Flush(); fightLog?.Flush(); services.TfDump?.Flush();
Report();
Console.WriteLine($"[{tag}] macro vocabulary: " + string.Join(", ", mvocab.Seen.Where(x => x.Value > 0).Select(x => $"{x.Key} unknown {mvocab.Unknown[x.Key]}/{x.Value}")));
Console.WriteLine($"[{tag}] run-level card selections: " + string.Join(", ", services.SelectionKinds.OrderByDescending(x => x.Value).Select(x => $"{x.Key} {x.Value}")));
Console.WriteLine($"[{tag}] T_f HP-axis formula check: max |logit(a, c at own HP) - network logit| = {services.TfFormulaMaxError:E2} over {services.TfRows} table rows");
Console.WriteLine($"[{tag}] notes: " + string.Join(", ", services.Notes.OrderByDescending(x => x.Value).Take(30).Select(x => $"{x.Key} {x.Value}")));
if (Get("shutdown", "0") == "1") { bridge?.Shutdown(); mbridge?.Shutdown(); }
bridge?.Dispose(); mbridge?.Dispose();
return 0;
