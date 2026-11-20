// S2 actor: plays battle starts with the network-guided search (self-play or evaluation).
//   dotnet S2.dll play key=value ...
// keys: bridge (shm path or "uniform"), starts, tables, vocab, out (fight log .jsonl), data (chunk dir, self-play),
//   tag, concurrent (fights in flight), fights (total), seconds (time limit), first/count (evaluation: play
//   starts[first..first+count) in order, each `reps` times), mode (selfplay|eval), seed, sims_full, sims_fast,
//   pfull, noise (0/1), cskip (confident-skip threshold), candidates, chunk (decisions per data chunk), shutdown (0/1),
//   exact (1 = G2 arm c, true future), oneturn (1 = G2 arm b, one-turn search), group (self-play fights per loaded start, default 3), match (1 = reproduce the logged monsters; default eval 1, self-play 0),
//   endturn_prompts (0/1: end turn split at the search root into one action per answer of the prompt it opens; 1)
//   ic113 warm-up (2026-11-18, src/ic113/WARMUP.md): mutate (self-play: share of loaded starts whose deck gets 1-2 cards
//   swapped for cards drawn from mutate_pool; 0), mutate_pool (JSON map card id -> draw weight), mutate_upg (chance that a
//   swapped-in card is upgraded; 0.3), card_counts (JSON file, rewritten at every report: per card id the fights whose
//   start deck held it, the plays in those fights, and the same for mutated fights only), fp_ext (0/1: the search
//   fingerprint with card damage / block and hidden counters, as S3's fp_ext; 1)
using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Models;
using Sts2Sim.S0;
using Sts2Sim.S2;

ModelDb.Init(ContentRegistry.AllTypes);
if (args.Length == 0 || args[0] != "play") { Console.WriteLine("usage: S2 play key=value ..."); return 2; }
var kv = args.Skip(1).Select(a => a.Split('=', 2)).ToDictionary(p => p[0], p => p.Length > 1 ? p[1] : "");
string Get(string k, string d) => kv.TryGetValue(k, out var v) ? v : d;
int GetI(string k, int d) => int.Parse(Get(k, d.ToString()));
double GetD(string k, double d) => double.Parse(Get(k, d.ToString(System.Globalization.CultureInfo.InvariantCulture)), System.Globalization.CultureInfo.InvariantCulture);

string mode = Get("mode", "selfplay");
bool eval = mode == "eval";
string tag = Get("tag", "a0");
var vocab = ObsLayout.Vocab.Load(Get("vocab", "src/s0/data/combat_vocab_v113.json"));
// obs=5 (2026-11-15): relic counters in the masked v5 slots; obs=6 also corrects the intent damage (existing slot).
ObservableState.Extended = GetI("fp_ext", 1) == 1;
int obsVersion = GetI("obs", 4);
CombatObservation.RelicCounters = obsVersion >= 5;
CombatObservation.ModifiedIntentDamage = obsVersion >= 6 || GetI("intent_damage", 0) == 1;
// shard=i/N: this actor uses only the battle starts with line index % N == i (and loads only their tables)
string[] shard = Get("shard", "0/1").Split('/');
int shardI = int.Parse(shard[0]), shardN = int.Parse(shard[1]);
var specLines = File.ReadLines(Get("starts", "src/s0/data/battle_starts_v0.111.0_defect_998.jsonl"))
    .Where(l => !string.IsNullOrWhiteSpace(l)).Where((l, i) => i % shardN == shardI).ToList();
var allSpecs = specLines.Select(l => JsonSerializer.Deserialize<BattleStartSpec>(l)!).ToList();
var ids = allSpecs.Select(s => s.Id).ToHashSet(StringComparer.Ordinal);
TfScorer.RepriceMode = Get("reprice", "");
var tf = new TfScorer(Get("tables", "src/s0/data/tf_tables_v0.111.0_defect_1000.jsonl"), ids.Contains);
var specs = allSpecs.Where(s => tf.Has(s.Id)).ToList();
specLines = null; allSpecs = null;
var scfg = new NetSearchConfig
{
    GumbelNoise = GetI("noise", eval ? 0 : 1) == 1, ExactWorlds = GetI("exact", 0) == 1, OneTurn = GetI("oneturn", 0) == 1, ConfidentSkip = GetD("cskip", 2.0), RootCandidates = GetI("candidates", 16),
    CScale = GetD("cscale", 0.1), CVisit = GetD("cvisit", 50), QSpanMin = GetD("qspan", 0), WinBound = GetD("win_bound", -1),
    EndTurnPrompts = GetI("endturn_prompts", 1) == 1,
};
var fcfg = new FightConfig
{
    SimsFull = GetI("sims_full", 64), SimsFast = GetI("sims_fast", eval ? GetI("sims_full", 64) : 16),
    PFull = GetD("pfull", eval ? 1.0 : 0.25), Record = !eval && kv.ContainsKey("data"), Context = GetI("context", 1) == 1,
};
ulong seed = ulong.Parse(Get("seed", "1"));
int concurrent = GetI("concurrent", 64);
double seconds = GetD("seconds", 1e18);
int first = GetI("first", 0), count = GetI("count", specs.Count - first), reps = GetI("reps", 1);
int total = eval ? count * reps : GetI("fights", int.MaxValue);
string bridgePath = Get("bridge", "uniform");
IEvaluator net;
BridgeEvaluator? bridge = null;
if (bridgePath == "uniform") net = new UniformEvaluator();
else net = bridge = new BridgeEvaluator(bridgePath, vocab);
if (bridge is not null && concurrent > bridge.Capacity) concurrent = bridge.Capacity;
var writer = kv.ContainsKey("data") ? new ChunkWriter(Get("data", ""), tag, GetI("chunk", 4096)) : null;
string outPath = Get("out", $"src/s2/out/{tag}.jsonl");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
using var log = new StreamWriter(outPath, append: true);
var rng = new Random((int)(seed % int.MaxValue));
Console.WriteLine($"[{tag}] mode {mode}, {specs.Count} battle starts with tables, concurrent {concurrent}, sims {fcfg.SimsFull}/{fcfg.SimsFast} " +
    $"pfull {fcfg.PFull}, noise {scfg.GumbelNoise}, evaluator {bridgePath}, fights {(total == int.MaxValue ? "unbounded" : total)}, " +
    $"table entries raised by free disposal {tf.FreeDisposalRaised}");

if (mode == "human")
{
    // Human fight outcomes (src/s3/py/human_rows_prep.py adds h_surv and h_end_hp to the start lines): the first
    // decision of each battle start, encoded as the actors encode it, labelled with the logged human outcome; no policy
    // target, held-slot mask 0 (no potion labels). Writes one chunk row per start to data=DIR. Use match=1.
    var hum = new Dictionary<string, (bool Surv, int Hp)>(StringComparer.Ordinal);
    foreach (var l in File.ReadLines(Get("starts", "")))
    {
        if (string.IsNullOrWhiteSpace(l)) continue;
        using var doc = JsonDocument.Parse(l);
        var e = doc.RootElement;
        if (e.TryGetProperty("h_surv", out var hs))
            hum[e.GetProperty("id").GetString()!] = (hs.GetInt32() == 1, e.GetProperty("h_end_hp").GetInt32());
    }
    var uni = new UniformEvaluator();
    int hOk = 0, hBad = 0, hNext = 0;
    async Task HumanWorker()
    {
        while (true)
        {
            int k = hNext++;
            if (k >= specs.Count) return;
            var spec = specs[k];
            if (!hum.TryGetValue(spec.Id, out var h)) continue;
            try
            {
                var st = await FightRunner.LoadBaseAsync(spec, "HUM-" + spec.Id, GetI("match", 1) == 1);
                var ctx = BattleContext.From(spec);
                double tMax = Math.Max(1e-3, tf.MaxValue(spec.Id));
                var sig = fcfg.Context ? new FightSignals(tf, ctx, spec, tMax) : null;
                var r = await new NetSearch(scfg, uni, tf, ctx, tMax, sig).SearchAsync(st, 0, 1);
                if (r.RootActions.Count <= 1) { hBad++; continue; }
                var o = new FightOutcome { Id = spec.Id, RoomType = spec.RoomType, Encounter = spec.EncounterId,
                    Result = h.Surv ? "won" : "lost", EndHp = h.Hp, TableMaxHp = tf.TableMaxHp(spec.Id) };
                o.Records.Add(FightRunner.Encode(st, r, vocab, false, sig));
                writer!.Add(o);
                hOk++;
                if (hOk % 2000 == 0) Console.WriteLine($"[{tag}] human rows {hOk}");
            }
            catch (Exception ex) { if (++hBad <= 20) Console.WriteLine($"[{tag}] {spec.Id}: {ex.GetType().Name}: {ex.Message}"); }
        }
    }
    Pump.Run(() => Task.WhenAll(Enumerable.Range(0, Math.Max(1, concurrent)).Select(_ => HumanWorker())), () => false);
    writer?.Flush();
    Console.WriteLine($"[{tag}] human mode: {hOk} rows written, {hBad} skipped, {hum.Count} labelled starts in the file, {specs.Count} with tables in this shard");
    return 0;
}

var wall = Stopwatch.StartNew();
int next = 0, done = 0, won = 0; long sims = 0, decisions = 0, leaves = 0; double relSum = 0;
var lastReport = Stopwatch.StartNew();

int group = GetI("group", eval ? 1 : 3);
double mutateShare = GetD("mutate", 0), mutateUpg = GetD("mutate_upg", 0.3);
var mutatePool = kv.ContainsKey("mutate_pool")
    ? JsonSerializer.Deserialize<Dictionary<string, double>>(File.ReadAllText(Get("mutate_pool", "")))!.Where(x => x.Value > 0).OrderBy(x => x.Key, StringComparer.Ordinal).Select(x => (Id: x.Key, W: x.Value)).ToList()
    : new List<(string Id, double W)>();
double mutateTotal = mutatePool.Sum(x => x.W);
int mutateDone = 0, mutateFailed = 0;
var curseOrStatus = new HashSet<string>(StringComparer.Ordinal);
foreach (var cm in ModelDb.All<Sts2Sim.Core.Models.CardModel>())
    if (cm.Type.ToString() is "Curse" or "Status") curseOrStatus.Add(cm.Id.ToString());
string? cardCountsPath = kv.ContainsKey("card_counts") ? Get("card_counts", "") : null;
var cardCounts = new Dictionary<string, long[]>(StringComparer.Ordinal);
if (mutatePool.Count > 0) Console.WriteLine($"[{tag}] mutation: share {mutateShare}, pool {mutatePool.Count} cards, upgraded {mutateUpg}");
bool match = GetI("match", eval ? 1 : 0) == 1;

async Task Worker()
{
    while (true)
    {
        if (next >= total || wall.Elapsed.TotalSeconds > seconds) return;
        // self-play: one battle start, loaded once, played `group` times with different chance; eval: start f / reps
        int f0 = next;
        int g = eval ? 1 : Math.Min(group, total - f0);
        next += g;
        BattleStartSpec spec = eval ? specs[first + f0 / reps] : specs[rng.Next(specs.Count)];
        string? mutated = null;
        if (!eval && mutatePool.Count > 0 && rng.NextDouble() < mutateShare)
        {
            var (ms, desc) = Mutate(spec);
            if (desc is not null) { spec = ms; mutated = desc; }
        }
        CombatState baseState;
        try { baseState = await FightRunner.LoadBaseAsync(spec, eval ? "EVAL-" + spec.Id : $"S2-{tag}-{seed}-{f0}", match); }
        catch (Exception e)
        {
            Console.WriteLine($"[{tag}] load failed {spec.Id}{(mutated is null ? "" : " (mutated " + mutated + ")")}: {e.GetType().Name}: {e.Message}");
            if (mutated is not null) mutateFailed++;
            continue;
        }
        if (mutated is not null) mutateDone++;
        var deckIds = spec.Deck.Select(c => c.Id).ToHashSet(StringComparer.Ordinal);
        for (int f = f0; f < f0 + g; f++)
        {
            ulong fs = ResamplingClone.Mix(seed, (ulong)f);
            FightOutcome o = await FightRunner.PlayAsync(spec, baseState, fs, net, tf, vocab, scfg, fcfg);
            var rec = FightRunner.LogRecord(o, f, tag);
            if (mutated is not null) rec["mutated"] = mutated;
            log.WriteLine(JsonSerializer.Serialize(rec));
            if (cardCountsPath is not null) CountCards(deckIds, o, mutated is not null);
            writer?.Add(o);
            done++; if (o.Survived) won++;
            relSum += o.Rel; sims += o.Simulations; decisions += o.Decisions; leaves += o.Leaves;
        }
        if (lastReport.Elapsed.TotalSeconds > 30)
        {
            lastReport.Restart();
            log.Flush();
            Report();
        }
    }
}

// coverage mutation: 1 or 2 deck cards (not curses / statuses) replaced by cards drawn from the pool by weight; the start
// keeps its id, so the fight is scored by the original deck's T_f table (an approximation for the swapped cards)
(BattleStartSpec, string?) Mutate(BattleStartSpec sp)
{
    var deck = sp.Deck.ToList();
    var idx = Enumerable.Range(0, deck.Count).Where(i => !curseOrStatus.Contains(deck[i].Id)).ToList();
    int n = Math.Min(rng.NextDouble() < 0.5 ? 1 : 2, idx.Count);
    if (n == 0) return (sp, null);
    var parts = new List<string>();
    for (int j = 0; j < n; j++)
    {
        int pick = rng.Next(idx.Count); int i = idx[pick]; idx.RemoveAt(pick);
        double u = rng.NextDouble() * mutateTotal; string id = mutatePool[^1].Id;
        foreach (var (cid, wgt) in mutatePool) { u -= wgt; if (u <= 0) { id = cid; break; } }
        int upg = rng.NextDouble() < mutateUpg ? 1 : 0;
        parts.Add($"{deck[i].Id}->{id}{(upg > 0 ? "+" : "")}");
        deck[i] = new BattleCard(id, upg, null, 0);
    }
    return (sp with { Deck = deck }, string.Join(",", parts));
}

void CountCards(HashSet<string> deckIds, FightOutcome o, bool mut)
{
    foreach (string id in deckIds)
    {
        var c = cardCounts.TryGetValue(id, out var x) ? x : cardCounts[id] = new long[4];
        c[0]++; if (mut) c[2]++;
    }
    if (o.Played is null) return;
    foreach (var (id, k) in o.Played)
    {
        var c = cardCounts.TryGetValue(id, out var x) ? x : cardCounts[id] = new long[4];
        c[1] += k; if (mut) c[3] += k;
    }
}

void WriteCardCounts()
{
    if (cardCountsPath is null) return;
    var d = cardCounts.OrderBy(x => x.Key, StringComparer.Ordinal).ToDictionary(x => x.Key,
        x => new Dictionary<string, long> { ["fights_in_deck"] = x.Value[0], ["plays"] = x.Value[1], ["mutated_fights"] = x.Value[2], ["mutated_plays"] = x.Value[3] });
    var meta = new Dictionary<string, object> { ["fights"] = done, ["mutated_starts"] = mutateDone, ["mutate_load_failed"] = mutateFailed, ["cards"] = d };
    File.WriteAllText(cardCountsPath + ".tmp", JsonSerializer.Serialize(meta));
    File.Move(cardCountsPath + ".tmp", cardCountsPath, true);
}

void Report()
{
    WriteCardCounts();
    double t = wall.Elapsed.TotalSeconds;
    Console.WriteLine($"[{tag}] {t:F0}s fights {done} (won {won}, mean rel {relSum / Math.Max(1, done):F4}), decisions {decisions}, " +
        $"sims {sims} = {sims / t:F0}/s, leaves {leaves / t:F0}/s" +
        (bridge is null ? "" : $", evals {bridge.States} in {bridge.Requests} batches (mean {bridge.States / (double)Math.Max(1, bridge.Requests):F0}), bridge wait {100 * bridge.WaitSeconds / t:F0}%") +
        (writer is null ? "" : $", records {writer.Written}") +
        $"; CPU s: load {Timing.Load:F1}, root legal {Timing.RootLegal:F1}, clone {Timing.Clone:F1}, apply {Timing.Apply:F1}, hash {Timing.Hash:F1}, encode {Timing.Encode:F1}");
}

Pump.Run(() => Task.WhenAll(Enumerable.Range(0, concurrent).Select(_ => Worker())), () => bridge?.Flush() ?? false);
writer?.Flush();
log.Flush();
Report();
if (bridge is not null && Get("shutdown", "0") == "1") bridge.Shutdown();
bridge?.Dispose();
return 0;
