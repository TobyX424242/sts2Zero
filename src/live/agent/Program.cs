// LiveAgent: the S3 agent (RunAgent: macro network + network-guided fight search) plays the real game through the
// AgentBridge mod (src/live/README.md, protocol v1).
//   dotnet LiveAgent.dll play key=value ...     connect to the mod and play
//   dotnet LiveAgent.dll replay file=LOG.jsonl  feed a controller log's states to the decision code (no game)
//   dotnet LiveAgent.dll selfcheck runs=N ...   simulator runs exported as protocol states and run back through the
//                                               import + decision + mapping path (SelfCheck.cs)
// Agent keys: those of `S3 play` (src/s3/S3/Program.cs) with the same meaning; defaults = the s3ic chain's evaluation
//   (AgentSetup.ChainEval: mode=eval sims_full=64 obs=5 character=ironclad era=6 tf_heal=1 greedy_map=1 look_beta=20
//   look_smith=20 look_shop=20 potion_mix_shrink=0.2 potion_mix_extra=1 fake_merchant=1 win_bound=0.001 deck_route=1
//   lethal_backstop=1; preset=none drops them). bridge / mbridge: shm paths of the inference servers or "uniform";
//   potion_mix=FILE (the chain's vbar file), vocab, mvocab.
// Play keys: asc (0-10: start a run at this level when the game is at the main menu; omitted = attach to the run in
//   progress or the one started from the in-game panel), seed, force_start (1: start_run even if a run is in progress),
//   loop (1, default: stay online after a game over for the next run; 0 = exit after one run), host (127.0.0.1), port (47213), log (dir, src/live/logs),
//   dry (1: decide and log, never send act), poll_ms (150), delay_ms (sent with config if given), root (tree root).
using System.Text.Json.Nodes;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Models;
using Sts2Sim.Live;
using Pump = Sts2Sim.S2.Pump;
using Decision = Sts2Sim.Live.Decision;

ModelDb.Init(ContentRegistry.AllTypes);
if (args.Length == 0 || args[0] is not ("play" or "replay" or "selfcheck"))
{
    Console.WriteLine("usage: LiveAgent play|replay|selfcheck key=value ...");
    return 2;
}
var kv = AgentSetup.ParseArgs(args.Skip(1));
string root = kv.TryGetValue("root", out var rt) ? rt : FindRoot();
AgentSetup? setup = null;
int code = 0;
// everything runs on one pump thread with a large stack: the simulator's engine, RunAgent and the search nest deeply
// (the self-check additionally nests a whole simulated run around the controller)
// The setup (and with it the inference bridges) is built on the pump thread: BridgeEvaluator binds to the thread that
// creates it and refuses calls from any other (first live game, 2026-11-20: built on the main thread, every search failed).
void OnBigStack(Func<AgentSetup, Task> main)
{
    Exception? err = null;
    var t = new Thread(() =>
    {
        try
        {
            var su = setup = AgentSetup.Build(kv, root);
            Console.WriteLine($"[live] {su.Describe()}");
            Pump.Run(() => main(su), su.Flush);
        }
        catch (Exception e) { err = e; }
    }, 1024 * 1024 * 1024);
    t.Start(); t.Join();
    if (err is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(err).Throw();
}
if (args[0] == "selfcheck")
{
    OnBigStack(async su => code = await SelfCheck.RunAsync(su));
    return code;
}
if (args[0] == "replay")
{
    OnBigStack(async su => code = await ReplayAsync(su));
    return code;
}
OnBigStack(async su => code = await PlayAsync(su, root));
setup?.Bridge?.Dispose(); setup?.MBridge?.Dispose();
return code;

static string FindRoot()
{
    foreach (string start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        for (var d = new DirectoryInfo(start); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "src", "s3", "data", "macro_vocab_v113.json"))) return d.FullName;
    return Directory.GetCurrentDirectory();
}

static async Task<int> PlayAsync(AgentSetup s, string root)
{
    string host = s.Get("host", "127.0.0.1");
    int port = s.GetI("port", 47213), pollMs = s.GetI("poll_ms", 150);
    bool dry = s.GetI("dry", 0) == 1, loop = s.GetI("loop", 1) == 1;
    int? asc = s.Kv.ContainsKey("asc") ? s.GetI("asc", 0) : null;
    if (asc is < 0 or > 10) { Console.WriteLine("asc must be 0-10"); return 2; }
    string logDir = s.Get("log", Path.Combine(root, "src", "live", "logs"));
    using var log = new LiveLog(logDir, "live");
    using var game = new GameClient(host, port);
    var ctl = new Controller(s, m => { Console.WriteLine($"[live] {m}"); log.Write("say", new JsonObject { ["text"] = m }); });
    Console.WriteLine($"[live] log {log.Path}{(dry ? " (dry run: no actions sent)" : "")}");
    JsonObject hello;
    while (true)
    {
        try { hello = game.Hello(); break; }
        catch (Exception e) { Console.WriteLine($"[live] waiting for the game on {host}:{port} ({e.Message})"); Thread.Sleep(3000); }
    }
    Console.WriteLine($"[live] connected: game {hello.Str("game_version")}, mod {hello.Str("mod_version")}, protocol {hello.Str("protocol")}");
    if (hello.Str("warning") is { Length: > 0 } warn) Console.WriteLine($"[live] mod warning: {warn}");
    log.Write("hello", hello);
    var waitSince = System.Diagnostics.Stopwatch.StartNew();
    if (s.Kv.ContainsKey("delay_ms")) game.Config(s.GetI("delay_ms", 400), null);
    bool startPending = asc is not null;
    int lastId = -1, rejects = 0, runsDone = 0, runsWon = 0;
    bool runWon = false, gameOverReported = false;
    string lastWait = "";
    while (true)
    {
        JsonObject st;
        try { st = game.State(); }
        catch (Exception e) { Console.WriteLine($"[live] state request failed: {e.Message}"); lastId = -1; Thread.Sleep(2000); continue; }
        int id = st.Int("state_id", -1);
        string screen = st.Str("screen") ?? "?";
        if (screen != "GAME_OVER") gameOverReported = false;
        if (startPending)
        {
            if (screen == "MAIN_MENU" || s.GetI("force_start", 0) == 1)
            {
                var r = game.StartRun(asc!.Value, s.Kv.GetValueOrDefault("seed"));
                Console.WriteLine($"[live] start_run A{asc}: {(r.Bool("ok") ? "ok" : r.Str("error"))}");
                log.Write("start_run", new JsonObject { ["ascension"] = asc, ["reply"] = r.DeepClone() });
                startPending = false;
                Thread.Sleep(1000);
                continue;
            }
            if (st.Obj("run") is not null)
            {
                Console.WriteLine($"[live] a run is in progress: attaching to it (asc={asc} applies to the next run; force_start=1 starts a new one now)");
                startPending = false;
            }
        }
        if (!st.Bool("stable") || st.Bool("paused") || !st.Bool("agent_mode", true) || id == lastId)
        {
            string w = st.Bool("paused") ? $"paused: {st.Str("pause_reason")}" : !st.Bool("agent_mode", true) ? "agent mode off"
                : waitSince.Elapsed.TotalSeconds > 30 ? $"waiting for the game ({screen}): {st.Str("wait_reason") ?? "not stable"}{(st.Str("last_error") is { } le ? $"; last error: {le}" : "")}" : "";
            if (w != "" && w != lastWait) Console.WriteLine($"[live] {w}");
            lastWait = w;
            if (st.Bool("paused") || !st.Bool("agent_mode", true)) lastId = -1;   // after a resume the same state is decided again
            Thread.Sleep(pollMs);
            continue;
        }
        lastWait = "";
        waitSince.Restart();
        Decision d;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try { d = await ctl.DecideAsync(st); }
        catch (Exception e)
        {
            d = Decision.Paused($"controller error on {screen}: {e.GetType().Name}: {e.Message}");
            Console.WriteLine(e);
        }
        if (d.Why?.StartsWith("the Architect") == true) runWon = true;   // the ending scene only follows a won run
        var entry = new JsonObject { ["state"] = log.Compact(st), ["decision"] = LiveLog.DecisionJson(d), ["ms"] = sw.ElapsedMilliseconds };
        lastId = id;
        if (d.Wait) { log.Write("wait", entry); if (d.Info.ContainsKey("game_over")) goto over; Thread.Sleep(pollMs); continue; }
        if (d.Pause is not null)
        {
            Console.WriteLine($"[live] PAUSE ({screen}): {d.Pause}");
            if (!dry) entry["reply"] = game.Pause(d.Pause).DeepClone();
            log.Write("decide", entry);
            continue;
        }
        Console.WriteLine($"[live] {screen} #{id}: {d.Why} -> option {d.Option}{(d.Target is { } t ? $" target {t}" : "")}{(d.Cards is { } c ? $" cards [{string.Join(",", c)}]" : "")} ({sw.ElapsedMilliseconds} ms)");
        if (dry) { log.Write("decide", entry); continue; }
        JsonObject reply = game.Act(id, d);
        entry["reply"] = reply.DeepClone();
        entry["sent"] = reply.Bool("ok");
        log.Write("decide", entry);
        if (reply.Bool("ok"))
        {
            rejects = 0;
            await ctl.AfterSentAsync(d);
        }
        else
        {
            Console.WriteLine($"[live] act rejected: {reply.Str("error")}");
            lastId = -1;
            if (++rejects >= 3)
            {
                game.Pause($"the game rejected the agent's action 3 times: {reply.Str("error")} ({d.Why})");
                rejects = 0;
            }
        }
        over:
        if (d.Info.ContainsKey("game_over") && !gameOverReported)
        {
            gameOverReported = true;
            JsonObject? run = st.Obj("run");
            runsDone++; if (runWon) runsWon++;
            Console.WriteLine($"[live] RUN {(runWon ? "WON" : "LOST")}: A{run?.Int("ascension")}, floor {run?.Int("floor")}, act {run?.Int("act_index") + 1}" +
                (runWon ? "" : $", hp {run?.Int("hp")}/{run?.Int("max_hp")}") + $" ({runsWon} won of {runsDone} this session)");
            log.Write("game_over", new JsonObject { ["won"] = runWon, ["detail"] = st.Str("screen_detail"), ["floor"] = run?.Int("floor"), ["hp"] = run?.Int("hp") });
            runWon = false;
            if (loop && asc is not null) { startPending = true; Thread.Sleep(2000); continue; }
            // loop without asc: stay connected and play the next run started from the in-game panel
            if (loop) { Console.WriteLine("[live] waiting for the next run (start one from the in-game panel)"); Thread.Sleep(2000); continue; }
            if (!dry && d.Option >= 0) return 0;
            return 0;
        }
    }
}

static async Task<int> ReplayAsync(AgentSetup s)
{
    string file = s.Get("file", "");
    if (!File.Exists(file)) { Console.WriteLine("replay needs file=LOG.jsonl"); return 2; }
    var ctl = new Controller(s, m => Console.WriteLine($"[replay] {m}"));
    JsonNode? save = null;
    int n = 0, same = 0, differ = 0, errors = 0;
    foreach (string line in File.ReadLines(file))
    {
        if (JsonNode.Parse(line) is not JsonObject e || e.Str("type") != "decide" || e.Obj("state") is not { } st) continue;
        if (st.Obj("run") is { } run)
        {
            if (run.At("save") is JsonValue v && v.TryGetValue(out string? sv) && sv == "same") run["save"] = save?.DeepClone();
            else save = run.At("save")?.DeepClone();
        }
        n++;
        Decision d;
        try { d = await ctl.DecideAsync(st); }
        catch (Exception ex) { errors++; Console.WriteLine($"[replay] #{st.Int("state_id")} {st.Str("screen")}: error {ex.GetType().Name}: {ex.Message}"); continue; }
        int logged = e.Obj("decision").Int("option", -1);
        bool eq = d.Option == logged && d.Pause is null == (e.Obj("decision")?.Str("pause") is null);
        if (eq) same++; else differ++;
        Console.WriteLine($"[replay] #{st.Int("state_id")} {st.Str("screen")}: {d.Why} -> {(d.Pause is not null ? "PAUSE " + d.Pause : "option " + d.Option)} (logged {logged}{(eq ? "" : ", DIFFERENT")})");
        if (e.Bool("sent")) await ctl.AfterSentAsync(d);
    }
    Console.WriteLine($"[replay] {n} decisions: {same} as logged, {differ} different (sampled heads and search seeds differ between runs), {errors} errors");
    return errors > 0 ? 1 : 0;
}
