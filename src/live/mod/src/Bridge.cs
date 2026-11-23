using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace AgentBridge;

/// <summary>Entry point called by the game's ModManager (ModInitializerAttribute).</summary>
[ModInitializer("Init")]
public static class AgentBridgeMod
{
    public static void Init()
    {
        try { Bridge.Init(); }
        catch (Exception e) { U.Error("init failed: " + e); }
    }
}

/// <summary>
/// The mod's core, all on the Godot main thread (driven by SceneTree.ProcessFrame): drains socket requests,
/// tracks when the game is stable (waiting for input), runs actions after the configured delay, auto-advances
/// non-decisions (ancient dialogue), handles the F9 pause hotkey and refreshes the panel.
/// With agent mode off nothing here touches the game: requests are read-only and act/start_run are refused.
/// </summary>
internal static class Bridge
{
    public const string ModVersion = "1.0.0";
    public const int Protocol = 1;

    // ------------------------------------------------------------------ settings (panel / config command)
    public static bool AgentMode;
    public static bool Paused;
    public static string? PauseReason;
    public static int DelayMs = 400;
    public static int PanelAscension;
    public static string PanelSeed = "";
    public static bool FastMode;
    private static FastModeType? _fastModeBefore;
    public static JsonObject? RequestedRun;

    public static string LastAction = "";
    public static string LastError = "";

    // ------------------------------------------------------------------ stability tracking
    private const int SettleMs = 250;        // readiness must hold this long
    private const int NewScreenMs = 600;     // and a new screen must be this old (card screens ignore clicks < 350 ms)
    private const int AfterActionMs = 250;   // after our own action finished
    private static Probe _probe = new();
    private static string _lastKey = "";
    private static long _keySince, _readySince, _lastExecEnd;
    private static bool _rawStable;

    // ------------------------------------------------------------------ state ids
    private static int _stateId;
    private static string _lastFp = "";
    private static bool _bump;
    private static int _lastLoggedStateId = -1;
    private static string _lastSaveLoggedKey = "";

    // ------------------------------------------------------------------ actions
    private sealed class PendingAct
    {
        public required Option Opt;
        public required ActArgs Args;
        public long Due;
        public string Fp = "";
        public int StateId;
    }

    private static PendingAct? _pending;
    private static Task<string?>? _exec;
    private static string _execLabel = "";
    private static long _execStart;
    private static Task<string?>? _start;
    private static Task? _shopTask;
    private static long _autoDue;
    private static string _autoKey = "";

    public static bool ShopPurchaseInFlight => _shopTask != null && !_shopTask.IsCompleted;
    public static void TrackShopPurchase(Task t) => _shopTask = t;
    public static bool Busy => _pending != null || _exec != null || _start != null;
    /// <summary>Multi-step actions stop between steps when the player pauses or switches agent mode off.</summary>
    public static bool Halted => !AgentMode || Paused;
    public const string HaltText = "stopped between steps: agent mode off or paused (finish this screen by hand)";

    private static long _canStartAt;
    private static string? _canStartCache;
    /// <summary>RunStarter.CanStart, cached for a second (it reads the save state).</summary>
    public static string? CanStartCached()
    {
        if (U.NowMs >= _canStartAt) { _canStartCache = RunStarter.CanStart(); _canStartAt = U.NowMs + 1000; }
        return _canStartCache;
    }

    // ------------------------------------------------------------------ timers (main-thread delays)
    private static readonly List<(long due, TaskCompletionSource tcs)> _timers = new();

    /// <summary>Main-thread delay: completed from the frame pump, so continuations resume on the main thread.</summary>
    public static Task Delay(int ms)
    {
        var tcs = new TaskCompletionSource();
        _timers.Add((U.NowMs + Math.Max(0, ms), tcs));
        return tcs.Task;
    }

    // ------------------------------------------------------------------ infrastructure
    private static BridgeServer? _server;
    private static Panel? _panel;
    private static bool _f9Down;
    private static long _panelRefresh;
    private static string _gameVersion = "?";
    private static bool _initialized;

    public static bool ControllerConnected => _server?.Connected ?? false;
    public static string LogFile => JsonlLog.FilePath;

    public static void Init()
    {
        if (_initialized) return;
        _initialized = true;
        U.Info($"AgentBridge {ModVersion} initializing (protocol {Protocol})");
        JsonlLog.Init();
        LoadConfig();
        _gameVersion = ReadGameVersion();
        try
        {
            _server = new BridgeServer();
            _server.Start();
        }
        catch (Exception e)
        {
            U.Error("could not open 127.0.0.1:" + BridgeServer.Port + ": " + e.Message);
            LastError = "socket: " + e.Message;
        }
        var tree = (SceneTree)Engine.GetMainLoop();
        tree.Connect(SceneTree.SignalName.ProcessFrame, Callable.From(Pump));
        Callable.From(() =>
        {
            try { _panel = Panel.Create(tree.Root); }
            catch (Exception e) { U.Error("panel creation failed: " + e); }
        }).CallDeferred();
        U.Info("user data dir: " + OS.GetUserDataDir() + "  logs: " + JsonlLog.Dir);
    }

    private static string ReadGameVersion()
    {
        try
        {
            string dir = Path.GetDirectoryName(OS.GetExecutablePath()) ?? ".";
            string f = Path.Combine(dir, "release_info.json");
            if (!File.Exists(f)) return "?";
            var n = JsonNode.Parse(File.ReadAllText(f));
            return U.StrOf(n?["version"]) ?? "?";
        }
        catch { return "?"; }
    }

    // ------------------------------------------------------------------ config persistence
    private static string ConfigPath => ProjectSettings.GlobalizePath("user://AgentBridge/config.json");

    private static void LoadConfig()
    {
        try
        {
            if (!File.Exists(ConfigPath)) return;
            var n = JsonNode.Parse(File.ReadAllText(ConfigPath)) as JsonObject;
            if (n == null) return;
            DelayMs = Math.Clamp(U.IntOf(n["delay_ms"]) ?? DelayMs, 0, 10000);
            PanelAscension = Math.Clamp(U.IntOf(n["ascension"]) ?? 0, 0, 10);
        }
        catch (Exception e) { U.Warn("config load failed: " + e.Message); }
    }

    public static void SaveConfig()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ConfigPath)!);
            File.WriteAllText(ConfigPath, new JsonObject { ["delay_ms"] = DelayMs, ["ascension"] = PanelAscension }.ToJsonString());
        }
        catch (Exception e) { U.Warn("config save failed: " + e.Message); }
    }

    // ------------------------------------------------------------------ the frame pump
    private static void Pump()
    {
        try { RunTimers(); } catch (Exception e) { U.Error("timers: " + e); }
        try { PollTasks(); } catch (Exception e) { U.Error("poll: " + e); }
        try { UpdateStability(); } catch (Exception e) { U.Error("stability: " + e); }
        try { DrainRequests(); } catch (Exception e) { U.Error("requests: " + e); }
        try { RunPending(); } catch (Exception e) { U.Error("pending: " + e); }
        try { AutoSteps(); } catch (Exception e) { U.Error("auto: " + e); }
        try { Hotkeys(); } catch (Exception e) { U.Error("hotkeys: " + e); }
        try
        {
            if (U.NowMs >= _panelRefresh)
            {
                _panelRefresh = U.NowMs + 200;
                _panel?.Refresh();
            }
        }
        catch (Exception e) { U.Error("panel: " + e); }
    }

    private static void RunTimers()
    {
        if (_timers.Count == 0) return;
        long now = U.NowMs;
        var due = _timers.Where(t => t.due <= now).ToList();
        if (due.Count == 0) return;
        _timers.RemoveAll(t => t.due <= now);
        foreach (var t in due) t.tcs.TrySetResult();
    }

    private static void PollTasks()
    {
        if (_exec != null)
        {
            if (_exec.IsCompleted)
            {
                string? err = _exec.IsFaulted ? "exception: " + _exec.Exception?.GetBaseException().Message
                    : _exec.IsCanceled ? "cancelled" : _exec.Result;
                if (_exec.IsFaulted) U.Error("action '" + _execLabel + "' threw: " + _exec.Exception);
                FinishExec(err);
            }
            else if (U.NowMs - _execStart > 150_000)
            {
                FinishExec("action '" + _execLabel + "' did not finish within 150 s");
            }
        }
        if (_start != null && _start.IsCompleted)
        {
            string? err = _start.IsFaulted ? "exception: " + _start.Exception?.GetBaseException().Message : _start.Result;
            _start = null;
            _bump = true;
            _lastExecEnd = U.NowMs;
            if (err != null)
            {
                RequestedRun = null;
                SetError("start_run failed: " + err, pause: true);
            }
            else
            {
                LastAction = "run started";
                JsonlLog.Write("run_started", new JsonObject { ["requested"] = RequestedRun?.DeepClone() });
            }
        }
        if (RequestedRun != null && _start == null && RunManager.Instance.IsInProgress) RequestedRun = null;
    }

    private static void FinishExec(string? err)
    {
        JsonlLog.Write("act_done", new JsonObject { ["action"] = _execLabel, ["error"] = err });
        _exec = null;
        _lastExecEnd = U.NowMs;
        _bump = true;
        if (err != null) SetError("action '" + _execLabel + "' failed: " + err, pause: true);
    }

    public static void SetError(string msg, bool pause)
    {
        LastError = msg;
        U.Warn(msg);
        if (pause && AgentMode)
        {
            Paused = true;
            PauseReason = msg;
        }
    }

    private static void UpdateStability()
    {
        if (!AgentMode && !ControllerConnected) { _rawStable = false; return; }
        long now = U.NowMs;
        _probe = Screens.Detect();
        if (_probe.Key != _lastKey) { _lastKey = _probe.Key; _keySince = now; }
        if (!_probe.Ready) _readySince = now;
        _rawStable = _probe.Ready && now - _readySince >= SettleMs && now - _keySince >= NewScreenMs
                     && now - _lastExecEnd >= AfterActionMs;
    }

    private static bool StableNow => _rawStable && !Busy;

    // ------------------------------------------------------------------ requests
    private static void DrainRequests()
    {
        if (_server == null) return;
        int n = 0;
        while (n++ < 8 && _server.Requests.TryDequeue(out var p))
        {
            string reply;
            try { reply = Handle(p.Request); }
            catch (Exception e)
            {
                U.Error("request handler: " + e);
                reply = new JsonObject { ["ok"] = false, ["error"] = "internal error: " + e.GetType().Name + ": " + e.Message }.ToJsonString(U.Compact);
            }
            p.Reply.TrySetResult(reply);
        }
    }

    private static JsonObject Ok() => new() { ["ok"] = true };
    private static JsonObject Fail(string e) => new() { ["ok"] = false, ["error"] = e };

    private static string Handle(JsonObject req)
    {
        string cmd = U.StrOf(req["cmd"]) ?? "";
        JsonObject r = cmd switch
        {
            "hello" => Hello(req),
            "state" => StateReply(),
            "act" => Act(req),
            "start_run" => StartRunCmd(req, fromPanel: false),
            "pause" => PauseCmd(U.StrOf(req["reason"]) ?? "paused by controller"),
            "resume" => ResumeCmd(),
            "config" => ConfigCmd(req),
            _ => Fail("unknown cmd '" + cmd + "'"),
        };
        return r.ToJsonString(U.Compact);
    }

    private static JsonObject Hello(JsonObject req)
    {
        if (_server != null) _server.ClientName = U.StrOf(req["client"]) ?? "";
        int proto = U.IntOf(req["protocol"]) ?? Protocol;
        JsonlLog.Write("hello", req);
        var r = Ok();
        r["protocol"] = Protocol;
        r["game_version"] = _gameVersion;
        r["mod_version"] = ModVersion;
        r["agent_mode"] = AgentMode;
        r["paused"] = Paused;
        r["log_file"] = JsonlLog.FilePath;
        if (proto != Protocol) r["warning"] = $"controller speaks protocol {proto}, mod speaks {Protocol}";
        return r;
    }

    /// <summary>Builds the state object; bumps state_id when anything changed since the last one.</summary>
    private static Snapshot CurrentSnapshot(bool withOptions, bool stableOverride = false)
    {
        if (!AgentMode && !ControllerConnected) UpdateStabilityForced();
        bool stable = stableOverride || StableNow;
        Snapshot s = StateBuilder.Build(_probe, stable, withOptions);
        if (_bump || s.Fingerprint != _lastFp)
        {
            _stateId++;
            _lastFp = s.Fingerprint;
            _bump = false;
        }
        return s;
    }

    private static void UpdateStabilityForced()
    {
        _probe = Screens.Detect();
    }

    private static JsonObject StateReply()
    {
        Snapshot s = CurrentSnapshot(withOptions: true);
        var o = new JsonObject
        {
            ["ok"] = true,
            ["state_id"] = _stateId,
            ["stable"] = s.Stable,
            ["agent_mode"] = AgentMode,
            ["paused"] = Paused,
            ["pause_reason"] = PauseReason,
            ["requested_run"] = RequestedRun?.DeepClone(),
            ["screen"] = s.Probe.Screen,
            ["screen_detail"] = s.Probe.Detail,
            ["wait_reason"] = s.Stable ? null : (Busy ? (_start != null ? "starting run" : _exec != null ? "executing " + _execLabel : "action delay") : s.Probe.Wait),
            ["run"] = s.Run,
            ["combat"] = s.Combat,
            ["options"] = U.Arr(s.Options, x => x.Json.DeepClone()),
            ["last_action"] = LastAction,
            ["last_error"] = LastError,
        };
        if (s.Stable && s.Run != null)
        {
            JsonNode? save = StateBuilder.BuildSave(out string? err);
            s.Run["save"] = save;
            if (err != null) s.Run["save_error"] = err;
        }
        if (s.Stable && _stateId != _lastLoggedStateId)
        {
            // Every new stable state is logged; the (large) save JSON only on the first state of each screen.
            _lastLoggedStateId = _stateId;
            var logged = (JsonObject)o.DeepClone();
            if (s.Probe.Key == _lastSaveLoggedKey && logged["run"] is JsonObject lr && lr["save"] != null)
                lr["save"] = "omitted: same screen as an earlier logged state";
            _lastSaveLoggedKey = s.Probe.Key;
            JsonlLog.Write("state", logged);
        }
        return o;
    }

    private static ActArgs ParseArgs(JsonObject req, out string? error)
    {
        error = null;
        var a = new ActArgs { Target = U.IntOf(req["target"]) };
        if (req["target"] != null && a.Target == null && req["target"] is not null) error = "target must be an integer";
        if (req["cards"] is JsonArray arr)
        {
            a.Cards = new List<int>();
            foreach (JsonNode? n in arr)
            {
                int? i = U.IntOf(n);
                if (i == null) { error = "cards must be a list of integers"; break; }
                a.Cards.Add(i.Value);
            }
        }
        else if (req["cards"] != null) error = "cards must be a list";
        return a;
    }

    private static JsonObject Act(JsonObject req)
    {
        JsonlLog.Write("act", req);
        if (!AgentMode) return Fail("agent mode is off (switch it on in the in-game panel)");
        if (Paused) return Fail("paused: " + (PauseReason ?? ""));
        if (Busy) return Fail("busy: " + (_start != null ? "starting a run" : _exec != null ? "executing " + _execLabel : "an action is already scheduled"));
        int? sid = U.IntOf(req["state_id"]);
        int? idx = U.IntOf(req["option"]);
        if (sid == null || idx == null) return Fail("act needs integer state_id and option");
        ActArgs args = ParseArgs(req, out string? perr);
        if (perr != null) return Fail(perr);

        Snapshot s = CurrentSnapshot(withOptions: true);
        if (!s.Stable) return Fail("game not stable: " + (s.Probe.Wait.Length > 0 ? s.Probe.Wait : "settling"));
        if (sid.Value != _stateId) return Fail($"stale state_id {sid} (current {_stateId})");
        if (idx.Value < 0 || idx.Value >= s.Options.Count) return Fail($"option {idx} out of range (0..{s.Options.Count - 1})");
        Option opt = s.Options[idx.Value];
        string? verr;
        try { verr = opt.Validate(args); } catch (Exception e) { verr = "validation error: " + e.Message; }
        if (verr != null) return Fail("illegal: " + verr);

        _pending = new PendingAct { Opt = opt, Args = args, Due = U.NowMs + DelayMs, Fp = s.Fingerprint, StateId = _stateId };
        LastAction = opt.Label + (args.Target != null ? " -> " + args.Target : "") + (args.Cards != null && args.Cards.Count > 0 ? " [" + string.Join(",", args.Cards) + "]" : "");
        var r = Ok();
        r["state_id"] = _stateId;
        r["scheduled_in_ms"] = DelayMs;
        return r;
    }

    private static void RunPending()
    {
        if (_pending == null) return;
        if (!AgentMode || Paused) { _pending = null; _bump = true; return; }
        if (U.NowMs < _pending.Due) return;
        PendingAct pa = _pending;
        _pending = null;
        // The player may have clicked something during the delay: only act on the exact state the controller saw.
        Snapshot sWithOpts = StateBuilder.Build(_probe, true, withOptions: true);
        if (!_rawStable || sWithOpts.Fingerprint != pa.Fp)
        {
            _bump = true;
            SetError("state changed during the action delay; action '" + pa.Opt.Label + "' dropped", pause: false);
            JsonlLog.Write("act_dropped", new JsonObject { ["action"] = pa.Opt.Label });
            return;
        }
        _execLabel = pa.Opt.Label;
        _execStart = U.NowMs;
        _bump = true;
        U.Info("action: " + LastAction);
        JsonlLog.Write("act_exec", new JsonObject { ["action"] = LastAction, ["state_id"] = pa.StateId, ["option"] = pa.Opt.Json.DeepClone() });
        try { _exec = pa.Opt.Exec(pa.Args); }
        catch (Exception e) { _exec = Task.FromResult<string?>("exception: " + e.Message); }
    }

    public static JsonObject StartRunCmd(JsonObject req, bool fromPanel)
    {
        JsonlLog.Write("start_run", req);
        if (!AgentMode) return Fail("agent mode is off (switch it on in the in-game panel)");
        if (!fromPanel && Paused) return Fail("paused: " + (PauseReason ?? ""));
        if (Busy) return Fail("busy");
        string ch = (U.StrOf(req["character"]) ?? "IRONCLAD").ToUpperInvariant();
        int asc = U.IntOf(req["ascension"]) ?? -1;
        string? seed = U.StrOf(req["seed"]);
        if (ch != "IRONCLAD") return Fail("only IRONCLAD is supported");
        if (asc < 0 || asc > 10) return Fail("ascension must be an integer 0..10");
        string? why = RunStarter.CanStart();
        if (why != null) return Fail(why);
        if (fromPanel) { Paused = false; PauseReason = null; }
        RequestedRun = new JsonObject { ["character"] = ch, ["ascension"] = asc, ["seed"] = string.IsNullOrWhiteSpace(seed) ? null : seed };
        LastAction = $"start run IRONCLAD A{asc}" + (string.IsNullOrWhiteSpace(seed) ? "" : " seed " + seed);
        _bump = true;
        try { _start = RunStarter.Start(ch, asc, seed); }
        catch (Exception e) { _start = Task.FromResult<string?>("exception: " + e.Message); }
        return Ok();
    }

    public static JsonObject PauseCmd(string reason)
    {
        Paused = true;
        PauseReason = reason;
        _pending = null;
        _bump = true;
        JsonlLog.Write("pause", new JsonObject { ["reason"] = reason });
        U.Info("paused: " + reason);
        return Ok();
    }

    public static JsonObject ResumeCmd()
    {
        Paused = false;
        PauseReason = null;
        _bump = true;
        JsonlLog.Write("resume", null);
        U.Info("resumed");
        return Ok();
    }

    private static JsonObject ConfigCmd(JsonObject req)
    {
        JsonlLog.Write("config", req);
        int? d = U.IntOf(req["delay_ms"]);
        if (d != null) { DelayMs = Math.Clamp(d.Value, 0, 10000); SaveConfig(); }
        bool? fm = U.BoolOf(req["fast_mode"]);
        if (fm != null)
        {
            if (!AgentMode) return Fail("agent mode is off: fast_mode not changed");
            SetFastMode(fm.Value);
        }
        var r = Ok();
        r["delay_ms"] = DelayMs;
        r["fast_mode"] = FastMode;
        return r;
    }

    /// <summary>fast_mode=true switches the game's own animation speed setting to Fast (restored when turned off).</summary>
    public static void SetFastMode(bool on)
    {
        try
        {
            var prefs = SaveManager.Instance.PrefsSave;
            if (on && !FastMode)
            {
                _fastModeBefore = prefs.FastMode;
                prefs.FastMode = FastModeType.Fast;
            }
            else if (!on && FastMode && _fastModeBefore != null)
            {
                prefs.FastMode = _fastModeBefore.Value;
                _fastModeBefore = null;
            }
            FastMode = on;
        }
        catch (Exception e) { U.Warn("fast mode change failed: " + e.Message); }
    }

    public static void SetAgentMode(bool on)
    {
        if (AgentMode == on) return;
        AgentMode = on;
        _pending = null;
        _bump = true;
        if (!on && FastMode) SetFastMode(false);
        JsonlLog.Write("agent_mode", new JsonObject { ["on"] = on });
        U.Info("agent mode " + (on ? "ON" : "OFF"));
    }

    public static void TogglePause()
    {
        if (Paused) ResumeCmd();
        else PauseCmd("paused by the player (F9 / panel)");
    }

    // ------------------------------------------------------------------ auto steps (no decision involved)
    private static void AutoSteps()
    {
        if (!AgentMode || Paused || Busy) { _autoDue = 0; return; }
        // Ancient dialogue: lines are clicked through before the options appear.
        if (_probe.Screen == "ANCIENT" && !_probe.Ready)
        {
            var layout = NEventRoom.Instance?.Layout;
            if (layout != null && Screens.AncientDialogueActive(layout))
            {
                string key = _probe.Key + ":dlg";
                if (_autoKey != key || _autoDue == 0) { _autoKey = key; _autoDue = U.NowMs + Math.Max(DelayMs, 700); return; }
                if (U.NowMs >= _autoDue)
                {
                    var hit = Screens.AncientDialogueHitbox(layout);
                    if (U.Usable(hit))
                    {
                        hit!.ForceClick();
                        JsonlLog.Write("auto", new JsonObject { ["step"] = "ancient dialogue" });
                    }
                    _autoDue = U.NowMs + Math.Max(DelayMs, 700);
                }
                return;
            }
        }
        _autoDue = 0;
    }

    private static void Hotkeys()
    {
        bool down = Input.IsPhysicalKeyPressed(Key.F9);
        if (down && !_f9Down && AgentMode) TogglePause();
        _f9Down = down;
    }

    // ------------------------------------------------------------------ panel helpers
    public static string StatusLine()
    {
        string conn = ControllerConnected ? "controller connected" : "no controller";
        string st = !AgentMode ? "agent off" : Paused ? "PAUSED" : _start != null ? "starting run" : _exec != null ? "acting" : _pending != null ? "waiting delay" : _rawStable ? "ready" : "game busy";
        return $"{conn} | {st} | {_probe.Screen}";
    }

    public static string WaitHint() =>
        !_rawStable && !Busy && _probe.Screen == "OTHER" && _probe.Wait.Length > 0 ? _probe.Wait : "";

    public static void PanelStartRun()
    {
        var req = new JsonObject { ["cmd"] = "start_run", ["character"] = "IRONCLAD", ["ascension"] = PanelAscension, ["seed"] = string.IsNullOrWhiteSpace(PanelSeed) ? null : PanelSeed.Trim() };
        var r = StartRunCmd(req, fromPanel: true);
        if (U.BoolOf(r["ok"]) != true) LastError = "start: " + U.StrOf(r["error"]);
    }
}
