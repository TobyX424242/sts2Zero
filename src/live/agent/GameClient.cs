using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sts2Sim.Live;

/// <summary>Protocol v1 client (src/live/README.md): TCP, newline-delimited JSON, one request then one reply.
/// Reconnects on failure (the mod accepts a new connection at any time).</summary>
public sealed class GameClient : IDisposable
{
    private readonly string _host;
    private readonly int _port;
    private TcpClient? _tcp;
    private StreamReader? _r;
    private StreamWriter? _w;
    public int TimeoutMs = 30_000;

    public GameClient(string host, int port) { _host = host; _port = port; }

    private void Connect()
    {
        Dispose();
        _tcp = new TcpClient { NoDelay = true, ReceiveTimeout = TimeoutMs, SendTimeout = TimeoutMs };
        _tcp.Connect(_host, _port);
        var s = _tcp.GetStream();
        _r = new StreamReader(s, new UTF8Encoding(false));
        _w = new StreamWriter(s, new UTF8Encoding(false)) { NewLine = "\n", AutoFlush = true };
    }

    /// <summary>One request and its reply; one reconnect and retry on a broken connection.</summary>
    public JsonObject Request(JsonObject req)
    {
        string line = req.ToJsonString();
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                if (_tcp is not { Connected: true }) Connect();
                _w!.WriteLine(line);
                string? reply = _r!.ReadLine() ?? throw new IOException("connection closed by the game");
                return JsonNode.Parse(reply) as JsonObject ?? throw new InvalidDataException("reply is not a JSON object");
            }
            catch (Exception e) when (attempt == 0 && e is IOException or SocketException or ObjectDisposedException)
            {
                Dispose();
            }
        }
    }

    public JsonObject Hello() => Request(new JsonObject { ["cmd"] = "hello", ["client"] = "LiveAgent", ["protocol"] = 1 });
    public JsonObject State() => Request(new JsonObject { ["cmd"] = "state" });
    public JsonObject Pause(string reason) => Request(new JsonObject { ["cmd"] = "pause", ["reason"] = reason });
    public JsonObject Resume() => Request(new JsonObject { ["cmd"] = "resume" });

    public JsonObject StartRun(int ascension, string? seed)
    {
        var o = new JsonObject { ["cmd"] = "start_run", ["character"] = "IRONCLAD", ["ascension"] = ascension };
        if (!string.IsNullOrEmpty(seed)) o["seed"] = seed;
        return Request(o);
    }

    public JsonObject Act(int stateId, Decision d)
    {
        var o = new JsonObject { ["cmd"] = "act", ["state_id"] = stateId, ["option"] = d.Option };
        if (d.Target is { } t) o["target"] = t;
        if (d.Cards is { } c) o["cards"] = new JsonArray(c.Select(x => (JsonNode)x).ToArray());
        return Request(o);
    }

    public JsonObject Config(int? delayMs, bool? fast)
    {
        var o = new JsonObject { ["cmd"] = "config" };
        if (delayMs is { } d) o["delay_ms"] = d;
        if (fast is { } f) o["fast_mode"] = f;
        return Request(o);
    }

    public void Dispose()
    {
        try { _r?.Dispose(); _w?.Dispose(); _tcp?.Dispose(); } catch { }
        _tcp = null; _r = null; _w = null;
    }
}

/// <summary>JSONL log of the controller (src/live/logs/): every state acted on, the decision and the reply. The
/// run save is written only when it changed since the last logged one ("save": "same"), so replay restores it.</summary>
public sealed class LiveLog : IDisposable
{
    private readonly StreamWriter _w;
    private string? _lastSave;
    public string Path { get; }

    public LiveLog(string dir, string tag)
    {
        Directory.CreateDirectory(dir);
        Path = System.IO.Path.Combine(dir, $"{tag}_{DateTime.Now:yyyyMMdd_HHmmss}.jsonl");
        _w = new StreamWriter(Path, append: true) { AutoFlush = true };
    }

    public void Write(string type, JsonObject body)
    {
        body["type"] = type;
        body["t"] = DateTime.UtcNow.ToString("o");
        _w.WriteLine(body.ToJsonString());
    }

    /// <summary>A state with its save replaced by "same" when unchanged.</summary>
    public JsonObject Compact(JsonObject state)
    {
        var copy = (JsonObject)state.DeepClone();
        if (copy.Obj("run") is { } run && run.At("save") is JsonNode save)
        {
            string s = save.ToJsonString();
            if (s == _lastSave) run["save"] = "same";
            else _lastSave = s;
        }
        return copy;
    }

    public static JsonObject DecisionJson(Decision d) => new()
    {
        ["option"] = d.Option, ["target"] = d.Target, ["cards"] = d.Cards is null ? null : new JsonArray(d.Cards.Select(x => (JsonNode)x).ToArray()),
        ["pause"] = d.Pause, ["wait"] = d.Wait, ["why"] = d.Why,
        ["info"] = JsonSerializer.SerializeToNode(d.Info),
    };

    public void Dispose() => _w.Dispose();
}
