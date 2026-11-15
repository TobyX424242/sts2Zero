using System.IO.MemoryMappedFiles;

namespace Sts2Sim.S3;

/// <summary>Macro-network answer for one row (see <see cref="ML"/> for the output layout).</summary>
public sealed class MacroEval
{
    public required float[] Out;
    public long ModelVersion;
    /// <summary>Ensemble value P(win) at the row's own HP.</summary>
    public double Value => ML.Sigmoid(Out[ML.OutV]);
    public double ValueStd => Out[ML.OutStd];
    public ReadOnlySpan<float> OptLogits => Out.AsSpan(ML.OutOpt, ML.K);
    public ReadOnlySpan<float> SlotLogits => Out.AsSpan(ML.OutSlot, ML.MaxDeck);
}

public interface IMacroEvaluator
{
    Task<MacroEval> EvaluateAsync(MacroRow row);
}

/// <summary>No network: value 0.5 at any HP, uniform policy. For tests of the C# side.</summary>
public sealed class UniformMacroEvaluator : IMacroEvaluator
{
    public Task<MacroEval> EvaluateAsync(MacroRow row) => Task.FromResult(new MacroEval { Out = new float[ML.OW] });
}

/// <summary>
/// Shared-memory client of the macro half of src/s3/py/s3_server.py. Same ping-pong protocol as the combat
/// bridge (src/s0/S0.Sim/Bridge.cs): the server creates the file; the client writes rows and publishes a
/// request sequence number; the server answers and publishes the response sequence number.
/// <code>
///   0  u64 magic "S3MACRO1"      8 i32 capacity B   12 i32 IW   16 i32 FW   20 i32 OW
///  40  i64 model version (server)
///  64  i64 request seq (client)  72 i32 request rows n
/// 128  i64 response seq (server) 136 i64 model version that answered
/// 192  i32 shutdown flag (client sets 1)
/// 256  ints int32[B, IW]   then floats float32[B, FW]   then out float32[B, OW]
/// </code>
/// </summary>
public sealed unsafe class MacroBridgeClient : IDisposable
{
    public const ulong Magic = 0x314F5243414D3353UL; // "S3MACRO1" little endian
    public const int HeaderBytes = 256;
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _p;
    private long _seq;
    public int Capacity { get; }
    public long LastModelVersion { get; private set; }
    public static TimeSpan ResponseTimeout { get; set; } = TimeSpan.FromSeconds(300);

    public MacroBridgeClient(string path, TimeSpan waitForServer)
    {
        var deadline = DateTime.UtcNow + waitForServer;
        while (!File.Exists(path) || new FileInfo(path).Length < HeaderBytes)
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException("macro bridge file not created by the server: " + path);
            Thread.Sleep(20);
        }
        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.ReadWrite);
        _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.ReadWrite);
        byte* p = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        _p = p;
        while (Volatile.Read(ref *(ulong*)_p) != Magic)
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException("macro bridge header not ready");
            Thread.Sleep(5);
        }
        if (*(int*)(_p + 12) != ML.IW || *(int*)(_p + 16) != ML.FW || *(int*)(_p + 20) != ML.OW)
            throw new InvalidOperationException("macro bridge layout mismatch");
        Capacity = *(int*)(_p + 8);
        _seq = Volatile.Read(ref *(long*)(_p + 64));
    }

    private long IntBytes => (long)Capacity * ML.IW * 4;
    private long FloatBytes => (long)Capacity * ML.FW * 4;
    public Span<int> IntRegion => new(_p + HeaderBytes, Capacity * ML.IW);
    public Span<float> FloatRegion => new(_p + HeaderBytes + IntBytes, Capacity * ML.FW);
    public ReadOnlySpan<float> OutRegion => new(_p + HeaderBytes + IntBytes + FloatBytes, Capacity * ML.OW);

    public void Submit(int n)
    {
        if (n < 1 || n > Capacity) throw new ArgumentOutOfRangeException(nameof(n));
        long k = ++_seq;
        *(int*)(_p + 72) = n;
        Volatile.Write(ref *(long*)(_p + 64), k);
    }

    public bool Ready => Volatile.Read(ref *(long*)(_p + 128)) == _seq;

    public void Wait()
    {
        var spin = new SpinWait();
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        while (Volatile.Read(ref *(long*)(_p + 128)) != _seq)
        {
            spin.SpinOnce();
            if (spin.Count % 1024 == 0 && System.Diagnostics.Stopwatch.GetElapsedTime(start) > ResponseTimeout)
                throw new TimeoutException($"no response from the macro server in {ResponseTimeout.TotalSeconds:F0} s");
        }
        LastModelVersion = Volatile.Read(ref *(long*)(_p + 136));
    }

    public void Shutdown() => Volatile.Write(ref *(int*)(_p + 192), 1);

    public void Dispose()
    {
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
    }
}

/// <summary>Batched macro evaluation: rows queue up while runs wait; <see cref="Submit"/> / <see cref="Complete"/>
/// are called by the actor's flush together with the combat bridge. More pending rows than the bridge capacity are
/// sent in several requests.</summary>
public sealed class MacroBridgeEvaluator : IMacroEvaluator, IDisposable
{
    private readonly MacroBridgeClient _c;
    private readonly Queue<(MacroRow Row, TaskCompletionSource<MacroEval> Tcs)> _pending = new();
    private readonly List<TaskCompletionSource<MacroEval>> _inFlight = new();
    public long Requests, Rows;
    public int Pending => _pending.Count;
    /// <summary>A request has been published and not yet completed.</summary>
    public bool InFlight => _inFlight.Count > 0;
    public bool Ready => _c.Ready;

    public MacroBridgeEvaluator(string path) => _c = new MacroBridgeClient(path, TimeSpan.FromSeconds(300));

    public Task<MacroEval> EvaluateAsync(MacroRow row)
    {
        var tcs = new TaskCompletionSource<MacroEval>();
        _pending.Enqueue((row, tcs));
        return tcs.Task;
    }

    /// <summary>Publishes a request with up to one bridge capacity of pending rows; false if nothing is pending.</summary>
    public bool Submit()
    {
        if (_pending.Count == 0) return false;
        int n = Math.Min(_pending.Count, _c.Capacity);
        var ints = _c.IntRegion; var floats = _c.FloatRegion;
        for (int i = 0; i < n; i++)
        {
            var (row, tcs) = _pending.Dequeue();
            row.I.CopyTo(ints.Slice(i * ML.IW, ML.IW));
            row.F.CopyTo(floats.Slice(i * ML.FW, ML.FW));
            _inFlight.Add(tcs);
        }
        _c.Submit(n);
        Requests++; Rows += n;
        return true;
    }

    /// <summary>Waits for the request published by <see cref="Submit"/> and completes its tasks.</summary>
    public void Complete()
    {
        _c.Wait();
        var o = _c.OutRegion;
        var done = new (TaskCompletionSource<MacroEval>, MacroEval)[_inFlight.Count];
        for (int i = 0; i < _inFlight.Count; i++)
            done[i] = (_inFlight[i], new MacroEval { Out = o.Slice(i * ML.OW, ML.OW).ToArray(), ModelVersion = _c.LastModelVersion });
        _inFlight.Clear();
        foreach (var (tcs, ev) in done) tcs.SetResult(ev);
    }

    public void Shutdown() => _c.Shutdown();
    public void Dispose() => _c.Dispose();
}
