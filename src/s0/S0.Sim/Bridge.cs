using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

namespace Sts2Sim.S0;

/// <summary>
/// Shared-memory client for batched network evaluation (protocol v1; server: src/s0/py/bridge_server.py).
/// <para>The server creates the file (e.g. /dev/shm/s0_bridge) and writes the header; the magic is written last.
/// Byte layout (little endian):</para>
/// <code>
///   0  u64 magic "S0BRIDGE"      8 i32 protocol version (1)   12 i32 observation layout version (ObsLayout.Version)
///  16  i32 capacity B            20 i32 T                     24 i32 token width   28 i32 A   32 i32 action width
///  36  i32 output width (A + 2)  40 i64 model version (server; bumped after each weight load)
///  64  i64 request seq (client)  72 i32 request batch size n
/// 128  i64 response seq (server) 136 i64 model version that answered this response
/// 192  i32 shutdown flag (client sets 1)
/// 256  obs  int32[B, T, token width]  then  act int32[B, A, action width]  then  out float32[B, A + 2]
/// </code>
/// <para>Synchronization is a single-producer/single-consumer ping-pong on the two sequence numbers: the client writes
/// obs/act rows, then publishes request seq = k (release store); the server spins on it, evaluates, writes out rows,
/// then publishes response seq = k; the client spins until it sees k (acquire load). One region per actor process;
/// a multi-actor server polls several such files and concatenates the ready requests into one GPU batch.</para>
/// </summary>
public sealed unsafe class BridgeClient : IDisposable
{
    public const ulong Magic = 0x4547444952423053UL; // "S0BRIDGE" little endian
    public const int ProtocolVersion = 1, HeaderBytes = 256;
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _p;
    private long _seq;
    public int Capacity { get; }
    public int OutWidth { get; }
    public long LastModelVersion { get; private set; }

    public BridgeClient(string path, TimeSpan waitForServer)
    {
        var deadline = DateTime.UtcNow + waitForServer;
        while (!File.Exists(path) || new FileInfo(path).Length < HeaderBytes)
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException("bridge file not created by the server: " + path);
            Thread.Sleep(20);
        }
        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.ReadWrite);
        _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.ReadWrite);
        byte* p = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        _p = p;
        while (Volatile.Read(ref *(ulong*)_p) != Magic)
        {
            if (DateTime.UtcNow > deadline) throw new TimeoutException("bridge header not ready");
            Thread.Sleep(5);
        }
        if (I32(8) != ProtocolVersion || I32(12) != ObsLayout.Version || I32(20) != ObsLayout.MaxTokens ||
            I32(24) != ObsLayout.TokenWidth || I32(28) != ObsLayout.MaxActions || I32(32) != ObsLayout.ActionWidth)
            throw new InvalidOperationException("bridge layout/protocol version mismatch");
        Capacity = I32(16);
        OutWidth = I32(36);
        _seq = Volatile.Read(ref *(long*)(_p + 64));
    }

    private int I32(int off) => *(int*)(_p + off);
    private long ObsBytes => (long)Capacity * ObsLayout.MaxTokens * ObsLayout.TokenWidth * 4;
    private long ActBytes => (long)Capacity * ObsLayout.MaxActions * ObsLayout.ActionWidth * 4;

    public Span<int> ObsRegion => new(_p + HeaderBytes, (int)(ObsBytes / 4));
    public Span<int> ActRegion => new(_p + HeaderBytes + ObsBytes, (int)(ActBytes / 4));
    public ReadOnlySpan<float> OutRegion => new(_p + HeaderBytes + ObsBytes + ActBytes, Capacity * OutWidth);

    /// <summary>Evaluates the first <paramref name="n"/> rows already written to <see cref="ObsRegion"/> /
    /// <see cref="ActRegion"/>; returns when the results are in <see cref="OutRegion"/>.</summary>
    public void Evaluate(int n)
    {
        Submit(n);
        Wait();
    }

    /// <summary>Publishes a request for the first <paramref name="n"/> rows without waiting.</summary>
    public void Submit(int n)
    {
        if (n < 1 || n > Capacity) throw new ArgumentOutOfRangeException(nameof(n));
        long k = ++_seq;
        *(int*)(_p + 72) = n;
        Volatile.Write(ref *(long*)(_p + 64), k);           // publish request
    }

    public bool Ready => Volatile.Read(ref *(long*)(_p + 128)) == _seq;

    /// <summary>Waits for the response to the last submitted request. Spins first, then yields/sleeps, so that more
    /// actor processes than cores can share the CPU while their batches are on the GPU.</summary>
    public void Wait()
    {
        var spin = new SpinWait();
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        while (Volatile.Read(ref *(long*)(_p + 128)) != _seq)   // wait for response
        {
            spin.SpinOnce();
            // fail fast: a dead server must not leave a rented box idle (2026-11-11 the server crashed and 24 actors waited 28 min)
            if (spin.Count % 1024 == 0 && System.Diagnostics.Stopwatch.GetElapsedTime(start) > ResponseTimeout)
                throw new TimeoutException($"no response from the inference server in {ResponseTimeout.TotalSeconds:F0} s");
        }
        LastModelVersion = Volatile.Read(ref *(long*)(_p + 136));
    }

    public static TimeSpan ResponseTimeout { get; set; } = TimeSpan.FromSeconds(300);

    public void Shutdown() => Volatile.Write(ref *(int*)(_p + 192), 1);

    public void Dispose()
    {
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
    }
}
