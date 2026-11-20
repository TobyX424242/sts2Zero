using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;

namespace AgentBridge;

/// <summary>
/// Protocol v1 transport: TCP 127.0.0.1:47213, newline-delimited JSON, one request then one reply.
/// Runs on a background thread; every request is handed to the Godot main thread through <see cref="Requests"/>
/// and the reply is written when the main thread has answered (or after a timeout).
/// A new connection replaces the previous one.
/// </summary>
internal sealed class BridgeServer
{
    public const int Port = 47213;

    public sealed class Pending
    {
        public required JsonObject Request;
        public readonly TaskCompletionSource<string> Reply = new(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public readonly ConcurrentQueue<Pending> Requests = new();

    private TcpListener? _listener;
    private Thread? _acceptThread;
    private TcpClient? _client;
    private readonly object _clientLock = new();
    private volatile bool _stopping;

    public volatile bool Connected;
    public string ClientName = "";
    public long LastRequestMs;   // written from the socket thread, read on the main thread (approximate is fine)

    public void Start()
    {
        _listener = new TcpListener(IPAddress.Loopback, Port);
        _listener.Server.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
        _listener.Start();
        _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "AgentBridge-accept" };
        _acceptThread.Start();
        U.Info($"listening on 127.0.0.1:{Port}");
    }

    public void Stop()
    {
        _stopping = true;
        try { _listener?.Stop(); } catch { }
        lock (_clientLock) { try { _client?.Close(); } catch { } }
    }

    private void AcceptLoop()
    {
        while (!_stopping)
        {
            TcpClient c;
            try { c = _listener!.AcceptTcpClient(); }
            catch (Exception e)
            {
                if (_stopping) return;
                U.Warn("accept failed: " + e.Message);
                Thread.Sleep(500);
                continue;
            }
            lock (_clientLock)
            {
                try { _client?.Close(); } catch { }
                _client = c;
            }
            var t = new Thread(() => ClientLoop(c)) { IsBackground = true, Name = "AgentBridge-client" };
            t.Start();
        }
    }

    private void ClientLoop(TcpClient c)
    {
        Connected = true;
        U.Info("controller connected");
        try
        {
            c.NoDelay = true;
            using NetworkStream ns = c.GetStream();
            using var reader = new StreamReader(ns, new UTF8Encoding(false));
            using var writer = new StreamWriter(ns, new UTF8Encoding(false)) { NewLine = "\n", AutoFlush = true };
            while (!_stopping)
            {
                string? line = reader.ReadLine();
                if (line == null) break;
                if (line.Trim().Length == 0) continue;
                LastRequestMs = Environment.TickCount64;
                string reply;
                try
                {
                    JsonNode? node = JsonNode.Parse(line);
                    if (node is not JsonObject req)
                    {
                        reply = Err("request must be a JSON object");
                    }
                    else
                    {
                        var p = new Pending { Request = req };
                        Requests.Enqueue(p);
                        // The main thread answers within a frame or two; a long wait means the game is loading or frozen.
                        if (p.Reply.Task.Wait(TimeSpan.FromSeconds(20)))
                            reply = p.Reply.Task.Result;
                        else
                            reply = Err("timeout: the game's main thread did not answer within 20 s (loading screen?)");
                    }
                }
                catch (Exception e)
                {
                    reply = Err("bad request: " + e.Message);
                }
                lock (_clientLock)
                {
                    if (_client != c) break; // replaced by a newer connection
                }
                writer.WriteLine(reply);
            }
        }
        catch (Exception e)
        {
            if (!_stopping) U.Info("controller connection ended: " + e.Message);
        }
        finally
        {
            lock (_clientLock)
            {
                if (_client == c) { Connected = false; _client = null; }
            }
            try { c.Close(); } catch { }
            U.Info("controller disconnected");
        }
    }

    private static string Err(string msg)
    {
        return new JsonObject { ["ok"] = false, ["error"] = msg }.ToJsonString(U.Compact);
    }
}
