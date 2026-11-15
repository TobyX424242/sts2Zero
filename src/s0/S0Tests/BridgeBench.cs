using System.Diagnostics;
using System.Runtime.InteropServices;
using Sts2Sim.S0;

/// <summary>
/// bridge &lt;bridge file&gt; &lt;obs dir&gt; &lt;seconds per size&gt; &lt;sizes...&gt;: round trips through a running
/// bridge_server.py. Each request copies n encoded states (cycling through obs dir) into shared memory, waits for the
/// answer and reads it back; latency = copy-in + server + read-out. Also checks that the policy logits of padded
/// action rows came back as -inf and valid ones finite.
/// </summary>
static class BridgeBench
{
    public static int Run(string[] a)
    {
        string path = a[1], dir = a[2]; double secs = double.Parse(a[3]);
        int[] sizes = a.Skip(4).Select(int.Parse).ToArray();
        int[] obs = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(dir, "obs.bin"))).ToArray();
        int[] act = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(dir, "act.bin"))).ToArray();
        int ow = ObsLayout.MaxTokens * ObsLayout.TokenWidth, aw = ObsLayout.MaxActions * ObsLayout.ActionWidth;
        int nStates = obs.Length / ow;
        using var c = new BridgeClient(path, TimeSpan.FromSeconds(120));
        Console.WriteLine($"connected: capacity {c.Capacity}, output width {c.OutWidth}, {nStates} distinct states");
        int cursor = 0; long badMask = 0;
        foreach (int n in sizes)
        {
            var lat = new List<double>();
            for (int warm = 0; warm < 3; warm++) Once(n);
            var total = Stopwatch.StartNew();
            long states = 0;
            while (total.Elapsed.TotalSeconds < secs)
            {
                long t0 = Stopwatch.GetTimestamp();
                Once(n);
                lat.Add((Stopwatch.GetTimestamp() - t0) * 1e3 / Stopwatch.Frequency);
                states += n;
            }
            lat.Sort();
            Console.WriteLine($"batch {n,5}: {lat.Count} round trips, latency mean {lat.Average():F2} ms, p50 {lat[lat.Count / 2]:F2} ms, " +
                $"p99 {lat[(int)(lat.Count * 0.99)]:F2} ms; throughput {states / total.Elapsed.TotalSeconds:F0} states/s; model version {c.LastModelVersion}");
        }
        Console.WriteLine($"padding/finiteness check failures: {badMask}");
        c.Shutdown();
        return 0;

        void Once(int n)
        {
            var o = c.ObsRegion; var ac = c.ActRegion;
            int first = cursor;
            for (int i = 0; i < n; i++)
            {
                int s = (cursor + i) % nStates;
                obs.AsSpan(s * ow, ow).CopyTo(o.Slice(i * ow, ow));
                act.AsSpan(s * aw, aw).CopyTo(ac.Slice(i * aw, aw));
            }
            cursor = (cursor + n) % nStates;
            c.Evaluate(n);
            var outp = c.OutRegion;
            for (int i = 0; i < n; i++)
            {
                int s = (first + i) % nStates;
                for (int j = 0; j < ObsLayout.MaxActions; j++)
                {
                    float v = outp[i * c.OutWidth + j];
                    bool valid = act[s * aw + j * ObsLayout.ActionWidth] > 0;
                    if (valid ? !float.IsFinite(v) : !float.IsNegativeInfinity(v)) badMask++;
                }
            }
        }
    }
}
