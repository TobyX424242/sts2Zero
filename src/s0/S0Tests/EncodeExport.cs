using System.Runtime.InteropServices;
using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.S0;

/// <summary>
/// encode &lt;runs&gt; &lt;states&gt; &lt;outdir&gt; [vocab.json]: encodes combat states of random Defect A10 runs (HP 3000,
/// potions used) with the frozen layout v1 and writes outdir/{obs.bin, act.bin, meta.json, check.jsonl}; writes the
/// vocabulary to vocab.json if it does not exist (otherwise loads it, so the layout stays frozen).
/// check.jsonl holds the C# originals (string ids) for the Python round-trip test.
/// </summary>
static class EncodeExport
{
    public static async Task<int> Run(int runs, int n, string outDir, string vocabPath)
    {
        Directory.CreateDirectory(outDir);
        ObsLayout.Vocab vocab;
        if (File.Exists(vocabPath)) vocab = ObsLayout.Vocab.Load(vocabPath);
        else { vocab = new ObsLayout.Vocab(ObsLayout.BuildVocab()); vocab.Save(vocabPath); Console.WriteLine($"wrote vocabulary {vocabPath}"); }
        var obs = new int[n * ObsLayout.MaxTokens * ObsLayout.TokenWidth];
        var act = new int[n * ObsLayout.MaxActions * ObsLayout.ActionWidth];
        using var check = new StreamWriter(Path.Combine(outDir, "check.jsonl"));
        int count = 0, decision = 0, truncTok = 0, truncAct = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        double encSec = 0;
        var policy = new ActionRunPolicy(new Random(11));
        policy.OnDecision = async state =>
        {
            if (count >= n || decision++ % 3 != 0) return;
            var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var tokens = CombatObservation.Encode(state, ObsLayout.MaxTokens);
            var actions = await ActionSpace.LegalAsync(state);
            ObsLayout.Write(tokens, actions, state, vocab,
                obs.AsSpan(count * ObsLayout.MaxTokens * ObsLayout.TokenWidth, ObsLayout.MaxTokens * ObsLayout.TokenWidth),
                act.AsSpan(count * ObsLayout.MaxActions * ObsLayout.ActionWidth, ObsLayout.MaxActions * ObsLayout.ActionWidth));
            encSec += (System.Diagnostics.Stopwatch.GetTimestamp() - t0) / (double)System.Diagnostics.Stopwatch.Frequency;
            var real = tokens.Where(x => x.Kind != TokenKind.Pad).ToList();
            if (real.Count >= ObsLayout.MaxTokens) truncTok++;
            if (actions.Count > ObsLayout.MaxActions) truncAct++;
            check.WriteLine(JsonSerializer.Serialize(new
            {
                i = count,
                tokens = real.Select(x => new object[] { (int)x.Kind, x.Id, x.Id2, x.F }).ToArray(),
                actions = actions.Take(ObsLayout.MaxActions).Select(a => new object[] { (int)a.Kind, a.Key, a.SourceIndex }).ToArray(),
            }));
            count++;
        };
        for (int i = 0; i < runs && count < n; i++)
        {
            try { await Util.PlayRun($"ENC{i:D7}", 3000, policy); }
            catch (Exception e) { Console.Error.WriteLine($"run {i} failed: {Util.Key(e)}"); }
        }
        File.WriteAllBytes(Path.Combine(outDir, "obs.bin"), MemoryMarshal.AsBytes(obs.AsSpan(0, count * ObsLayout.MaxTokens * ObsLayout.TokenWidth)).ToArray());
        File.WriteAllBytes(Path.Combine(outDir, "act.bin"), MemoryMarshal.AsBytes(act.AsSpan(0, count * ObsLayout.MaxActions * ObsLayout.ActionWidth)).ToArray());
        File.WriteAllText(Path.Combine(outDir, "meta.json"), JsonSerializer.Serialize(new
        {
            version = CombatObservation.Version, n = count, max_tokens = ObsLayout.MaxTokens, token_width = ObsLayout.TokenWidth,
            max_actions = ObsLayout.MaxActions, action_width = ObsLayout.ActionWidth, dtype = "int32", byteorder = "little",
        }));
        Console.WriteLine($"encoded {count} states in {sw.Elapsed.TotalSeconds:F1} s (encode + action enumeration {encSec * 1e6 / Math.Max(1, count):F0} us/state); " +
            $"unknown ids {vocab.Unknown}; states at the token cap {truncTok}, over the action cap {truncAct}");
        return 0;
    }
}
