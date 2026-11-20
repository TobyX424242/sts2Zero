using System.Text.Json;

namespace Sts2Sim.S3;

/// <summary>
/// ic113 mixed potion valuation (PLAN step 8; `S3 play potion_mix=FILE`; src/ic113/CHAIN.md). The file is written by
/// src/ic113/potion_mix.py from the macro network the actors start a segment with:
/// <code>
/// {"version": 1, "units": "vrun", "macro": PATH, "source": TEXT,
///  "acts": {"1": {"POTION.X": vbar, ...}, "2": {...}, "3": {...}},   // Vbar_act(k), V_run units (probability)
///  "n": {"1": states, ...},
///  "phase2": {"table": {"POTION.X": T, ...}, "c": c, "w": {"1": 0, "2": 0.35, "3": 0.70}}   // optional (phase 2)
/// }
/// </code>
/// dV(s, k) = V_run(s with the belt replaced by the single potion k) - V_run(s with an empty belt), both read from the macro
/// network's value of the live state row (MacroRow.SetPotions; the potion slots of the run unchanged). Vbar_act(k) is the
/// mean of the same quantity over sampled self-play states of act `act` (potion_mix.py, the same row change in Python).
/// <para>Phase 1: V_mix(s, k) = Vbar_act(k) + shrink (dV(s, k) - Vbar_act(k)).</para>
/// <para>Phase 2 (potion_mix_phase=2): w_act T(k) + (1 - w_act) c Vbar_act(k) + c shrink (dV(s, k) - Vbar_act(k)), T(k) in
/// win points from the paired final-boss probe (src/ic113/potion_probe.sh), c = sd_k(T) / sd_k(Vbar_3) (puts the V_run
/// terms on T's scale), w_act = 0 / 0.35 / 0.70 in acts 1 / 2 / 3 (from the file). A kind without T gets the median T.</para>
/// <para>A kind without a Vbar for the act (no states sampled yet, e.g. the chain's first segment, or a kind the file does
/// not list) is scored by its state term alone: Vbar_act(k) := dV(s, k) (phase 1: V_mix = dV; phase 2: w T + (1 - w) c dV).</para>
/// </summary>
public sealed class PotionMix
{
    public readonly Dictionary<int, Dictionary<string, double>> Vbar = new();
    public readonly Dictionary<int, int> N = new();
    public Dictionary<string, double>? Table;
    public double C = double.NaN, TableMedian;
    public readonly Dictionary<int, double> W = new() { [1] = 0.0, [2] = 0.35, [3] = 0.70 };
    public string Path = "", Source = "", Macro = "";
    public int Phase = 1;

    public static PotionMix Load(string path, int phase)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        var m = new PotionMix { Path = path, Phase = phase };
        if (r.TryGetProperty("source", out var src) && src.ValueKind == JsonValueKind.String) m.Source = src.GetString() ?? "";
        if (r.TryGetProperty("macro", out var mac) && mac.ValueKind == JsonValueKind.String) m.Macro = mac.GetString() ?? "";
        if (r.TryGetProperty("acts", out var acts))
            foreach (var a in acts.EnumerateObject())
            {
                int act = int.Parse(a.Name);
                var d = new Dictionary<string, double>();
                foreach (var k in a.Value.EnumerateObject()) if (k.Value.ValueKind == JsonValueKind.Number && double.IsFinite(k.Value.GetDouble())) d[k.Name] = k.Value.GetDouble();
                m.Vbar[act] = d;
            }
        if (r.TryGetProperty("n", out var n))
            foreach (var a in n.EnumerateObject()) m.N[int.Parse(a.Name)] = a.Value.GetInt32();
        if (phase == 2)
        {
            if (!r.TryGetProperty("phase2", out var p2)) throw new InvalidDataException($"potion_mix_phase=2: {path} has no \"phase2\" block (potion_mix.py vbar --table)");
            m.Table = new Dictionary<string, double>();
            foreach (var k in p2.GetProperty("table").EnumerateObject()) m.Table[k.Name] = k.Value.GetDouble();
            if (m.Table.Count == 0) throw new InvalidDataException($"potion_mix_phase=2: {path}: empty table");
            var sorted = m.Table.Values.OrderBy(x => x).ToList();
            m.TableMedian = sorted[sorted.Count / 2];
            m.C = p2.GetProperty("c").GetDouble();
            if (!double.IsFinite(m.C) || m.C <= 0) throw new InvalidDataException($"potion_mix_phase=2: {path}: c must be a positive number");
            if (p2.TryGetProperty("w", out var w)) foreach (var a in w.EnumerateObject()) m.W[int.Parse(a.Name)] = a.Value.GetDouble();
        }
        else if (phase != 1) throw new ArgumentException($"potion_mix_phase {phase} (1|2)");
        return m;
    }

    /// <summary>Vbar_act(kind) if the file has it.</summary>
    public bool TryMean(int act, string kind, out double vbar)
    {
        vbar = 0;
        return Vbar.TryGetValue(act, out var d) && d.TryGetValue(kind, out vbar);
    }

    /// <summary>The score of one potion kind in a state of act <paramref name="act"/> (1-based, clamped to 1-3) whose
    /// state term is <paramref name="dv"/>; <paramref name="hasMean"/> = false when the state term stood in for Vbar.</summary>
    public double Score(int act, string kind, double dv, double shrink, out bool hasMean)
    {
        act = Math.Clamp(act, 1, 3);
        hasMean = TryMean(act, kind, out double vbar);
        if (!hasMean) vbar = dv;
        double mix = vbar + shrink * (dv - vbar);
        if (Phase == 1) return mix;
        double w = W.GetValueOrDefault(act, 0.0);
        double t = Table!.GetValueOrDefault(kind, TableMedian);
        return w * t + (1 - w) * C * vbar + C * shrink * (dv - vbar);
    }

    public string Describe() =>
        $"phase {Phase}, Vbar acts " + string.Join("/", Enumerable.Range(1, 3).Select(a => Vbar.TryGetValue(a, out var d) ? d.Count.ToString() : "0")) +
        " kinds (states " + string.Join("/", Enumerable.Range(1, 3).Select(a => N.GetValueOrDefault(a).ToString())) + ")" +
        (Phase == 2 ? $", table {Table!.Count} kinds (median {TableMedian:F2}), c {C:F3}, w {string.Join("/", Enumerable.Range(1, 3).Select(a => W.GetValueOrDefault(a).ToString("F2")))}" : "") +
        (Source != "" ? $", source {Source}" : "") + (Macro != "" ? $", macro {Macro}" : "");
}
