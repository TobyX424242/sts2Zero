namespace Sts2Sim.S3;

/// <summary>One recorded macro row of a run: the state after a floor (value example) or a macro decision.</summary>
public sealed class MacroStep
{
    public required MacroRow Row;
    /// <summary>0 = state after a floor, 1 = decision.</summary>
    public int Kind;
    public int Floor, Act, Dtype = ML.DNone, Chosen = -1;
    public float LogP, Value, ValueStd, ValueLogitPolicy;
    public long ModelVersion;
    /// <summary>Deck slots that were legal answers (slot decisions only).</summary>
    public ulong SlotMask;
}

/// <summary>Everything recorded about one run.</summary>
public sealed class RunRecord
{
    public string Seed = "";
    public ulong SeedHash;
    public int Ascension;
    public bool Won, Perturbed, Error;
    public string Outcome = "", ErrorText = "", KilledBy = "";
    public int Floors, ActIndex, FinalHp, MaxHp, Fights, FightsWon, Elites, Bosses, BossesWon, Rests, Smiths, Shops, Events,
        CardsTaken, CardsSkipped, PotionsUsed, Perturbations, DeckSize, Relics;
    public long Simulations, CombatDecisions, ModelVersion, MacroVersion;
    public double Seconds;
    /// <summary>Practice fights (not counted in the fight statistics above; logged only when practice is on).</summary>
    public bool PracticeOn;
    public int PracticeFights, PracticeWon, PracticeAbandoned;
    public long PracticeSimulations, PracticeDecisions;
    public double PracticeSeconds;
    public readonly List<MacroStep> Steps = new();
    public readonly Dictionary<string, int> DecisionCounts = new();

    public Dictionary<string, object?> Log(string tag, int index)
    {
        var d = LogBase(tag, index);
        if (PracticeOn)
        {
            d["practice_fights"] = PracticeFights; d["practice_won"] = PracticeWon; d["practice_abandoned"] = PracticeAbandoned;
            d["practice_simulations"] = PracticeSimulations; d["practice_decisions"] = PracticeDecisions;
            d["practice_seconds"] = Math.Round(PracticeSeconds, 2);
        }
        return d;
    }

    private Dictionary<string, object?> LogBase(string tag, int index) => new()
    {
        ["run"] = index, ["actor"] = tag, ["seed"] = Seed, ["ascension"] = Ascension, ["won"] = Won, ["outcome"] = Outcome,
        ["error"] = Error ? ErrorText : null, ["perturbed"] = Perturbed, ["floors"] = Floors, ["act"] = ActIndex,
        ["final_hp"] = FinalHp, ["max_hp"] = MaxHp, ["killed_by"] = KilledBy, ["fights"] = Fights, ["fights_won"] = FightsWon,
        ["elites"] = Elites, ["bosses"] = Bosses, ["bosses_won"] = BossesWon, ["rests"] = Rests, ["smiths"] = Smiths,
        ["shops"] = Shops, ["events"] = Events, ["cards_taken"] = CardsTaken, ["cards_skipped"] = CardsSkipped,
        ["potions_used"] = PotionsUsed, ["perturbations"] = Perturbations, ["deck"] = DeckSize, ["relics"] = Relics,
        ["simulations"] = Simulations, ["combat_decisions"] = CombatDecisions, ["macro_steps"] = Steps.Count,
        ["decisions"] = DecisionCounts, ["model_version"] = ModelVersion, ["macro_version"] = MacroVersion,
        ["seconds"] = Math.Round(Seconds, 2),
    };
}

/// <summary>
/// Macro training chunks for research/s3/py/train_macro_s3.py. One file per chunk, written as .tmp and renamed:
/// <code>
/// header 64 bytes: "S3RUNS01", i32 version (1), i32 runs R, i32 rows N, i32 IW, i32 FW, i32 run-meta width, i32 row-meta width
/// run meta int32[R, 16]: seed lo, seed hi, ascension, won, floors, act index, final HP, max HP, perturbed, first row,
///                       rows, fights, fights won, elites, combat-model version, macro-model version
/// ints   int16[N, IW]    floats float32[N, FW]
/// row meta float32[N, 16]: run index in chunk, kind (0 floor state, 1 decision), floor, dtype, chosen, log-prob,
///                       value (ensemble P(win)), value-logit std, macro-model version, options, slot mask in four
///                       16-bit pieces (low first), act index, the policy member's own value logit
/// </code>
/// Rows of a run are in play order. Runs that ended in an error are not written.
/// </summary>
public sealed class RunChunkWriter(string dir, string tag, int runsPerChunk)
{
    public const int RunMeta = 16, RowMeta = 16;
    private readonly List<RunRecord> _buf = new();
    private int _seq;
    public long RunsWritten, RowsWritten;

    public void Add(RunRecord r)
    {
        if (r.Error || r.Steps.Count == 0) return;
        _buf.Add(r);
        if (_buf.Count >= runsPerChunk) Flush();
    }

    public void Flush()
    {
        if (_buf.Count == 0) return;
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"{tag}_{_seq++:D6}.s3r");
        int n = _buf.Sum(r => r.Steps.Count);
        using (var w = new BinaryWriter(File.Create(path + ".tmp")))
        {
            w.Write("S3RUNS01"u8.ToArray());
            foreach (int x in new[] { 1, _buf.Count, n, ML.IW, ML.FW, RunMeta, RowMeta }) w.Write(x);
            w.Write(new byte[64 - 8 - 7 * 4]);
            int first = 0;
            foreach (RunRecord r in _buf)
            {
                foreach (int x in new[]
                {
                    (int)(r.SeedHash & 0xFFFFFFFF), (int)(r.SeedHash >> 32), r.Ascension, r.Won ? 1 : 0, r.Floors, r.ActIndex, r.FinalHp,
                    r.MaxHp, r.Perturbed ? 1 : 0, first, r.Steps.Count, r.Fights, r.FightsWon, r.Elites, (int)r.ModelVersion, (int)r.MacroVersion,
                }) w.Write(x);
                first += r.Steps.Count;
            }
            foreach (RunRecord r in _buf) foreach (MacroStep s in r.Steps) foreach (int x in s.Row.I) w.Write((short)Math.Clamp(x, short.MinValue, short.MaxValue));
            foreach (RunRecord r in _buf) foreach (MacroStep s in r.Steps) foreach (float x in s.Row.F) w.Write(x);
            for (int i = 0; i < _buf.Count; i++)
                foreach (MacroStep s in _buf[i].Steps)
                {
                    w.Write((float)i); w.Write((float)s.Kind); w.Write((float)s.Floor); w.Write((float)s.Dtype); w.Write((float)s.Chosen);
                    w.Write(s.LogP); w.Write(s.Value); w.Write(s.ValueStd); w.Write((float)s.ModelVersion); w.Write((float)s.Row.NOptions);
                    for (int k = 0; k < 4; k++) w.Write((float)((s.SlotMask >> (16 * k)) & 0xFFFF));
                    w.Write((float)s.Act); w.Write(s.ValueLogitPolicy);
                }
        }
        File.Move(path + ".tmp", path, overwrite: true);
        RunsWritten += _buf.Count; RowsWritten += n;
        _buf.Clear();
    }
}
