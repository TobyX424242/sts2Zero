using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.S0;

namespace Sts2Sim.S2;

/// <summary>Process-wide CPU time counters (single-threaded actor, so synchronous sections are CPU time).</summary>
public static class Timing
{
    public static double Load, RootLegal, Clone, Apply, Hash, Encode;
    /// <summary>Step 0 profiling (2026-11-13). Sim: all synchronous time inside NetSearch simulations (clone, steps,
    /// hashes, legal actions, leaf encodes, selection, backup; not the wait for the network). Apply split by what the
    /// step led to: ApplyReplay = a step that reached a node already in the table (re-playing the path from the root),
    /// ApplyFront = the step that ended the simulation (new leaf or fight end); ApplyEnd = the end-turn part of Apply.
    /// Legal: legal actions at new nodes below the root. SimEval: synchronous part of leaf evaluation calls (encode +
    /// queueing). Decode: turning a returned batch into NetEvals (before continuations run). Record: training records.</summary>
    public static double Sim, ApplyReplay, ApplyFront, ApplyEnd, Legal, SimEval, Decode, Record;
    public static long Sims, Steps, EndSteps, Leaves, Terminals, Hits;
}

public sealed class FightConfig
{
    public int SimsFull { get; set; } = 64;
    public int SimsFast { get; set; } = 16;
    /// <summary>Playout cap randomisation (Wu 2019): share of moves searched with <see cref="SimsFull"/>; only these
    /// give policy targets.</summary>
    public double PFull { get; set; } = 0.25;
    public int MaxRounds { get; set; } = 50;
    public bool Record { get; set; } = true;
    /// <summary>Give the network the fight's Context token (layout v3).</summary>
    public bool Context { get; set; } = true;
}

/// <summary>One recorded decision: the network input as the search saw it, and the policy target if searched fully.</summary>
public sealed class Decision
{
    public required short[] Obs, Act;
    public float[]? Policy;
    public int NActions, Sims, HeldMask;
    public long ModelVersion;
}

public sealed class FightOutcome
{
    public required string Id, RoomType, Encounter;
    public string Result = "error";
    public string? Error;
    public int EndHeldMask, TableMaxHp;
    public int StartHp, EndHp, MaxHp, Turns, Decisions, Searched, Simulations, PotionsStart, PotionsUsed, Leaves, Steps;
    public double Score, Rel, TMax, Seconds, RootQSum;
    public long ModelVersion;
    /// <summary>S3: a practice copy of a live fight (not part of the run); root visits taken over by tree reuse.</summary>
    public bool Practice;
    public long ReusedVisits;
    /// <summary>S3: the fight's T_f of a death (death_value=progress; 0 otherwise) and the table's source when it is not
    /// the macro value network ("boss2": the double-boss model). Logged by S3 only when the options are on.</summary>
    public double DeathValue;
    public string? TfKind;
    public List<Decision> Records = new();
    public bool Survived => Result == "won";
}

public static class FightRunner
{
    /// <summary>Battle-start id -> simulator seed that reproduces the logged starting enemies.</summary>
    public static readonly Dictionary<string, string> MatchedSeeds = new(StringComparer.Ordinal);

    /// <summary>The fight's start state. With <paramref name="match"/>, the simulator seed whose starting enemies equal
    /// the log is searched once per battle start and cached (evaluation); otherwise <paramref name="simSeed"/> is used
    /// as is (self-play: the encounter is the logged one, a random-roster encounter may roll other monsters).</summary>
    public static async Task<CombatState> LoadBaseAsync(BattleStartSpec spec, string simSeed, bool match)
    {
        long tl = Stopwatch.GetTimestamp();
        CombatState state;
        if (!match) state = await BattleStartLoader.LoadAsync(spec, simSeed);
        else if (MatchedSeeds.TryGetValue(spec.Id, out string? matched)) state = await BattleStartLoader.LoadAsync(spec, matched);
        else
        {
            var (s0, ms, _) = await BattleStartLoader.LoadMatchedAsync(spec, "S2-" + spec.Id);
            MatchedSeeds[spec.Id] = ms;
            state = s0;
        }
        Timing.Load += Stopwatch.GetElapsedTime(tl).TotalSeconds;
        return state;
    }

    /// <summary>Plays one fight from <paramref name="baseState"/> (not modified): the fight's own chance (draw order,
    /// enemy and card RNG) comes from a resampling clone seeded with <paramref name="seed"/>.</summary>
    public static async Task<FightOutcome> PlayAsync(BattleStartSpec spec, CombatState baseState, ulong seed, IEvaluator net,
        TfScorer tf, ObsLayout.Vocab vocab, NetSearchConfig scfg, FightConfig fcfg)
    {
        var sw = Stopwatch.StartNew();
        var o = new FightOutcome { Id = spec.Id, RoomType = spec.RoomType, Encounter = spec.EncounterId, StartHp = spec.Hp, PotionsStart = spec.Potions.Count };
        var rng = new Random((int)(seed ^ (seed >> 32)));
        try
        {
            CombatState state = ResamplingClone.Resampled(baseState, ResamplingClone.Mix(seed, 0xF1647UL));
            SearchCardSelectionSource.InstallOn(state, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(rng.Next())));
            var ctx = BattleContext.From(spec);
            o.TMax = Math.Max(1e-3, tf.MaxValue(spec.Id));
            FightSignals? signals = fcfg.Context ? new FightSignals(tf, ctx, spec, o.TMax) : null;
            var search = new NetSearch(scfg, net, tf, ctx, o.TMax, signals);
            while (!CombatActions.IsOver(state) && state.RoundNumber <= fcfg.MaxRounds)
            {
                bool full = rng.NextDouble() < fcfg.PFull;
                var r = await search.SearchAsync(state, full ? fcfg.SimsFull : fcfg.SimsFast, ResamplingClone.Mix(seed, (ulong)o.Decisions));
                o.Simulations += r.Simulations; o.Leaves += r.Leaves; o.Steps += r.StepsApplied;
                if (r.RootEval is not null) { o.ModelVersion = r.RootEval.ModelVersion; }
                if (r.Simulations > 0) { o.Searched++; o.RootQSum += r.RootQ; }
                if (fcfg.Record && r.RootActions.Count > 1)
                {
                    var d = Encode(state, r, vocab, full, signals);
                    d.HeldMask = signals?.HeldMask(state) ?? 0;
                    o.Records.Add(d);
                }
                if (r.Action.Kind == ActionKind.UsePotion) o.PotionsUsed++;
                if (!await ActionSpace.ApplyAsync(state, r.Action)) { o.Error = "chosen action not applicable: " + r.Action.Key; break; }
                o.Decisions++;
            }
            var pc = state.Players[0].Creature;
            o.EndHp = Math.Max(0, pc.CurrentHp); o.MaxHp = pc.MaxHp; o.Turns = state.RoundNumber;
            bool over = CombatActions.IsOver(state);
            o.Result = o.Error is not null ? "error" : !over ? "capped" : state.Engine!.Won ? "won" : "lost";
            o.Score = o.Result == "won" ? tf.Score(state, ctx) : 0.0;
            o.TableMaxHp = tf.TableMaxHp(spec.Id);
            if (signals is not null) o.EndHeldMask = signals.HeldMask(state);
            o.Rel = o.Score / o.TMax;
        }
        catch (Exception e)
        {
            o.Result = "error";
            o.Error = e.GetType().Name + ": " + (e.Message.Length > 160 ? e.Message[..160] : e.Message);
        }
        o.Seconds = sw.Elapsed.TotalSeconds;
        return o;
    }

    public static Decision Encode(CombatState state, NetSearchResult r, ObsLayout.Vocab vocab, bool full, FightSignals? signals)
    {
        int ow = ObsLayout.MaxTokens * ObsLayout.TokenWidth, aw = ObsLayout.MaxActions * ObsLayout.ActionWidth;
        var obs = new int[ow];
        var act = new int[aw];
        ObsLayout.Write(Encoding.Tokens(state, signals), r.RootActions, state, vocab, obs, act);
        return new Decision
        {
            Obs = obs.Select(Sat).ToArray(), Act = act.Select(Sat).ToArray(),
            Policy = full ? r.PolicyTarget : null, NActions = r.RootActions.Count, Sims = r.Simulations,
            ModelVersion = r.RootEval?.ModelVersion ?? 0,
        };
    }

    private static short Sat(int x) => (short)Math.Clamp(x, short.MinValue, short.MaxValue);

    public static Dictionary<string, object?> LogRecord(FightOutcome o, int fight, string tag) => new()
    {
        ["fight"] = fight, ["actor"] = tag, ["id"] = o.Id, ["room_type"] = o.RoomType, ["encounter"] = o.Encounter,
        ["result"] = o.Result, ["error"] = o.Error, ["start_hp"] = o.StartHp, ["end_hp"] = o.EndHp, ["max_hp"] = o.MaxHp,
        ["score"] = Math.Round(o.Score, 5), ["rel"] = Math.Round(o.Rel, 5), ["t_max"] = Math.Round(o.TMax, 5),
        ["potions_start"] = o.PotionsStart, ["potions_used"] = o.PotionsUsed, ["turns"] = o.Turns,
        ["decisions"] = o.Decisions, ["searched"] = o.Searched, ["simulations"] = o.Simulations, ["leaves"] = o.Leaves,
        ["steps"] = o.Steps, ["mean_root_q"] = o.Searched > 0 ? Math.Round(o.RootQSum / o.Searched, 4) : null,
        ["model_version"] = o.ModelVersion, ["seconds"] = Math.Round(o.Seconds, 3), ["records"] = o.Records.Count,
    };
}

/// <summary>
/// Training-data chunks for src/s2/py/train_combat.py. One file per chunk, written as .tmp and renamed:
/// <code>
/// header 64 bytes: "S2DATA01", i32 version (2), i32 n, i32 T, i32 token width, i32 A, i32 action width, i32 meta width
/// obs  int16[n, T, token width]   act int16[n, A, action width]   policy float32[n, A] (zeros if none)
/// meta float32[n, 12]: has_policy, survived, rel (T_f / T_max at fight end, 0 if lost), n_actions, model_version,
///                     sims, decision index, decisions in fight, end-HP bin (Outcome.Bin; 0 = death),
///                     held-slot mask at the decision, held-slot mask at the fight end, end HP / table max HP
/// </code>
/// </summary>
public sealed class ChunkWriter(string dir, string tag, int chunkSize)
{
    public const int MetaWidth = 12;
    private readonly List<(Decision D, FightOutcome F, int Index)> _buf = new();
    private int _seq;
    public long Written;

    public void Add(FightOutcome f)
    {
        if (f.Result == "error") return;
        for (int i = 0; i < f.Records.Count; i++) _buf.Add((f.Records[i], f, i));
        if (_buf.Count >= chunkSize) Flush();
    }

    public void Flush()
    {
        if (_buf.Count == 0) return;
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"{tag}_{_seq++:D6}.s2d");
        int n = _buf.Count, T = ObsLayout.MaxTokens, TW = ObsLayout.TokenWidth, A = ObsLayout.MaxActions, AW = ObsLayout.ActionWidth;
        using (var w = new BinaryWriter(File.Create(path + ".tmp")))
        {
            w.Write("S2DATA01"u8.ToArray());
            // Header 2 is the format every build reads; the opt-in layout-v5 features are tagged by the sidecar below and
            // do not change the binary layout, so switching them off keeps the bytes identical to the old build.
            foreach (int x in new[] { 2, n, T, TW, A, AW, MetaWidth }) w.Write(x);
            w.Write(new byte[64 - 8 - 7 * 4]);
            foreach (var (d, _, _) in _buf) foreach (short s in d.Obs) w.Write(s);
            foreach (var (d, _, _) in _buf) foreach (short s in d.Act) w.Write(s);
            foreach (var (d, _, _) in _buf)
                for (int j = 0; j < A; j++) w.Write(d.Policy is { } p && j < p.Length ? p[j] : 0f);
            foreach (var (d, f, i) in _buf)
            {
                w.Write(d.Policy is null ? 0f : 1f); w.Write(f.Survived ? 1f : 0f); w.Write((float)f.Rel);
                w.Write((float)d.NActions); w.Write((float)d.ModelVersion); w.Write((float)d.Sims);
                w.Write((float)i); w.Write((float)f.Records.Count);
                int endHp = f.Survived ? f.EndHp : 0;
                w.Write((float)Outcome.Bin(endHp, f.TableMaxHp)); w.Write((float)d.HeldMask); w.Write((float)f.EndHeldMask);
                w.Write((float)Math.Clamp(endHp / (double)Math.Max(1, f.TableMaxHp), 0, 1));
            }
        }
        File.Move(path + ".tmp", path, overwrite: true);
        // Layout-v5/v6 data: a sidecar beside the chunk (the format itself is unchanged, so the off switch writes exactly
        // the old files). It records which opt-in features the actor had on.
        if (CombatObservation.Version >= 5)
            File.WriteAllText(path + ".obs",
                $"obs={CombatObservation.Version} intent_damage={(CombatObservation.ModifiedIntentDamage ? 1 : 0)} " +
                $"relic_counters={(CombatObservation.RelicCounters ? 1 : 0)}\n");
        Written += n;
        _buf.Clear();
    }
}
