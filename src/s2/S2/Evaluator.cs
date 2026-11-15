using Sts2Sim.Core.Combat;
using Sts2Sim.S0;

namespace Sts2Sim.S2;

/// <summary>Network output for one state: one policy logit per action (in the order the actions were given,
/// after <see cref="Actions.Cap"/>), P(survive) and the state value in relative T_f (T_f / T_max).</summary>
public sealed class NetEval
{
    public float[] Logits = Array.Empty<float>();
    public float Survive;
    public double Value;
    public long ModelVersion;
}

/// <summary>
/// Outcome value head (PROPOSAL.md §4.2, owner's decision 2026-11-11): the combat network predicts the fight's
/// outcome and the macro network's table prices it. Output row after the policy logits: <see cref="Bins"/> logits of
/// the end-HP distribution (bin 0 = death, bin b = end HP in ((b-1)/40, b/40] of the table's max HP) and
/// <see cref="Slots"/> logits of P(the potion in slot i is still held at the end | survive). The scalar head (old: P(survive)
/// logit, relative T_f) is recognised by its output width.
/// </summary>
public static class Outcome
{
    public const int Bins = 41, Slots = 6;
    public static bool IsOutcome(int outWidth) => outWidth == ObsLayout.MaxActions + Bins + Slots;
    public static int Bin(int endHp, int maxHp) => endHp <= 0 ? 0 : Math.Clamp((int)Math.Ceiling(40.0 * endHp / Math.Max(1, maxHp)), 1, Bins - 1);
}

public interface IEvaluator
{
    /// <param name="signals">The fight's T_f signals for the Context token and potion values (null: none).</param>
    Task<NetEval> EvaluateAsync(CombatState state, IReadOnlyList<SearchAction> actions, FightSignals? signals);
}

/// <summary>
/// What the macro network says about the current combat state (layout v3), from the fight's T_f table
/// (T_f = V_run of the post-fight state, PROPOSAL.md §4.2), in thousandths of T_max:
/// <para>Context token, inserted after the global token: [room (0 monster, 1 elite, 2 boss), act index, floor,
/// T_f(current HP, potions held) (the fight's worth if it ended now), T_f(HP) - T_f(HP - 10) (the price of 10 HP here),
/// T_f(HP, held) - T_f(HP, none) (the worth of the potions still held), 1000 T_max (absolute stakes), final boss].</para>
/// <para>Potion tokens: feature 1 = marginal T_f of that potion, T_f(HP, held) - T_f(HP, held without it), for potions
/// of the starting belt; -1 for a potion gained in the fight (not in the table).</para>
/// </summary>
public sealed class FightSignals(TfScorer tf, BattleContext ctx, BattleStartSpec spec, double tMax)
{
    private int Rel(double x) => (int)Math.Round(1000 * x / tMax);
    private readonly Dictionary<int, double[]> _binned = new();   // slot mask -> relative T_f per end-HP bin

    public int HeldMask(CombatState s)
    {
        int m = 0;
        foreach (int i in ctx.HeldBeltIndices(s.Players[0])) if (i < Outcome.Slots) m |= 1 << i;
        return m;
    }

    /// <summary>Relative T_f per end-HP bin for the belt slots in <paramref name="mask"/>: the mean of the table over the
    /// bin's HP range, divided by T_max (bin 0 = death: the table's death value / T_max, 0 unless S3 sets one).</summary>
    private double[] Binned(int mask)
    {
        if (_binned.TryGetValue(mask, out var b)) return b;
        var held = Enumerable.Range(0, Outcome.Slots).Where(i => (mask >> i & 1) == 1).ToList();
        double[] t = tf.TableAt(ctx.Id, tf.SubsetIndex(ctx.Id, held));
        int M = tf.TableMaxHp(ctx.Id);
        b = new double[Outcome.Bins];
        b[0] = tf.DeathValue(ctx.Id) / tMax;
        for (int k = 1; k < Outcome.Bins; k++)
        {
            int lo = (int)Math.Floor((k - 1) * M / 40.0) + 1, hi = (int)Math.Floor(k * M / 40.0);
            if (hi < lo) lo = hi = Math.Max(1, (int)Math.Round(k * M / 40.0));
            double sum = 0;
            for (int h = lo; h <= hi; h++) sum += t[Math.Clamp(h, 0, t.Length - 1)];
            b[k] = sum / (hi - lo + 1) / tMax;
        }
        return _binned[mask] = b;
    }

    /// <summary>Value of a predicted outcome: sum over subsets Q of the held potions (kept independently with
    /// probability <paramref name="keep"/>) and end-HP bins of P(Q) P(bin) T_f(bin, Q) / T_max, plus P(death) x the
    /// death value / T_max when the table has a death value (bin 0; potions do not matter there).</summary>
    public double OutcomeValue(double[] pHp, double[] keep, int heldMask)
    {
        var slots = Enumerable.Range(0, Outcome.Slots).Where(i => (heldMask >> i & 1) == 1).ToArray();
        double v = 0;
        for (int sub = 0; sub < 1 << slots.Length; sub++)
        {
            double pq = 1; int mask = 0;
            for (int j = 0; j < slots.Length; j++)
                if ((sub >> j & 1) == 1) { pq *= keep[slots[j]]; mask |= 1 << slots[j]; } else pq *= 1 - keep[slots[j]];
            if (pq < 1e-9) continue;
            double[] bins = Binned(mask);
            double e = 0;
            for (int k = 1; k < Outcome.Bins; k++) e += pHp[k] * bins[k];
            v += pq * e;
        }
        double d = tf.DeathValue(ctx.Id);
        if (d != 0) v += pHp[0] * d / tMax;   // skipped at d = 0 so that the default is bit-identical to the old sum
        return v;
    }

    public int[] Context(CombatState s, List<int> held, int hp)
    {
        double v = tf.Value(ctx.Id, held, hp);
        return new[]
        {
            spec.RoomType == "elite" ? 1 : spec.RoomType == "boss" ? 2 : 0, spec.ActIndex, spec.Floor,
            Rel(v), Rel(v - tf.Value(ctx.Id, held, Math.Max(1, hp - 10))), Rel(v - tf.Value(ctx.Id, Array.Empty<int>(), hp)),
            (int)Math.Round(1000 * tMax), tf.FinalBoss(ctx.Id) ? 1 : 0,
        };
    }

    public void Apply(CombatState s, List<ObsToken> tokens)
    {
        var p = s.Players[0];
        int hp = Math.Max(1, p.Creature.CurrentHp);
        var held = ctx.HeldBeltIndices(p);
        double v = tf.Value(ctx.Id, held, hp);
        foreach (var t in tokens)
            if (t.Kind == TokenKind.Potion)
            {
                int slot = t.F[0];
                t.F[1] = held.Contains(slot) ? Rel(v - tf.Value(ctx.Id, held.Where(x => x != slot).ToList(), hp)) : -1;
            }
        tokens.Insert(1, new ObsToken(TokenKind.Context, "", "", Context(s, held, hp)));
        if (tokens.Count > ObsLayout.MaxTokens) tokens.RemoveAt(tokens.Count - 1);
    }
}

public static class Encoding
{
    /// <summary>Tokens of a state, with the fight's T_f signals (<see cref="FightSignals"/>) if given.</summary>
    public static List<ObsToken> Tokens(CombatState state, FightSignals? signals)
    {
        var t = CombatObservation.Encode(state, ObsLayout.MaxTokens);
        signals?.Apply(state, t);
        return t;
    }
}

public static class Actions
{
    /// <summary>At most <see cref="ObsLayout.MaxActions"/> actions, end turn always kept (it is last in
    /// <see cref="ActionSpace"/> lists): the list the network sees and the search uses.</summary>
    public static List<SearchAction> Cap(List<SearchAction> a)
    {
        if (a.Count <= ObsLayout.MaxActions) return a;
        var r = a.Take(ObsLayout.MaxActions - 1).ToList();
        r.Add(a[^1]);
        return r;
    }
}

/// <summary>No network: uniform policy, value 0.5. For speed tests of the C# side.</summary>
public sealed class UniformEvaluator : IEvaluator
{
    public Task<NetEval> EvaluateAsync(CombatState state, IReadOnlyList<SearchAction> actions, FightSignals? signals) =>
        Task.FromResult(new NetEval { Logits = new float[actions.Count], Survive = 0.5f, Value = 0.5 });
}

/// <summary>
/// Batched evaluation through the shared-memory bridge (<see cref="BridgeClient"/>). <see cref="EvaluateAsync"/>
/// encodes the state straight into the next free row of the bridge region and returns a pending task;
/// <see cref="Flush"/> (called by the <see cref="Pump"/> when every fight of the process is waiting) sends all
/// pending rows as one request, waits for the answer and completes the tasks. Single-threaded by design.
/// </summary>
public sealed class BridgeEvaluator : IEvaluator, IDisposable
{
    private readonly BridgeClient _c;
    private readonly ObsLayout.Vocab _vocab;
    private List<(TaskCompletionSource<NetEval> Tcs, int NActions, FightSignals? Signals, int HeldMask)> _pending = new(), _spare = new();
    private readonly int _thread = Environment.CurrentManagedThreadId;
    public long Requests, States;
    public double WaitSeconds;
    public int Capacity => _c.Capacity;
    public int Pending => _pending.Count;

    public BridgeEvaluator(string path, ObsLayout.Vocab vocab)
    {
        _c = new BridgeClient(path, TimeSpan.FromSeconds(300));
        _vocab = vocab;
    }

    public Task<NetEval> EvaluateAsync(CombatState state, IReadOnlyList<SearchAction> actions, FightSignals? signals)
    {
        if (Environment.CurrentManagedThreadId != _thread) throw new InvalidOperationException("BridgeEvaluator used from another thread");
        int i = _pending.Count;
        if (i >= _c.Capacity) throw new InvalidOperationException("more pending evaluations than bridge capacity");
        int ow = ObsLayout.MaxTokens * ObsLayout.TokenWidth, aw = ObsLayout.MaxActions * ObsLayout.ActionWidth;
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var tokens = Encoding.Tokens(state, signals);
        ObsLayout.Write(tokens, actions, state, _vocab, _c.ObsRegion.Slice(i * ow, ow), _c.ActRegion.Slice(i * aw, aw));
        Timing.Encode += System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalSeconds;
        var tcs = new TaskCompletionSource<NetEval>();
        _pending.Add((tcs, Math.Min(actions.Count, ObsLayout.MaxActions), signals, signals?.HeldMask(state) ?? 0));
        return tcs.Task;
    }

    /// <summary>Evaluates all pending rows; returns false if there were none.</summary>
    public bool Flush()
    {
        if (!Submit()) return false;
        Complete();
        return true;
    }

    private long _submitT0;
    private int _submitted;

    /// <summary>Publishes the pending rows as one request without waiting (S3: the actor submits the combat and the
    /// macro request together, then waits for both); false if there were none.</summary>
    public bool Submit()
    {
        int n = _pending.Count;
        if (n == 0) return false;
        _submitT0 = System.Diagnostics.Stopwatch.GetTimestamp();
        _submitted = n;
        _c.Submit(n);
        return true;
    }

    /// <summary>Waits for the request published by <see cref="Submit"/> and completes its tasks.</summary>
    public void Complete()
    {
        int n = _submitted;
        long t0 = _submitT0;
        _c.Wait();
        WaitSeconds += (System.Diagnostics.Stopwatch.GetTimestamp() - t0) / (double)System.Diagnostics.Stopwatch.Frequency;
        Requests++; States += n;
        long td = System.Diagnostics.Stopwatch.GetTimestamp();
        var o = _c.OutRegion;
        int w = _c.OutWidth;
        var batch = _pending;
        _pending = _spare; _pending.Clear();
        var results = new NetEval[n];
        bool outcome = Outcome.IsOutcome(w);
        if (!outcome && w != ObsLayout.MaxActions + 2) throw new InvalidOperationException($"unknown output width {w}");
        var pHp = new double[Outcome.Bins];
        var keep = new double[Outcome.Slots];
        for (int i = 0; i < n; i++)
        {
            var row = o.Slice(i * w, w);
            int na = batch[i].NActions;
            var lg = new float[na];
            row[..na].CopyTo(lg);
            var ev = new NetEval { Logits = lg, ModelVersion = _c.LastModelVersion };
            if (!outcome)
            {
                float s = row[ObsLayout.MaxActions], r = row[ObsLayout.MaxActions + 1];
                ev.Survive = 1f / (1f + MathF.Exp(-s));
                ev.Value = ev.Survive * Math.Clamp(r, 0f, 1f);
            }
            else
            {
                var h = row.Slice(ObsLayout.MaxActions, Outcome.Bins);
                float mx = float.NegativeInfinity;
                foreach (float x in h) mx = Math.Max(mx, x);
                double z = 0;
                for (int k = 0; k < Outcome.Bins; k++) { pHp[k] = Math.Exp(h[k] - mx); z += pHp[k]; }
                for (int k = 0; k < Outcome.Bins; k++) pHp[k] /= z;
                for (int j = 0; j < Outcome.Slots; j++) keep[j] = 1.0 / (1.0 + Math.Exp(-row[ObsLayout.MaxActions + Outcome.Bins + j]));
                ev.Survive = (float)(1 - pHp[0]);
                if (batch[i].Signals is { } sig) ev.Value = sig.OutcomeValue(pHp, keep, batch[i].HeldMask);
                else { double e = 0; for (int k = 1; k < Outcome.Bins; k++) e += pHp[k] * (k - 0.5) / 40; ev.Value = e; }
            }
            results[i] = ev;
        }
        Timing.Decode += System.Diagnostics.Stopwatch.GetElapsedTime(td).TotalSeconds;
        for (int i = 0; i < n; i++) batch[i].Tcs.SetResult(results[i]); // continuations run here and refill _pending
        batch.Clear();
        _spare = batch;
    }

    public void Shutdown() => _c.Shutdown();
    public void Dispose() => _c.Dispose();
}

/// <summary>
/// Single-threaded cooperative scheduler: every fight of the process is an async task on this thread; a fight runs
/// until it awaits a network evaluation; when no work is left, <c>onIdle</c> (the evaluator's flush) runs the batch
/// and resumes the waiting fights.
/// </summary>
public sealed class Pump : SynchronizationContext
{
    private readonly Queue<(SendOrPostCallback D, object? S)> _q = new();

    public override void Post(SendOrPostCallback d, object? state) { lock (_q) _q.Enqueue((d, state)); }
    public override void Send(SendOrPostCallback d, object? state) => throw new NotSupportedException();

    public static void Run(Func<Task> main, Func<bool> onIdle)
    {
        var pump = new Pump();
        var prev = Current;
        SetSynchronizationContext(pump);
        try
        {
            Task t = main();
            var idleSince = System.Diagnostics.Stopwatch.StartNew();
            while (true)
            {
                bool did = false;
                while (true)
                {
                    (SendOrPostCallback D, object? S) item;
                    lock (pump._q) { if (!pump._q.TryDequeue(out item)) break; }
                    item.D(item.S);
                    did = true;
                }
                if (t.IsCompleted) break;
                if (onIdle()) did = true;
                if (did) { idleSince.Restart(); continue; }
                if (idleSince.Elapsed.TotalSeconds > 60) throw new InvalidOperationException("pump deadlock: nothing queued, nothing pending");
                Thread.Sleep(0);
            }
            t.GetAwaiter().GetResult();
        }
        finally { SetSynchronizationContext(prev); }
    }
}
