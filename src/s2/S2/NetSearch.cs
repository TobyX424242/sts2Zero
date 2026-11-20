using Sts2Sim.Core.Combat;
using Sts2Sim.S0;

namespace Sts2Sim.S2;

public sealed class NetSearchConfig
{
    /// <summary>Root candidates of Gumbel-top-k (mctx default 16).</summary>
    public int RootCandidates { get; set; } = 16;
    public double CVisit { get; set; } = 50;
    public double CScale { get; set; } = 0.1;
    /// <summary>Floor of the span used to normalise completed Q in sigma (Q is relative T_f, 0..1). 0 = the original
    /// min-max normalisation (span at least 1e-8): the best of a few noisy samples is pushed to the full
    /// (c_visit + max N) c_scale whatever the real gap. With a floor, gaps below it stay proportionally small.</summary>
    public double QSpanMin { get; set; } = 0;
    /// <summary>Win-now bound (src/ic113/STALEMATE.md; &lt; 0 = off, the old search): the network value of a
    /// non-terminal node is capped at the exact T_f of winning in that state (current HP, potions held) minus this
    /// margin. HP and potions rarely rise inside a fight, so no continuation is worth more than winning now; without the
    /// cap a value head that overrates a harmless position by a few thousandths makes the search refuse the kill.</summary>
    public double WinBound { get; set; } = -1;
    /// <summary>Gumbel noise at the root (self-play); off for evaluation.</summary>
    public bool GumbelNoise { get; set; } = true;
    /// <summary>Actions below the root per simulation before the network value is used (cycle guard).</summary>
    public int MaxDepth { get; set; } = 64;
    /// <summary>If the prior's top move has at least this probability, play it without search (A19). Off at &gt; 1.</summary>
    public double ConfidentSkip { get; set; } = 2.0;
    /// <summary>Potions in the action lists below the root (always at the root).</summary>
    public bool PotionsBelowRoot { get; set; } = true;
    /// <summary>G2 arm (c): simulate in exact clones (the true future) instead of resampled worlds. Upper bound only.</summary>
    public bool ExactWorlds { get; set; }
    /// <summary>G2 arm (b): one-turn search instead of the tree search (see <see cref="NetSearch.OneTurnAsync"/>).</summary>
    public bool OneTurn { get; set; }
    /// <summary>Keep the node table from one move to the next of the same fight (tree reuse; see
    /// <see cref="NetSearch.SearchAsync"/>). Off: a fresh table per move (the original behaviour).</summary>
    public bool Reuse { get; set; }
    /// <summary>Tree reuse: the kept table is cleared when it holds more nodes than this after pruning.</summary>
    public int ReuseCap { get; set; } = 5000;
    /// <summary>End-turn prompts at the root (<c>endturn_prompts</c>): end turn is probed once per root on a resampled
    /// clone and, if it opens a prompt, split into one root action per answer (<see cref="ActionSpace.LegalAsync"/>).
    /// Off: the root list of the original build (end-turn prompts answered by the selection policy).</summary>
    public bool EndTurnPrompts { get; set; } = true;
    /// <summary>Common random numbers at the root (<c>crn</c>, 2026-11-15, off by default): the j-th simulation of every
    /// root candidate draws world j (the seed is the candidate's visit count, not the global simulation index), so the
    /// candidates are compared on the same draws and the noise of their Q differences shrinks.</summary>
    public bool Crn { get; set; }
    /// <summary>A copy of these settings (for per-fight variants, e.g. exact worlds in boss fights only).</summary>
    /// <summary>Decomposition of the true-future gain (evaluation only): keep the true draw order (<c>KeepDraw</c>) or
    /// the true enemy randomness (<c>KeepEnemy</c>) in the search's worlds; see <see cref="ResamplingClone.ResampledPartial"/>.</summary>
    public bool KeepDraw { get; set; }
    public bool KeepEnemy { get; set; }
    /// <summary>At most this many distinct sampled worlds per search (0 = a fresh world per simulation, the original).
    /// Simulation k uses world k mod Worlds, so states after a chance event (draws, end turn) recur and the tree can grow
    /// past them (2026-11-15 test of tree shattering; every world is drawn from the information set, no leak).</summary>
    public int Worlds { get; set; }
    public NetSearchConfig Copy() => (NetSearchConfig)MemberwiseClone();
}

public sealed class NetSearchResult
{
    public required SearchAction Action;
    public required List<SearchAction> RootActions;
    public NetEval? RootEval;
    /// <summary>Improved policy over <see cref="RootActions"/> (null when the move was not searched).</summary>
    public float[]? PolicyTarget;
    public int Simulations, Leaves, Terminals, TreeHits, StepsApplied;
    /// <summary>Tree reuse: root visits taken over from the kept table (not counted in <see cref="Simulations"/>).</summary>
    public int ReusedVisits;
    /// <summary>Tree reuse: nodes in the kept table after the search (and its pruning).</summary>
    public int TableNodes;
    public double RootQ;
}

/// <summary>
/// Network-guided search for one combat decision (PROPOSAL.md §6.4), Gumbel MuZero style (Danihelka et al. 2022):
/// <list type="bullet">
/// <item>Root: Gumbel-top-k over the prior logits, sequential halving, completed-Q transform
/// sigma(q) = (c_visit + max N) c_scale q with q min-max normalised over the root's completed Q values; improved
/// policy softmax(logits + sigma(completed Q)) as the policy target.</item>
/// <item>Information sets and chance: every simulation draws a fresh world with
/// <see cref="ResamplingClone.Resampled"/>. Tree nodes are keyed by <see cref="ObservableState.Hash"/> of the state the
/// player sees, so an end turn (or any random effect) leads to a sampled chance outcome with its own node, and the same
/// state reached by different card orders is one node (transpositions; statistics live on the edges).</item>
/// <item>Below the root: the deterministic Gumbel selection argmax(pi'(a) - N(a) / (1 + sum N)).</item>
/// <item>Leaves: a new node is evaluated by the network (value = P(survive) * relative T_f); a finished fight is
/// scored exactly by the T_f table, divided by the fight's T_max.</item>
/// </list>
/// Values are relative T_f in [0, 1]. Prompt options are separate actions at the root; below the root, prompts are
/// answered at random. With <see cref="NetSearchConfig.EndTurnPrompts"/> this includes the first prompt opened by end
/// turn (probed on a resampled clone): each answer is its own root end-turn action (key "E|S:..."), applied in every
/// simulated world through the key script (policy fallback where the prompt does not come or the keys do not match),
/// encoded like a card action's answers and given its own share of the policy target. Under tree reuse these edges do
/// not inherit the kept node's plain "E" edge (its statistics mix random answers).
/// <para>Tree reuse (<see cref="NetSearchConfig.Reuse"/>): one NetSearch per fight keeps its node table between moves.
/// Nodes are keyed by the observable state, so the next move's root is found in the table when the previous search
/// reached that state (the same information set: its statistics came from worlds consistent with what is observed).
/// The root keeps its own node (fresh network evaluation, actions with prompt options), but each root edge whose key
/// matches an edge of the kept node starts with that edge's visits and value sum. Sequential halving then spends the
/// full budget of fresh simulations as before; ranking, the final pick and the completed-Q policy target use reused
/// plus fresh statistics (reuse adds information, as more simulations would). <see cref="NetSearchResult.Simulations"/>
/// counts fresh simulations only. Memory: nodes not touched by this search or the previous one are dropped after every
/// search, and the table is cleared above <see cref="NetSearchConfig.ReuseCap"/> nodes; <see cref="Clear"/> at fight end.</para>
/// </summary>
public sealed class NetSearch(NetSearchConfig cfg, IEvaluator net, ITerminalScorer scorer, BattleContext ctx, double tMax, FightSignals? signals = null)
{
    private sealed class Edge(SearchAction a, float logit)
    {
        public readonly SearchAction Action = a;
        public readonly float Logit = logit;
        public int N;
        public double W;
        public double Q => W / N;
    }

    private sealed class Node(Edge[] edges, double v)
    {
        public readonly Edge[] Edges = edges;
        public readonly double V = v;
        public int SumN;
        /// <summary>Tree reuse: number of the last search that created or passed through this node.</summary>
        public int Stamp;
    }

    private Dictionary<ulong, Node>? _kept;
    private int _searchNo;

    /// <summary>Tree reuse: drops the kept table (fight end).</summary>
    public void Clear() { _kept?.Clear(); _kept = null; }

    public double TerminalValue(CombatState s) => scorer.Score(s, ctx) / tMax;

    /// <summary>Value of a non-terminal node: the network's, capped by <see cref="NetSearchConfig.WinBound"/> if set.</summary>
    private double NodeValue(CombatState s, double v) => cfg.WinBound < 0 ? v : Math.Min(v, TerminalValue(s) - cfg.WinBound);

    public async Task<NetSearchResult> SearchAsync(CombatState root, int simulations, ulong seed)
    {
        var rng = new Random((int)(seed ^ (seed >> 32)));
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var actions = CapRoot(await ActionSpace.LegalAsync(root, endTurnPrompts: cfg.EndTurnPrompts));
        Timing.RootLegal += System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalSeconds;
        var res = new NetSearchResult { Action = actions[0], RootActions = actions };
        if (actions.Count == 1) return res;
        NetEval ev = await net.EvaluateAsync(root, actions, signals);
        res.RootEval = ev;
        float[] logits = ev.Logits;
        int n = actions.Count;
        if (simulations <= 0 || MaxProb(logits) >= cfg.ConfidentSkip)
        {
            res.Action = actions[ArgMax(logits)];
            return res;
        }
        if (cfg.OneTurn) return await OneTurnAsync(root, actions, ev, simulations, seed, rng, res);
        var rootNode = new Node(actions.Select((a, i) => new Edge(a, logits[i])).ToArray(), NodeValue(root, ev.Value));
        Edge[] edges = rootNode.Edges;
        Dictionary<ulong, Node> table;
        if (!cfg.Reuse) table = new Dictionary<ulong, Node>();
        else
        {
            table = _kept ??= new Dictionary<ulong, Node>();
            _searchNo++;
            if (table.TryGetValue(ObservableState.Hash(root), out Node? kept))
            {
                kept.Stamp = _searchNo;
                var byKey = new Dictionary<string, Edge>(kept.Edges.Length, StringComparer.Ordinal);
                foreach (Edge e in kept.Edges) byKey[e.Action.Key] = e;
                foreach (Edge e in edges)
                    if (byKey.TryGetValue(e.Action.Key, out Edge? old) && old.N > 0)
                    {
                        e.N = old.N; e.W = old.W; rootNode.SumN += old.N; res.ReusedVisits += old.N;
                    }
            }
        }
        double[] g = new double[n];
        if (cfg.GumbelNoise)
            for (int i = 0; i < n; i++) g[i] = -Math.Log(-Math.Log(Math.Max(1e-12, rng.NextDouble())));
        // no more candidates than simulations: with a small budget every candidate considered gets at least one visit
        int m = Math.Min(Math.Min(cfg.RootCandidates, n), Math.Max(2, simulations));
        List<int> remaining = Enumerable.Range(0, n).OrderByDescending(i => g[i] + logits[i]).Take(m).ToList();
        int phases = Math.Max(1, (int)Math.Ceiling(Math.Log2(m)));
        int used = 0;
        ulong simSeed = seed * 0x9E3779B97F4A7C15UL;
        for (int phase = 0; phase < phases && used < simulations; phase++)
        {
            int per = Math.Max(1, simulations / (phases * remaining.Count));
            if (phase == phases - 1) per = Math.Max(1, (simulations - used) / remaining.Count);
            for (int s = 0; s < per; s++)
                foreach (int i in remaining)
                {
                    if (used >= simulations) break;
                    await SimulateAsync(root, rootNode, edges[i], table, ResamplingClone.Mix(simSeed, WorldIndex(cfg.Crn || cfg.Worlds > 0 ? edges[i].N : used)), rng, res);
                    used++;
                }
            if (remaining.Count > 1)
            {
                double[] sq = Sigma(rootNode);
                remaining = remaining.OrderByDescending(i => g[i] + logits[i] + sq[i])
                    .Take(phase == phases - 1 ? 1 : (remaining.Count + 1) / 2).ToList();
            }
        }
        double[] sigma = Sigma(rootNode);
        int pick = remaining.OrderByDescending(i => g[i] + logits[i] + sigma[i]).First();
        res.Action = actions[pick];
        res.Simulations = used;
        res.RootQ = rootNode.SumN > 0 ? edges.Where(e => e.N > 0).Sum(e => e.W) / rootNode.SumN : ev.Value;
        var target = new float[n];
        double mx = double.NegativeInfinity;
        for (int i = 0; i < n; i++) mx = Math.Max(mx, logits[i] + sigma[i]);
        double z = 0;
        for (int i = 0; i < n; i++) { double e = Math.Exp(logits[i] + sigma[i] - mx); target[i] = (float)e; z += e; }
        for (int i = 0; i < n; i++) target[i] = (float)(target[i] / z);
        res.PolicyTarget = target;
        if (cfg.Reuse) Prune(table, res);
        return res;
    }

    /// <summary><see cref="Actions.Cap"/> for the root list, keeping every end-turn action of the tail (end-turn
    /// prompt variants, at most 16 by <see cref="ActionSpace.LegalAsync"/>); with a single end turn it is Actions.Cap.</summary>
    public static List<SearchAction> CapRoot(List<SearchAction> a)
    {
        int k = 0;
        while (k < a.Count && a[a.Count - 1 - k].Kind == ActionKind.EndTurn) k++;
        if (k <= 1 || a.Count <= ObsLayout.MaxActions) return Actions.Cap(a);
        k = Math.Min(k, ObsLayout.MaxActions / 2);
        var r = a.Take(ObsLayout.MaxActions - k).ToList();
        r.AddRange(a.Skip(a.Count - k));
        return r;
    }

    private void Prune(Dictionary<ulong, Node> table, NetSearchResult res)
    {
        int keepFrom = _searchNo - 1;
        List<ulong>? drop = null;
        foreach (var (h, node) in table)
            if (node.Stamp < keepFrom) (drop ??= new()).Add(h);
        if (drop is not null) foreach (ulong h in drop) table.Remove(h);
        if (table.Count > cfg.ReuseCap) table.Clear();
        res.TableNodes = table.Count;
    }

    private async Task SimulateAsync(CombatState root, Node rootNode, Edge first, Dictionary<ulong, Node> table, ulong seed,
        Random rng, NetSearchResult stats)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        long tSim = t0;   // start of the current synchronous section (Timing.Sim leaves out the wait for the network)
        CombatState world = World(root, seed);
        SearchCardSelectionSource.InstallOn(world, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(rng));
        Timing.Clone += System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalSeconds;
        var path = new List<(Node Node, Edge Edge)>(16) { (rootNode, first) };
        var seen = new HashSet<ulong>();
        double v;
        Edge edge = first;
        int depth = 0;
        while (true)
        {
            long ta = System.Diagnostics.Stopwatch.GetTimestamp();
            bool ok = await ActionSpace.ApplyAsync(world, edge.Action);
            double dt = System.Diagnostics.Stopwatch.GetElapsedTime(ta).TotalSeconds;
            Timing.Apply += dt; Timing.Steps++;
            if (edge.Action.Kind == ActionKind.EndTurn) { Timing.ApplyEnd += dt; Timing.EndSteps++; }
            stats.StepsApplied++;
            if (!ok) { Timing.ApplyFront += dt; v = path[^1].Node.V; break; }  // not applicable in this world (should not happen): parent's estimate
            if (CombatActions.IsOver(world)) { Timing.ApplyFront += dt; Timing.Terminals++; v = TerminalValue(world); stats.Terminals++; break; }
            long th = System.Diagnostics.Stopwatch.GetTimestamp();
            ulong h = ObservableState.Hash(world);
            Timing.Hash += System.Diagnostics.Stopwatch.GetElapsedTime(th).TotalSeconds;
            if (!table.TryGetValue(h, out Node? node))
            {
                Timing.ApplyFront += dt;
                long tl = System.Diagnostics.Stopwatch.GetTimestamp();
                var legal = Actions.Cap(ActionSpace.Base(world, cfg.PotionsBelowRoot));
                long te = System.Diagnostics.Stopwatch.GetTimestamp();
                Task<NetEval> pending = net.EvaluateAsync(world, legal, signals);
                long tw = System.Diagnostics.Stopwatch.GetTimestamp();
                double f = System.Diagnostics.Stopwatch.Frequency;
                Timing.Legal += (te - tl) / f; Timing.SimEval += (tw - te) / f; Timing.Sim += (tw - tSim) / f;
                NetEval e = await pending;
                tSim = System.Diagnostics.Stopwatch.GetTimestamp();
                node = new Node(legal.Select((a, i) => new Edge(a, e.Logits[i])).ToArray(), NodeValue(world, e.Value)) { Stamp = _searchNo };
                table[h] = node;
                v = node.V;
                stats.Leaves++; Timing.Leaves++;
                break;
            }
            Timing.ApplyReplay += dt; Timing.Hits++;
            stats.TreeHits++;
            node.Stamp = _searchNo;
            if (++depth >= cfg.MaxDepth || !seen.Add(h) || node.Edges.Length == 0) { v = node.V; break; }
            edge = SelectInner(node);
            path.Add((node, edge));
        }
        foreach (var (nd, e) in path) { e.N++; e.W += v; nd.SumN++; }
        Timing.Sim += System.Diagnostics.Stopwatch.GetElapsedTime(tSim).TotalSeconds;
        Timing.Sims++;
    }

    private ulong WorldIndex(int k) => (ulong)(cfg.Worlds > 0 ? k % cfg.Worlds : k);

    /// <summary>Lethal backstop (S3 lethal_backstop): the index in <paramref name="actions"/> of a root action that ends the
    /// fight with a win at once, applied to a resampled clone of <paramref name="root"/> (one world per action, the
    /// search's world kind and random selection policy); among several such actions the one with the best exact end value
    /// (T_f; ties: the earlier action). -1 = none. The live state is not touched.</summary>
    public async Task<int> WinNowAsync(CombatState root, IReadOnlyList<SearchAction> actions, ulong seed)
    {
        var rng = new Random((int)(seed ^ (seed >> 32)));
        int best = -1;
        double bestV = double.NegativeInfinity;
        for (int i = 0; i < actions.Count; i++)
        {
            CombatState world = World(root, ResamplingClone.Mix(seed, (ulong)i));
            SearchCardSelectionSource.InstallOn(world, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(rng));
            if (!await ActionSpace.ApplyAsync(world, actions[i])) continue;
            if (world.Engine is not { IsInProgress: false, Won: true } || world.Players[0].Creature.CurrentHp <= 0) continue;
            double v = TerminalValue(world);
            if (v > bestV) { bestV = v; best = i; }
        }
        return best;
    }

    private CombatState World(CombatState root, ulong seed) => cfg.ExactWorlds ? ResamplingClone.Exact(root)
        : cfg.KeepDraw || cfg.KeepEnemy ? ResamplingClone.ResampledPartial(root, seed, cfg.KeepDraw, cfg.KeepEnemy)
        : ResamplingClone.Resampled(root, seed);

    /// <summary>
    /// One-turn search (G2 arm b): <paramref name="samples"/> times, draw a world, take a root action (round-robin
    /// over the Gumbel-top-k candidates, so every candidate gets samples), then finish the turn by sampling moves from
    /// the network's prior until the turn ends or the fight is over; score the end-of-turn state with the value head
    /// (or exactly if the fight ended). The move played is the candidate with the best mean score.
    /// </summary>
    public async Task<NetSearchResult> OneTurnAsync(CombatState root, List<SearchAction> actions, NetEval ev, int samples,
        ulong seed, Random rng, NetSearchResult res)
    {
        int n = actions.Count;
        float[] logits = ev.Logits;
        double[] g = new double[n];
        if (cfg.GumbelNoise)
            for (int i = 0; i < n; i++) g[i] = -Math.Log(-Math.Log(Math.Max(1e-12, rng.NextDouble())));
        int m = Math.Min(cfg.RootCandidates, n);
        int[] cand = Enumerable.Range(0, n).OrderByDescending(i => g[i] + logits[i]).Take(m).ToArray();
        var sum = new double[n]; var cnt = new int[n];
        ulong simSeed = seed * 0x9E3779B97F4A7C15UL;
        int startRound = root.RoundNumber;
        for (int s = 0; s < samples; s++)
        {
            int i = cand[s % m];
            CombatState world = World(root, ResamplingClone.Mix(simSeed, (ulong)s));
            SearchCardSelectionSource.InstallOn(world, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(rng));
            SearchAction a = actions[i];
            double v;
            while (true)
            {
                await ActionSpace.ApplyAsync(world, a);
                res.StepsApplied++;
                if (CombatActions.IsOver(world)) { v = TerminalValue(world); res.Terminals++; break; }
                var legal = Actions.Cap(ActionSpace.Base(world, cfg.PotionsBelowRoot));
                NetEval e = await net.EvaluateAsync(world, legal, signals);
                res.Leaves++;
                if (world.RoundNumber != startRound || a.Kind == ActionKind.EndTurn) { v = e.Value; break; }
                a = legal[SampleSoftmax(e.Logits, rng)];
            }
            sum[i] += v; cnt[i]++;
        }
        int pick = cand.Where(i => cnt[i] > 0).OrderByDescending(i => sum[i] / cnt[i]).First();
        res.Action = actions[pick];
        res.Simulations = samples;
        res.RootQ = cand.Where(i => cnt[i] > 0).Sum(i => sum[i]) / Math.Max(1, cnt.Sum());
        return res;
    }

    private static int SampleSoftmax(float[] logits, Random rng)
    {
        float mx = logits.Max();
        double z = 0;
        foreach (float l in logits) z += Math.Exp(l - mx);
        double u = rng.NextDouble() * z, c = 0;
        for (int i = 0; i < logits.Length; i++) { c += Math.Exp(logits[i] - mx); if (u <= c) return i; }
        return logits.Length - 1;
    }

    /// <summary>Completed Q (visited: Q, unvisited: v_mix), min-max normalised, through sigma.</summary>
    private double[] Sigma(Node node)
    {
        Edge[] e = node.Edges;
        int n = e.Length, maxN = 0;
        double[] q = CompletedQ(node);
        double lo = q.Min(), hi = q.Max(), span = Math.Max(hi - lo, Math.Max(cfg.QSpanMin, 1e-8));
        foreach (var x in e) maxN = Math.Max(maxN, x.N);
        var s = new double[n];
        for (int i = 0; i < n; i++) s[i] = (cfg.CVisit + maxN) * cfg.CScale * (q[i] - lo) / span;
        return s;
    }

    private static double[] CompletedQ(Node node)
    {
        Edge[] e = node.Edges;
        int n = e.Length;
        double[] pi = Softmax(e.Select(x => (double)x.Logit).ToArray());
        double sumPiVisited = 0, sumPiQ = 0;
        for (int i = 0; i < n; i++)
            if (e[i].N > 0) { sumPiVisited += pi[i]; sumPiQ += pi[i] * e[i].Q; }
        double vMix = node.SumN == 0 || sumPiVisited <= 0 ? node.V
            : (node.V + node.SumN * sumPiQ / sumPiVisited) / (1 + node.SumN);
        var q = new double[n];
        for (int i = 0; i < n; i++) q[i] = e[i].N > 0 ? e[i].Q : vMix;
        return q;
    }

    /// <summary>Gumbel MuZero's non-root rule: argmax of improved policy minus visit share.</summary>
    private Edge SelectInner(Node node)
    {
        Edge[] e = node.Edges;
        if (e.Length == 1) return e[0];
        double[] s = Sigma(node);
        var z = new double[e.Length];
        for (int i = 0; i < e.Length; i++) z[i] = e[i].Logit + s[i];
        double[] pi = Softmax(z);
        int best = 0; double bestScore = double.NegativeInfinity;
        for (int i = 0; i < e.Length; i++)
        {
            double sc = pi[i] - e[i].N / (1.0 + node.SumN);
            if (sc > bestScore) { bestScore = sc; best = i; }
        }
        return e[best];
    }

    private static double[] Softmax(double[] x)
    {
        double mx = x.Max(), z = 0;
        var r = new double[x.Length];
        for (int i = 0; i < x.Length; i++) { r[i] = Math.Exp(x[i] - mx); z += r[i]; }
        for (int i = 0; i < x.Length; i++) r[i] /= z;
        return r;
    }

    public static double MaxProb(float[] logits)
    {
        float mx = logits.Max();
        double z = logits.Sum(l => Math.Exp(l - mx));
        return 1.0 / z;
    }

    public static int ArgMax(float[] x)
    {
        int b = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[b]) b = i;
        return b;
    }
}
