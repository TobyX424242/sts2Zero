using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Players;

namespace Sts2Sim.S0;

/// <summary>What a terminal scorer needs to know about the fight: battle-start id, max HP at the start and the
/// potion belt at the start (slot order = battle-start <c>potions</c> order).</summary>
public sealed record BattleContext(string Id, int MaxHp, IReadOnlyList<string> Belt)
{
    public static BattleContext From(BattleStartSpec spec) => new(spec.Id, spec.MaxHp, spec.Potions);

    /// <summary>Belt indices still held: slot i still holds the potion id it started with (potions gained in the
    /// fight are ignored; a used potion re-gained into the same slot is counted as held, a known approximation).</summary>
    public List<int> HeldBeltIndices(Player p)
    {
        var held = new List<int>();
        for (int i = 0; i < Belt.Count; i++)
            if (i < p.PotionSlots.Count && p.PotionSlots[i]?.Id.ToString() == Belt[i]) held.Add(i);
        return held;
    }
}

/// <summary>Scores a (terminal or turn-capped) state in [0, 1].</summary>
public interface ITerminalScorer
{
    string Name { get; }
    double Score(CombatState state, BattleContext ctx);
}

/// <summary>End HP / max HP, 0 on death.</summary>
public sealed class HpScorer : ITerminalScorer
{
    public string Name => "hp";
    public double Score(CombatState state, BattleContext ctx)
    {
        var c = state.Players[0].Creature;
        return c.IsDead || c.CurrentHp <= 0 ? 0.0 : (double)c.CurrentHp / Math.Max(1, c.MaxHp);
    }
}

/// <summary>
/// T_f lookup from a per-battle table file (JSON lines: id, max_hp, belt, subsets, tables, final_boss).
/// Score = tables[k][hp] where k is the subset equal to the belt indices still held and hp the end HP (death: the
/// table's death value, 0 for table files and by default). If the held set is not among the subsets, the largest listed subset contained in it is used
/// (<see cref="SubsetFallbacks"/> counts this); HP beyond the table is clamped to its last entry.
/// <para>Free disposal (2026-11-11): at load, every entry is raised to the best entry of the same HP over the subsets it
/// contains (holding an extra potion can never be worth less than not holding it, since it need not be drunk);
/// <see cref="FreeDisposalRaised"/> counts the raised entries (0.18% of entries on the held-out pool).</para>
/// </summary>
public sealed class TfScorer : ITerminalScorer
{
    private sealed record Table(int MaxHp, List<string> Belt, List<int[]> Subsets, List<double[]> Tables, bool FinalBoss, double Death = 0.0);
    private readonly Dictionary<string, Table> _tables = new(StringComparer.Ordinal);
    public string Name => "tf";
    public long SubsetFallbacks;
    public long FreeDisposalRaised;

    public TfScorer(string path) : this(path, null) { }

    /// <summary>Empty scorer for tables built at run time (S3: one table per live fight, <see cref="Set"/>).</summary>
    public TfScorer() { }

    /// <summary>Adds or replaces the table of fight <paramref name="id"/> (S3 live tables). <paramref name="subsets"/>
    /// must be listed by size (smallest first), as in the table files; free disposal is applied. <paramref name="death"/>:
    /// the value of a death in this fight (default 0; S3 <c>death_value=progress</c> gives the run-outcome value of dying
    /// here); the tables' HP-0 entries should hold the same value.</summary>
    public void Set(string id, int maxHp, IEnumerable<string> belt, IEnumerable<int[]> subsets, IEnumerable<double[]> tables, bool finalBoss,
        double death = 0.0)
    {
        var t = new Table(maxHp, belt.ToList(), subsets.Select(s => s.OrderBy(x => x).ToArray()).ToList(), tables.ToList(), finalBoss, death);
        FreeDisposal(t);
        _tables[id] = t;
    }

    public void Remove(string id) => _tables.Remove(id);

    /// <summary>Loads only the battle starts whose id passes <paramref name="keep"/> (read from the line's leading
    /// <c>{"id": "..."</c> before parsing, so a sharded actor skips the rest cheaply).</summary>
    public TfScorer(string path, Func<string, bool>? keep)
    {
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (keep is not null)
            {
                int a = line.IndexOf("\"id\": \"", StringComparison.Ordinal);
                if (a >= 0)
                {
                    int b = line.IndexOf('"', a + 7);
                    if (b > a && !keep(line[(a + 7)..b])) continue;
                }
            }
            using var doc = JsonDocument.Parse(line);
            var r = doc.RootElement;
            var t = new Table(
                r.GetProperty("max_hp").GetInt32(),
                r.GetProperty("belt").EnumerateArray().Select(x => x.GetString()!).ToList(),
                r.GetProperty("subsets").EnumerateArray().Select(s => s.EnumerateArray().Select(x => x.GetInt32()).OrderBy(x => x).ToArray()).ToList(),
                r.GetProperty("tables").EnumerateArray().Select(t => t.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToList(),
                r.TryGetProperty("final_boss", out var fb) && fb.GetBoolean());
            FreeDisposal(t);
            Reprice(t);
            _tables[r.GetProperty("id").GetString()!] = t;
        }
    }

    /// <summary>Altered prices for the re-pricing test (PROPOSAL.md A20), set before loading: "" = as built;
    /// "hp_only" = potions are worth nothing (every subset gets the empty-belt table); "potion3x" = each subset's gain
    /// over the empty belt is tripled.</summary>
    public static string RepriceMode { get; set; } = "";

    private static void Reprice(Table t)
    {
        if (RepriceMode == "") return;
        double[] none = t.Tables[0];
        for (int k = 1; k < t.Tables.Count; k++)
        {
            double[] tk = t.Tables[k];
            for (int h = 1; h < tk.Length && h < none.Length; h++)
                tk[h] = RepriceMode switch
                {
                    "hp_only" => none[h],
                    "potion3x" => none[h] + 3 * (tk[h] - none[h]),
                    _ => throw new ArgumentException("unknown reprice mode " + RepriceMode),
                };
        }
    }

    private void FreeDisposal(Table t)
    {
        // subsets are listed by size, so raising in list order sees every smaller subset already raised
        for (int k = 0; k < t.Subsets.Count; k++)
            for (int j = 0; j < t.Subsets.Count; j++)
            {
                if (j == k || t.Subsets[j].Length >= t.Subsets[k].Length || !t.Subsets[j].All(t.Subsets[k].Contains)) continue;
                double[] tk = t.Tables[k], tj = t.Tables[j];
                for (int h = 1; h < Math.Min(tk.Length, tj.Length); h++)
                    if (tj[h] > tk[h] + 1e-12) { tk[h] = tj[h]; FreeDisposalRaised++; }
            }
    }

    /// <summary>Table max HP (the HP axis runs 0..MaxHp).</summary>
    public int TableMaxHp(string id) => _tables[id].MaxHp;

    /// <summary>Index of the table for the held belt indices (exact match, else the largest listed subset contained in it).</summary>
    public int SubsetIndex(string id, IReadOnlyCollection<int> held)
    {
        Table t = _tables[id];
        int k = -1, best = -1;
        for (int i = 0; i < t.Subsets.Count; i++)
        {
            int[] s = t.Subsets[i];
            if (s.Length == held.Count && s.All(held.Contains)) return i;
            if (s.All(held.Contains) && s.Length > best) { best = s.Length; k = i; }
        }
        SubsetFallbacks++;
        return Math.Max(0, k);
    }

    public double[] TableAt(string id, int k) => _tables[id].Tables[k];

    public bool Has(string id) => _tables.ContainsKey(id);

    /// <summary>Largest table entry of a battle start (the value of winning at full HP with the whole belt): the
    /// scale that turns T_f into the combat network's relative value.</summary>
    public double MaxValue(string id) => _tables[id].Tables.Max(t => t.Max());

    public bool FinalBoss(string id) => _tables[id].FinalBoss;

    /// <summary>T_f of a death in this fight (0 unless set by <see cref="Set"/>).</summary>
    public double DeathValue(string id) => _tables[id].Death;

    /// <summary>T_f of battle start <paramref name="id"/> ending with <paramref name="hp"/> HP (0 = death) and the belt
    /// indices <paramref name="held"/> still held. A held set missing from the table falls back to its largest listed
    /// subset; HP beyond the table is clamped.</summary>
    public double Value(string id, IReadOnlyCollection<int> held, int hp)
    {
        if (hp <= 0) return _tables[id].Death;
        double[] table = _tables[id].Tables[SubsetIndex(id, held)];
        return table[Math.Clamp(hp, 0, table.Length - 1)];
    }

    public double Score(CombatState state, BattleContext ctx)
    {
        var c = state.Players[0].Creature;
        if (c.IsDead || c.CurrentHp <= 0) return _tables[ctx.Id].Death;
        return Value(ctx.Id, ctx.HeldBeltIndices(state.Players[0]), c.CurrentHp);
    }
}

public sealed class SearchConfig
{
    public int Simulations { get; set; } = 200;
    /// <summary>Root candidates sampled by Gumbel-top-k (PROPOSAL 6.4: 4-8).</summary>
    public int RootCandidates { get; set; } = 8;
    public int MaxRolloutTurns { get; set; } = 50;
    public double CVisit { get; set; } = 50;
    public double CScale { get; set; } = 1.0;
    public double Ucb { get; set; } = 0.7;
    /// <summary>Rollout policy: end the turn with this probability while a card is playable.</summary>
    public double RolloutEndTurnP { get; set; } = 0.1;
    /// <summary>Potions are searched in the tree; rollouts do not use them unless this is set.</summary>
    public bool RolloutPotions { get; set; }
    public int MaxTreeDepth { get; set; } = 30;
}

public sealed record SearchResult(SearchAction Action, int Simulations, double Seconds, IReadOnlyList<(SearchAction A, int N, double Q)> Root);

/// <summary>
/// M1 search, no network (PROPOSAL 6.2/6.4). Root: Gumbel-top-k over a uniform prior, then sequential halving with
/// the completed-Q transform sigma(q) = (c_visit + max N) * c_scale * q. Every simulation draws a fresh world with
/// <see cref="ResamplingClone.Resampled"/> and walks an open-loop tree keyed by action keys (single-observer ISMCTS:
/// UCB with availability counts over the actions legal in that world), expands one node, then plays a random
/// rollout to the end of the fight (capped at <see cref="SearchConfig.MaxRolloutTurns"/> turns from the root; a capped
/// rollout is scored on its current state). Card prompts below the root are answered at random; at the root the
/// prompt options are separate actions (<see cref="ActionSpace.LegalAsync"/>).
/// </summary>
public sealed class GumbelSearch(SearchConfig cfg, ITerminalScorer scorer, BattleContext ctx)
{
    private sealed class Node
    {
        public int N, Avail;
        public double W;
        public Dictionary<string, Node>? Children;
        public double Q => N == 0 ? 0 : W / N;
        public Node Child(string key)
        {
            Children ??= new Dictionary<string, Node>(StringComparer.Ordinal);
            if (!Children.TryGetValue(key, out Node? n)) Children[key] = n = new Node();
            return n;
        }
    }

    public long RolloutSteps { get; private set; }

    public async Task<SearchResult> SearchAsync(CombatState root, ulong seed)
    {
        var sw = Stopwatch.StartNew();
        var rng = new System.Random((int)(seed ^ (seed >> 32)));
        List<SearchAction> actions = await ActionSpace.LegalAsync(root);
        if (actions.Count == 1)
            return new SearchResult(actions[0], 0, sw.Elapsed.TotalSeconds, new[] { (actions[0], 0, 0.0) });
        double[] g = actions.Select(_ => -Math.Log(-Math.Log(Math.Max(1e-12, rng.NextDouble())))).ToArray();
        var nodes = actions.Select(_ => new Node()).ToArray();
        int m = Math.Min(cfg.RootCandidates, actions.Count);
        List<int> remaining = Enumerable.Range(0, actions.Count).OrderByDescending(i => g[i]).Take(m).ToList();
        int phases = Math.Max(1, (int)Math.Ceiling(Math.Log2(m)));
        int used = 0;
        ulong simSeed = seed * 0x9E3779B97F4A7C15UL;
        for (int phase = 0; phase < phases && remaining.Count > 0; phase++)
        {
            int left = cfg.Simulations - used;
            int per = phase == phases - 1 ? Math.Max(1, left / remaining.Count)
                : Math.Max(1, cfg.Simulations / (phases * remaining.Count));
            foreach (int i in remaining)
                for (int s = 0; s < per && used < cfg.Simulations; s++, used++)
                    await SimulateAsync(root, actions[i], nodes[i], ResamplingClone.Mix(simSeed, (ulong)used), rng);
            if (remaining.Count > 1)
            {
                int maxN = nodes.Max(n => n.N);
                remaining = remaining.OrderByDescending(i => g[i] + Sigma(nodes[i].Q, maxN))
                    .Take(phase == phases - 1 ? 1 : (remaining.Count + 1) / 2).ToList();
            }
        }
        int pick = remaining[0];
        var rootStats = Enumerable.Range(0, actions.Count).Select(i => (actions[i], nodes[i].N, nodes[i].Q)).ToList();
        return new SearchResult(actions[pick], used, sw.Elapsed.TotalSeconds, rootStats);
    }

    private double Sigma(double q, int maxN) => (cfg.CVisit + maxN) * cfg.CScale * q;

    private async Task SimulateAsync(CombatState root, SearchAction first, Node firstNode, ulong seed, System.Random rng)
    {
        CombatState world = ResamplingClone.Resampled(root, seed);
        var src = SearchCardSelectionSource.InstallOn(world, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(rng));
        var path = new List<Node>(16) { firstNode };
        await ActionSpace.ApplyAsync(world, first);
        Node node = firstNode;
        int depth = 0;
        while (!CombatActions.IsOver(world) && depth < cfg.MaxTreeDepth && node.N > 0)
        {
            var legal = ActionSpace.Base(world, includePotions: true);
            Node? bestChild = null; SearchAction? best = null; double bestU = double.NegativeInfinity;
            List<(SearchAction, Node)>? unvisited = null;
            foreach (var a in legal)
            {
                Node c = node.Child(a.Key);
                c.Avail++;
                if (c.N == 0) (unvisited ??= new()).Add((a, c));
                else
                {
                    double u = c.Q + cfg.Ucb * Math.Sqrt(Math.Log(c.Avail) / c.N);
                    if (u > bestU) { bestU = u; best = a; bestChild = c; }
                }
            }
            if (unvisited is not null) (best, bestChild) = unvisited[rng.Next(unvisited.Count)];
            await ActionSpace.ApplyAsync(world, best!);
            path.Add(bestChild!);
            node = bestChild!;
            depth++;
            if (unvisited is not null) break; // expanded one node
        }
        int startRound = root.RoundNumber;
        while (!CombatActions.IsOver(world) && world.RoundNumber - startRound < cfg.MaxRolloutTurns)
        {
            var legal = ActionSpace.Base(world, includePotions: cfg.RolloutPotions);
            SearchAction a = legal.Count == 1 || rng.NextDouble() < cfg.RolloutEndTurnP ? SearchAction.End
                : legal[rng.Next(legal.Count - 1)];
            await ActionSpace.ApplyAsync(world, a);
            RolloutSteps++;
        }
        double v = scorer.Score(world, ctx);
        foreach (Node n in path) { n.N++; n.W += v; }
    }
}
