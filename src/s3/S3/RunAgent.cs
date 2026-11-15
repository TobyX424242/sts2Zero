using System.Diagnostics;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Merchant;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Events;
using Sts2Sim.Core.Factories;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Random;
using Sts2Sim.Core.Rewards;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S2;

namespace Sts2Sim.S3;

public sealed class AgentConfig
{
    public int Ascension { get; set; } = 10;
    public int SimsFull { get; set; } = 64;
    public int SimsFast { get; set; } = 16;
    /// <summary>Share of combat moves searched with <see cref="SimsFull"/> (only these give policy targets).</summary>
    public double PFull { get; set; } = 0.25;
    /// <summary>Search budget multiplier in boss fights (1 = none).</summary>
    public double BossSimsFactor { get; set; } = 1.0;
    /// <summary>Search budget multiplier in hallway (normal monster) fights (1 = none).</summary>
    public double HallSimsFactor { get; set; } = 1.0;
    /// <summary>Share of boss-fight moves searched with <see cref="SimsFull"/> (policy targets); negative = <see cref="PFull"/>.</summary>
    public double BossPFull { get; set; } = -1.0;
    /// <summary>Beyond this round the prior's best move is played without search; beyond three times it, end turn.</summary>
    public int MaxRounds { get; set; } = 50;
    public bool RecordCombat { get; set; } = true;
    public bool Context { get; set; } = true;
    /// <summary>Macro decisions: argmax instead of sampling (evaluation).</summary>
    public bool Greedy { get; set; }
    /// <summary>Probability of a perturbed fight start in a flagged run (PROPOSAL.md §6.5).</summary>
    public double PerturbFightP { get; set; } = 0.35;
    /// <summary>"tf": fights scored by T_f from the macro network; "hp": by end HP / max HP (the G3 comparison arm).</summary>
    public string Scorer { get; set; } = "tf";
    /// <summary>Pessimism: T_f = sigmoid(mean logit - kappa x std of the members' logits).</summary>
    public double Kappa { get; set; }
    /// <summary>Practice fights (self-play): at the start of a live boss (elite) fight, this many extra copies of it are
    /// played to the end on resampled clones of the live combat and recorded as training data; the live run is not
    /// affected. 0 = off.</summary>
    public int PracticeBoss { get; set; }
    public int PracticeElite { get; set; }
    /// <summary>Practice fights only from this act on (1-based: 2 = acts 2 and 3).</summary>
    public int PracticeMinAct { get; set; } = 1;
    /// <summary>Share of practice-fight moves searched with <see cref="SimsFull"/> (policy targets); negative = as in the live
    /// fight (boss_pfull / pfull).</summary>
    public double PracticePFull { get; set; } = -1.0;
    /// <summary>Value of a death in a fight scored by T_f: "zero" (default, the original scoring), "progress" or
    /// "progress_raw". Both progress modes start from the run outcome of dying here under the macro value target,
    /// d = (1 - <see cref="DeathAlpha"/>) x acts cleared before this fight / 3 (decision D14: outcome = alpha won +
    /// (1 - alpha) acts cleared / 3). "progress": d is capped at the table's lowest survival entry T_f(1 HP, no potions),
    /// so a death is never worth more than surviving and the table itself is unchanged (the macro network's act-3 tables
    /// at A10 often lie below d: its level error there is larger than the HP effect). "progress_raw": d as it is, and
    /// every table entry below d is raised to d (surviving can never clear fewer acts than dying here).</summary>
    public string DeathValue { get; set; } = "zero";
    /// <summary>alpha of the macro trainer's outcome (window win rate at the target ascension / 0.2, at most 1).</summary>
    public double DeathAlpha { get; set; } = 0.05;
    /// <summary>Share of the progress value a death keeps: d = scale x (1 - alpha) x acts cleared / 3 (1 = the full value of
    /// the macro target; below 1 a death stays priced well under any survival and a won run stays the dominant goal).</summary>
    public double DeathScale { get; set; } = 1.0;
    /// <summary>Double-boss model (null = off): the first of two act-3 bosses is scored by
    /// T_f(h, P) = d + (1 - d) W2(h / max HP, |P|) instead of the macro value network (d = the fight's death value).</summary>
    public Boss2Model? Boss2 { get; set; }
    /// <summary>Stake-priced tables (2026-11-14, off by default): every T_f table of a fight that does not end the run is
    /// rescaled to the part of the run outcome not yet secured, T' = (T - b) / (1 - b) with b = (1 - <see cref="DeathAlpha"/>)
    /// x acts cleared / 3 (the outcome of losing the run here, decision D14), floored at <see cref="StakeFloor"/> x T so that
    /// HP and potion order survive where the value sits under b. A death stays 0, which is now exact. The search is
    /// unchanged by an affine map of the table except through the death value; what changes is every price the combat
    /// network reads (Context token and potion features are relative to the table's maximum), which at A10 in act 3 are
    /// today dominated by the 0.55 already banked.</summary>
    public bool TfStake { get; set; }
    public double StakeFloor { get; set; } = 0.02;
    public NetSearchConfig Search { get; set; } = new();
}

/// <summary>W2: chance of winning the second act-3 boss given the HP share and the number of potions held at its start,
/// a logistic fitted on self-play logs by research/s3/fit_boss2.py (JSON: b0, b_hp, b_pot, b_hp2, pot_cap).</summary>
public sealed class Boss2Model
{
    public double B0, BHp, BPot, BHp2;
    public int PotCap = 3;
    public string Source = "";

    public static Boss2Model Load(string path)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        double G(string k, double d) => r.TryGetProperty(k, out var v) ? v.GetDouble() : d;
        var m = new Boss2Model { B0 = G("b0", double.NaN), BHp = G("b_hp", double.NaN), BPot = G("b_pot", double.NaN), BHp2 = G("b_hp2", 0), PotCap = (int)G("pot_cap", 3), Source = path };
        if (!double.IsFinite(m.B0) || !double.IsFinite(m.BHp) || !double.IsFinite(m.BPot) || !double.IsFinite(m.BHp2))
            throw new InvalidDataException($"boss2 model {path}: b0, b_hp, b_pot must be finite numbers");
        return m;
    }

    public double W2(double hpShare, int potions)
    {
        double h = Math.Clamp(hpShare, 0, 1);
        return ML.Sigmoid(B0 + BHp * h + BHp2 * h * h + BPot * Math.Min(potions, PotCap));
    }

    public override string ToString() => $"b0 {B0:F4} b_hp {BHp:F4} b_pot {BPot:F4} b_hp2 {BHp2:F4} pot_cap {PotCap} ({Source})";
}

/// <summary>What the runs of one actor process share.</summary>
public sealed class ActorServices
{
    public required IEvaluator Net;
    public required IMacroEvaluator Macro;
    public required ObsLayout.Vocab Vocab;
    public required MacroVocab MVocab;
    public required AgentConfig Cfg;
    public readonly TfScorer Tf = new();
    public ChunkWriter? CombatWriter;
    public Action<FightOutcome, RunRecord, int>? FightLog;
    public long Simulations, CombatDecisions, MacroDecisions, TfRows, TfBuilds, Fights, FightsWon, Progress;
    /// <summary>Practice fights (not in the counters above): finished, won, abandoned (round cap or error), their cost.</summary>
    public long PracticeFights, PracticeWon, PracticeAbandoned, PracticeSimulations, PracticeDecisions;
    public double PracticeSeconds;
    /// <summary>Tree reuse: searched moves, root visits reused over them, largest kept table seen.</summary>
    public long ReuseSearches, ReusedVisits, ReuseTableMax;
    /// <summary>The last agent callback (run seed, callback name, floor): printed by the actor's stall check.</summary>
    public volatile string WhereSeed = "", WhereWho = "";
    public volatile int WhereFloor;
    /// <summary>Largest difference between the value logit rebuilt from a served row's (a, c) at the row's own HP and
    /// the logit the network computed there: a running check of the HP-axis formula behind every live T_f table.</summary>
    public double TfFormulaMaxError;
    /// <summary>Debugging: print one line per callback kind and floor.</summary>
    public bool Trace;
    /// <summary>T_f table dump (tf_dump=FILE): the first <see cref="TfDumpK"/> live fights of each kind (act-3 boss, hallway
    /// fight of each act) are written as JSON lines.</summary>
    public StreamWriter? TfDump;
    public int TfDumpK = 20;
    public readonly Dictionary<string, int> TfDumped = new();
    /// <summary>death_value=progress_raw: table entries of survived fights raised to the death value; death_value=progress:
    /// tables whose death value was capped at T_f(1 HP, no potions); fights scored by the double-boss model.</summary>
    public long TfDeathFloorRaised, TfDeathCapped, TfBoss2Fights;
    /// <summary>tf_stake: fights whose table was rescaled, table entries that took the floor, sum of b over those fights.</summary>
    public long TfStakeFights, TfStakeFloored, TfMsFights;
    public double TfStakeBankedSum;
    public readonly Dictionary<string, long> SelectionKinds = new(), Notes = new();
    public void Note(string k) => Notes[k] = Notes.GetValueOrDefault(k) + 1;
}

/// <summary>
/// The S3 agent for one run (PROPOSAL.md §6.5): every out-of-combat decision goes to the macro policy, every fight is
/// played by the network-guided search (research/s2 NetSearch) on the live combat, scored by a T_f table built at the
/// fight start from the current macro value network (V_run of the post-fight state, §4.2).
/// <para>Macro decisions and how they are made: card rewards, Ancient choices, rest-site options, smith and removal
/// targets and shop purchases by the S1-trained heads; map points and event options by heads that start uniform
/// (no human data) and are trained by PPO; potion rewards are taken when a slot is free; custom event interactions
/// use the simulator's default.</para>
/// </summary>
public sealed class RunAgent : IRunDecisionSource, ICombatCompletionValidator
{
    private sealed class LiveFight
    {
        public required CombatEngine Engine;
        public required string Id;
        public required BattleContext Ctx;
        public required NetSearch Search;
        public required FightOutcome O;
        public required RoomType Room;
        public FightSignals? Signals;
        public ulong Seed;
        public Stopwatch Watch = Stopwatch.StartNew();
    }

    private readonly ActorServices _s;
    private readonly AgentConfig _cfg;
    private readonly RunState _run;
    private readonly Player _p;
    private readonly Random _rng;
    private readonly FloorFacts _facts = new();
    private readonly ulong _seed;
    private readonly int _asc;
    private string _lastWho = "";
    private int _lastTraceFloor = -1;
    private readonly Queue<string[]> _selScript = new();
    /// <summary>The queued answers in <see cref="_selScript"/> belong to an end-turn action (statistics only).</summary>
    private bool _selEndTurn;
    private readonly Stopwatch _watch = Stopwatch.StartNew();
    private LiveFight? _fight;
    private int _lastSnapFloor, _fightIndex, _shopSteps, _shopFloor = -1;
    private ActMap? _featMap;
    private readonly Dictionary<MapPoint, (int[] Min, int[] Max)> _feat = new();
    public RunRecord Rec { get; }
    public Dictionary<RelicModel, int> RelicFloors => _facts.RelicFloor;

    /// <param name="ascension">This run's ascension level (self-play may mix levels; -1 = the configured one).</param>
    public RunAgent(ActorServices services, RunState run, Player player, string seed, bool perturbed, int ascension = -1)
    {
        _s = services; _cfg = services.Cfg; _run = run; _p = player;
        _seed = ActionSpace.Fnv("S3:" + seed);
        _rng = new Random((int)(_seed ^ (_seed >> 32)));
        _asc = ascension >= 0 ? ascension : _cfg.Ascension;
        _facts.Ascension = _asc;
        Rec = new RunRecord
        {
            Seed = seed, SeedHash = _seed, Ascension = _asc, Perturbed = perturbed,
            PracticeOn = _cfg.PracticeBoss > 0 || _cfg.PracticeElite > 0,
        };
    }

    // ------------------------------------------------------------------ run bookkeeping

    private void Pre([System.Runtime.CompilerServices.CallerMemberName] string who = "")
    {
        _s.Progress++;
        _s.WhereSeed = Rec.Seed; _s.WhereWho = who; _s.WhereFloor = _run.TotalFloor;
        if (_s.Trace && (who != _lastWho || _run.TotalFloor != _lastTraceFloor))
        {
            _lastWho = who; _lastTraceFloor = _run.TotalFloor;
            Console.WriteLine($"TRACE {Rec.Seed} floor {_run.TotalFloor} act {_run.CurrentActIndex} {_run.CurrentMapPoint?.PointType} room {_run.CurrentRoom?.GetType().Name} " +
                $"{(_run.BaseRoom is { } b ? FloorFacts.RoomModelId(b) : "")} hp {_p.Creature.CurrentHp}/{_p.Creature.MaxHp} gold {_p.Gold} {who}");
        }
        _facts.NoteRoom(_run);
        _facts.NoteRelics(_p, _run.TotalFloor);
    }

    private bool IsFinalBossPoint() =>
        _run.CurrentActIndex >= _run.Acts.Count - 1 && _run.CurrentMapPoint is { PointType: MapPointType.Boss } pt &&
        (_run.Map.SecondBossMapPoint is not { } second || pt.coord.Equals(second.coord));

    /// <summary>The first boss of the last act when the act has a second boss (ascension 10: two bosses back to back).</summary>
    private bool IsFirstOfDoubleBoss() =>
        _run.CurrentActIndex >= _run.Acts.Count - 1 && _run.CurrentMapPoint is { PointType: MapPointType.Boss } pt &&
        _run.Map.SecondBossMapPoint is { } second && !pt.coord.Equals(second.coord);

    /// <summary>T_f of dying in the current fight: 0, or with death_value=progress the run outcome of a death here,
    /// (1 - alpha) x acts cleared / 3 (the act index; the trainer counts acts cleared as min(act index, 2) for a lost run).</summary>
    private double DeathValueHere() =>
        _cfg.DeathValue != "zero" && _cfg.Scorer == "tf" ? _cfg.DeathScale * (1 - _cfg.DeathAlpha) * Math.Min(_run.CurrentActIndex, 2) / 3.0 : 0.0;

    /// <summary>RunDriver.OnRoomResolved: the floor is over (room exited, before any act change).</summary>
    public void OnRoomResolved(MapPoint point, RoomType roomType)
    {
        Pre();
        SnapshotFloor();
    }

    /// <summary>Value example: the state after the current floor (as in the S1 dataset: not the run's last floor, not after death).</summary>
    private void SnapshotFloor()
    {
        int g = _run.TotalFloor;
        if (g <= _lastSnapFloor) return;
        _lastSnapFloor = g;
        if (_p.Creature.CurrentHp <= 0 || _run.IsGameOver || IsFinalBossPoint()) return;
        MacroRow row = MacroEncoder.Encode(_run, _p, _facts, _s.MVocab);
        Rec.Steps.Add(new MacroStep { Row = row, Kind = 0, Floor = g, Act = _run.CurrentActIndex });
    }

    public void Finish(RunDriver.Result res)
    {
        Rec.Won = res.Won; Rec.Outcome = res.Outcome.ToString();
        FillEnd();
    }

    public void Fail(Exception e)
    {
        Rec.Error = true; Rec.Outcome = "error";
        Rec.ErrorText = e.GetType().Name + ": " + (e.Message.Length > 200 ? e.Message[..200] : e.Message);
        if (_fight is not null) { _fight.Search.Clear(); _s.Tf.Remove(_fight.Id); _fight = null; }
        FillEnd();
    }

    private void FillEnd()
    {
        Rec.Floors = _run.TotalFloor; Rec.ActIndex = _run.CurrentActIndex;
        Rec.FinalHp = Math.Max(0, _p.Creature.CurrentHp); Rec.MaxHp = _p.Creature.MaxHp;
        Rec.DeckSize = _p.Deck.Cards.Count; Rec.Relics = _p.Relics.Count;
        Rec.Seconds = _watch.Elapsed.TotalSeconds;
    }

    // ------------------------------------------------------------------ macro decisions

    private static double[] Softmax(ReadOnlySpan<float> logits)
    {
        float mx = float.NegativeInfinity;
        foreach (float x in logits) mx = Math.Max(mx, x);
        var p = new double[logits.Length];
        double z = 0;
        for (int i = 0; i < p.Length; i++) { p[i] = Math.Exp(logits[i] - mx); z += p[i]; }
        for (int i = 0; i < p.Length; i++) p[i] /= z;
        return p;
    }

    /// <param name="greedy">Evaluation takes the most probable option for the decision types S1 trained; the heads
    /// without human data (map, events) are sampled also in evaluation: their argmax repeats itself on repeatable event
    /// options (a run was lost by holding on at Slippery Bridge until death).</param>
    private int Pick(double[] p, bool greedy)
    {
        if (greedy)
        {
            int b = 0;
            for (int i = 1; i < p.Length; i++) if (p[i] > p[b]) b = i;
            return b;
        }
        double u = _rng.NextDouble(), c = 0;
        for (int i = 0; i < p.Length; i++) { c += p[i]; if (u <= c) return i; }
        return p.Length - 1;
    }

    private void Record(MacroRow row, int dtype, int chosen, double prob, MacroEval ev, ulong slotMask)
    {
        _s.MacroDecisions++;
        Rec.MacroVersion = ev.ModelVersion;
        string name = ML.DtNames[dtype];
        Rec.DecisionCounts[name] = Rec.DecisionCounts.GetValueOrDefault(name) + 1;
        Rec.Steps.Add(new MacroStep
        {
            Row = row, Kind = 1, Floor = _run.TotalFloor, Act = _run.CurrentActIndex, Dtype = dtype, Chosen = chosen,
            LogP = (float)Math.Log(Math.Max(prob, 1e-12)), Value = (float)ev.Value, ValueStd = (float)ev.ValueStd,
            ValueLogitPolicy = ev.Out[ML.OutVPol], ModelVersion = ev.ModelVersion, SlotMask = slotMask,
        });
    }

    /// <summary>Samples one of the options already added to <paramref name="row"/>; returns its index.</summary>
    private async Task<int> DecideOptionsAsync(MacroRow row, int dtype)
    {
        row.I[ML.ODtype] = dtype;
        int n = row.NOptions;
        if (n <= 1) return 0;
        MacroEval ev = await _s.Macro.EvaluateAsync(row);
        double[] p = Softmax(ev.OptLogits[..n]);
        int pick = Pick(p, _cfg.Greedy && dtype != ML.DMap && dtype != ML.DEvent);
        Record(row, dtype, pick, p[pick], ev, 0);
        return pick;
    }

    /// <summary>Samples one deck slot among <paramref name="slots"/> (all &lt; <see cref="ML.MaxDeck"/>).</summary>
    private async Task<int> DecideSlotAsync(MacroRow row, int dtype, IReadOnlyList<int> slots)
    {
        if (slots.Count == 1) return slots[0];
        row.ClearDecision();
        row.I[ML.ODtype] = dtype;
        MacroEval ev = await _s.Macro.EvaluateAsync(row);
        var lg = new float[slots.Count];
        ulong mask = 0;
        for (int i = 0; i < slots.Count; i++) { lg[i] = ev.SlotLogits[slots[i]]; mask |= 1UL << slots[i]; }
        double[] p = Softmax(lg);
        int pick = Pick(p, _cfg.Greedy);
        Record(row, dtype, slots[pick], p[pick], ev, mask);
        return slots[pick];
    }

    private MacroRow StateRow() => MacroEncoder.Encode(_run, _p, _facts, _s.MVocab);

    private int DeckIndex(CardModel c)
    {
        var deck = _p.Deck.Cards;
        for (int i = 0; i < deck.Count; i++) if (ReferenceEquals(deck[i], c)) return i;
        return -1;
    }

    /// <summary>Chooses one of <paramref name="cards"/> (deck cards) with the smith or removal head.</summary>
    private async Task<CardModel> DecideDeckCardAsync(IReadOnlyList<CardModel> cards, int dtype)
    {
        var bySlot = new Dictionary<int, CardModel>();
        foreach (CardModel c in cards)
        {
            int i = DeckIndex(c);
            if (i >= 0 && i < ML.MaxDeck) bySlot[i] = c;
        }
        if (bySlot.Count == 0) return cards[_rng.Next(cards.Count)];
        int slot = await DecideSlotAsync(StateRow(), dtype, bySlot.Keys.OrderBy(x => x).ToList());
        return bySlot[slot];
    }

    // ---- map

    private static readonly MapPointType[] MinTypes = { MapPointType.Monster, MapPointType.Elite, MapPointType.RestSite, MapPointType.Shop, MapPointType.Treasure, MapPointType.Unknown };
    private static readonly MapPointType[] MaxTypes = { MapPointType.Monster, MapPointType.Elite, MapPointType.RestSite, MapPointType.Shop, MapPointType.Unknown };

    /// <summary>Per room type, the fewest and the most rooms of that type on any path from <paramref name="pt"/>
    /// (inclusive) to the end of the act.</summary>
    private (int[] Min, int[] Max) PathCounts(MapPoint pt)
    {
        if (!ReferenceEquals(_featMap, _run.Map)) { _feat.Clear(); _featMap = _run.Map; }
        if (_feat.TryGetValue(pt, out var got)) return got;
        var mn = new int[MinTypes.Length]; var mx = new int[MaxTypes.Length];
        bool first = true;
        foreach (MapPoint ch in pt.Children)
        {
            var (cmn, cmx) = PathCounts(ch);
            for (int i = 0; i < mn.Length; i++) mn[i] = first ? cmn[i] : Math.Min(mn[i], cmn[i]);
            for (int i = 0; i < mx.Length; i++) mx[i] = first ? cmx[i] : Math.Max(mx[i], cmx[i]);
            first = false;
        }
        for (int i = 0; i < mn.Length; i++) if (pt.PointType == MinTypes[i]) mn[i]++;
        for (int i = 0; i < mx.Length; i++) if (pt.PointType == MaxTypes[i]) mx[i]++;
        return _feat[pt] = (mn, mx);
    }

    private float[] MapFeatures(MapPoint pt)
    {
        var (mn, mx) = PathCounts(pt);
        var f = new float[ML.MF];
        f[0] = pt.coord.row / 20f;
        for (int i = 0; i < mn.Length; i++) f[1 + i] = mn[i] / 5f;
        for (int i = 0; i < mx.Length; i++) f[1 + mn.Length + i] = mx[i] / 5f;
        return f;
    }

    public async Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options)
    {
        Pre();
        SnapshotFloor();   // the Ancient floor of an act has no OnRoomResolved
        if (options.Count == 1) return options[0];
        var sorted = options.OrderBy(o => o.coord.col).ThenBy(o => o.coord.row).Take(ML.K).ToList();
        MacroRow row = StateRow();
        foreach (MapPoint pt in sorted) row.AddOption(ML.KMap, (int)pt.PointType, 0, MapFeatures(pt));
        return sorted[await DecideOptionsAsync(row, ML.DMap)];
    }

    // ---- rewards

    public async Task<RewardDecision> ChooseRewardActionAsync(RewardsSet rewards)
    {
        Pre();
        var cls = RewardDecisionClassifier.Classify(rewards);
        if (cls is RewardDecisionClassification.Automatic auto) return auto.Decision;
        var cands = ((RewardDecisionClassification.Choice)cls).Candidates;
        // card reward (main or extra): cards + skip; alternatives (relic-given substitutes) are not offered to the policy
        var cardOpts = new List<(RewardDecision D, CardModel? Card)>();
        foreach (RewardDecision d in cands)
            switch (d)
            {
                case RewardDecision.TakeCard t: cardOpts.Add((d, t.Card)); break;
                case RewardDecision.SkipCard: cardOpts.Add((d, null)); break;
                case RewardDecision.ResolveExtra { Reward: CardReward } x when !x.Skip: cardOpts.Add((d, x.SelectedCard)); break;
            }
        if (cardOpts.Count > 0)
        {
            if (cardOpts.Count > ML.K) cardOpts = cardOpts.Take(ML.K - 1).Append(cardOpts.First(o => o.Card is null)).ToList();
            MacroRow row = StateRow();
            foreach (var (_, card) in cardOpts)
                if (card is null) row.AddOption(ML.KSpecial, 1);   // SKIP
                else row.AddOption(ML.KCard, _s.MVocab.Tok("card", MacroEncoder.CardId(card)), card.CurrentUpgradeLevel);
            int pick = await DecideOptionsAsync(row, ML.DCard);
            if (cardOpts[pick].Card is null) Rec.CardsSkipped++; else Rec.CardsTaken++;
            return cardOpts[pick].D;
        }
        // optional merchant potion / card-removal reward: take when possible (a removal then asks for the card)
        _s.Note("reward_choice:" + string.Join("|", cands.Select(c => c is RewardDecision.ResolveExtra x ? x.Reward.GetType().Name + (x.Skip ? "-skip" : "") : c.GetType().Name)));
        return cands[0];
    }

    // ---- shop

    public async Task<ShopDecision> ChooseShopActionAsync(MerchantInventory inventory, Player player)
    {
        Pre();
        if (_shopFloor != _run.TotalFloor) { _shopFloor = _run.TotalFloor; _shopSteps = 0; Rec.Shops++; }
        if (++_shopSteps > 12) return new ShopDecision.Leave();
        MacroRow row = StateRow();
        var opts = new List<ShopDecision>();
        int gold = player.Gold;
        foreach (MerchantCardEntry e in inventory.Cards)
            if (!e.Purchased && e.Price <= gold && row.NOptions < ML.K - 2)
            { row.AddOption(ML.KCard, _s.MVocab.Tok("card", MacroEncoder.CardId(e.Card)), e.Card.CurrentUpgradeLevel); opts.Add(new ShopDecision.BuyCard(e)); }
        foreach (MerchantRelicEntry e in inventory.Relics)
            if (!e.Purchased && e.Price <= gold && row.NOptions < ML.K - 2)
            { row.AddOption(ML.KRelic, _s.MVocab.Tok("relic", e.Relic.Id.ToString())); opts.Add(new ShopDecision.BuyRelic(e)); }
        bool freeSlot = player.PotionSlots.Contains(null);
        foreach (MerchantPotionEntry e in inventory.Potions)
            if (!e.Purchased && e.Price <= gold && freeSlot && row.NOptions < ML.K - 2)
            { row.AddOption(ML.KPotion, _s.MVocab.Tok("potion", e.Potion.Id.ToString())); opts.Add(new ShopDecision.BuyPotion(e)); }
        var removable = player.Deck.Cards.Where(c => c.IsRemovable).ToList();
        int removeAt = -1;
        if (!inventory.CardRemoval.Purchased && inventory.CardRemoval.Price <= gold && removable.Count > 0)
        { removeAt = row.AddOption(ML.KSpecial, 2); opts.Add(new ShopDecision.Leave()); }   // REMOVE (card chosen below)
        if (opts.Count == 0) return new ShopDecision.Leave();
        row.AddOption(ML.KZero, 0); opts.Add(new ShopDecision.Leave());
        int pick = await DecideOptionsAsync(row, ML.DShop);
        if (pick == removeAt)
            return new ShopDecision.BuyCardRemoval(await DecideDeckCardAsync(removable, ML.DRemove));
        return opts[pick];
    }

    // ---- rest site

    public async Task<RestSiteDecision> ChooseRestSiteActionAsync(Player player, IReadOnlyList<RestSiteDecision> candidates)
    {
        Pre();
        var enabled = candidates.Where(c => c.IsEnabled).ToList();
        if (enabled.Count == 0) return RestSiteDecisionPolicy.ChooseDefault(candidates);
        var groups = enabled.GroupBy(c => c.OptionId).Select(g => (Id: g.Key, Items: g.ToList())).ToList();
        int pick = 0;
        if (groups.Count > 1)
        {
            MacroRow row = StateRow();
            foreach (var (id, _) in groups.Take(ML.K))
            {
                int sid = _s.MVocab.SpecialId("REST:" + id.ToUpperInvariant());
                if (sid >= 0) row.AddOption(ML.KSpecial, sid);
                else row.AddOption(ML.KEvent, ML.EventBucket("REST:" + id));
            }
            pick = await DecideOptionsAsync(row, ML.DRest);
        }
        var chosen = groups[pick];
        if (chosen.Id == "heal") Rec.Rests++;
        if (chosen.Id == "smith")
        {
            Rec.Smiths++;
            var cards = chosen.Items.OfType<RestSiteDecision.Smith>().Select(s => s.Card).ToList();
            CardModel card = await DecideDeckCardAsync(cards, ML.DSmith);
            return chosen.Items.OfType<RestSiteDecision.Smith>().First(s => ReferenceEquals(s.Card, card));
        }
        return chosen.Items[0];
    }

    // ---- events

    public async Task<EventOption> ChooseEventOptionAsync(IReadOnlyList<EventOption> options)
    {
        Pre();
        var open = options.Where(o => !o.IsLocked).Take(ML.K).ToList();
        if (open.Count == 0) return EventOptionDecisionPolicy.ChooseDefault(options);
        if (open.Count == 1) return open[0];
        string eventId = (_run.BaseRoom as EventRoom)?.Event.Id.ToString() ?? "";
        MacroRow row = StateRow();
        // an Ancient offers relics: the option's offered content is the relic (S1's ANCIENT decision)
        bool ancient = _run.CurrentMapPoint?.PointType == MapPointType.Ancient && _run.CurrentRoomCount == 1 &&
            open.All(o => o.OfferedContent is { ContentId.Length: > 0 } c && _s.MVocab.Has("relic", "RELIC." + c.ContentId));
        if (!ancient && _run.CurrentMapPoint?.PointType == MapPointType.Ancient) _s.Note("ancient_options_not_relics:" + eventId);
        if (ancient)
        {
            foreach (EventOption o in open) row.AddOption(ML.KRelic, _s.MVocab.Tok("relic", "RELIC." + o.OfferedContent!.ContentId));
            return open[await DecideOptionsAsync(row, ML.DAncient)];
        }
        Rec.Events++;
        foreach (EventOption o in open) row.AddOption(ML.KEvent, ML.EventBucket(eventId + ":" + o.Key));
        int ev = await DecideOptionsAsync(row, ML.DEvent);
        if (_s.Trace) Console.WriteLine($"TRACE {Rec.Seed} event {eventId} options {string.Join("|", open.Select(o => o.Key))} -> {open[ev].Key} (hp {_p.Creature.CurrentHp})");
        return open[ev];
    }

    public Task<CustomEventDecision> ChooseCustomEventActionAsync(EventModel @event)
    {
        Pre();
        return Task.FromResult(CustomEventDecisionPolicy.ChooseDefault(@event));
    }

    // ---- card prompts

    public async Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request)
    {
        if (request.IsLiveCombatSelection)
        {
            if (_selScript.TryDequeue(out string[]? keys))
            {
                if (CardSelectionOptions.TryMatch(request, keys, out SelectionOption matched)) return matched.Apply(request);
                // an end-turn answer chosen at the root from a resampled probe world that does not fit the real prompt
                // (e.g. a hand-dependent start-of-turn prompt): answered at random below, as without the option
                if (_selEndTurn) _s.Note("endturn_prompt_key_miss");
            }
            var options = CardSelectionOptions.Enumerate(request);
            return options[_rng.Next(options.Count)].Apply(request);
        }
        Pre();
        string? purpose = CardSelectCmd.CurrentDeckPurpose;
        string kind = purpose ?? request.Source?.GetType().Name ?? "none";
        _s.SelectionKinds[kind] = _s.SelectionKinds.GetValueOrDefault(kind) + 1;
        var cands = request.Candidates;
        if (cands.Count == 0) return Array.Empty<CardModel>();
        bool inDeck = cands.All(c => DeckIndex(c) >= 0);
        if (inDeck && purpose is not null && request.MaxCount >= 1 && !request.IsBundleSelection)
        {
            int dtype = purpose == "upgrade" ? ML.DSmith : ML.DRemove;
            var left = cands.ToList();
            var picked = new List<CardModel>();
            while (picked.Count < request.MaxCount && left.Count > 0)
            {
                CardModel c = await DecideDeckCardAsync(left, dtype);
                picked.Add(c); left.Remove(c);
            }
            return picked;
        }
        if (!inDeck && request.MaxCount == 1 && !request.IsBundleSelection && cands.Count < ML.K)
        {
            // a card offered from outside the deck: the card-reward head (skip allowed when the prompt allows none)
            MacroRow row = StateRow();
            foreach (CardModel c in cands) row.AddOption(ML.KCard, _s.MVocab.Tok("card", MacroEncoder.CardId(c)), c.CurrentUpgradeLevel);
            bool canSkip = request.MinCount == 0 || request.Cancelable;
            if (canSkip) row.AddOption(ML.KSpecial, 1);
            int pick = await DecideOptionsAsync(row, ML.DCard);
            return pick < cands.Count ? new[] { cands[pick] } : Array.Empty<CardModel>();
        }
        // anything else: a random legal answer of the largest size
        var pool = cands.ToList();
        var res = new List<CardModel>();
        int want = request.IsBundleSelection ? 1 : request.MaxCount;
        while (res.Count < want && pool.Count > 0) { int i = _rng.Next(pool.Count); res.Add(pool[i]); pool.RemoveAt(i); }
        return res;
    }

    // ------------------------------------------------------------------ combat

    private static string RoomName(RoomType t) => t switch { RoomType.Elite => "elite", RoomType.Boss => "boss", _ => "monster" };

    private async Task<LiveFight> StartFightAsync(CombatState state)
    {
        var room = _run.CurrentRoom as CombatRoom;
        RoomType rt = room?.RoomType is RoomType.Elite or RoomType.Boss ? room.RoomType : RoomType.Monster;
        Creature pc = _p.Creature;
        if (Rec.Perturbed && _rng.NextDouble() < _cfg.PerturbFightP) await PerturbAsync();
        int maxHp = Math.Max(1, pc.MaxHp);
        string id = $"{Rec.Seed}:{_fightIndex}";
        ulong seed = ResamplingClone.Mix(_seed, (ulong)_fightIndex);
        _fightIndex++;
        var belt = _p.PotionSlots.Select(x => x?.Id.ToString() ?? "").ToList();
        var held = Enumerable.Range(0, belt.Count).Where(i => belt[i] != "" && i < Outcome.Slots).ToList();
        var subsets = new List<int[]>();
        for (int size = 0; size <= held.Count; size++)
            for (int m = 0; m < 1 << held.Count; m++)
                if (System.Numerics.BitOperations.PopCount((uint)m) == size)
                    subsets.Add(held.Where((_, j) => (m >> j & 1) == 1).ToArray());
        bool final = rt == RoomType.Boss && room is not null && _run.CurrentRoomCount == 1 && IsFinalBossPoint();
        var tables = new double[subsets.Count][];
        bool useMs = false; double msAlpha = 0;
        if (final || _cfg.Scorer == "hp")
        {
            for (int k = 0; k < subsets.Count; k++)
            {
                tables[k] = new double[maxHp + 1];
                for (int h = 1; h <= maxHp; h++) tables[k][h] = final ? 1.0 : h / (double)maxHp;
            }
        }
        else
        {
            MacroRow post = MacroEncoder.PostFight(_run, _p, _facts, _s.MVocab, rt);
            post.SetHp(maxHp, maxHp);
            var tasks = new Task<MacroEval>[subsets.Count];
            for (int k = 0; k < subsets.Count; k++)
            {
                MacroRow r = post.Clone();
                r.SetPotions(subsets[k].Select(i => _s.MVocab.Tok("potion", belt[i])).Take(ML.MaxPot).ToList());
                tasks[k] = _s.Macro.EvaluateAsync(r);
            }
            MacroEval[] evs = await Task.WhenAll(tasks);
            _s.TfRows += subsets.Count; _s.TfBuilds++;
            // milestone heads (served when the macro file has trained ones): tables from their blend, at the stage of
            // the post-fight state (a boss floor counts as the stage after that boss, as in the trainer)
            useMs = ML.HasMilestones(evs[0].Out);
            int stagePost = ML.Stage(_run.CurrentActIndex, rt == RoomType.Boss);
            bool dbl = _asc >= ML.DoubleBossAsc;
            if (useMs) { msAlpha = evs[0].Out[ML.OutAlpha]; _s.TfMsFights++; }
            for (int k = 0; k < subsets.Count; k++)
            {
                tables[k] = new double[maxHp + 1];
                double shift = _cfg.Kappa * evs[k].ValueStd;
                if (useMs)
                {
                    double v0 = Math.Clamp(ML.MilestoneValue(evs[k].Out, maxHp, maxHp, stagePost, dbl), 1e-6, 1 - 1e-6);
                    _s.TfFormulaMaxError = Math.Max(_s.TfFormulaMaxError, Math.Abs(Math.Log(v0 / (1 - v0)) - evs[k].Out[ML.OutV]));
                    for (int h = 1; h <= maxHp; h++)
                    {
                        double v = Math.Clamp(ML.MilestoneValue(evs[k].Out, h, maxHp, stagePost, dbl), 1e-9, 1 - 1e-9);
                        tables[k][h] = shift == 0 ? v : ML.Sigmoid(Math.Log(v / (1 - v)) - shift);
                    }
                }
                else
                {
                    _s.TfFormulaMaxError = Math.Max(_s.TfFormulaMaxError, Math.Abs(ML.LogitAt(evs[k].Out, maxHp, maxHp) - evs[k].Out[ML.OutV]));
                    for (int h = 1; h <= maxHp; h++) tables[k][h] = ML.Sigmoid(ML.LogitAt(evs[k].Out, h, maxHp) - shift);
                }
            }
            Rec.MacroVersion = evs[0].ModelVersion;
        }
        double death = DeathValueHere();
        double[][]? raw = _s.TfDump is not null ? tables.Select(t => (double[])t.Clone()).ToArray() : null;
        bool firstOfTwo = rt == RoomType.Boss && room is not null && _run.CurrentRoomCount == 1 && IsFirstOfDoubleBoss();
        string? tfKind = null;
        if (firstOfTwo && _cfg.Boss2 is { } b2 && _cfg.Scorer == "tf")
        {
            // first of two act-3 bosses: what winning it is worth is the chance of then winning the second one
            for (int k = 0; k < subsets.Count; k++)
                for (int h = 1; h <= maxHp; h++) tables[k][h] = death + (1 - death) * b2.W2(h / (double)maxHp, subsets[k].Length);
            tfKind = "boss2";
            _s.TfBoss2Fights++;
        }
        if (_cfg.TfStake && _cfg.Scorer == "tf" && !final && tfKind is null)
        {
            // the outcome of dying in this fight; with milestone heads alpha is the trainer's own, served with the rows
            double b = (1 - (useMs ? msAlpha : _cfg.DeathAlpha)) * Math.Min(_run.CurrentActIndex, 2) / 3.0;
            if (b > 0)
            {
                for (int k = 0; k < subsets.Count; k++)
                    for (int h = 1; h <= maxHp; h++)
                    {
                        double t = tables[k][h], s = (t - b) / (1 - b), f = _cfg.StakeFloor * t;
                        if (s < f) { s = f; _s.TfStakeFloored++; }
                        tables[k][h] = s;
                    }
                _s.TfStakeFights++; _s.TfStakeBankedSum += b;
            }
        }
        double deathFormula = death;
        if (death != 0 && _cfg.DeathValue == "progress" && tables[0][1] < death)
        {
            // subsets are listed by size: tables[0] is the empty belt, its HP-1 entry the worst survival
            death = tables[0][1];
            _s.TfDeathCapped++;
        }
        if (death != 0)
            for (int k = 0; k < subsets.Count; k++)
            {
                tables[k][0] = death;
                if (_cfg.DeathValue == "progress_raw")
                    for (int h = 1; h <= maxHp; h++)
                        if (tables[k][h] < death) { tables[k][h] = death; _s.TfDeathFloorRaised++; }
            }
        if (_s.TfDump is not null) DumpTables(id, rt, room, final, firstOfTwo, tfKind, maxHp, pc.CurrentHp, belt, subsets, death, deathFormula, raw!, tables);
        _s.Tf.Set(id, maxHp, belt, subsets, tables, final, death);
        var ctx = new BattleContext(id, maxHp, belt);
        string enc = room?.Encounter is { } e ? "ENCOUNTER." + e.IdEntry : room?.EncounterName ?? "";
        var spec = new BattleStartSpec
        {
            Id = id, RoomType = RoomName(rt), ActIndex = _run.CurrentActIndex, Floor = _run.TotalFloor, EncounterId = enc,
            Hp = pc.CurrentHp, MaxHp = maxHp, Potions = belt, Ascension = _asc,
        };
        double tMax = Math.Max(1e-3, _s.Tf.MaxValue(id));
        FightSignals? signals = _cfg.Context ? new FightSignals(_s.Tf, ctx, spec, tMax) : null;
        var o = new FightOutcome
        {
            Id = id, RoomType = spec.RoomType, Encounter = enc, StartHp = pc.CurrentHp, PotionsStart = held.Count, TMax = tMax,
            TableMaxHp = maxHp, DeathValue = death, TfKind = tfKind,
        };
        return new LiveFight
        {
            Engine = state.Engine!, Id = id, Ctx = ctx, Signals = signals, O = o, Room = rt, Seed = seed,
            Search = new NetSearch(_cfg.Search, _s.Net, _s.Tf, ctx, tMax, signals),
        };
    }

    /// <summary>F3 debug dump: the first TfDumpK live fights per kind (act-3 boss; hallway fight of act 1, 2, 3) of this
    /// actor, with the macro network's table before any change ("raw", before free disposal; all 1 at the final boss) and
    /// the table the fight is scored by ("tables", before free disposal, which TfScorer applies).</summary>
    private void DumpTables(string id, RoomType rt, CombatRoom? room, bool final, bool firstOfTwo, string? tfKind, int maxHp, int hp,
        List<string> belt, List<int[]> subsets, double death, double deathFormula, double[][] raw, double[][] tables)
    {
        string? key = rt == RoomType.Boss && _run.CurrentActIndex == 2 ? "boss3" : rt == RoomType.Monster ? $"monster{_run.CurrentActIndex + 1}" : null;
        if (key is null || _s.TfDumped.GetValueOrDefault(key) >= _s.TfDumpK) return;
        _s.TfDumped[key] = _s.TfDumped.GetValueOrDefault(key) + 1;
        static double[] R(double[] t) => t.Select(x => Math.Round(x, 6)).ToArray();
        var d = new Dictionary<string, object?>
        {
            ["kind"] = key, ["id"] = id, ["seed"] = Rec.Seed, ["ascension"] = _asc, ["act"] = _run.CurrentActIndex, ["floor"] = _run.TotalFloor,
            ["encounter"] = room?.Encounter is { } e ? "ENCOUNTER." + e.IdEntry : room?.EncounterName ?? "", ["final_boss"] = final,
            ["first_of_two_bosses"] = firstOfTwo, ["tf_kind"] = tfKind, ["max_hp"] = maxHp, ["hp"] = hp, ["belt"] = belt,
            ["subsets"] = subsets, ["death_value_option"] = _cfg.DeathValue, ["alpha"] = _cfg.DeathAlpha, ["death"] = death,
            ["death_formula"] = deathFormula, ["death_scale"] = _cfg.DeathScale, ["death_progress"] = (1 - _cfg.DeathAlpha) * Math.Min(_run.CurrentActIndex, 2) / 3.0,
            ["raw"] = raw.Select(R).ToArray(), ["tables"] = tables.Select(R).ToArray(),
        };
        _s.TfDump!.WriteLine(System.Text.Json.JsonSerializer.Serialize(d));
    }

    /// <summary>Perturbed fight start (PROPOSAL.md §6.5): HP shifted by up to 30% of max HP and a potion added or
    /// removed, independent of deck and play, so V_run learns what HP and potions are worth to this agent.</summary>
    private async Task PerturbAsync()
    {
        Creature pc = _p.Creature;
        int shift = (int)Math.Round((_rng.NextDouble() * 0.6 - 0.3) * pc.MaxHp);
        pc.SetCurrentHpInternal(Math.Clamp(pc.CurrentHp + shift, 1, pc.MaxHp));
        var heldSlots = _p.PotionSlots.Where(x => x is not null).ToList();
        bool free = _p.PotionSlots.Contains(null);
        if (_rng.NextDouble() < 0.5 && free)
        {
            PotionModel? pot = PotionFactory.CreateRandomOutOfCombat(_p, new Rng(ResamplingClone.Mix(_seed, 0x9E77UL + (ulong)_fightIndex)));
            if (pot is not null) await PotionCmd.TryToProcure(pot, _p);
        }
        else if (heldSlots.Count > 0)
            _p.RemovePotionInternal(heldSlots[_rng.Next(heldSlots.Count)]!);
        Rec.Perturbations++;
    }

    public async Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
    {
        Pre();
        if (_fight is null || !ReferenceEquals(_fight.Engine, state.Engine))
        {
            if (_fight is not null) EndFight(_fight.Engine.State);
            _fight = await StartFightAsync(state);
            if (_cfg.PracticeBoss > 0 || _cfg.PracticeElite > 0) await PracticeAsync(_fight, state);
        }
        LiveFight f = _fight;
        FightOutcome o = f.O;
        _selScript.Clear();
        if (state.RoundNumber > 3 * _cfg.MaxRounds)
        {
            // A fight that neither side can finish never ends (the engine has no round limit). Ending the turn here
            // completes synchronously, so the actor's single thread would never return to its other runs (2026-11-12:
            // three actors spun for up to two hours). The run is failed instead: recorded as an error, not trained on.
            _s.Note("combat_stalemate:" + o.Encounter);
            throw new InvalidOperationException($"combat stalemate: {o.Encounter} still in progress at round {state.RoundNumber}");
        }
        bool capped = state.RoundNumber > _cfg.MaxRounds;
        // one draw per move whatever the settings, so the actor's random stream does not depend on them
        bool full = DrawFull(_rng, f.Room);
        int sims = MoveSims(f.Room, capped, full);
        NetSearchResult r = await f.Search.SearchAsync(state, sims, ResamplingClone.Mix(f.Seed, (ulong)o.Decisions));
        Account(f, o, state, r, full, capped);
        o.Decisions++;
        return ToDecision(state, r.Action);
    }

    private bool DrawFull(Random rng, RoomType room) =>
        rng.NextDouble() < (room == RoomType.Boss && _cfg.BossPFull >= 0 ? _cfg.BossPFull : _cfg.PFull);

    private int MoveSims(RoomType room, bool capped, bool full)
    {
        int sims = capped ? 0 : full ? _cfg.SimsFull : _cfg.SimsFast;
        if (room == RoomType.Boss) sims = (int)Math.Round(sims * _cfg.BossSimsFactor);
        else if (room == RoomType.Monster) sims = (int)Math.Round(sims * _cfg.HallSimsFactor);
        return sims;
    }

    /// <summary>Statistics and the training record of one searched move (live fight or practice copy).</summary>
    private void Account(LiveFight f, FightOutcome o, CombatState state, NetSearchResult r, bool full, bool capped)
    {
        o.Simulations += r.Simulations; o.Leaves += r.Leaves; o.Steps += r.StepsApplied;
        if (r.RootEval is not null) o.ModelVersion = r.RootEval.ModelVersion;
        if (r.Simulations > 0) { o.Searched++; o.RootQSum += r.RootQ; }
        if (_cfg.Search.Reuse && r.Simulations > 0)
        {
            o.ReusedVisits += r.ReusedVisits;
            _s.ReuseSearches++; _s.ReusedVisits += r.ReusedVisits; _s.ReuseTableMax = Math.Max(_s.ReuseTableMax, r.TableNodes);
        }
        if (_cfg.RecordCombat && _s.CombatWriter is not null && r.RootActions.Count > 1 && !capped)
        {
            long tr = Stopwatch.GetTimestamp();
            int ow = ObsLayout.MaxTokens * ObsLayout.TokenWidth, aw = ObsLayout.MaxActions * ObsLayout.ActionWidth;
            var obs = new int[ow]; var act = new int[aw];
            ObsLayout.Write(Encoding.Tokens(state, f.Signals), r.RootActions, state, _s.Vocab, obs, act);
            o.Records.Add(new Decision
            {
                Obs = Array.ConvertAll(obs, Sat), Act = Array.ConvertAll(act, Sat), Policy = full ? r.PolicyTarget : null,
                NActions = r.RootActions.Count, Sims = r.Simulations, ModelVersion = r.RootEval?.ModelVersion ?? 0,
                HeldMask = f.Signals?.HeldMask(state) ?? 0,
            });
            Timing.Record += Stopwatch.GetElapsedTime(tr).TotalSeconds;
        }
        if (r.Action.Kind == ActionKind.UsePotion) o.PotionsUsed++;
    }

    // ------------------------------------------------------------------ practice fights

    /// <summary>
    /// Practice fights (KataGo-style: more training from hard positions). At the first decision of a live boss (elite)
    /// fight, <see cref="AgentConfig.PracticeBoss"/> (<see cref="AgentConfig.PracticeElite"/>) copies of the fight are
    /// played to the end, one after another, before the live fight goes on. Each copy starts from a resampling clone of
    /// the live combat (hidden information drawn afresh, as the search does, with its own chance seed), is played by the
    /// same search settings with the live fight's T_f table, signals and T_max, and is recorded to the combat data like
    /// a live fight. The live run is untouched: clones only, a random stream of its own (not the agent's), its own
    /// search tables. A copy still running at round 3 x max_rounds, or failing, is abandoned (not recorded); the run goes on.
    /// </summary>
    private async Task PracticeAsync(LiveFight f, CombatState live)
    {
        int k = f.Room == RoomType.Boss ? _cfg.PracticeBoss : f.Room == RoomType.Elite ? _cfg.PracticeElite : 0;
        if (k <= 0 || _run.CurrentActIndex + 1 < _cfg.PracticeMinAct) return;
        for (int j = 0; j < k; j++) await PlayPracticeAsync(f, live, j);
        f.Watch.Restart();   // the live fight's own time
    }

    private async Task PlayPracticeAsync(LiveFight f, CombatState live, int copy)
    {
        var sw = Stopwatch.StartNew();
        ulong seed = ResamplingClone.Mix(f.Seed, 0x9AC71CE0UL + (ulong)copy);
        var rng = new Random((int)(seed ^ (seed >> 32)));
        FightOutcome lo = f.O;
        var o = new FightOutcome
        {
            Id = $"{f.Id}:p{copy}", RoomType = lo.RoomType, Encounter = lo.Encounter, StartHp = live.Players[0].Creature.CurrentHp,
            PotionsStart = lo.PotionsStart, TMax = lo.TMax, TableMaxHp = lo.TableMaxHp, Practice = true,
        };
        var search = new NetSearch(_cfg.Search, _s.Net, _s.Tf, f.Ctx, lo.TMax, f.Signals);
        bool abandoned = false;
        try
        {
            CombatState state = ResamplingClone.Resampled(live, ResamplingClone.Mix(seed, 0xF1647UL));
            SearchCardSelectionSource.InstallOn(state, SelectionMode.Policy, SearchCardSelectionSource.RandomPolicy(new Random(rng.Next())));
            while (!CombatActions.IsOver(state))
            {
                if (state.RoundNumber > 3 * _cfg.MaxRounds) { abandoned = true; o.Error = "round cap"; _s.Note("practice_stalemate:" + o.Encounter); break; }
                _s.Progress++;
                bool capped = state.RoundNumber > _cfg.MaxRounds;
                bool full = _cfg.PracticePFull >= 0 ? rng.NextDouble() < _cfg.PracticePFull : DrawFull(rng, f.Room);
                NetSearchResult r = await search.SearchAsync(state, MoveSims(f.Room, capped, full), ResamplingClone.Mix(seed, (ulong)o.Decisions));
                Account(f, o, state, r, full, capped);
                if (!await ActionSpace.ApplyAsync(state, r.Action))
                { abandoned = true; o.Error = "chosen action not applicable: " + r.Action.Key; _s.Note("practice_action_not_applicable"); break; }
                o.Decisions++;
            }
            Creature pc = state.Players[0].Creature;
            o.EndHp = Math.Max(0, pc.CurrentHp); o.MaxHp = pc.MaxHp; o.Turns = state.RoundNumber;
            if (!abandoned)
            {
                bool won = state.Engine is { IsInProgress: false, Won: true } && o.EndHp > 0;
                o.Result = won ? "won" : "lost";
                o.Score = won ? _s.Tf.Score(state, f.Ctx) : 0.0;
                o.Rel = o.Score / o.TMax;
                if (f.Signals is not null) o.EndHeldMask = f.Signals.HeldMask(state);
            }
        }
        catch (Exception e)
        {
            abandoned = true;
            o.Error = e.GetType().Name + ": " + (e.Message.Length > 160 ? e.Message[..160] : e.Message);
            _s.Note("practice_error:" + e.GetType().Name);
        }
        search.Clear();
        o.Seconds = sw.Elapsed.TotalSeconds;
        Rec.PracticeSimulations += o.Simulations; Rec.PracticeDecisions += o.Decisions; Rec.PracticeSeconds += o.Seconds;
        _s.PracticeSimulations += o.Simulations; _s.PracticeDecisions += o.Decisions; _s.PracticeSeconds += o.Seconds;
        if (abandoned)
        {
            o.Result = "abandoned"; o.Records.Clear();
            Rec.PracticeAbandoned++; _s.PracticeAbandoned++;
        }
        else
        {
            _s.CombatWriter?.Add(o);
            Rec.PracticeFights++; _s.PracticeFights++;
            if (o.Survived) { Rec.PracticeWon++; _s.PracticeWon++; }
        }
        _s.FightLog?.Invoke(o, Rec, _run.TotalFloor);
    }

    private static short Sat(int x) => (short)Math.Clamp(x, short.MinValue, short.MaxValue);

    /// <summary>The live-engine decision for a search action (resolved by identity, as ActionSpace.ApplyAsync does);
    /// the action's prompt answers are queued for <see cref="ChooseCardsAsync"/>. This includes an end-turn action's
    /// answer (endturn_prompts): the engine asks for it during the enemy turn or the next turn start, before the next
    /// <see cref="ChooseCombatActionAsync"/> (which clears what was not used).</summary>
    private CombatDecision ToDecision(CombatState state, SearchAction a)
    {
        foreach (string[] k in a.Selections) _selScript.Enqueue(k);
        _selEndTurn = a.Kind == ActionKind.EndTurn && a.Selections.Count > 0;
        if (_selEndTurn) _s.Note("endturn_prompt_answer");
        Player p = state.Players[0];
        switch (a.Kind)
        {
            case ActionKind.PlayCard:
            {
                var hand = p.PlayerCombatState!.Hand.Cards;
                CardModel? card = a.SourceIndex >= 0 && a.SourceIndex < hand.Count && hand[a.SourceIndex].CanPlay(out _) &&
                    CardSelectionOptions.EquivalenceKey(hand[a.SourceIndex]) == a.Source ? hand[a.SourceIndex]
                    : hand.FirstOrDefault(c => c.CanPlay(out _) && CardSelectionOptions.EquivalenceKey(c) == a.Source);
                if (card is null) break;
                Creature? target = null;
                if (card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer)
                {
                    target = CombatTargetCandidates.ForCard(state, p, card.TargetType).FirstOrDefault(c => c.CombatId == a.TargetId);
                    if (target is null) break;
                }
                return new CombatDecision.PlayCard(card, target);
            }
            case ActionKind.UsePotion:
            {
                PotionModel? potion = a.SourceIndex >= 0 && a.SourceIndex < p.PotionSlots.Count && p.PotionSlots[a.SourceIndex]?.Id.ToString() == a.Source
                    ? p.PotionSlots[a.SourceIndex] : p.PotionSlots.FirstOrDefault(x => x?.Id.ToString() == a.Source);
                if (potion is null) break;
                Creature? target = a.TargetId is { } tid ? ActionSpace.PotionTargetCandidates(state).FirstOrDefault(c => c?.CombatId == tid) : null;
                if (a.TargetId is not null && target is null) break;
                return new CombatDecision.UsePotion(potion, target);
            }
            case ActionKind.EndTurn:
                return new CombatDecision.EndTurn();
        }
        _selScript.Clear();
        _s.Note("combat_action_unresolved");
        return new CombatDecision.EndTurn();
    }

    /// <summary>ICombatCompletionValidator: the live fight is over, before its outcome and rewards are resolved. This
    /// is the state the search scores, so the fight's record (end HP, potions kept) is taken here.</summary>
    public void ValidateCompletedCombat(CombatState state)
    {
        if (_fight is not null && ReferenceEquals(_fight.Engine, state.Engine)) EndFight(state);
    }

    private void EndFight(CombatState state)
    {
        LiveFight f = _fight!;
        _fight = null;
        FightOutcome o = f.O;
        Creature pc = state.Players[0].Creature;
        o.EndHp = Math.Max(0, pc.CurrentHp); o.MaxHp = pc.MaxHp; o.Turns = state.RoundNumber;
        bool won = state.Engine is { IsInProgress: false, Won: true } && o.EndHp > 0;
        o.Result = won ? "won" : "lost";
        o.Score = won ? _s.Tf.Score(state, f.Ctx) : 0.0;
        o.Rel = o.Score / o.TMax;
        if (f.Signals is not null) o.EndHeldMask = f.Signals.HeldMask(state);
        o.Seconds = f.Watch.Elapsed.TotalSeconds;
        f.Search.Clear();
        _s.Tf.Remove(f.Id);
        _s.CombatWriter?.Add(o);
        _s.Simulations += o.Simulations; _s.CombatDecisions += o.Decisions; _s.Fights++;
        Rec.Fights++; Rec.Simulations += o.Simulations; Rec.CombatDecisions += o.Decisions; Rec.PotionsUsed += o.PotionsUsed;
        Rec.ModelVersion = o.ModelVersion;
        if (f.Room == RoomType.Elite) Rec.Elites++;
        if (f.Room == RoomType.Boss) Rec.Bosses++;
        if (won) { Rec.FightsWon++; _s.FightsWon++; if (f.Room == RoomType.Boss) Rec.BossesWon++; }
        else Rec.KilledBy = o.Encounter;
        _s.FightLog?.Invoke(o, Rec, _run.TotalFloor);
    }
}
