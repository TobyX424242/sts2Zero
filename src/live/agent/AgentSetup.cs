using System.Globalization;
using System.Text.Json;
using Sts2Sim.Core.Models;
using Sts2Sim.S0;
using Sts2Sim.S2;
using Sts2Sim.S3;

namespace Sts2Sim.Live;

/// <summary>
/// The agent's settings and services, built from key=value arguments exactly as src/s3/S3/Program.cs builds them
/// for `S3 play mode=eval` (same keys, same meaning; the parsing below is a copy of Program.cs's, restricted to the keys
/// that matter for one live run). On top, the live defaults (<see cref="ChainEval"/>) are the s3ic chain's evaluation:
/// eval_s3.sh / s3_supervise.sh pass `mode=eval sims_full=64 obs=5` plus EVAL_ARGS, so the live agent is the agent the
/// chain measures. `preset=none` drops these defaults (S3 Program.cs defaults then apply).
/// </summary>
public sealed class AgentSetup
{
    /// <summary>s3ic evaluation (src/ic113/s3ic_launch.sh EVAL_BASE_ARGS + the hook's potion_mix file, coordinator
    /// 2026-11-20): greedy macro, 64 simulations on every move, 16 root candidates (eval default), no search noise.</summary>
    public static readonly Dictionary<string, string> ChainEval = new()
    {
        ["mode"] = "eval", ["sims_full"] = "64", ["obs"] = "5", ["character"] = "ironclad", ["era"] = "6", ["tf_heal"] = "1",
        ["greedy_map"] = "1", ["look_beta"] = "20", ["look_smith"] = "20", ["look_shop"] = "20", ["potion_mix_shrink"] = "0.2",
        ["potion_mix_extra"] = "1", ["fake_merchant"] = "1", ["win_bound"] = "0.001", ["deck_route"] = "1", ["lethal_backstop"] = "1",
        ["kindle_cap"] = "1",   // live only (owner, 2026-11-20): not part of the chain's evaluation
    };

    public required Dictionary<string, string> Kv;
    public required AgentConfig Cfg;
    public required ActorServices Services;
    public required ObsLayout.Vocab Vocab;
    public required MacroVocab MVocab;
    public BridgeEvaluator? Bridge;
    public MacroBridgeEvaluator? MBridge;
    public required CharacterModel Character;
    public bool Eval;

    public string Get(string k, string d) => Kv.TryGetValue(k, out var v) ? v : d;
    public int GetI(string k, int d) => int.Parse(Get(k, d.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
    public double GetD(string k, double d) => double.Parse(Get(k, d.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);

    public static Dictionary<string, string> ParseArgs(IEnumerable<string> args)
    {
        var kv = new Dictionary<string, string>();
        foreach (string[] p in args.Select(a => a.Split('=', 2))) kv[p[0]] = p.Length > 1 ? p[1] : "";   // the last one wins
        if (!kv.TryGetValue("preset", out string? preset) || preset != "none")
            foreach (var (k, v) in ChainEval) kv.TryAdd(k, v);
        return kv;
    }

    /// <summary>Builds config and services (Program.cs's block). Repository-relative default paths are resolved
    /// against <paramref name="root"/> (the tree root).</summary>
    public static AgentSetup Build(Dictionary<string, string> kv, string root)
    {
        string Get(string k, string d) => kv.TryGetValue(k, out var v) ? v : d;
        int GetI(string k, int d) => int.Parse(Get(k, d.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
        double GetD(string k, double d) => double.Parse(Get(k, d.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
        string P(string p) => Path.IsPathRooted(p) ? p : Path.Combine(root, p);

        bool eval = Get("mode", "eval") == "eval";
        var vocab = ObsLayout.Vocab.Load(P(Get("vocab", "src/s0/data/combat_vocab_v113.json")));
        ObservableState.Extended = GetI("fp_ext", 1) == 1;
        int obsVersion = GetI("obs", 4);
        CombatObservation.RelicCounters = obsVersion >= 5;
        CombatObservation.ModifiedIntentDamage = obsVersion >= 6 || GetI("intent_damage", 0) == 1;
        var mvocab = new MacroVocab(P(Get("mvocab", "src/s3/data/macro_vocab_v113.json")));
        ML.Era = GetI("era", ML.EraV0113);
        string ch = Get("character", "ironclad");
        string chId = ch.StartsWith("CHARACTER.", StringComparison.Ordinal) ? ch : "CHARACTER." + ch.ToUpperInvariant();
        var character = ModelDb.GetById<CharacterModel>(ModelId.Deserialize(chId));
        ML.CharIndex(character.Id.ToString());   // refuses a character without an S1 token
        var cfg = new AgentConfig
        {
            Ascension = GetI("ascension", 10), SimsFull = GetI("sims_full", 64), SimsFast = GetI("sims_fast", eval ? GetI("sims_full", 64) : 16),
            PFull = GetD("pfull", eval ? 1.0 : 0.25), BossSimsFactor = GetD("boss_sims", 1.0), RecordCombat = false,
            Context = GetI("context", 1) == 1, Greedy = GetI("greedy", eval ? 1 : 0) == 1, Scorer = Get("scorer", "tf"), Kappa = GetD("kappa", 0),
            MaxRounds = GetI("max_rounds", 50), HallSimsFactor = GetD("hall_sims", 1.0), BossPFull = GetD("boss_pfull", -1.0),
            Search = new NetSearchConfig
            {
                GumbelNoise = GetI("noise", eval ? 0 : 1) == 1, ConfidentSkip = GetD("cskip", 2.0), RootCandidates = GetI("candidates", 16),
                CScale = GetD("cscale", 0.1), CVisit = GetD("cvisit", 50), QSpanMin = GetD("qspan", 0), WinBound = GetD("win_bound", -1),
                Reuse = GetI("reuse", 0) == 1, ReuseCap = GetI("reuse_cap", 5000),
                EndTurnPrompts = GetI("endturn_prompts", 1) == 1,
                Crn = GetI("crn", 0) == 1,
            },
            PracticeBoss = 0, PracticeElite = 0,   // practice fights are training data
            DeathValue = Get("death_value", "zero"), DeathAlpha = GetD("death_alpha", 0.05), DeathScale = GetD("death_scale", 1.0),
            Boss2 = kv.ContainsKey("boss2_model") ? Boss2Model.Load(P(Get("boss2_model", ""))) : null,
            TfStake = GetI("tf_stake", 0) == 1, StakeFloor = GetD("stake_floor", 0.02),
            ExactBoss = 0,   // needs the true future: impossible against the game
            GreedyMap = GetI("greedy_map", 0) == 1,
            GreedyEvent = GetI("greedy_event", 0), MapRule = Get("map_rule", ""),
            RemoveStrike = GetI("remove_strike", 0) == 1, BadCardCut = GetD("bad_card_cut", double.NaN), BadCardAct = GetI("bad_card_act", 3),
            SkipBias = GetD("skip_bias", 0), SkipDeck = GetI("skip_deck", 34), SkipAct = GetI("skip_act", 2),
            LookBeta = GetD("look_beta", 0), LookKappa = GetD("look_kappa", 0), LookAct = GetI("look_act", 1),
            LookRemove = GetD("look_remove", 0), LookSmith = GetD("look_smith", 0), LookRest = GetD("look_rest", 0),
            LookShop = GetD("look_shop", 0), LookRelic = GetD("look_relic", 0), LookRelicAny = GetI("look_relic_any", 0) == 1,
            PotionScale = GetD("potion_scale", 1.0), HpScale = GetD("hp_scale", 1.0),
            PotionSwap = GetI("potion_swap", 1) == 1, PotionKindScale = GetI("potion_kind_scale", 0) == 1,
            PotionReserve = GetD("potion_reserve", 1.0), PotionReserveMin = GetD("potion_reserve_min", 12.0),
            PotionReserveHpLo = GetD("potion_reserve_hp_lo", 0.2), PotionReserveHpHi = GetD("potion_reserve_hp_hi", 0.3),
            CloneProbe = 0,   // needs RunState.CloneReseeded of a live room: not with an imported run
            TfHeal = GetI("tf_heal", 1) == 1,
            PotionMixShrink = GetD("potion_mix_shrink", 0.2), PotionMixPhase = GetI("potion_mix_phase", 1),
            PotionMixMargin = GetD("potion_mix_margin", 0), PotionMixExtra = GetI("potion_mix_extra", 0) == 1,
            FakeMerchant = GetI("fake_merchant", 0) == 1,
            DeckRoute = GetI("deck_route", 0) == 1, LethalBackstop = GetI("lethal_backstop", 0) == 1,
            KindleCap = GetI("kindle_cap", 0) == 1, KindleActFights = GetI("kindle_act_fights", 10),
        };
        if (kv.ContainsKey("potion_mix") && kv.ContainsKey("potion_values")) throw new ArgumentException("potion_mix replaces the potion_values belt rule: use one of them");
        if (cfg.TfStake && (cfg.DeathValue != "zero" || cfg.Boss2 is not null)) throw new ArgumentException("tf_stake=1 replaces death_value and boss2_model: use it alone");
        if (cfg.DeathValue is not ("zero" or "progress" or "progress_raw")) throw new ArgumentException($"unknown death_value {cfg.DeathValue}");
        if (cfg.PotionReserve != 1.0 && !kv.ContainsKey("potion_values")) throw new ArgumentException("potion_reserve needs potion_values=FILE");
        string bridgePath = Get("bridge", "uniform"), mbridgePath = Get("mbridge", "uniform");
        BridgeEvaluator? bridge = bridgePath == "uniform" ? null : new BridgeEvaluator(bridgePath, vocab);
        MacroBridgeEvaluator? mbridge = mbridgePath == "uniform" ? null : new MacroBridgeEvaluator(mbridgePath);
        var services = new ActorServices
        {
            Net = bridge is null ? new UniformEvaluator() : bridge, Macro = mbridge is null ? new UniformMacroEvaluator() : mbridge,
            Vocab = vocab, MVocab = mvocab, Cfg = cfg, CombatWriter = null,
        };
        if (kv.ContainsKey("potion_values"))
        {
            services.PotionValues = JsonSerializer.Deserialize<Dictionary<string, double>>(File.ReadAllText(P(Get("potion_values", ""))))!;
            var sorted = services.PotionValues.Values.OrderBy(x => x).ToList();
            services.PotionValueDefault = sorted.Count > 0 ? sorted[sorted.Count / 2] : 0;
            services.PotionSwapMargin = GetD("potion_swap_margin", 0);
            services.PotionBuyMin = GetD("potion_buy_min", 0);
        }
        if (kv.ContainsKey("potion_mix") && Get("potion_mix", "") != "") services.PotionMix = PotionMix.Load(P(Get("potion_mix", "")), cfg.PotionMixPhase);
        if (kv.ContainsKey("card_values"))
        {
            services.CardValues = JsonSerializer.Deserialize<Dictionary<string, double>>(File.ReadAllText(P(Get("card_values", ""))))!;
            services.CardBonus = GetD("card_bonus", 0.5); services.CardBonusAct = GetI("card_bonus_act", 3);
        }
        return new AgentSetup
        {
            Kv = kv, Cfg = cfg, Services = services, Vocab = vocab, MVocab = mvocab, Bridge = bridge, MBridge = mbridge,
            Character = character, Eval = eval,
        };
    }

    /// <summary>Program.cs's pump idle step: combat batches are answered within a flush; a macro request stays in
    /// flight while fights go on and is waited for only when nothing else can run.</summary>
    public bool Flush()
    {
        bool did = false;
        if (MBridge is not null)
        {
            if (MBridge.InFlight && MBridge.Ready) { MBridge.Complete(); did = true; }
            if (!MBridge.InFlight) MBridge.Submit();
        }
        if (Bridge is not null && Bridge.Submit()) { Bridge.Complete(); did = true; }
        if (!did && MBridge is not null && MBridge.InFlight) { MBridge.Complete(); did = true; }
        return did;
    }

    public string Describe() =>
        $"character {Character.Id}, era {ML.Era}, mode {(Eval ? "eval" : "selfplay")}, sims {Cfg.SimsFull}/{Cfg.SimsFast} pfull {Cfg.PFull}, " +
        $"candidates {Cfg.Search.RootCandidates}, noise {Cfg.Search.GumbelNoise}, greedy {Cfg.Greedy} (map {Cfg.GreedyMap}), look beta/smith/shop " +
        $"{Cfg.LookBeta}/{Cfg.LookSmith}/{Cfg.LookShop}, potion_mix {(Services.PotionMix is null ? "off" : Get("potion_mix", ""))}, " +
        $"fake_merchant {Cfg.FakeMerchant}, deck_route {Cfg.DeckRoute}, lethal_backstop {Cfg.LethalBackstop}, kindle_cap {Cfg.KindleCap}, win_bound {Cfg.Search.WinBound}, " +
        $"tf_heal {Cfg.TfHeal}, obs v{CombatObservation.Version}, combat {Get("bridge", "uniform")}, macro {Get("mbridge", "uniform")}";
}
