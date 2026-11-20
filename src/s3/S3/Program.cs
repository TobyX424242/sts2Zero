// S3 actor: plays whole runs (character= ironclad|silent|defect|necrobinder|regent, default defect); macro decisions by
// the macro network, fights by the network-guided search.
//   dotnet S3.dll play key=value ...
// keys: bridge (combat shm path or "uniform"), mbridge (macro shm path or "uniform"), vocab, mvocab,
//   out (run log .jsonl), fights (fight log .jsonl, optional), data (combat chunk dir), mdata (macro chunk dir),
//   tag, concurrent (runs in flight), runs (total), seconds (time limit: no new run starts after it),
//   mode (selfplay|eval), ascension, seed (actor seed), seeds (eval: seed prefix, runs use <prefix><index>),
//   first/count (eval: seed indices), sims_full, sims_fast, pfull, boss_sims (budget factor in boss fights), hall_sims (budget factor in hallway
//   fights), boss_pfull (share of boss moves searched with sims_full; default pfull), max_rounds, noise (0/1),
//   cskip (confident-skip threshold), candidates, cscale / cvisit (sigma of the completed Q: 0.1 / 50), qspan (floor of the
//   Q span in sigma, relative T_f; 0 = min-max as before), perturb (share of self-play runs with perturbed fight starts),
//   scorer (tf|hp), kappa (T_f pessimism), greedy (macro argmax; default eval 1), context (0/1), chunk, mchunk,
//   greedy_map (0/1: with greedy, the map head's argmax too), greedy_event (N > 0: with greedy, the event head's argmax
//   except options already taken N times on the floor; 0 = events sampled), starts_dump (JSONL file, "{tag}" = actor
//   tag: every live boss fight's start as a BattleStartSpec with its T_f table; starts_dump_elite=1 adds elite fights),
//   potion_values (JSON map potion id -> value: with a full belt and a potion reward waiting, the lowest-valued held
//   potion is discarded when the offered one is worth more by potion_swap_margin (0); off; potion_swap=0 loads the values
//   without this rule), potion_kind_scale (0/1, needs potion_values: a potion subset's gain in the macro tables x the mean
//   of value / median value of its potions, so the potions worth most at the final boss are spared),
//   potion_buy_min (X > 0, needs potion_values: at a shop an affordable potion worth at least X is bought first; off),
//   map_rule (route probes: uniform | elite_max | elite_min | hp:T, see AgentConfig.MapRule; "" = the map head),
//   remove_strike (0/1), bad_card_cut (card value below which a reward card gets -20; needs card_values; card_bonus=0 to
//   load the values without the bonus) and bad_card_act (3), skip_bias / skip_deck (34) / skip_act (2), look_beta /
//   look_kappa / look_act (1): deployment probes of 2026-11-16, see AgentConfig.RemoveStrike,
//   look_remove / look_smith / look_rest / look_shop / look_relic (betas of the one-step value lookahead of removal
//   targets, upgrade targets, rest-site options, shop purchases and Ancient relic choices; 0 = off; AgentConfig.LookRemove),
//   look_relic_any (0/1: look_relic also when an offered relic has a pickup effect, the relic token alone), look_dump
//   (JSONL file, "{tag}" = actor tag: each look_* decision's state row, post-state rows, pick and next state row, for
//   src/exp/look_rowcheck.py; off),
//   card_values (JSON map card id -> value: in card rewards from act card_bonus_act (3) on, card_bonus (0.5) x value is
//   added to the card's policy logit; off),
//   potion_scale / hp_scale (price probes, 1 = off: the macro tables' potion gains / HP drops multiplied by this),
//   max_floors, shutdown (0/1), asc_mix (self-play: share of runs at `ascension`, the rest uniform over the levels below; 1),
//   asc_weights (self-play: weights of the levels below `ascension`, comma-separated; default uniform),
//   practice_boss / practice_elite (self-play: extra copies of each live boss / elite fight played on resampled clones
//   and recorded; 0), practice_min_act (1-based act from which practice copies are played; 1), practice_pfull (share of
//   practice moves at sims_full; default as in the live fight),
//   reuse (0/1: keep the search tree between moves of a fight; 0), reuse_cap (node cap of a kept tree; 5000),
//   death_value (zero|progress|progress_raw: T_f of a death = 0, or d = (1 - alpha) x acts cleared / 3 as in the macro value
//   target, capped at T_f(1 HP, no potions) (progress) or with table entries below d raised to d (progress_raw); zero),
//   death_alpha (alpha for death_value=progress; 0.05), death_scale (d multiplied by this; 1), death_alpha_log (macro_train_log.jsonl: alpha = its last "alpha"
//   entry when the file has one, else death_alpha), boss2_model (JSON of src/s3/fit_boss2.py: the first of two act-3
//   bosses scored by d + (1 - d) W2(HP share, potions); off), tf_dump (JSONL file, "{tag}" replaced by the actor tag: T_f
//   tables of the first tf_dump_k act-3 boss and hallway fights per act of this actor; off), tf_dump_k (20),
//   tf_stake (0/1: tables rescaled to the outcome not yet secured, (T - b) / (1 - b), b = (1 - alpha) x acts cleared / 3;
//   alpha from death_alpha / death_alpha_log; 0), stake_floor (floor of the rescaled table as a share of T; 0.02),
//   endturn_prompts (0/1: the search root probes end turn once on a resampled clone and splits it into one action per
//   answer of the prompt it opens, e.g. Knowledge Demon's curse; 0 = such prompts answered at random, the old build; 1)
//   fp_ext (0/1: the search's observable-state fingerprint includes card damage / block and hidden power counters
//   (S0 ObservableState.Extended); 0 = the old fingerprint; 1)
//   potion_reserve (W; 1 = off; needs potion_values; not with boss2_model, tf_stake or the price probes: in every fight
//   scored by the macro table except the act-3 bosses, a potion subset's gain over the same subset without its top
//   potions is multiplied by W at end HP shares >= potion_reserve_hp_hi (0.3), fading to x1 at potion_reserve_hp_lo (0.2);
//   top = potion value >= potion_reserve_min (12); AgentConfig.PotionReserve), tf_dump_all (0/1: tf_dump covers every
//   room kind of every act; 0),
//   clone_probe (0/1/2: at every non-Ancient event decision take RunState.CloneReseeded and drop it (1), and also play
//   each open option once on a clone of its own (2); notes clone_ok / clone_reject:Reason:event; play unchanged; 0),
//   clone_dump (JSONL file, "{tag}" = actor tag: one line per probed event decision; off),
//   character (ironclad|silent|defect|necrobinder|regent or a CHARACTER.X id; defect), era (macro version-era token: 6 =
//   v0.113, the simulator's content; 4 = v0.111; ML.Era), tf_heal (0/1: T_f(h) = V_run at the HP after the simulator's
//   post-combat heals (Burning Blood, Black Blood, Meat on the Bone, Chosen Cheese; PostCombatHeal); 0 = the table at the
//   fight's end HP as before; 1),
//   potion_mix (FILE of src/ic113/potion_mix.py: ic113 mixed potion valuation, a new mode of the belt rule: with a
//   full belt and a potion reward waiting, the held potion of the lowest V_mix is discarded when the offered one scores
//   more by potion_mix_margin (0); V_mix(s, k) = Vbar_act(k) + potion_mix_shrink (0.2) x (dV(s, k) - Vbar_act(k)),
//   dV = V_run(belt = {k}) - V_run(empty belt) of the state row; potion_mix_phase=2: the table blend of PotionMix.cs;
//   potion_mix_extra=1: extra potion rewards of events / relics too; not with potion_values; off; AgentConfig.PotionMixShrink),
//   win_bound (EPS >= 0: the search caps a non-terminal node's value at the exact T_f of winning now minus EPS;
//   -1 = off; src/ic113/STALEMATE.md), stale_trace (FILE, "{tag}" = actor tag: per-decision trace of every
//   stalemated fight; stale_trace_enc=ENCOUNTER.X or all: also of every finished fight of it), seed_list (FILE of
//   "SEED ASCENSION" lines: the runs played, in order),
//   fake_merchant (0/1: the Fake Merchant event's relic purchases by the shop head, a Foul Potion throw by the event head;
//   0 = the simulator's default: leave; AgentConfig.FakeMerchant),
//   deck_route (0/1: run-level card prompts that were answered at random go to the smith / removal / card-reward head;
//   src/ic113/DECK_ROUTE.md; AgentConfig.DeckRoute), lethal_backstop (0/1: past max_rounds a root move that wins at
//   once on a clone is played instead of the prior's argmax; AgentConfig.LethalBackstop), kindle_cap (0/1: no Kindle at a
//   rest site while Pumpkin Candle's charges cover an upper bound of the fights left; kindle_act_fights per later act, 10;
//   AgentConfig.KindleCap; off)
// dotnet S3.dll encode out=DIR runs=N ascension=A [maps=K]: encoder round-trip data (EncodeTest); also prints K (8) map
//   decisions of the first run with the layout v2 inputs (boss ids, extra scalars, the 8 extra features per option)
// dotnet S3.dll props out=FILE mvocab=FILE: layout v3 content properties of the macro vocabulary (ContentProps)
// dotnet S3.dll prompttest: checks of the end-turn prompt expansion on a Knowledge Demon fight (PASS/FAIL, exit 1 on failure)
// dotnet S3.dll vocab out=FILE ids=FILE: the combat vocabulary of this simulator (ObsLayout.BuildVocab) and every id the macro
//   vocabulary could need (cards, enchantments, relics, potions, encounters ENCOUNTER.X of every act pool, events EVENT.X)
// dotnet S3.dll cardinfo out=FILE: type / rarity / pool of every card of this simulator (src/ic113/convert_v113.py)
// dotnet S3.dll healtest: PostCombatHeal against the simulator's own post-combat hooks (PASS/FAIL)
// dotnet S3.dll portcheck what=root|token|cards|events ...: v0.113 port checks (PortChecks.cs)
// dotnet S3.dll tables starts=F out=F mbridge=PATH [shard=i/N] [tf_heal=1] [check=1]: T_f tables of human battle starts by
//   the macro network (ic113 combat warm-up; HumanTables.cs)
using System.Diagnostics;
using System.Text.Json;
using Sts2Sim.Core.Content;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Characters;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S2;
using Sts2Sim.S3;

ModelDb.Init(ContentRegistry.AllTypes);
if (args.Length == 0) { Console.WriteLine("usage: S3 play|encode key=value ..."); return 2; }
var kv = args.Skip(1).Select(a => a.Split('=', 2)).ToDictionary(p => p[0], p => p.Length > 1 ? p[1] : "");
string Get(string k, string d) => kv.TryGetValue(k, out var v) ? v : d;
int GetI(string k, int d) => int.Parse(Get(k, d.ToString()));
double GetD(string k, double d) => double.Parse(Get(k, d.ToString(System.Globalization.CultureInfo.InvariantCulture)), System.Globalization.CultureInfo.InvariantCulture);
if (args[0] == "tftest") return TfDeathTest.Run();
if (args[0] == "tables") return HumanTables.Run(kv);   // ic113 warm-up: T_f tables of human battle starts (HumanTables.cs)
if (args[0] == "vocab") { new ObsLayout.Vocab(ObsLayout.BuildVocab()).Save(Get("out", "src/s0/data/combat_vocab_v113.json")); return ContentProps.SimIds(Get("ids", "src/s3/data/sim_ids_v113.json")); }
if (args[0] == "cardinfo") return ContentProps.CardInfo(Get("out", "src/s3/data/card_info_v113.json"));
if (args[0] == "healtest") return await PostCombatHeal.Test();
if (args[0] == "portcheck") return await PortChecks.Run(kv);
if (args[0] == "prompttest") return await PromptTest.Run(Get("vocab", "src/s0/data/combat_vocab_v113.json"));
if (args[0] == "props") return await ContentProps.Run(Get("out", "src/s3/data/content_props_v113.json"), Get("mvocab", "src/s3/data/macro_vocab_v113.json"));
if (args[0] == "encode") return await EncodeTest.Run(Get("out", "src/s3/out/encode"), GetI("runs", 20), GetI("ascension", 10), Get("mvocab", "src/s3/data/macro_vocab_v113.json"), GetI("maps", 8), Get("character", "defect"));
if (args[0] != "play") { Console.WriteLine("usage: S3 play|encode key=value ..."); return 2; }

string mode = Get("mode", "selfplay");
bool eval = mode == "eval";
string tag = Get("tag", "a0");
var vocab = ObsLayout.Vocab.Load(Get("vocab", "src/s0/data/combat_vocab_v113.json"));
Sts2Sim.S0.ObservableState.Extended = GetI("fp_ext", 1) == 1;
// obs=5 (2026-11-15): relic counters in the masked v5 slots (the live v3 network is bit-exact); obs=6 also corrects the
// enemy intent damage (an existing slot, so only for a retrained network). Default 4 keeps the pre-v5 input bit for bit.
int obsVersion = GetI("obs", 4);
CombatObservation.RelicCounters = obsVersion >= 5;
CombatObservation.ModifiedIntentDamage = obsVersion >= 6 || GetI("intent_damage", 0) == 1;
var mvocab = new MacroVocab(Get("mvocab", "src/s3/data/macro_vocab_v113.json"));
ML.Era = GetI("era", ML.EraV0113);
var character = CharacterOf(Get("character", "defect"));
int charToken = ML.CharIndex(character.Id.ToString());   // refuses a character without an S1 token
var cfg = new AgentConfig
{
    Ascension = GetI("ascension", 10), SimsFull = GetI("sims_full", 64), SimsFast = GetI("sims_fast", eval ? GetI("sims_full", 64) : 16),
    PFull = GetD("pfull", eval ? 1.0 : 0.25), BossSimsFactor = GetD("boss_sims", 1.0), RecordCombat = !eval && kv.ContainsKey("data"),
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
    // practice fights are training data: self-play only
    PracticeBoss = eval ? 0 : GetI("practice_boss", 0), PracticeElite = eval ? 0 : GetI("practice_elite", 0),
    PracticeMinAct = GetI("practice_min_act", 1), PracticePFull = GetD("practice_pfull", -1.0),
    DeathValue = Get("death_value", "zero"), DeathAlpha = GetD("death_alpha", 0.05), DeathScale = GetD("death_scale", 1.0),
    Boss2 = kv.ContainsKey("boss2_model") ? Boss2Model.Load(Get("boss2_model", "")) : null,
    TfStake = GetI("tf_stake", 0) == 1, StakeFloor = GetD("stake_floor", 0.02),
    AuditP = GetD("audit_p", 0), AuditSims = GetI("audit_sims", 1024), AuditAll = GetI("audit_all", 0) == 1,
    AuditPath = Get("audit", "").Replace("{tag}", tag),
    ExactBoss = eval ? GetI("exact_boss", 0) : 0,
    GreedyMap = GetI("greedy_map", 0) == 1,
    GreedyEvent = GetI("greedy_event", 0), MapRule = Get("map_rule", ""),
    RemoveStrike = GetI("remove_strike", 0) == 1, BadCardCut = GetD("bad_card_cut", double.NaN), BadCardAct = GetI("bad_card_act", 3),
    SkipBias = GetD("skip_bias", 0), SkipDeck = GetI("skip_deck", 34), SkipAct = GetI("skip_act", 2),
    LookBeta = GetD("look_beta", 0), LookKappa = GetD("look_kappa", 0), LookAct = GetI("look_act", 1),
    LookRemove = GetD("look_remove", 0), LookSmith = GetD("look_smith", 0), LookRest = GetD("look_rest", 0),
    LookShop = GetD("look_shop", 0), LookRelic = GetD("look_relic", 0), LookRelicAny = GetI("look_relic_any", 0) == 1,
    PotionScale = GetD("potion_scale", 1.0), HpScale = GetD("hp_scale", 1.0),
    PotionSwap = GetI("potion_swap", 1) == 1, PotionKindScale = GetI("potion_kind_scale", 0) == 1,
    ExactPart = Get("exact_part", ""),
    BossWorlds = GetI("boss_worlds", 16),
    PotionReserve = GetD("potion_reserve", 1.0), PotionReserveMin = GetD("potion_reserve_min", 12.0),
    PotionReserveHpLo = GetD("potion_reserve_hp_lo", 0.2), PotionReserveHpHi = GetD("potion_reserve_hp_hi", 0.3),
    CloneProbe = GetI("clone_probe", 0),
    TfHeal = GetI("tf_heal", 1) == 1,
    PotionMixShrink = GetD("potion_mix_shrink", 0.2), PotionMixPhase = GetI("potion_mix_phase", 1),
    PotionMixMargin = GetD("potion_mix_margin", 0), PotionMixExtra = GetI("potion_mix_extra", 0) == 1,
    FakeMerchant = GetI("fake_merchant", 0) == 1,
    DeckRoute = GetI("deck_route", 0) == 1, LethalBackstop = GetI("lethal_backstop", 0) == 1,
    KindleCap = GetI("kindle_cap", 0) == 1, KindleActFights = GetI("kindle_act_fights", 10),
};
if (kv.ContainsKey("potion_mix") && kv.ContainsKey("potion_values")) { Console.WriteLine("potion_mix replaces the potion_values belt rule: use one of them"); return 2; }
if (!kv.ContainsKey("potion_mix") && (cfg.PotionMixExtra || kv.ContainsKey("potion_mix_phase"))) { Console.WriteLine("potion_mix_extra / potion_mix_phase need potion_mix=FILE"); return 2; }
if (cfg.PotionReserve != 1.0)
{
    if (!kv.ContainsKey("potion_values")) { Console.WriteLine("potion_reserve needs potion_values=FILE"); return 2; }
    if (cfg.Boss2 is not null || cfg.TfStake || cfg.PotionScale != 1.0 || cfg.HpScale != 1.0 || cfg.PotionKindScale || cfg.Scorer != "tf")
    { Console.WriteLine("potion_reserve is not combined with boss2_model, tf_stake, potion_scale, hp_scale, potion_kind_scale or scorer=hp"); return 2; }
}
if (cfg.TfStake && (cfg.DeathValue != "zero" || cfg.Boss2 is not null)) { Console.WriteLine("tf_stake=1 replaces death_value and boss2_model: use it alone"); return 2; }
if (cfg.DeathValue is not ("zero" or "progress" or "progress_raw")) { Console.WriteLine($"unknown death_value {cfg.DeathValue} (zero|progress|progress_raw)"); return 2; }
string alphaFrom = $"death_alpha {cfg.DeathAlpha}";
if (kv.ContainsKey("death_alpha_log"))
{
    // the macro trainer's alpha at its last iteration (one JSON object per line; a line cut off by a crash is skipped)
    string lp = Get("death_alpha_log", "");
    double? found = null;
    try
    {
        foreach (string line in File.ReadLines(lp))
        {
            try
            {
                using var doc = JsonDocument.Parse(line);
                if (doc.RootElement.TryGetProperty("alpha", out var av) && av.ValueKind == JsonValueKind.Number) found = av.GetDouble();
            }
            catch (JsonException) { }
        }
    }
    catch (Exception e) { alphaFrom += $" (death_alpha_log {lp} unreadable: {e.GetType().Name})"; }
    if (found is { } fa && double.IsFinite(fa) && fa >= 0 && fa <= 1) { cfg.DeathAlpha = fa; alphaFrom = $"alpha {fa} from {lp}"; }
    else if (!alphaFrom.Contains("unreadable")) alphaFrom += $" (no alpha in {lp})";
}
if (eval && (GetI("practice_boss", 0) > 0 || GetI("practice_elite", 0) > 0)) Console.WriteLine($"[{tag}] practice fights ignored in eval mode");
double perturb = GetD("perturb", eval ? 0.0 : 0.1);
ulong seed = ulong.Parse(Get("seed", "1"));
int concurrent = GetI("concurrent", 32), maxFloors = GetI("max_floors", 80);
double seconds = GetD("seconds", 1e18);
string seedPrefix = Get("seeds", eval ? "EVAL" : $"S3-{tag}-{seed}-");
int first = GetI("first", 0);
int total = eval ? GetI("count", 100) : GetI("runs", int.MaxValue);
// seed_list=FILE (src/ic113/STALEMATE.md): lines "SEED ASCENSION"; the runs are these seeds at these levels, in order
List<(string Seed, int Asc)>? seedList = kv.ContainsKey("seed_list")
    ? File.ReadAllLines(Get("seed_list", "")).Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries)).Where(x => x.Length >= 2)
        .Select(x => (x[0], int.Parse(x[1]))).ToList() : null;
if (seedList is not null) total = Math.Min(total, seedList.Count);

string bridgePath = Get("bridge", "uniform"), mbridgePath = Get("mbridge", "uniform");
BridgeEvaluator? bridge = bridgePath == "uniform" ? null : new BridgeEvaluator(bridgePath, vocab);
MacroBridgeEvaluator? mbridge = mbridgePath == "uniform" ? null : new MacroBridgeEvaluator(mbridgePath);
// a run waits for at most one combat evaluation at a time (the search awaits its simulations one by one), so the
// number of runs in flight is limited by the bridge capacity
if (bridge is not null && concurrent > bridge.Capacity) concurrent = bridge.Capacity;
var services = new ActorServices
{
    Net = bridge is null ? new UniformEvaluator() : bridge, Macro = mbridge is null ? new UniformMacroEvaluator() : mbridge,
    Vocab = vocab, MVocab = mvocab, Cfg = cfg,
    CombatWriter = cfg.RecordCombat ? new ChunkWriter(Get("data", ""), tag, GetI("chunk", 4096)) : null,
};
services.Trace = GetI("trace", 0) == 1;
if (kv.ContainsKey("tf_dump"))
{
    string dumpPath = Get("tf_dump", "").Replace("{tag}", tag);   // {tag}: one file per actor
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dumpPath))!);
    services.TfDump = new StreamWriter(dumpPath, append: true);
    services.TfDumpK = GetI("tf_dump_k", 20);
    services.TfDumpAll = GetI("tf_dump_all", 0) == 1;
}
if (kv.ContainsKey("clone_dump"))
{
    string cdPath = Get("clone_dump", "").Replace("{tag}", tag);   // {tag}: one file per actor
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(cdPath))!);
    services.CloneDump = new StreamWriter(cdPath, append: true);
}
if (kv.ContainsKey("look_dump"))
{
    string ldPath = Get("look_dump", "").Replace("{tag}", tag);   // {tag}: one file per actor
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(ldPath))!);
    services.LookDump = new StreamWriter(ldPath, append: true);
}
if (kv.ContainsKey("potion_values"))
{
    services.PotionValues = JsonSerializer.Deserialize<Dictionary<string, double>>(File.ReadAllText(Get("potion_values", "")))!;
    var sorted = services.PotionValues.Values.OrderBy(x => x).ToList();
    services.PotionValueDefault = sorted.Count > 0 ? sorted[sorted.Count / 2] : 0;
    services.PotionSwapMargin = GetD("potion_swap_margin", 0);
    services.PotionBuyMin = GetD("potion_buy_min", 0);
}
if (kv.ContainsKey("potion_mix")) services.PotionMix = PotionMix.Load(Get("potion_mix", ""), cfg.PotionMixPhase);
if (kv.ContainsKey("card_values"))
{
    services.CardValues = JsonSerializer.Deserialize<Dictionary<string, double>>(File.ReadAllText(Get("card_values", "")))!;
    services.CardBonus = GetD("card_bonus", 0.5); services.CardBonusAct = GetI("card_bonus_act", 3);
}
if (kv.ContainsKey("starts_dump"))
{
    string sdPath = Get("starts_dump", "").Replace("{tag}", tag);   // {tag}: one file per actor
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(sdPath))!);
    services.StartsDump = new StreamWriter(sdPath, append: true);
    services.StartsDumpElite = GetI("starts_dump_elite", 0) == 1;
}
var runWriter = !eval && kv.ContainsKey("mdata") ? new RunChunkWriter(Get("mdata", ""), tag, GetI("mchunk", 16)) : null;
string outPath = Get("out", $"src/s3/out/{tag}.jsonl");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
using var log = new StreamWriter(outPath, append: true);
StreamWriter? fightLog = kv.ContainsKey("fights") ? new StreamWriter(Get("fights", ""), append: true) : null;
if (fightLog is not null)
    services.FightLog = (o, rec, floor) =>
    {
        var d = FightRunner.LogRecord(o, rec.Fights, tag);
        d["run_seed"] = rec.Seed; d["floor"] = floor;
        if (o.Practice) d["practice"] = true;
        if (cfg.Search.Reuse) d["reused_visits"] = o.ReusedVisits;
        if (cfg.DeathValue != "zero") d["death_value"] = Math.Round(o.DeathValue, 5);
        if (o.TfKind is not null) d["tf_kind"] = o.TfKind;
        if (!double.IsNaN(o.ReserveTMax0)) d["reserve_t_max0"] = Math.Round(o.ReserveTMax0, 5);
        fightLog.WriteLine(JsonSerializer.Serialize(d));
    };
if (kv.ContainsKey("stale_trace"))
{
    // src/ic113/STALEMATE.md: per-decision trace of stalemated fights (and of every fight of stale_trace_enc)
    string stPath = Get("stale_trace", "").Replace("{tag}", tag);
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(stPath))!);
    RunAgent.StaleTrace = new StreamWriter(stPath, append: true);
    RunAgent.StaleTraceEnc = kv.TryGetValue("stale_trace_enc", out var ste) ? ste : null;
}
var rng = new Random((int)(seed % int.MaxValue));
Console.WriteLine($"[{tag}] mode {mode}, character {character.Id} (token {charToken}), era {ML.Era}, T_f post-combat heals {(cfg.TfHeal ? "on" : "off")}, ascension {cfg.Ascension} (share {(eval ? 1.0 : GetD("asc_mix", 1.0))}), concurrent {concurrent}, sims {cfg.SimsFull}/{cfg.SimsFast} pfull {cfg.PFull}, " +
    $"noise {cfg.Search.GumbelNoise}, greedy macro {cfg.Greedy}, scorer {cfg.Scorer}, perturb {perturb}, combat {bridgePath}, macro {mbridgePath}, " +
    $"runs {(total == int.MaxValue ? "unbounded" : total)}" +
    (cfg.PracticeBoss > 0 || cfg.PracticeElite > 0 ? $", practice boss {cfg.PracticeBoss} elite {cfg.PracticeElite} from act {cfg.PracticeMinAct}" : "") +
    (cfg.Search.Reuse ? $", tree reuse (cap {cfg.Search.ReuseCap})" : "") +
    (cfg.Search.EndTurnPrompts ? ", end-turn prompts at the root" : "") +
    (CombatObservation.Version >= 5 ? $", obs v{CombatObservation.Version} ({(CombatObservation.RelicCounters ? "relic counters" : "")}{(CombatObservation.ModifiedIntentDamage ? (CombatObservation.RelicCounters ? ", " : "") + "modified intent damage" : "")})" : "") +
    (cfg.DeathValue != "zero" ? $", death value {cfg.DeathValue} x {cfg.DeathScale} ({alphaFrom})" : "") +
    (cfg.Boss2 is not null ? $", double-boss model {cfg.Boss2}" : "") +
    (services.TfDump is not null ? $", T_f dump {Get("tf_dump", "").Replace("{tag}", tag)} (k {services.TfDumpK})" : ""));
if (services.PotionMix is { } pmix)
    Console.WriteLine($"[{tag}] potion_mix {pmix.Path}: shrink {cfg.PotionMixShrink}, margin {cfg.PotionMixMargin}, extra rewards {(cfg.PotionMixExtra ? "on" : "off")}, {pmix.Describe()}");

var wall = Stopwatch.StartNew();
int next = 0, done = 0, won = 0, errors = 0;
long floors = 0;
var lastReport = Stopwatch.StartNew();

// fail fast if the single thread hangs inside the engine (a blocked wait cannot be seen by the pump's own check).
// Progress is real work: answered network requests and finished runs. Agent callbacks do not count (2026-11-12: an
// endless fight kept calling the agent, so a callback counter never stalled). Without bridges, callbacks are all there is.
long lastProgress = -1; var stall = Stopwatch.StartNew();
var watchdog = new Thread(() =>
{
    while (true)
    {
        Thread.Sleep(10_000);
        long p = bridge is null && mbridge is null ? Interlocked.Read(ref services.Progress)
            : (bridge?.Requests ?? 0) + (mbridge?.Requests ?? 0) + Volatile.Read(ref done);
        if (p != lastProgress) { lastProgress = p; stall.Restart(); }
        else if (stall.Elapsed.TotalSeconds > 300)
        {
            Console.WriteLine($"[{tag}] no progress for 300 s (last callback {services.WhereWho}, run {services.WhereSeed}, floor {services.WhereFloor}): exiting");
            Environment.Exit(3);
        }
    }
}) { IsBackground = true };
watchdog.Start();

// Self-play ascension: a share `asc_mix` of the runs at the target level (`ascension`), the rest uniformly at the
// levels below it. Evaluation always plays the target level.
double ascMix = GetD("asc_mix", 1.0);
// asc_weights=w0,w1,... (one weight per level below `ascension`): the levels below are drawn in proportion to these
// weights instead of uniformly; absent = uniform (the original draw, unchanged)
double[]? ascW = null;
if (kv.ContainsKey("asc_weights"))
{
    ascW = Get("asc_weights", "").Split(',').Select(x => double.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
    if (ascW.Length != cfg.Ascension || ascW.Any(w => w < 0) || ascW.Sum() <= 0)
        throw new ArgumentException($"asc_weights needs {cfg.Ascension} non-negative weights with a positive sum");
}
int Lower()
{
    if (ascW is null) return rng.Next(cfg.Ascension);
    double u = rng.NextDouble() * ascW.Sum();
    for (int k = 0; k < ascW.Length; k++) { u -= ascW[k]; if (u < 0) return k; }
    return Array.FindLastIndex(ascW, w => w > 0);
}
int SampleAscension() => eval || cfg.Ascension == 0 || rng.NextDouble() < ascMix ? cfg.Ascension : Lower();

async Task<RunRecord> PlayRunAsync(string runSeed, bool perturbed, int ascFixed = -1)
{
    int asc = ascFixed >= 0 ? ascFixed : SampleAscension();
    var acts = ActDefinition.GetRandomList(runSeed);
    var run = new RunState(runSeed, acts, ascensionLevel: asc);
    var player = Player.CreateForNewRun(character, run);
    run.AddPlayer(player);
    var agent = new RunAgent(services, run, player, runSeed, perturbed, asc);
    var driver = new RunDriver(run, agent);
    driver.OnRoomResolved += agent.OnRoomResolved;
    try { agent.Finish(await driver.RunAsync(maxFloors)); }
    catch (Exception e) { agent.Fail(e); }
    return agent.Rec;
}

async Task Worker()
{
    while (true)
    {
        if (next >= total || wall.Elapsed.TotalSeconds > seconds) return;
        int i = next++;
        RunRecord rec = seedList is not null ? await PlayRunAsync(seedList[i].Seed, false, seedList[i].Asc)
            : await PlayRunAsync($"{seedPrefix}{first + i:D6}", !eval && rng.NextDouble() < perturb);
        log.WriteLine(JsonSerializer.Serialize(rec.Log(tag, first + i)));
        runWriter?.Add(rec);
        done++; floors += rec.Floors;
        if (rec.Won) won++;
        if (rec.Error) { errors++; if (errors <= 20) Console.WriteLine($"[{tag}] run {rec.Seed} error at floor {rec.Floors}: {rec.ErrorText}"); }
        if (lastReport.Elapsed.TotalSeconds > 30) { lastReport.Restart(); log.Flush(); fightLog?.Flush(); services.TfDump?.Flush(); services.StartsDump?.Flush(); services.LookDump?.Flush(); services.CloneDump?.Flush(); Report(); }
    }
}

void Report()
{
    double t = wall.Elapsed.TotalSeconds;
    Console.WriteLine($"[{tag}] {t:F0}s runs {done} (won {won}, errors {errors}, mean floors {floors / (double)Math.Max(1, done):F1}) = {3600 * done / t:F0}/h, " +
        $"fights {services.Fights} (won {services.FightsWon}), combat decisions {services.CombatDecisions}, sims {services.Simulations} = {services.Simulations / t:F0}/s, " +
        $"macro decisions {services.MacroDecisions}, T_f tables {services.TfBuilds} ({services.TfRows} rows)" +
        (bridge is null ? "" : $", combat evals {bridge.States} in {bridge.Requests} batches (mean {bridge.States / (double)Math.Max(1, bridge.Requests):F0}), bridge wait {100 * bridge.WaitSeconds / t:F0}%") +
        (mbridge is null ? "" : $", macro rows {mbridge.Rows} in {mbridge.Requests} requests") +
        (services.CombatWriter is null ? "" : $", combat records {services.CombatWriter.Written}") +
        (cfg.PracticeBoss > 0 || cfg.PracticeElite > 0 ? $", practice fights {services.PracticeFights} (won {services.PracticeWon}, abandoned {services.PracticeAbandoned}), " +
            $"practice decisions {services.PracticeDecisions}, practice sims {services.PracticeSimulations}, practice time {services.PracticeSeconds:F0}s" : "") +
        (cfg.DeathValue != "zero" || cfg.Boss2 is not null ? $", death values capped {services.TfDeathCapped}, T_f entries raised to the death value {services.TfDeathFloorRaised}, double-boss tables {services.TfBoss2Fights}" : "") +
        (services.TfMsFights > 0 ? $", milestone tables {services.TfMsFights}" : "") +
        (cfg.TfStake ? $", stake tables {services.TfStakeFights} (mean banked {services.TfStakeBankedSum / Math.Max(1, services.TfStakeFights):F3}, entries floored {services.TfStakeFloored}; {alphaFrom})" : "") +
        (cfg.Search.EndTurnPrompts ? $", end-turn probes {ActionSpace.EndTurnProbes} (prompt {ActionSpace.EndTurnProbesWithPrompt}, failed {ActionSpace.EndTurnProbeErrors})" : "") +
        (cfg.Search.Reuse ? $", reuse: {services.ReusedVisits / (double)Math.Max(1, services.ReuseSearches):F1} root visits reused per searched move over {services.ReuseSearches}, largest table {services.ReuseTableMax}" : "") +
        (runWriter is null ? "" : $", macro rows written {runWriter.RowsWritten}") +
        (services.PotionMix is null ? "" : $", potion_mix decisions {services.PmDecisions[0]}+{services.PmDecisions[1]} swaps {services.PmSwaps[0]}+{services.PmSwaps[1]}"));
    if (GetI("timing", 1) == 1) Console.WriteLine(TimingLine(t));
}

// Step 0 profiling (2026-11-13): where the actor's CPU goes. Process CPU covers every thread (GC, spin-waits); the
// listed parts are synchronous sections of the single actor thread; "outside search" is the actor thread's busy time
// (wall - bridge wait) minus the search parts listed (run driver, live steps, macro encoding, T_f tables, pump, ...).
string TimingLine(double t)
{
    double cpu = Process.GetCurrentProcess().TotalProcessorTime.TotalSeconds;
    double busy = t - (bridge?.WaitSeconds ?? 0);
    double inSearch = Timing.Sim + Timing.RootLegal + Timing.Decode;
    double rootEnc = Math.Max(0, Timing.Encode - Timing.SimEval);
    double simRest = Timing.Sim - Timing.Clone - Timing.Apply - Timing.Hash - Timing.Legal - Timing.SimEval;
    string P(double x) => $"{x:F1}s ({100 * x / Math.Max(1e-9, busy):F0}%)";
    long sims = Math.Max(1, Timing.Sims);
    return $"[{tag}] timing {t:F0}s: process CPU {cpu:F0}s = {cpu / t:F2} cores, actor busy {busy:F0}s; " +
        $"in simulations {P(Timing.Sim)} = clone {P(Timing.Clone)}, apply {P(Timing.Apply)} (replay {P(Timing.ApplyReplay)}, " +
        $"frontier {P(Timing.ApplyFront)}; end turns {P(Timing.ApplyEnd)}), hash {P(Timing.Hash)}, legal {P(Timing.Legal)}, " +
        $"leaf encode {P(Timing.SimEval)}, select/backup {P(simRest)}; root legal {P(Timing.RootLegal)}, root encode {P(rootEnc)}, " +
        $"decode {P(Timing.Decode)}, records {P(Timing.Record)}; outside search {P(busy - inSearch - rootEnc - Timing.Record)}; " +
        $"sims {Timing.Sims} ({Timing.Sims / t:F0}/s, {1e6 * Timing.Sim / sims:F0} us each), steps/sim {Timing.Steps / (double)sims:F2} " +
        $"(end turns {Timing.EndSteps / (double)sims:F2}), leaves {Timing.Leaves / (double)sims:F2}, terminals {Timing.Terminals / (double)sims:F2}, " +
        $"tree hits {Timing.Hits / (double)sims:F2} per sim; us/step {1e6 * Timing.Apply / Math.Max(1, Timing.Steps):F0} " +
        $"(end turn {1e6 * Timing.ApplyEnd / Math.Max(1, Timing.EndSteps):F0}), clone {1e6 * Timing.Clone / sims:F0}; " +
        $"GC alloc {GC.GetTotalAllocatedBytes() / 1e9:F1} GB ({GC.GetTotalAllocatedBytes() / 1e6 / t:F0} MB/s), " +
        $"collections {GC.CollectionCount(0)}/{GC.CollectionCount(1)}/{GC.CollectionCount(2)}, pause {GC.GetTotalPauseDuration().TotalSeconds:F1}s";
}

// Combat evaluations are answered within a flush. A macro request (few rows, slower per row) stays in flight while
// fights go on: it is completed when its answer is there, and waited for only when nothing else can run.
bool Flush()
{
    bool did = false;
    if (mbridge is not null)
    {
        if (mbridge.InFlight && mbridge.Ready) { mbridge.Complete(); did = true; }
        if (!mbridge.InFlight) mbridge.Submit();
    }
    if (bridge is not null && bridge.Submit()) { bridge.Complete(); did = true; }
    if (!did && mbridge is not null && mbridge.InFlight) { mbridge.Complete(); did = true; }
    return did;
}

Pump.Run(() => Task.WhenAll(Enumerable.Range(0, concurrent).Select(_ => Worker())), Flush);
services.CombatWriter?.Flush();
runWriter?.Flush();
log.Flush(); fightLog?.Flush(); services.TfDump?.Flush(); services.StartsDump?.Flush(); services.LookDump?.Flush(); services.CloneDump?.Flush();
Report();
Console.WriteLine($"[{tag}] macro vocabulary: " + string.Join(", ", mvocab.Seen.Where(x => x.Value > 0).Select(x => $"{x.Key} unknown {mvocab.Unknown[x.Key]}/{x.Value}"))
    + (mvocab.UnknownIds.Count > 0 ? "; unknown ids: " + string.Join(" ", mvocab.UnknownIds) : ""));
Console.WriteLine($"[{tag}] run-level card selections: " + string.Join(", ", services.SelectionKinds.OrderByDescending(x => x.Value).Select(x => $"{x.Key} {x.Value}")));
Console.WriteLine($"[{tag}] T_f HP-axis formula check: max |logit(a, c at own HP) - network logit| = {services.TfFormulaMaxError:E2} over {services.TfRows} table rows");
Console.WriteLine($"[{tag}] notes: " + string.Join(", ", services.Notes.OrderByDescending(x => x.Value).Take(120).Select(x => $"{x.Key} {x.Value}")));
if (cfg.PotionReserve != 1.0)
    Console.WriteLine($"[{tag}] potion reserve W {cfg.PotionReserve} (top >= {cfg.PotionReserveMin}, HP ramp {cfg.PotionReserveHpLo}-{cfg.PotionReserveHpHi}): " +
        $"fights {services.PotionReserveFights}, entries raised {services.PotionReserveEntries}, mean T_max ratio {services.PotionReserveTMaxRatioSum / Math.Max(1, services.PotionReserveFights):F4}");
if (cfg.CloneProbe > 0)
    Console.WriteLine($"[{tag}] clone probe: event decisions {services.CloneProbes}, clones ok {services.CloneOk}, " +
        $"mean {1000 * services.CloneSeconds / Math.Max(1, services.CloneProbes):F2} ms, max {1000 * services.CloneMaxSeconds:F1} ms" +
        (cfg.CloneProbe >= 2 ? $"; option probes {services.OptionProbes} (failed {services.OptionProbeErrors}), mean {1000 * services.OptionProbeSeconds / Math.Max(1, services.OptionProbes):F2} ms" : ""));
if (services.LookN > 0) Console.WriteLine($"[{tag}] lookahead: {services.LookN} cards scored, mean |value-logit change| {services.LookDz / services.LookN:F4}");
if (services.LookStats.Count > 0)
    Console.WriteLine($"[{tag}] look types: " + string.Join(", ", services.LookStats.OrderBy(x => x.Key).Select(x =>
        $"{x.Key} decisions {services.Notes.GetValueOrDefault("lookahead_" + x.Key)} changed {services.Notes.GetValueOrDefault("look_changed_" + x.Key)} " +
        $"same {services.Notes.GetValueOrDefault("look_same_" + x.Key)} candidates {x.Value.N} mean |dz| {x.Value.SumAbsDz / Math.Max(1, x.Value.N):F4}")));
if (services.PotionMix is not null)
    for (int src = 0; src < (cfg.PotionMixExtra ? 2 : 1); src++)
        Console.WriteLine($"[{tag}] potion_mix {(src == 0 ? "reward" : "extra")}: decisions {services.PmDecisions[src]}, swaps {services.PmSwaps[src]}, " +
            $"kinds scored {services.PmKinds[src]} (without Vbar {services.PmNoMean[src]}), macro rows {services.PmRows[src]}" +
            (src == 0 ? $", mean |dV - Vbar| {services.PmSumAbsDev / Math.Max(1, services.PmDevN):F4} over {services.PmDevN}" : ""));
if (Get("shutdown", "0") == "1") { bridge?.Shutdown(); mbridge?.Shutdown(); }
bridge?.Dispose(); mbridge?.Dispose();
return 0;

/// <summary>`character=`: a short name (ironclad, silent, defect, necrobinder, regent, fool) or a CHARACTER.X id.</summary>
static Sts2Sim.Core.Models.CharacterModel CharacterOf(string name)
{
    string id = name.StartsWith("CHARACTER.", StringComparison.Ordinal) ? name : "CHARACTER." + name.ToUpperInvariant();
    return ModelDb.GetById<Sts2Sim.Core.Models.CharacterModel>(ModelId.Deserialize(id));
}
