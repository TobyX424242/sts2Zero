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
    /// <summary>With <see cref="Greedy"/>: also take the map head's most probable option (2026-11-15 test; default
    /// off: map choices are sampled in evaluation too).</summary>
    public bool GreedyMap { get; set; }
    /// <summary>With <see cref="Greedy"/> (2026-11-16 test; 0 = off: event options are sampled): take the event head's
    /// most probable option, except options already taken this many times on the current floor (the repeat guard: the
    /// plain argmax held on at Slippery Bridge until death).</summary>
    public int GreedyEvent { get; set; }
    /// <summary>Route probes (2026-11-16, evaluation; "" = the map head): "uniform" = a random next node; "elite_max" /
    /// "elite_min" = the next node with the most / fewest elites reachable below it (most: the path maximum, fewest: the
    /// path minimum), ties broken by the map head; "hpT" (e.g. hp0.7) = elite_max while HP >= T x max HP, else elite_min
    /// (the hand rule "fight elites when healthy, avoid them when hurt"); "emax23" = elite_max in acts 2 and 3 only, the
    /// map head in the other acts.</summary>
    public string MapRule { get; set; } = "";
    /// <summary>Deployment probes of 2026-11-16 (all off by default).
    /// RemoveStrike: a removal (shop service or a "remove" prompt) takes a Strike, un-upgraded first, unless a curse or
    /// status card can be removed (those stay with the removal head).
    /// BadCardCut: in card rewards from act <see cref="BadCardAct"/> on, a card whose value (card_values file) is below
    /// the cut gets a logit penalty of 20 (it is taken only when nothing else, skip included, is offered).
    /// SkipBias: from act <see cref="SkipAct"/> on and with at least <see cref="SkipDeck"/> cards, Skip gets this logit bonus.
    /// LookBeta: one-step value lookahead in card rewards from act <see cref="LookAct"/> on: each card's logit gets
    /// beta x (value logit of the state with that card - value logit of the state without it) - kappa x the rise of the
    /// ensemble spread; Skip gets 0. With the argmax this picks argmax log pi(a) + bonus(a).</summary>
    public bool RemoveStrike { get; set; }
    public double BadCardCut { get; set; } = double.NaN;
    public int BadCardAct { get; set; } = 3;
    public double SkipBias { get; set; }
    public int SkipDeck { get; set; } = 34;
    public int SkipAct { get; set; } = 2;
    public double LookBeta { get; set; }
    public double LookKappa { get; set; }
    public int LookAct { get; set; } = 1;
    /// <summary>One-step value lookahead for the other deterministic macro decisions (2026-11-16, evaluation; each a beta
    /// like <see cref="LookBeta"/>, 0 = off, from act <see cref="LookAct"/> on). Each candidate's post-state is the state
    /// row (StateRow) with the candidate's effect applied as the encoder would encode it (MacroEncoder.With*); its logit
    /// gets beta x (z(post) - z(state)), z = the ensemble value logit; candidates without a modelled post-state get 0.
    /// LookRemove: removal targets (shop removal service, "remove" prompts): that deck slot removed (later slots move down).
    /// LookSmith: upgrade targets (rest-site smith, "upgrade" prompts): that card's upgrade level + 1.
    /// LookRest: rest-site options: heal = the state row's value at HP + the heal the game will give (30% of max HP
    /// through the rest-site heal hooks, e.g. Regal Pillow; read off the HP ramps, no evaluation); smith = the row with the
    /// smith head's argmax target upgraded (one evaluation); other options 0.
    /// LookShop: each purchase: card -> deck + card, relic -> relics + relic, potion -> first free belt slot, all with the
    /// gold reduced by the price; the removal service and LEAVE 0 (the removal target is then LookRemove's).
    /// LookRelic: Ancient relic choices: relic added; only when no offered relic has an on-pickup effect (overrides
    /// RelicModel.AfterObtained or HasUponPickupEffect), unless <see cref="LookRelicAny"/> (then the relic token alone
    /// stands for every relic, its pickup effect left out).</summary>
    public double LookRemove { get; set; }
    public double LookSmith { get; set; }
    public double LookRest { get; set; }
    public double LookShop { get; set; }
    public double LookRelic { get; set; }
    public bool LookRelicAny { get; set; }
    /// <summary>Price probes (2026-11-16, 1 = off): in every fight scored by the macro network's table (not the final
    /// boss, not the double-boss model), the table's potion gains (a subset's value over the empty belt at the same HP)
    /// are multiplied by <see cref="PotionScale"/> and its HP drops (the empty belt's value at full HP minus its value at
    /// HP h) by <see cref="HpScale"/>, floored at 2% of the unscaled value. Tests whether the value network prices HP and
    /// potions too flatly.</summary>
    public double PotionScale { get; set; } = 1.0;
    /// <summary>With potion values loaded (potion_values=FILE): a subset's potion gain is also multiplied by the mean
    /// over its potions of value / median value (clamped to 0.1-4), so the search spares the potions that are worth
    /// most at the final boss (Focus Potion about 3.3) and spends the worthless ones first.</summary>
    public bool PotionKindScale { get; set; }
    /// <summary>With potion values loaded: the belt management rule of <see cref="ActorServices.PotionValues"/>.</summary>
    public bool PotionSwap { get; set; }
    public double HpScale { get; set; } = 1.0;
    /// <summary>Reservation price for the top potions (2026-11-17, PROPOSAL §6.1 step 8; 1 = off; needs potion values).
    /// In every fight scored by the macro network's table except the act-3 bosses (so every fight of acts 1-2 and the
    /// act-3 hallway and elite fights), and only when no other table option is on, each subset P's table becomes
    /// T'(h, P) = min(1, T(h, P) + (W - 1) ramp(h / max HP) max(0, T(h, P) - T(h, P without its top potions))),
    /// ramp(x) = clamp((x - lo) / (hi - lo), 0, 1); a kind is "top" when its potion value is at least
    /// <see cref="PotionReserveMin"/>. Applied to the network's table before free disposal (TfScorer.Set keeps it).</summary>
    public double PotionReserve { get; set; } = 1.0;
    public double PotionReserveMin { get; set; } = 12.0;
    public double PotionReserveHpLo { get; set; } = 0.2;
    public double PotionReserveHpHi { get; set; } = 0.3;
    /// <summary>Scoping probe (2026-11-17, PROPOSAL §6.1 step 10; 0 = off): at every event decision with two or more open
    /// options (not Ancients), take <c>RunState.CloneReseeded</c> and drop it (1), and also play each option once on a fresh
    /// clone of its own (2). Timed and noted (clone_ok / clone_reject:Reason:event); play is unchanged.</summary>
    public int CloneProbe { get; set; }
    /// <summary>ic113 (2026-11-18): T_f(h) = V_run(post-fight state at the HP the simulator's post-combat hooks leave after a
    /// fight won at end HP h) (<see cref="PostCombatHeal"/>: Burning Blood, Black Blood, Meat on the Bone, Chosen Cheese).
    /// The search's terminal states and the combat network's end-HP target read the HP before those hooks (the fight
    /// ends before CombatRoom resolves its outcome), so the heal enters exactly once, through the table. Death (h = 0) is
    /// never healed. false = the table at the end HP as before.</summary>
    public bool TfHeal { get; set; } = true;
    /// <summary>ic113 mixed potion valuation (potion_mix=FILE, 2026-11-18; off without the file; <see cref="PotionMix"/>):
    /// a new mode of the belt rule. With a full belt and a potion reward about to be taken, every held kind and the offered
    /// one are scored by V_mix (phase 1: Vbar_act(k) + <see cref="PotionMixShrink"/> (dV(s, k) - Vbar_act(k)); phase 2 with
    /// <see cref="PotionMixPhase"/> = 2), and the lowest-scored held potion is discarded when the offered one scores more by
    /// <see cref="PotionMixMargin"/>. <see cref="PotionMixExtra"/>: extra potion rewards (events, relics; not the optional
    /// merchant potion) met with a full belt get the same rule.</summary>
    public double PotionMixShrink { get; set; } = 0.2;
    public int PotionMixPhase { get; set; } = 1;
    public double PotionMixMargin { get; set; }
    public bool PotionMixExtra { get; set; }
    /// <summary>ic113 (fake_merchant=1, 2026-11-18; off = the simulator's default, which always leaves): the Fake Merchant
    /// event's purchases are agent decisions. Its stock (6 FAKE_* relics at event prices) goes through the SHOP head like a
    /// normal shop's relics (option = relic + price features, LEAVE = the zero option; look_shop as in shops), one purchase
    /// per step until LEAVE or nothing affordable. With a usable Foul Potion the EVENT head first chooses between throwing it
    /// (a fight; reward: the merchant's rug and every unbought relic) and the shop (hashed keys FAKE_MERCHANT:use_potion /
    /// FAKE_MERCHANT:shop), once per visit.</summary>
    public bool FakeMerchant { get; set; }
    /// <summary>deck_route=1 (src/ic113/DECK_ROUTE.md): run-level card prompts without a deck purpose that used to get a
    /// random answer go to the closest existing head: deck cards that get better (enchant, upgrade, duplicate) to the smith
    /// head, deck cards lost or replaced (remove, transform) to the removal head, new cards to the card-reward head (several
    /// picks one by one). Bundles and Scroll Boxes keep the random answer. Off = the old behaviour exactly.</summary>
    public bool DeckRoute { get; set; }
    /// <summary>lethal_backstop=1: past max_rounds (search off, the prior's argmax plays) a root action that wins the fight at
    /// once on a resampled clone is played instead (the best end value among such actions). Off = the old behaviour.</summary>
    public bool LethalBackstop { get; set; }
    /// <summary>kindle_cap=1 (owner, 2026-11-20, live play): at a rest site, Kindle (Pumpkin Candle: +5 combats of +1 energy)
    /// is not offered while the candle's remaining charges already cover an upper bound of the fights left in the run:
    /// the most Monster + Elite + Unknown rooms of each kind on any path from the rest site to the act's boss, plus the
    /// act's boss(es), plus <see cref="KindleActFights"/> per act still to come. The macro row has no relic counters, so
    /// the network picked Kindle at 53% of candle rest sites in s3ic self-play (segments 20-21) and left ~7 charges unused.
    /// Off = the old behaviour exactly.</summary>
    public bool KindleCap { get; set; }
    /// <summary>Fights counted per act not yet reached (kindle_cap): won s3ic runs had 22 fights on average, at most 29.</summary>
    public int KindleActFights { get; set; } = 10;
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
    /// <summary>Target audit (2026-11-15, off by default): on a share <see cref="AuditP"/> of fully searched boss moves
    /// (all rooms with <see cref="AuditAll"/>), the same state is searched again with the same budget and a fresh seed,
    /// and once with <see cref="AuditSims"/> simulations; the agreement of the policy targets is written to
    /// <see cref="AuditPath"/> ({tag} = actor tag). The live decision is the first search's, as without the audit.</summary>
    public double AuditP { get; set; }
    public int AuditSims { get; set; } = 1024;
    public bool AuditAll { get; set; }
    public string AuditPath { get; set; } = "";
    /// <summary>Evaluation only (2026-11-15): 1 = every boss fight searches in exact clones (the true future), 3 = act-3
    /// boss fights only; 0 = off.</summary>
    public int ExactBoss { get; set; }
    /// <summary>With <see cref="ExactBoss"/>: "" = the whole true future; "draw" = only the true draw order (and shuffle
    /// stream); "enemy" = only the true enemy randomness.</summary>
    public string ExactPart { get; set; } = "";
    /// <summary>With ExactPart "worlds": no true future, but at most this many sampled worlds per search in those fights.</summary>
    public int BossWorlds { get; set; } = 16;
    public NetSearchConfig Search { get; set; } = new();
}

/// <summary>W2: chance of winning the second act-3 boss given the HP share and the number of potions held at its start,
/// a logistic fitted on self-play logs by src/s3/fit_boss2.py (JSON: b0, b_hp, b_pot, b_hp2, pot_cap).</summary>
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
    /// <summary>Battle-start dump (starts_dump=FILE, 2026-11-16): one JSON line per live boss fight (elite fights too with
    /// <see cref="StartsDumpElite"/>) in the S0 BattleStartSpec format plus the fight's T_f table, so the fight can be
    /// replayed from its start by the S2 fight harness.</summary>
    /// <summary>Potion belt management (potion_values=FILE, 2026-11-16; null = off): a JSON map potion id -> value (win
    /// points at the final boss, src/exp/potion_values.json). With a full belt and a potion reward waiting, the
    /// held potion of the lowest value is discarded when the offered one is worth more by <see cref="PotionSwapMargin"/>;
    /// the reward is then taken as usual. Unknown potions get the median value.</summary>
    public Dictionary<string, double>? PotionValues;
    /// <summary>Card-reward bonus (card_values=FILE, 2026-11-16; null = off): a JSON map card id -> value (points of run
    /// win chance of one more copy at the first act-3 boss, src/exp/out/card_values.json). In card rewards from act
    /// <see cref="CardBonusAct"/> on (1-based), <see cref="CardBonus"/> x value is added to the card's policy logit
    /// (skip and unknown cards: 0).</summary>
    public Dictionary<string, double>? CardValues;
    public double CardBonus = 0.5;
    /// <summary>Lookahead statistics: sum of |value-logit change| of a card, number of cards scored.</summary>
    public double LookDz;
    public long LookN;
    /// <summary>look_* lookaheads by decision type: candidates scored and the sum of their |value-logit change|.</summary>
    public readonly Dictionary<string, (long N, double SumAbsDz)> LookStats = new();
    /// <summary>look_dump=FILE (row check): per lookahead the state row, every candidate's post-state row, the pick and
    /// the state row at the agent's next callback (src/exp/look_rowcheck.py).</summary>
    public StreamWriter? LookDump;
    public int CardBonusAct = 3;
    public double PotionSwapMargin, PotionValueDefault;
    /// <summary>Shop rule (potion_buy_min=X > 0, needs potion values): an affordable potion worth at least X is bought
    /// before the policy is asked (the most valuable one first), discarding the lowest-valued held potion when the belt
    /// is full and that one is worth less. 0 = off.</summary>
    public double PotionBuyMin;
    public long PotionBuys;
    public long PotionSwaps;
    /// <summary>potion_mix=FILE (AgentConfig.PotionMixShrink; null = off). Counters, index 0 = the reward set's potion,
    /// 1 = extra potion rewards (potion_mix_extra): decisions (full belt, potion about to be taken), swaps, kinds scored,
    /// kinds scored without a Vbar (state term alone), macro rows evaluated; sum of |dV - Vbar| over kinds with a Vbar.</summary>
    public PotionMix? PotionMix;
    public readonly long[] PmDecisions = new long[2], PmSwaps = new long[2], PmKinds = new long[2], PmNoMean = new long[2], PmRows = new long[2];
    public double PmSumAbsDev;
    public long PmDevN;
    public StreamWriter? StartsDump;
    public bool StartsDumpElite;
    public int TfDumpK = 20;
    /// <summary>tf_dump_all=1: the dump covers every room kind of every act (keys monsterA, eliteA, bossA; boss3 as before).</summary>
    public bool TfDumpAll;
    public readonly Dictionary<string, int> TfDumped = new();
    /// <summary>potion_reserve: fights whose table was changed, table entries raised, sum of T_max (after) / T_max (before).</summary>
    public long PotionReserveFights, PotionReserveEntries;
    public double PotionReserveTMaxRatioSum;
    /// <summary>clone_probe: event decisions probed, clones taken, their total and largest time (s), option probes and
    /// their failures; clone_dump=FILE: one JSON line per probed decision.</summary>
    public long CloneProbes, CloneOk, OptionProbes, OptionProbeErrors;
    public double CloneSeconds, CloneMaxSeconds, OptionProbeSeconds;
    public StreamWriter? CloneDump;
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
/// played by the network-guided search (src/s2 NetSearch) on the live combat, scored by a T_f table built at the
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
        /// <summary>stale_trace: one line per decision of this fight (null when tracing is off).</summary>
        public List<string>? Trace;
    }

    /// <summary>stale_trace (diagnosis of combat stalemates, src/ic113/STALEMATE.md): every live fight keeps one
    /// line per decision (state the player sees, the root actions with prior and policy target, the chosen action);
    /// the lines of a fight that hits the stalemate cap are written here, and with <see cref="StaleTraceEnc"/> also the
    /// lines of every finished fight of that encounter. Null = off (the chain's actors).</summary>
    public static StreamWriter? StaleTrace;
    public static string? StaleTraceEnc;

    private void TraceDecision(LiveFight f, CombatState state, NetSearchResult r)
    {
        var sb = new System.Text.StringBuilder();
        Player p = state.Players[0];
        PlayerCombatState pcs = p.PlayerCombatState!;
        sb.Append($"R{state.RoundNumber} d{f.O.Decisions} hp {p.Creature.CurrentHp}/{p.Creature.MaxHp} blk {p.Creature.Block} en {pcs.Energy} ");
        sb.Append("pw [").Append(string.Join(",", p.Creature.Powers.Select(x => $"{x.Id.Entry}:{x.Amount}"))).Append("] ");
        sb.Append($"draw {pcs.DrawPile.Cards.Count} disc {pcs.DiscardPile.Cards.Count} exh {pcs.ExhaustPile.Cards.Count} ");
        sb.Append("hand [").Append(string.Join(",", pcs.Hand.Cards.Select(c => c.Id.Entry + (c.CurrentUpgradeLevel > 0 ? "+" : "") + "$" + c.EnergyCost
            + (c.CanPlay(out _) ? "" : "x")))).Append("] ");
        foreach (Creature e in state.Enemies)
        {
            sb.Append($"| {e.Monster?.Id.Entry}#{e.CombatId} {e.CurrentHp}/{e.MaxHp} b{e.Block}{(e.IsDead ? " DEAD" : "")}");
            if (e.Powers.Any()) sb.Append(" [").Append(string.Join(",", e.Powers.Select(x => $"{x.Id.Entry}:{x.Amount}"))).Append(']');
            if (e.Monster?.NextMove is { } mv)
            {
                sb.Append(' ').Append(mv.Id);
                foreach (var it in mv.Intents)
                    if (it is Sts2Sim.Core.MonsterMoves.Intents.AttackIntent a && !e.IsDead) sb.Append('=').Append(a.GetTotalDamage(state.Allies, e));
            }
            sb.Append(' ');
        }
        static string A(SearchAction a) => a.Kind == ActionKind.EndTurn ? "END" : a.Source.Split('+')[0] + (a.TargetId is { } t ? "@" + t : "");
        float[]? lg = r.RootEval?.Logits;
        // the exact score if the fight ended now (current HP and belt), relative to T_max: the value of winning this move
        sb.Append($"|| win {_s.Tf.Score(state, f.Ctx) / f.O.TMax:F3} sims {r.Simulations} q {r.RootQ:F3} v {r.RootEval?.Value:F3} surv {r.RootEval?.Survive:F3} ");
        if (lg is not null && lg.Length >= r.RootActions.Count)
        {
            double m = lg.Take(r.RootActions.Count).Max(), z = lg.Take(r.RootActions.Count).Sum(x => Math.Exp(x - m));
            sb.Append("roots [").Append(string.Join(" ", r.RootActions.Select((a, i) =>
                $"{A(a)}:{Math.Exp(lg[i] - m) / z:F2}" + (r.PolicyTarget is { } pt && i < pt.Length ? $"/{pt[i]:F2}" : "")))).Append("] ");
        }
        sb.Append("-> ").Append(A(r.Action));
        f.Trace!.Add(sb.ToString());
    }

    private void WriteTrace(LiveFight f, string end)
    {
        if (StaleTrace is null || f.Trace is null) return;
        lock (StaleTrace)
        {
            StaleTrace.WriteLine($"=== {Rec.Seed} asc {_asc} floor {_run.TotalFloor} {f.O.Encounter} start hp {f.O.StartHp}: {end}; T_max {f.O.TMax:F4} death {f.O.DeathValue:F4}; deck " +
                string.Join(",", _p.Deck.Cards.Select(c => c.Id.Entry + (c.CurrentUpgradeLevel > 0 ? "+" : ""))) + "; relics " +
                string.Join(",", _p.Relics.Select(x => x.Id.Entry)));
            foreach (string l in f.Trace) StaleTrace.WriteLine(l);
            StaleTrace.Flush();
        }
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
    private readonly Dictionary<MapPoint, (int ToRest, int ToShop, int Elites, int Fights)> _ahead = new();
    private readonly Dictionary<MapPoint, float[]> _feat2 = new();
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
            Seed = seed, SeedHash = _seed, Ascension = _asc, Perturbed = perturbed, Character = player.Character.Id.ToString(),
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
        if (_lookRec is not null) LookFlush(StateRow());
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
        HealCheck();
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
        LookFlush(null);
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
    /// <param name="look">The bonus is a look_* lookahead of this type (notes look_changed_TYPE instead of bonus_*).</param>
    private async Task<int> DecideOptionsAsync(MacroRow row, int dtype, bool[]? greedyBan = null, double[]? logitBonus = null, string? look = null)
    {
        row.I[ML.ODtype] = dtype;
        int n = row.NOptions;
        if (n <= 1) return 0;
        MacroEval ev = await _s.Macro.EvaluateAsync(row);
        double[] p;
        if (logitBonus is null) p = Softmax(ev.OptLogits[..n]);
        else
        {
            var lg = new float[n];
            for (int i = 0; i < n; i++) lg[i] = ev.OptLogits[i] + (float)logitBonus[i];
            p = Softmax(lg);
        }
        int pick;
        if (logitBonus is not null && _cfg.Greedy)
        {
            int a0 = 0, a1 = 0;
            for (int i = 1; i < n; i++) { if (ev.OptLogits[i] > ev.OptLogits[a0]) a0 = i; if (p[i] > p[a1]) a1 = i; }
            if (look is null) _s.Note(a0 == a1 ? "bonus_same_pick" : "bonus_changed_pick");
            else LookChanged(look, a0 != a1);
        }
        if (greedyBan is not null)
        {
            // greedy_event: the most probable option that the repeat guard has not banned; all banned = sample
            pick = -1;
            for (int i = 0; i < n; i++) if (!greedyBan[i] && (pick < 0 || p[i] > p[pick])) pick = i;
            if (pick < 0) pick = Pick(p, false);
        }
        else pick = Pick(p, _cfg.Greedy && (dtype != ML.DMap || _cfg.GreedyMap) && dtype != ML.DEvent);
        if (look is not null) LookPicked(pick);
        Record(row, dtype, pick, p[pick], ev, 0);
        return pick;
    }

    /// <summary>Samples one deck slot among <paramref name="slots"/> (all &lt; <see cref="ML.MaxDeck"/>).</summary>
    /// <param name="bonus">Logit bonus per entry of <paramref name="slots"/> (a look_* lookahead of type <paramref name="look"/>).</param>
    private async Task<int> DecideSlotAsync(MacroRow row, int dtype, IReadOnlyList<int> slots, double[]? bonus = null, string? look = null)
    {
        if (slots.Count == 1) return slots[0];
        row.ClearDecision();
        row.I[ML.ODtype] = dtype;
        MacroEval ev = await _s.Macro.EvaluateAsync(row);
        var lg = new float[slots.Count];
        ulong mask = 0;
        for (int i = 0; i < slots.Count; i++) { lg[i] = ev.SlotLogits[slots[i]]; mask |= 1UL << slots[i]; }
        int a0 = ArgMax(lg);
        if (bonus is not null) for (int i = 0; i < slots.Count; i++) lg[i] += (float)bonus[i];
        double[] p = Softmax(lg);
        int pick = Pick(p, _cfg.Greedy);
        if (bonus is not null && look is not null)
        {
            if (_cfg.Greedy) LookChanged(look, ArgMax(lg) != a0);
            LookPicked(pick);
        }
        Record(row, dtype, slots[pick], p[pick], ev, mask);
        return slots[pick];
    }

    private static int ArgMax(float[] x)
    {
        int b = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[b]) b = i;
        return b;
    }

    private MacroRow StateRow() => MacroEncoder.Encode(_run, _p, _facts, _s.MVocab);

    // ---- one-step lookahead of the other deterministic decisions (look_remove / smith / rest / shop / relic; see
    // AgentConfig.LookRemove). Card rewards keep their own code (look_beta, ChooseRewardActionAsync).

    private bool LookOn(double beta) => beta != 0 && _run.CurrentActIndex + 1 >= _cfg.LookAct;

    /// <summary>Pending look_dump record: written at the agent's next callback with the state row then (the outcome).</summary>
    private Dictionary<string, object?>? _lookRec;

    /// <summary>Value-logit change z(post) - z(base) of each candidate post-state (null = no modelled change: 0); the base
    /// row and every post-state are evaluated in one batch (<paramref name="base0"/>: the base already evaluated).</summary>
    private async Task<(double[] Dz, MacroEval Base)> LookDeltasAsync(MacroRow b0, IReadOnlyList<MacroRow?> posts, MacroEval? base0 = null)
    {
        var rows = new List<MacroRow>();
        if (base0 is null) rows.Add(b0);
        foreach (MacroRow? r in posts) if (r is not null) rows.Add(r);
        MacroEval[] evs = await Task.WhenAll(rows.Select(r => _s.Macro.EvaluateAsync(r)));
        MacroEval b = base0 ?? evs[0];
        var dz = new double[posts.Count];
        for (int i = 0, k = base0 is null ? 1 : 0; i < posts.Count; i++) if (posts[i] is not null) dz[i] = evs[k++].Out[ML.OutV] - b.Out[ML.OutV];
        return (dz, b);
    }

    /// <summary>beta x dz as a logit bonus; statistics, notes (lookahead_TYPE) and the look_dump record of this decision.</summary>
    private double[] LookBonus(string type, double beta, double[] dz, MacroRow b0, IReadOnlyList<MacroRow?> posts, IReadOnlyList<string> info,
        Dictionary<string, object?>? extra = null)
    {
        _s.Note("lookahead_" + type);
        Rec.Look[type] = Rec.Look.GetValueOrDefault(type) + 1;
        var (n, sum) = _s.LookStats.GetValueOrDefault(type);
        for (int i = 0; i < dz.Length; i++) if (posts[i] is not null || dz[i] != 0) { n++; sum += Math.Abs(dz[i]); }
        _s.LookStats[type] = (n, sum);
        if (_s.LookDump is not null)
        {
            LookFlush(null);
            static object? R(MacroRow? r) => r is null ? null : new Dictionary<string, object> { ["i"] = r.I, ["f"] = r.F };
            _lookRec = new Dictionary<string, object?>
            {
                ["seed"] = Rec.Seed, ["floor"] = _run.TotalFloor, ["act"] = _run.CurrentActIndex, ["type"] = type,
                ["hp"] = _p.Creature.CurrentHp, ["max_hp"] = _p.Creature.MaxHp, ["gold"] = _p.Gold, ["deck"] = _p.Deck.Cards.Count,
                ["relics"] = _p.Relics.Count, ["info"] = info, ["dz"] = dz, ["base"] = R(b0), ["posts"] = posts.Select(R).ToList(),
            };
            if (extra is not null) foreach (var kv in extra) _lookRec[kv.Key] = kv.Value;
        }
        return dz.Select(d => beta * d).ToArray();
    }

    private void LookChanged(string type, bool changed)
    {
        _s.Note((changed ? "look_changed_" : "look_same_") + type);
        if (changed) Rec.Look[type + "_changed"] = Rec.Look.GetValueOrDefault(type + "_changed") + 1;
    }

    private void LookPicked(int pick)
    {
        if (_lookRec is not null) _lookRec["chosen"] = pick;
    }

    /// <summary>Writes the pending look_dump record, with the state row <paramref name="after"/> (null: the run ended).</summary>
    private void LookFlush(MacroRow? after)
    {
        if (_lookRec is null || _s.LookDump is null) return;
        if (after is not null)
        {
            _lookRec["after"] = new Dictionary<string, object> { ["i"] = after.I, ["f"] = after.F };
            _lookRec["after_hp"] = _p.Creature.CurrentHp; _lookRec["after_floor"] = _run.TotalFloor; _lookRec["after_gold"] = _p.Gold;
        }
        lock (_s.LookDump) _s.LookDump.WriteLine(System.Text.Json.JsonSerializer.Serialize(_lookRec, DumpJson));
        _lookRec = null;
    }

    private static readonly System.Text.Json.JsonSerializerOptions DumpJson =
        new() { NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };

    /// <summary>Value logit at HP <paramref name="hp"/> from a served row (milestone blend when the file has the heads).</summary>
    private double ZAt(float[] o, int hp, int maxHp)
    {
        if (!ML.HasMilestones(o)) return ML.LogitAt(o, hp, maxHp);
        double v = Math.Clamp(ML.MilestoneValue(o, hp, maxHp, ML.Stage(_run.CurrentActIndex, false), _asc >= ML.DoubleBossAsc), 1e-9, 1 - 1e-9);
        return Math.Log(v / (1 - v));
    }

    /// <summary>A relic whose pickup does nothing but add it (no AfterObtained override, no upon-pickup flag).</summary>
    private static bool PlainRelic(RelicModel m) =>
        !m.HasUponPickupEffect && m.GetType().GetMethod(nameof(RelicModel.AfterObtained), Type.EmptyTypes)?.DeclaringType == typeof(RelicModel);

    private int DeckIndex(CardModel c)
    {
        var deck = _p.Deck.Cards;
        for (int i = 0; i < deck.Count; i++) if (ReferenceEquals(deck[i], c)) return i;
        return -1;
    }

    /// <summary>Chooses one of <paramref name="cards"/> (deck cards) with the smith or removal head.</summary>
    private async Task<CardModel> DecideDeckCardAsync(IReadOnlyList<CardModel> cards, int dtype, bool removal = false)
    {
        if (removal && _cfg.RemoveStrike && !cards.Any(c => c.Type is CardType.Curse or CardType.Status))
        {
            string strikeId = "CARD.STRIKE_" + _p.Character.Id.Entry;   // the character's basic Strike (CARD.STRIKE_IRONCLAD, ...)
            CardModel? strike = cards.Where(c => c.Id.ToString() == strikeId).OrderBy(c => c.CurrentUpgradeLevel).FirstOrDefault();
            if (strike is not null) { _s.Note("remove_strike_rule"); _s.Note("remove_target:" + strike.Id); return strike; }
        }
        var bySlot = new Dictionary<int, CardModel>();
        foreach (CardModel c in cards)
        {
            int i = DeckIndex(c);
            if (i >= 0 && i < ML.MaxDeck) bySlot[i] = c;
        }
        if (bySlot.Count == 0) return cards[_rng.Next(cards.Count)];
        var slots = bySlot.Keys.OrderBy(x => x).ToList();
        // look_remove / look_smith: the state with that slot removed / upgraded (a multi-card prompt: the earlier picks
        // are not applied yet, in the base row as in every post-state)
        string? look = removal ? "remove" : dtype == ML.DSmith ? "smith" : null;
        double beta = look == "remove" ? _cfg.LookRemove : look == "smith" ? _cfg.LookSmith : 0;
        double[]? bonus = null;
        if (slots.Count > 1 && LookOn(beta))
        {
            MacroRow b0 = StateRow();
            var posts = slots.Select(j => (MacroRow?)(removal ? MacroEncoder.WithoutCard(b0, _p, j, _s.MVocab, _run.TotalFloor) : MacroEncoder.WithUpgrade(b0, _p, j))).ToList();
            var (dz, _) = await LookDeltasAsync(b0, posts);
            bonus = LookBonus(look!, beta, dz, b0, posts, slots.Select(j => $"{j}:{bySlot[j].Id}:{bySlot[j].CurrentUpgradeLevel}").ToList());
        }
        int slot = await DecideSlotAsync(StateRow(), dtype, slots, bonus, bonus is null ? null : look);
        if (removal) _s.Note("remove_target:" + bySlot[slot].Id);
        return bySlot[slot];
    }

    // ---- map

    private static readonly MapPointType[] MinTypes = { MapPointType.Monster, MapPointType.Elite, MapPointType.RestSite, MapPointType.Shop, MapPointType.Treasure, MapPointType.Unknown };
    private static readonly MapPointType[] MaxTypes = { MapPointType.Monster, MapPointType.Elite, MapPointType.RestSite, MapPointType.Shop, MapPointType.Unknown };

    /// <summary>Per room type, the fewest and the most rooms of that type on any path from <paramref name="pt"/>
    /// (inclusive) to the end of the act.</summary>
    private (int[] Min, int[] Max) PathCounts(MapPoint pt)
    {
        SyncMapMemo();
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

    /// <summary>The memoised map features belong to one act map: dropped when the run has another.</summary>
    private void SyncMapMemo()
    {
        if (ReferenceEquals(_featMap, _run.Map)) return;
        _feat.Clear(); _ahead.Clear(); _feat2.Clear();
        _featMap = _run.Map;
    }

    private const int NoShop = 1000;

    /// <summary>
    /// What lies ahead of <paramref name="pt"/> on the paths toward the boss (children = the next row), by map point
    /// type (an Unknown point counts as nothing):
    /// ToRest = fewest rows to a rest site (0 on one; the first boss counts as the end, so without a rest site this is the
    /// rows to the boss); ToShop = fewest rows to a shop (<see cref="NoShop"/>: none before the boss); Elites / Fights =
    /// fewest elite / monster + elite points on a path from this point (inclusive) to that path's first rest site (or
    /// the boss; the boss fight itself is not counted). Each of the four is minimised on its own.
    /// </summary>
    private (int ToRest, int ToShop, int Elites, int Fights) Ahead(MapPoint pt)
    {
        if (_ahead.TryGetValue(pt, out var got)) return got;
        MapPointType t = pt.PointType;
        int rest = 0, shop = t == MapPointType.Shop ? 0 : NoShop, elites = 0, fights = 0;
        if (t != MapPointType.Boss)   // the paths end at the first boss (a second boss is its child)
        {
            int cRest = int.MaxValue, cShop = NoShop, cElites = int.MaxValue, cFights = int.MaxValue;
            foreach (MapPoint ch in pt.Children)
            {
                var c = Ahead(ch);
                cRest = Math.Min(cRest, c.ToRest); cShop = Math.Min(cShop, c.ToShop);
                cElites = Math.Min(cElites, c.Elites); cFights = Math.Min(cFights, c.Fights);
            }
            bool leaf = pt.Children.Count == 0;
            if (t != MapPointType.RestSite)
            {
                rest = leaf ? 0 : 1 + cRest;
                elites = (t == MapPointType.Elite ? 1 : 0) + (leaf ? 0 : cElites);
                fights = (t is MapPointType.Elite or MapPointType.Monster ? 1 : 0) + (leaf ? 0 : cFights);
            }
            if (shop != 0 && cShop < NoShop) shop = 1 + cShop;
        }
        return _ahead[pt] = (rest, shop, elites, fights);
    }

    /// <summary>Most elite points on a path of <paramref name="rows"/> rows that starts at <paramref name="pt"/> (inclusive).</summary>
    private static int EliteMax(MapPoint pt, int rows)
    {
        int best = 0;
        if (rows > 1) foreach (MapPoint ch in pt.Children) best = Math.Max(best, EliteMax(ch, rows - 1));
        return best + (pt.PointType == MapPointType.Elite ? 1 : 0);
    }

    /// <summary>
    /// Layout v2 features of a map option (ML.MF2 = 8), which carry the order of rooms that the path counts of
    /// <see cref="MapFeatures"/> do not: 0 rows to the nearest rest site / 8 (0 on one; rows to the boss if none);
    /// 1 rows to the nearest shop / 8, at most 1 (1 = none before the boss, or 8 rows and more); 2 fewest elites up to
    /// the first rest site (or the boss) / 3; 3 fewest fights up to there / 5; 4 most elites within three rows from
    /// this point (inclusive) / 3; 5 an elite with a rest site among its children; 6 a rest site with an elite among
    /// its children; 7 rows to the act's (first) boss / 16.
    /// </summary>
    private float[] MapFeatures2(MapPoint pt)
    {
        SyncMapMemo();
        if (_feat2.TryGetValue(pt, out float[]? got)) return got;
        var a = Ahead(pt);
        var f = new float[ML.MF2];
        f[0] = a.ToRest / 8f;
        f[1] = a.ToShop >= NoShop ? 1f : Math.Min(1f, a.ToShop / 8f);
        f[2] = a.Elites / 3f;
        f[3] = a.Fights / 5f;
        f[4] = EliteMax(pt, 3) / 3f;
        f[5] = pt.PointType == MapPointType.Elite && pt.Children.Any(c => c.PointType == MapPointType.RestSite) ? 1f : 0f;
        f[6] = pt.PointType == MapPointType.RestSite && pt.Children.Any(c => c.PointType == MapPointType.Elite) ? 1f : 0f;
        f[7] = Math.Max(0, _run.Map.BossMapPoint.coord.row - pt.coord.row) / 16f;
        return _feat2[pt] = f;
    }

    public async Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options)
    {
        Pre();
        SnapshotFloor();   // the Ancient floor of an act has no OnRoomResolved
        if (options.Count == 1) return options[0];
        var sorted = options.OrderBy(o => o.coord.col).ThenBy(o => o.coord.row).Take(ML.K).ToList();
        MacroRow row = StateRow();
        foreach (MapPoint pt in sorted) row.AddOption(ML.KMap, (int)pt.PointType, 0, MapFeatures(pt), MapFeatures2(pt));
        MacroEncoder.FillMap(row, _run, sorted);   // layout v3: the option-reach bits of the map nodes follow these slots
        // map_rule: "emaxN..." applies elite_max only in the acts listed (1-based digits, e.g. emax23), the head elsewhere
        string rule = _cfg.MapRule;
        if (rule.StartsWith("emax", StringComparison.Ordinal) && rule.Length > 4)
            rule = rule[4..].Contains((char)('1' + _run.CurrentActIndex)) ? "elite_max" : "";
        if (rule != "")
        {
            if (rule == "uniform") return sorted[_rng.Next(sorted.Count)];
            bool most = rule == "elite_max";
            if (rule.StartsWith("hp", StringComparison.Ordinal))   // hp0.7 (no colon: the evaluation script splits arms at colons)
                most = _p.Creature.CurrentHp >= double.Parse(rule[2..].TrimStart(':', '@'), System.Globalization.CultureInfo.InvariantCulture) * _p.Creature.MaxHp;
            // index 1 of both count arrays is the elite count (MinTypes / MaxTypes)
            var key = sorted.Select(pt => most ? -PathCounts(pt).Max[1] : PathCounts(pt).Min[1]).ToList();
            int best = key.Min();
            return sorted[await DecideOptionsAsync(row, ML.DMap, key.Select(k => k != best).ToArray())];
        }
        return sorted[await DecideOptionsAsync(row, ML.DMap)];
    }

    // ---- rewards

    public async Task<RewardDecision> ChooseRewardActionAsync(RewardsSet rewards)
    {
        Pre();
        if (_s.PotionMix is not null) await PotionMixRewardAsync(rewards);   // ic113: replaces the table belt rule below
        else if (_cfg.PotionSwap && _s.PotionValues is { } pv && rewards.Potion is { IsResolved: false, Potion: { } offered } && !_p.PotionSlots.Contains(null) &&
            _p.CanUseOrRemovePotions)
        {
            double V(PotionModel m) => pv.GetValueOrDefault(m.Id.ToString(), _s.PotionValueDefault);
            PotionModel? worst = null;
            foreach (PotionModel? m in _p.PotionSlots) if (m is not null && (worst is null || V(m) < V(worst))) worst = m;
            if (worst is not null && V(offered) > V(worst) + _s.PotionSwapMargin)
            {
                await PotionCmd.Discard(worst);
                _s.PotionSwaps++; _s.Note("potion_swap");
            }
        }
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
            double[]? bonus = null;
            int act1 = _run.CurrentActIndex + 1;
            double[] Bonus() => bonus ??= new double[cardOpts.Count];
            if (_s.CardValues is { } cv && act1 >= _s.CardBonusAct && _s.CardBonus != 0)
            {
                for (int i = 0; i < cardOpts.Count; i++)
                    if (cardOpts[i].Card is { } c0) Bonus()[i] += _s.CardBonus * cv.GetValueOrDefault(MacroEncoder.CardId(c0), 0.0);
                _s.Note("card_bonus_decision");
            }
            if (_s.CardValues is { } cv2 && !double.IsNaN(_cfg.BadCardCut) && act1 >= _cfg.BadCardAct)
                for (int i = 0; i < cardOpts.Count; i++)
                    if (cardOpts[i].Card is { } c1 && cv2.TryGetValue(MacroEncoder.CardId(c1), out double val) && val < _cfg.BadCardCut)
                    { Bonus()[i] -= 20; _s.Note("bad_card_penalised"); }
            if (_cfg.SkipBias != 0 && act1 >= _cfg.SkipAct && _p.Deck.Cards.Count >= _cfg.SkipDeck)
                for (int i = 0; i < cardOpts.Count; i++)
                    if (cardOpts[i].Card is null) { Bonus()[i] += _cfg.SkipBias; _s.Note("skip_bias_decision"); }
            if (_cfg.LookBeta != 0 && act1 >= _cfg.LookAct && _p.Deck.Cards.Count < ML.MaxDeck)
            {
                // one-step lookahead: the state row with each offered card added to the deck, against the row without it
                MacroRow b0 = StateRow();
                int j = _p.Deck.Cards.Count;
                var rows = new List<MacroRow> { b0 };
                foreach (var (_, card) in cardOpts)
                {
                    if (card is null) { rows.Add(b0); continue; }
                    MacroRow r = b0.Clone();
                    r.I[ML.ODIds + j] = _s.MVocab.Tok("card", MacroEncoder.CardId(card));
                    r.I[ML.ODUpg + j] = Math.Clamp(card.CurrentUpgradeLevel, 0, 3);
                    r.I[ML.ODEnch + j] = 0; r.I[ML.ODEamt + j] = 0; r.I[ML.ODAge + j] = 0;
                    r.F[ML.SDeck] = (j + 1) / 40f;
                    if (card.CurrentUpgradeLevel > 0) r.F[ML.SUpg] += 1 / 20f;
                    rows.Add(r);
                }
                MacroEval[] evs = await Task.WhenAll(rows.Select(r => _s.Macro.EvaluateAsync(r)));
                double z0 = evs[0].Out[ML.OutV], s0 = evs[0].ValueStd;
                for (int i = 0; i < cardOpts.Count; i++)
                {
                    if (cardOpts[i].Card is null) continue;
                    double dz = evs[i + 1].Out[ML.OutV] - z0, ds = Math.Max(0, evs[i + 1].ValueStd - s0);
                    Bonus()[i] += _cfg.LookBeta * dz - _cfg.LookKappa * ds;
                    _s.LookDz += Math.Abs(dz); _s.LookN++;
                }
                _s.Note("lookahead_decision");
            }
            int pick = await DecideOptionsAsync(row, ML.DCard, null, bonus);
            if (cardOpts[pick].Card is null) Rec.CardsSkipped++; else Rec.CardsTaken++;
            return cardOpts[pick].D;
        }
        // optional merchant potion / card-removal reward: take when possible (a removal then asks for the card)
        _s.Note("reward_choice:" + string.Join("|", cands.Select(c => c is RewardDecision.ResolveExtra x ? x.Reward.GetType().Name + (x.Skip ? "-skip" : "") : c.GetType().Name)));
        return cands[0];
    }

    // ---- ic113 mixed potion valuation (potion_mix=FILE; AgentConfig.PotionMixShrink, PotionMix.cs)

    /// <summary>The belt rule by V_mix at a reward set (the step that takes the reward's potion, after its gold; or, with
    /// potion_mix_extra, the step that resolves an extra potion reward), with a full belt. Decides at most once per potion:
    /// a drop frees a slot and the reward step then takes the offered potion.</summary>
    private async Task PotionMixRewardAsync(RewardsSet rewards)
    {
        if (_p.PotionSlots.Count == 0 || _p.PotionSlots.Contains(null) || !_p.CanUseOrRemovePotions) return;
        var next = RewardDecisionClassifier.Classify(rewards);
        PotionModel? offered = null;
        int source = 0;
        if (next is RewardDecisionClassification.Automatic { Decision: RewardDecision.TakePotion } && rewards.Potion is { IsResolved: false, Potion: { } o0 })
            offered = o0;
        else if (_cfg.PotionMixExtra && next is RewardDecisionClassification.Automatic
                 { Decision: RewardDecision.ResolveExtra { Reward: PotionReward { IsOptionalMerchantChoice: false, IsResolved: false, Potion: { } o1 } } })
        { offered = o1; source = 1; }
        if (offered is null) return;
        if (!PotionCmd.CanProcure(offered, _p)) { _s.Note("potion_mix_not_procurable"); return; }
        await PotionMixDecisionAsync(offered, source);
    }

    /// <summary>One belt decision: dV of every held kind and of the offered one from the state row (belt = that single
    /// potion vs an empty belt, one batch), scored by <see cref="PotionMix.Score"/>; the first lowest-scored held potion is
    /// discarded when the offered potion scores more than it by potion_mix_margin. Run record: look potion_mix(_extra),
    /// potion_mix(_extra)_swap, potion_mix_nomean (kinds scored by the state term alone); look_dump type "potion_mix".</summary>
    private async Task PotionMixDecisionAsync(PotionModel offered, int source)
    {
        PotionMix pm = _s.PotionMix!;
        string key = source == 0 ? "potion_mix" : "potion_mix_extra";
        var held = _p.PotionSlots.Select(m => m!).ToList();
        string oid = offered.Id.ToString();
        var kinds = held.Select(h => h.Id.ToString()).Append(oid).Distinct().ToList();
        MacroRow b0 = StateRow();
        var rows = new List<MacroRow>();
        MacroRow empty = b0.Clone(); empty.SetPotions(Array.Empty<int>()); rows.Add(empty);
        foreach (string k in kinds)
        {
            MacroRow r = b0.Clone();
            r.SetPotions(new[] { _s.MVocab.Tok("potion", k) });
            rows.Add(r);
        }
        MacroEval[] evs = await Task.WhenAll(rows.Select(r => _s.Macro.EvaluateAsync(r)));
        int act = _run.CurrentActIndex + 1;
        var dv = new Dictionary<string, double>();
        var score = new Dictionary<string, double>();
        var noMean = new List<string>();
        for (int i = 0; i < kinds.Count; i++)
        {
            dv[kinds[i]] = evs[i + 1].Value - evs[0].Value;
            score[kinds[i]] = pm.Score(act, kinds[i], dv[kinds[i]], _cfg.PotionMixShrink, out bool hasMean);
            if (!hasMean) noMean.Add(kinds[i]);
            else if (pm.TryMean(Math.Clamp(act, 1, 3), kinds[i], out double vb)) { _s.PmSumAbsDev += Math.Abs(dv[kinds[i]] - vb); _s.PmDevN++; }
        }
        int w = 0;
        for (int j = 1; j < held.Count; j++) if (score[held[j].Id.ToString()] < score[held[w].Id.ToString()]) w = j;
        int choice = score[oid] > score[held[w].Id.ToString()] + _cfg.PotionMixMargin ? w : -1;
        _s.PmDecisions[source]++; _s.PmKinds[source] += kinds.Count; _s.PmNoMean[source] += noMean.Count; _s.PmRows[source] += rows.Count;
        _s.Note("lookahead_" + key);
        Rec.Look[key] = Rec.Look.GetValueOrDefault(key) + 1;
        if (noMean.Count > 0) Rec.Look["potion_mix_nomean"] = Rec.Look.GetValueOrDefault("potion_mix_nomean") + noMean.Count;
        if (_s.LookDump is not null)
        {
            var d = new Dictionary<string, object?>
            {
                ["seed"] = Rec.Seed, ["floor"] = _run.TotalFloor, ["act"] = _run.CurrentActIndex, ["type"] = "potion_mix", ["source"] = source == 0 ? "reward" : "extra",
                ["asc"] = _asc, ["phase"] = pm.Phase, ["shrink"] = _cfg.PotionMixShrink, ["margin"] = _cfg.PotionMixMargin,
                ["offered"] = oid, ["belt"] = held.Select(h => h.Id.ToString()).ToList(), ["kinds"] = kinds,
                ["v_empty"] = evs[0].Value, ["dv"] = kinds.Select(k => dv[k]).ToList(), ["score"] = kinds.Select(k => score[k]).ToList(),
                ["vbar"] = kinds.Select(k => pm.TryMean(Math.Clamp(act, 1, 3), k, out double vb) ? (double?)vb : null).ToList(),
                ["choice"] = choice, ["base"] = new Dictionary<string, object> { ["i"] = b0.I, ["f"] = b0.F },
            };
            lock (_s.LookDump) _s.LookDump.WriteLine(System.Text.Json.JsonSerializer.Serialize(d, DumpJson));
        }
        if (choice >= 0)
        {
            await PotionCmd.Discard(held[choice]);
            _s.PmSwaps[source]++; _s.Note(key + "_swap");
            Rec.Look[key + "_swap"] = Rec.Look.GetValueOrDefault(key + "_swap") + 1;
        }
    }

    // ---- shop

    public async Task<ShopDecision> ChooseShopActionAsync(MerchantInventory inventory, Player player)
    {
        Pre();
        if (_shopFloor != _run.TotalFloor) { _shopFloor = _run.TotalFloor; _shopSteps = 0; Rec.Shops++; }
        if (++_shopSteps > 12) return new ShopDecision.Leave();
        if (_s.PotionValues is { } pv && _s.PotionBuyMin > 0 && player.CanUseOrRemovePotions)
        {
            double V(PotionModel m) => pv.GetValueOrDefault(m.Id.ToString(), _s.PotionValueDefault);
            MerchantPotionEntry? best = null;
            foreach (MerchantPotionEntry e in inventory.Potions)
                if (!e.Purchased && e.Price <= player.Gold && V(e.Potion) >= _s.PotionBuyMin && (best is null || V(e.Potion) > V(best.Potion))) best = e;
            if (best is not null)
            {
                bool free = player.PotionSlots.Contains(null);
                if (!free)
                {
                    PotionModel? worst = null;
                    foreach (PotionModel? m in player.PotionSlots) if (m is not null && (worst is null || V(m) < V(worst))) worst = m;
                    if (worst is not null && V(best.Potion) > V(worst) + _s.PotionSwapMargin) { await PotionCmd.Discard(worst); free = true; }
                }
                if (free) { _s.PotionBuys++; _s.Note("potion_buy_rule"); return new ShopDecision.BuyPotion(best); }
            }
        }
        MacroRow row = StateRow();
        var opts = new List<ShopDecision>();
        int gold = player.Gold;
        // layout v3: shop options carry their price in the extra option features (price / 300, price / gold)
        float[] Price(int price) => new[] { price / 300f, price / (float)Math.Max(1, gold) };
        foreach (MerchantCardEntry e in inventory.Cards)
            if (!e.Purchased && e.Price <= gold && row.NOptions < ML.K - 2)
            { row.AddOption(ML.KCard, _s.MVocab.Tok("card", MacroEncoder.CardId(e.Card)), e.Card.CurrentUpgradeLevel, default, Price(e.Price)); opts.Add(new ShopDecision.BuyCard(e)); }
        foreach (MerchantRelicEntry e in inventory.Relics)
            if (!e.Purchased && e.Price <= gold && row.NOptions < ML.K - 2)
            { row.AddOption(ML.KRelic, _s.MVocab.Tok("relic", e.Relic.Id.ToString()), 0, default, Price(e.Price)); opts.Add(new ShopDecision.BuyRelic(e)); }
        bool freeSlot = player.PotionSlots.Contains(null);
        foreach (MerchantPotionEntry e in inventory.Potions)
            if (!e.Purchased && e.Price <= gold && freeSlot && row.NOptions < ML.K - 2)
            { row.AddOption(ML.KPotion, _s.MVocab.Tok("potion", e.Potion.Id.ToString()), 0, default, Price(e.Price)); opts.Add(new ShopDecision.BuyPotion(e)); }
        var removable = player.Deck.Cards.Where(c => c.IsRemovable).ToList();
        int removeAt = -1;
        if (!inventory.CardRemoval.Purchased && inventory.CardRemoval.Price <= gold && removable.Count > 0)
        { removeAt = row.AddOption(ML.KSpecial, 2, 0, default, Price(inventory.CardRemoval.Price)); opts.Add(new ShopDecision.Leave()); }   // REMOVE (card chosen below)
        if (opts.Count == 0) return new ShopDecision.Leave();
        row.AddOption(ML.KZero, 0); opts.Add(new ShopDecision.Leave());
        double[]? bonus = null;
        if (LookOn(_cfg.LookShop))
        {
            // look_shop: each purchase's post-state, gold reduced by the price; removal service and LEAVE: 0
            MacroRow b0 = StateRow();
            int g = _run.TotalFloor;
            var posts = opts.Select(o => o switch
            {
                ShopDecision.BuyCard c => MacroEncoder.WithGold(MacroEncoder.WithCard(b0, _p, c.Entry.Card, _s.MVocab, g), gold - c.Entry.Price),
                ShopDecision.BuyRelic r => MacroEncoder.WithGold(MacroEncoder.WithRelic(b0, _p, _s.MVocab.Tok("relic", r.Entry.Relic.Id.ToString())), gold - r.Entry.Price),
                ShopDecision.BuyPotion q => MacroEncoder.WithGold(MacroEncoder.WithPotion(b0, _p, q.Entry.Potion, _s.MVocab), gold - q.Entry.Price),
                _ => null,
            }).ToList();
            var info = opts.Select((o, i) => o switch
            {
                ShopDecision.BuyCard c => $"card:{c.Entry.Card.Id}:{c.Entry.Card.CurrentUpgradeLevel}:{c.Entry.Price}",
                ShopDecision.BuyRelic r => $"relic:{r.Entry.Relic.Id}:{r.Entry.Price}",
                ShopDecision.BuyPotion q => $"potion:{q.Entry.Potion.Id}:{q.Entry.Price}",
                _ => i == removeAt ? $"remove:{inventory.CardRemoval.Price}" : "leave",
            }).ToList();
            var (dz, _) = await LookDeltasAsync(b0, posts);
            bonus = LookBonus("shop", _cfg.LookShop, dz, b0, posts, info);
        }
        int pick = await DecideOptionsAsync(row, ML.DShop, null, bonus, bonus is null ? null : "shop");
        if (pick == removeAt)
            return new ShopDecision.BuyCardRemoval(await DecideDeckCardAsync(removable, ML.DRemove, removal: true));
        return opts[pick];
    }

    // ---- rest site

    public async Task<RestSiteDecision> ChooseRestSiteActionAsync(Player player, IReadOnlyList<RestSiteDecision> candidates)
    {
        Pre();
        var enabled = candidates.Where(c => c.IsEnabled).ToList();
        if (enabled.Count == 0) { _s.Note("rest_no_enabled_option"); return RestSiteDecisionPolicy.ChooseDefault(candidates); }
        foreach (string oid in enabled.Select(c => c.OptionId).Distinct()) _s.Note("rest_offered:" + oid);
        if (_cfg.KindleCap && enabled.Any(c => c.OptionId == "kindle") && enabled.Any(c => c.OptionId != "kindle"))
        {
            int charges = _p.Relics.OfType<Sts2Sim.Core.Models.Relics.PumpkinCandle>().Sum(c => c.KindleCount);
            int left = FightsLeftUpper();
            if (charges >= left)
            {
                enabled = enabled.Where(c => c.OptionId != "kindle").ToList();
                _s.Note("kindle_cap");
                LastKindleCap = $"kindle not offered: {charges} charges >= {left} fights left at most";
            }
            else LastKindleCap = $"kindle kept: {charges} charges < {left} fights left at most";
        }
        else LastKindleCap = null;
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
            double[]? bonus = LookOn(_cfg.LookRest) ? await RestLookAsync(groups.Take(ML.K).ToList()) : null;
            pick = await DecideOptionsAsync(row, ML.DRest, null, bonus, bonus is null ? null : "rest");
        }
        var chosen = groups[pick];
        _s.Note("rest_chosen:" + chosen.Id);
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

    /// <summary>kindle_cap's last verdict at a rest site with Kindle on offer (null otherwise), for the live controller's log.</summary>
    public string? LastKindleCap { get; private set; }

    /// <summary>Upper bound of the fights left in the run from the current map point (kindle_cap): per room kind, the most
    /// Monster, Elite and Unknown rooms on any path to the act's boss (each maximised on its own), the boss(es) of this
    /// act, and <see cref="AgentConfig.KindleActFights"/> per later act. Without a current point: a whole act.</summary>
    private int FightsLeftUpper()
    {
        int later = Math.Max(0, _run.Acts.Count - 1 - _run.CurrentActIndex) * _cfg.KindleActFights;
        MapPoint? cur = _run.CurrentMapPoint;
        if (cur is null) return later + _cfg.KindleActFights;
        var (_, mx) = PathCounts(cur);
        int rooms = 0;
        for (int i = 0; i < MaxTypes.Length; i++)
            if (MaxTypes[i] is MapPointType.Monster or MapPointType.Elite or MapPointType.Unknown) rooms += mx[i];
        int bosses = 1 + (_run.Map.SecondBossMapPoint is null ? 0 : 1);
        return rooms + bosses + later;
    }

    /// <summary>look_rest: heal = the state row's value at HP + the heal the game will give (RestSiteDecision.Heal: 30% of
    /// max HP through Hook.ModifyRestSiteHealAmount, the integer part of HP + amount, at most max HP), read off the HP
    /// ramps of the state row's output; smith = the state row with the smith head's argmax target upgraded (the slot
    /// choice DecideDeckCardAsync would make without look_smith); other options 0.</summary>
    private async Task<double[]> RestLookAsync(IReadOnlyList<(string Id, List<RestSiteDecision> Items)> groups)
    {
        MacroRow b0 = StateRow();
        int hp = b0.Hp, maxHp = b0.MaxHp;
        var info = groups.Select(g => g.Id).ToList();
        var posts = new MacroRow?[groups.Count];
        int si = info.IndexOf("smith"), target = -1;
        MacroEval base0;
        if (si >= 0)
        {
            var slots = groups[si].Items.OfType<RestSiteDecision.Smith>().Select(s => DeckIndex(s.Card)).Where(j => j >= 0 && j < ML.MaxDeck).Distinct().OrderBy(j => j).ToList();
            MacroRow sr = b0.Clone();
            sr.ClearDecision(); sr.I[ML.ODtype] = ML.DSmith;   // the smith head's row, as DecideSlotAsync builds it
            MacroEval[] evs = await Task.WhenAll(_s.Macro.EvaluateAsync(b0), _s.Macro.EvaluateAsync(sr));
            base0 = evs[0];
            foreach (int j in slots) if (target < 0 || evs[1].SlotLogits[j] > evs[1].SlotLogits[target]) target = j;
            if (target >= 0) { posts[si] = MacroEncoder.WithUpgrade(b0, _p, target); info[si] = $"smith:{target}:{_p.Deck.Cards[target].Id}"; }
        }
        else base0 = await _s.Macro.EvaluateAsync(b0);
        var (dz, _) = await LookDeltasAsync(b0, posts, base0);
        int hi = info.IndexOf("heal"), hpAfter = hp;
        if (hi >= 0)
        {
            decimal amount = Sts2Sim.Core.Hooks.Hook.ModifyRestSiteHealAmount(_run, _p.Creature, _p.Creature.MaxHp * 0.3m);
            hpAfter = (int)Math.Clamp(hp + amount, 0m, maxHp);
            dz[hi] = ZAt(base0.Out, hpAfter, maxHp) - ZAt(base0.Out, hp, maxHp);
            info[hi] = $"heal:{hp}->{hpAfter}";
        }
        return LookBonus("rest", _cfg.LookRest, dz, b0, posts, info, new() { ["heal_hp_after"] = hpAfter, ["smith_target"] = target });
    }

    // ---- events

    private int _evtFloor = -1;
    private readonly Dictionary<string, int> _evtChosen = new();

    public async Task<EventOption> ChooseEventOptionAsync(IReadOnlyList<EventOption> options)
    {
        Pre();
        var open = options.Where(o => !o.IsLocked).Take(ML.K).ToList();
        string evNote = (_run.BaseRoom as EventRoom)?.Event.Id.Entry ?? "?";
        _s.Note("event_options:" + evNote);
        if (open.Count == 0) { _s.Note("event_no_open_option:" + evNote); _lastEventKey = null; return EventOptionDecisionPolicy.ChooseDefault(options); }
        if (open.Count == 1) { _lastEventKey = open[0].Key; return open[0]; }
        string eventId = (_run.BaseRoom as EventRoom)?.Event.Id.ToString() ?? "";
        MacroRow row = StateRow();
        // an Ancient offers relics: the option's offered content is the relic (S1's ANCIENT decision)
        bool ancient = _run.CurrentMapPoint?.PointType == MapPointType.Ancient && _run.CurrentRoomCount == 1 &&
            open.All(o => o.OfferedContent is { ContentId.Length: > 0 } c && _s.MVocab.Has("relic", "RELIC." + c.ContentId));
        if (!ancient && _run.CurrentMapPoint?.PointType == MapPointType.Ancient) _s.Note("ancient_options_not_relics:" + eventId);
        if (ancient)
        {
            foreach (EventOption o in open) row.AddOption(ML.KRelic, _s.MVocab.Tok("relic", "RELIC." + o.OfferedContent!.ContentId));
            double[]? bonus = null;
            if (LookOn(_cfg.LookRelic))
            {
                // look_relic: relic added; only when no offered relic has a pickup effect (look_relic_any: always, token only)
                var models = open.Select(o => ModelDb.GetByIdOrNull<RelicModel>(new ModelId("RELIC", o.OfferedContent!.ContentId))).ToList();
                if (_cfg.LookRelicAny || models.All(m => m is not null && PlainRelic(m)))
                {
                    MacroRow b0 = StateRow();
                    var posts = open.Select(o => (MacroRow?)MacroEncoder.WithRelic(b0, _p, _s.MVocab.Tok("relic", "RELIC." + o.OfferedContent!.ContentId))).ToList();
                    var (dz, _) = await LookDeltasAsync(b0, posts);
                    bonus = LookBonus("relic", _cfg.LookRelic, dz, b0, posts,
                        open.Select((o, i) => $"{o.OfferedContent!.ContentId}:{(models[i] is { } m && PlainRelic(m) ? "plain" : "pickup")}").ToList());
                }
                else _s.Note("look_relic_skipped_pickup");
            }
            return open[await DecideOptionsAsync(row, ML.DAncient, null, bonus, bonus is null ? null : "relic")];
        }
        Rec.Events++;
        if (_cfg.CloneProbe > 0) await CloneProbeAsync(eventId, open);
        var keys = open.Select(o => eventId + ":" + o.Key).ToList();
        foreach (string k in keys) row.AddOption(ML.KEvent, ML.EventBucket(k));
        bool[]? ban = null;
        if (_cfg.Greedy && _cfg.GreedyEvent > 0)
        {
            if (_evtFloor != _run.TotalFloor) { _evtFloor = _run.TotalFloor; _evtChosen.Clear(); }
            ban = keys.Select(k => _evtChosen.GetValueOrDefault(k) >= _cfg.GreedyEvent).ToArray();
        }
        int ev = await DecideOptionsAsync(row, ML.DEvent, ban);
        if (ban is not null) _evtChosen[keys[ev]] = _evtChosen.GetValueOrDefault(keys[ev]) + 1;
        if (_s.Trace) Console.WriteLine($"TRACE {Rec.Seed} event {eventId} options {string.Join("|", open.Select(o => o.Key))} -> {open[ev].Key} (hp {_p.Creature.CurrentHp})");
        _lastEventKey = open[ev].Key;
        return open[ev];
    }

    private int _cloneProbeNo;

    /// <summary>clone_probe (scoping, PROPOSAL §6.1 step 10): a parallel world of the run at this event decision
    /// (<c>RunState.CloneReseeded</c>: hidden future redrawn), timed and dropped; with clone_probe=2 each open option is
    /// also chosen once on a fresh clone of its own (the event's own callback only: no rewards, fights or later rooms are
    /// driven) and its visible effect is recorded. Seeds come from the run seed and a counter, never from the agent's
    /// random stream, so the live run is unchanged.</summary>
    private async Task CloneProbeAsync(string eventId, List<EventOption> open)
    {
        ulong cseed = ResamplingClone.Mix(_seed, 0xC1013E0000UL + (ulong)_cloneProbeNo++);
        var sw = Stopwatch.StartNew();
        RunState? clone = null;
        string status, detail = "";
        try { clone = _run.CloneReseeded(cseed); status = "ok"; }
        catch (RunCloneNotSupportedException e) { status = "reject:" + e.Reason; detail = e.Message; }
        catch (Exception e) { status = "error:" + e.GetType().Name; detail = e.Message; }
        double sec = sw.Elapsed.TotalSeconds;
        _s.CloneProbes++; _s.CloneSeconds += sec; _s.CloneMaxSeconds = Math.Max(_s.CloneMaxSeconds, sec);
        if (clone is not null) { _s.CloneOk++; _s.Note("clone_ok"); }
        else _s.Note($"clone_{status}:{eventId}");
        static List<EventOption> Open(IReadOnlyList<EventOption> o) => o.Where(x => !x.IsLocked).Take(ML.K).ToList();
        EventModel? CloneEvent(RunState c) => (c.CurrentRoom as EventRoom ?? c.BaseRoom as EventRoom)?.Event;
        bool? match = null;
        if (clone is not null)
        {
            EventModel? ce = CloneEvent(clone);
            match = ce is not null && Open(ce.CurrentOptions).Select(o => o.Key).SequenceEqual(open.Select(o => o.Key));
            if (match != true) _s.Note("clone_options_differ:" + eventId);
        }
        var outcomes = new List<Dictionary<string, object?>>();
        if (_cfg.CloneProbe >= 2 && clone is not null)
        {
            for (int k = 0; k < open.Count; k++)
            {
                var ow = Stopwatch.StartNew();
                var res = new Dictionary<string, object?> { ["key"] = open[k].Key };
                try
                {
                    RunState c = _run.CloneReseeded(ResamplingClone.Mix(cseed, (ulong)k + 1));
                    EventModel ev = CloneEvent(c) ?? throw new InvalidOperationException("no event room in the clone");
                    EventOption opt = ev.CurrentOptions.First(o => o.Key == open[k].Key && !o.IsLocked);
                    Player q = c.Players[0];
                    int hp0 = q.Creature.CurrentHp, mhp0 = q.Creature.MaxHp, gold0 = q.Gold, deck0 = q.Deck.Cards.Count, rel0 = q.Relics.Count,
                        pot0 = q.PotionSlots.Count(x => x is not null);
                    Task t = ev.ChooseOption(opt);
                    res["sync"] = t.IsCompleted;
                    await t;
                    res["result"] = c.IsGameOver ? "game_over" : ev.IsAwaitingForcedCombat ? "fight" : ev.IsFinished ? "finished" : "next_page";
                    res["rewards"] = ev.HasPendingRewardOffers;
                    res["next_options"] = ev.IsFinished ? 0 : ev.CurrentOptions.Count;
                    res["dhp"] = q.Creature.CurrentHp - hp0; res["dmaxhp"] = q.Creature.MaxHp - mhp0; res["dgold"] = q.Gold - gold0;
                    res["ddeck"] = q.Deck.Cards.Count - deck0; res["drelics"] = q.Relics.Count - rel0;
                    res["dpotions"] = q.PotionSlots.Count(x => x is not null) - pot0;
                    _s.Note("clone_option_" + res["result"]);
                    res["ms"] = Math.Round(ow.Elapsed.TotalMilliseconds, 3);
                    // a second world for the same option: is its visible result the same (deterministic option)?
                    string f1 = Fingerprint(c, ev);
                    RunState c2 = _run.CloneReseeded(ResamplingClone.Mix(cseed, 1000 + (ulong)k));
                    EventModel ev2 = CloneEvent(c2)!;
                    await ev2.ChooseOption(ev2.CurrentOptions.First(o => o.Key == open[k].Key && !o.IsLocked));
                    res["same_in_2nd_world"] = Fingerprint(c2, ev2) == f1;
                }
                catch (Exception e)
                {
                    res["result"] = "error:" + e.GetType().Name;
                    res["detail"] = e.Message.Length > 200 ? e.Message[..200] : e.Message;
                    _s.OptionProbeErrors++; _s.Note($"clone_option_error:{e.GetType().Name}:{eventId}");
                }
                res.TryAdd("ms", Math.Round(ow.Elapsed.TotalMilliseconds, 3));
                _s.OptionProbes++; _s.OptionProbeSeconds += ow.Elapsed.TotalSeconds;
                outcomes.Add(res);
            }
        }
        // the player's visible state after an option on a clone (deck ids and upgrades, relics, potions, HP, gold) and the
        // event's state: equal fingerprints in two worlds = the option's result did not depend on the hidden future
        static string Fingerprint(RunState c, EventModel ev)
        {
            Player q = c.Players[0];
            return string.Join("|", q.Deck.Cards.Select(x => x.Id + "+" + x.CurrentUpgradeLevel).OrderBy(x => x, StringComparer.Ordinal)) + "#" +
                string.Join("|", q.Relics.Select(x => x.Id.ToString()).OrderBy(x => x, StringComparer.Ordinal)) + "#" +
                string.Join("|", q.PotionSlots.Select(x => x?.Id.ToString() ?? "")) + $"#{q.Creature.CurrentHp}/{q.Creature.MaxHp}#{q.Gold}" +
                $"#{ev.IsFinished}/{ev.IsAwaitingForcedCombat}/{ev.HasPendingRewardOffers}/" + string.Join("|", ev.CurrentOptions.Select(o => o.Key));
        }
        if (_s.CloneDump is not null)
        {
            var d = new Dictionary<string, object?>
            {
                ["seed"] = Rec.Seed, ["floor"] = _run.TotalFloor, ["act"] = _run.CurrentActIndex, ["event"] = eventId,
                ["room_count"] = _run.CurrentRoomCount, ["options"] = open.Select(o => o.Key).ToList(), ["status"] = status,
                ["detail"] = detail.Length > 200 ? detail[..200] : detail, ["ms"] = Math.Round(1000 * sec, 3), ["options_match"] = match,
            };
            if (_cfg.CloneProbe >= 2) d["option_probes"] = outcomes;
            lock (_s.CloneDump) _s.CloneDump.WriteLine(System.Text.Json.JsonSerializer.Serialize(d));
        }
    }

    public async Task<CustomEventDecision> ChooseCustomEventActionAsync(EventModel @event)
    {
        Pre();
        if (@event is Sts2Sim.Core.Models.Events.FakeMerchant fm && _cfg.FakeMerchant) return await FakeMerchantAsync(fm);
        _s.Note("custom_event_default:" + @event.Id.Entry);   // the simulator's default (Crystal Sphere: first hidden cell; Fake Merchant: leave)
        return CustomEventDecisionPolicy.ChooseDefault(@event);
    }

    private int _fakeFloor = -1, _fakeSteps;
    private bool _fakeShop;

    /// <summary>fake_merchant=1 (AgentConfig.FakeMerchant): one step of the Fake Merchant event.</summary>
    private async Task<CustomEventDecision> FakeMerchantAsync(Sts2Sim.Core.Models.Events.FakeMerchant fm)
    {
        if (_fakeFloor != _run.TotalFloor) { _fakeFloor = _run.TotalFloor; _fakeSteps = 0; _fakeShop = false; Rec.Look["fake_merchant"] = Rec.Look.GetValueOrDefault("fake_merchant") + 1; _s.Note("fake_merchant_visit"); }
        if (++_fakeSteps > 12) { _s.Note("fake_merchant_step_cap"); return new CustomEventDecision.Leave(); }
        string eventId = fm.Id.ToString();
        PotionModel? foul = _p.PotionSlots.FirstOrDefault(q => q is Sts2Sim.Core.Models.Potions.FoulPotion && q.PassesCustomUsabilityCheck);
        if (foul is not null && !_fakeShop)
        {
            _fakeShop = true;   // asked once per visit
            MacroRow er = StateRow();
            er.AddOption(ML.KEvent, ML.EventBucket(eventId + ":use_potion"));
            er.AddOption(ML.KEvent, ML.EventBucket(eventId + ":shop"));
            if (await DecideOptionsAsync(er, ML.DEvent) == 0)
            {
                _s.Note("fake_merchant_foul_thrown"); Rec.Look["fake_merchant_foul"] = Rec.Look.GetValueOrDefault("fake_merchant_foul") + 1;
                return new CustomEventDecision.UsePotion(foul);
            }
        }
        int gold = _p.Gold;
        var entries = fm.Inventory.Relics.Where(e => !e.Purchased && e.Price <= gold).Take(ML.K - 1).ToList();
        if (entries.Count == 0) { _s.Note("fake_merchant_nothing_affordable"); return new CustomEventDecision.Leave(); }
        MacroRow row = StateRow();
        float[] Price(int price) => new[] { price / 300f, price / (float)Math.Max(1, gold) };   // as in ChooseShopActionAsync
        foreach (var e in entries) row.AddOption(ML.KRelic, _s.MVocab.Tok("relic", e.Relic.Id.ToString()), 0, default, Price(e.Price));
        row.AddOption(ML.KZero, 0);   // LEAVE
        double[]? bonus = null;
        if (LookOn(_cfg.LookShop))
        {
            MacroRow b0 = StateRow();
            var posts = entries.Select(e => (MacroRow?)MacroEncoder.WithGold(MacroEncoder.WithRelic(b0, _p, _s.MVocab.Tok("relic", e.Relic.Id.ToString())), gold - e.Price))
                .Append(null).ToList();
            var info = entries.Select(e => $"fake_relic:{e.Relic.Id}:{e.Price}").Append("leave").ToList();
            var (dz, _) = await LookDeltasAsync(b0, posts);
            bonus = LookBonus("fake_shop", _cfg.LookShop, dz, b0, posts, info);
        }
        int pick = await DecideOptionsAsync(row, ML.DShop, null, bonus, bonus is null ? null : "fake_shop");
        if (pick >= entries.Count) { _s.Note("fake_merchant_leave"); return new CustomEventDecision.Leave(); }
        _s.Note("fake_merchant_buy:" + entries[pick].Relic.Id.Entry);
        Rec.Look["fake_merchant_buy"] = Rec.Look.GetValueOrDefault("fake_merchant_buy") + 1;
        return new CustomEventDecision.BuyRelic(entries[pick]);
    }

    // ---- card prompts

    private static readonly SearchAction[] ValueProbeAction = { new(ActionKind.EndTurn, "", null, Array.Empty<string[]>()) };

    /// <summary>
    /// Value-network expected-value selector for Gambling Chip: chooses a subset of the current hand (up to
    /// <paramref name="maxDiscard"/> cards) to discard and redraw from the remaining draw pile, maximizing
    /// E[V(s')] under the combat value head priced by this fight's T_f table. Uses stratified Common Random Numbers
    /// over the draw pile (exact expectation for 1-card replacements when draw &lt;= 32) and forward-backward greedy
    /// subset maximization.
    /// </summary>
    public async Task<(List<int> DiscardIndices, double BaseValue, double BestValue)> ChooseGamblingChipAsync(
        CombatState state, int maxDiscard)
    {
        Pre();
        if (_fight is null || !ReferenceEquals(_fight.Engine, state.Engine))
        {
            if (_fight is not null) EndFight(_fight.Engine.State);
            _fight = await StartFightAsync(state);
        }
        LiveFight f = _fight;
        CombatState work = state.Clone();
        PlayerCombatState pcs = work.Players[0].PlayerCombatState!;
        var origHand = pcs.Hand.Cards.ToList();
        var origDraw = pcs.DrawPile.Cards.ToList();
        var origDisc = pcs.DiscardPile.Cards.ToList();
        int n = origHand.Count, m = origDraw.Count;
        maxDiscard = Math.Clamp(maxDiscard, 0, n);
        if (n == 0 || maxDiscard == 0 || (m == 0 && origDisc.Count == 0))
            return (new List<int>(), 0.0, 0.0);

        // Build K stratified permutations of the draw pile (Common Random Numbers across all subsets).
        // When m <= 32, K = m and permutation r starts with origDraw[r], making 1-card replacement expectation exact.
        var rng = new Random((int)(ResamplingClone.Mix(f.Seed, 0x6A4B11C0UL) & 0x7FFFFFFF));
        int kSamples = m == 0 ? 1 : Math.Min(m, 32);
        var firstOrder = Enumerable.Range(0, m).ToArray();
        if (m > kSamples)
            for (int i = 0; i < kSamples; i++) { int j = rng.Next(i, m); (firstOrder[i], firstOrder[j]) = (firstOrder[j], firstOrder[i]); }
        var perms = new CardModel[kSamples][];
        for (int r = 0; r < kSamples; r++)
        {
            var arr = origDraw.ToArray();
            if (m > 0)
            {
                int first = firstOrder[r];
                (arr[0], arr[first]) = (arr[first], arr[0]);
                for (int i = 1; i < m - 1; i++) { int j = rng.Next(i, m); (arr[i], arr[j]) = (arr[j], arr[i]); }
            }
            perms[r] = arr;
        }

        int cap = _s.Net is BridgeEvaluator be ? Math.Max(1, be.Capacity) : 16;

        void SetPiles(HashSet<int> discard, int sampleIdx)
        {
            foreach (CardModel c in pcs.Hand.Cards.ToList()) pcs.Hand.RemoveInternal(c);
            foreach (CardModel c in pcs.DrawPile.Cards.ToList()) pcs.DrawPile.RemoveInternal(c);
            foreach (CardModel c in pcs.DiscardPile.Cards.ToList()) pcs.DiscardPile.RemoveInternal(c);

            var discList = new List<CardModel>(origDisc.Count + discard.Count);
            discList.AddRange(origDisc);
            for (int i = 0; i < n; i++)
                if (discard.Contains(i)) discList.Add(origHand[i]);
                else pcs.Hand.AddInternal(origHand[i]);

            int need = discard.Count;
            if (need == 0)
            {
                foreach (CardModel c in origDraw) pcs.DrawPile.AddInternal(c);
                foreach (CardModel c in origDisc) pcs.DiscardPile.AddInternal(c);
            }
            else if (need <= m)
            {
                CardModel[] perm = perms[sampleIdx];
                for (int i = 0; i < need; i++) pcs.Hand.AddInternal(perm[i]);
                for (int i = need; i < m; i++) pcs.DrawPile.AddInternal(perm[i]);
                foreach (CardModel c in discList) pcs.DiscardPile.AddInternal(c);
            }
            else
            {
                CardModel[] perm = perms[sampleIdx];
                for (int i = 0; i < m; i++) pcs.Hand.AddInternal(perm[i]);
                int rem = need - m;
                var reshuf = discList.ToArray();
                var r2 = new Random(sampleIdx * 7919 + 17);
                for (int i = 0; i < reshuf.Length - 1; i++) { int j = r2.Next(i, reshuf.Length); (reshuf[i], reshuf[j]) = (reshuf[j], reshuf[i]); }
                int take2 = Math.Min(rem, reshuf.Length);
                for (int i = 0; i < take2; i++) pcs.Hand.AddInternal(reshuf[i]);
                for (int i = take2; i < reshuf.Length; i++) pcs.DrawPile.AddInternal(reshuf[i]);
            }
            pcs.DrawPile.ResetPositionKnowledgeInternal();
        }

        async Task<double> EvalSubsetAsync(HashSet<int> discard)
        {
            if (discard.Count == 0)
            {
                SetPiles(discard, 0);
                return (await _s.Net.EvaluateAsync(work, ValueProbeAction, f.Signals)).Value;
            }
            double sum = 0.0;
            for (int off = 0; off < kSamples; off += cap)
            {
                int batch = Math.Min(cap, kSamples - off);
                var tasks = new Task<NetEval>[batch];
                for (int b = 0; b < batch; b++)
                {
                    SetPiles(discard, off + b);
                    tasks[b] = _s.Net.EvaluateAsync(work, ValueProbeAction, f.Signals);
                }
                NetEval[] evs = await Task.WhenAll(tasks);
                for (int b = 0; b < batch; b++) sum += evs[b].Value;
            }
            return sum / kSamples;
        }

        string[] handKeys = origHand.Select(CardSelectionOptions.EquivalenceKey).ToArray();
        var best = new HashSet<int>();
        double baseVal = await EvalSubsetAsync(best);
        double bestVal = baseVal;

        // Forward greedy: iteratively discard whichever remaining card increases E[V(s')] the most.
        while (best.Count < maxDiscard)
        {
            int pick = -1;
            double pickVal = bestVal;
            var seenKeys = new HashSet<string>();
            for (int i = 0; i < n; i++)
            {
                if (best.Contains(i) || !seenKeys.Add(handKeys[i])) continue;
                best.Add(i);
                double v = await EvalSubsetAsync(best);
                best.Remove(i);
                if (v > pickVal + 1e-6) { pickVal = v; pick = i; }
            }
            if (pick < 0) break;
            best.Add(pick);
            bestVal = pickVal;
        }

        // Backward check: undo any discard that hurts E[V(s')] once other cards were discarded.
        while (best.Count >= 2)
        {
            int drop = -1;
            double dropVal = bestVal;
            var seenKeys = new HashSet<string>();
            foreach (int i in best.ToList())
            {
                if (!seenKeys.Add(handKeys[i])) continue;
                best.Remove(i);
                double v = await EvalSubsetAsync(best);
                best.Add(i);
                if (v > dropVal + 1e-6) { dropVal = v; drop = i; }
            }
            if (drop < 0) break;
            best.Remove(drop);
            bestVal = dropVal;
        }

        return (best.OrderBy(x => x).ToList(), baseVal, bestVal);
    }

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
        if (purpose is not null) _s.Note($"deck_select:{purpose}:{request.Source?.GetType().Name ?? "none"}");
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
                CardModel c = await DecideDeckCardAsync(left, dtype, removal: purpose == "remove");
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
        if (_cfg.DeckRoute && !request.IsBundleSelection && request.MaxCount >= 1)
        {
            IReadOnlyList<CardModel>? routed = await DeckRouteAsync(request, inDeck);
            if (routed is not null) return routed;
        }
        // anything else: a random legal answer of the largest size
        var pool = cands.ToList();
        var res = new List<CardModel>();
        int want = request.IsBundleSelection ? 1 : request.MaxCount;
        while (res.Count < want && pool.Count > 0) { int i = _rng.Next(pool.Count); res.Add(pool[i]); pool.RemoveAt(i); }
        return res;
    }

    /// <summary>The last event option this agent chose (deck_route: tells Symbiote's and Trial's two prompts apart).</summary>
    private string? _lastEventKey;

    /// <summary>deck_route: in-deck sources without a deck purpose (src/ic113/DECK_ROUTE.md). "smith" = the chosen
    /// card gets better (enchantment, upgrade, copy); "remove" = it leaves the deck; "transform" = it is replaced; null =
    /// not routed (random answer as before).</summary>
    private static string? DeckRouteOf(string source, string? eventKey) => source switch
    {
        "SelfHelpBook" or "SpiralingWhirlpool" or "Kifuda" or "TriBoomerang" or "GnarledHammer" or "RoyalStamp" or
            "ElectricShrymp" or "DollysMirror" or "PunchDagger" or "GraveOfTheForgotten" or "StoneOfAllTime" or
            "WaterloggedScriptorium" or "PaelsGrowth" => "smith",
        "LordsParasol" or "PreciseScissors" or "DoorsOfLightAndDark" or "PreservedFog" or "EmptyCage" or "PaelsTooth" => "remove",
        "EndlessConveyor" or "Claws" => "transform",
        "Symbiote" => eventKey switch { "APPROACH" => "smith", "KILL_WITH_FIRE" => "transform", _ => null },
        "Trial" => eventKey switch { "MERCHANT_INNOCENT" => "smith", "NONDESCRIPT_INNOCENT" => "transform", _ => null },
        _ => null,
    };

    /// <summary>deck_route=1 (AgentConfig.DeckRoute): a run-level card prompt the old code answered at random. In-deck
    /// sources go to the smith / removal head (MaxCount picks one by one, as the purpose prompts); new cards to the
    /// card-reward head, one pick at a time (skip offered once MinCount is reached, or at the first pick of a cancelable
    /// prompt), at most K-1 candidates per step (a random shortlist when more are offered). Null = not routed.
    /// Every decision is recorded by DecideSlotAsync / DecideOptionsAsync like the other smith / removal / card decisions.</summary>
    private async Task<IReadOnlyList<CardModel>?> DeckRouteAsync(CardSelectionRequest request, bool inDeck)
    {
        var cands = request.Candidates;
        string src = request.Source?.GetType().Name ?? "none";
        string? route = inDeck ? DeckRouteOf(src, _lastEventKey) : cands.All(c => DeckIndex(c) < 0) ? "card" : null;
        if (route is null) { _s.Note("deck_route_none:" + src); return null; }
        string key = $"deck_route:{src}:{route}";
        Rec.Look[key] = Rec.Look.GetValueOrDefault(key) + 1;
        _s.Note(key);
        var left = cands.ToList();
        var picked = new List<CardModel>();
        if (route != "card")
        {
            int dtype = route == "smith" ? ML.DSmith : ML.DRemove;
            while (picked.Count < request.MaxCount && left.Count > 0)
            {
                CardModel c = await DecideDeckCardAsync(left, dtype, removal: route == "remove");
                picked.Add(c); left.Remove(c);
            }
            return picked;
        }
        while (picked.Count < request.MaxCount && left.Count > 0)
        {
            bool canSkip = picked.Count >= request.MinCount || (request.Cancelable && picked.Count == 0);
            List<CardModel> shortlist = left;
            if (left.Count > ML.K - 1)
            {
                var idx = Enumerable.Range(0, left.Count).ToList();
                for (int i = idx.Count - 1; i > 0; i--) { int j = _rng.Next(i + 1); (idx[i], idx[j]) = (idx[j], idx[i]); }
                shortlist = idx.Take(ML.K - 1).OrderBy(i => i).Select(i => left[i]).ToList();
                _s.Note("deck_route_shortlist:" + src);
            }
            MacroRow row = StateRow();
            foreach (CardModel c in shortlist) row.AddOption(ML.KCard, _s.MVocab.Tok("card", MacroEncoder.CardId(c)), c.CurrentUpgradeLevel);
            if (canSkip) row.AddOption(ML.KSpecial, 1);
            int pick = await DecideOptionsAsync(row, ML.DCard);
            if (pick >= shortlist.Count) break;
            picked.Add(shortlist[pick]); left.Remove(shortlist[pick]);
        }
        if (picked.Count == 0) _s.Note("deck_route_skip:" + src);
        return picked;
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
            // ic113: the HP (and max HP) the post-combat hooks leave after a win at end HP h; the post-fight row carries
            // the max HP after them (Chosen Cheese), the HP axis is read at heal(h) (PostCombatHeal; bin 0 = death untouched)
            var heal = _cfg.TfHeal ? PostCombatHeal.Table(_p, maxHp) : PostCombatHeal.Identity(maxHp);
            int postMax = heal.MaxHpAfter;
            post.SetHp(postMax, postMax);
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
                    double v0 = Math.Clamp(ML.MilestoneValue(evs[k].Out, postMax, postMax, stagePost, dbl), 1e-6, 1 - 1e-6);
                    _s.TfFormulaMaxError = Math.Max(_s.TfFormulaMaxError, Math.Abs(Math.Log(v0 / (1 - v0)) - evs[k].Out[ML.OutV]));
                    for (int h = 1; h <= maxHp; h++)
                    {
                        double v = Math.Clamp(ML.MilestoneValue(evs[k].Out, heal.Hp[h], postMax, stagePost, dbl), 1e-9, 1 - 1e-9);
                        tables[k][h] = shift == 0 ? v : ML.Sigmoid(Math.Log(v / (1 - v)) - shift);
                    }
                }
                else
                {
                    _s.TfFormulaMaxError = Math.Max(_s.TfFormulaMaxError, Math.Abs(ML.LogitAt(evs[k].Out, postMax, postMax) - evs[k].Out[ML.OutV]));
                    for (int h = 1; h <= maxHp; h++) tables[k][h] = ML.Sigmoid(ML.LogitAt(evs[k].Out, heal.Hp[h], postMax) - shift);
                }
            }
            Rec.MacroVersion = evs[0].ModelVersion;
            if (heal.Healing) _s.Note("tf_heal_fight");
        }
        double death = DeathValueHere();
        double[][]? raw = _s.TfDump is not null ? tables.Select(t => (double[])t.Clone()).ToArray() : null;
        bool firstOfTwo = rt == RoomType.Boss && room is not null && _run.CurrentRoomCount == 1 && IsFirstOfDoubleBoss();
        string? tfKind = null;
        if (firstOfTwo && _cfg.Boss2 is { } b2 && _cfg.Scorer == "tf")
        {
            // first of two act-3 bosses: what winning it is worth is the chance of then winning the second one
            for (int k = 0; k < subsets.Count; k++)
            {
                var heal2 = _cfg.TfHeal ? PostCombatHeal.Table(_p, maxHp) : PostCombatHeal.Identity(maxHp);
                for (int h = 1; h <= maxHp; h++)
                    tables[k][h] = death + (1 - death) * b2.W2(heal2.Hp[h] / (double)heal2.MaxHpAfter, subsets[k].Length);
            }
            tfKind = "boss2";
            _s.TfBoss2Fights++;
        }
        bool kindScale = _cfg.PotionKindScale && _s.PotionValues is not null && _s.PotionValueDefault > 0;
        if (!final && _cfg.Scorer == "tf" && tfKind is null && (_cfg.PotionScale != 1.0 || _cfg.HpScale != 1.0 || kindScale))
        {
            double[] empty = (double[])tables[0].Clone();   // subsets are listed by size: tables[0] is the empty belt
            for (int k = 0; k < subsets.Count; k++)
            {
                double w = _cfg.PotionScale;
                if (kindScale && subsets[k].Length > 0)
                    w *= subsets[k].Average(i => Math.Clamp(_s.PotionValues!.GetValueOrDefault(belt[i], _s.PotionValueDefault) / _s.PotionValueDefault, 0.1, 4.0));
                for (int h = 1; h <= maxHp; h++)
                {
                    double pot = tables[k][h] - empty[h];
                    double hpPart = Math.Max(empty[maxHp] - _cfg.HpScale * (empty[maxHp] - empty[h]), 0.02 * empty[h]);
                    tables[k][h] = Math.Clamp(hpPart + w * pot, 1e-6, 1.0);
                }
            }
            _s.Note("price_probe_fight");
        }
        List<int>? reserveTop = null;
        double reserveTMax0 = double.NaN;
        if (_cfg.PotionReserve != 1.0 && _cfg.Scorer == "tf" && !final && tfKind is null && _s.PotionValues is not null
            && !(rt == RoomType.Boss && _run.CurrentActIndex >= 2))
        {
            // reservation price for the top potions (not at the act-3 bosses): a subset's gain over the same subset
            // without its top potions is multiplied by W at end HP shares from hi up, fading to x1 at lo
            var top = held.Where(i => _s.PotionValues.TryGetValue(belt[i], out double pv) && pv >= _cfg.PotionReserveMin).ToHashSet();
            if (top.Count > 0)
            {
                var index = new Dictionary<string, int>();
                for (int k = 0; k < subsets.Count; k++) index[string.Join(",", subsets[k])] = k;
                double[][] orig = tables.Select(t => (double[])t.Clone()).ToArray();
                reserveTMax0 = orig.Max(t => t.Max());
                double lo = _cfg.PotionReserveHpLo, hi = _cfg.PotionReserveHpHi;
                for (int k = 0; k < subsets.Count; k++)
                {
                    if (!subsets[k].Any(top.Contains)) continue;
                    int j = index[string.Join(",", subsets[k].Where(i => !top.Contains(i)))];
                    for (int h = 1; h <= maxHp; h++)
                    {
                        double x = h / (double)maxHp;
                        double ramp = hi > lo ? Math.Clamp((x - lo) / (hi - lo), 0, 1) : x >= lo ? 1 : 0;
                        double gain = orig[k][h] - orig[j][h];
                        if (ramp <= 0 || gain <= 0) continue;
                        tables[k][h] = Math.Min(1.0, orig[k][h] + (_cfg.PotionReserve - 1) * ramp * gain);
                        _s.PotionReserveEntries++;
                    }
                }
                reserveTop = top.OrderBy(i => i).ToList();
                _s.PotionReserveFights++;
                _s.Note("potion_reserve_fight");
            }
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
        if (_s.TfDump is not null) DumpTables(id, rt, room, final, firstOfTwo, tfKind, maxHp, pc.CurrentHp, belt, subsets, death, deathFormula, raw!, tables, reserveTop);
        _s.Tf.Set(id, maxHp, belt, subsets, tables, final, death);
        var ctx = new BattleContext(id, maxHp, belt);
        string enc = room?.Encounter is { } e ? "ENCOUNTER." + e.IdEntry : room?.EncounterName ?? "";
        var spec = new BattleStartSpec
        {
            Id = id, RoomType = RoomName(rt), ActIndex = _run.CurrentActIndex, Floor = _run.TotalFloor, EncounterId = enc,
            Hp = pc.CurrentHp, MaxHp = maxHp, Potions = belt, Ascension = _asc,
        };
        if (_s.StartsDump is not null && !Rec.Perturbed && (rt == RoomType.Boss || (rt == RoomType.Elite && _s.StartsDumpElite)))
            DumpStart(id, rt, enc, state, maxHp, belt, held, subsets, tables, final, firstOfTwo);
        double tMax = Math.Max(1e-3, _s.Tf.MaxValue(id));
        if (reserveTop is not null) _s.PotionReserveTMaxRatioSum += tMax / Math.Max(1e-3, reserveTMax0);
        FightSignals? signals = _cfg.Context ? new FightSignals(_s.Tf, ctx, spec, tMax) : null;
        var o = new FightOutcome
        {
            Id = id, RoomType = spec.RoomType, Encounter = enc, StartHp = pc.CurrentHp, PotionsStart = held.Count, TMax = tMax,
            TableMaxHp = maxHp, DeathValue = death, TfKind = tfKind,
            ReserveTMax0 = reserveTop is null ? double.NaN : Math.Max(1e-3, reserveTMax0),
        };
        return new LiveFight
        {
            Engine = state.Engine!, Id = id, Ctx = ctx, Signals = signals, O = o, Room = rt, Seed = seed,
            Search = new NetSearch(SearchConfigFor(rt), _s.Net, _s.Tf, ctx, tMax, signals),
        };
    }

    /// <summary>starts_dump: this fight's start as a BattleStartSpec line (potions compacted to the held ones) with its
    /// table under "table" (the format of the S2 tables file).</summary>
    private void DumpStart(string id, RoomType rt, string enc, CombatState state, int maxHp, List<string> belt, List<int> held,
        List<int[]> subsets, double[][] tables, bool final, bool firstOfTwo)
    {
        var pos = new Dictionary<int, int>();
        for (int j = 0; j < held.Count; j++) pos[held[j]] = j;
        var pots = held.Select(i => belt[i]).ToList();
        var d = new Dictionary<string, object?>
        {
            ["id"] = id, ["build_id"] = ML.BuildId, ["character"] = _p.Character.Id.ToString(), ["ascension"] = _asc,
            ["act_ids"] = _run.Acts.Select(a => "ACT." + a.GetType().Name.ToUpperInvariant()).ToList(),
            ["act_index"] = _run.CurrentActIndex, ["floor"] = _run.TotalFloor, ["room_type"] = RoomName(rt), ["encounter_id"] = enc,
            ["monster_ids"] = state.Enemies.Select(e => e.Monster!.Id.ToString()).ToList(),
            ["hp"] = _p.Creature.CurrentHp, ["max_hp"] = maxHp, ["gold"] = _p.Gold,
            ["deck"] = _p.Deck.Cards.Select(c => new Dictionary<string, object?>
            {
                ["id"] = c.Id.ToString(), ["upgrade"] = c.CurrentUpgradeLevel,
                ["enchantment"] = c.Enchantments.Count > 0 ? c.Enchantments[0].Id.ToString() : null,
                ["enchantment_amount"] = c.Enchantments.Count > 0 ? (int)c.Enchantments[0].Magnitude : 0,
            }).ToList(),
            ["relics"] = _p.Relics.Select(r => r.Id.ToString()).ToList(), ["potions"] = pots, ["potion_slots"] = belt.Count,
            ["final_boss"] = final, ["first_of_two_bosses"] = firstOfTwo, ["run_seed"] = Rec.Seed,
            ["table"] = new Dictionary<string, object?>
            {
                ["id"] = id, ["max_hp"] = maxHp, ["belt"] = pots,
                ["subsets"] = subsets.Select(s => s.Select(i => pos[i]).ToArray()).ToList(),
                ["tables"] = tables.Select(t => t.Select(x => Math.Round(x, 6)).ToArray()).ToArray(), ["final_boss"] = final,
            },
        };
        _s.StartsDump!.WriteLine(System.Text.Json.JsonSerializer.Serialize(d));
    }

    private NetSearchConfig? _exactCfg;
    /// <summary>Search settings of a live fight: with <see cref="AgentConfig.ExactBoss"/> boss fights (act 3 only with
    /// value 3) search in exact clones that see the true future (an upper bound, evaluation only); otherwise the agent's.</summary>
    private NetSearchConfig SearchConfigFor(RoomType rt)
    {
        if (_cfg.ExactBoss == 0 || rt != RoomType.Boss || (_cfg.ExactBoss == 3 && _run.CurrentActIndex < 2)) return _cfg.Search;
        if (_exactCfg is null)
        {
            _exactCfg = _cfg.Search.Copy();
            if (_cfg.ExactPart == "draw") _exactCfg.KeepDraw = true;
            else if (_cfg.ExactPart == "enemy") _exactCfg.KeepEnemy = true;
            else if (_cfg.ExactPart == "worlds") _exactCfg.Worlds = _cfg.BossWorlds;
            else _exactCfg.ExactWorlds = true;
        }
        _s.Note("exact_boss_fight");
        return _exactCfg;
    }

    /// <summary>F3 debug dump: the first TfDumpK live fights per kind (act-3 boss; hallway fight of act 1, 2, 3) of this
    /// actor, with the macro network's table before any change ("raw", before free disposal; all 1 at the final boss) and
    /// the table the fight is scored by ("tables", before free disposal, which TfScorer applies).</summary>
    private void DumpTables(string id, RoomType rt, CombatRoom? room, bool final, bool firstOfTwo, string? tfKind, int maxHp, int hp,
        List<string> belt, List<int[]> subsets, double death, double deathFormula, double[][] raw, double[][] tables, List<int>? reserveTop = null)
    {
        string? key = rt == RoomType.Boss && _run.CurrentActIndex == 2 ? "boss3" : rt == RoomType.Monster ? $"monster{_run.CurrentActIndex + 1}"
            : _s.TfDumpAll ? $"{RoomName(rt)}{_run.CurrentActIndex + 1}" : null;
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
        if (_cfg.PotionReserve != 1.0)
        {
            d["potion_reserve"] = _cfg.PotionReserve; d["reserve_min"] = _cfg.PotionReserveMin;
            d["reserve_hp_lo"] = _cfg.PotionReserveHpLo; d["reserve_hp_hi"] = _cfg.PotionReserveHpHi;
            d["reserve_top"] = reserveTop;   // belt indices of the top potions; null = the fight was left unchanged
        }
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

    /// <summary>ic113 port checks only (PortChecks.cs; null in every actor): called with the live root state before each
    /// search (result null) and after it (with the result).</summary>
    public static Func<CombatState, NetSearchResult?, Task>? SearchHook;

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
            WriteTrace(f, "STALEMATE");
            throw new InvalidOperationException($"combat stalemate: {o.Encounter} still in progress at round {state.RoundNumber}");
        }
        bool capped = state.RoundNumber > _cfg.MaxRounds;
        // one draw per move whatever the settings, so the actor's random stream does not depend on them
        bool full = DrawFull(_rng, f.Room);
        int sims = MoveSims(f.Room, capped, full);
        if (SearchHook is { } hook) await hook(state, null);
        NetSearchResult r = await f.Search.SearchAsync(state, sims, ResamplingClone.Mix(f.Seed, (ulong)o.Decisions));
        if (capped && _cfg.LethalBackstop) await LethalBackstopAsync(f.Search, state, r, ResamplingClone.Mix(f.Seed ^ 0x1E7A1UL, (ulong)o.Decisions), false);
        if (SearchHook is { } hook2) await hook2(state, r);
        if (StaleTrace is not null) { f.Trace ??= new(); TraceDecision(f, state, r); }
        Account(f, o, state, r, full, capped);
        if (_cfg.AuditP > 0 && r.PolicyTarget is not null && r.RootEval is not null && (_cfg.AuditAll || f.Room == RoomType.Boss)
            && _auditRng.NextDouble() < _cfg.AuditP)
            await AuditAsync(f, state, r, sims, o.Decisions);
        o.Decisions++;
        return ToDecision(state, r.Action);
    }

    private readonly Random _auditRng = new(7919);
    private static StreamWriter? _auditWriter;
    private static readonly object _auditLock = new();

    /// <summary>Target audit (<see cref="AgentConfig.AuditP"/>): repeat the search of this state and log how far the
    /// policy targets of two equal-budget searches, a large search and the prior lie apart (KL in nats, argmax agreement).</summary>
    private async Task AuditAsync(LiveFight f, CombatState state, NetSearchResult a, int sims, int dec)
    {
        NetSearchResult b = await f.Search.SearchAsync(state, sims, ResamplingClone.Mix(f.Seed ^ 0xA0D17A0D17UL, (ulong)dec));
        NetSearchResult big = await f.Search.SearchAsync(state, _cfg.AuditSims, ResamplingClone.Mix(f.Seed ^ 0xB16B16B16UL, (ulong)dec));
        int n = a.RootActions.Count;
        if (b.PolicyTarget is null || big.PolicyTarget is null || b.RootActions.Count != n || big.RootActions.Count != n) return;
        float[] pa = a.PolicyTarget!, pb = b.PolicyTarget, pg = big.PolicyTarget, lg = a.RootEval!.Logits;
        var prior = new double[n];
        double mx = lg.Take(n).Max(), z = 0;
        for (int i = 0; i < n; i++) { prior[i] = Math.Exp(lg[i] - mx); z += prior[i]; }
        for (int i = 0; i < n; i++) prior[i] /= z;
        static double Kl(IReadOnlyList<double> p, IReadOnlyList<double> q)
        {
            double s = 0;
            for (int i = 0; i < p.Count; i++) if (p[i] > 1e-12) s += p[i] * Math.Log(p[i] / Math.Max(q[i], 1e-12));
            return s;
        }
        static double H(IReadOnlyList<double> p) => -p.Where(x => x > 1e-12).Sum(x => x * Math.Log(x));
        static int Arg(IReadOnlyList<double> p) { int k = 0; for (int i = 1; i < p.Count; i++) if (p[i] > p[k]) k = i; return k; }
        double[] A = pa.Take(n).Select(x => (double)x).ToArray(), B = pb.Take(n).Select(x => (double)x).ToArray(),
                 G = pg.Take(n).Select(x => (double)x).ToArray();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        string F(double x) => x.ToString("0.#####", ci);
        string line = $"{{\"id\":\"{f.Id}\",\"room\":\"{f.Room}\",\"round\":{state.RoundNumber},\"n\":{n},\"sims\":{sims},\"big\":{_cfg.AuditSims}," +
            $"\"kl_ab\":{F(Kl(A, B))},\"kl_ba\":{F(Kl(B, A))},\"kl_ga\":{F(Kl(G, A))},\"kl_gb\":{F(Kl(G, B))},\"kl_gp\":{F(Kl(G, prior))}," +
            $"\"kl_ap\":{F(Kl(A, prior))},\"kl_ag\":{F(Kl(A, G))},\"h_a\":{F(H(A))},\"h_g\":{F(H(G))},\"h_p\":{F(H(prior))}," +
            $"\"same_ab\":{(Arg(A) == Arg(B) ? 1 : 0)},\"same_ag\":{(Arg(A) == Arg(G) ? 1 : 0)},\"same_pg\":{(Arg(prior) == Arg(G) ? 1 : 0)}," +
            $"\"act_a\":{(a.RootActions.IndexOf(a.Action) == Arg(A) ? 1 : 0)},\"same_act_ab\":{(a.RootActions.IndexOf(a.Action) == b.RootActions.IndexOf(b.Action) ? 1 : 0)}}}";
        lock (_auditLock)
        {
            _auditWriter ??= new StreamWriter(_cfg.AuditPath, append: true);
            _auditWriter.WriteLine(line); _auditWriter.Flush();
        }
    }

    /// <summary>lethal_backstop=1 (AgentConfig.LethalBackstop), moves past max_rounds only: if a root action wins the fight
    /// at once on a resampled clone (NetSearch.WinNowAsync), it replaces the prior's argmax in <paramref name="r"/>.
    /// Capped moves are not recorded as training data (Account), so only the move played changes.</summary>
    private async Task LethalBackstopAsync(NetSearch search, CombatState state, NetSearchResult r, ulong seed, bool practice)
    {
        if (r.RootActions.Count < 2) return;
        int k = await search.WinNowAsync(state, r.RootActions, seed);
        string tag = practice ? "practice_" : "";
        if (k < 0) { _s.Note(tag + "lethal_backstop_none"); return; }
        if (r.RootActions[k].Key == r.Action.Key) { _s.Note(tag + "lethal_backstop_same"); return; }
        r.Action = r.RootActions[k];
        _s.Note(tag + "lethal_backstop_played");
        if (!practice) Rec.Look["lethal_backstop"] = Rec.Look.GetValueOrDefault("lethal_backstop") + 1;
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
        o.NoteRoot(r);
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
                if (capped && _cfg.LethalBackstop) await LethalBackstopAsync(search, state, r, ResamplingClone.Mix(seed ^ 0x1E7A1UL, (ulong)o.Decisions), true);
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

    /// <summary>ic113 live check of <see cref="PostCombatHeal"/>: after a won fight (EndFight, before the room resolves),
    /// the HP the hooks should leave; compared with the player's HP when the room is resolved (notes heal_check_ok /
    /// heal_check_mismatch; skipped when a relic was gained in between, e.g. an elite's relic reward that changes max HP).</summary>
    private (int Hp, int MaxHp, int EndHp, int Relics)? _healExpect;

    private void HealCheck()
    {
        if (_healExpect is not { } e) return;
        _healExpect = null;
        if (_p.Relics.Count != e.Relics) { _s.Note("heal_check_skipped_relic_reward"); return; }
        if (_p.Creature.CurrentHp == e.Hp && _p.Creature.MaxHp == e.MaxHp) _s.Note(e.Hp != e.EndHp ? "heal_check_ok_healed" : "heal_check_ok");
        else
        {
            _s.Note("heal_check_mismatch");
            if (_s.Notes.GetValueOrDefault("heal_check_mismatch") <= 5)
                Console.WriteLine($"heal check mismatch {Rec.Seed} floor {_run.TotalFloor}: end HP {e.EndHp}, expected {e.Hp}/{e.MaxHp}, " +
                    $"got {_p.Creature.CurrentHp}/{_p.Creature.MaxHp}, relics {string.Join(",", _p.Relics.Select(r => r.Id.Entry))}");
        }
    }

    private void EndFight(CombatState state)
    {
        LiveFight f = _fight!;
        _fight = null;
        FightOutcome o = f.O;
        Creature pc = state.Players[0].Creature;
        o.EndHp = Math.Max(0, pc.CurrentHp); o.MaxHp = pc.MaxHp; o.Turns = state.RoundNumber;
        bool won = state.Engine is { IsInProgress: false, Won: true } && o.EndHp > 0;
        if (won)
        {
            var (hpAfter, maxAfter) = PostCombatHeal.Apply(state.Players[0].Relics, o.EndHp, o.MaxHp);
            _healExpect = (hpAfter, maxAfter, o.EndHp, state.Players[0].Relics.Count);
        }
        o.Result = won ? "won" : "lost";
        o.Score = won ? _s.Tf.Score(state, f.Ctx) : 0.0;
        o.Rel = o.Score / o.TMax;
        if (f.Signals is not null) o.EndHeldMask = f.Signals.HeldMask(state);
        o.Seconds = f.Watch.Elapsed.TotalSeconds;
        f.Search.Clear();
        _s.Tf.Remove(f.Id);
        if (StaleTraceEnc is not null && (StaleTraceEnc == "all" || StaleTraceEnc == o.Encounter)) WriteTrace(f, $"{o.Result} end hp {o.EndHp} rounds {o.Turns}");
        _s.CombatWriter?.Add(o);
        _s.Simulations += o.Simulations; _s.CombatDecisions += o.Decisions; _s.Fights++;
        Rec.Fights++; Rec.Simulations += o.Simulations; Rec.CombatDecisions += o.Decisions; Rec.PotionsUsed += o.PotionsUsed;
        Rec.ModelVersion = o.ModelVersion;
        // layout v3: the fight for the fight tokens of later rows
        _facts.Fights.Add(new FightNote(_s.MVocab.Tok("encounter", o.Encounter), o.Encounter,
            f.Room == RoomType.Boss ? ML.FightBoss : f.Room == RoomType.Elite ? ML.FightElite : ML.FightHall,
            o.StartHp, o.EndHp, o.TableMaxHp, o.PotionsUsed, o.Turns, _run.TotalFloor, _run.CurrentActIndex));
        if (f.Room == RoomType.Elite) Rec.Elites++;
        if (f.Room == RoomType.Boss) Rec.Bosses++;
        if (won) { Rec.FightsWon++; _s.FightsWon++; if (f.Room == RoomType.Boss) Rec.BossesWon++; }
        else Rec.KilledBy = o.Encounter;
        _s.FightLog?.Invoke(o, Rec, _run.TotalFloor);
    }
}
