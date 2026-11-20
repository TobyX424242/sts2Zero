using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Relics;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;

namespace Sts2Sim.S3;

/// <summary>
/// ic113 (2026-11-18): HP and max HP after the simulator's post-combat hooks of a WON fight, as a function of the HP at
/// the fight's end. In iRyougi v0.113.1.1 a combat room resolves a win as (CombatRoom.ResolveVictoryOnceAsync):
/// <c>Hook.AfterCombatEnd</c> (every listener), then <c>Hook.AfterCombatVictory</c> = an AfterCombatVictoryEarly pass
/// over every listener followed by an AfterCombatVictory pass. Of all models only four change the player's HP or max HP
/// there (code search of every AfterCombatEnd / AfterCombatVictory(Early) override, 2026-11-18; `S3 healtest` checks
/// every relic of the simulator against its own hooks):
/// <list type="bullet">
/// <item>Chosen Cheese (AfterCombatEnd): max HP + 1 (CreatureCmd.GainMaxHp: also heals the 1);</item>
/// <item>Meat on the Bone (AfterCombatVictoryEarly): heals 12 if alive and HP &lt;= 50% of max HP (checked before the
/// Victory pass, so before Burning / Black Blood);</item>
/// <item>Burning Blood (AfterCombatVictory): heals 6 if alive;</item>
/// <item>Black Blood (AfterCombatVictory): heals 12 if alive.</item>
/// </list>
/// Heals are not modified by hooks in the simulator (CreatureCmd.Heal -> HealInternal, clamped at max HP). Melted relics
/// are not hook listeners. The fight's end (what the search's terminal states and the combat network's end-HP target
/// read: RunAgent.EndFight runs in ICombatCompletionValidator before the room resolves) is BEFORE these hooks.
/// </summary>
public static class PostCombatHeal
{
    public sealed class HealTable
    {
        /// <summary>HP after the hooks for a win at end HP h (index 0 = death: 0, never healed).</summary>
        public required int[] Hp;
        public required int MaxHpAfter;
        public bool Healing;
    }

    /// <summary>(HP, max HP) after the post-combat hooks of a win at <paramref name="hp"/> / <paramref name="maxHp"/>.</summary>
    public static (int Hp, int MaxHp) Apply(IReadOnlyList<RelicModel> relics, int hp, int maxHp)
    {
        if (hp <= 0) return (Math.Max(0, hp), maxHp);
        var live = relics.Where(r => !r.IsMelted).ToList();
        // AfterCombatEnd pass
        foreach (RelicModel r in live)
            if (r is ChosenCheese) { maxHp += 1; hp = Math.Min(hp + 1, maxHp); }
        // AfterCombatVictoryEarly pass
        foreach (RelicModel r in live)
            if (r is MeatOnTheBone && hp > 0 && hp <= maxHp * 0.5m) hp = Math.Min(hp + 12, maxHp);
        // AfterCombatVictory pass
        foreach (RelicModel r in live)
        {
            if (r is BurningBlood && hp > 0) hp = Math.Min(hp + 6, maxHp);
            else if (r is BlackBlood && hp > 0) hp = Math.Min(hp + 12, maxHp);
        }
        return (hp, maxHp);
    }

    public static HealTable Table(Player p, int maxHp)
    {
        var hp = new int[maxHp + 1];
        int maxAfter = maxHp;
        bool healing = false;
        for (int h = 1; h <= maxHp; h++)
        {
            var (a, m) = Apply(p.Relics, h, maxHp);
            hp[h] = a; maxAfter = m;
            if (a != h) healing = true;
        }
        if (maxAfter != maxHp) healing = true;
        return new HealTable { Hp = hp, MaxHpAfter = maxAfter, Healing = healing };
    }

    public static HealTable Identity(int maxHp) =>
        new() { Hp = Enumerable.Range(0, maxHp + 1).ToArray(), MaxHpAfter = maxHp, Healing = false };

    // ------------------------------------------------------------------ test against the simulator

    /// <summary>`S3 healtest`: wins real fights (RunDriver injected combat: the fight ends, the validator reads the end HP,
    /// then CombatRoom resolves the victory with the simulator's own hooks) holding each relic of the simulator alone, and
    /// the heal relics in combinations, at end HPs around the thresholds; the HP / max HP after the room resolved must
    /// equal <see cref="Apply"/> of the end HP. Any relic outside the table that changes HP or max HP there fails.</summary>
    public static async Task<int> Test()
    {
        var relicIds = ModelDb.All<RelicModel>().Select(r => r.Id.ToString()).Distinct().Order(StringComparer.Ordinal).ToList();
        var sets = relicIds.Select(id => new List<string> { id }).ToList();
        string[] heal = { "RELIC.BURNING_BLOOD", "RELIC.BLACK_BLOOD", "RELIC.MEAT_ON_THE_BONE", "RELIC.CHOSEN_CHEESE" };
        sets.Add(new() { heal[0], heal[2] }); sets.Add(new() { heal[2], heal[0] }); sets.Add(new() { heal[1], heal[2] });
        sets.Add(new() { heal[0], heal[2], heal[3] }); sets.Add(new() { heal[3], heal[2], heal[1] }); sets.Add(new() { heal[0], heal[3] });
        sets.Add(new());
        const int maxHp = 80;
        int[] hps = { 1, 5, 30, 39, 40, 41, 50, 70, 74, 75, 79, 80 };
        long cases = 0, pass = 0, fail = 0, notWon = 0, loadFail = 0, healed = 0;
        var errors = new Dictionary<string, int>();
        var failures = new List<string>();
        var changed = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var set in sets)
        {
            foreach (int h in hps)
            {
                // relics whose effect does not depend on HP: a few HP levels are enough
                if (set.Count == 1 && !heal.Contains(set[0]) && h is not (1 or 40 or 80)) continue;
                cases++;
                try
                {
                    var (won, endHp, endMax, relics, afterHp, afterMax) = await WinAtAsync(set, h, maxHp);
                    if (!won) { notWon++; continue; }
                    var (eh, em) = Apply(relics, endHp, endMax);
                    if (eh != endHp || em != endMax) healed++;
                    if (afterHp != endHp || afterMax != endMax) changed.Add(string.Join("+", set));
                    if (eh == afterHp && em == afterMax) pass++;
                    else
                    {
                        fail++;
                        if (failures.Count < 20) failures.Add($"{string.Join("+", set)} end {endHp}/{endMax}: simulator {afterHp}/{afterMax}, table {eh}/{em}");
                    }
                }
                catch (BattleImportException e) { loadFail++; errors[$"import {e.Cause}"] = errors.GetValueOrDefault($"import {e.Cause}") + 1; }
                catch (Exception e)
                {
                    string k = e.GetType().Name + ": " + (e.Message.Length > 80 ? e.Message[..80] : e.Message);
                    errors[k] = errors.GetValueOrDefault(k) + 1;
                }
            }
        }
        Console.WriteLine($"healtest: {sets.Count} relic sets ({relicIds.Count} relics alone + 7 combinations + none), {cases} cases: " +
            $"pass {pass}, FAIL {fail}, not won {notWon}, import failures {loadFail}, errors {errors.Values.Sum()}; cases the table heals {healed}");
        Console.WriteLine("relic sets whose post-combat hooks changed HP or max HP: " + string.Join(", ", changed));
        foreach (string f in failures) Console.WriteLine("  FAIL " + f);
        foreach (var kv in errors.OrderByDescending(kv => kv.Value).Take(10)) Console.WriteLine($"  error x{kv.Value}: {kv.Key}");
        bool ok = fail == 0 && pass > 0 && healed > 0;
        Console.WriteLine(ok ? "HEALTEST PASSED" : "HEALTEST FAILED");
        return ok ? 0 : 1;
    }

    private static async Task<(bool Won, int EndHp, int EndMax, IReadOnlyList<RelicModel> Relics, int AfterHp, int AfterMax)>
        WinAtAsync(List<string> relics, int hp, int maxHp)
    {
        BattleImportException? last = null;
        foreach (string enc in new[] { "SHRINKER_BEETLE_WEAK", "NIBBITS_WEAK", "AXEBOTS_NORMAL" })
            foreach (string act in new[] { "ACT.OVERGROWTH", "ACT.UNDERDOCKS" })
            {
                var spec = new BattleStartSpec
                {
                    Id = "healtest", Character = "CHARACTER.IRONCLAD", Ascension = 0, ActIds = new() { act, "ACT.HIVE", "ACT.GLORY" },
                    ActIndex = 0, Floor = 3, RoomType = "monster", EncounterId = "ENCOUNTER." + enc, Hp = maxHp, MaxHp = maxHp,
                    Deck = Enumerable.Repeat(new BattleCard("CARD.STRIKE_IRONCLAD", 0, null, 0), 10).ToList(), Relics = relics,
                    PotionSlots = 3,
                };
                var policy = new WinPolicy(hp);
                try
                {
                    CombatRoom room = await BattleStartLoader.PlayLiveAsync(spec, "HEALTEST", policy);
                    Player p = room.Engine.State.Players[0];
                    if (policy.EndHp is null) return (false, 0, 0, p.Relics, 0, 0);
                    return (policy.Won, policy.EndHp.Value, policy.EndMax, policy.Relics!, p.Creature.CurrentHp, p.Creature.MaxHp);
                }
                catch (BattleImportException e) { last = e; }
            }
        throw last!;
    }

    /// <summary>At its first decision: every enemy at 1 HP, the player at the test HP and 10 energy; then Strikes at the
    /// first living enemy until the fight is won. Records the end HP when the fight is over (before the room resolves).</summary>
    private sealed class WinPolicy(int hp) : IRunDecisionSource, ICombatCompletionValidator
    {
        private bool _set;
        public int? EndHp;
        public int EndMax;
        public bool Won;
        public IReadOnlyList<RelicModel>? Relics;

        public Task<Sts2Sim.Core.Map.MapPoint> ChooseMapPointAsync(IReadOnlyList<Sts2Sim.Core.Map.MapPoint> options) => Task.FromResult(options[0]);

        public Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request) =>
            Task.FromResult(CardSelectionOptions.Enumerate(request)[0].Apply(request));

        public Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
        {
            Player p = state.Players[0];
            if (!_set)
            {
                _set = true;
                foreach (var e in state.Enemies) if (e.IsAlive) e.SetCurrentHpInternal(1);
                p.Creature.SetCurrentHpInternal(hp);
            }
            p.PlayerCombatState!.Energy = 10;
            var target = state.Enemies.FirstOrDefault(e => e.IsAlive && e.IsHittable);
            var card = p.PlayerCombatState.Hand.Cards.FirstOrDefault(c => c.Id.Entry == "STRIKE_IRONCLAD" && c.CanPlay(out _));
            if (target is null || card is null) return Task.FromResult<CombatDecision>(new CombatDecision.EndTurn());
            return Task.FromResult<CombatDecision>(new CombatDecision.PlayCard(card, target));
        }

        public void ValidateCompletedCombat(CombatState state)
        {
            var pc = state.Players[0].Creature;
            EndHp = Math.Max(0, pc.CurrentHp); EndMax = pc.MaxHp;
            Won = state.Engine is { IsInProgress: false, Won: true } && EndHp > 0;
            Relics = state.Players[0].Relics.ToList();
        }
    }
}
