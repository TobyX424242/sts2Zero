using System.Collections.Concurrent;
using System.Reflection;
using Sts2Sim.Core.Entities.Relics;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Models.Powers;
using Sts2Sim.Core.Models.Relics;
using CardFace = Sts2Sim.Core.Models.Cards.CardFaceBaseValues;
using WitherCard = Sts2Sim.Core.Models.Cards.Wither;

namespace Sts2Sim.S0;

/// <summary>
/// Current damage / block printed on a card (layout v4 features, observable-state fingerprint): the card's own base
/// values including upgrades and permanent or in-fight changes (Claw's growth, Genetic Algorithm, ...), without
/// Strength, Dexterity, Vulnerable, Weak or other target/owner modifiers. Source: the simulator's audited
/// <see cref="CardFace"/> ("Damage" / "Block" literal DynamicVar base values; live fields for cards whose values
/// change). Wither (Aeonglass) keeps its value in a private upgrade counter that the audit table does not read:
/// its turn-end damage <see cref="WitherCard.TurnEndDamage"/> is used. Cards without audited values give 0.
/// </summary>
public static class CardValues
{
    // types for which CardFaceBaseValues throws (no audited values): 0 without another attempt
    private static readonly ConcurrentDictionary<Type, bool> Unsupported = new();

    public static int Damage(CardModel c) => c is WitherCard w ? w.TurnEndDamage : Get(c, "Damage");

    public static int Block(CardModel c) => c is WitherCard ? 0 : Get(c, "Block");

    private static int Get(CardModel c, string key)
    {
        Type t = c.GetType();
        if (Unsupported.ContainsKey(t)) return 0;
        double v;
        try { v = CardFace.Get(c, key); }
        catch (InvalidOperationException) { Unsupported[t] = true; return 0; }
        return (int)Math.Clamp(Math.Round(v), -1e9, 1e9);
    }
}

/// <summary>
/// Hidden counters of powers (layout v4: EnemyPower f2, PlayerPower f1; observable-state fingerprint): state the game
/// shows on the power (or that decides when it fires next) but that iRyougi keeps outside <c>Amount</c>. Private
/// fields are read through <see cref="FieldInfo"/>s resolved once per type at first use (a missing field throws at
/// that point: the simulator changed and this table must be checked again).
/// <code>
/// power                counter
/// WitheringPresence    cards the player can still play before the next Wither is created (Amount - cards played)
/// Sloth                cards the owner may still play this turn (Amount - cards played this turn, at least 0)
/// Automation           cards left to draw before the energy gain (CardsLeft)
/// Cacophony            cards left to draw before the damage trigger
/// Panache              cards left to play this turn before the damage trigger
/// Orbit                energy left to spend before the energy gain (4 - energy spent)
/// Slow                 cards played since the owner's turn start (damage taken x (1 + 0.1 n))
/// Juggling             attacks played this turn (the third is copied)
/// VoidForm             free cards left this turn (Amount - cards played this turn, at least 0)
/// Tender, Feral        the game's displayed number (cards played this turn; zero-cost attacks still returned)
/// HardenedShell        the game's displayed number (HP loss still allowed this turn)
/// Monologue            the game's displayed number (Strength gained this turn)
/// Inferno, CrimsonMantle  HP lost at the start of the next turn (grows each turn)
/// </code>
/// Every other power: 0 (<see cref="TryGet"/> false).
/// </summary>
public static class PowerCounters
{
    private const BindingFlags Inst = BindingFlags.Instance | BindingFlags.NonPublic;

    private static readonly Dictionary<Type, Func<PowerModel, int>> Getters = Build();

    public static bool TryGet(PowerModel p, out int value)
    {
        if (Getters.TryGetValue(p.GetType(), out var g)) { value = g(p); return true; }
        value = 0;
        return false;
    }

    public static int Get(PowerModel p) => TryGet(p, out int v) ? v : 0;

    private static FieldInfo F<T>(string name) =>
        typeof(T).GetField(name, Inst) ?? throw new InvalidOperationException($"{typeof(T).Name}.{name} not found (simulator changed)");

    private static int I(FieldInfo f, object o) => Convert.ToInt32(f.GetValue(o));

    private static Dictionary<Type, Func<PowerModel, int>> Build()
    {
        var withering = F<WitheringPresencePower>("_cardsPlayed");
        var sloth = F<SlothPower>("_cardsPlayed");
        var cacophony = F<CacophonyPower>("_cardsLeft");
        var panache = F<PanachePower>("_cardsLeft");
        var slow = F<SlowPower>("_cardsPlayedSinceOwnerTurnStart");
        var juggling = F<JugglingPower>("_attacksPlayedThisTurn");
        var inferno = F<InfernoPower>("_selfDamage");
        var mantle = F<CrimsonMantlePower>("_selfDamage");
        int orbitPer = Convert.ToInt32((typeof(OrbitPower).GetField("EnergyPerTrigger", BindingFlags.Static | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("OrbitPower.EnergyPerTrigger not found (simulator changed)")).GetRawConstantValue());
        return new Dictionary<Type, Func<PowerModel, int>>
        {
            [typeof(WitheringPresencePower)] = p => p.Amount - I(withering, p),
            [typeof(SlothPower)] = p => Math.Max(0, p.Amount - I(sloth, p)),
            [typeof(AutomationPower)] = p => ((AutomationPower)p).CardsLeft,
            [typeof(CacophonyPower)] = p => I(cacophony, p),
            [typeof(PanachePower)] = p => I(panache, p),
            [typeof(OrbitPower)] = p => orbitPer - ((OrbitPower)p).EnergySpent,
            [typeof(SlowPower)] = p => I(slow, p),
            [typeof(JugglingPower)] = p => I(juggling, p),
            [typeof(VoidFormPower)] = p => Math.Max(0, p.Amount - Math.Min(((VoidFormPower)p).CardsPlayedThisTurn, 1_000_000_000)),
            [typeof(TenderPower)] = p => ((TenderPower)p).DisplayAmount,
            [typeof(FeralPower)] = p => ((FeralPower)p).DisplayAmount,
            [typeof(HardenedShellPower)] = p => ((HardenedShellPower)p).DisplayAmount,
            [typeof(MonologuePower)] = p => ((MonologuePower)p).DisplayAmount,
            [typeof(InfernoPower)] = p => I(inferno, p),
            [typeof(CrimsonMantlePower)] = p => I(mantle, p),
        };
    }
}

/// <summary>
/// Counters of the relics that show one on screen (layout v5: Relic F[0], Relic F[1] = 1). The simulator keeps these
/// in private fields or plain properties; the values here are the progress toward the relic's next trigger, i.e. what
/// the game's badge shows (Pen Nib / Nunchaku modulo 10, Happy Flower / Pendulum per-turn counters, attack counters for
/// Kunai / Shuriken / Ornamental Fan, cards played for Pocketwatch / Velvet Choker, and the public counters of Girya,
/// Book of Five Rings, Pael's Wing and Toy Box).
/// <para>Reflection is resolved once per type at first use; a renamed field throws at that point (the simulator changed
/// and this table must be checked again), the same contract as <see cref="PowerCounters"/>.</para>
/// </summary>
public static class RelicCounterValues
{
    private const BindingFlags Inst = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    private static readonly Dictionary<Type, Func<RelicModel, int>> Getters = Build();

    public static bool TryGet(RelicModel relic, out int value)
    {
        if (Getters.TryGetValue(relic.GetType(), out var g)) { value = g(relic); return true; }
        value = 0;
        return false;
    }

    private static FieldInfo F<T>(string name) =>
        typeof(T).GetField(name, Inst) ?? throw new InvalidOperationException($"{typeof(T).Name}.{name} not found (simulator changed)");

    private static Func<RelicModel, int> Field<T>(string name)
    {
        FieldInfo f = F<T>(name);
        return r => Convert.ToInt32(f.GetValue(r));
    }

    private static Func<RelicModel, int> FieldMod<T>(string name, int mod)
    {
        FieldInfo f = F<T>(name);
        return r => ((Convert.ToInt32(f.GetValue(r)) % mod) + mod) % mod;
    }

    private static Func<RelicModel, int> Property<T>(string name)
    {
        PropertyInfo p = typeof(T).GetProperty(name, Inst)
            ?? throw new InvalidOperationException($"{typeof(T).Name}.{name} not found (simulator changed)");
        return r => Convert.ToInt32(p.GetValue(r));
    }

    private static Dictionary<Type, Func<RelicModel, int>> Build() => new()
    {
        [typeof(PenNib)] = FieldMod<PenNib>("_attacksPlayed", 10),
        [typeof(Nunchaku)] = FieldMod<Nunchaku>("_attacksPlayed", 10),
        [typeof(HappyFlower)] = Field<HappyFlower>("_turnCounter"),
        [typeof(LetterOpener)] = Field<LetterOpener>("_skillsThisTurn"),
        [typeof(Pendulum)] = Field<Pendulum>("_turnCounter"),
        [typeof(Kunai)] = Field<Kunai>("_attacksThisTurn"),
        [typeof(Shuriken)] = Field<Shuriken>("_attacksThisTurn"),
        [typeof(OrnamentalFan)] = Field<OrnamentalFan>("_attacksThisTurn"),
        // ic113: v0.113.1 Velvet Choker counts the card plays started this turn (PlayerCombatState.CardPlaysStartedThisTurn,
        // read through the relic's private property); v0.111 kept its own _cardsPlayedThisTurn field
        [typeof(VelvetChoker)] = Property<VelvetChoker>("CardPlaysStartedThisTurn"),
        [typeof(Pocketwatch)] = Property<Pocketwatch>("CardsPlayedThisTurn"),
        [typeof(Girya)] = Property<Girya>("TimesLifted"),
        [typeof(BookOfFiveRings)] = Property<BookOfFiveRings>("DisplayAmount"),
        [typeof(PaelsWing)] = Property<PaelsWing>("DisplayAmount"),
        [typeof(ToyBox)] = Property<ToyBox>("DisplayAmount"),
    };
}
