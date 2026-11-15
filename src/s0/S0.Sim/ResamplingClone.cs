using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Cards;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Entities.Rngs;
using Sts2Sim.Core.Random;

namespace Sts2Sim.S0;

/// <summary>
/// Combat clones for search. <see cref="Exact"/> is iRyougi's own clone (the clone sees the true future:
/// draw order and every random roll). <see cref="Resampled"/> returns a clone in which everything the
/// player cannot observe is drawn afresh from <paramref name="seed"/>:
/// <list type="bullet">
/// <item>every run-level stream of the clone's <c>RunRngSet</c> (shuffle, monster AI, card generation, card
/// selection, energy costs, targets, potions, orbs, niche, ...);</item>
/// <item>every per-player stream (<c>PlayerRngSet</c>: rewards, shops, transformations);</item>
/// <item>the private <c>Rng</c> of every monster in the fight;</item>
/// <item>the order of the draw pile, except the cards whose position the player knows: the known top and
/// known bottom runs tracked by the S0 patch in <c>CardPile</c> (cards put on top/bottom deterministically
/// since the last shuffle, and Innate cards at combat start).</item>
/// </list>
/// Observable state (piles as seen, HP, block, powers, shown intents, energy, orbs, potions, relics) is
/// untouched. Reseeding is done in place on the clone's own generator objects (counters are kept), so every
/// reference the clone graph holds to a generator sees the new stream.
/// </summary>
public static class ResamplingClone
{
    public static CombatState Exact(CombatState state) => state.Clone();

    public static CombatState Resampled(CombatState state, ulong seed) => Resampled(state, seed, out _);

    public static CombatState Resampled(CombatState state, ulong seed, out CombatCloneMap map)
    {
        CombatState clone = state.Clone(out map);
        ResampleInPlace(clone, seed);
        return clone;
    }

    /// <summary>Resamples hidden information of a search clone in place. Refuses a real (non-projection)
    /// combat so the true game can never be changed by accident.</summary>
    public static void ResampleInPlace(CombatState clone, ulong seed) => ResampleInPlace(clone, seed, shuffleDrawPile: true);

    /// <param name="shuffleDrawPile">false only for the leak test's negative control (RNG reseeded, draw order kept).</param>
    public static void ResampleInPlace(CombatState clone, ulong seed, bool shuffleDrawPile)
    {
        ArgumentNullException.ThrowIfNull(clone);
        if (!clone.IsProjection)
            throw new InvalidOperationException("Only a search clone (CombatState.Clone) can be resampled.");
        var runRng = clone.RunState.Rng;
        if (runRng.UsesSemanticKeys)
            throw new NotSupportedException("Keyed (semantic-key) RNG mode cannot be resampled by stream.");

        foreach (RunRngType type in RunTypes)
            Reseed(runRng.GetRng(type), Mix(seed, 0x100 + (ulong)type));

        for (int p = 0; p < clone.Players.Count; p++)
        {
            Player player = clone.Players[p];
            foreach (PlayerRngType type in PlayerTypes)
                Reseed(player.PlayerRng.GetRng(type), Mix(seed, 0x200 + 0x10 * (ulong)p + (ulong)type));
            if (shuffleDrawPile && player.PlayerCombatState is { } pcs)
                pcs.DrawPile.ShuffleUnknownRegionInternal(new Rng(Mix(seed, 0x300 + (ulong)p)));
        }

        foreach (Creature creature in clone.Allies.Concat(clone.Enemies))
            if (creature.Monster is { Rng: { } monsterRng })
                Reseed(monsterRng, Mix(seed, 0x400 + (ulong)(creature.CombatId ?? 0)));
    }

    /// <summary>Number of draw-pile cards whose order the clone keeps fixed (top run, bottom run).</summary>
    public static (int Top, int Bottom, int Unknown) DrawPileKnowledge(Player player)
    {
        CardPile pile = player.PlayerCombatState!.DrawPile;
        return (pile.KnownTopCount, pile.KnownBottomCount,
            pile.Cards.Count - pile.KnownTopCount - pile.KnownBottomCount);
    }

    private static readonly RunRngType[] RunTypes = Enum.GetValues<RunRngType>();
    private static readonly PlayerRngType[] PlayerTypes = Enum.GetValues<PlayerRngType>();

    private static void Reseed(Rng rng, ulong newSeed)
    {
        var fresh = new Rng(newSeed).ToSerializable();
        fresh.counter = rng.Counter; // keep the visible draw count; only the generator state changes
        rng.LoadFromSerializable(fresh);
    }

    /// <summary>SplitMix64 of (seed, salt).</summary>
    public static ulong Mix(ulong seed, ulong salt)
    {
        ulong z = seed + 0x9E3779B97F4A7C15UL * (salt + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}
