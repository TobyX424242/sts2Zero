using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.CompilerServices;
using Sts2Sim.Core.Models;

namespace Sts2Sim.Core.Hooks;

/// <summary>Hook methods whose dispatch loops may pass over listeners that keep AbstractModel's own implementation
/// (see <see cref="HookSkipIndex"/>). Each value names the AbstractModel virtual of the same name.</summary>
internal enum HookSlot : byte
{
    None = 0,
    ShouldAllowHitting, ShouldPlay,
    ModifyDamageAdditive, ModifyDamageMultiplicative, ModifyDamageCap,
    ModifyHpLostBeforeOsty, ModifyHpLostBeforeOstyLate, ModifyHpLostAfterOsty, ModifyHpLostAfterOstyLate,
    AfterModifyingHpLostBeforeOsty, AfterModifyingHpLostAfterOsty,
    AfterCardDrawnEarly, AfterCardDrawn,
    BeforeDamageReceived, AfterDamageReceived, AfterCurrentHpChanged,
    AfterSideTurnStart, AfterSideTurnStartLate,
    ModifyPowerAmountGivenAdditive, ModifyPowerAmountGivenMultiplicative, TryModifyPowerAmountReceived,
    AfterModifyingPowerAmountReceived, BeforePowerAmountChanged, AfterPowerAmountChanged,
    ModifyBlockAdditive, ModifyBlockMultiplicative, AfterModifyingBlockAmount,
    ModifyCardPlayCount, AfterModifyingCardPlayCount, ModifyCardPlayResultLocation, ModifyOrbValue,
    AfterBlockCleared, BeforeCardPlayed, BeforeAttack,
    Count,
}

/// <summary>
/// Perf only. A listener may be passed over by a dispatch loop for <see cref="HookSlot"/> S when that loop would
/// do nothing observable with it:
/// <list type="bullet">
/// <item>its runtime type is one of the eight known hook model types (others keep the eligibility guard, which
/// throws for unsupported types);</item>
/// <item>the type's implementation of S is AbstractModel's own: the neutral constant or pass-through
/// (<c>true</c>, <c>0m</c>, <c>1m</c>, <c>decimal.MaxValue</c>, the incoming value, <c>false</c> with the
/// incoming value, or <c>Task.CompletedTask</c>), which each converted dispatcher folds as an exact no-op
/// (decimal <c>x + 0m</c> and <c>x * 1m</c> keep every bit of x, including scale and sign);</item>
/// <item>no <see cref="AbstractModel.ExecutionFinished"/> handler is attached (for slots whose loop calls
/// <c>InvokeExecutionFinished</c>).</item>
/// </list>
/// The eligibility guard is then pure and its result irrelevant, so skipping it changes nothing for well-formed
/// listeners. Type data is immutable; metadata failures make a type never skippable (fail-closed).
/// </summary>
internal static class HookSkipIndex
{
    private static readonly MethodInfo?[] BaseSlots = ResolveBaseSlots();
    private static readonly ulong NotifyingSlots = MaskOf(
        HookSlot.AfterModifyingHpLostBeforeOsty, HookSlot.AfterModifyingHpLostAfterOsty,
        HookSlot.AfterCardDrawnEarly, HookSlot.AfterCardDrawn,
        HookSlot.BeforeDamageReceived, HookSlot.AfterDamageReceived, HookSlot.AfterCurrentHpChanged,
        HookSlot.AfterSideTurnStart, HookSlot.AfterSideTurnStartLate,
        HookSlot.AfterModifyingPowerAmountReceived, HookSlot.BeforePowerAmountChanged, HookSlot.AfterPowerAmountChanged,
        HookSlot.AfterModifyingBlockAmount, HookSlot.AfterModifyingCardPlayCount,
        HookSlot.AfterBlockCleared, HookSlot.BeforeCardPlayed, HookSlot.BeforeAttack);
    private static readonly ConcurrentDictionary<Type, ulong> Masks = new();

    private const int CacheSize = 256;
    [ThreadStatic] private static Type?[]? t_types;
    [ThreadStatic] private static ulong[]? t_masks;

    internal static bool CanSkip(AbstractModel model, HookSlot slot)
    {
        ulong bit = 1UL << (int)slot;
        if ((SkippableMask(model.GetType()) & bit) == 0) return false;
        return (NotifyingSlots & bit) == 0 || !model.HasExecutionFinishedHandlers;
    }

    private static ulong SkippableMask(Type type)
    {
        Type?[]? types = t_types;
        ulong[]? masks = t_masks;
        if (types is null || masks is null)
        {
            t_types = types = new Type?[CacheSize];
            t_masks = masks = new ulong[CacheSize];
        }
        int index = RuntimeHelpers.GetHashCode(type) & (CacheSize - 1);
        if (ReferenceEquals(types[index], type)) return masks[index];
        ulong mask = Masks.GetOrAdd(type, static t => Resolve(t));
        types[index] = type;
        masks[index] = mask;
        return mask;
    }

    private static ulong MaskOf(params HookSlot[] slots)
    {
        ulong mask = 0;
        foreach (HookSlot slot in slots) mask |= 1UL << (int)slot;
        return mask;
    }

    private static MethodInfo?[] ResolveBaseSlots()
    {
        var slots = new MethodInfo?[(int)HookSlot.Count];
        MethodInfo[] declared = typeof(AbstractModel).GetMethods(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly);
        for (int i = 1; i < slots.Length; i++)
        {
            string name = ((HookSlot)i).ToString();
            MethodInfo[] candidates = Array.FindAll(declared, m => m.Name == name && m.IsVirtual);
            // A missing or overloaded name disables the slot: it is never skipped.
            slots[i] = candidates.Length == 1 ? candidates[0] : null;
        }
        return slots;
    }

    // Bit i set: type keeps AbstractModel's implementation of slot i.
    private static ulong Resolve(Type runtimeType)
    {
        try
        {
            if (!typeof(PowerModel).IsAssignableFrom(runtimeType) && !typeof(MonsterModel).IsAssignableFrom(runtimeType)
                && !typeof(OrbModel).IsAssignableFrom(runtimeType) && !typeof(CardModel).IsAssignableFrom(runtimeType)
                && !typeof(RelicModel).IsAssignableFrom(runtimeType) && !typeof(PotionModel).IsAssignableFrom(runtimeType)
                && !typeof(Models.Afflictions.AfflictionModel).IsAssignableFrom(runtimeType)
                && !typeof(EnchantmentModel).IsAssignableFrom(runtimeType))
                return 0;
            ulong mask = 0;
            for (int i = 1; i < BaseSlots.Length; i++)
            {
                MethodInfo? baseSlot = BaseSlots[i];
                if (baseSlot is null) continue;
                MethodInfo? implementation = null;
                for (Type? current = runtimeType; current is not null && implementation is null; current = current.BaseType)
                {
                    // DeclaredOnly keeps hidden new slots from obscuring inherited overrides.
                    foreach (MethodInfo method in current.GetMethods(BindingFlags.Instance |
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                    {
                        if (method.IsVirtual && method.GetBaseDefinition().MethodHandle == baseSlot.MethodHandle)
                        {
                            implementation = method;
                            break;
                        }
                    }
                }
                if (implementation is not null && implementation.MethodHandle == baseSlot.MethodHandle) mask |= 1UL << i;
            }
            return mask;
        }
        catch (Exception error) when (error is not OutOfMemoryException and not AccessViolationException)
        {
            return 0;
        }
    }
}
