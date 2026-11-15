using Sts2Sim.Core.Models;

namespace Sts2Sim.Core.Hooks;

/// <summary>
/// Per-thread free list of the snapshot buffers that hook dispatch fills before any listener runs
/// (perf only: dispatch order, snapshot timing and contents are unchanged).
/// A buffer belongs to exactly one live enumerator from <see cref="Rent"/> until that enumerator is
/// finished or disposed (iterator <c>finally</c>); <see cref="Return"/> clears it, so a pooled buffer
/// holds no model references and nothing is shared between combat states, clones or runs.
/// An enumerator that is abandoned without Dispose just leaves its buffer to the GC.
/// </summary>
internal static class HookListenerBuffers
{
    private const int MaxPooled = 64;
    private const int MaxCapacity = 4096;

    [ThreadStatic] private static Stack<List<AbstractModel>>? t_free;

    internal static List<AbstractModel> Rent()
    {
        Stack<List<AbstractModel>>? free = t_free;
        return free is not null && free.TryPop(out List<AbstractModel>? list) ? list : new List<AbstractModel>(64);
    }

    internal static void Return(List<AbstractModel> list)
    {
        list.Clear();
        if (list.Capacity > MaxCapacity) return;
        Stack<List<AbstractModel>> free = t_free ??= new Stack<List<AbstractModel>>();
        if (free.Count < MaxPooled) free.Push(list);
    }
}
