using System.Collections;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.Runs;

namespace Sts2Sim.Core.Hooks;

/// <summary>
/// Perf only: allocation-free, pattern-based enumeration of <c>runState.IterateHookListeners(combatState)</c>
/// (optionally behind the <c>IterateCombatHookListeners</c> end-of-combat gate) with the same listeners, order and
/// snapshot timing as the iterators it replaces. Every step happens inside <see cref="Enumerator.MoveNext"/>,
/// exactly where the iterator body would run it:
/// <list type="bullet">
/// <item>gate (when requested): <c>IsOverOrEnding() &amp;&amp; !IsStarting()</c> at the first MoveNext;</item>
/// <item>players: the run's own <c>List&lt;Player&gt;</c> enumerator (same version checks);</item>
/// <item>relics: <see cref="RunState"/> snapshots them when the player is reached (as <c>Relics.ToArray()</c>);
/// the combat clone's run snapshot enumerates them live (same <c>List</c> enumerator); <c>IsMelted</c> is read
/// when each relic is reached;</item>
/// <item>potions: the live <c>List</c> enumerator, non-null slots (as <c>OfType&lt;PotionModel&gt;()</c>);</item>
/// <item>combat listeners: <see cref="CombatState.AppendHookListeners"/> when enumeration reaches them, after all
/// players' relics and potions have been yielded.</item>
/// </list>
/// Any other run/combat implementation, and a null combat state, use the original iterator unchanged.
/// With a <see cref="HookSlot"/>, listeners that <see cref="HookSkipIndex.CanSkip"/> are passed over (decided when
/// each one is reached); the snapshots themselves are unchanged.
/// Pooled buffers go back to <see cref="HookListenerBuffers"/> when a segment ends or on Dispose (foreach).
/// </summary>
internal readonly struct HookListenerSequence(IRunState runState, ICombatState? combatState, bool gate,
    HookSlot slot = HookSlot.None) : IEnumerable<AbstractModel>
{
    public Enumerator GetEnumerator() => new(runState, combatState, gate, slot);

    IEnumerator<AbstractModel> IEnumerable<AbstractModel>.GetEnumerator() => GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    internal struct Enumerator : IEnumerator<AbstractModel>
    {
        private enum Phase : byte { Start, Players, Relics, Potions, Combat, Fallback, Done }

        private readonly IRunState _run;
        private readonly ICombatState? _combatState;
        private readonly bool _gate;
        private readonly HookSlot _slot;
        private Phase _phase;
        private CombatState? _combat;
        private bool _snapshotRelics;
        private List<Player>.Enumerator _players;
        private List<AbstractModel>? _relicBuffer;
        private int _index;
        private List<RelicModel>.Enumerator _liveRelics;
        private List<PotionModel?>.Enumerator _potions;
        private List<AbstractModel>? _combatBuffer;
        private IEnumerator<AbstractModel>? _fallback;
        private AbstractModel? _current;

        internal Enumerator(IRunState runState, ICombatState? combatState, bool gate, HookSlot slot)
        {
            _run = runState;
            _combatState = combatState;
            _gate = gate;
            _slot = slot;
        }

        private readonly bool Skip(AbstractModel? model) =>
            _slot != HookSlot.None && model is not null && HookSkipIndex.CanSkip(model, _slot);

        public readonly AbstractModel Current => _current!;

        readonly object IEnumerator.Current => _current!;

        public bool MoveNext()
        {
            while (true)
            {
                switch (_phase)
                {
                    case Phase.Start:
                        if (_gate && _combatState!.IsOverOrEnding() && !_combatState.IsStarting())
                        {
                            _phase = Phase.Done;
                            return false;
                        }
                        List<Player>? players = null;
                        if (_combatState is CombatState combat)
                        {
                            if (_run is RunState run)
                            {
                                players = run.PlayerList;
                                _snapshotRelics = true;
                            }
                            else
                            {
                                players = CombatState.SnapshotPlayerList(_run);
                            }
                            _combat = combat;
                        }
                        if (players is null)
                        {
                            _fallback = _run.IterateHookListeners(_combatState).GetEnumerator();
                            _phase = Phase.Fallback;
                            continue;
                        }
                        _players = players.GetEnumerator();
                        _phase = Phase.Players;
                        continue;

                    case Phase.Players:
                        if (!_players.MoveNext())
                        {
                            _combatBuffer = HookListenerBuffers.Rent();
                            _combat!.AppendHookListeners(_combatBuffer, includePlayerInventory: false);
                            _index = 0;
                            _phase = Phase.Combat;
                            continue;
                        }
                        Player player = _players.Current;
                        if (_snapshotRelics)
                        {
                            _relicBuffer = HookListenerBuffers.Rent();
                            IReadOnlyList<RelicModel> relics = player.Relics;
                            for (int i = 0; i < relics.Count; i++)
                                _relicBuffer.Add(relics[i]);
                            _index = 0;
                        }
                        else
                        {
                            _liveRelics = player.RelicList.GetEnumerator();
                        }
                        _phase = Phase.Relics;
                        continue;

                    case Phase.Relics:
                        if (_snapshotRelics)
                        {
                            while (_index < _relicBuffer!.Count)
                            {
                                var relic = (RelicModel)_relicBuffer[_index++];
                                if (!relic.IsMelted && !Skip(relic))
                                {
                                    _current = relic;
                                    return true;
                                }
                            }
                            HookListenerBuffers.Return(_relicBuffer);
                            _relicBuffer = null;
                        }
                        else
                        {
                            while (_liveRelics.MoveNext())
                            {
                                RelicModel relic = _liveRelics.Current;
                                if (!relic.IsMelted && !Skip(relic))
                                {
                                    _current = relic;
                                    return true;
                                }
                            }
                        }
                        _potions = _players.Current.PotionSlotList.GetEnumerator();
                        _phase = Phase.Potions;
                        continue;

                    case Phase.Potions:
                        while (_potions.MoveNext())
                        {
                            if (_potions.Current is { } potion && !Skip(potion))
                            {
                                _current = potion;
                                return true;
                            }
                        }
                        _phase = Phase.Players;
                        continue;

                    case Phase.Combat:
                        while (_index < _combatBuffer!.Count)
                        {
                            AbstractModel listener = _combatBuffer[_index++];
                            if (Skip(listener)) continue;
                            _current = listener;
                            return true;
                        }
                        HookListenerBuffers.Return(_combatBuffer);
                        _combatBuffer = null;
                        _phase = Phase.Done;
                        return false;

                    case Phase.Fallback:
                        while (_fallback!.MoveNext())
                        {
                            AbstractModel listener = _fallback.Current;
                            if (Skip(listener)) continue;
                            _current = listener;
                            return true;
                        }
                        _phase = Phase.Done;
                        return false;

                    default:
                        return false;
                }
            }
        }

        public void Dispose()
        {
            if (_relicBuffer is not null)
            {
                HookListenerBuffers.Return(_relicBuffer);
                _relicBuffer = null;
            }
            if (_combatBuffer is not null)
            {
                HookListenerBuffers.Return(_combatBuffer);
                _combatBuffer = null;
            }
            _fallback?.Dispose();
            _fallback = null;
            _phase = Phase.Done;
        }

        public void Reset() => throw new NotSupportedException();
    }
}
