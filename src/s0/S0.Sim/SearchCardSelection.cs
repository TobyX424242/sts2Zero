using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Models;

namespace Sts2Sim.S0;

/// <summary>One legal answer to a card-selection prompt: indices into <see cref="CardSelectionRequest.Candidates"/>
/// (ascending). An empty array is "select nothing" (legal when MinCount is 0 or the prompt is cancelable).</summary>
public readonly record struct SelectionOption(int[] Indices)
{
    public IReadOnlyList<CardModel> Apply(CardSelectionRequest request) =>
        Indices.Select(i => request.Candidates[i]).ToArray();

    public override string ToString() => "[" + string.Join(",", Indices) + "]";
}

/// <summary>Raised inside a clone when a prompt is reached in <see cref="SelectionMode.Suspend"/> mode.
/// Derives from iRyougi's <see cref="DecisionSuspendedException"/>, which the engine lets propagate.
/// The clone that raised it is left mid-action and must be discarded.</summary>
public sealed class SelectionPendingException(CardSelectionRequest request, IReadOnlyList<SelectionOption> options,
    int promptIndex) : DecisionSuspendedException("A card selection is pending in a search clone.")
{
    public CardSelectionRequest Request { get; } = request;
    public IReadOnlyList<SelectionOption> Options { get; } = options;
    /// <summary>0-based index of this prompt among the prompts answered by the same source.</summary>
    public int PromptIndex { get; } = promptIndex;
}

public enum SelectionMode
{
    /// <summary>Answer with <see cref="SearchCardSelectionSource.Policy"/> (e.g. a random or network policy).</summary>
    Policy,
    /// <summary>Throw <see cref="SelectionPendingException"/> with the enumerated options (probe; the clone is discarded,
    /// the search replays the action on a fresh clone with the chosen option in <see cref="SearchCardSelectionSource.Script"/>).</summary>
    Suspend,
    /// <summary>Return an unfinished task: the engine call (PlayCardAsync, EndPlayerTurnAsync, ...) stays incomplete,
    /// <see cref="SearchCardSelectionSource.Pending"/> exposes the prompt, and <see cref="SearchCardSelectionSource.Resolve"/>
    /// continues the same clone with the chosen option.</summary>
    Pause,
}

/// <summary>
/// Card-selection decision source for search clones. iRyougi clones get a source that always throws
/// ("A non-forced combat card selection requires a configured decision source"); install this one with
/// <see cref="InstallOn"/>. Prompts are answered first from <see cref="Script"/> (pre-chosen options, in order),
/// then according to <see cref="Mode"/>.
/// Forced prompts (fewer candidates than the minimum, no candidates) never reach a decision source; the
/// simulator resolves them itself. Run-level prompts inside a clone (deck transform/upgrade/remove and bundle
/// screens, which read <c>RunState.CardSelectionSource</c>) keep iRyougi's deterministic default policy.
/// </summary>
public sealed class SearchCardSelectionSource : ICardSelectionDecisionSource
{
    public SelectionMode Mode { get; set; }
    public Func<CardSelectionRequest, IReadOnlyList<SelectionOption>, SelectionOption>? Policy { get; set; }
    public Queue<SelectionOption> Script { get; } = new();

    /// <summary>World-independent answers: each entry is the multiset of candidate equivalence keys to select
    /// (see <see cref="CardSelectionOptions.EquivalenceKey"/>). Used by open-loop search, where the same option is
    /// replayed in different resampled worlds. Consulted after <see cref="Script"/>; if the keys cannot be matched
    /// against the offered candidates the prompt falls through to <see cref="Mode"/> and <see cref="KeyMisses"/> counts it.</summary>
    public Queue<string[]> KeyScript { get; } = new();
    public int KeyMisses { get; private set; }
    public int MaxOptions { get; set; } = 256;

    /// <summary>Prompts answered so far (scripted, policy or resolved).</summary>
    public int PromptsSeen { get; private set; }
    public List<SelectionOption> History { get; } = new();

    public CardSelectionRequest? Pending { get; private set; }
    public IReadOnlyList<SelectionOption>? PendingOptions { get; private set; }
    private TaskCompletionSource<IReadOnlyList<CardModel>>? _pendingTask;

    public static SearchCardSelectionSource InstallOn(CombatState clone, SelectionMode mode,
        Func<CardSelectionRequest, IReadOnlyList<SelectionOption>, SelectionOption>? policy = null)
    {
        if (!clone.IsProjection)
            throw new InvalidOperationException("Install the search selection source on clones only.");
        var source = new SearchCardSelectionSource { Mode = mode, Policy = policy };
        clone.CardSelectionSource = source;
        return source;
    }

    public static Func<CardSelectionRequest, IReadOnlyList<SelectionOption>, SelectionOption> RandomPolicy(System.Random rng) =>
        (_, options) => options[rng.Next(options.Count)];

    public Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request)
    {
        if (Pending is not null)
            throw new InvalidOperationException("A previous selection is still pending.");
        if (Script.TryDequeue(out SelectionOption scripted))
            return Task.FromResult(Record(request, scripted));
        if (KeyScript.TryDequeue(out string[]? keys))
        {
            if (CardSelectionOptions.TryMatch(request, keys, out SelectionOption matched))
                return Task.FromResult(Record(request, matched));
            KeyMisses++;
        }
        IReadOnlyList<SelectionOption> options = CardSelectionOptions.Enumerate(request, MaxOptions);
        switch (Mode)
        {
            case SelectionMode.Policy:
                if (Policy is null) throw new InvalidOperationException("Policy mode needs a policy.");
                return Task.FromResult(Record(request, Policy(request, options)));
            case SelectionMode.Suspend:
                throw new SelectionPendingException(request, options, PromptsSeen);
            default:
                Pending = request;
                PendingOptions = options;
                _pendingTask = new TaskCompletionSource<IReadOnlyList<CardModel>>();
                return _pendingTask.Task;
        }
    }

    /// <summary>Pause mode: answers the pending prompt. The engine continues synchronously inside this call
    /// until the action finishes or the next prompt pauses it again.</summary>
    public void Resolve(SelectionOption option)
    {
        CardSelectionRequest request = Pending ?? throw new InvalidOperationException("No selection is pending.");
        TaskCompletionSource<IReadOnlyList<CardModel>> task = _pendingTask!;
        Pending = null;
        PendingOptions = null;
        _pendingTask = null;
        task.SetResult(Record(request, option));
    }

    private IReadOnlyList<CardModel> Record(CardSelectionRequest request, SelectionOption option)
    {
        PromptsSeen++;
        History.Add(option);
        return option.Apply(request);
    }
}

public static class CardSelectionOptions
{
    /// <summary>Key under which two candidates are interchangeable for a selection.</summary>
    public static string EquivalenceKey(CardModel card) =>
        $"{card.Id}#{card.CurrentUpgradeLevel}#{card.EnergyCost}#{card.Affliction?.Id}#" +
        string.Join(",", card.Enchantments.Select(e => $"{e.Id}:{e.Magnitude}").Order(StringComparer.Ordinal));

    /// <summary>
    /// All legal answers to <paramref name="request"/>, with interchangeable candidates merged (choosing either of
    /// two identical Strikes is one option). Sizes run from MinCount to MaxCount; "nothing" is added for a
    /// cancelable prompt. Bundle prompts are single choices among the bundle representatives. If there are more
    /// than <paramref name="maxOptions"/> distinct answers, a deterministic subset is returned (smallest sizes first).
    /// </summary>
    /// <summary>Sorted candidate keys of an option (its world-independent identity).</summary>
    public static string[] Keys(CardSelectionRequest request, SelectionOption option) =>
        option.Indices.Select(i => request.IsBundleSelection ? "bundle:" + string.Join("+", request.Bundles![i].Select(c => c.Id.ToString()))
            : EquivalenceKey(request.Candidates[i])).Order(StringComparer.Ordinal).ToArray();

    /// <summary>Maps a key multiset back to candidate indices of <paramref name="request"/> (false if impossible
    /// or if the size is not legal for this prompt).</summary>
    public static bool TryMatch(CardSelectionRequest request, string[] keys, out SelectionOption option)
    {
        option = default;
        int lo = request.Cancelable ? 0 : request.MinCount;
        if (keys.Length > request.MaxCount || (keys.Length < lo && !(keys.Length == 0 && request.Candidates.Count == 0))) return false;
        var used = new bool[request.Candidates.Count];
        var idx = new List<int>();
        foreach (string k in keys)
        {
            int found = -1;
            for (int i = 0; i < request.Candidates.Count && found < 0; i++)
            {
                if (used[i]) continue;
                string ck = request.IsBundleSelection ? "bundle:" + string.Join("+", request.Bundles![i].Select(c => c.Id.ToString()))
                    : EquivalenceKey(request.Candidates[i]);
                if (ck == k) found = i;
            }
            if (found < 0) return false;
            used[found] = true; idx.Add(found);
        }
        idx.Sort();
        option = new SelectionOption(idx.ToArray());
        return true;
    }

    public static IReadOnlyList<SelectionOption> Enumerate(CardSelectionRequest request, int maxOptions = 256)
    {
        // Group interchangeable candidates; an answer is a count per group (first x indices of each group).
        var groups = new List<List<int>>();
        var byKey = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < request.Candidates.Count; i++)
        {
            string key = request.IsBundleSelection ? i.ToString() : EquivalenceKey(request.Candidates[i]);
            if (!byKey.TryGetValue(key, out int g)) { g = groups.Count; byKey.Add(key, g); groups.Add(new List<int>()); }
            groups[g].Add(i);
        }
        var result = new List<SelectionOption>();
        int lo = request.MinCount, hi = request.MaxCount;
        if (request.Cancelable && lo > 0) result.Add(new SelectionOption(Array.Empty<int>()));
        var counts = new int[groups.Count];
        for (int size = lo; size <= hi && result.Count < maxOptions; size++)
            Rec(0, size);
        if (result.Count == 0) result.Add(new SelectionOption(Array.Empty<int>()));
        return result;

        void Rec(int g, int remaining)
        {
            if (result.Count >= maxOptions) return;
            if (remaining == 0)
            {
                var idx = new List<int>();
                for (int k = 0; k < groups.Count; k++)
                    for (int j = 0; j < counts[k]; j++) idx.Add(groups[k][j]);
                idx.Sort();
                result.Add(new SelectionOption(idx.ToArray()));
                return;
            }
            if (g == groups.Count) return;
            for (int x = Math.Min(remaining, groups[g].Count); x >= 0; x--)
            {
                counts[g] = x;
                Rec(g + 1, remaining - x);
                counts[g] = 0;
                if (result.Count >= maxOptions) return;
            }
        }
    }
}
