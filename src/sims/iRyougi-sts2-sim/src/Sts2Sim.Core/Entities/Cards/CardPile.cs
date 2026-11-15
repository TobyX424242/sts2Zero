using Sts2Sim.Core.Models;
using Sts2Sim.Core.Random;

namespace Sts2Sim.Core.Entities.Cards;

/// <summary>
/// 一种牌堆容器。逐字移植增删/重排核心（<c>MegaCrit.Sts2.Core.Entities.Cards.CardPile</c>）；
/// <c>Get(PileType, Player)</c> 静态查找放在 <c>CardPileCmd</c>（Task 4）而非这里,避免 CardPile
/// 反向依赖 Player。
/// </summary>
/// <remarks>
/// S0 patch (src/s0/iryougi_s0.patch): a draw pile also tracks how many cards at its top and at its
/// bottom sit at positions the player can know (cards placed there deterministically since the last random
/// reorder). The counters never change game behaviour; a resampling search clone keeps those cards in place
/// and reshuffles only the cards in between. Rules: a random reorder or a random-position insert forgets all
/// knowledge; a deterministic insert at/inside the known top run extends it, an append (or insert inside the
/// known bottom run) extends the bottom run; an insert into the unknown middle is not tracked (conservative);
/// removals shrink the run they come from.
/// </remarks>
public sealed class CardPile(PileType type)
{
    private readonly List<CardModel> _cards = new();
    private int _knownTop;
    private int _knownBottom;

    public const int MaxCardsInHand = 10;

    public PileType Type { get; } = type;

    public IReadOnlyList<CardModel> Cards => _cards;

    /// <summary>S0: number of cards at the top (index 0 upward) whose order the player knows. Draw pile only.</summary>
    public int KnownTopCount => _knownTop;

    /// <summary>S0: number of cards at the bottom (last index downward) whose order the player knows.</summary>
    public int KnownBottomCount => _knownBottom;

    public void AddInternal(CardModel card, int index = -1) => AddInternal(card, index, randomPosition: false);

    /// <summary>S0: <paramref name="randomPosition"/> marks an insert whose index came from an RNG.</summary>
    public void AddInternal(CardModel card, int index, bool randomPosition)
    {
        if (card.Pile != null)
        {
            card.Pile.RemoveInternal(card);
        }

        int before = _cards.Count;
        if (index < 0 || index > _cards.Count)
        {
            _cards.Add(card);
            index = before;
        }
        else
        {
            _cards.Insert(index, card);
        }
        card.AssignPileInternal(this);
        if (Type == PileType.Draw) NoteInsert(index, before, randomPosition);
    }

    public void RemoveInternal(CardModel card)
    {
        int index = _cards.IndexOf(card);
        if (index < 0) return;
        int before = _cards.Count;
        _cards.RemoveAt(index);
        if (Type == PileType.Draw) NoteRemove(index, before);
        if (card.Pile == this)
        {
            card.AssignPileInternal(null);
        }
    }

    public void MoveToBottomInternal(CardModel card)
    {
        RemoveForMove(card);
        int before = _cards.Count;
        _cards.Add(card);
        if (Type == PileType.Draw) NoteInsert(before, before, randomPosition: false);
    }

    public void MoveToTopInternal(CardModel card)
    {
        RemoveForMove(card);
        int before = _cards.Count;
        _cards.Insert(0, card);
        if (Type == PileType.Draw) NoteInsert(0, before, randomPosition: false);
    }

    /// <summary>开局使用不稳定洗牌；普通重洗由 CardPileCmd.Shuffle 的 StableShuffle 处理。</summary>
    public void RandomizeOrderInternal(Rng rng)
    {
        List<CardModel> list = _cards.ToList();
        rng.Shuffle(list);
        _cards.Clear();
        _cards.AddRange(list);
        ResetPositionKnowledgeInternal();
    }

    /// <summary>S0: the current order came from a random shuffle; the player knows no position.</summary>
    public void ResetPositionKnowledgeInternal()
    {
        _knownTop = 0;
        _knownBottom = 0;
    }

    /// <summary>S0: copies the knowledge counters of a pile with the same card order (combat clone).</summary>
    internal void CopyPositionKnowledgeFrom(CardPile source)
    {
        _knownTop = source._knownTop;
        _knownBottom = source._knownBottom;
    }

    /// <summary>S0: reshuffles the cards between the known top run and the known bottom run (Fisher-Yates
    /// with <paramref name="rng"/>). Known cards keep their positions; counters are unchanged.</summary>
    internal void ShuffleUnknownRegionInternal(Rng rng)
    {
        int lo = _knownTop, hi = _cards.Count - _knownBottom; // [lo, hi) is unknown
        for (int i = hi - 1; i > lo; i--)
        {
            int j = lo + rng.NextInt(i - lo + 1);
            (_cards[i], _cards[j]) = (_cards[j], _cards[i]);
        }
    }

    private void RemoveForMove(CardModel card)
    {
        int index = _cards.IndexOf(card);
        if (index < 0) return;
        int before = _cards.Count;
        _cards.RemoveAt(index);
        if (Type == PileType.Draw) NoteRemove(index, before);
    }

    private void NoteInsert(int index, int before, bool randomPosition)
    {
        if (randomPosition)
        {
            _knownTop = 0;
            _knownBottom = 0;
        }
        else if (index <= _knownTop)
        {
            _knownTop++;
        }
        else if (index >= before - _knownBottom)
        {
            _knownBottom++;
        }
    }

    private void NoteRemove(int index, int before)
    {
        if (index < _knownTop) _knownTop--;
        else if (index >= before - _knownBottom) _knownBottom--;
    }
}
