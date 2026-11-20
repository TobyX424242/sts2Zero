using System.Text.Json.Nodes;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Commands;
using Sts2Sim.Core.Entities.Creatures;
using Sts2Sim.Core.Entities.Players;
using Sts2Sim.Core.Map;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.MonsterMoves;
using Sts2Sim.Core.Rooms;
using Sts2Sim.Core.Runs;
using Sts2Sim.S0;
using Sts2Sim.S3;

namespace Sts2Sim.Live;

/// <summary>
/// The simulator's copy of the game's fight ("shadow"). Built at the fight's first decision: the run is imported
/// (<see cref="RunImport"/>), the encounter is entered through the simulator's injected-combat path
/// (RunDriver.RunOneInjectedCombatAsync, as src/s0 BattleStart does) and stopped at the player's first decision,
/// so relic and power setup run as in the game. The engine then stays live (not a clone): RunAgent's search clones it as
/// it clones the live combat in self-play, and every action the agent takes is also applied to it (monster spawns,
/// card and power counters, move state machines follow). The simulator's own draws and rolls are not the game's, so
/// before every decision everything the game shows is written over it (<see cref="Sync"/>).
/// <para>Synced from the protocol at every decision: round, player HP / max HP / block / energy / powers (id, amount),
/// the four piles (cards by id, upgrade, enchantment, this-turn cost; the game's card instance ids keep simulator card
/// objects across decisions, so in-fight card state survives), the draw pile in the game's order with no position
/// knowledge (the search reshuffles it), the potion belt by slot, every monster's HP / max HP / block / powers / next
/// move (state machine id), monsters that died (removed) or spawned in the game but not in the copy (created).</para>
/// <para>Not synced (kept from the simulator's own run of the fight): hidden power counters (PowerCounters), relic
/// counters inside the fight, monsters' move history beyond the next move, cards' in-fight properties for cards the
/// copy had to create, the per-turn play counters of the player, the known top / bottom of the draw pile, orbs and
/// stars (not Ironclad), afflictions.</para>
/// </summary>
public sealed class Shadow : IRunDecisionSource
{
    public required Imported Imp;
    public required RunAgent Agent;
    public required string Key;
    public required RoomType Room;
    public required string Encounter;
    public CombatState? State;
    public Task? DriverTask;
    private readonly TaskCompletionSource<bool> _ready = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly Dictionary<int, CardModel> _inst = new();
    private readonly Dictionary<int, Creature> _mons = new();
    /// <summary>Answers RunAgent gave to the copy's own card prompts while the last action was applied (EquivalenceKeys).</summary>
    public readonly List<string[]> SimAnswers = new();
    /// <summary>Prompt answers of the last chosen search action that the copy did not consume.</summary>
    public readonly List<string[]> PlannedKeys = new();
    public bool Broken;
    public string? BrokenWhy;
    public int Restarts;
    public readonly List<string> SyncNotes = new();

    public bool Alive => State is { Engine: { IsInProgress: true } } && !Broken;

    // ---- IRunDecisionSource of the injected fight: the first player decision captures the live state and never returns
    public Task<MapPoint> ChooseMapPointAsync(IReadOnlyList<MapPoint> options) => throw new InvalidOperationException("no map in a shadow fight");

    public Task<CombatDecision> ChooseCombatActionAsync(CombatState state)
    {
        State = state;
        _ready.TrySetResult(true);
        return new TaskCompletionSource<CombatDecision>().Task;   // the driver stays here; the controller drives the engine
    }

    public async Task<IReadOnlyList<CardModel>> ChooseCardsAsync(CardSelectionRequest request)
    {
        IReadOnlyList<CardModel> res = await Agent.ChooseCardsAsync(request);
        if (request.IsLiveCombatSelection) SimAnswers.Add(res.Select(CardSelectionOptions.EquivalenceKey).ToArray());
        return res;
    }

    /// <summary>Enters the encounter in the imported run and waits for the player's first decision. An encounter that is
    /// not in the act's pools (event fights) is entered as a custom combat room with the game's monsters.</summary>
    public async Task StartAsync(JsonObject combat)
    {
        string entry = Encounter.StartsWith("ENCOUNTER.", StringComparison.Ordinal) ? Encounter[10..] : Encounter;
        RoomType rt = Room;
        var act = Imp.Run.Act;
        bool InPool(RoomType t) => (t switch
        {
            RoomType.Elite => act.EliteEncounterCandidates, RoomType.Boss => act.BossEncounterCandidates, _ => act.MonsterEncounterCandidates,
        }).Count(e => e.IdEntry == entry) == 1;
        if (!InPool(rt) && new[] { RoomType.Monster, RoomType.Elite, RoomType.Boss }.Where(InPool).Cast<RoomType?>().FirstOrDefault() is { } alt)
        {
            SyncNotes.Add($"encounter {Encounter} injected as {alt} (game room type {rt})");
            rt = alt;
        }
        if (InPool(rt))
        {
            var driver = new RunDriver(Imp.Run, this);
            DriverTask = driver.RunOneInjectedCombatAsync(rt, entry);
            await Task.WhenAny(_ready.Task, DriverTask);
        }
        else
        {
            // custom room with the game's monsters (alive ones, in protocol order, with their slots)
            var mons = new List<(MonsterModel Monster, string? SlotName)>();
            foreach (JsonObject m in combat.Objs("monsters").Where(m => m.Bool("alive", m.Int("hp") > 0)))
            {
                string mid = J.ModelId(m.Str("id"), "MONSTER");
                MonsterModel canon = ModelDb.GetByIdOrNull<MonsterModel>(ModelId.Deserialize(mid))
                    ?? throw new LiveException($"monster {mid} of {Encounter} is not in the simulator");
                mons.Add(((MonsterModel)canon.MutableClone(), m.Str("slot")));
            }
            if (mons.Count == 0) throw new LiveException($"fight {Encounter}: no living monster to build it from");
            var room = new CombatRoom(() => (IReadOnlyList<(MonsterModel, string?)>)mons, rt, string.IsNullOrEmpty(entry) ? CombatRoom.CustomEncounterName : entry);
            Priv.Call(room, "ConfigureCardSelectionSource", this);
            Imp.Run.PushRoom(room);
            await room.Enter(Imp.Run);
            State = room.Engine.State;
            SyncNotes.Add($"encounter {Encounter} is not in the act's pools: custom room with the game's monsters");
        }
        if (State is null)
        {
            string why = DriverTask is { IsFaulted: true } ? DriverTask.Exception!.GetBaseException().Message : "the fight ended before the first decision";
            throw new LiveException($"shadow fight {Encounter}: {why}");
        }
    }

    // ------------------------------------------------------------------ sync

    private static string Pid(string? id) => J.ModelId(id, "POWER");

    /// <summary>Writes the game's fight (protocol combat object, run belt) over the copy. Returns notes about what could
    /// not be synced.</summary>
    public List<string> Sync(JsonObject combat, JsonObject? run)
    {
        var notes = new List<string>();
        CombatState s = State!;
        Player p = s.Players[0];
        Creature pc = p.Creature;
        PlayerCombatState pcs = p.PlayerCombatState!;
        JsonObject pl = combat.Obj("player") ?? new JsonObject();
        s.CurrentSide = CombatSide.Player;
        if (combat.IntOrNull("round") is { } round) s.RoundNumber = round;
        SyncCreature(pc, pl, notes, "player");
        if (pl.IntOrNull("energy") is { } en) pcs.Energy = en;
        if (pl.IntOrNull("stars") is { } stars && stars != pcs.Stars) Priv.SetProp(pcs, "Stars", stars);

        // ---- piles
        var all = pcs.Hand.Cards.Concat(pcs.DrawPile.Cards).Concat(pcs.DiscardPile.Cards).Concat(pcs.ExhaustPile.Cards).Concat(pcs.PlayPile.Cards).ToList();
        var allSet = new HashSet<CardModel>(all, ReferenceEqualityComparer.Instance);
        var used = new HashSet<CardModel>(ReferenceEqualityComparer.Instance);
        var mapped = new HashSet<CardModel>(_inst.Values, ReferenceEqualityComparer.Instance);
        int created = 0, upgraded = 0, costs = 0;
        CardModel Resolve(JsonObject c, IReadOnlyList<CardModel> samePile)
        {
            CardSpec spec = CardSpec.Read(c);
            int? inst = c.IntOrNull("instance");
            CardModel? m = null;
            if (inst is { } i && _inst.TryGetValue(i, out CardModel? known) && allSet.Contains(known) && !used.Contains(known))
            {
                if (CardSpec.KeyOf(known) == spec.Key) m = known;
                else if (known.Id.ToString() == spec.Id && known.CurrentUpgradeLevel < spec.Upgrade)
                {
                    while (known.CurrentUpgradeLevel < spec.Upgrade && known.IsUpgradable) { known.Upgrade(); upgraded++; }
                    if (CardSpec.KeyOf(known) == spec.Key) m = known;
                }
            }
            m ??= samePile.FirstOrDefault(x => !used.Contains(x) && !(inst is not null && mapped.Contains(x) && !ReferenceEquals(_inst.GetValueOrDefault(inst.Value), x)) && CardSpec.KeyOf(x) == spec.Key)
               ?? Best(all.Where(x => !used.Contains(x) && !mapped.Contains(x) && CardSpec.KeyOf(x) == spec.Key))
               ?? Best(all.Where(x => !used.Contains(x) && CardSpec.KeyOf(x) == spec.Key));
            // among equal cards, the one whose current cost is the game's (in-fight changes follow the card)
            CardModel? Best(IEnumerable<CardModel> xs) => xs.OrderBy(x => c.IntOrNull("cost_this_turn") is { } ct && x.EnergyCost != ct ? 1 : 0).FirstOrDefault();
            if (m is null)
            {
                try { m = RunImport.CreateCard(p, spec); created++; }
                catch (Exception e) { notes.Add($"card {spec.Id} not creatable: {e.Message}"); return null!; }
            }
            used.Add(m);
            if (inst is { } j) _inst[j] = m;
            SyncAffliction(m, c, notes);
            if (c.IntOrNull("cost_this_turn") is { } ct && ct >= 0 && !m.CostsXEnergy && m.EnergyCost != ct)
            {
                try { m.SetTemporaryCostOverrideThisTurn(ct); costs++; } catch (Exception e) { notes.Add($"cost of {spec.Id}: {e.Message}"); }
            }
            return m;
        }
        var want = new List<(Sts2Sim.Core.Entities.Cards.CardPile Pile, List<CardModel> Cards)>();
        foreach (var (name, pile) in new[] { ("hand", pcs.Hand), ("draw", pcs.DrawPile), ("discard", pcs.DiscardPile), ("exhaust", pcs.ExhaustPile) })
        {
            var cards = new List<CardModel>();
            foreach (JsonObject c in pl.Objs(name)) { CardModel m = Resolve(c, pile.Cards); if (m is not null) cards.Add(m); }
            want.Add((pile, cards));
        }
        // the copy's known top / bottom of the draw pile (cards the player put there) survive when the game's draw
        // order has the same cards at that end (the protocol's draw order: index 0 = top, as in the simulator)
        var oldDraw = pcs.DrawPile.Cards.Select(CardSpec.KeyOf).ToList();
        int kt = pcs.DrawPile.KnownTopCount, kb = pcs.DrawPile.KnownBottomCount;
        foreach (var pile in pcs.AllPiles) foreach (CardModel c in pile.Cards.ToList()) pile.RemoveInternal(c);
        foreach (var (pile, cards) in want) foreach (CardModel c in cards) pile.AddInternal(c);
        pcs.DrawPile.ResetPositionKnowledgeInternal();
        var newDraw = pcs.DrawPile.Cards.Select(CardSpec.KeyOf).ToList();
        if (kt > 0 && kt <= newDraw.Count && newDraw.Take(kt).SequenceEqual(oldDraw.Take(kt))) Priv.Set(pcs.DrawPile, "_knownTop", kt);
        if (kb > 0 && kb <= newDraw.Count && newDraw.TakeLast(kb).SequenceEqual(oldDraw.TakeLast(kb)) && kt + kb <= newDraw.Count) Priv.Set(pcs.DrawPile, "_knownBottom", kb);
        if (created > 0) notes.Add($"cards created {created}");
        if (upgraded > 0) notes.Add($"cards upgraded {upgraded}");
        if (costs > 0) notes.Add($"card costs set {costs}");

        // ---- potion belt (by slot)
        if (run is not null && run.At("potions") is JsonArray belt)
        {
            var list = (List<PotionModel?>)Priv.GetProp(p, "PotionSlotList")!;
            while (list.Count < belt.Count) list.Add(null);
            for (int k = 0; k < list.Count; k++)
            {
                string? want1 = k < belt.Count && belt[k] is JsonValue jv && jv.TryGetValue(out string? sid) && !string.IsNullOrEmpty(sid) ? J.ModelId(sid, "POTION")
                    : k < belt.Count && belt[k] is JsonObject bo ? J.ModelId(bo.Str("id"), "POTION") : null;
                if (want1 == "") want1 = null;
                if (list[k]?.Id.ToString() == want1) continue;
                if (want1 is null) { list[k] = null; continue; }
                PotionModel? canon = ModelDb.GetByIdOrNull<PotionModel>(ModelId.Deserialize(want1));
                if (canon is null) { notes.Add("unknown potion " + want1); list[k] = null; continue; }
                var pm = (PotionModel)canon.MutableClone();
                pm.AssignOwner(p);
                list[k] = pm;
            }
        }

        // ---- monsters
        // monsters: the protocol index is the position in the game's Enemies list (it shifts when the game removes a
        // dead monster), so each sync matches by id + slot, then by order among equal ids; _mons maps this state's
        // indices to the copy's creatures (targets of this decision)
        var ms = combat.Objs("monsters").ToList();
        var prev = new Dictionary<int, Creature>(_mons);
        _mons.Clear();
        var matched = new HashSet<Creature>(ReferenceEqualityComparer.Instance);
        for (int pos = 0; pos < ms.Count; pos++)
        {
            JsonObject m = ms[pos];
            int idx = m.Int("index", pos);
            string mid = J.ModelId(m.Str("id"), "MONSTER");
            string? slot = m.Str("slot");
            bool alive = m.Bool("alive", m.Int("hp") > 0);
            bool Free(Creature e) => !matched.Contains(e) && e.Monster?.Id.ToString() == mid;
            Creature? c = !string.IsNullOrEmpty(slot) ? s.Enemies.FirstOrDefault(e => Free(e) && e.SlotName == slot) : null;
            c ??= s.Enemies.FirstOrDefault(e => Free(e) && string.IsNullOrEmpty(slot) == string.IsNullOrEmpty(e.SlotName)) ?? s.Enemies.FirstOrDefault(Free);
            if (c is null && alive)   // spawned in the game, not in the copy
            {
                MonsterModel? canon = ModelDb.GetByIdOrNull<MonsterModel>(ModelId.Deserialize(mid));
                if (canon is null) { notes.Add("unknown monster " + mid); continue; }
                var mon = (MonsterModel)canon.MutableClone();
                c = s.AddMonster(mon, CombatSide.Enemy, slot);
                mon.SetUpForCombat();
                notes.Add("monster created " + mid);
            }
            if (c is null) continue;
            matched.Add(c);
            _mons[idx] = c;
            if (!alive && string.IsNullOrEmpty(m.Str("move_id")))
            {
                if (s.Enemies.Contains(c)) { if (c.IsAlive) c.SetCurrentHpInternal(0); s.RemoveCreature(c); }
                continue;
            }
            // a dead monster that still has a move (revive / reattach: Decimillipede, Test Subject, ...) stays at 0 HP
            SyncCreature(c, m, notes, mid);
            if (!alive && c.IsAlive) c.SetCurrentHpInternal(0);
            string? move = m.Str("move_id");
            if (!string.IsNullOrEmpty(move) && c.Monster is { MoveStateMachine: { } sm } mm && mm.NextMove?.Id != move)
            {
                if (sm.States.TryGetValue(move, out MonsterState? st) && st is MoveState mv) mm.SetMoveImmediate(mv, forceTransition: true);
                else notes.Add($"move {move} of {mid} not in the simulator's state machine");
            }
        }
        foreach (Creature e in s.Enemies.ToList())
            if (!matched.Contains(e)) { s.RemoveCreature(e); notes.Add("monster removed " + e.Monster?.Id); }
        SyncNotes.Clear();
        SyncNotes.AddRange(notes);
        return notes;
    }

    /// <summary>A card's affliction (BOUND, HEXED, ...): protocol CARD "affliction" ({id, amount} or an id with
    /// "affliction_amount").</summary>
    private static void SyncAffliction(CardModel m, JsonObject c, List<string> notes)
    {
        string? id = c.At("affliction") is JsonObject ao ? ao.Str("id") : c.Str("affliction");
        int amount = c.At("affliction") is JsonObject ao2 ? ao2.Int("amount") : c.Int("affliction_amount");
        if (string.IsNullOrEmpty(id)) id = null; else id = J.ModelId(id, "AFFLICTION");
        if (m.Affliction?.Id.ToString() == id)
        {
            if (m.Affliction is { } a && a.Amount != amount && c.At("affliction") is JsonObject) Priv.Call(a, "SetAmount", (decimal)amount);
            return;
        }
        if (m.Affliction is not null) Priv.Call(m, "ClearAfflictionInternal");
        if (id is null) return;
        var canon = ModelDb.GetByIdOrNull<Sts2Sim.Core.Models.Afflictions.AfflictionModel>(ModelId.Deserialize(id));
        if (canon is null) { notes.Add("unknown affliction " + id); return; }
        Priv.Call(m, "AttachAffliction", canon.MutableClone(), (decimal)amount);
    }

    private static void SyncCreature(Creature c, JsonObject o, List<string> notes, string who)
    {
        if (o.IntOrNull("max_hp") is { } mx && mx > 0 && mx != c.MaxHp) c.SetMaxHpInternal(mx);
        if (o.IntOrNull("hp") is { } hp && hp != c.CurrentHp) c.SetCurrentHpInternal(hp);
        if (o.IntOrNull("block") is { } b && b != c.Block)
        {
            if (b > c.Block) c.GainBlockInternal(b - c.Block); else c.LoseBlockInternal(c.Block - b);
        }
        if (o.At("powers") is not JsonArray pw) return;
        var want = pw.OfType<JsonObject>().Select(x => (Id: Pid(x.Str("id")), Amount: x.Int("amount"))).ToList();
        var have = c.Powers.ToList();
        var keep = new HashSet<PowerModel>(ReferenceEqualityComparer.Instance);
        var order = new List<PowerModel>();
        foreach (var (id, amount) in want)
        {
            PowerModel? cur = have.FirstOrDefault(x => x.Id.ToString() == id && !keep.Contains(x));
            if (cur is null)
            {
                PowerModel? canon = ModelDb.GetByIdOrNull<PowerModel>(ModelId.Deserialize(id));
                if (canon is null) { notes.Add($"unknown power {id} on {who}"); continue; }
                cur = (PowerModel)canon.MutableClone();
                cur.ApplyInternal(c, amount);
            }
            else if (cur.Amount != amount) cur.SetAmount(amount);
            keep.Add(cur); order.Add(cur);
        }
        foreach (PowerModel x in have) if (!keep.Contains(x)) x.RemoveInternal();
        // the game's order (hooks fire in list order)
        if (!c.Powers.SequenceEqual(order))
            foreach (PowerModel x in order) { c.RemovePowerInternal(x); c.ApplyPowerInternal(x); }
    }

    // ------------------------------------------------------------------ mapping

    /// <summary>Protocol creature index of a copy creature (monsters by the sync map; the player: player.index if given).</summary>
    public int? IndexOf(Creature? c, JsonObject combat)
    {
        if (c is null) return null;
        foreach (var (k, v) in _mons) if (ReferenceEquals(v, c)) return k;
        if (State is { } s && ReferenceEquals(c, s.Players[0].Creature)) return combat.Obj("player")?.IntOrNull("index");
        return null;
    }

    /// <summary>The copy's creature for a protocol monster index (after <see cref="Sync"/>).</summary>
    public Creature? CreatureAt(int index) => _mons.TryGetValue(index, out Creature? c) ? c : null;

    /// <summary>Applies the agent's decision to the copy (as RunDriver.DriveCombatAsync does in self-play). A failure
    /// marks the copy broken; it is rebuilt at the next decision.</summary>
    public async Task ApplyAsync(CombatDecision d)
    {
        if (!Alive) return;
        CombatEngine engine = State!.Engine!;
        Player player = State.Players[0];
        SimAnswers.Clear();
        try
        {
            switch (d)
            {
                case CombatDecision.PlayCard pc:
                    await engine.PlayCardAsync(player, pc.Card, pc.Target);
                    engine.CheckWinCondition();
                    break;
                case CombatDecision.UsePotion up:
                    await PotionCmd.Use(up.Potion, player, up.Target);
                    engine.CheckWinCondition();
                    break;
                case CombatDecision.EndTurn:
                    if (engine.IsInProgress) await engine.EndPlayerTurnAsync();
                    break;
            }
        }
        catch (Exception e) { Broken = true; BrokenWhy = e.GetType().Name + ": " + e.Message; }
    }
}

public sealed class LiveException(string message) : Exception(message);
