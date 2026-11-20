using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace AgentBridge;

internal sealed class ActArgs
{
    public int? Target;
    public List<int>? Cards;
}

/// <summary>One legal action: its JSON for the controller, a validator for the act arguments, and the executor
/// that performs it through the game's UI path (runs on the main thread, may span several frames).</summary>
internal sealed class Option
{
    public JsonObject Json = new();
    public Func<ActArgs, string?> Validate = _ => null;
    public Func<ActArgs, Task<string?>> Exec = _ => Task.FromResult<string?>("not executable");
    public string Label = "";
}

internal sealed class Snapshot
{
    public Probe Probe = new();
    public bool Stable;
    public JsonObject? Run;
    public JsonObject? Combat;
    public List<Option> Options = new();
    public string Fingerprint = "";
}

/// <summary>Builds the protocol state (run, combat, options) from the live game. Read-only.</summary>
internal static class StateBuilder
{
    public static Snapshot Build(Probe p, bool stable, bool withOptions)
    {
        var s = new Snapshot { Probe = p, Stable = stable };
        try { s.Run = BuildRun(p); }
        catch (Exception e) { s.Run = new JsonObject { ["error"] = "run: " + e.Message }; }
        try { s.Combat = BuildCombat(p); }
        catch (Exception e) { s.Combat = new JsonObject { ["error"] = "combat: " + e.Message }; }
        if (stable && withOptions)
        {
            try { s.Options = BuildOptions(p); }
            catch (Exception e)
            {
                s.Options = new List<Option>();
                p.Wait = "options error: " + e.GetType().Name + ": " + e.Message;
                U.Error("BuildOptions failed: " + e);
            }
        }
        var fp = new JsonObject
        {
            ["screen"] = p.Screen,
            ["detail"] = p.Detail,
            ["key"] = p.Key,
            ["stable"] = stable,
            ["run"] = s.Run?.DeepClone(),
            ["combat"] = s.Combat?.DeepClone(),
            ["options"] = U.Arr(s.Options, o => o.Json.DeepClone()),
        };
        s.Fingerprint = U.Hash(U.Str(fp));
        return s;
    }

    // ===================================================================== run
    private static JsonObject? BuildRun(Probe p)
    {
        RunState? rs = p.Run;
        Player? me = p.Me;
        if (rs == null || me == null) return null;
        var o = new JsonObject
        {
            ["character"] = me.Character.Id.Entry,
            ["ascension"] = rs.AscensionLevel,
            ["seed"] = SafeSeed(rs),
            ["act_index"] = rs.CurrentActIndex,
            ["act_ids"] = U.Arr(rs.Acts, a => JsonValue.Create(a.Id.Entry)),
            ["floor"] = rs.TotalFloor,
            ["act_floor"] = rs.ActFloor,
            ["hp"] = me.Creature.CurrentHp,
            ["max_hp"] = me.Creature.MaxHp,
            ["gold"] = me.Gold,
            ["deck"] = U.Arr(me.Deck.Cards, c => DeckCard(c)),
            ["relics"] = U.Arr(me.Relics, r => new JsonObject
            {
                ["id"] = r.Id.Entry,
                ["counter"] = SafeCounter(r),
            }),
            ["potions"] = U.Arr(me.PotionSlots, pt => pt == null ? null : JsonValue.Create(pt.Id.Entry)),
            ["map"] = BuildMap(rs),
            ["save"] = null,
        };
        return o;
    }

    private static string SafeSeed(RunState rs)
    {
        try { return rs.Rng.StringSeed; } catch { return ""; }
    }

    private static JsonNode? SafeCounter(RelicModel r)
    {
        try { return r.ShowCounter ? JsonValue.Create(r.DisplayAmount) : null; } catch { return null; }
    }

    private static JsonObject DeckCard(CardModel c)
    {
        return new JsonObject
        {
            ["id"] = c.Id.Entry,
            ["upgrade"] = c.CurrentUpgradeLevel,
            ["enchantment"] = c.Enchantment?.Id.Entry,
            ["enchantment_amount"] = c.Enchantment == null ? null : JsonValue.Create(c.Enchantment.Amount),
            ["floor_added"] = c.FloorAddedToDeck == null ? null : JsonValue.Create(c.FloorAddedToDeck.Value),
        };
    }

    private static JsonObject BuildMap(RunState rs)
    {
        var map = new JsonObject();
        MapCoord? cur = rs.CurrentMapCoord;
        map["current"] = cur == null ? null : Coord(cur.Value);
        map["visited"] = U.Arr(rs.VisitedMapCoords, c => Coord(c));
        var points = new JsonArray();
        try
        {
            ActMap m = rs.Map;
            var all = new Dictionary<(int, int), MapPoint>();
            foreach (MapPoint mp in m.GetAllMapPoints()) all[(mp.coord.col, mp.coord.row)] = mp;
            foreach (MapPoint? extra in new[] { SafePoint(() => m.StartingMapPoint), SafePoint(() => m.BossMapPoint), SafePoint(() => m.SecondBossMapPoint) })
                if (extra != null) all[(extra.coord.col, extra.coord.row)] = extra;
            foreach (MapPoint mp in all.Values.OrderBy(x => x.coord.row).ThenBy(x => x.coord.col))
            {
                points.Add(new JsonObject
                {
                    ["col"] = mp.coord.col,
                    ["row"] = mp.coord.row,
                    ["type"] = mp.PointType.ToString(),
                    ["children"] = U.Arr(mp.Children.OrderBy(c => c.coord.col), c => Coord(c.coord)),
                });
            }
        }
        catch (Exception e) { map["error"] = e.Message; }
        map["points"] = points;
        return map;
    }

    private static MapPoint? SafePoint(Func<MapPoint?> f)
    {
        try { return f(); } catch { return null; }
    }

    private static JsonObject Coord(MapCoord c) => new() { ["col"] = c.col, ["row"] = c.row };

    /// <summary>The game's own save JSON of the live run (RunManager.ToSave + the game's JSON context).</summary>
    public static JsonNode? BuildSave(out string? error)
    {
        error = null;
        try
        {
            if (!RunManager.Instance.IsInProgress) return null;
            SerializableRun save = RunManager.Instance.ToSave(null);
            string json = JsonSerializationUtility.ToJson(save);
            return JsonNode.Parse(json);
        }
        catch (Exception e)
        {
            error = e.GetType().Name + ": " + e.Message;
            return null;
        }
    }

    // ===================================================================== combat
    private static JsonObject? BuildCombat(Probe p)
    {
        if (!p.InCombat || p.Me == null) return null;
        CombatState? cs = CombatManager.Instance.DebugOnlyGetState();
        if (cs == null) return null;
        Player me = p.Me;
        var pcs = me.PlayerCombatState;
        var player = new JsonObject
        {
            ["hp"] = me.Creature.CurrentHp,
            ["max_hp"] = me.Creature.MaxHp,
            ["block"] = me.Creature.Block,
            ["energy"] = Safe(() => pcs?.Energy ?? 0),
            ["max_energy"] = Safe(() => pcs?.MaxEnergy ?? 0),
            ["stars"] = Safe(() => pcs?.Stars ?? 0),
            ["powers"] = Powers(me.Creature),
            ["hand"] = pcs == null ? new JsonArray() : U.Arr(pcs.Hand.Cards, c => Card(c, true)),
            ["draw"] = pcs == null ? new JsonArray() : U.Arr(pcs.DrawPile.Cards, c => Card(c, false)),
            ["discard"] = pcs == null ? new JsonArray() : U.Arr(pcs.DiscardPile.Cards, c => Card(c, false)),
            ["exhaust"] = pcs == null ? new JsonArray() : U.Arr(pcs.ExhaustPile.Cards, c => Card(c, false)),
            ["orbs"] = pcs == null ? new JsonArray() : U.Arr(pcs.OrbQueue.Orbs, orb => new JsonObject
            {
                ["id"] = orb.Id.Entry,
                ["value"] = Safe(() => (double)orb.PassiveVal),
                ["evoke"] = Safe(() => (double)orb.EvokeVal),
            }),
        };
        var monsters = new JsonArray();
        var enemies = cs.Enemies;
        for (int i = 0; i < enemies.Count; i++)
        {
            Creature c = enemies[i];
            var mo = new JsonObject
            {
                ["index"] = i,
                ["id"] = c.Monster?.Id.Entry,
                ["slot"] = c.SlotName,
                ["hp"] = c.CurrentHp,
                ["max_hp"] = c.MaxHp,
                ["block"] = c.Block,
                ["alive"] = c.IsAlive,
                ["hittable"] = Safe(() => c.IsHittable),
                ["powers"] = Powers(c),
                ["move_id"] = SafeStr(() => c.Monster?.NextMove?.Id),
                ["intents"] = Intents(c, cs),
            };
            monsters.Add(mo);
        }
        RoomType rt = RoomType.Monster;
        try { if (cs.Encounter != null) rt = cs.Encounter.RoomType; } catch { }
        return new JsonObject
        {
            ["round"] = cs.RoundNumber,
            ["turn_side"] = cs.CurrentSide == CombatSide.Player ? "PLAYER" : "ENEMY",
            ["encounter_id"] = cs.Encounter?.Id.Entry,
            ["room_type"] = rt.ToString().ToUpperInvariant(),
            ["player"] = player,
            ["monsters"] = monsters,
        };
    }

    private static JsonArray Powers(Creature c)
    {
        var a = new JsonArray();
        foreach (PowerModel pw in c.Powers.ToList())
        {
            a.Add(new JsonObject { ["id"] = pw.Id.Entry, ["amount"] = Safe(() => pw.Amount) });
        }
        return a;
    }

    private static JsonArray Intents(Creature c, CombatState cs)
    {
        var a = new JsonArray();
        try
        {
            var move = c.Monster?.NextMove;
            if (move == null || !c.IsAlive) return a;
            var targets = cs.PlayerCreatures;
            foreach (AbstractIntent it in move.Intents)
            {
                var io = new JsonObject { ["type"] = it.IntentType.ToString() };
                if (it is AttackIntent ai)
                {
                    io["damage"] = Safe(() => ai.GetSingleDamage(targets, c));
                    io["hits"] = Math.Max(1, Safe(() => ai.Repeats));
                }
                a.Add(io);
            }
        }
        catch (Exception e) { a.Add(new JsonObject { ["error"] = e.Message }); }
        return a;
    }

    public static JsonObject Card(CardModel c, bool inHand)
    {
        bool x = Safe(() => c.EnergyCost.CostsX);
        bool inCombat = CombatManager.Instance.IsInProgress;
        var o = new JsonObject
        {
            ["id"] = c.Id.Entry,
            ["upgrade"] = c.CurrentUpgradeLevel,
            ["enchantment"] = c.Enchantment?.Id.Entry,
            ["enchantment_amount"] = c.Enchantment == null ? null : JsonValue.Create(Safe(() => c.Enchantment.Amount)),
            ["affliction"] = c.Affliction?.Id.Entry,
            ["cost"] = x ? -1 : Safe(() => c.EnergyCost.Canonical),
            ["cost_this_turn"] = x ? -1 : (inCombat ? Safe(() => c.EnergyCost.GetAmountToSpend()) : Safe(() => c.EnergyCost.Canonical)),
            ["x_cost"] = x,
            ["playable"] = inHand && inCombat && Safe(() => c.CanPlay()),
            ["needs_target"] = c.TargetType == TargetType.AnyEnemy || c.TargetType == TargetType.AnyAlly,
            ["target_type"] = c.TargetType.ToString(),
            ["instance"] = InstanceIds.Of(c),
        };
        return o;
    }

    private static T Safe<T>(Func<T> f)
    {
        try { return f(); } catch { return default!; }
    }

    private static string? SafeStr(Func<string?> f)
    {
        try { return f(); } catch { return null; }
    }

    // ===================================================================== options
    private static List<Option> BuildOptions(Probe p)
    {
        switch (p.Screen)
        {
            case "COMBAT": return CombatOptions(p);
            case "COMBAT_SELECT":
            case "CARD_SELECT":
                return SelectOptions(p);
            case "MAP": return MapOptions(p);
            case "CARD_REWARD": return CardRewardOptions(p);
            case "REWARDS": return RewardOptions(p);
            case "EVENT":
            case "ANCIENT":
                return EventOptions(p);
            case "REST": return RestOptions(p);
            case "SHOP": return ShopOptions(p);
            case "TREASURE": return TreasureOptions(p);
            case "GAME_OVER": return GameOverOptions(p);
            case "CRYSTAL_SPHERE": return SphereOptions(p);
            case "OTHER":
                if (p.Overlay is NChooseARelicSelection rs) return RelicSelectOptions(rs);
                return new List<Option>();
            default: return new List<Option>();
        }
    }

    private static Option Opt(string kind, string label)
    {
        var o = new Option { Label = label };
        o.Json["kind"] = kind;
        return o;
    }

    private static string? NoArgs(ActArgs a)
    {
        if (a.Target != null) return "this option takes no target";
        if (a.Cards != null && a.Cards.Count > 0) return "this option takes no cards";
        return null;
    }

    private static async Task<string?> Click(NClickableControl? c, string what)
    {
        if (!U.Usable(c)) return what + " is not clickable any more";
        c!.ForceClick();
        await Bridge.Delay(150);
        return null;
    }

    // ---------------------------------------------------------------- combat
    private static List<Option> CombatOptions(Probe p)
    {
        var list = new List<Option>();
        Player me = p.Me!;
        CombatState cs = CombatManager.Instance.DebugOnlyGetState()!;
        var pcs = me.PlayerCombatState!;
        var enemies = cs.Enemies.ToList();

        var hand = pcs.Hand.Cards.ToList();
        for (int i = 0; i < hand.Count; i++)
        {
            CardModel card = hand[i];
            bool playable;
            try { playable = card.CanPlay(); } catch { playable = false; }
            if (!playable) continue;
            var targets = new List<int>();
            bool needsTarget = card.TargetType == TargetType.AnyEnemy;
            if (card.TargetType == TargetType.AnyAlly) continue; // no ally targets for Ironclad; not supported in v1
            if (needsTarget)
            {
                for (int t = 0; t < enemies.Count; t++)
                {
                    Creature e = enemies[t];
                    bool ok;
                    try { ok = e.IsAlive && e.IsHittable && card.CanPlayTargeting(e); } catch { ok = false; }
                    if (ok) targets.Add(t);
                }
                if (targets.Count == 0) continue;
            }
            else
            {
                bool ok;
                try { ok = card.CanPlayTargeting(null); } catch { ok = true; }
                if (!ok) continue;
            }
            int handIndex = i;
            var o = Opt("play", "play " + card.Id.Entry);
            o.Json["hand_index"] = handIndex;
            o.Json["card"] = Card(card, true);
            o.Json["targets"] = U.Arr(targets, t => JsonValue.Create(t));
            o.Validate = a =>
            {
                if (a.Cards != null && a.Cards.Count > 0) return "play takes no cards";
                if (needsTarget)
                {
                    if (a.Target == null) return "this card needs a target (one of " + string.Join(",", targets) + ")";
                    if (!targets.Contains(a.Target.Value)) return "illegal target " + a.Target + " (legal: " + string.Join(",", targets) + ")";
                }
                else if (a.Target != null) return "this card takes no target";
                return null;
            };
            o.Exec = async a =>
            {
                var pc = me.PlayerCombatState;
                if (pc == null || !pc.Hand.Cards.Contains(card)) return "card is no longer in hand";
                Creature? target = needsTarget ? enemies[a.Target!.Value] : null;
                if (target != null && !target.IsAlive) return "target is dead";
                // Same call the hand UI makes when a card is released on a target (NCardPlay.TryPlayCard).
                if (needsTarget && NCombatRoom.Instance != null) NCombatRoom.Instance.LastTargetedCreature = target;
                bool ok = card.TryManualPlay(target);
                if (!ok) return "the game refused to play " + card.Id.Entry;
                await Bridge.Delay(100);
                return null;
            };
            list.Add(o);
        }

        // potions
        bool potionsLocked = false;
        try { potionsLocked = RunManager.Instance.IsPotionUseLocked || !me.CanUseOrRemovePotions; } catch { }
        if (!potionsLocked)
        {
            var slots = me.PotionSlots.ToList();
            for (int s = 0; s < slots.Count; s++)
            {
                PotionModel? pot = slots[s];
                if (pot == null) continue;
                bool usable;
                try
                {
                    usable = (pot.Usage == PotionUsage.CombatOnly || pot.Usage == PotionUsage.AnyTime)
                             && !pot.IsQueued && pot.PassesCustomUsabilityCheck;
                }
                catch { usable = false; }
                if (!usable) continue;
                TargetType tt = pot.TargetType;
                if (tt == TargetType.TargetedNoCreature || tt == TargetType.AnyAlly) continue;
                bool needsTarget = tt == TargetType.AnyEnemy;
                var targets = new List<int>();
                if (needsTarget)
                {
                    for (int t = 0; t < enemies.Count; t++)
                    {
                        bool ok;
                        try { ok = enemies[t].IsAlive && enemies[t].IsHittable && pot.IsValidTarget(enemies[t]); } catch { ok = false; }
                        if (ok) targets.Add(t);
                    }
                    if (targets.Count == 0) continue;
                }
                int slot = s;
                var o = Opt("potion", "potion " + pot.Id.Entry);
                o.Json["slot"] = slot;
                o.Json["id"] = pot.Id.Entry;
                o.Json["targets"] = U.Arr(targets, t => JsonValue.Create(t));
                o.Validate = a =>
                {
                    if (a.Cards != null && a.Cards.Count > 0) return "potion takes no cards";
                    if (needsTarget)
                    {
                        if (a.Target == null) return "this potion needs a target (one of " + string.Join(",", targets) + ")";
                        if (!targets.Contains(a.Target.Value)) return "illegal target " + a.Target;
                    }
                    else if (a.Target != null) return "this potion takes no target";
                    return null;
                };
                o.Exec = async a =>
                {
                    if (me.GetPotionAtSlotIndex(slot) != pot) return "potion is no longer in that slot";
                    Creature? target = needsTarget ? enemies[a.Target!.Value] : (tt == TargetType.Self || tt == TargetType.AnyPlayer ? me.Creature : null);
                    // Same call the potion UI makes (NPotionHolder.UsePotion / TargetNode).
                    pot.EnqueueManualUse(target);
                    await Bridge.Delay(100);
                    return null;
                };
                list.Add(o);
            }
        }

        // end turn
        {
            var o = Opt("end_turn", "end turn");
            o.Validate = NoArgs;
            o.Exec = async a =>
            {
                var btn = NCombatRoom.Instance?.Ui?.EndTurnButton;
                if (btn != null && U.Usable(btn))
                {
                    btn.CallReleaseLogic(); // the end-turn button's own click logic
                }
                else
                {
                    var pc = me.PlayerCombatState;
                    if (pc == null) return "no player combat state";
                    RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(me, pc.TurnNumber));
                }
                await Bridge.Delay(100);
                return null;
            };
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- card selections
    private static string PromptKey(CardSelectorPrefs prefs)
    {
        try { return prefs.Prompt?.LocEntryKey ?? ""; } catch { return ""; }
    }

    private static string PromptText(CardSelectorPrefs prefs)
    {
        try { return prefs.Prompt == null ? "" : U.Plain(prefs.Prompt.GetFormattedText()); } catch { return ""; }
    }

    private static string PurposeOf(object screen, string promptKey)
    {
        string k = promptKey.ToUpperInvariant();
        if (screen is NDeckUpgradeSelectScreen || k.Contains("UPGRADE")) return "upgrade";
        if (screen is NDeckTransformSelectScreen || k.Contains("TRANSFORM")) return "transform";
        if (screen is NDeckEnchantSelectScreen || k.Contains("ENCHANT")) return "enchant";
        if (k.Contains("REMOVE")) return "remove";
        if (k.Contains("EXHAUST")) return "exhaust";
        if (k.Contains("DISCARD")) return "discard";
        if (k.Contains("ABSORB")) return "absorb";
        if (screen is NChooseACardSelectionScreen) return "choose";
        return promptKey.Length > 0 ? promptKey.ToLowerInvariant() : "select";
    }

    private static Option SelectOption(string purpose, string promptKey, string prompt, int min, int max, List<CardModel> cards, string source,
        Func<List<CardModel>, Task<string?>> exec)
    {
        var o = Opt("select", "select (" + purpose + ")");
        o.Json["purpose"] = purpose;
        o.Json["prompt_key"] = promptKey;
        o.Json["prompt"] = prompt;
        o.Json["min"] = min;
        o.Json["max"] = max;
        o.Json["source"] = source;
        o.Json["cards"] = U.Arr(cards, c => Card(c, false));
        o.Validate = a =>
        {
            if (a.Target != null) return "select takes no target";
            var idx = a.Cards ?? new List<int>();
            if (idx.Count < min || idx.Count > max) return $"select needs between {min} and {max} cards, got {idx.Count}";
            if (idx.Distinct().Count() != idx.Count) return "duplicate card indices";
            foreach (int i in idx) if (i < 0 || i >= cards.Count) return "card index " + i + " out of range";
            return null;
        };
        o.Exec = a => exec((a.Cards ?? new List<int>()).Select(i => cards[i]).ToList());
        return o;
    }

    private static List<Option> SelectOptions(Probe p)
    {
        var list = new List<Option>();
        if (p.Overlay is NCardGridSelectionScreen grid)
        {
            var cards = (U.GetRaw(grid, "_cards") as IEnumerable<CardModel>)?.ToList() ?? new List<CardModel>();
            U.TryGetStruct(grid, "_prefs", out CardSelectorPrefs prefs);
            int min = prefs.MinSelect, max = prefs.MaxSelect;
            if (max <= 0) { min = Math.Max(0, min); max = Math.Max(1, max); }
            string key = PromptKey(prefs);
            string source = grid is NCombatPileCardSelectScreen ? "pile" : "deck";
            int gmax = Math.Min(max, cards.Count), gmin = Math.Min(Math.Max(0, min), gmax);
            list.Add(SelectOption(PurposeOf(grid, key), key, PromptText(prefs), gmin, gmax, cards, source,
                picks => Actions.GridSelect(grid, picks)));
            var close = grid.GetNodeOrNull<NClickableControl>("%Close") ?? U.Get<NClickableControl>(grid, "_closeButton");
            if (U.Usable(close))
            {
                var sk = Opt("skip", "skip selection");
                sk.Validate = NoArgs;
                sk.Exec = _ => Click(close, "close button");
                list.Add(sk);
            }
            return list;
        }
        if (p.Overlay is NChooseACardSelectionScreen choose)
        {
            var cards = (U.GetRaw(choose, "_cards") as IEnumerable<CardModel>)?.ToList()
                        ?? U.FindAll<NCardHolder>(choose).Select(h => h.CardModel).Where(c => c != null).Cast<CardModel>().ToList();
            list.Add(SelectOption("choose", "", "Choose a card", 1, 1, cards, "generated",
                picks => Actions.ChooseACard(choose, picks[0])));
            var skip = U.Get<NClickableControl>(choose, "_skipButton");
            if (U.Usable(skip))
            {
                var sk = Opt("skip", "skip choice");
                sk.Validate = NoArgs;
                sk.Exec = _ => Click(skip, "skip button");
                list.Add(sk);
            }
            return list;
        }
        NPlayerHand? hand = NPlayerHand.Instance;
        if (hand != null && hand.IsInCardSelection)
        {
            U.TryGetStruct(hand, "_prefs", out CardSelectorPrefs prefs);
            var cards = hand.ActiveHolders.Select(h => h.CardModel).Where(c => c != null).Cast<CardModel>().ToList();
            string key = PromptKey(prefs);
            string purpose = hand.CurrentMode == NPlayerHand.Mode.UpgradeSelect ? "upgrade" : PurposeOf(hand, key);
            int min = Math.Max(0, prefs.MinSelect), max = Math.Min(Math.Max(prefs.MaxSelect, min), cards.Count);
            list.Add(SelectOption(purpose, key, PromptText(prefs), Math.Min(min, max), max, cards, "hand",
                picks => Actions.HandSelect(hand, picks)));
        }
        return list;
    }

    // ---------------------------------------------------------------- map
    private static List<Option> MapOptions(Probe p)
    {
        var list = new List<Option>();
        NMapScreen ms = NMapScreen.Instance!;
        var pts = U.FindAll<NMapPoint>(ms).Where(mp => mp.State == MapPointState.Travelable && U.Usable(mp) && mp.Point != null)
            .OrderBy(mp => mp.Point.coord.row).ThenBy(mp => mp.Point.coord.col).ToList();
        foreach (NMapPoint mp in pts)
        {
            NMapPoint node = mp;
            MapPoint pt = mp.Point;
            var o = Opt("map", $"map ({pt.coord.col},{pt.coord.row}) {pt.PointType}");
            o.Json["col"] = pt.coord.col;
            o.Json["row"] = pt.coord.row;
            o.Json["type"] = pt.PointType.ToString();
            o.Validate = NoArgs;
            o.Exec = async _ =>
            {
                if (!U.Usable(node) || node.State != MapPointState.Travelable) return "map node no longer travelable";
                if (!ms.IsTravelEnabled || ms.IsTraveling) return "map travel not enabled";
                // What NMapPoint.OnRelease calls on a click. Called directly because the click itself is also gated
                // on the first-map FTUE, map drawing mode and the node being on screen.
                ms.OnMapPointSelectedLocally(node);
                await Bridge.Delay(200);
                return null;
            };
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- card reward
    private static List<Option> CardRewardOptions(Probe p)
    {
        var list = new List<Option>();
        if (p.Overlay is not NCardRewardSelectionScreen scr) return list;
        var results = (U.GetRaw(scr, "_options") as System.Collections.IEnumerable)?.Cast<object>().ToList() ?? new List<object>();
        var alts = (U.GetRaw(scr, "_extraOptions") as System.Collections.IEnumerable)?.Cast<object>().ToList() ?? new List<object>();
        for (int i = 0; i < results.Count; i++)
        {
            CardModel? card = U.GetRaw(results[i], "Card") as CardModel;
            if (card == null) continue;
            CardModel c = card;
            var o = Opt("card", "take " + c.Id.Entry);
            o.Json["index"] = i;
            o.Json["id"] = c.Id.Entry;
            o.Json["upgrade"] = c.CurrentUpgradeLevel;
            o.Json["enchantment"] = c.Enchantment?.Id.Entry;
            o.Json["enchantment_amount"] = c.Enchantment == null ? null : JsonValue.Create(c.Enchantment.Amount);
            o.Json["card"] = Card(c, false);
            o.Validate = NoArgs;
            o.Exec = async _ =>
            {
                if (!U.Valid(scr) || !scr.IsInsideTree()) return "card reward screen closed";
                NCardHolder holder;
                try { holder = scr.GetCardHolder(c); } catch { return "card holder not found"; }
                holder.EmitSignal(NCardHolder.SignalName.Pressed, holder); // same signal a click emits
                await Bridge.Delay(200);
                return null;
            };
            list.Add(o);
        }
        var altContainer = U.Get<Node>(scr, "_rewardAlternativesContainer");
        var altButtons = altContainer == null ? new List<NCardRewardAlternativeButton>()
            : altContainer.GetChildren().OfType<NCardRewardAlternativeButton>().Where(b => !b.IsQueuedForDeletion()).ToList();
        for (int j = 0; j < alts.Count; j++)
        {
            string id = U.GetRaw(alts[j], "OptionId") as string ?? ("ALT" + j);
            NCardRewardAlternativeButton? btn = j < altButtons.Count ? altButtons[j] : null;
            bool isSkip = string.Equals(id, "Skip", StringComparison.OrdinalIgnoreCase);
            var o = Opt(isSkip ? "skip" : "alternative", isSkip ? "skip card reward" : "card reward " + id);
            o.Json["id"] = id;
            o.Validate = NoArgs;
            o.Exec = _ => Click(btn, "card reward button " + id);
            if (btn != null && U.Usable(btn)) list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- rewards
    private static string RewardType(Reward r) => r switch
    {
        GoldReward => "GOLD",
        PotionReward => "POTION",
        RelicReward => "RELIC",
        SpecialCardReward => "SPECIAL_CARD",
        CardReward => "CARD",
        CardRemovalReward => "CARD_REMOVAL",
        CardUpgradeReward => "CARD_UPGRADE",
        _ => r.GetType().Name.ToUpperInvariant(),
    };

    private static List<Option> RewardOptions(Probe p)
    {
        var list = new List<Option>();
        if (p.Overlay is NRewardsScreen scr && NMapScreen.Instance?.IsOpen != true)
        {
            bool canTakePotion = p.Me?.HasOpenPotionSlots ?? false;
            var buttons = U.FindAll<NRewardButton>(scr).Where(b => U.Usable(b) && b.Reward != null).ToList();
            for (int i = 0; i < buttons.Count; i++)
            {
                NRewardButton b = buttons[i];
                Reward r = b.Reward!;
                if (r is PotionReward && !canTakePotion) continue; // the game would refuse: potion belt full
                var o = Opt("reward", "reward " + RewardType(r));
                o.Json["index"] = i;
                o.Json["type"] = RewardType(r);
                switch (r)
                {
                    case GoldReward g: o.Json["amount"] = g.Amount; break;
                    case PotionReward pr: o.Json["id"] = pr.Potion?.Id.Entry; break;
                    case RelicReward rr: o.Json["id"] = rr.Relic?.Id.Entry; break;
                }
                o.Json["linked"] = b.GetParent() is not null && U.HasAncestorNamed(b, "Linked", scr);
                o.Json["text"] = U.SafeText(() => r.Description);
                o.Validate = NoArgs;
                o.Exec = _ => Click(b, "reward button");
                list.Add(o);
            }
            var proceed = U.FindFirst<NProceedButton>(scr);
            if (U.Usable(proceed))
            {
                var o = Opt("proceed", "proceed");
                o.Validate = NoArgs;
                o.Exec = _ => Click(proceed, "proceed button");
                list.Add(o);
            }
            return list;
        }
        // after-combat room with only a proceed button
        var cp = NCombatRoom.Instance?.ProceedButton;
        if (U.Usable(cp))
        {
            var o = Opt("proceed", "proceed");
            o.Validate = NoArgs;
            o.Exec = _ => Click(cp, "proceed button");
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- events
    private static List<Option> EventOptions(Probe p)
    {
        var list = new List<Option>();
        var layout = NEventRoom.Instance?.Layout;
        if (layout == null) return list;
        var buttons = layout.OptionButtons.Where(b => U.Valid(b) && !b.IsQueuedForDeletion()).ToList();
        for (int i = 0; i < buttons.Count; i++)
        {
            NEventOptionButton b = buttons[i];
            EventOption opt = b.Option;
            bool locked = opt.IsLocked;
            if (!locked && !U.Usable(b)) continue;
            var o = Opt("event_option", "event option " + opt.TextKey);
            o.Json["index"] = i;
            o.Json["key"] = opt.TextKey;
            o.Json["title_key"] = SafeStr(() => opt.Title?.LocEntryKey);
            o.Json["text"] = U.SafeText(() => opt.Title);
            o.Json["description"] = U.SafeText(() => opt.Description);
            o.Json["locked"] = locked;
            o.Json["proceed"] = opt.IsProceed;
            o.Json["relic"] = opt.Relic?.Id.Entry;
            o.Validate = a => locked ? "this event option is locked" : NoArgs(a);
            o.Exec = _ => Click(b, "event option button"); // NEventOptionButton -> NEventRoom.OptionButtonClicked
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- rest site
    private static List<Option> RestOptions(Probe p)
    {
        var list = new List<Option>();
        var rr = NRestSiteRoom.Instance;
        if (rr == null) return list;
        var opts = rr.Options.ToList();
        for (int i = 0; i < opts.Count; i++)
        {
            RestSiteOption opt = opts[i];
            NRestSiteButton? btn = rr.GetButtonForOption(opt);
            if (!U.Usable(btn)) continue;
            bool enabled = opt.IsEnabled;
            var o = Opt("rest_option", "rest " + opt.OptionId);
            o.Json["id"] = opt.OptionId;
            o.Json["index"] = i;
            o.Json["enabled"] = enabled;
            o.Validate = a => enabled ? NoArgs(a) : "rest option " + opt.OptionId + " is disabled";
            o.Exec = _ => Click(btn, "rest site button");
            if (enabled) list.Add(o);
        }
        if (U.Usable(rr.ProceedButton))
        {
            var o = Opt("proceed", "leave rest site");
            o.Validate = NoArgs;
            o.Exec = _ => Click(rr.ProceedButton, "proceed button");
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- shop
    private static List<Option> ShopOptions(Probe p)
    {
        var list = new List<Option>();
        var mr = NMerchantRoom.Instance;
        if (mr == null) return list;
        MerchantInventory inv = RunManager.Instance.MerchantSynchronizer.GetLocalInventory();
        var entries = new List<(MerchantEntry e, string item, string? id, int upgrade, JsonObject? card, bool onSale)>();
        foreach (MerchantCardEntry ce in inv.CardEntries)
        {
            var c = ce.CreationResult?.Card;
            entries.Add((ce, "card", c?.Id.Entry, c?.CurrentUpgradeLevel ?? 0, c == null ? null : Card(c, false), ce.IsOnSale));
        }
        foreach (MerchantRelicEntry re in inv.RelicEntries) entries.Add((re, "relic", re.Model?.Id.Entry, 0, null, false));
        foreach (MerchantPotionEntry pe in inv.PotionEntries) entries.Add((pe, "potion", pe.Model?.Id.Entry, 0, null, false));
        if (inv.CardRemovalEntry != null) entries.Add((inv.CardRemovalEntry, "remove", null, 0, null, false));
        bool canTakePotion = p.Me?.HasOpenPotionSlots ?? false;
        for (int i = 0; i < entries.Count; i++)
        {
            var (entry, item, id, upgrade, card, onSale) = entries[i];
            bool stocked, afford;
            int price;
            try { stocked = entry.IsStocked; afford = entry.EnoughGold; price = entry.Cost; } catch { continue; }
            if (!stocked || !afford) continue;
            if (item == "potion" && !canTakePotion) continue;
            MerchantEntry en = entry;
            var o = Opt("buy", "buy " + item + " " + (id ?? ""));
            o.Json["item"] = item;
            o.Json["id"] = id;
            o.Json["price"] = price;
            o.Json["entry_index"] = i;
            if (item == "card") { o.Json["upgrade"] = upgrade; o.Json["on_sale"] = onSale; o.Json["card"] = card; }
            o.Validate = NoArgs;
            o.Exec = _ => Actions.ShopBuy(mr, en);
            list.Add(o);
        }
        {
            var o = Opt("leave", "leave shop");
            o.Validate = NoArgs;
            o.Exec = _ => Actions.ShopLeave(mr);
            if (U.Usable(mr.ProceedButton) || mr.Inventory.IsOpen) list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- treasure
    private static List<Option> TreasureOptions(Probe p)
    {
        var list = new List<Option>();
        var tr = NRun.Instance?.TreasureRoom;
        if (tr == null) return list;
        var chest = U.Get<NClickableControl>(tr, "_chestButton");
        if (U.Usable(chest))
        {
            var o = Opt("open", "open chest");
            o.Validate = NoArgs;
            o.Exec = _ => Click(chest, "chest");
            list.Add(o);
        }
        var holders = Screens.TreasureRelicHolders(tr);
        for (int i = 0; i < holders.Count; i++)
        {
            var h = holders[i];
            var o = Opt("relic", "take relic " + h.Relic.Model.Id.Entry);
            o.Json["id"] = h.Relic.Model.Id.Entry;
            o.Json["index"] = i;
            o.Validate = NoArgs;
            o.Exec = _ => Click(h, "relic");
            list.Add(o);
        }
        if (U.Usable(tr.ProceedButton))
        {
            var o = Opt("proceed", "leave treasure room");
            o.Validate = NoArgs;
            o.Exec = _ => Click(tr.ProceedButton, "proceed button");
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- crystal sphere minigame
    private static List<Option> SphereOptions(Probe p)
    {
        var list = new List<Option>();
        if (p.Overlay is not NCrystalSphereScreen sph) return list;
        int left = Screens.SphereDivinations(sph);
        if (left > 0)
            foreach (var cell in Screens.SphereHiddenCells(sph))
            {
                var c = cell;
                var o = Opt("sphere_cell", $"crystal sphere cell ({c.Entity.X},{c.Entity.Y})");
                o.Json["x"] = c.Entity.X;
                o.Json["y"] = c.Entity.Y;
                o.Json["divinations"] = left;
                o.Validate = NoArgs;
                o.Exec = async _ =>
                {
                    if (!GodotObject.IsInstanceValid(c) || !c.Visible || !c.Entity.IsHidden) return "cell is not clickable any more";
                    c.EmitSignal(NClickableControl.SignalName.Released, c);   // the game's AutoSlay click
                    await Bridge.Delay(500);
                    return null;
                };
                list.Add(o);
            }
        var proceed = U.Get<NProceedButton>(sph, "_proceedButton");
        if (U.Usable(proceed))
        {
            var o = Opt("proceed", "leave crystal sphere");
            o.Validate = NoArgs;
            o.Exec = _ => Click(proceed, "proceed button");
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- relic selection (OTHER)
    private static List<Option> RelicSelectOptions(NChooseARelicSelection scr)
    {
        var list = new List<Option>();
        var holders = U.FindAll<NRelicBasicHolder>(scr).Where(h => U.Usable(h)).ToList();
        for (int i = 0; i < holders.Count; i++)
        {
            var h = holders[i];
            string? id = null;
            try { id = h.Relic?.Model?.Id.Entry; } catch { }
            var o = Opt("relic", "choose relic " + id);
            o.Json["id"] = id;
            o.Json["index"] = i;
            o.Validate = NoArgs;
            o.Exec = _ => Click(h, "relic");
            list.Add(o);
        }
        var skip = U.Get<NClickableControl>(scr, "_skipButton");
        if (U.Usable(skip))
        {
            var o = Opt("skip", "skip relic");
            o.Validate = NoArgs;
            o.Exec = _ => Click(skip, "skip button");
            list.Add(o);
        }
        return list;
    }

    // ---------------------------------------------------------------- game over
    private static List<Option> GameOverOptions(Probe p)
    {
        var list = new List<Option>();
        if (p.Overlay is not NGameOverScreen scr) return list;
        var o = Opt("to_menu", "return to main menu");
        o.Validate = NoArgs;
        o.Exec = _ => Actions.GameOverToMenu(scr);
        list.Add(o);
        return list;
    }
}
