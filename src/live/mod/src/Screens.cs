using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace AgentBridge;

/// <summary>What is on screen right now, and whether the game waits for input there.</summary>
internal sealed class Probe
{
    public string Screen = "OTHER";
    public string Detail = "";
    /// <summary>Raw readiness this frame (no settle time applied).</summary>
    public bool Ready;
    /// <summary>Why not ready (diagnostics only).</summary>
    public string Wait = "";
    /// <summary>Identity of the screen instance; a change restarts the settle timer.</summary>
    public string Key = "";

    public RunState? Run;
    public Player? Me;
    public object? Overlay;
    public bool InCombat;
}

/// <summary>
/// Screen detection. Read-only: never changes the game. Order matters: modal popups and overlays sit above rooms;
/// the rewards / card-reward / crystal-sphere overlays linger on the overlay stack after the map opens, so the
/// map wins over them (same rule as the game's own AutoSlay / STS2MCP).
/// </summary>
internal static class Screens
{
    public static int SphereDivinations(NCrystalSphereScreen sph)
    {
        try { return U.Get<CrystalSphereMinigame>(sph, "_entity")?.DivinationCount ?? 0; } catch { return 0; }
    }

    /// <summary>Clickable cells of the Crystal Sphere grid (visible and still hidden), as AutoSlay picks them.</summary>
    public static List<NCrystalSphereCell> SphereHiddenCells(NCrystalSphereScreen sph)
    {
        var box = sph.GetNodeOrNull<Control>("%Cells");
        if (box == null) return new List<NCrystalSphereCell>();
        return U.FindAll<NCrystalSphereCell>(box).Where(c => c.Visible && c.Entity != null && c.Entity.IsHidden).ToList();
    }

    public static Probe Detect()
    {
        var p = new Probe();
        try { DetectInto(p); }
        catch (Exception e)
        {
            p.Screen = "OTHER";
            p.Detail = "probe error: " + e.GetType().Name + ": " + e.Message;
            p.Ready = false;
            p.Key = "ERR";
        }
        return p;
    }

    private static void DetectInto(Probe p)
    {
        NGame? game = NGame.Instance;
        if (game == null) { p.Detail = "game not ready"; p.Key = "NOGAME"; return; }

        RunManager rm = RunManager.Instance;
        if (!rm.IsInProgress)
        {
            var mm = game.MainMenu;
            if (mm != null)
            {
                p.Screen = "MAIN_MENU";
                p.Detail = "NMainMenu";
                p.Ready = !game.Transition.InTransition;
                if (!p.Ready) p.Wait = "transition";
                p.Key = "MAIN_MENU";
            }
            else
            {
                p.Detail = U.TypeName(game.RootSceneContainer?.CurrentScene);
                p.Key = "OTHER:" + p.Detail;
                p.Wait = "no run and no main menu (loading?)";
            }
            return;
        }

        RunState? rs = rm.DebugOnlyGetState();
        p.Run = rs;
        p.Me = rs == null ? null : LocalContext.GetMe(rs);
        if (rs == null || p.Me == null) { p.Detail = "run starting"; p.Key = "RUN_STARTING"; p.Wait = "run state not ready"; return; }

        p.InCombat = CombatManager.Instance.IsInProgress;

        // Modal popups (tutorial FTUEs, confirmations) block the whole UI.
        var modal = NModalContainer.Instance?.OpenModal;
        if (modal != null)
        {
            p.Detail = "modal:" + U.TypeName(modal);
            p.Key = "MODAL:" + p.Detail;
            p.Wait = "a popup is open: dismiss it by hand";
            return;
        }
        // Deck view / settings-like screens the player may open.
        var capstone = NCapstoneContainer.Instance?.CurrentCapstoneScreen;
        if (capstone != null)
        {
            p.Detail = "capstone:" + U.TypeName(capstone);
            p.Key = "CAPSTONE:" + p.Detail;
            p.Wait = "a full-screen view is open: close it";
            return;
        }

        IOverlayScreen? overlay = NOverlayStack.Instance?.Peek();
        p.Overlay = overlay;
        bool mapOpen = NMapScreen.Instance?.IsOpen == true;
        string okey = overlay is GodotObject go ? go.GetInstanceId().ToString() : "";

        if (overlay is NGameOverScreen gos)
        {
            p.Screen = "GAME_OVER"; p.Detail = "NGameOverScreen"; p.Key = "GAME_OVER:" + okey;
            var cont = U.FindFirst<NGameOverContinueButton>(gos);
            var menu = U.FindFirst<NReturnToMainMenuButton>(gos);
            p.Ready = U.Usable(cont) || U.Usable(menu);
            if (!p.Ready) p.Wait = "game over animation";
            return;
        }

        if (!mapOpen && overlay is NCardRewardSelectionScreen crs)
        {
            p.Screen = "CARD_REWARD"; p.Detail = "NCardRewardSelectionScreen"; p.Key = "CARD_REWARD:" + okey;
            p.Ready = U.Visible(crs) && U.FindAll<NGridCardHolder>(crs).Count + U.FindAll<NCardRewardAlternativeButton>(crs).Count > 0;
            if (!p.Ready) p.Wait = "card reward screen animating";
            return;
        }
        if (!mapOpen && overlay is NRewardsScreen rws)
        {
            p.Screen = "REWARDS"; p.Detail = "NRewardsScreen"; p.Key = "REWARDS:" + okey;
            bool anyButton = U.FindAll<NRewardButton>(rws).Any(b => U.Usable(b) && b.Reward != null);
            var proceed = U.FindFirst<NProceedButton>(rws);
            p.Ready = U.Visible(rws) && (anyButton || U.Usable(proceed));
            if (!p.Ready) p.Wait = "rewards screen animating";
            return;
        }
        if (!mapOpen && overlay is NCrystalSphereScreen sph)
        {
            // Crystal Sphere minigame (2026-11-20: purely random, the agent clicks random hidden cells as the
            // game's own AutoSlay CrystalSphereScreenHandler does; the rewards that follow are the agent's choice)
            p.Screen = "CRYSTAL_SPHERE"; p.Detail = "NCrystalSphereScreen";
            int left = SphereDivinations(sph);
            p.Key = "SPHERE:" + okey + ":" + left;
            bool cells = left > 0 && SphereHiddenCells(sph).Count > 0;
            var proceed = U.Get<NProceedButton>(sph, "_proceedButton");
            p.Ready = U.Visible(sph) && (cells || U.Usable(proceed));
            if (!p.Ready) p.Wait = "crystal sphere animating";
            return;
        }
        if (overlay is NCardGridSelectionScreen grid)
        {
            p.Screen = p.InCombat ? "COMBAT_SELECT" : "CARD_SELECT";
            p.Detail = U.TypeName(grid); p.Key = "GRID:" + okey;
            var g = U.Get<NCardGrid>(grid, "_grid");
            p.Ready = U.Visible(grid) && g != null && !g.IsAnimatingOut && U.FindAll<NGridCardHolder>(grid).Count > 0;
            if (!p.Ready) p.Wait = "card grid animating";
            return;
        }
        if (overlay is NChooseACardSelectionScreen choose)
        {
            p.Screen = p.InCombat ? "COMBAT_SELECT" : "CARD_SELECT";
            p.Detail = "NChooseACardSelectionScreen"; p.Key = "CHOOSE:" + okey;
            p.Ready = U.Visible(choose) && U.FindAll<NCardHolder>(choose).Count > 0;
            if (!p.Ready) p.Wait = "choose-a-card screen animating";
            return;
        }
        if (overlay is NChooseARelicSelection relicSel)
        {
            // Not in protocol v1's screen list: reported as OTHER, but with options (relic / skip).
            p.Screen = "OTHER"; p.Detail = "NChooseARelicSelection"; p.Key = "RELICSEL:" + okey;
            p.Ready = U.Visible(relicSel) && U.FindAll<MegaCrit.Sts2.Core.Nodes.Relics.NRelicBasicHolder>(relicSel).Any(h => U.Usable(h));
            if (!p.Ready) p.Wait = "relic selection animating";
            return;
        }
        if (overlay != null && overlay is not NRewardsScreen && overlay is not NCardRewardSelectionScreen && overlay is not NCrystalSphereScreen)
        {
            p.Detail = U.TypeName(overlay); p.Key = "OVERLAY:" + okey;
            p.Wait = "unhandled overlay " + p.Detail;
            return;
        }

        if (mapOpen)
        {
            NMapScreen ms = NMapScreen.Instance!;
            p.Screen = "MAP"; p.Detail = "NMapScreen"; p.Key = "MAP:" + rs.CurrentActIndex + ":" + rs.VisitedMapCoords.Count;
            bool any = U.FindAll<NMapPoint>(ms).Any(mp => mp.State == MapPointState.Travelable && mp.IsEnabled);
            p.Ready = ms.IsTravelEnabled && !ms.IsTraveling && any && !game.Transition.InTransition;
            if (!p.Ready) p.Wait = !ms.IsTravelEnabled ? "map travel disabled" : ms.IsTraveling ? "travelling" : !any ? "no travelable node yet" : "transition";
            return;
        }

        if (p.InCombat)
        {
            NPlayerHand? hand = NPlayerHand.Instance;
            if (hand != null && hand.IsInCardSelection)
            {
                p.Screen = "COMBAT_SELECT"; p.Detail = "hand:" + hand.CurrentMode; p.Key = "HANDSEL:" + rs.TotalFloor;
                p.Ready = hand.ActiveHolders.Count > 0 || U.Usable(U.Get<NConfirmButton>(hand, "_selectModeConfirmButton"));
                if (!p.Ready) p.Wait = "hand selection animating";
                return;
            }
            p.Screen = "COMBAT";
            CombatState? cs = CombatManager.Instance.DebugOnlyGetState();
            p.Detail = cs?.Encounter?.Id.Entry ?? "combat";
            p.Key = "COMBAT:" + rs.TotalFloor + ":" + (cs?.Encounter?.Id.Entry ?? "");
            p.Wait = CombatWait(p.Me, cs, hand);
            p.Ready = p.Wait.Length == 0;
            return;
        }

        var room = rs.CurrentRoom;
        switch (room)
        {
            case CombatRoom:
            {
                var proceed = NCombatRoom.Instance?.ProceedButton;
                p.Detail = "combat room (after combat)";
                p.Key = "POSTCOMBAT:" + rs.TotalFloor;
                if (U.Usable(proceed))
                {
                    p.Screen = "REWARDS"; p.Detail = "NCombatRoom.ProceedButton"; p.Ready = true;
                }
                else p.Wait = "combat ending / rewards not shown yet";
                return;
            }
            case EventRoom er:
            {
                EventModel ev = SafeEvent(er);
                bool ancient = ev is AncientEventModel;
                p.Screen = ancient ? "ANCIENT" : "EVENT";
                p.Detail = ev.Id.Entry;
                p.Key = "EVENT:" + rs.TotalFloor + ":" + ev.Id.Entry;
                var layout = NEventRoom.Instance?.Layout;
                if (layout == null)
                {
                    var fm = FakeMerchantNode();
                    if (fm != null)
                    {
                        p.Screen = "FAKE_MERCHANT";
                        p.Detail = "NFakeMerchant";
                        p.Key = "FAKE_MERCHANT:" + rs.TotalFloor;
                        var proceed = U.Get<NProceedButton>(fm, "_proceedButton") ?? U.FindFirst<NProceedButton>(fm);
                        p.Ready = U.Visible(fm) && !game.Transition.InTransition && (U.Usable(proceed) || fm.Inventory?.IsOpen == true);
                        if (!p.Ready) p.Wait = "fake merchant loading";
                        return;
                    }
                    p.Screen = "OTHER"; p.Detail = "event:" + ev.Id.Entry + " (custom event screen)";
                    p.Wait = "custom event screen: play it by hand";
                    return;
                }
                var buttons = layout.OptionButtons.Where(b => U.Valid(b) && !b.IsQueuedForDeletion()).ToList();
                bool anyEnabled = buttons.Any(b => U.Usable(b));
                bool dialogue = ancient && AncientDialogueActive(layout);
                p.Ready = anyEnabled && !dialogue && !game.Transition.InTransition;
                if (!p.Ready) p.Wait = dialogue ? "ancient dialogue (auto-advanced in agent mode)" : buttons.Count == 0 ? "no event options yet" : "event options disabled / animating";
                return;
            }
            case RestSiteRoom:
            {
                p.Screen = "REST"; p.Detail = "NRestSiteRoom"; p.Key = "REST:" + rs.TotalFloor;
                var rr = NRestSiteRoom.Instance;
                if (rr == null) { p.Wait = "rest site loading"; return; }
                bool anyOpt = rr.Options.Any(o => U.Usable(rr.GetButtonForOption(o)));
                p.Ready = anyOpt || U.Usable(rr.ProceedButton);
                if (!p.Ready) p.Wait = "rest site busy";
                return;
            }
            case MerchantRoom:
            {
                p.Screen = "SHOP"; p.Detail = "NMerchantRoom"; p.Key = "SHOP:" + rs.TotalFloor;
                var mr = NMerchantRoom.Instance;
                if (mr == null || mr.Inventory == null) { p.Wait = "shop loading"; return; }
                bool purchasing = Bridge.ShopPurchaseInFlight;
                p.Ready = U.Visible(mr) && !purchasing && (U.Usable(mr.ProceedButton) || mr.Inventory.IsOpen);
                if (!p.Ready) p.Wait = purchasing ? "purchase in progress" : "shop animating";
                return;
            }
            case TreasureRoom:
            {
                p.Screen = "TREASURE"; p.Detail = "NTreasureRoom"; p.Key = "TREASURE:" + rs.TotalFloor;
                var tr = NRun.Instance?.TreasureRoom;
                if (tr == null) { p.Wait = "treasure room loading"; return; }
                var chest = U.Get<NClickableControl>(tr, "_chestButton");
                bool relics = TreasureRelicHolders(tr).Count > 0;
                p.Ready = U.Usable(chest) || relics || U.Usable(tr.ProceedButton);
                if (!p.Ready) p.Wait = "chest opening";
                return;
            }
            default:
                p.Detail = "room:" + U.TypeName(room);
                p.Key = "ROOM:" + p.Detail + ":" + rs.TotalFloor;
                p.Wait = "unknown room";
                return;
        }
    }

    public static string CombatWait(Player? me, CombatState? cs, NPlayerHand? hand)
    {
        var cm = CombatManager.Instance;
        if (cs == null || me == null) return "no combat state";
        var pcs = me.PlayerCombatState;
        if (pcs == null) return "no player combat state";
        if (cm.IsOverOrEnding) return "combat ending";
        if (cs.CurrentSide != CombatSide.Player) return "enemy turn";
        if (pcs.Phase != PlayerTurnPhase.Play) return "player phase " + pcs.Phase;
        if (cm.PlayerActionsDisabled) return "player actions disabled";
        if (cm.IsPlayerReadyToEndTurn(me)) return "turn ending";
        if (!me.Creature.IsAlive) return "player dead";
        var rm = RunManager.Instance;
        if (rm.ActionExecutor.IsRunning) return "action executing";
        if (!rm.ActionQueueSet.IsEmpty) return "action queue not empty";
        if (hand == null) return "no hand UI";
        if (hand.InCardPlay) return "card play in progress";
        if (hand.CurrentMode != NPlayerHand.Mode.Play) return "hand mode " + hand.CurrentMode;
        var endTurn = NCombatRoom.Instance?.Ui?.EndTurnButton;
        if (endTurn != null && !endTurn.IsEnabled) return "end turn button disabled";
        if (NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0) return "overlay open";
        return "";
    }

    public static EventModel SafeEvent(EventRoom er)
    {
        try { return er.LocalMutableEvent ?? er.CanonicalEvent; } catch { return er.CanonicalEvent; }
    }

    public static bool AncientDialogueActive(NEventLayout layout)
    {
        if (layout is not NAncientEventLayout) return false;
        var hit = U.Get<NClickableControl>(layout, "_dialogueHitbox") ?? layout.GetNodeOrNull<NClickableControl>("%DialogueHitbox");
        return U.Usable(hit);
    }

    public static NClickableControl? AncientDialogueHitbox(NEventLayout layout)
    {
        return U.Get<NClickableControl>(layout, "_dialogueHitbox") ?? layout.GetNodeOrNull<NClickableControl>("%DialogueHitbox");
    }

    public static NFakeMerchant? FakeMerchantNode()
    {
        var er = NEventRoom.Instance;
        if (er == null) return null;
        return er.CustomEventNode as NFakeMerchant ?? U.FindFirst<NFakeMerchant>(er);
    }

    public static List<NTreasureRoomRelicHolder> TreasureRelicHolders(NTreasureRoom tr)
    {
        var coll = U.Get<Node>(tr, "_relicCollection");
        if (coll == null || !U.Visible(coll as CanvasItem)) return new List<NTreasureRoomRelicHolder>();
        return U.FindAll<NTreasureRoomRelicHolder>(coll).Where(h => U.Usable(h) && h.Relic?.Model != null).ToList();
    }
}
