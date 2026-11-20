using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;

namespace AgentBridge;

/// <summary>
/// Multi-step UI sequences (select cards then confirm, open shop then buy, ...). Each step is a click / signal the
/// UI itself would produce, with short pauses so the owner can follow. Runs on the main thread.
/// Returns null on success or an error text (which pauses the agent).
/// </summary>
internal static class Actions
{
    private const int StepMs = 250;

    private static bool Gone(Control screen)
    {
        if (!U.Valid(screen) || !screen.IsInsideTree() || screen.IsQueuedForDeletion()) return true;
        var top = NOverlayStack.Instance?.Peek();
        return top is GodotObject go && go != screen;
    }

    /// <summary>Confirm button to press next: a visible enabled one inside a "*Preview*" container first (the
    /// screens show a preview after the selection), otherwise the screen's main confirm button.</summary>
    private static NClickableControl? NextConfirm(Control screen)
    {
        var buttons = U.FindAll<NConfirmButton>(screen).Where(b => U.Usable(b)).ToList();
        var preview = buttons.FirstOrDefault(b => U.HasAncestorNamed(b, "Preview", screen));
        return preview ?? buttons.FirstOrDefault();
    }

    // ---------------------------------------------------------------- deck / pile grid screens
    public static async Task<string?> GridSelect(NCardGridSelectionScreen screen, List<CardModel> picks)
    {
        var grid = U.Get<NCardGrid>(screen, "_grid");
        foreach (CardModel card in picks)
        {
            if (Gone(screen)) return "selection screen closed before all cards were selected";
            if (Bridge.Halted) return Bridge.HaltText;
            NGridCardHolder? holder = null;
            try { holder = grid?.GetCardHolder(card); } catch { }
            if (grid != null && holder != null)
                grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holder); // what a click on the card emits
            else
                U.Call(screen, "OnCardClicked", card); // card not laid out (scrolled away): call the same handler
            await Bridge.Delay(StepMs);
        }
        return await ConfirmUntilGone(screen, "card selection");
    }

    private static async Task<string?> ConfirmUntilGone(Control screen, string what)
    {
        for (int i = 0; i < 16; i++)
        {
            if (Gone(screen)) return null;
            if (Bridge.Halted) return Bridge.HaltText;
            var btn = NextConfirm(screen);
            if (btn != null)
            {
                btn.ForceClick();
                await Bridge.Delay(StepMs + 150);
                continue;
            }
            await Bridge.Delay(StepMs);
        }
        return Gone(screen) ? null : what + " screen did not close (no enabled confirm button)";
    }

    // ---------------------------------------------------------------- choose one of N generated cards
    public static async Task<string?> ChooseACard(NChooseACardSelectionScreen screen, CardModel card)
    {
        if (Gone(screen)) return "choose-a-card screen closed";
        var holder = U.FindAll<NCardHolder>(screen).FirstOrDefault(h => h.CardModel == card);
        if (holder == null) return "card holder not found";
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
        await Bridge.Delay(StepMs);
        return null;
    }

    // ---------------------------------------------------------------- in-combat hand selection
    public static async Task<string?> HandSelect(NPlayerHand hand, List<CardModel> picks)
    {
        foreach (CardModel card in picks)
        {
            if (!U.Valid(hand) || !hand.IsInCardSelection) return "hand selection ended before all cards were selected";
            if (Bridge.Halted) return Bridge.HaltText;
            var holder = hand.ActiveHolders.FirstOrDefault(h => h.CardModel == card);
            if (holder == null) return "card " + card.Id.Entry + " is not selectable in hand";
            holder.EmitSignal(NCardHolder.SignalName.Pressed, holder); // -> NPlayerHand.OnHolderPressed
            await Bridge.Delay(StepMs);
        }
        for (int i = 0; i < 12; i++)
        {
            if (!U.Valid(hand) || !hand.IsInCardSelection) return null;
            if (Bridge.Halted) return Bridge.HaltText;
            var confirm = U.Get<NClickableControl>(hand, "_selectModeConfirmButton") ?? hand.GetNodeOrNull<NClickableControl>("%SelectModeConfirmButton");
            if (confirm != null && U.Valid(confirm) && confirm.IsEnabled)
            {
                confirm.ForceClick();
                await Bridge.Delay(StepMs + 150);
                continue;
            }
            await Bridge.Delay(StepMs);
        }
        return hand.IsInCardSelection ? "hand selection did not finish (confirm button not enabled)" : null;
    }

    // ---------------------------------------------------------------- shop
    public static async Task<string?> ShopBuy(NMerchantRoom room, MerchantEntry entry)
    {
        if (!U.Valid(room)) return "shop closed";
        if (!entry.IsStocked) return "item sold out";
        if (!entry.EnoughGold) return "not enough gold";
        if (room.Inventory != null && !room.Inventory.IsOpen)
        {
            room.OpenInventory(); // what clicking the merchant does
            await Bridge.Delay(700);
        }
        // Same call the shop slot makes on click (NMerchantSlot.OnSelected). Card removal opens a deck selection
        // screen and the purchase task only completes after it, so it is not awaited here.
        Task<bool> t = RunManager.Instance.MerchantSynchronizer.PurchaseLocalEntry(entry);
        Bridge.TrackShopPurchase(t);
        await Bridge.Delay(400);
        if (t.IsCompleted && !t.IsFaulted && !t.Result) return "the shop refused the purchase";
        if (t.IsFaulted) return "purchase failed: " + t.Exception?.GetBaseException().Message;
        return null;
    }

    public static async Task<string?> ShopLeave(NMerchantRoom room)
    {
        if (!U.Valid(room)) return "shop closed";
        if (room.Inventory != null && room.Inventory.IsOpen)
        {
            var back = U.FindAll<NBackButton>(room).FirstOrDefault(b => U.Usable(b));
            if (back != null) back.ForceClick(); else room.Inventory.Close();
            await Bridge.Delay(500);
        }
        for (int i = 0; i < 10; i++)
        {
            if (!U.Valid(room) || !room.IsInsideTree()) return null;
            if (Bridge.Halted) return Bridge.HaltText;
            if (U.Usable(room.ProceedButton))
            {
                room.ProceedButton.ForceClick();
                await Bridge.Delay(200);
                return null;
            }
            await Bridge.Delay(StepMs);
        }
        return "shop proceed button never became enabled";
    }

    // ---------------------------------------------------------------- game over
    public static async Task<string?> GameOverToMenu(NGameOverScreen screen)
    {
        for (int i = 0; i < 400; i++) // up to ~100 s: the summary animation is long
        {
            if (!U.Valid(screen) || !screen.IsInsideTree()) return null;
            if (Bridge.Halted) return Bridge.HaltText;
            var menu = U.FindFirst<NReturnToMainMenuButton>(screen);
            if (U.Usable(menu))
            {
                menu!.ForceClick();
                await Bridge.Delay(500);
                return null;
            }
            var cont = U.FindFirst<NGameOverContinueButton>(screen);
            if (U.Usable(cont))
            {
                cont!.ForceClick();
                await Bridge.Delay(800);
                continue;
            }
            await Bridge.Delay(250);
        }
        return "game over screen: main menu button never appeared";
    }
}
