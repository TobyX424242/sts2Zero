using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;

namespace AgentBridge;

/// <summary>
/// Starts a fresh single-player Ironclad run at any ascension 0-10 through the game's own run-setup path:
/// the same calls NCharacterSelectScreen.StartNewSingleplayerRun makes (Player.CreateForNewRun, act list from the
/// lobby's "act_selection" RNG, NGame.StartNewSingleplayerRun with shouldSave=true, GameMode.Standard). Only the
/// lobby's ascension clamp (StartRunLobby.BeginRunLocally: min(ascension, MaxAscension)) is skipped.
/// </summary>
internal static class RunStarter
{
    public static string? CanStart()
    {
        NGame? game = NGame.Instance;
        if (game == null) return "game not ready";
        if (RunManager.Instance.IsInProgress) return "a run is in progress: finish or abandon it first";
        if (game.MainMenu == null) return "not on the main menu";
        if (game.Transition.InTransition) return "screen transition in progress";
        try
        {
            if (SaveManager.Instance.HasRunSave)
                return "a saved run exists: continue it, or abandon it in the main menu, before starting an agent run";
        }
        catch (Exception e) { return "cannot read save state: " + e.Message; }
        return null;
    }

    public static async Task<string?> Start(string character, int ascension, string? seedText)
    {
        if (!string.Equals(character, "IRONCLAD", StringComparison.OrdinalIgnoreCase))
            return "only IRONCLAD is supported";
        if (ascension < 0 || ascension > 10) return "ascension must be 0..10";
        string? why = CanStart();
        if (why != null) return why;

        string seed = string.IsNullOrWhiteSpace(seedText) ? SeedHelper.GetRandomSeed() : SeedHelper.CanonicalizeSeed(seedText.Trim());
        U.Info($"starting agent run: IRONCLAD A{ascension} seed {seed}");

        Player player;
        List<ActModel> acts;
        try
        {
            var unlocks = new UnlockState(SaveManager.Instance.Progress);
            var rng = new Rng((ulong)StringHelper.GetDeterministicHashCode(seed), "act_selection");
            acts = ActModel.GetRandomList(rng, unlocks, false).ToList();
            player = Player.CreateForNewRun<Ironclad>(unlocks, 1uL); // NetSingleplayerGameService.NetId == 1
        }
        catch (Exception e)
        {
            U.Error("run setup failed: " + e);
            return "run setup failed: " + e.Message;
        }

        try
        {
            try { await NGame.Instance!.Transition.FadeOut(0.8f); }
            catch (Exception e) { U.Warn("fade out failed (continuing): " + e.Message); }
            await NGame.Instance!.StartNewSingleplayerRun(player, shouldSave: true, acts, Array.Empty<ModifierModel>(), seed,
                GameMode.Standard, isCustomGame: false, ascension);
        }
        catch (Exception e)
        {
            U.Error("exception starting the run: " + e);
            try { await NGame.Instance!.ReturnToMainMenuWithInternalError(e); } catch { }
            return "exception starting the run: " + e.Message;
        }
        return null;
    }
}
