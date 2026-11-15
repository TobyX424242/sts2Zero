using System.Text.Json;
using Sts2Sim.Core.Combat;
using Sts2Sim.Core.Models;
using Sts2Sim.Core.MonsterMoves.Intents;

namespace Sts2Sim.S0;

/// <summary>
/// Frozen integer layout (v4 shape; v5 values opt-in) of the combat observation (mirrored in src/s0/py/combat_obs.py).
/// <para>Observation: int32[<see cref="MaxTokens"/>, <see cref="TokenWidth"/>]; row = [kind, id, id2, f0..f7].
/// <c>id</c> indexes the vocabulary of the token kind's category (hand/draw/discard/exhaust: card; enemy: monster;
/// enemy/player power: power; orb: orb; potion: potion; relic: relic; global: 0), <c>id2</c> the secondary category
/// (cards: enchantment; enemy: intent). In every category 0 = &lt;pad&gt;/none, 1 = &lt;unk&gt;. Kind 0 = padding.</para>
/// <para>Actions: int32[<see cref="MaxActions"/>, <see cref="ActionWidth"/>]; row = [kind+1 (0 = padding; 1 end,
/// 2 card, 3 potion), source token index (hand / potion token; global token 0 for end turn), target token index
/// (enemy token, or the global token 0 for the player, -1 none), prompt code (0 = no prompt answer given,
/// 1 + number of chosen cards over all prompts), then for the first <see cref="MaxSelCards"/> chosen cards:
/// card vocabulary id, upgrade level + 1 (0 = slot unused)].</para>
/// <para>v4 (2026-11-15): new values in feature slots that were always 0 in v3 (same width): Hand f5 = current damage,
/// f6 = current block of the card; Draw f4 / f5 and Discard, Exhaust f3 / f4 = damage / block (the group's largest,
/// see <see cref="CombatObservation.SplitPilesByValue"/>); EnemyPower f2 and PlayerPower f1 = the power's hidden
/// counter (<see cref="PowerCounters"/>; e.g. WitheringPresence: cards left before the next Wither). Networks without
/// config "ext" multiply the features by the v3 slot mask, so they see exactly their v3 input; v3 chunks have zeros
/// there.</para>
/// <para>v3 (2026-11-11): optional Context token (kind 12) right after the global token, features from the fight's T_f table
/// (src/s2/S2/Evaluator.cs FightSignals); potion tokens' feature 1 = the potion's marginal T_f.</para>
/// <para>v5 (2026-11-15, opt-in, no shape change): Enemy f5 may carry the intent damage after Strength / Weak /
/// Vulnerable (<see cref="CombatObservation.ModifiedIntentDamage"/>) and Relic f0/f1 the relic counters
/// (<see cref="CombatObservation.RelicCounters"/>). The Python mask keeps the relic slots out of v3/v4 networks; the
/// intent damage changes an existing slot by design.</para>
/// <para>v2 (2026-11-11): action rows carry the chosen cards of prompt answers (v1 had only the prompt count, so two
/// answers to the same prompt were indistinguishable to the network).</para>
/// </summary>
public static class ObsLayout
{
    public const int Version = 4, MaxTokens = 128, TokenWidth = 3 + CombatObservation.FeatureCount, MaxActions = 64, MaxSelCards = 2,
        ActionWidth = 4 + 2 * MaxSelCards;

    public static readonly string[] Categories = { "card", "enchantment", "monster", "intent", "power", "orb", "potion", "relic" };

    public static (string Primary, string Secondary) CategoriesOf(TokenKind k) => k switch
    {
        TokenKind.Hand or TokenKind.Draw or TokenKind.Discard or TokenKind.Exhaust => ("card", "enchantment"),
        TokenKind.Enemy => ("monster", "intent"),
        TokenKind.EnemyPower or TokenKind.PlayerPower => ("power", ""),
        TokenKind.Orb => ("orb", ""),
        TokenKind.Potion => ("potion", ""),
        TokenKind.Relic => ("relic", ""),
        _ => ("", ""),
    };

    /// <summary>Vocabulary from the simulator's content registry: per category ["&lt;pad&gt;", "&lt;unk&gt;", sorted ids].</summary>
    public static Dictionary<string, List<string>> BuildVocab()
    {
        List<string> V(IEnumerable<string> ids) => new[] { "<pad>", "<unk>" }.Concat(ids.Distinct().Order(StringComparer.Ordinal)).ToList();
        return new Dictionary<string, List<string>>
        {
            ["card"] = V(ModelDb.All<CardModel>().Select(x => x.Id.ToString())),
            ["enchantment"] = V(ModelDb.All<EnchantmentModel>().Select(x => x.Id.ToString())),
            ["monster"] = V(ModelDb.All<MonsterModel>().Select(x => x.Id.ToString())),
            ["intent"] = V(Enum.GetNames<IntentType>().Select(n => "INTENT." + n)),
            ["power"] = V(ModelDb.All<PowerModel>().Select(x => x.Id.ToString())),
            ["orb"] = V(ModelDb.All<OrbModel>().Select(x => x.Id.ToString())),
            ["potion"] = V(ModelDb.All<PotionModel>().Select(x => x.Id.ToString())),
            ["relic"] = V(ModelDb.All<RelicModel>().Select(x => x.Id.ToString())),
        };
    }

    public sealed class Vocab
    {
        private readonly Dictionary<string, Dictionary<string, int>> _maps;
        public Dictionary<string, List<string>> Lists { get; }
        public long Unknown;
        public Vocab(Dictionary<string, List<string>> lists)
        {
            Lists = lists;
            _maps = lists.ToDictionary(kv => kv.Key, kv => kv.Value.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i, StringComparer.Ordinal));
        }
        public static Vocab Load(string path) => new(JsonSerializer.Deserialize<Dictionary<string, List<string>>>(File.ReadAllText(path))!
            .Where(kv => kv.Key != "_meta").ToDictionary(kv => kv.Key, kv => kv.Value));
        public void Save(string path)
        {
            var o = new Dictionary<string, object>(Lists.ToDictionary(kv => kv.Key, kv => (object)kv.Value));
            File.WriteAllText(path, JsonSerializer.Serialize(o));
        }
        public int Id(string category, string id)
        {
            if (category == "" || id == "") return 0;
            if (_maps[category].TryGetValue(id, out int i)) return i;
            Unknown++;
            return 1;
        }
    }

    /// <summary>Writes one state (tokens + legal actions) into the int32 buffers at the given offsets.</summary>
    public static void Write(List<ObsToken> tokens, IReadOnlyList<SearchAction> actions, CombatState state, Vocab vocab,
        Span<int> obs, Span<int> act)
    {
        obs.Clear();
        act.Clear();
        int handBase = -1, potionBase = -1, enemyBase = -1;
        for (int i = 0; i < Math.Min(tokens.Count, MaxTokens); i++)
        {
            ObsToken t = tokens[i];
            var (c1, c2) = CategoriesOf(t.Kind);
            Span<int> row = obs.Slice(i * TokenWidth, TokenWidth);
            row[0] = (int)t.Kind;
            row[1] = vocab.Id(c1, t.Id);
            row[2] = vocab.Id(c2, t.Id2);
            for (int f = 0; f < CombatObservation.FeatureCount; f++) row[3 + f] = t.F[f];
            if (t.Kind == TokenKind.Hand && handBase < 0) handBase = i;
            if (t.Kind == TokenKind.Potion && potionBase < 0) potionBase = i;
            if (t.Kind == TokenKind.Enemy && enemyBase < 0) enemyBase = i;
        }
        // token index lookups: hand tokens are in hand order (F0 = hand position); potion tokens carry the slot (F0);
        // enemy tokens carry the enemy slot (F0 = index in state.Enemies)
        var potionTok = new Dictionary<int, int>();
        var enemyTok = new Dictionary<uint, int>();
        for (int i = 0; i < Math.Min(tokens.Count, MaxTokens); i++)
        {
            if (tokens[i].Kind == TokenKind.Potion) potionTok[tokens[i].F[0]] = i;
            if (tokens[i].Kind == TokenKind.Enemy && state.Enemies[tokens[i].F[0]].CombatId is { } cid) enemyTok[cid] = i;
        }
        uint? playerId = state.Players[0].Creature.CombatId;
        for (int j = 0; j < Math.Min(actions.Count, MaxActions); j++)
        {
            SearchAction a = actions[j];
            Span<int> row = act.Slice(j * ActionWidth, ActionWidth);
            row[0] = (int)a.Kind + 1;
            row[1] = a.Kind switch
            {
                ActionKind.PlayCard => handBase >= 0 ? handBase + a.SourceIndex : 0,
                ActionKind.UsePotion => potionTok.GetValueOrDefault(a.SourceIndex, 0),
                _ => 0,
            };
            row[2] = a.TargetId is not { } tid ? -1 : tid == playerId ? 0 : enemyTok.GetValueOrDefault(tid, -1);
            if (a.Selections.Count > 0)
            {
                int nCards = 0;
                foreach (string[] sel in a.Selections)
                    foreach (string key in sel)
                    {
                        if (nCards < MaxSelCards)
                        {
                            var (cid, up) = ParseSelectionKey(key);
                            row[4 + 2 * nCards] = vocab.Id("card", cid);
                            row[5 + 2 * nCards] = up + 1;
                        }
                        nCards++;
                    }
                row[3] = 1 + nCards;
            }
        }
    }

    /// <summary>Card id and upgrade level of a selection key (<see cref="CardSelectionOptions.EquivalenceKey"/>
    /// "CARD.X#up#cost#..." or a bundle key "bundle:CARD.A+CARD.B", which maps to its first card, upgrade 0).</summary>
    public static (string Id, int Upgrade) ParseSelectionKey(string key)
    {
        if (key.StartsWith("bundle:", StringComparison.Ordinal))
        {
            string rest = key["bundle:".Length..];
            int plus = rest.IndexOf('+');
            return (plus < 0 ? rest : rest[..plus], 0);
        }
        int h1 = key.IndexOf('#');
        if (h1 < 0) return (key, 0);
        int h2 = key.IndexOf('#', h1 + 1);
        int up = int.TryParse(h2 < 0 ? key[(h1 + 1)..] : key[(h1 + 1)..h2], out int u) ? u : 0;
        return (key[..h1], up);
    }
}
