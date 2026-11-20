# Findings

This file keeps the lessons of the ic113 phase (Ironclad A10 on game v0.113.1). Each finding gives its evidence and
links to the experiment in [experiments.md](experiments.md). Numbers are in [results.md](results.md). Terms such as
V_run, T_f and segment are defined at the top of experiments.md. The last section lists the lessons carried over from
the Defect phase (branch `v0.111-defect`, `doc/findings.md`).

## Lessons of the ic113 phase

### 1. Networks move across game versions if ids are remapped, but not across characters

**Evidence.** Converted by id, the Defect networks played Defect on v0.113 at 46.6% (CI 43.6-49.7) against 44.8% on
v0.111. The same networks won 0% of 512 Ironclad A10 runs and 5.1% at A0. A human-trained macro network did no better
(0% / 2.4%). [Experiment 3.]

**Implication.** Vocabularies must be remapped by id, never by position: new content shifts every sorted index. A
version change is cheap once that is done. A character change is not: the combat network must learn the new cards
before full-run training can produce wins.

### 2. A short combat warm-up on human battle starts bridges a new character

**Evidence.** 4.1 GPU hours of fight self-play from 372,680 human Ironclad battle starts lifted held-out elite and boss
starts from 59.8% to 85.1% (+25.3 pp). Full runs went from 0% to 3.1% at A10 and from 5.1% to 52.9% at A0. The
hourly harness read 61.5 / 85.6 / 86.8 / 85.0 / 87.3%: almost all of the gain came in the first hour. The chain that
started from this network reached 23.3% at its first evaluation. [Experiments 4, 5.]

**Implication.** Human battle starts are a cheap curriculum for combat: they put the network in realistic mid-run
fights at once. One hour is enough to make full-run self-play productive. The macro network did not need a human
warm-up; the converted Defect macro network was good enough to start from.

### 3. A tiny value bias plus min-max normalisation can stall a fight forever

**Evidence.** Against one harmless slime, the network valued the open fight a few thousandths above the exact value
of winning now. Min-max normalisation at the root turned that gap into a hard "do not kill" policy target. Stalled
fights were never trained on, so nothing pushed back. Self-play stalemates grew from 149 to 269 per segment. Capping
searched values at the exact win-now value (`win_bound=0.001`) took replayed stalemate seeds from 6 stalemates to 0
and self-play from 269 to 1, with the same end HP. [Experiment 6.]

**Implication.** When a search compares exact terminal values with network estimates, an upward bias in the network
can beat every terminal move. Where no future state can be worth more than winning now, enforce that bound in the
search. A per-round time cost would also work, but it must be larger than the bias, and then it trades HP for speed.

### 4. The act-3 bosses are the wall, and more search does not move it

**Evidence.** In the segment-29 evaluation the three act-3 bosses cause 90 of 148 deaths (60.8%). In T3's 320 runs
they caused 118 of 199 (59.3%). On 904 paired act-3 boss fights, 128 and 256 simulations did not beat 64 (+0.22 pp
and +0.55 pp, both n.s.); on the final boss they were slightly worse (-0.23 and -0.94 pp). The only measurable gain was
about 1.4 HP. [Experiment 14; results "Segment 29 in detail".]

**Implication.** At 64 simulations the search already gets what the current combat network can give. Act-3 boss
wins must come from training: a better combat network, better HP and potion preparation, more act-3 boss practice.
This is why the chain added act-3 boss practice and a larger shadow combat network in segment 29.

### 5. The run value is too flat where it matters, and one member makes it flatter

**Evidence.** After act-3 boss 1 at A10, the actual win rate ranges from 4.5% to 87.2% across HP fifths, but V_run
predicts 16.5% to 71.0%. Low HP is overpriced by about 12 pp and high HP underpriced by 14-16 pp. Member 0, the
member that also trains the policy, predicts 28.3% to 66.0% where members 1-4 predict 9.8% to 79.9%. The network uses
34.4% of the possible improvement over a 9-cell table on its training window but only 8.9% on later runs. Milestone
heads cut the held-out value loss by 0.015 nats (0.071 nats on the logit average that T_f uses) and narrowed the gap.
[Experiments 11, 12.]

**Implication.** One value head that mixes banked act progress with the final boss's steep HP curve flattens that
curve. Separate heads per milestone fit it better offline. Whether this changes play was not measured: the chain
switched milestone heads on without a fork test (experiment 13).

### 6. A better price does not help while the fighter cannot act on it

**Evidence.** Pricing T_f with members 1-4 only (steeper HP values) and a measured potion table each changed play
the intended way: the agent entered the final boss with more HP (0.616 vs 0.592 of max) and more potions (up to 1.66
vs 1.35), and won it more often (up to 64% vs 58%). But it paid for this at act-3 boss 1: 12-14 more deaths there. Net
effect on 320 paired seeds: -2.2, 0.0 and +0.6 pp. [Experiment 16.]

**Implication.** The same result appeared in the Defect phase (finding 14 there). With a frozen combat network,
re-pricing fights moves losses from one fight to another. Price changes should be tested through training, or
together with a stronger combat network.

### 7. Distilling a lookahead with forward KL collapses the policy

**Evidence.** 75.6% of PPO's initial behaviour-lag KL came from the card-reward lookahead (`look_beta=20`). Training
the card head toward the lookahead distribution with forward KL halved its logit range (9.42 -> 4.35), doubled its
entropy (0.362 -> 0.752), and lowered its agreement with the lookahead by 7.8-12.8 pp. Each update made it worse
(agreement 0.871 -> 0.687 over five updates). [Experiment 12, arm M3.]

**Implication.** Forward KL covers modes: against a noisy target that disagrees on a fifth of the rows, the cheapest
answer is to flatten. Cleaner options are to record the raw log-probability so PPO's ratio stays on-policy, or to
distill a target recorded by the actors with reverse KL. A residual adapter that feeds member 0 from members 1-4
(P2) also failed: it cost 31% of PPO steps for no gain in card agreement.

### 8. One 300-seed panel per segment cannot rank neighbouring segments

**Evidence.** Neighbouring segments moved by 6-8 pp on the panel (40.6% -> 46.7% -> 46.7% -> 38.3% for segments
24-27) while self-play A10 stayed at 24.0-24.8% on 3,273-5,103 A10 runs each. The CI of one reading is about +-5.5 pp.
Segment 29's 50.7% is the highest of 23 readings on the same seeds, so it is biased upward. [Results "What the numbers
do not show".]

**Implication.** Use the panel for trends and large steps, and self-play A10 for smaller trends. Confirm any release
or setting change on a fresh, paired panel of about 1,000 seeds, and replicate it on a second fresh panel (the Defect
phase once saw a 1,024-seed arm 5 points off).

### 9. On an interruptible server, only what is on disk counts

**Evidence.** Stops came without a signal. Periodic carries every 10 minutes (about 1 s per save) and adoption of
interrupted segments saved 9 segments. 3 segments were interrupted before their first carry, 1 was lost with its
host, and 3 failed after a restore because the restored tree lacked a Python module. The run counter fell from
314,598 to 294,723 at the restore, because it counts only backed-up run records. [Experiment 8.]

**Implication.** Write state often and atomically, adopt partial work, back up carries off the machine, and test
the restore path end to end, including imports. After a restore, watch the first segment; three identical failures
in a row point to a broken tree, not bad luck. Say what a counter counts.

### 10. More actors can be slower

**Evidence.** On a machine with more cores, 84 actors and 3 combat servers played 17.9k runs/h; 60 actors and 2
servers played 19.8k. The GPU was at its power cap and actors waited on it 46% of the time instead of 30%.
[Experiment 10.]

**Implication.** Check where the bottleneck is before scaling actors. When the GPU is saturated, more actors only
queue.

### 11. Every new character or version hides untrained decision paths

**Evidence.** About 0.67 run-level card prompts per run were answered at random because no head handled them
(Self-Help Book, Symbiote, Spiraling Whirlpool, Lord's Parasol, ...). The Fake Merchant event always left by the
default option, so the 11.8% of runs that entered it never bought anything. New rest options and events start with
untrained embeddings. [Experiments 5, 9.]

**Implication.** After a port, list every place where the agent falls back to a default or a random answer (log
them by source) and route each to the closest trained head. The Defect phase found the same class of bug in prompts
that open below the root (finding 16 there).

### 12. Human data needs the same filters as the target setting

**Evidence.** The battle-start export kept non-standard runs: 40% of v0.113 A10 starts were Dread mode. The warm-up's
coverage mutation drew from a pool with 5 multiplayer-only cards, so about 6.5% of training starts held a card that a
single-player run never sees; two of them cannot be played solo at all. [Experiments 3, 4.]

**Implication.** Filter human data by game mode, modifiers and mods, not only by version, character and ascension.
Build synthetic variations only from content the target setting can produce.

### 13. Value-based potion prices keep stale priors for content the character never sees

**Evidence.** The phase-1 potion rule ranks potions by their mean V_run difference. At segment 30 it still placed
Focus Potion third in act 3 (+4.90 points), although it is not among the 46 potions seen in Ironclad belts. A
measured probe on 80 final-boss starts ranked Entropic Brew, Fairy in a Bottle and Heart of Iron at the top (+17.5,
+15.0, +13.8 points), with standard errors of 5-6 points. [Experiments 5, 16.]

**Implication.** Embeddings of content that never appears are never corrected. They do no harm while that content is
never offered, but averages over all ids should be limited to content the character can obtain. Measured tables need
far more than 80 starts to rank potions reliably.

### 14. The simulator-trained agent wins in the real game

**Evidence.** Three finished live runs in the real game, all won: A0 and A6 with segment-21 weights, A10 with
segment-29 weights. The live runs exposed harness gaps that the simulator never showed: a threading bug, the ending
scene, a minigame, and a game-over screen that shows HP 0 after a win. [Experiment 19.]

**Implication.** The simulator is close enough to the game for this agent to win there. Three runs do not measure a
win rate; a live estimate needs many more runs. The live harness needs its own tests against the real game.

## Carried over from the Defect phase (v0.111)

These findings from branch `v0.111-defect` shaped the ic113 recipe and still hold.

| Defect finding | What ic113 kept |
|---|---|
| 1. A pure win target cannot start a run-value network | The value target mixes wins with acts cleared (`alpha` 0.70, raised to 0.85 in segment 29) |
| 2. A value gradient into a shared policy encoder breaks PPO's trust region | The policy member's value learning stays head-only |
| 4. Human data ranks states but sets the wrong prices for a strong agent | No human refresh of the macro network; prices come from self-play |
| 5-6. The value memorises runs; rows per run bias it | 50k-run window, 16% row share, dropout and weight decay on members 1-4, probability averaging |
| 8. Sixteen root candidates are too many in self-play | 4 root candidates in self-play, 16 in evaluation |
| 10. Search pays off most on bosses | Hallways at half search, every boss move fully searched |
| 14. Better-calibrated values did not improve frozen-network play | Seen again (finding 6 above) |
| 16. Decision-surface bugs hide outside the root | Seen again after the port (finding 11 above) |
| 18. Evaluation noise can reverse decisions | 300 seeds per segment; fresh 1,000-seed panels for decisions (finding 8 above) |
| Shadow trainer method | A new combat network trains beside the playing one before any switch (experiment 18) |
