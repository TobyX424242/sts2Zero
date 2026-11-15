# Findings

The experiments are in `doc/experiments.md`; this file keeps the conclusions that changed the design. Each finding
is stated with the evidence that produced it and what it implies for a future run.

## 1. A pure win target cannot start a run-value network

**Evidence.** In the first S3 test the agent won 0 of 1,024 runs. With the target `P(win)` alone, V_run went to ~0
within two updates everywhere; T_f and the PPO advantages had no signal. Switching to
`alpha * won + (1-alpha) * acts_cleared / 3` with `alpha = win rate / 0.2` restored calibration (0.336 predicted vs
0.333 observed after 4,600 runs).

**Implication.** The value target needs a dense progress component until the agent wins about a fifth of its runs.
`alpha` should be computed from the current window, not fixed. This is decision D14.

## 2. A value gradient into a policy-shared encoder leaves PPO's trust region

**Evidence.** In the first minutes of the server run, the policy member's value loss updated the encoder it shares
with the policy. Behaviour-lag KL reached 0.41 and KL to the S1 policy 0.23 after four updates; PPO stopped after
3 steps. Detaching the state vector for the policy member's value loss dropped lag KL to 0.004-0.006 and the KL to S1
to 0.008-0.013, and PPO completed its steps.

**Implication.** In a shared-trunk actor-critic, the critic loss is not automatically safe for the policy. Keep the
policy member's value learning head-only, or give the critic its own trunk. This is a general RL finding, not an
S3 quirk.

## 3. Monotone HP is necessary, but the policy still needs HP

**Evidence.** S1 removed HP from the encoder so the value could be monotone by construction (0 violations in 39,850
probes). The S1 policy query therefore could not see HP; after 40 minutes of S3 training the policy rested 92% of the
time. S3 adds the 26 HP ramps to the policy query through a zero-initialised layer.

**Implication.** A structural constraint on the value head should not silently remove an input from the policy head.
Check the policy's input coverage separately from the value's monotonicity.

## 4. Human data ranks, but its scale and prices are wrong for a strong agent

**Evidence.** The POC model transferred in ranking to a 70.9% agent's states (within-floor AUC 0.670 vs 0.676 on
human states) but predicted 39% wins where the true rate was 74%; after recalibration, +10 HP was worth 1.1-1.4
points against 1.7 / 2.9 / 3.9 raw. The S1 model's human HP price was 2.2 / 3.5 / 4.2 points per 10 HP by act.

**Implication.** Use human data to initialise representations, not to set the prices of a stronger agent. This is why
S3 retrains V_run on its own outcomes and rebuilds T_f at every fight. The "value is a prediction, not a price" issue
is real; perturbed fight starts are the intended causal correction.

## 5. The S3 value network memorised runs, and a constant predictor beat it on fresh runs

**Evidence.** From segment 9 on, the value loss on fresh runs was worse than a single constant (segment 22: 0.718 vs
0.632 for the 9-cell table). Training loss 0.52-0.58 was below any honest predictor. The five members disagreed by
1.8-2.8 logits at A5-A8. Causes: ~95 rows per run with the same label, a 4,096-run window (8 minutes), dropout off
during value training, plain Adam, and sigmoid(mean logit) amplifying overconfidence.

**Implication.** A large window is not enough by itself; the row mix and regularisation matter. The working
configuration is 50k runs, at most 16 rows per run (later a 16% row share), 150 value steps, dropout + weight decay
on members 1-4, and probability averaging. Fresh loss fell to 0.640, slightly better than the table.

## 6. Rows per run create a length bias that looks like a value bias

**Evidence.** A run dying in act 1 has 33 floor/decision rows; a won run has 120. Keeping 16 rows per run therefore
overweights short runs 3.6x among act-1 rows. Value minus outcome in act 1 on fresh runs was -0.049 (A10), -0.070
(A9), -0.072 (A0-8) before the fix. Keeping a fixed 16% share of each run's rows moved these to +0.005 / +0.001 /
-0.021 and improved fresh loss.

**Implication.** When a trajectory contributes many rows with one outcome label, row sampling is a causal
intervention on the value's calibration. Log the row mix and check per-act bias, not only overall loss.

## 7. The combat value head overfits

**Evidence.** At 32.8k steps on a frozen stream, training survival loss was 0.05 while held-out was 0.16-0.18; the
from-scratch arms' held-out end-HP loss rose while training loss fell. The larger d256 x 4 network learned faster per
sample but did not approach the trained network in 1.5 h.

**Implication.** More capacity is not the current bottleneck; the value target/replay distribution is. Candidate
fixes: stronger regularisation, more diverse chunks, a value target that is less fight-specific, or a true held-out
early stop. The shadow trainer is the safe way to test a change without risking the playing network.

## 8. Sixteen root candidates are too many for four to eight samples per candidate

**Evidence.** With 64 simulations and 16 candidates, the first halving round gives each candidate one simulation in
one resampled world, then drops half of them; min-max normalisation pushes the best single sample to 5-6 logits above
the worst. Four candidates improved mean relative T_f by +0.0075 at 64 simulations (CI +0.004 to +0.011) and +0.0125
at 8 simulations; eight candidates was +0.001 n.s.; two was -0.015.

**Implication.** The mctx default (16) comes from deterministic games. In a stochastic game with one world per
simulation, the root candidate count is a variance control. Four is the current setting.

## 9. Hidden information costs about 0.026 relative T_f, and the true future is an upper bound, not a target

**Evidence.** G2: search 64 0.594, true-future search 64 0.620 (-0.026 for hidden information, about 2.1 win points
in that comparison). The exact clone is useful as a diagnostic but must never be used for training or deployment.

**Implication.** Report the gap, but do not "fix" it by leaking the future. The gap is the value of information; it
shrinks with a better value/policy, not with a stronger search alone (16x more search added only ~5 boss points).

## 10. MCTS beats one-turn search, but only by a little; search pays off most on bosses

**Evidence.** G2: one-turn 0.585 vs MCTS 0.594 (-0.009, CI -0.012 to -0.006). Search allocation: halving hallway
search cost nothing (69% of simulations), every boss move at 64 gained +0.064 bosses per run at 103% of the
simulations; doubling boss search instead gained +0.046 at 95%.

**Implication.** Keep MCTS, but spend the budget where the game is decided. Hallway fights can be searched cheaply.

## 11. The outcome head tied with the scalar head in play; its value is the separation of prices from weights

**Evidence.** A20: outcome - scalar rel. T_f was -0.004 (network alone), -0.002 (search 16), -0.006 (search 64),
-0.001 (one-turn), +0.002 (true future); none significant. The outcome head used 5-7 fewer potion points. Re-pricing
without retraining: both heads followed new tables (potion use 95-99% when worthless, 28-34% at 3x).

**Implication.** Do not expect a learning-speed win from the outcome head. The structural win is that T_f can be
rebuilt from a retrained V_run without relabelling combat data; test that property, not just the loss.

## 12. The boss gap is mostly the agent's judgement, not the simulator

**Evidence.** With true-future search 1,024 the agent won 76.1% of 117 held-out boss fights (human 83.8%), and it
matched or beat humans on Queen, Kaiser Crab, Lagavulin Matriarch, The Kin, Vantom, Soul Fysh and The Insatiable.
A code review of Test Subject, Queen, Aeonglass and Knowledge Demon against the decompiled v0.111.0 game found no
difference that changes a single-player fight (HP, damage, ascension values, phases). The remaining suspicious bosses
were not explained by the simulator.

**Implication.** Invest in longer-horizon combat value and boss-specific data/search, not in rewriting the bosses. But
keep the caveat: the simulator is not byte-exact, and the review was code reading, not replay parity.

## 13. At A10, the run value has very little headroom over a 9-cell table

**Evidence.** The value beat the table by 0.007 log loss. A predictor that knows every run's outcome could beat it by
only 0.086 / 0.061 / 0.043 in acts 1 / 2 / 3; the network takes about 10% of that room. Act-3 values sit on the 0.55
already banked by reaching act 3, and 40.4% of act-3 decision rows are below it.

**Implication.** Do not spend a training run chasing a few thousandths of value loss. Fix the target's
representation (milestones, stake-pricing, banked progress) only if it changes decisions; the first two attempts did
not change wins with frozen networks.

## 14. Death value, stake-priced tables and milestone heads did not improve frozen-network play

**Evidence.** Death value F1, double-boss F2, both, and stake-priced tables: bosses won per run -0.042 to +0.004,
all CIs include 0; stake 18.3% vs 18.0% wins. Milestone heads: 16.2% vs 14.5% (-1.6 points, -4.2 to +0.9), with a
better replay loss and no act-3 values below banked.

**Implication.** A better-calibrated value does not automatically make a better player when the search's prices and
the policy's behaviour are unchanged. These ideas remain untested *through training* (two chains from one
checkpoint); that is the only way to know whether they help.

## 15. Argmax loops on repeatable event options

**Evidence.** 26% of evaluation runs of the first laptop agent died outside combat; the canonical case was holding on
at Slippery Bridge page after page from full HP. The new EVENT head's argmax repeats the same option, and some options
are repeatable.

**Implication.** Evaluation must sample the heads that have no human prior (MAP, EVENT) and take argmax only where the
S1 policy is calibrated. More generally, a deterministic policy on a repeatable option needs a loop guard.

## 16. Decision-surface bugs hide in prompts that are not at the root

**Evidence.** Knowledge Demon's Curse of Knowledge prompt opens during End Turn. `ActionSpace` never expanded it, so
the live agent answered at random three times per fight. The same applied to start-of-turn power prompts. The fix
(root probe of End Turn, one action per answer) passed a dedicated test.

**Implication.** Audit the decision surface against the real game, not only against the states the actor happens to
visit. A prompt inside an animation/end-turn phase is still a decision.

## 17. Throughput work should target our code first, then the engine

**Evidence.** S3 actor profile: inside simulations 91% (engine steps 55%, clone 11%, observable-state hash 11-12%,
leaf encode 9%, legal actions 8%, selection/backup 1%); outside the search only 4%. GC pauses 5.5% of wall time,
~300 MB/s allocated per actor. The cheap wins were a string-free hash (~10%), cached vocabulary ids (~5-7%), cheaper
legal-action keys, then the engine's hook-listener lists. The perf patch (hook caching, exact-size lists, ModelDb
cache) gave 1.31x with real networks and passed 4,662 upstream tests. MPS + CUDA graphs + overlap gave +18% runs/hour
and about 20 -> 45 combat-trainer steps/s.

**Implication.** Profile the actor before optimizing the search algorithm. Replay avoidance is worth at most ~16%
because paths are short; the encoder/hash/action-key path is where the time is.

## 18. Evaluation noise is large enough to reverse decisions at 150 seeds

**Evidence.** Early 150-seed evaluations read 0 / 0.7 / 1.3 / 2.0% for four consecutive segments; at 300 seeds the
curve became informative. Shadow comparisons of 300 paired seeds had McNemar p ~0.5 at a +1.3 point difference.

**Implication.** Use at least 300 seeds for a segment decision and 1,000 for a gate; report paired tests, not just
win rates. Do not switch a network on a single segment.

## 19. Warm starts help, but the KL schedule and the hand-off need to be explicit

**Evidence.** S3 starts from S1/S2 and uses a KL penalty to the S1 policy that decays to zero over 75k runs. The first
runs showed that a strong KL term plus a value gradient can stall PPO; after the fixes, PPO completed its steps and
the agent improved from 0% to 31.3% at A10. The human data's risk preferences (rest vs upgrade, elites) were tracked
but never fully removed.

**Implication.** Treat the warm start as a prior with an expiration date. Log the KL to the prior, the lag KL, and
the behavioural statistics (rest rate, elites per act) so "human habits" are visible after the KL weight reaches
zero.

## 20. A simulator review and a harness review are different things

**Evidence.** The boss fidelity review found no simulator difference for four bosses, but it did find a harness bug
(random curse choice) and two observation gaps. The simulator patch also fixed a decision-surface omission (End Turn
prompts). The 1,000-start import test had 2 failures from a modded relic, not from combat.

**Implication.** When a boss looks weak, check three layers separately: the simulator's numbers, the harness's
decisions, and the network's input. A clean boss implementation does not prove the agent is being asked the right
question.
