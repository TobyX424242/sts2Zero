# History

A short account of how the project got here. Details are in [experiments.md](experiments.md) (this phase) and on
branch `v0.111-defect` (the previous phase). Dates are UTC.

## The design in one paragraph

`sts2Zero` plays full runs of Slay the Spire 2 in the MIT-licensed simulator iRyougi/sts2-sim. A **macro network**
makes every decision outside combat and predicts the run value V_run. A **combat network** guides an AlphaZero-style
search in every fight. The two are linked by one price: a fight is scored by `T_f(h, P)`, the run value after the
fight with HP `h` and potion belt `P`. Training has three stages: **S1** warms the macro network up on human runs,
**S2** warms the combat network up on human battle starts, and **S3** trains both on the agent's own full runs. See
[architecture.md](architecture.md) and [training.md](training.md).

## Phase 1: Defect on game v0.111.0 (2026-11-10 to 2026-11-18)

- **Proof of concept.** A set-transformer value model on a 100k-run sample of human data. It ranked states well but
  priced them wrongly for a strong agent. Its review shaped the S1 design.
- **S0, S1, S2 (2026-11-10 and 11).** Simulator extensions (exact and resampled clones, battle-start import, the
  legal-action space, the bridge to the networks). S1 trained the macro network on 1.2M clean human runs. S2 trained the
  combat network on 1M self-play fights from human battle starts; with search 64 it won 91.8% of held-out starts.
- **S3 (from 2026-11-11).** Full-run self-play at A10. Four design problems surfaced in the first day: a pure win
  target gave V_run no signal, the policy could not see HP, argmax on the event head looped, and the value loss moved
  the policy encoder. Later the run value was found to memorise runs. After the fixes the A10 win rate rose from 0% (53k runs) to **31.3%** at segment 45 after
  1.28M runs (2026-11-15). Human Defect A10 players win 15.5%.
- **Release v0.1.0-s045** (2026-11-15): the segment-45 weights and the code, now on branch `v0.111-defect`.
- **After the release (2026-11-15 to 2026-11-18).** The Defect work continued on a new observation layout with
  attention (layout v3). Its networks measured **44.8%** on a fresh 1,020-seed panel (2026-11-17). A further layout
  with public forecasts (v4) fell behind and was dropped. All Defect training stopped on 2026-11-18. The layout-v3
  networks of segment 5 became the starting point of the next phase.

## Phase 2: Ironclad on game v0.113.1 (from 2026-11-18)

On 2026-11-18 the project moved to the new game version v0.113.1 (standard mode only) and to a new character,
Ironclad, still at A10. It kept the design and the Defect networks, converted to the new version.

- **Port (2026-11-18).** The simulator was moved to iRyougi's v0.113.1.1 tag. Character and game version became
  options, vocabularies were remapped by id, and T_f learned post-combat heals. Every check passed; the converted
  networks played Defect on v0.113 at their old level (46.6%).
- **Untrained Ironclad: 0%.** The same networks won no Ironclad A10 run.
- **Combat warm-up (2026-11-18).** Four GPU hours of fight self-play on human Ironclad battle starts. Full runs went
  from 0% to 3.1% at A10 and from 5.1% to 52.9% at A0. The macro network got no human refresh.
- **Self-play chain `s3ic` (from 2026-11-18 12:19).** A10 evaluations went from 23.3% (segment 0) to 43.5%
  (segment 5) in about 13 hours, then moved between 37% and 47% while self-play kept improving. Along the way the
  chain fixed combat stalemates (`win_bound`), routed random card prompts to trained heads, switched on milestone
  value heads, and climbed an ascension curriculum. It survived an interruptible server, a lost host and several
  machine moves; segment numbers 9-10 and 13-17 produced nothing.
- **Offline tests (2026-11-19 and 20).** Four controlled tests on a separate GPU: V_run is too flat after the first
  act-3 boss (T1); milestone heads fit better offline (T2); more search does not win act-3 boss fights (T3); repricing
  fights with the same combat network does not win more (T4). These pointed the work at the combat network.
- **Segment 29 (2026-11-20).** With curriculum stage 4, a higher win weight in the value target, act-3 boss practice
  and a larger shadow combat network training beside the playing one, segment 29 evaluated at **50.7%** (152 / 300,
  CI 45.0-56.3) after 528,810 self-play runs. Its weights are the current release ([checkpoints.md](checkpoints.md)).
  Human Ironclad A10 players win 21.2% (v0.111.0).

## Live play in the real game (from 2026-11-19)

A game mod (AgentBridge) and a controller (LiveAgent) let the agent play the real game: the controller imports the run
from the game's save into the simulator and sends the agent's choices back. The first finished runs were wins: A0 and
A6 with segment-21 weights on 2026-11-19, and A10 with segment-29 weights on 2026-11-20. See
[live-play.md](live-play.md).

## Where things stand

- The release reading (50.7%) comes from one 300-seed panel and needs a fresh panel to confirm it.
- The act-3 bosses cause about 60% of A10 losses, and more search does not help there. The open bet is the larger
  combat network that trains in the shadow since segment 29.
- Results: [results.md](results.md). Lessons: [findings.md](findings.md).
