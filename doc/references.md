# References

Annotated bibliography. The "used for" column says what the project took from the work, not that the whole paper was
implemented. Links are given where the project itself relied on an online resource; paper citations are given in
full so they can be found without a fragile URL.

## Read these first

1. **Danihelka et al. 2022, Gumbel AlphaZero** — the root policy-improvement rule (Gumbel-top-k + sequential
   halving) used in every search.
2. **Antonoglou et al. 2022, Stochastic MuZero** — the framing of STS combat as a stochastic single-agent problem with
   chance nodes.
3. **Silver & Veness 2010, POMCP / Cowling et al. 2012, ISMCTS** — the information-set view: one sampled world per
   simulation, never the true RNG.
4. **Wu 2019, KataGo** — the score/outcome distribution head whose utility is applied outside the network; the direct
   precedent for pricing the combat network's outcome with T_f.
5. **Schulman et al. 2017, PPO** — the macro policy optimiser.
6. **Farebrother et al. 2024, HL-Gauss** — the classification target for the end-HP distribution.
7. **Sill 1998 / Daniels & Velikova 2010, monotone networks** — the HP-monotone value head.
8. **Guo et al. 2017, calibration** — why the value is checked with ECE/floor bands before it is used as a price.

## Core search and RL

| Work | Used for |
|---|---|
| Silver et al. 2016. Mastering the game of Go with deep neural networks and tree search. *Nature* 529, 484-489. doi:10.1038/nature16961 | AlphaZero-style policy/value search; the general architecture |
| Silver et al. 2018. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science* 362, 1140-1144. doi:10.1126/science.aar6404 | Self-play, expert iteration, symmetry of policy and value |
| Schrittwieser et al. 2020. Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature* 588, 604-609. doi:10.1038/s41586-020-03051-4 | Learned model + search; value targets |
| Schrittwieser et al. 2021. Online and offline reinforcement learning by planning with a learned model. NeurIPS | Re-searching old data with the current network |
| Anthony, Tian & Barber 2017. Thinking fast and slow with deep learning and tree search (Expert Iteration). NeurIPS | The S2/S3 combat training loop: search produces targets for the next network |
| Danihelka, Guez, Schrittwieser & Silver 2022. Policy improvement by planning with Gumbel. ICLR. arXiv:2107.00504 | Gumbel root, sequential halving, improved policy target |
| Antonoglou et al. 2022. Planning in stochastic environments with a learned model (Stochastic MuZero). ICLR. arXiv:2112.03113 | Chance nodes in a stochastic single-agent game |
| Couëtoux, Hoock, Sokolovska, Teytaud & Bonnard 2011. Continuous upper confidence trees. LION 5 | UCT/UCB in continuous action spaces |
| Silver & Veness 2010. Monte-Carlo planning in large POMDPs (POMCP). NeurIPS | Information-set MCTS with sampled worlds |
| Cowling, Powley & Whitehouse 2012. Information Set Monte Carlo Tree Search. *IEEE TCIAIG* 4(2) | Single-observer ISMCTS; the search's information-set convention |
| Perez Liebana et al. 2015. Open loop search for general video game playing. GECCO | Open-loop tree beyond chance nodes |
| Czech, Korus & Kersting 2021. Improving AlphaZero using Monte-Carlo graph search. ICAPS | Transpositions; nodes keyed by observable state |
| Schulman et al. 2017. Proximal Policy Optimization algorithms. arXiv:1707.06347 | Macro PPO: clipped ratio, trust region |
| Schulman et al. 2015. High-dimensional continuous control using generalized advantage estimation. arXiv:1506.02438 | GAE (gamma = 1, lambda = 0.95) |
| Espeholt et al. 2018. IMPALA: scalable distributed deep-RL with importance weighted actor-learner architectures. ICML | Actor/learner decoupling, lag tolerance, V-trace as a fallback |
| Espeholt et al. 2020. SEED RL: scalable and efficient deep-RL with accelerated central inference. ICLR | Central batched inference server; the shared-memory bridge design |
| Huang et al. 2022. CleanRL: high-quality single-file implementations of deep reinforcement learning algorithms. *JMLR* 23 | Reference implementation style for PPO |
| Wu 2019. Accelerating self-play learning in Go (KataGo). arXiv:1902.10565 | Playout cap randomization (64/16), score-distribution head + utility outside the net |
| Henderson et al. 2018. Deep reinforcement learning that matters. AAAI | Seeds, variance, reporting |
| Agarwal et al. 2021. Deep reinforcement learning at the edge of the statistical precipice. NeurIPS | Stratified bootstrap / interval reporting; we use paired tests and Wilson intervals |
| Sutton 1988. Learning to predict by the methods of temporal differences. *Machine Learning* 3 | TD(lambda); considered for V_run and rejected in favour of Monte Carlo |

## Architectures and value learning

| Work | Used for |
|---|---|
| Vaswani et al. 2017. Attention is all you need. NeurIPS. arXiv:1706.03762 | Combat token transformer |
| Zaheer et al. 2017. Deep Sets. NeurIPS. arXiv:1703.06114 | The no-attention macro encoder (per-token MLP + mean pooling) |
| Lee et al. 2019. Set Transformer. ICML. arXiv:1810.00825 | The macro transformer alternative |
| Xiong et al. 2020. On layer normalization in the Transformer architecture. ICML | Pre-norm transformer layers |
| Farebrother et al. 2024. Stop regressing: training value functions via classification for scalable deep RL. ICML. arXiv:2403.03950 | HL-Gauss target for the end-HP bins |
| Sill 1998. Monotonic networks. NeurIPS | Monotone value head by construction |
| Daniels & Velikova 2010. Monotone and partially monotone neural networks. *IEEE TNN* 21(6) | Partial monotonicity (HP only) |
| Guo, Pleiss, Sun & Weinberger 2017. On calibration of modern neural networks. ICML. arXiv:1706.04599 | ECE, temperature scaling (tested and dropped) |
| Lakshminarayanan, Pritzel & Blundell 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS. arXiv:1612.01474 | Five-member ensemble; average probabilities/logits |
| Smith & Winkler 2006. The optimizer's curse. *Management Science* 52(3) | Why a search maximising a noisy value picks its errors; kappa/pessimism |
| Levine, Kumar, Tucker & Fu 2020. Offline reinforcement learning: tutorial, review, and perspectives. arXiv:2005.01643 | Why a value fitted to logs is a prediction, not a causal price |
| Peng, Kumar, Zhang & Levine 2019. Advantage-weighted regression. arXiv:1910.00177 | Considered for behaviour-cloning weighting (not run) |
| Nair, Gupta, Dalal & Levine 2020. AWAC. arXiv:2006.09359 | Considered for offline-to-online fine-tuning (not run) |
| Barreto et al. 2017. Successor features for transfer in reinforcement learning. NeurIPS | Conceptual precedent: learned outcome features priced by separate weights |
| Loshchilov & Hutter 2019. Decoupled weight decay regularization (AdamW). ICLR. arXiv:1711.05101 | S1/S3 value regularisation |
| Srivastava et al. 2014. Dropout. *JMLR* 15 | Value-head regularisation in S3 |
| Polyak & Juditsky 1992. Acceleration of stochastic approximation by averaging. *SIAM J. Control Optim.* 30(4) | Weight averaging / EMA |
| King & Zeng 2001. Logistic regression in rare events data. *Political Analysis* 9(2) | Rare-win calibration; considered, not central |

## Slay the Spire agents, benchmarks and data

| Work | Used for |
|---|---|
| AlphaDefect evaluation runs: https://github.com/hmty-CHEN/alphadefect-evaluation-runs | External reference: 2,000 published games of a 70.9% A10 Defect agent in its own engine; audited for consistency; used for POC transfer analysis, not in the final training |
| Epoch SpireClimb: https://epoch.ai/benchmarks/spireclimb | Best LLM agent reference (35% at A10, all characters, N=75) |
| SpireBench. arXiv:2610.12596 | LLM + strategy guides on STS1; guides slightly hurt; risk valuation is the failure mode |
| AgenticSTS. arXiv:2607.02255 | LLM + situation-triggered skills; small, not significant |
| sts-rl-agent (STS1 hybrid): https://huggingface.co/Jialeiv/sts-rl-agent | MCTS combat + REINFORCE macro; rollout MCTS needs tens of thousands of simulations without a learned prior; distillation failed because search saw the future draw order |
| evangambit/sts2: https://github.com/evangambit/sts2 | Determinised MCTS + transformer prior/value for combat; +2.0 HP per fight vs rollout; the design we build on |
| iRyougi/sts2-sim: https://github.com/iRyougi/sts2-sim | The chosen simulator (MIT); source snapshot and patches in `src/sims` |
| wuhao21/sts2-cli: https://github.com/wuhao21/sts2-cli | Real-DLL parity oracle; designed for, not included |
| STS2MCP: https://github.com/Gennadiyev/STS2MCP | Live-client bridge candidate for a real-game check |
| CharTyr/STS2-Agent: https://github.com/CharTyr/STS2-Agent | Live-client bridge candidate |
| xuxy-tech/An-agent-for-slay-the-spire-2: https://github.com/xuxy-tech/An-agent-for-slay-the-spire-2 | Related STS2 agent |
| HalfVulpes/sts2ai. arXiv:2611.12088 | Reviewed after the snapshot for ideas (replayable run logs, observation hash, failure-driven restarts, common random numbers); not part of the architecture |
| Spire Codex developer docs and exports: https://spire-codex.com/developers | Human-run data and entity exports; use confirmed with the maintainer, data not redistributed here |
| STS2 v0.111.0 beta patch notes: https://patchbot.io/games/slay-the-spire-2/articles/1131-beta-patch-notes-v01110 | Version pin |

## Tools

| Tool | Used for |
|---|---|
| DeepMind mctx: https://github.com/google-deepmind/mctx | Reference for the Gumbel root; not a dependency |
| CleanRL: https://github.com/vwxyzjn/cleanrl | Reference for a single-file PPO |
| PyTorch 2.6 | All networks; CUDA graphs for inference |
| .NET 9 | Simulator, actors, search, encoders |

## Statistics and evaluation

- Wilson, E. B. 1927. Probable inference, the law of succession, and statistical inference. *JASA* 22.
- McNemar, Q. 1947. Note on the sampling error of the difference between correlated proportions or percentages.
  *Psychometrika* 12.
- Efron, B. 1979. Bootstrap methods: another look at the jackknife. *Annals of Statistics* 7.
- Agarwal et al. 2021 (above) for the caution about point estimates in deep RL.
