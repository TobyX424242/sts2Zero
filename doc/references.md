# References

Works the project used or compared against. The "Used for" column says what was taken from each work, not that the
whole method was implemented. Papers are cited in full so they can be found without a link; links are given for
software and data the project relied on.

## The core ideas, in brief

| Idea in this agent | Source |
|---|---|
| Gumbel root selection and sequential halving; the improved-policy training target | Danihelka et al. 2022 |
| Combat as a stochastic single-agent problem with chance nodes | Antonoglou et al. 2022 (Stochastic MuZero) |
| One sampled world per simulation, never the true random state | Silver & Veness 2010 (POMCP); Cowling et al. 2012 (ISMCTS) |
| An outcome distribution head whose utility is applied outside the network (here: end HP priced by T_f) | Wu 2019 (KataGo) |
| Search output as the policy target for the next network | Anthony et al. 2017 (Expert Iteration); Silver et al. 2018 |
| Macro policy optimisation | Schulman et al. 2017 (PPO); Schulman et al. 2015 (GAE) |
| Classification target for the end-HP bins | Farebrother et al. 2024 (HL-Gauss) |
| Value monotone in HP by construction | Sill 1998; Daniels & Velikova 2010 |
| Checking calibration before using a value as a price | Guo et al. 2017 |

## Search and reinforcement learning

| Work | Used for |
|---|---|
| Silver et al. 2016. Mastering the game of Go with deep neural networks and tree search. *Nature* 529, 484–489. doi:10.1038/nature16961 | Policy/value-guided search; the general architecture |
| Silver et al. 2018. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science* 362, 1140–1144. doi:10.1126/science.aar6404 | Self-play with search targets |
| Schrittwieser et al. 2020. Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature* 588, 604–609. doi:10.1038/s41586-020-03051-4 | Value targets with search |
| Schrittwieser et al. 2021. Online and offline reinforcement learning by planning with a learned model. NeurIPS | Re-searching old data with the current network |
| Anthony, Tian & Barber 2017. Thinking fast and slow with deep learning and tree search. NeurIPS | Expert iteration: the combat training loop |
| Danihelka, Guez, Schrittwieser & Silver 2022. Policy improvement by planning with Gumbel. ICLR. arXiv:2107.00504 | Gumbel root, sequential halving, completed-Q policy target |
| Antonoglou et al. 2022. Planning in stochastic environments with a learned model. ICLR. arXiv:2112.03113 | Chance nodes |
| Silver & Veness 2010. Monte-Carlo planning in large POMDPs. NeurIPS | Information-set search with sampled worlds |
| Cowling, Powley & Whitehouse 2012. Information Set Monte Carlo Tree Search. *IEEE TCIAIG* 4(2) | Single-observer information sets |
| Couëtoux, Hoock, Sokolovska, Teytaud & Bonnard 2011. Continuous upper confidence trees. LION 5 | UCT in continuous action spaces (background) |
| Perez Liebana et al. 2015. Open loop search for general video game playing. GECCO | Open-loop trees beyond chance nodes |
| Czech, Korus & Kersting 2021. Improving AlphaZero using Monte-Carlo graph search. ICAPS | Transpositions: nodes keyed by the observable state |
| Schulman et al. 2017. Proximal policy optimization algorithms. arXiv:1707.06347 | Macro policy: clipped ratio, trust region |
| Schulman et al. 2015. High-dimensional continuous control using generalized advantage estimation. arXiv:1506.02438 | GAE (γ = 1, λ = 0.95) |
| Espeholt et al. 2018. IMPALA. ICML | Actor/learner split and lag tolerance |
| Espeholt et al. 2020. SEED RL: scalable and efficient deep-RL with accelerated central inference. ICLR | Central batched inference; the shared-memory bridge design |
| Huang et al. 2022. CleanRL: high-quality single-file implementations of deep reinforcement learning algorithms. *JMLR* 23 | Reference PPO implementation |
| Wu 2019. Accelerating self-play learning in Go. arXiv:1902.10565 | Playout-cap randomisation (64 / 16 simulations), score head priced outside the network |
| Sutton 1988. Learning to predict by the methods of temporal differences. *Machine Learning* 3 | TD(λ); considered for V_run, Monte Carlo targets kept |
| Henderson et al. 2018. Deep reinforcement learning that matters. AAAI | Seeds, variance and reporting |
| Agarwal et al. 2021. Deep reinforcement learning at the edge of the statistical precipice. NeurIPS | Caution about point estimates; interval reporting |

## Networks and value learning

| Work | Used for |
|---|---|
| Vaswani et al. 2017. Attention is all you need. NeurIPS. arXiv:1706.03762 | Combat transformer; macro self-attention (layout v3) |
| Zaheer et al. 2017. Deep Sets. NeurIPS. arXiv:1703.06114 | Per-token MLP + mean pooling in the macro encoder |
| Lee et al. 2019. Set Transformer. ICML. arXiv:1810.00825 | The attention alternative for sets |
| Xiong et al. 2020. On layer normalization in the Transformer architecture. ICML | Pre-norm layers |
| Farebrother et al. 2024. Stop regressing: training value functions via classification for scalable deep RL. ICML. arXiv:2403.03950 | HL-Gauss targets for end-HP bins |
| Sill 1998. Monotonic networks. NeurIPS | HP-monotone value head |
| Daniels & Velikova 2010. Monotone and partially monotone neural networks. *IEEE TNN* 21(6) | Monotone in HP only |
| Guo, Pleiss, Sun & Weinberger 2017. On calibration of modern neural networks. ICML. arXiv:1706.04599 | Calibration checks (ECE); temperature scaling tested and dropped |
| Lakshminarayanan, Pritzel & Blundell 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS. arXiv:1612.01474 | Five-member macro ensemble |
| Smith & Winkler 2006. The optimizer's curse. *Management Science* 52(3) | Why a search that maximises a noisy value picks its errors |
| Levine, Kumar, Tucker & Fu 2020. Offline reinforcement learning: tutorial, review, and perspectives. arXiv:2005.01643 | A value fitted to logs is a prediction, not a causal price; perturbed fight starts |
| Barreto et al. 2017. Successor features for transfer in reinforcement learning. NeurIPS | Precedent: learned outcome features priced by separate weights |
| Loshchilov & Hutter 2019. Decoupled weight decay regularization. ICLR. arXiv:1711.05101 | AdamW; decay on weight matrices only |
| Srivastava et al. 2014. Dropout. *JMLR* 15 | Value regularisation |
| Polyak & Juditsky 1992. Acceleration of stochastic approximation by averaging. *SIAM J. Control Optim.* 30(4) | Weight averaging (EMA) |
| Peng, Kumar, Zhang & Levine 2019. Advantage-weighted regression. arXiv:1910.00177 | Considered, not run |
| Nair, Gupta, Dalal & Levine 2020. AWAC. arXiv:2006.09359 | Considered, not run |
| King & Zeng 2001. Logistic regression in rare events data. *Political Analysis* 9(2) | Rare-win calibration; background |

## Slay the Spire agents, simulators and data

| Work | Used for |
|---|---|
| iRyougi/sts2-sim: https://github.com/iRyougi/sts2-sim | The simulator (MIT); tag v0.113.1.1 in this phase, commit `db844ad` (v0.111.0) before |
| Spire Codex: https://spire-codex.com/ (developer docs: https://spire-codex.com/developers) | Human runs and entity exports; use confirmed with the maintainer |
| AlphaDefect evaluation runs: https://github.com/hmty-CHEN/alphadefect-evaluation-runs | 2,000 published games of a Defect A10 agent (70.9% in its own engine); early feasibility analysis only |
| HalfVulpes/sts2ai. arXiv:2611.12088 | A search-free PPO agent; its reported win rates (Ironclad 15.6%, Defect 8.1%) were a planning reference; ideas on run logs and restarts |
| evangambit/sts2: https://github.com/evangambit/sts2 | Determinised MCTS with a transformer prior and value for combat; the design this project builds on |
| sts-rl-agent (Slay the Spire 1): https://huggingface.co/Jialeiv/sts-rl-agent | MCTS combat + REINFORCE macro; lessons on rollout cost and leaking the draw order |
| Epoch SpireClimb: https://epoch.ai/benchmarks/spireclimb | LLM agent reference (35% at A10, all characters, 75 runs) |
| SpireBench. arXiv:2610.12596 | LLM + strategy guides on Slay the Spire 1; risk valuation as the failure mode |
| AgenticSTS. arXiv:2607.02255 | LLM + situation-triggered skills |
| wuhao21/sts2-cli: https://github.com/wuhao21/sts2-cli | Candidate parity oracle against the real game DLL (not used) |
| STS2MCP: https://github.com/Gennadiyev/STS2MCP | Live-game bridge candidate (the project wrote its own mod) |
| CharTyr/STS2-Agent: https://github.com/CharTyr/STS2-Agent | Live-game bridge candidate |
| xuxy-tech/An-agent-for-slay-the-spire-2: https://github.com/xuxy-tech/An-agent-for-slay-the-spire-2 | Related agent |
| STS2 v0.111.0 beta patch notes: https://patchbot.io/games/slay-the-spire-2/articles/1131-beta-patch-notes-v01110 | Version pin of the previous phase |

## Tools

| Tool | Used for |
|---|---|
| PyTorch | All networks; CUDA graphs for inference |
| .NET 9 | Simulator, actors, search, encoders |
| CUDA MPS | Trainers and servers sharing one GPU |
| DeepMind mctx: https://github.com/google-deepmind/mctx | Reference for the Gumbel root (not a dependency) |
| CleanRL: https://github.com/vwxyzjn/cleanrl | Reference PPO (not a dependency) |

## Statistics

- Wilson, E. B. 1927. Probable inference, the law of succession, and statistical inference. *JASA* 22.
- McNemar, Q. 1947. Note on the sampling error of the difference between correlated proportions or percentages.
  *Psychometrika* 12.
- Efron, B. 1979. Bootstrap methods: another look at the jackknife. *Annals of Statistics* 7.
