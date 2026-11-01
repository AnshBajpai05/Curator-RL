# CURATOR — Related Work, Competing Systems and Positioning

**Date:** 24 October 2026  ·  **Scope:** papers and systems that decide *what an LLM trains on* during RL with verifiable rewards (RLVR), and systems that check *whether a training reward can be trusted*.  ·  **Status of our numbers:** taken from `docs/DECISIONS.md` (D-87 … D-106); GPU results are from three seeds of Qwen2.5-0.5B on Kaggle T4s.

---

## 1. The problem, in one paragraph

An LLM trained with RLVR learns from several environments at once (math, code, puzzles). They differ in cost per prompt, their usefulness changes as the model improves, and some rewards are broken: random, hackable, or rewarding the wrong thing. The question is how to spend a fixed GPU budget across them. CURATOR answers it with a bandit scheduler around an unmodified GRPO trainer that divides learning signal by measured cost, checks each reward source before and during training, and accounts every GPU-second against the budget.

## 2. Research papers

### 2.1 Which environment or domain to train on next (closest competitors)

| Work | Venue / date | What it does | Reported result |
|---|---|---|---|
| **TAC**: Transferability for General Reasoning ([arXiv 2606.25178](https://arxiv.org/abs/2606.25178)) | ICML 2026 | Bandit curriculum over domains: learnability (per-domain advantage) + transferability (projected GRPO gradients), < 1% overhead | Best macro accuracy on a 6-domain suite, Qwen3-1.7B and Llama3.2-3B; beats random, a hand schedule and a learnability-only bandit |
| **HarnessBandit** ([arXiv 2609.13739](https://arxiv.org/html/2609.13739)) | Sept 2026 | One harness per optimiser step, scored by mean \|advantage\| and the cosine of a gradient sketch with other harnesses | Multi-harness agentic RL |
| **RISED** ([arXiv 2610.00979](https://arxiv.org/abs/2610.00979)) | Oct 2026 | Rubric tags on rollouts drive prompt-group selection across environments, plus self-distillation | Highest mean pass rate on ALFWorld, WebShop, DBBench; argues rubric similarity predicts transfer better than gradient similarity |
| **SEC**: Self-Evolving Curriculum ([arXiv 2505.14970](https://arxiv.org/abs/2505.14970)) | COLM 2026 | Non-stationary bandit over problem categories, reward = mean \|advantage\|, TD(0) | Gains over random curricula on reasoning tasks |
| **DUMP** ([arXiv 2504.09710](https://arxiv.org/pdf/2504.09710)) | 2025 | UCB over data distributions on mean \|advantage\| | Faster convergence on multi-distribution RL |
| **DataFlex-RL** ([arXiv 2609.06107](https://arxiv.org/abs/2609.06107)) | Sept 2026 | Evaluation platform for RLVR data policies; 12 matched seeds, Qwen2.5-7B, 12 benchmarks | **No** selection, reweighting or adaptive mixing policy beat uniform GRPO sampling with a paired 95% interval excluding zero; rankings flip with the benchmark summary (ρ = −0.33) |
| M2RL ([arXiv 2602.12566](https://arxiv.org/html/2602.12566)), RLVR fusion paradigms ([arXiv 2608.27409](https://arxiv.org/html/2608.27409v1)) | 2026 | Mixed multi-task RLVR vs separate training + merging / distillation | Comparisons of how to combine domains, not online allocation |

### 2.2 Which prompts inside a dataset (complementary, prompt level)

| Work | Venue / date | Idea |
|---|---|---|
| Online Difficulty Filtering ([arXiv 2504.03380](https://arxiv.org/abs/2504.03380)) | EACL 2026 | Keep prompts of intermediate accuracy; expected improvement is bounded by the variance of success probabilities; up to +12% with less than half the GRPO steps |
| Prompt Replay ([arXiv 2603.21177](https://arxiv.org/html/2603.21177v1)) | 2026 | Reuse high-signal prompts on-policy at no extra cost |
| DEPO ([arXiv 2602.06375](https://arxiv.org/html/2602.06375)), GPS ([arXiv 2602.01970](https://arxiv.org/html/2602.01970)), Train at Moving Edge ([2603.25184](https://www.alphaxiv.org/abs/2603.25184)) | 2026 | Predict prompt difficulty before rollout to skip uninformative prompts |

### 2.3 Data mixing for pretraining (the older relative)

Online Data Mixing ([arXiv 2312.02406](https://arxiv.org/abs/2312.02406), EXP3 bandit over domains), RegMix ([arXiv 2407.01492](https://arxiv.org/abs/2407.01492)) and its 2026 schedule extension, Olmix ([ICML 2026](https://proceedings.mlr.press/v306/chen26dd.html)), Data Mixing Agent ([arXiv 2507.15640](https://arxiv.org/html/2507.15640v2)), Bayesian domain reweighting ([arXiv 2607.27928](https://arxiv.org/html/2607.27928v1)). All optimise a loss on pretraining domains; none is cost-aware per domain or deals with broken rewards.

### 2.4 Can the reward be trusted? (reward reliability)

| Work | Date | Finding |
|---|---|---|
| Spurious Rewards ([arXiv 2506.10947](https://arxiv.org/abs/2506.10947)) | ICML 2026 | Random rewards lift Qwen2.5-Math-7B on MATH-500 by 21.4 points (29.1 with true rewards); the effect does not carry to Llama or OLMo |
| Fuzzing RLVR Verifiers ([arXiv 2606.01066](https://arxiv.org/abs/2606.01066)) | May 2026 | Adversarial completions expose verifiers that accept wrong answers; verifier reliability is measurable before training |
| Checking the Checkers ([arXiv 2610.17824](https://arxiv.org/html/2610.17824v1)) | Oct 2026 | Synthesis of verifier noise, spurious rewards and judge vulnerabilities |
| Verifier Errors in RLVR ([arXiv 2609.35677](https://arxiv.org/pdf/2609.35677)) | Sept 2026 | Theory: training observations alone give no advantage in detecting accepted verifier errors |
| Grift, gradient fingerprints ([arXiv 2604.16242](https://arxiv.org/html/2604.16242v2)) | 2026 | Gradient-based reward-hacking detector, about 80% F1 on code |
| Leaky reward suites ([arXiv 2607.11022](https://arxiv.org/pdf/2607.11022)) | July 2026 | Pre-registered causal study of natural verifier false positives |

## 3. Systems and products

No hardware devices address this problem; the relevant "systems" are training frameworks, environment hubs and managed RL services. None of them decides the environment mix for you: they leave the mix as a fixed user setting.

| System | Kind | What it gives | Mix / budget control |
|---|---|---|---|
| [Prime Intellect](https://www.primeintellect.ai/) Environments Hub, `verifiers`, `prime-rl`, Lab | Hub + framework + hosted RL | 2,500+ community environments, rubrics, trainer, hosted post-training | User-set |
| NVIDIA NeMo Gym + NeMo-RL | Environments + framework | Tool servers with `/verify` reward endpoints; post-training framework | User-set |
| [Nous Research Atropos](https://github.com/NousResearch/atropos) | Environment framework | Environments own inference and post scored batches; runs `verifiers` environments | User-set |
| [Hugging Face RL Environments](https://huggingface.co/blog/rl-environments) + TRL | Hub + trainer | Environment filter on the Hub; `GRPOTrainer` (CURATOR's trainer) | User-set |
| verl, OpenRLHF, AReaL, slime ([overview](https://anyscale.com/blog/open-source-rl-libraries-for-llms)) | Open-source RL frameworks | Scale (Megatron/FSDP, vLLM/SGLang), async training | User-set |
| [OpenAI Reinforcement Fine-Tuning](https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning) | Managed API | RL on OpenAI reasoning models with user graders (string checks, model graders) | One dataset per job |
| [Thinking Machines Tinker](https://en.wikipedia.org/wiki/Thinking_Machines_Lab) | Managed fine-tuning API | Training on open-weight models on their clusters | User code |

## 4. Where CURATOR stands

### 4.1 Feature comparison

| | TAC | SEC / DUMP | RISED / HarnessBandit | DataFlex-RL | Difficulty filtering | Verifier fuzzing | Platforms (Prime, NeMo, TRL…) | **CURATOR** |
|---|---|---|---|---|---|---|---|---|
| Chooses the environment mix online | ✓ | ✓ | ✓ | (evaluates others) | prompt level | — | — | ✓ |
| Uses measured cost per environment | — | — | — | — | — | — | — | **✓** |
| Matched GPU-budget accounting | — | — | — | partial (matched recipe) | — | — | — | **✓** (ledger, ≤ 0.1% unattributed) |
| Checks rewards before training | — | — | — | — | — | ✓ | — | **✓** (pre-flight junk-answer check) |
| Checks rewards during training | — | — | — | — | — | — | — | **✓** (label-free dispersion audit) |
| Models cross-domain transfer | **✓** | — | **✓** | — | — | — | — | partial (simulator transfer, ROI only) |
| Agentic / multi-turn environments | — | — | **✓** | — | — | — | ✓ | — |
| Model scale in experiments | 1.7–3B | 1.5–7B | agentic LLMs | **7–8B** | 7B | — | any | 0.5B (1B OLMo pilot) |
| Seeds per comparison | few | few | few | **12** | few | — | — | 3 |
| Pre-registered tests, negative results published | — | — | — | partly | — | — | — | **✓** |

### 4.2 Where we are ahead

1. **Robust to broken reward sources.** On the real portfolio, the learnability bandits sent about a third of their questions to the random-reward arm (SEC 34%, DUMP 31%, Uniform 20%), CURATOR 6%; SEC and DUMP lost accuracy against Uniform (−2.3 and −1.5 points, seed 0; confirmation seeds pending). Any selector that steers on |advantage| is attracted by a reward that is pure noise, because noise always produces mixed groups.
2. **Cost is a first-class signal.** CURATOR divides learning signal by measured GPU cost per environment (measured spread 2.67× on the Level-1 portfolio) and charges every second of training, scoring, calibration and overhead to one budget. None of the selectors above does either.
3. **Two layers of reward checking inside the scheduler.** A pre-flight check (junk answers per arm before training; flags the random and hackable arms, real verifiers 0%) and an in-run, label-free audit (prompt dispersion of pass rates: real verifiers 3.0–4.4, random reward 1.0–1.09 on the Kaggle pilot). Verifier fuzzing papers check before training only; detection papers (Grift) need gradients and labels.
4. **Research discipline.** Every adoption decision has a written rule fixed before the run, fresh seeds, and published failures (CRE, H-γ1, the label-based audit of D-105). This matches the standard DataFlex-RL sets for the field.
5. **Runs on free hardware.** The whole pipeline (vLLM colocate, LoRA, checkpoints that survive lost sessions) runs on two Kaggle T4s.

### 4.3 Where we are behind

1. **No accuracy gain over Uniform.** CURATOR vs Uniform on sampled macro accuracy, three seeds: −0.1 points [−1.2, +1.0]. That is the DataFlex-RL result again. TAC and online difficulty filtering report real accuracy gains; we do not.
2. **No transfer term.** TAC and HarnessBandit model how training one domain helps another; CURATOR treats environments as independent arms (transfer only enters the simulator and the ROI table).
3. **Small scale and few seeds.** 0.5B models and three seeds, against 1.7–8B models and up to 12 seeds in the literature. Our intervals are wide.
4. **Not agentic.** All environments are single-turn; the agentic selection work (RISED, HarnessBandit) is where multi-environment training is moving.
5. **Calibration and ROI do not pay off at our budget.** Held-out calibration measures but does not raise the score; Curator without it scores slightly higher, and a one-seed ROI table is descriptive only.
6. **Blind spot.** A reward that depends on the prompt but rewards the wrong thing passes the in-run audit; only the pre-flight check can catch it, and only for the junk patterns it tries.
7. **Not a product.** The platforms above ship hubs, hosted training and hundreds of environments; CURATOR is a research scheduler with five environments.

### 4.4 Positioning in one sentence

At small scale the data mix does not move accuracy; CURATOR's contribution is that its mix is **not pulled into broken reward sources and is accounted in GPU-dollars**, while learnability-only selectors are. Accuracy gains, transfer and agentic environments are where the field is ahead.

## 5. Caveats

- Literature numbers are as reported by the authors; none of the papers was re-run here, and our setup (0.5B, five single-turn arms, 1.5 GPU-hour runs) differs from theirs, so comparisons are qualitative.
- Several 2026 entries were read from abstracts and search summaries, not full text.
- CURATOR's GPU results come from the integrated branch's decision log (D-101 … D-103) and were not re-run for this document.
