# Literature Review and Comparison

**Date:** 24 October 2026
**Purpose:** explain what CURATOR is and does, whether it solves a real problem, and — for each competing paper or system — what it does, what it lacks, and whether CURATOR or the competitor leads on that point.
**Sources:** `RELATED_WORK.md`, `docs/DECISIONS.md` (D-87 … D-106), `docs/ROADMAP_v4.md`. Literature numbers are as reported by their own authors and were not re-run here.

---

## 1. What this project is, in plain terms

Training a language model with **reinforcement learning from verifiable rewards (RLVR)** means practicing on several kinds of problems at once — math, code, logic puzzles — each checked by an automatic grader instead of a human. Every round of training, you must decide: *what fraction of the next batch of practice problems should come from each subject?*

That choice matters for two reasons:
1. **The subjects cost different amounts.** Some problems need longer answers or slower checking. A fixed GPU budget buys more practice in a cheap subject than an expensive one.
2. **Some graders are broken.** A grader might reward a lucky guess, accept a wrong answer, or never reward a genuinely hard problem. A method that chases "the subject where the model seems to be improving" can be fooled into spending its whole budget on a subject whose grader is simply noisy, because noise looks like learning.

**CURATOR is a scheduler that sits outside the training loop and makes that choice every round.** It does not change the model, the training algorithm, or the graders. It only decides the mix. It does three things no single competing method combines:
- It divides each subject's apparent learning signal by its **measured** GPU cost, not a guess.
- It checks, before and during training, whether each grader's reward actually depends on the answer at all — catching graders that are effectively random.
- It tracks every GPU-second spent (generating, checking, training, measuring) in one ledger, so every comparison is at matched cost.

It is tested first in a fast simulator with known right answers, then on real GPU training of a small open model (Qwen2.5-0.5B) on free Kaggle hardware, using five subjects: grade-school math, harder math, code, a number puzzle, and one subject with a deliberately broken (random) grader.

## 2. Is it solving a real problem?

**Yes, but a narrower one than first planned.**

- **The problem is real.** Multiple 2026 papers confirm it: a grader that looks fine can reward the wrong thing, and a selector that steers on "where the model is improving" can be fooled by a grader that rewards randomly — it always looks like something is happening. CURATOR measures this directly: on the real GPU run, its two closest learnability-based competitors spent 31–34% of their training budget on the broken grader and lost 1.5–2.3 points of accuracy to a method that simply trained on everything equally. CURATOR spent 6% of its budget there.
- **The problem CURATOR does *not* solve (yet) is raising accuracy.** On the actual GPU run, CURATOR does not beat training on everything equally: −0.1 points, with an interval wide enough to include both a real gain and a real loss. A large 2026 benchmark study (DataFlex-RL) found the same thing for every published method it tested, even at much larger scale.
- **So the honest claim is:** CURATOR reliably avoids wasting budget on a broken grader, and accounts for cost, where its direct competitors do not — but it has not yet been shown to make the model more accurate than simply training on everything equally.

## 3. Literature review: what each competitor does, what it lacks, who leads

### 3.1 Choosing which subject to train on next (closest competitors)

- **TAC — Transferability-Aware Curriculum** (arXiv 2606.25178, ICML 2026)
  - *Does:* a bandit that combines "is this subject improving" with "does training on it help other subjects," measured from the training gradients.
  - *Lacks:* no cost awareness, no check for a broken grader, no cost ledger.
  - *Verdict:* **TAC leads on accuracy and transfer.** Reports real gains across six subjects on larger models (1.7–3B). CURATOR has not matched this.

- **HarnessBandit** (arXiv 2609.13739)
  - *Does:* picks which multi-step task to practice next using advantage plus a gradient-similarity score to other tasks.
  - *Lacks:* no cost signal, no broken-grader check.
  - *Verdict:* **HarnessBandit leads on multi-step (agentic) task selection.** CURATOR only handles single-turn subjects.

- **RISED** (arXiv 2610.00979)
  - *Does:* tags each attempt with a rubric, then uses rubric similarity (instead of gradients) to decide what to train on across three multi-step environments.
  - *Lacks:* no cost signal, no broken-grader check.
  - *Verdict:* **RISED leads on agentic environment selection.** CURATOR has no multi-step environments yet (planned, Roadmap v4 Phase 5).

- **SEC (Self-Evolving Curriculum)** (arXiv 2505.14970) and **DUMP** (arXiv 2504.09710)
  - *Do:* a bandit that favors problem categories where the model's answers disagree most (highest "advantage").
  - *Lack:* exactly the property that makes them fail here — a random grader always produces disagreeing answers, so they are drawn to it. No cost signal.
  - *Verdict:* **CURATOR leads.** Measured directly: SEC and DUMP lost 1.5–2.3 points to simple equal training, by over-spending on the broken grader. This is the project's central, measured result.

- **DataFlex-RL** (arXiv 2609.06107, Sept 2026)
  - *Does:* a large, careful benchmark of RLVR data-selection methods (12 seeds, 7–8B models). Not a method itself — a report card on the whole field.
  - *Finding:* **no published selection or mixing method beat simply training on everything equally**, with a statistically solid result.
  - *Verdict:* **Neither leads — this sets the bar CURATOR must still clear.** It also means CURATOR's "no accuracy gain" result is not an isolated failure; it matches the field's best evidence so far.

### 3.2 Choosing which individual problems within a subject (a level below CURATOR)

- **Online Difficulty Filtering** (arXiv 2504.03380, EACL 2026): drop problems that are too easy or too hard; claims up to +12% accuracy using under half the training steps.
  - *Verdict:* **Leads on accuracy**, at the within-subject level CURATOR does not directly touch. CURATOR's own first attempt at a related idea (question-level filtering) was found in this project's own tests to fall into the same trap as SEC/DUMP — it over-selects a broken grader's "disagreeing" answers. This is a known, unresolved gap (Roadmap v4, Phase 2).
- **Prompt Replay, DEPO, GPS, Train at Moving Edge** (various, 2026): reuse or predict which individual problems are worth practicing, to cut wasted computation.
  - *Verdict:* **These lead on compute efficiency** within a subject. CURATOR does not yet have an equivalent mechanism that survives a broken grader.

### 3.3 Is the grader (reward) trustworthy?

- **Spurious Rewards** (arXiv 2506.10947, ICML 2026): shows a *completely random* reward can still raise accuracy on one specific model family, by 21 points — a warning that "broken reward = useless" is not always true, and the effect is model-specific.
  - *Verdict:* **Raises a real caveat CURATOR has not fully answered.** CURATOR assumes a broken grader should get no budget; this paper shows that assumption can occasionally be wrong. CURATOR's roadmap already plans a second model family to test this.
- **Fuzzing RLVR Verifiers** (arXiv 2606.01066): tests graders with adversarial fake answers before training, to find ones that accept wrong answers.
  - *Verdict:* **CURATOR already does this** (its pre-flight check is built on the same idea) but with a shorter, fixed list of fake-answer types. The published fuzzing work tests more kinds.
- **Checking the Checkers** (arXiv 2610.17824) and **Verifier Errors in RLVR** (arXiv 2609.35677): survey and prove, respectively, that a grader's mistakes are often invisible from the training data alone.
  - *Verdict:* **Neither side has solved this.** It directly motivates CURATOR's planned "wrong answer key" detector (Roadmap v4, Phase 1), which is built but not yet proven to work.
- **Grift** (arXiv 2604.16242): detects reward-gaming from the model's training gradients, about 80% accurate on code.
  - *Lacks:* needs gradient access and labelled examples of gaming.
  - *Verdict:* **CURATOR leads on simplicity and cost** (its check needs no gradients and no labels) but **Grift likely leads on catching subtler gaming**, since CURATOR's check only catches a grader that ignores the problem entirely, not one that depends on the problem but rewards the wrong thing.

### 3.4 Platforms and products (not research methods)

- **Prime Intellect Environments Hub, NVIDIA NeMo Gym, Nous Atropos, Hugging Face RL Environments + TRL, verl/OpenRLHF/AReaL/slime, OpenAI Reinforcement Fine-Tuning, Thinking Machines Tinker**
  - *Do:* give you thousands of ready-made subjects, scalable training infrastructure, or a hosted training service.
  - *Lack:* **none of them decides the mix for you.** Every one of them leaves "how much of each subject" as a fixed setting the user must choose by hand, and none checks whether a grader can be trusted.
  - *Verdict:* **Neither leads, because they don't compete on this question.** This is CURATOR's opening: it could plug into any of these as a small add-on (a planned "reward health check" command, Roadmap v4 Phase 6) rather than compete with them as a platform.

## 4. Scorecard

| Capability | Leader | Why |
|---|---|---|
| Avoiding a broken grader | **CURATOR** | 6% of budget vs 31–34% for SEC/DUMP; measured, not simulated |
| Accounting cost per subject in GPU-dollars | **CURATOR** | No competitor above does this |
| Checking a grader before training | Tie | CURATOR has a basic version of the fuzzing idea |
| Checking a grader during training | **CURATOR** (narrowly) | Its free, label-free check is novel; but it misses a grader that is wrong in a way that still depends on the problem |
| Raising accuracy over training on everything equally | **Competitors, weakly** | TAC and difficulty filtering report real gains; CURATOR does not yet, but neither does most of the published field (DataFlex-RL) |
| Transfer between subjects | **TAC / HarnessBandit** | They model it directly; CURATOR does not yet |
| Multi-step (agentic) tasks | **RISED / HarnessBandit** | CURATOR has none yet |
| Scale and number of seeds | **Published literature** | 1.7–8B models, up to 12 seeds, vs CURATOR's 0.5B and 3 seeds |
| Being a usable product | **The platforms** | They ship thousands of ready environments; CURATOR is a research prototype with five |

## 5. Bottom line

CURATOR solves one specific, verified problem well: it keeps a cost-aware scheduler from being fooled by a broken grader, something its closest learnability-based competitors (SEC, DUMP) measurably fail at. It does not yet solve the bigger problem the whole field is still stuck on — making the training mix itself raise accuracy over simply training on everything equally. The roadmap (`docs/ROADMAP_v4.md`) lays out what would need to be true to close that gap, without changing CURATOR's core design.
