# CURATOR: self-critique

*Written 21 October 2026, after 21 GPU runs (Qwen2.5-0.5B on Kaggle T4s, evaluation protocol 2). Every
number here comes from our own runs or from the cited papers.*

---

## 1. The honest answer: did we take on an impossible problem?

**Partly yes.** Two different things need separating.

- **The claim we set out to prove was close to impossible to demonstrate in our setting.** The claim:
  *adaptive choice between data sources raises accuracy over equal mixing (Uniform) at a fixed compute
  budget.* The setting: a 0.5B Qwen model, 1.5 GPU-hours per run, free T4s. The reason is not a bug in
  Curator. In this regime **the data mix barely changes the outcome**, so no mixing method can
  beat Uniform by much. Random mixtures, learning-progress (LP), Curator and Uniform all land within
  about ±1 point.
- **The underlying idea is not impossible.** Data selection for RL fine-tuning does give measurable
  gains in other regimes: larger models (1.5B–7B), longer training, diverse domains, and selection of
  *questions* rather than *sources* (Section 6). Curator's other claims (detecting broken rewards, not
  being fooled by them) are working or promising in our setting.

So the failure was mostly one of **choosing a regime and a claim that couldn't win**, compounded by
process mistakes that hid this for too long.

---

## 2. What Curator needed in order to beat Uniform, and what was missing

A data scheduler beats Uniform only if **all four** hold:

| # | condition | did it hold? | evidence |
|---|---|---|---|
| 1 | **Headroom:** some data mixes are clearly better than others | **No** | Random mixtures, LP, Curator and Uniform all within ±1 point of sampled accuracy. Two random-mix runs differ by more (0.305 vs 0.324 greedy) than any two methods. |
| 2 | **Findability:** the scheduler can find the better mix from in-run signals within the budget | Weak | About 6 questions per source per round. LP's mixture swung from 81% GSM8K to 75% MATH between rounds, which is noise chasing. Curator needed 8–23 rounds to flag the random-reward source. |
| 3 | **Net gain:** the advantage exceeds the scheduler's own overhead | No | Calibration costs 2–3 training rounds per run; without it Curator scored the same. |
| 4 | **Measurability:** the evaluation can see the gain | No at first, then yes | Greedy scoring (protocol 1) moved only +0.6 points with *all* of training. Protocol 2 resolves about ±1 point. |

**Condition 1 is the decisive failure.** When the best possible mix is only marginally better than
Uniform, no amount of scheduler cleverness can show a win, and better engineering of conditions 2–4
cannot fix that.

---

## 3. Where Curator goes wrong (the technical causes)

### 3.1 Root cause: wrong regime. Progress is step-limited, not data-limited.

- Early RL on a small model mostly **sharpens** what the model already knows:
  - entropy fell from about 0.50 to 0.26–0.39 in every run;
  - on GSM8K the gap between sampled and greedy accuracy halved (22 → 11 points).
- Sharpening comes from almost any math-like data, and on Qwen even from random rewards.
- **The decisive number:** Uniform reached the same score with **about 0.9** GPU-hours of useful data as
  Curator did with **about 1.3**. Giving useful data 45% more compute bought nothing.
- The likely mechanism: Adam takes a similar-sized step whether a batch is 50% or 90% informative. Early
  progress is set by the *number* of updates, not by how clean each batch is.
- The literature says the same:
  - one training example gives most of the early gain ([1-shot RLVR](https://arxiv.org/html/2504.20571v1));
  - RL mainly concentrates the model on answers it could already produce ([Yue et al.](https://arxiv.org/pdf/2504.13837)).

### 3.2 Wrong unit of selection: sources instead of questions

- **35–50% of all answer groups in every run had no learning signal.** All 8 answers got the same
  reward, so the group contributes no gradient. This was true for Uniform, Curator, LP, SEC and DUMP
  alike.
- That waste sits *inside* each source. A scheduler that only chooses *which source* cannot touch it.
- The published gains come from choosing *questions*
  ([online difficulty filtering](https://arxiv.org/html/2504.03380v2): +2.2 to +3.8 points at 3B–7B;
  [GRESO](https://arxiv.org/html/2506.02177v1): about 2× faster).

### 3.3 Wrong cost model

Curator's "cost-aware" scheduling charged each question by its own tokens. Measured reality:

- **93% of training steps contain an answer that runs to the 1,024-token cap.** Those steps take about
  76 s; steps whose longest answer is short take 40–60 s.
- System identification: **56 s of each step is fixed overhead** (R² 0.45). Per-source costs differ only
  0.89×–1.58×.
- So "cheap" questions save nothing while one long answer shares their step. Curator put 53% of its
  questions into MBPP (cheapest per question) and saved no time. That shift was close to zero-sum on the
  macro score (MBPP up, GSM8K down).

### 3.4 Machinery that costs more than it returns

| component | cost | return at this scale |
|---|---|---|
| Calibration (held-out quizzes every 10 rounds) | about 1.5% of budget plus slower rounds, so 2–3 fewer rounds | none: Curator without it scored the same |
| ROI leaderboard | analysis only | none: 3–6 calibrations give intervals like −173 to +203; "not a ranking" |
| Transfer matrix (system identification) | 2 extra runs | none: posterior SDs of 0.2–0.7 on effects of about 0.05 |
| Exploration floors | junk sources kept at 3–6% | small; each kept long-answer Countdown question also slows its step |

### 3.5 A portfolio that gave Curator nothing to win

The 5 sources were chosen so that Curator *could* show off, with two junk sources. But both turned out
to be **harmless junk**:

- **Countdown** is unsolvable for a 0.5B model (0 of about 1,800 answers), so it gives zero gradient.
  Training on it wastes time but does no damage.
- **The random-reward source** does not hurt Qwen. That is a published Qwen-specific effect
  ([Spurious Rewards](https://proceedings.mlr.press/v306/shao26d.html)); on Llama and OLMo, random
  rewards do not help.
- The three useful sources are all math or code, which leaves little real trade-off to manage.

The benchmark also blunts differences:

- Countdown is a quarter of the macro score and sits at about 0% for every method.
- MBPP's dev set has only about 60 problems, so one problem is worth about 1.7 points on that domain.

### 3.6 Too few training steps

- Each run did about 62–72 optimizer steps (31–36 rounds). The plan assumed 42 s per step and 60
  rounds; the measured step time was about 80 s, so the budget ended every run near round 33.
- Every method's learning curve was still rising at the end. All methods were on the same early
  sharpening path when they stopped.

### 3.7 Noisy steering signals

- Per round, each source contributes about 6 questions × 8 answers. Learning-progress estimates from
  that are noisy, which is why LP oscillated and Curator's proxy could not reliably rank the useful
  sources.

---

## 4. Where our *process* went wrong

These mistakes are why the regime problem was discovered after the system was built, not before.

1. **No cheap kill test first.** Three short runs would have shown the tie in an afternoon of GPU time,
   before any of the scheduler machinery existed: Uniform, a random mix, and Uniform minus the junk
   sources.
2. **The simulator encoded the hypothesis.** It assumed learning depends on the mixture and that cost is
   proportional to tokens, so it showed Curator +11–12%. Bit-exact reproducibility was verified (210/210
   episodes); **realism against a real GPU run never was.** A reproducible simulator of the wrong
   dynamics is still wrong.
3. **The evaluation was designed late.**
   - The first protocol (greedy only, no untrained-model reference) could not see a difference at this
     scale.
   - Its periodic check used the 100 lowest-index problems, not a random 100.
   - The fix (D-101) came after the first results. We managed the integrity risk by pre-registering the
     new metric before seeing its numbers and reporting both, but it should have been the first design
     decision.
4. **The literature check came after the build.** The key 2025 papers predicted our null result:
   1-shot RLVR, Spurious Rewards, Yue et al., A Sober Look, difficulty filtering. Either they were not
   weighed, or the design predates them; either way they were not used to choose the regime.
5. **Compute reality was underestimated:** 42 s per step planned, about 80 s measured, so about 35 rounds
   instead of 60.
6. **Complexity before the core effect.** D-UCB, the reward audit, calibration, the α/β fit, drift
   handling, weight caps, ROI, leave-one-out and system identification were all built before anyone
   checked that the data mix moves accuracy at all. Each component added overhead, bugs (log rollback,
   evaluation subset, port race) and analysis that turned out to be unidentifiable.
7. **One model family.** Qwen is the family where reward quality matters least. The literature already
   advised validating on others.
8. **Engineering ahead of science.** The run-survival infrastructure is genuinely good and was needed.
   But weeks went into making runs unkillable before a run had answered the core question.

---

## 5. What Curator gets right (so the critique stays balanced)

| result | evidence | strength |
|---|---|---|
| Detects rewards that ignore the prompt | flagged the random-reward source in all 3 seeds; κ about 50 vs 1.8–2.6 for real verifiers (about 20× separation) | strong |
| Pre-flight verifier check | flagged both broken sources (random rewards: 30% junk paid; buggy checker: 100% on number spray) and none of the 4 real checkers | strong (deterministic) |
| Reallocates compute away from junk | 6–13% on junk vs Uniform's 41% (3 seeds) | strong |
| **Not fooled where other adaptive methods are** | SEC put 34% and DUMP 31% of training into the random-reward source and lost 2.3 and 1.5 points; filtering alone lost 2.1; Curator tied Uniform, and with filtering beat filtering alone by +1.8 (2.9 SE) | promising (1 seed; seeds 1–2 pending) |
| Infrastructure and evaluation | 21 runs, none lost; paired, pre-registered evaluation with an untrained baseline | strong |

The "not fooled" result is Curator's genuine edge. Every learning-signal heuristic (variance, advantage,
learning progress) treats random rewards as excellent training data, because random rewards *look*
maximally informative. Curator is the only selector we tested that checks whether a reward means
anything.

---

## 6. Root-cause table

| symptom | cause | evidence | what would fix it | is the fix proven? |
|---|---|---|---|---|
| Curator ties Uniform | no headroom: progress is step-limited sharpening at 0.5B | all mixes within ±1; equal score with 0.9 vs 1.3 useful h | larger models and longer training, where data selection is shown to matter | yes in the literature (1.5B–7B), not on our hardware |
| 35–50% of groups wasted | selection at the source level | zero-signal fraction the same in every method | question-level selection | yes at 3B–7B; at 0.5B our pilot failed (and filtering alone was fooled by random rewards) |
| "Cost-aware" saves nothing | cost model ignores that the longest answer sets step time | 93% of steps capped; 76 s vs 40–60 s | step-time-based cost, length-aware batching | no, an idea only |
| Curator does fewer steps | calibration overhead | 32 vs 34–35 rounds; no-calibration variant ties | drop calibration at this scale | measured (1 seed for the ablation) |
| Junk does not hurt Uniform | Qwen insensitive to reward quality; Countdown gives zero gradient | random rewards harmless here; documented for Qwen | a non-Qwen model (OLMo run pending) | Qwen-specificity yes; our OLMo outcome unknown |
| ROI and transfer unusable | too few calibrations | intervals ±200; SD ≫ effect | many more calibrations or the pooled design | yes, but expensive |
| Early metric blind | greedy only, no baseline | greedy gain +0.6 from all of training | protocol 2 (done) | done |

---

## 7. Lessons (what we would do differently)

1. **Measure the headroom first.** Before building a selector, run the cheapest test of whether the
   thing being selected matters: best-guess mix vs Uniform vs random mix, a few seeds. If they tie,
   change the regime or the question before writing the scheduler.
2. **Validate the simulator against one real run** before trusting any simulated gain.
3. **Design the evaluation before the method:** the metric, the untrained baseline, paired
   comparisons, the number of seeds. All of it pre-registered.
4. **Read the most recent literature for the exact regime you can afford**, and pick a regime where
   positive results exist, or state up front that you are testing a regime where they do not.
5. **Prefer claims where every outcome is reportable.** "Does adaptive selection help at small scale?"
   is answerable either way. "Curator beats Uniform" is only good news if it is true.
6. **Add components only once the core effect exists.** Each one should earn its compute.
7. **Use at least two model families** for any claim about RL training signals.

---

## 8. Where this leaves the project

- **The original accuracy claim, at this scale:** answered, negatively, and the reasons are understood.
  That is a legitimate result, not a failed experiment.
- **What still stands and is worth finishing (measured, not speculative):**
  1. Confirm the robustness finding with seeds 1–2 for SEC, DUMP, `uniform_q` and `curator_q`. It was
     2–3.6 SE on seed 0, with a clear mechanism.
  2. Run OLMo (the code is fixed; the untrained score is done). It tests whether broken rewards hurt
     outside Qwen.
- **What would be needed for an accuracy win:** a regime where the literature shows gains (1.5B–3B or
  larger, longer training, diverse domains, question-level selection). That means rented GPUs, not free
  T4s.

**One-line verdict:** we picked a regime where data selection cannot show an accuracy gain and only found
out after building a large system. The system turned that into a clear, explained negative result, and
uncovered a real strength: unlike its competitors, Curator is not fooled by broken rewards.
