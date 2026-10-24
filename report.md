# CURATOR: full project report, from day 1 to 22 October 2026

Branch `audit-fixes-and-gpu-run` (PR #1), latest commit `4d64b1e`.
Companion documents:
- `self_critic.md`: why the original claim could not win, and what went wrong;
- `curator-rl/docs/DECISIONS.md`: every decision, including pre-registrations D-101, D-102 and D-103;
- `curator-rl/kaggle/README_KAGGLE.md`: how to run everything.

---

## 0. Summary

1. **What CURATOR is.** An adaptive data scheduler for RL fine-tuning with verifiable rewards (RLVR).
   While a model trains with GRPO, Curator decides how much compute each training source ("arm") gets,
   and it flags sources whose rewards are broken.
2. **The original claim failed, cleanly.** On Qwen2.5-0.5B with 1.5 GPU-hours per run, Curator does not
   beat equal mixing (Uniform): **−0.1 points, 95% interval [−1.2, +1.0], 3 seeds.** No mixing method
   does: LP, random mixtures and simple rules all tie within about 1 point. `self_critic.md` explains
   why: at this scale progress is limited by the number of training steps, not by which data fills
   them.
3. **Training itself works:** +4.7 to +6.8 points of sampled accuracy over the untrained model. The
   original greedy-only evaluation (protocol 1) could not see this (+0.6 points), so the measure was
   replaced, pre-registered, by evaluation protocol 2 (D-101).
4. **What works:**
   - Curator flags the random-reward source in every seed; its reward check separates broken from real
     rewards by about 20×.
   - A pre-flight probe catches broken checkers before training: 2 of 2 broken flagged, 0 of 4 real.
   - Curator gives useful data about 45% more compute than Uniform.
5. **The most promising finding (1 seed, being confirmed).** Published curricula (SEC, DUMP) and
   reward-variance filtering are **pulled into the broken source** and lose 1.5–2.3 points. Curator is
   not. With filtering, Curator's reward check is worth **+1.8 points** (2.9 SE).
6. **The second model family (OLMo-2-1B) is inconclusive.** It is about 3× slower per step, so it got
   only 14–15 training rounds, and neither method moved it measurably.
7. **Next:** the D-103 confirmation runs (seeds 1–2), then the story decision on analysis day. About
   12 GPU-hours, inside next week's 30.

---

## 1. Timeline

| phase | dates | what happened |
|---|---|---|
| A. Audit and fixes | until 12 Oct | Repo audit (`viva.md`, `improvement_required.md`, `errors_found.md`); errors fixed; reward audit, drift handling, α/β fit, weight cap, simulator realism v2, ROI and leave-one-out, figures; dependency lock, type checking, coverage gate, CI |
| B. GPU pipeline | 12–19 Oct | Kaggle package; run survival (checkpoint every round, Hugging Face backup, resume, clean stop before session limits, crash restarts); results folder and zip; one-cell notebook; progress lines |
| C. GPU session 1 | 19–20 Oct | Curator, Uniform, LP (seed 0); ablation (Curator without calibration, pilot_uniform); system identification (2 random-mixture runs). Evaluated greedy (protocol 1) |
| D. Protocol 2 and session 2 | 20–21 Oct | Diagnosed the blind metric; pre-registered protocol 2 (D-101); untrained-model reference; re-scored all session-1 runs; Curator, Uniform, LP seeds 1–2 |
| E. Research and D-102 build | 21 Oct | Literature review; built question filtering, the pre-flight probe, the hackable arm, second-model support, pass@k; pre-registered the pilots (D-102) |
| F. Pilots | 21 Oct | Filtering pilot; SEC and DUMP baselines; OLMo (crashed, fixed, rerun); the stress pilot did not run (round 1 only) |
| G. New plan | 21–22 Oct | Self-critique; D-103 (robustness thesis, Curator-lite, confirmation runs); full-results report (this file) |

---

## 2. What was built

| component | what it does |
|---|---|
| Scheduler (Curator: D-UCB) | chooses each round's mixture over 5 arms from learning signals, cost and status |
| Reward audit | per arm: does the reward depend on the prompt? (dispersion / κ); a failing arm goes to **S5** and is capped at the exploration floor |
| Pre-flight verifier probe (D-102) | before training, scores junk answers with each arm's reward; flags arms that pay for junk; `curator_q` starts them in S5 |
| Status signals | S1 unexplored, S2 learning, S4 too hard, S5 unreliable |
| Calibration | every 10 rounds, 100 held-out paired items per slice |
| Budget ledger | charged GPU-hours = training + calibration; evaluation is uncharged |
| Trainer | TRL 1.15 GRPO, unmodified; vLLM colocated; LoRA rank 16, learning rate 1e-5; prompts injected per step |
| Question filtering (`_q`, D-102) | draws 1.75× the questions, trains on groups with a reward signal; all generation charged |
| Evaluation protocol 2 (D-101) | greedy + sampled accuracy (4 samples per item at the training temperature), per-item results, item-paired comparisons, untrained baseline, learning curves on a fixed random subset, pass@k |
| Run survival | checkpoint every round (newest 3 kept), live Hugging Face mirror, automatic resume, clean stop before session limits, crash restarts with micro-batch halving |
| Kaggle tooling | one-cell notebook, progress lines, results folder and zip, namespaces for other models and portfolios |
| Analysis | comparison report, ROI leaderboard, leave-one-out, system identification, plots |

**Arms**
- **gsm8k**: grade-school math.
- **math35**: MATH, levels 3–5.
- **mbpp**: Python, tested in a sandbox.
- **countdown**: arithmetic puzzles.
- **noisy**: GSM8K-style problems with random rewards, paid 35% of the time regardless of the answer;
  a deliberately broken source.
- **hackable** (stress tests only): a GSM8K checker that pays when the right number appears anywhere in
  the answer.

**Benchmark:** dev split, 300 items per domain (MBPP has about 60), macro over gsm8k, math35, mbpp and
countdown. The sealed test sets were never opened.

---

## 3. Before the GPUs: simulator findings

- The simulator reproduces its golden runs bit for bit (210 of 210 episodes).
- In simulation, Curator with the reward audit gained **+11–12%** over Uniform (LP +6%). Several other
  components (drift handling, α/β fit, smoothing) showed no gain, and calibration added little.
- ROI vs leave-one-out in simulation (D-100): pooled Kendall τ +0.60 and +1.00 on two scenarios; a single
  run is much weaker (median τ 0.4–0.6).
- **In hindsight:** the simulator assumed that the data mix drives learning, so it was bound to show
  gains. It was never validated against a real GPU run (`self_critic.md` §4).

---

## 4. GPU setup

| item | value |
|---|---|
| model | Qwen2.5-0.5B-Instruct (main); OLMo-2-0425-1B-DPO (second family) |
| hardware | Kaggle T4 x2 (fp16, no bf16), one run per GPU |
| shape | 16 prompts × 8 answers per step, 2 steps per round, answers up to 1,024 tokens |
| budget | 1.5 charged GPU-hours per run; this ended every Qwen run at 31–36 rounds and every OLMo run at 14–15 |
| evaluation | protocol 2: sampled (k = 4, temperature 1.0) and greedy, item-paired, untrained baseline |

---

## 5. Results

### 5.1 Session 1, evaluated greedy (protocol 1), seed 0

| run | greedy macro | compute on junk arms |
|---|---|---|
| dirichlet seed 1 (random mix) | 0.324 | 44% |
| curator_nocal | 0.310 | 13% |
| curator | 0.308 | 13% |
| pilot_uniform | 0.308 | 22% |
| dirichlet seed 0 (random mix) | 0.305 | 39% |
| uniform | 0.302 | 41% |
| lp | 0.301 | 21% |

Two random-mix seeds differed by 1.9 points, more than any two methods differed. All seven runs are
inside the noise.

**System identification (two random-mixture runs)**
- Cost model: **56 s of every step is fixed overhead** (R² 0.45, 62 steps). Per-arm cost is 0.89×
  (mbpp) to 1.58× (math35).
- Reward audit κ: gsm8k 2.60, math35 2.50, mbpp 1.80; **noisy 49.95**.
- Transfer between arms is not identifiable: posterior SD 0.2–0.7 on effects of about 0.05. The effect
  of random rewards on GSM8K: +0.05 ± 0.49.

### 5.2 Why protocol 1 was blind, and D-101

Early RL mostly sharpens the sampling distribution; the single most likely (greedy) answer barely moves.
On top of that, greedy accuracy had about ±1.8 points of evaluation noise per run, and there was no
untrained reference. **D-101**, fixed before any of its numbers existed:
- the primary measure becomes sampled accuracy (4 samples per item), compared item by item;
- an untrained-model reference is added;
- learning curves use a fixed random subset. This fixed a bug: the round-20 check had used the 100
  lowest-index items.

Greedy is always reported alongside.

### 5.3 Session 2, protocol 2: 3 seeds of Curator, Uniform and LP

**Training works; greedy hides it**

| | sampled macro | greedy macro | pass@4 |
|---|---|---|---|
| untrained model | 17.9% | 30.0% | 35.1% |
| Uniform (3 seeds) | 23.4% | 30.6% | 38.6% |
| Curator (3 seeds) | 23.3% | 30.9% | 38.8% |

- Gain over the untrained model: **+4.7 to +6.8 points per run** (4–8 SE), mostly GSM8K (+13 to +16).
- On GSM8K the gap between sampled and greedy accuracy halved, from 22 points to 11.
- Entropy fell from about 0.50 to 0.26–0.39 in every run.

**Curator vs Uniform: a precise tie** (sampled, item-paired)

| | seed 0 | seed 1 | seed 2 | across seeds |
|---|---|---|---|---|
| Curator | +0.3 | −0.2 | −0.6 | **−0.1 [−1.2, +1.0], p 0.75** |
| LP | −0.7 | +0.4 | +0.8 | +0.2 [−1.7, +2.0] |

Per domain (3-seed means), Curator vs Uniform: GSM8K 48.7 vs 49.2, MATH 16.7 vs 17.0, MBPP 27.6 vs 27.4,
Countdown 0 vs 0.

**Where the compute went**

| run | compute on junk | useful GPU-h | random-reward arm flagged S5 |
|---|---|---|---|
| Uniform s0 / s1 / s2 | 41% / 42% / 43% | 0.89 / 0.90 / 0.87 | never |
| Curator s0 / s1 / s2 | 13% / 20% / 11% | 1.31 / 1.18 / 1.33 | round 8 / 23 / 8 |
| LP s0 / s1 / s2 | 21–24% | 1.14–1.20 | never |

- Uniform matched Curator's score with about 0.9 useful GPU-hours; Curator used about 1.3.
- Countdown was solved 0 times in about 1,800 training answers.
- In every method, **35–50% of answer groups had identical rewards**, so they produced no gradient.
- Reaching Uniform's final level took Curator 3% less compute on seed 0, and it did not reach that level
  on seeds 1 and 2. There is no consistent compute saving.
- The ROI leaderboards are "not a ranking": with 3–6 calibrations, every interval spans zero.

### 5.4 What limits step time

Across 1,094 steps:
- **93% contain an answer that hits the 1,024-token cap**, and those steps take about 76 s;
- steps whose longest answer is under 600 tokens take about 40 s, and 600–900 tokens about 59 s.

Step time tracks the longest answer (correlation 0.45) more than the mean length (0.29). Curator's
per-token cost model does not capture this: it treated MBPP as cheap, but a cheap question saves nothing
while a long answer shares its step.

### 5.5 Pre-flight verifier probe (D-102)

| arm | junk reward rate | flagged |
|---|---|---|
| gsm8k, math35, mbpp, countdown | 0.00 (every probe rejected; MBPP's sandbox resists early exit and global leaks) | no |
| noisy (random rewards) | 0.30 | **yes** |
| hackable (buggy checker) | 1.00 on number spray | **yes** |

### 5.6 Pilots (D-102), seed 0, sampled, item-paired against Uniform seed 0

| run | vs Uniform | greedy vs Uniform | gain over untrained | rounds | compute on the random-reward arm |
|---|---|---|---|---|---|
| `curator_q` (Curator + filtering + gate) | −0.3 (SE 0.6) | +0.7 (0.8) | +6.2 | 27 | 6% |
| `uniform_q` (filtering only) | **−2.1 (0.6)** | +0.3 (1.3) | +4.4 | 26 | 20% drawn; about a third of trained groups (estimate) |
| SEC (published) | **−2.3 (0.6)** | +1.3 (1.0) | +4.2 | 32 | **34%** |
| DUMP (published) | **−1.5 (0.7)** | +2.0 (1.0) | +5.0 | 34 | **31%** |
| Uniform (reference) | — | — | +6.5 | 35 | 20% |

- **curator_q vs uniform_q: +1.8 points (SE 0.6)**, greedy +0.3 (1.2).
- Filtering worked as designed: the share of trained groups with a learning signal rose from 52–57% drawn
  to 89–93% kept.
- Compute to reach Uniform's final level: curator_q 8% less; uniform_q, SEC and DUMP never reached it.

**D-102 verdicts**
- H-A (filtering raises accuracy per GPU-hour) **failed**: curator_q −0.3 points and 8% compute saved,
  against a bar of +1.5 points or 20%.
- Baselines (descriptive): SEC and DUMP below Uniform on sampled accuracy, above it on greedy (1–2 SE).

**Interpretation (exploratory, 1 seed):**
- Every learning-signal heuristic treats random rewards as ideal training data. Random rewards always
  produce mixed groups and large advantages, so variance filtering, SEC and DUMP all over-train on the
  broken source.
- Curator's reward check is the only defence among the methods tested.
- The greedy disagreement is real and must be reported.

### 5.7 Second model family: OLMo-2-1B (D-102 H-B)

| run | sampled | greedy | pass@4 | rounds | vs untrained (sampled) |
|---|---|---|---|---|---|
| untrained | 17.65% | 21.83% | 29.7% | 0 | — |
| Curator | 17.81% | 21.75% | 30.3% | 15 | +0.2 (SE 0.4) |
| Uniform | 17.58% | 21.75% | 28.9% | 14 | −0.1 (SE 0.3) |

- The first attempt crashed: vLLM serves OLMo-2 through its Transformers backend, which rewrites the
  shared model class; this is now fixed.
- After the fix, OLMo trained, but each round took about 6.5 minutes, so only 14–15 rounds fit in the
  budget. 60–68% of its answer groups carried no signal. Neither method changed the model measurably.
- The evaluation did use the trained weights: the vLLM-vs-trainer gap stayed at about 0.0015, the same as
  Qwen.
- Curator vs Uniform: +0.2 points (SE 0.4). **H-B not met; inconclusive.**
- Its "23% less compute" is an artifact: Uniform's final level equals the untrained level, so reaching it
  means nothing.
- OLMo evaluation is slow: about 21 minutes for a 100-item check and about 59 minutes for the final 300.

### 5.8 Stress pilot (hackable checker, D-102 H-C)

Not run. Both runs (curator_q and Uniform with the hackable arm) reached round 1 before their session
ended. They can resume. Under D-103 this step is optional.

---

## 6. What the evidence supports

| claim | evidence | strength |
|---|---|---|
| Training improves sampled accuracy (+5–7 points); greedy barely moves | 13 runs, 4–8 SE each | strong |
| Source-level mixing does not change accuracy at this scale | Curator, LP and random mixes all tie Uniform (−0.1 [−1.2, +1.0]) | strong |
| Curator detects rewards that ignore the prompt | S5 in 3/3 seeds; κ about 50 vs about 2 | strong |
| The pre-flight probe catches broken checkers | 2/2 flagged, 0/4 false alarms (deterministic) | strong |
| Curator reallocates compute away from junk | 6–13% vs 41% (3 seeds) | strong |
| Learning-signal selectors are fooled by broken rewards; Curator is not | SEC −2.3, DUMP −1.5, filtering −2.1; curator_q vs uniform_q +1.8 | **promising, 1 seed** (D-103 confirms) |
| Filtering raises accuracy per GPU-hour | curator_q −0.3, 8% saved | not supported at this scale |
| Calibration, ROI, transfer and the per-token cost model pay off | none measurable | not supported at this scale |
| Broken rewards hurt outside Qwen | OLMo got too little training | untested |

---

## 7. Literature context (why the results look like this)

**Early RL mostly sharpens.**
- One training example gives most of the early gain ([1-shot RLVR](https://arxiv.org/html/2504.20571v1)).
- RL models beat their base at 1 try but not at many tries ([Yue et al.](https://arxiv.org/pdf/2504.13837)).

**Qwen is a special case.**
- Random rewards improve Qwen but not Llama or OLMo
  ([Spurious Rewards](https://proceedings.mlr.press/v306/shao26d.html)).
- Qwen's pretraining likely contains benchmark data
  ([Reasoning or Memorization?](https://arxiv.org/html/2507.10532v2)).

**Where data selection does help** (all at 1.5B–7B or with longer training):
- difficulty filtering, +2.2 to +3.8 points ([Bae et al.](https://arxiv.org/html/2504.03380v2));
- [GRESO](https://arxiv.org/html/2506.02177v1), about 2× faster;
- [TAC](https://www.emergentmind.com/papers/2606.25178), up to +2.8 points on six diverse domains;
- related curricula: [DUMP](https://arxiv.org/html/2504.09710v3), SEC.

**Broken verifiers are a real problem.**
- False positives drive reward hacking ([Cai et al.](https://arxiv.org/html/2510.00915v3)).
- Verifier fuzzing finds exploitable bugs ([2026](https://www.alphaxiv.org/abs/2606.01066)).

**Evaluation practice.** Results are sensitive to seeds and decoding; report averages over samples and
seeds ([A Sober Look](https://arxiv.org/html/2504.07086v1)).

**Learning rate.** For LoRA in RL, the best learning rate is about 10× full fine-tuning's; our 1e-5 is in
range ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)).

---

## 8. Engineering record

**Bugs found and fixed**

*Evaluation*
- A relaunched finished run rewound its logs and dropped its final row (restored from the backup).
- The periodic check used the lowest-index items, not a random subset.
- ROI tables were presented as rankings.
- Progress estimates were wrong for evaluation-only jobs and for OLMo's long final evaluation.

*Operations*
- Hugging Face upload bars flooded the notebook, which looked like a crash, and restarting killed
  healthy runs.
- A virtualenv restored by Kaggle persistence lost its execute permission (now built in `/tmp`).
- A vLLM port race between two parallel runs.
- Notebooks pushed through the API cannot read secrets.
- Sessions die when the browser closes.
- A Kaggle CLI path bug on Windows.
- The package's "dirty" flag counted untracked downloads.

*Model*
- OLMo-2 under vLLM 0.31: the compile hook crashed the training copy (fixed with a test).

*Before the GPUs*
- The evaluation-suite test stub.
- ROI scaling.
- The full transfer matrix was unidentifiable, so "members" is the default.
- Leave-one-out reports were missing for relaunched finished runs.
- The smoke test and compare overwrote each other's backups.

**Quality**
- About 500 tests, coverage about 91% (gate 85%), type-checked, CI.
- Question filtering verified inside TRL 1.15 on the CPU.
- 19 completed GPU training runs, 2 partial (the stress pilot) and 2 untrained-model evaluations. None
  lost; the first 2 OLMo attempts crashed and were rerun after the fix.

**Key commits**

| commit | change |
|---|---|
| `b259f9c` | run survival |
| `412b8fd` | reward audit, GPU preset |
| `6523f7b` | ROI and leave-one-out |
| `3243fbd` | results folder |
| `2417db0` | port race |
| `0018010` | venv in `/tmp` |
| `bcab58d` | no progress bars |
| `203e074` | progress lines |
| `27c674b` | protocol 2 |
| `e393ab4` | D-102 changes |
| `2349080` | OLMo fix |
| `e3c64a1` | D-103 |
| `4d64b1e` | OLMo record |

---

## 9. What went well, and what went wrong

`self_critic.md` has the full version. In short:

**Went well**
- reliable infrastructure;
- a rigorous, pre-registered evaluation that turned a null into an explained result;
- working broken-reward detection;
- a distinctive robustness signal.

**Went wrong**
- The claim was chosen in a regime where it cannot win.
- No cheap test of whether the data mix matters before building.
- A simulator that assumed the hypothesis.
- The evaluation was designed late.
- Complex machinery (calibration, ROI, transfer) before the core effect was established.
- One model family until the end.
- Compute underestimated: about 80 s per step, not the planned 42 s.

---

## 10. Status (22 Oct)

| item | status |
|---|---|
| Qwen runs (main namespace) | 17 training runs finished: Curator, Uniform, LP × 3 seeds; curator_nocal, pilot_uniform, 2 dirichlet, curator_q, uniform_q, SEC, DUMP (seed 0); plus the untrained-model evaluation |
| OLMo namespace | untrained, Curator, Uniform (seed 0): finished, inconclusive |
| Stress namespace | 2 runs at round 1, resumable, optional |
| Friends' local GPU runs | instructions and package sent; no results yet |
| Kaggle quota | 25.8 of 30 h used; **4.2 h left until 24 Oct, 05:30 IST**, then 30 h |
| Documents | `report.md` (this file), `self_critic.md`, DECISIONS D-101 / D-102 / D-103 |

---

## 11. Next steps (D-103, pre-registered)

**Thesis.** At small scale the data mix does not move accuracy (explained). Selectors that steer on
learning signals are fooled by broken rewards; Curator checks whether a reward means anything and is not
fooled.

| session | `STEPS` | runs | time |
|---|---|---|---|
| A | `["confirm", "lite"]` | SEC, DUMP, uniform_q, curator_q seeds 1–2; Curator-lite (`curator_nocal`) seeds 1–2 | ~10 h (can start on this week's remaining 4.2 h and resume after the reset) |
| B | `["clean"]` | clean_uniform seeds 0–1 (the ceiling of a perfect junk detector) | ~2 h |

**Success rules** (3 seeds; the same sign in all three and a mean of at least 2 across-seed SE; the
t-interval is reported):
- **R1:** curator_q > uniform_q.
- **R2:** SEC < Uniform and DUMP < Uniform.
- **R3:** the mechanism, junk share (descriptive).
- **R4:** Curator-lite within ±1 point of Curator.

On analysis day the story is fixed: the robustness thesis if R1 and R2 hold, otherwise the fallback
(rigorous negative result, plus the probe and the reward check as the contribution).

**Then:** a slide deck (outline in the D-103 plan) and the write-up. An accuracy win would need models of
1.5B parameters or more and longer training, which means rented GPUs and is outside this project's
budget.

---

## Appendix: where things are

| item | location |
|---|---|
| downloaded results | `curator_results/` (session 2), `curator_results_new/` (pilots), `curator_results_olmo/` (OLMo) |
| live backup | Hugging Face dataset `AnshBajpai05/curator-runs` (private); namespaces `olmo1b/`, `stress/` |
| Kaggle package | `curator-rl/dist/curator-gpu-kaggle.zip`; Kaggle dataset `anshbajpai2005/curator-gpu-kaggle` |
| notebook | `curator-rl/kaggle/curator_gpu.ipynb` (import into the Kaggle notebook `curator-gpu-runs`) |
| decisions | `curator-rl/docs/DECISIONS.md` (D-101 protocol 2, D-102 pilots, D-103 thesis and confirmation) |
| local-GPU instructions | `curator-rl/kaggle/LOCAL_GPU_INSTRUCTIONS.md` |
