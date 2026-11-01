# CURATOR Roadmap v4: closing the gaps without changing the core

**Date:** 24 October 2026
**Status:** PROPOSED. Each phase becomes active only when its pre-registration is confirmed in `DECISIONS.md` (D-107 records this roadmap).
**Builds on:**
- `RELATED_WORK.md` (landscape, 24 October 2026);
- `DECISIONS.md` D-98 … D-106;
- `../improvement_required.md`;
- `../CHANGES_AND_IMPACT.md`;
- the literature search done for this document (section 10).

---

## 1. Where CURATOR stands today

**On real hardware:** Qwen2.5-0.5B-Instruct on Kaggle T4. LoRA, GRPO through TRL 1.15. Each run is capped at 1.5 charged GPU-hours, about 110 optimizer steps. Measured with evaluation protocol 2 (D-101): sampled macro accuracy, item-paired.

| Comparison | Result | Source |
|---|---|---|
| Curator − Uniform, 3 seeds | −0.1 points [−1.2, +1.0] | D-101, D-103 |
| Every method − untrained model | +4.7 to +6.8 points | protocol-2 comparison report (D-101) |
| SEC − Uniform, seed 0 | −2.3 (SE 0.6) | D-103 |
| DUMP − Uniform, seed 0 | −1.5 (SE 0.7) | D-103 |
| uniform_q − Uniform, seed 0 | −2.1 (SE 0.6) | D-103 |
| curator_q − uniform_q, seed 0 | +1.8 (SE 0.6) | D-103 |
| Share of questions on the random-reward arm | SEC 34%, DUMP 31%, Uniform 20%, Curator 6% | D-103 |
| GRPO groups with no reward signal | 35–50% of groups, in every method | D-102 |

**In the simulator:** realism v2, GPU shape, tuning seeds only. Curator with the reward audit scores +11.6% over Uniform, against +5.9% for the LP curriculum (D-98). That gain did not transfer to the GPU.

**Summary.**
- Training works: every method gains 5–7 points over the untrained model.
- The mixture does not change accuracy at this scale.
- Selectors that steer on learning signals get pulled into a broken reward and lose accuracy. Curator does not.
- D-103 turned this into the project's thesis. The D-103 confirmation runs have not been run yet.

---

## 2. The core that stays fixed

Every change in this roadmap must keep these five properties:

1. **Outer loop only.** CURATOR sets the per-round prompt mixture over whole environments. The GRPO trainer, its loss, the optimiser and the verifiers stay unmodified.
2. **Allocation rule.** A discounted UCB bandit over the proxy reward `(α·LP′ + β·SR) / c̃`, using measured cost `c̃`. Weights come from a softmax with an exploration floor. The status machine S1–S5 sets hard caps.
3. **Reward-health layers act on the mixture, never on the reward.** These layers are:
   - the pre-flight verifier fuzz;
   - the in-run dispersion audit;
   - benchmark calibration.

   Because they never feed the reward, the policy is never trained to evade a monitor. Monitors used as reward penalties are known to decay this way (section 10, monitoring).
4. **One budget ledger.** Generation, verification, update and calibration are all charged. Methods are compared at matched GPU-hours.
5. **Pre-registration.**
   - Tune on tuning seeds only.
   - Write the criteria before the run.
   - Use fresh seeds.
   - Report negative results.

**Rule for every addition.** A new component must take one of three forms:
- a new feature inside the proxy;
- a new trigger for an existing status;
- a step inside the trainer that the ledger charges.

It must also be off by default, and committed results must still reproduce bit for bit with it off. This is the same rule the realism-v2 switches followed (D-98).

---

## 3. What CURATOR lacks, ranked

| # | Gap | Evidence | Who does better | Severity |
|---|---|---|---|---|
| G1 | No accuracy gain over Uniform on GPU | −0.1 [−1.2, +1.0], 3 seeds. Uniform is hard to beat everywhere: DataFlex-RL found no policy significantly above it at 7–8B | TAC (+1.6 to +2.8 points, 1.7–3B); Actor-Curator (large AIME gains, under review); online difficulty filtering | Critical for an accuracy claim. Not needed for the robustness thesis |
| G2 | Evidence too thin to convince a reviewer | 3 seeds, one model family trained to completion (the OLMo run reached only about 14 rounds), runs of about 110 steps | DataFlex-RL uses 12 seeds and 2 families. Single-seed swings of 5–15 points are documented ("A Sober Look") | Critical |
| G3 | The robustness thesis rests on two planted faults | Only a random reward and a "number anywhere" checker were tested | Verifier-noise studies catalogue many more fault types: wrong keys, partial-credit leaks, format-only rewards (Checking the Checkers; Fuzzing RLVR Verifiers; CATCH) | High: it is the thesis |
| G4 | Blind spot for prompt-dependent wrong rewards | The dispersion audit passes any reward that depends on the prompt. The pre-flight probes are a fixed list | Noise-corrected GRPO and Online Label Refinement model label noise. Nobody checks it at the environment level | High |
| G5 | Question filtering falls into the same trap as SEC and DUMP, and wastes compute | The filter keeps groups with differing rewards first. With measured mixed-group rates, it lifts the random arm from about 18% of drawn questions to about 30% of trained groups (Monte Carlo, section 5.3). It also generates 1.75× and discards the rest | SARA and Pilot-Commit allocate rollouts sequentially: 22% fewer rollouts than dynamic sampling (SARA); up to 1.9× faster to target than GRPO (Pilot-Commit) | High: it is the cheapest accuracy lever |
| G6 | No transfer term | Arms are independent. The calibration-based transfer matrix ranked arms almost backwards at the Level-1 budget (D-100) | TAC and HarnessBandit use gradient transfer; RISED uses rubric transfer | Medium to high |
| G7 | Cost lever untested where it matters | The real spread is 2.67×. The simulator needs about 3.5× or more before cost normalisation pays (H3). There is no multi-turn arm | RISED, HarnessBandit and TRACE work on agentic, multi-turn tasks | Medium |
| G8 | Not usable by others | No adapter for environment hubs or for trainers other than TRL | Prime Intellect Environments Hub and verifiers, NeMo Gym, verl and OpenRLHF all leave the mixture to the user and ship no reward-health check. That is an opening, not a competitor | Medium |
| G9 | Calibration and ROI do not pay off at this scale | Curator without calibration scores +0.5 to +0.9 points higher (D-98, D-103) | Not addressed by anyone | Low: keep as measurement, Curator-lite by default |
| G10 | Name collision | Actor-Curator (Gu et al., arXiv 2602.20532, Feb 2026) is a learned, problem-level bandit curator for RL post-training | — | Low, but must be cited and distinguished |

---

## 4. Strategy

At 0.5B and 1.5 GPU-hours the whole field ties Uniform, so CURATOR should not compete on accuracy there. It should compete on what the direct competitors lack.

- **(a) Reliability.** CURATOR should not spend the budget on broken reward sources, across a realistic family of verifier faults rather than one planted fault. Phase 1 builds this.
- **(b) Efficiency.** The same accuracy for less compute, through question-level allocation that is health-gated and fully charged. Phase 2 builds this.
- **(c) A transfer feature.** This lets CURATOR beat Uniform where transfer between domains exists. Phase 3 builds it.

Each of these must be measured at a statistical power a reviewer accepts (Phase 4). A multi-turn arm then tests the cost lever where the cost spread is large (Phase 5). The reward-health layer is also packaged so that others can use it without the scheduler (Phase 6).

---

## 5. Phases

Each phase lists:
- the gap it closes;
- what to build, by file;
- how to measure it cheaply first;
- the pre-registered gate;
- the compute it needs;
- what happens if it fails.

GPU compute is given in Kaggle wall-hours on T4 ×2. One run of 1.5 GPU-hours takes about 1.9 h including evaluation, and two run in parallel, so about one run per wall-hour.

### Phase 0: finish the D-103 story (week 1, 26 Oct – 1 Nov)

- **Run the pre-registered runs, as written:**
  - D-103 A (`confirm`, `lite`): 10 runs, about 10 h.
  - D-103 B: `clean` (clean_uniform seeds 0 and 1), plus `olmo` if that run has not finished. About 2–4.5 h.
- **Hold the story decision day exactly as D-103 fixes it.** Nothing in this roadmap changes that story. Every later phase is a new pre-registration.
- **Check the filter mechanism from the logs.** This needs no GPU. From `steps.jsonl` of `uniform_q_s0` and `curator_q_s0`, measure each arm's share of trained groups.
  - Prediction from section 5.3: about 0.30 on the random-reward arm under uniform_q, about 0.02 under curator_q.
  - This is post hoc on seed 0, so label it exploratory. Seeds 1 and 2 from D-103 A then confirm or refute it.
- **Read the decisive number: clean_uniform − Uniform.** It is the ceiling of a perfect junk detector.
  - If it is below 1 point, removing junk alone cannot move accuracy at this scale. The accuracy gates of Phases 2 and 3 then move to the Phase 4 setting: a larger model or a longer budget.
  - If it is 1 point or more, Phases 2 and 3 run at the current scale.
- **Housekeeping:**
  - The owner confirms or withdraws the D-99 draft. Its seeds 700–799 stay reserved until then.
  - Remove the old worktree `../Curator-RL-ansh`.
  - Tag the merge commit, for example `v3-gpu-pilot`.

### Phase 1: a reward-health suite for a family of faults (G3, G4; weeks 1–3)

**Build.**

1. **`envs/faults.py`.** Wrappers that turn any real environment into a faulty one. Each fault cites the real failure it imitates.

   | Fault | Reward | Real-world source | Current coverage |
   |---|---|---|---|
   | F1 random | Bernoulli(p), ignores the response | Spurious Rewards | Dispersion audit (exists: `noisy`) |
   | F2 answer anywhere | Pays if the reference number appears anywhere | Leaky checkers | Pre-flight fuzz (exists: `hackable`) |
   | F3 format only | Pays for a well-formed `\boxed{}` whatever is inside | Format-reward bugs | Probably the dispersion audit; to be measured |
   | F4a wrong keys, inactive | A fraction ρ ∈ {0.5, 0.7} of keys replaced by another item's key | Misaligned answer files | None (blind spot) |
   | F4b wrong keys, active | ρ = 0.5 of keys replaced by the base model's most common wrong answer | Online Label Refinement's "active noise", the harmful kind | None (blind spot) |
   | F5 partial credit | Pays within a loose tolerance, or for any number in the problem statement | Checking the Checkers | Pre-flight fuzz, partly |
   | F6 weak code tests | MBPP with only the first assert run | CATCH (coding RL) | Pre-flight fuzz, partly; sandbox screening |

   A fully wrong key file (ρ = 1) makes an arm look unsolvable. The existing S4 status already caps that, as it does for Countdown. The dangerous faults are the partial ones. Studies of noisy verifiers report little harm up to about 30% noise and damage from about 50%, so ρ starts at 0.5.

2. **`signals/key_audit.py`: a label-free key-consistency audit for F4.** It costs nothing extra, because it uses answers the reward router already extracts.
   - The idea follows self-consistency: when the policy's G answers to a question agree strongly, the modal answer is usually correct. With a sound key, agreeing groups are therefore paid much more often than scattered ones.
   - With wrong keys, the agreeing answers lose their premium. With inactive noise it shrinks by about (1 − ρ). With active noise the confident wrong answer gets paid instead.
   - Statistic: `A = P(paid | modal share ≥ m) − P(paid | modal share < m)`, pooled over rounds, with a binomial standard error. The arm is flagged when the upper confidence bound of A is below a threshold.
   - It applies to arms with an extractable answer (GSM8K, MATH, Countdown), not to code.
   - **Whether the statistic separates sound from faulty keys is an open question.** Answering it is the first measurement below. If it does not separate them, F4 is listed as uncovered and only the opt-in audit (item 3) covers it, where it applies.

3. **Opt-in cross-verifier audit.** For arms that share prompts with another verifier, revive the archived L-98 φ audit, without benchmark-gain flags.
   - In the L-98 diagnosis, the audit alone detected 0.97 with no false flags (D-105).
   - The false flags that sank L-98 came from calibration, not from the audit.

4. **Extend the pre-flight fuzz** (`signals/verifier_fuzz.py`). Add mutation operators from Fuzzing RLVR Verifiers and CATCH:
   - every number in the prompt boxed;
   - units stuffed around the answer;
   - multiple boxed answers;
   - code stubs that read the test file;
   - variants that exit early.

**Measure cheaply first: a frozen completion bank.**
- One Kaggle job of about 1–2 h. The base model writes 8 samples for about 500 train prompts per math arm. Store each extracted answer and its correctness under the true key.
- Every fault (F3–F5) can then be applied offline, and every detector's ROC curve computed on a CPU over thousands of resamples.
- Choose the operating point on one half of the prompts, split by prompt hash. Report it on the other half.
- Then add the fault types to the realism-v2 simulator as scenario variants, to measure detection lag at the GPU batch shape.

**Gate:** pre-registered as D-108, on simulator seeds 800–899 plus the held-out half of the bank.
- Each of F1, F2, F3, F4b and F5 is detected by some layer in at least 90% of runs by round 20.
- Real-arm false flags, all layers combined, are at most 5%.
- F4a and F6 are reported, not gated.
- A fault type that fails is listed as uncovered in every claim.

**GPU confirmation:** pre-registered as D-109.
- Portfolio: one fault arm replaces `noisy`, for types F3, F4b and F5.
- Methods: Uniform, SEC and Curator-lite.
- Pilot on seed 0: 9 runs, about 9 h. Seeds 1 and 2 only for fault types where Uniform or SEC lose at least 1 point against clean_uniform: up to 18 more runs.
- **R5** (extends D-103 R3): Curator-lite puts at most 13% of questions on the fault arm for every covered type, and SEC puts more there than Uniform for the faults that create reward variance (F3, F5).
- Accuracy loss against the clean ceiling is reported for every method.

**If it fails:** the claim narrows to the fault types that pass, and that list is printed in the abstract.

### Phase 2: health-gated, sequential question allocation (G5, G1; weeks 3–5)

**Why the current filter loses.** The trap and the compute waste were measured as follows.
- The filter (`trainer/group_filter.py`) keeps groups with differing rewards first. Mixed-group rates were measured on the 0.5B pilot: gsm8k 0.77, math35 0.50, mbpp 0.47, countdown 0, noisy 1.0.
- A Monte Carlo of 20,000 steps with 28 candidates and 16 kept gave these shares of trained groups: gsm8k 0.29, math35 0.20, mbpp 0.19, countdown 0.02, noisy 0.30. The noisy arm had 0.18 of the drawn questions.
- That is the SEC and DUMP failure again, from a different mechanism. It is consistent with uniform_q −2.1 and with curator_q, whose pre-flight gate sends noisy to S5, at +1.8 over uniform_q.
- Generating 1.75× candidates also adds about 15% to step time. Generation is about 8.4 s of a 42 s step, so the extra 0.75× costs about 6 s.

**Build.**
1. **Health gate.** Filtering runs only within arms not in S5, and the in-run audit as well as the pre-flight check can send an arm to S5. Uniform's filtered variant uses the pre-flight check only, and this is documented as such.
2. **Sequential probing in place of generating 1.75× and discarding** (SARA-style). Each candidate question gets k₀ answers first. A sequential test on a Beta posterior, with the arm's κ-model prior, drops questions that are likely to be saturated. The remaining G − k₀ answers are generated only for the questions kept. Every token is charged. Method suffix `_qs`.
3. **Tune k₀ and the thresholds offline** on the Phase 1 completion bank. It holds 8 answers per prompt, so any probe policy can be replayed exactly on a CPU. The objective is groups with signal per generated token.

**Gate:** pre-registered as D-110.
- Runs: curator_qs and uniform_qs, seeds 0–2, 6 runs, about 6 h. They are compared with the existing curator_nocal and uniform seeds 0–2.
- Success follows D-102's H-A bar, unchanged: curator_qs at least 1.5 points and at least 2 across-seed SE above Curator-lite, or reaching Uniform's final level with at least 20% less compute.
- Secondary: uniform_qs's share on the noisy arm.

**If it fails:** filtering stays off, and the mechanism finding (filtering is pulled into broken rewards) is reported as part of the robustness thesis.

### Phase 3: a transfer feature that fits the bandit (G6, G1; weeks 5–8)

**Build.**

1. **`signals/transfer.py`: per-arm gradient sketches** (TAC-like, but cheap).
   - A micro-batch of G = 8 completions is one question's group. Halving it after an out-of-memory error keeps it within one question. Every micro-batch therefore belongs to one arm.
   - In the existing `CuratorGRPOTrainer` subclass, override `training_step` to record the change in the LoRA gradient after each micro-batch.
   - Project that change with a count sketch to d = 4,096. Keep a per-arm EMA of the sketches.
   - LoRA r16 on all linear layers of Qwen2.5-0.5B has about 8.8 M parameters. The sketch needs one bucket index (int16) and one sign per parameter, about 20 MB, and one scatter-add per micro-batch. Its cost is measured and charged as `curator_overhead`.
   - If a future configuration enlarges the micro-batch so that it spans several arms, those micro-batches are skipped, not split.
2. **Transfer score.** `Tᵢ = cos(EMAᵢ, Σ_{j≠i, healthy} w_j·EMA_j)`: how far training on arm i also moves the directions of the other healthy arms.
   - Arms in S5 never enter the target sum, so a broken arm cannot appear transferable.
   - Prediction: a random reward's gradients are close to orthogonal to the others, so T is near 0.
3. **Proxy change.** The proxy becomes `(α·LP′ + β·SR + δ·T⁺) / c̃`, with δ = 0 by default. This adds a feature without changing the bandit.
4. **TAC-lite baseline.** The same learnability and transfer signals with no cost normalisation and no health layers. This brings the direct competitor's mechanism onto our portfolio and into Phase 1's fault zoo.
   - Open question: is a transfer-aware selector pulled into broken arms?
   - Either answer is publishable.
5. **Free logging.** Sketch logging is switched on, uncharged and with no effect on the mixture, during the Phase 2 runs. Its signal-to-noise ratio and its agreement with the sysid transfer estimates are therefore known before any Phase 3 run.

**Gate:** pre-registered as D-111.
- Runs: curator_T and tac_lite, seeds 0–2. Escalate to seeds 3–5 by D-102's rule if the interval still crosses zero. 6 to 12 runs, about 6–12 h.
- Primary: curator_T − Uniform at 2 or more across-seed SE, the claim the field cares about.
- Secondary: curator_T against Curator-lite and against tac_lite, and each method's share of questions on the broken arm.

**Simulator caveat.** The simulator has no gradients. It can test only how the bandit uses a noisy transfer signal, with the noise level measured from the GPU sketches. Its results are labelled that way (simulator seeds 900–999).

**If it fails:** δ stays 0. Report the sketch's measured noise as the reason transfer does not pay at this budget.

### Phase 4: power and generality (G2; weeks 6–10, overlapping)

1. **Seed count from the measured spread.**
   - Phase 0 gives the across-seed SD of the item-paired differences.
   - The required seeds are `n = ((t_{0.975,n−1} + t_{0.80,n−1}) · sd / δ)²`.
   - For δ = 1.5 points and an SD of about 1 point, that is about 6 seeds.
   - Final matrix: Uniform, SEC, Curator-lite and the best new variant, seeds 0–5. That adds seeds 3–5 to what exists: 12 runs, about 12 h.
2. **A second model trained properly.**
   - Either OLMo-2-1B at 3 GPU-hours, about 28 rounds instead of 14, or Qwen2.5-1.5B. The pilot already shows the audit separating arms on 1.5B: D = 3.0–4.4 against 1.0–1.09.
   - Uniform and the best Curator, 3 seeds: 6 runs, about 10 h.
3. **Budget dependence.** One pair at 4.5 GPU-hours (Uniform and the best Curator, seed 0), about 5 h. It shows whether the mixture starts to matter with more steps. Descriptive only.
4. **External evaluation.** If DataFlex-RL's platform is public, run CURATOR there as a data policy on its domain-balanced 12-benchmark summary with its paired intervals.
   - It is the single strongest answer to the "nothing beats uniform" result.
   - Check availability first. If it is not public, cite and match its reporting instead.
5. **Report several summaries:** domain-balanced macro, each domain, pass@k, and rank stability across summaries. DataFlex-RL found rankings flip with the benchmark summary (ρ = −0.33).

**Gate:** D-112, written from the Phase 0 SD before any Phase 4 run.

### Phase 5: where cost matters, a multi-turn tool-use arm (G7; weeks 8–12)

- **Build `envs/toolcall.py`** (Level 2, B11 in `improvement_required.md`):
  - deterministic multi-turn function calling against a local mock API (calculator, key-value store, lookup table);
  - the verifier checks the final state;
  - tool execution is charged to the SCORE span;
  - target cost spread is at least 5× against GSM8K.
- **Multi-turn rollouts** need either the pinned TRL's environment or rollout hook (check what TRL 1.15 offers), or a loop inside `CuratorGRPOTrainer`'s generation hook. That hook already fills batches at generation time.
- **Write the arm in an existing environment format** (verifiers / Environments Hub style) so that it is reusable. It then doubles as Phase 6's first adapter.
- **Gate (D-113): H3 on hardware.** Curator against Curator with cost normalisation off (unit cost 1), at a measured spread of 5× or more, 3 seeds (9 runs with Uniform, about 10 h). Success: compute-to-target better by at least 20%, or accuracy higher by 2 or more SE.
- **Scope.** This is one agentic environment, not a suite. The gap to RISED and HarnessBandit narrows but stays open, and the write-up says so.

### Phase 6: make it usable (G8; CPU only, in parallel from week 4)

- **`curator-health` command.** Given a set of environments (verifiers or Environments Hub format, or TRL reward functions), it runs:
  - the pre-flight fuzz;
  - a short probe generation;
  - the dispersion audit;
  - the key-consistency audit.

  It then prints a reward-health report. It is useful without the scheduler, so it is the piece most likely to be adopted, and no platform offers it.
- **`MixController` interface.** `observe(round_stats) → weights`, with thin adapters for TRL (exists), verl and OpenRLHF (data-sampler hooks).
- **Package and docs.** PyPI package, the Kaggle notebook as the quick start, and the plots that already exist under `viz/`.
- **Positioning.** CURATOR is a plug-in for the hubs, not a competing platform.

---

## 6. Timeline and compute

Kaggle's GPU quota is usually about 30 h a week (check the account). At about one run per wall-hour:

| Phase | Weeks (from 26 Oct 2026) | Kaggle wall-hours | Runs |
|---|---|---|---|
| 0 D-103 runs | 1 | 12–15 | 12–14 |
| 1 completion bank | 1–2 | 1–2 | 1 job |
| 1 GPU confirmation | 3 | 9–27 | 9–27 |
| 2 sequential filtering | 4–5 | ~6 | 6 |
| 3 transfer feature | 6–8 | 6–12 | 6–12 |
| 4 power and second model | 7–10 | ~27 | ~20 |
| 5 tool-call arm | 9–12 | ~11 | 9 + profiling |
| **Total** | 12 weeks | **~72–100** | |

- That is about 3–3.5 weeks of quota spread over 12 calendar weeks, within the original Level-1 estimate of about 125 GPU-hours.
- CPU work runs in parallel: fault wrappers, detectors, the offline studies, the simulator variants, the sketch code and Phase 6.

---

## 7. Decision rules

- **A failed phase never changes the core.** Its component stays off, its result is reported, and later phases continue.
- **Phase 0.** If clean_uniform − Uniform is below 1 point, Phases 2 and 3 are judged in the Phase 4 setting (a larger model or a longer budget), not at 0.5B and 1.5 GPU-hours.
- **Phase 1.** A fault type that fails is listed as uncovered. The robustness claim covers only the types that pass.
- **Phases 2 and 3 both fail.** The paper is the robustness and accounting paper with no accuracy claim, as D-103's fallback already states.
- **Phase 3 passes and TAC-lite resists broken arms.** The robustness advantage narrows to learnability-only selectors (SEC, DUMP, variance filtering), and the paper says so.
- **No story change after an analysis day.** This is the rule from D-103.

---

## 8. Seeds and protocol

- **Simulator seed ranges:**
  - 700–799 stay reserved for D-99 until the owner confirms or withdraws it.
  - Phase 1 uses 800–899.
  - Phase 3 uses 900–999.
  - 1000–1099 are spare.
- **GPU seeds:**
  - Pilot on seed 0, confirm on seeds 1 and 2, escalate to seeds 3–5 by D-102's rule.
  - The Phase 4 final matrix uses seeds 0–5 for every method in it.
- **Completion bank:** split into tuning and test halves by prompt hash. Operating points are chosen on the tuning half only.
- **Sealed test:** opened once per final checkpoint, at the very end, through the existing guard.
- **Pre-registration:** one D-entry per gate, written before its first run (D-108 … D-113).

---

## 9. Positioning if the gates pass

| | Env-level mix | Measured cost | Reward-health checks | Question-level | Transfer | Multi-turn | Evidence |
|---|---|---|---|---|---|---|---|
| SEC, DUMP | yes | no | no | no | no | no | pulled into broken rewards (D-103) |
| TAC (ICML 2026) | yes | no | no | no | gradients | no | 6 domains, 1.7–3B |
| Actor-Curator (preprint) | problem-level | no | no | yes | no | no | AIME, ARC-1D |
| RISED | yes | no | rubrics | yes | rubrics | yes | AgentBench (3 environments) |
| SARA, Pilot-Commit | no | rollout count | no | yes | no | TRACE: yes | math, 1.5–14B |
| Platforms (Environments Hub, NeMo Gym, verl) | user setting | no | no | no | no | yes | n/a |
| CURATOR today | yes | yes | random + hackable faults | prototype | no | no | 0.5B, 3 seeds |
| **CURATOR after v4 (target)** | yes | yes | fault zoo F1–F6, reported per type | health-gated, sequential | gradient sketch | one tool-call arm | 2 models, 6 seeds |

---

## 10. References

Each entry was read from its abstract or project page during the 24 October 2026 search. Numbers are as their authors report them and were not re-run.

**Environment and curriculum selection**
- TAC, *Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR*, arXiv 2606.25178 (ICML 2026).
- HarnessBandit, arXiv 2609.13739.
- RISED, arXiv 2610.00979.
- SEC, arXiv 2505.14970.
- DUMP, arXiv 2504.09710.
- Actor-Curator, Gu et al., arXiv 2602.20532 (preprint, under review).
- MoDoMoDo, arXiv 2505.24871.
- Fusion paradigms, arXiv 2608.27409.

**Evaluation**
- DataFlex-RL, arXiv 2609.06107.
- *A Sober Look at Progress in LM Reasoning*, arXiv 2504.07086.

**Rollout allocation**
- SARA, arXiv 2607.26253.
- Pilot-Commit, arXiv 2605.26606.
- Learning-Zone Energy, arXiv 2605.17003.
- TRACE, arXiv 2606.11119.
- Online difficulty filtering, arXiv 2504.03380 (EACL 2026).

**Reward reliability**
- Spurious Rewards, arXiv 2506.10947.
- Fuzzing RLVR Verifiers, arXiv 2606.01066.
- Checking the Checkers, arXiv 2610.17824.
- Noise-corrected GRPO, arXiv 2510.18924 (ICML 2026).
- Online Label Refinement, arXiv 2604.03993.
- *An Imperfect Verifier is Good Enough*, arXiv 2604.07666.
- CATCH, arXiv 2609.39533.

**Monitoring during training**
- Representation probes for reward hacking, arXiv 2609.19101.
- Anthropic Alignment Science, *Reducing Reward Hacking by Localizing Bad Behavior* (2026): monitors used as penalties can be evaded.
