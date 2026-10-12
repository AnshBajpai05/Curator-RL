# CURATOR (`curator-rl`)

**Cost-aware, calibrated environment scheduling for RL training of LLMs.** CURATOR is the
scheduling layer around an unmodified GRPO trainer. While a language model is trained with
reinforcement learning on several task environments (math, code, puzzles), it decides every
round which share of the GPU budget each environment gets. The aim is the highest held-out score
when the budget runs out.

> **Status (2026-10-12, branch `audit-fixes-and-gpu-run`):** the code is complete for the Level-1 GPU
> runs. Design, simulator, scheduler, baselines, calibration and all five environments are built and
> tested; a full audit found and fixed 30+ errors (`../errors_found.md`). New since the audit: a
> label-free **reward audit** that catches a reward which ignores the prompt (the random-reward arm)
> at zero cost, the alpha/beta fit, drift-triggered discounting, realism-v2 simulator scenarios, the
> ROI leaderboard and leave-one-out, figures, and runs that **survive a lost Kaggle session** (every
> round checkpointed, mirrored to a Hugging Face dataset, resumed by rerunning the same command).
> On tuning seeds, Curator with the audit gains +11-12% over Uniform at the GPU shape versus +6% for
> the best baseline (`../CHANGES_AND_IMPACT.md`). Nothing has run on a real GPU yet: the Kaggle
> package (`kaggle/`) is the next step. 481 tests pass (1 Linux-only skip) with the datasets
> downloaded; pyright reports 0 errors; CI also rebuilds the datasets and enforces 85% coverage.

---

## Contents

1. [The problem](#1-the-problem)
2. [How CURATOR works](#2-how-curator-works)
3. [What is new compared with prior work](#3-what-is-new-compared-with-prior-work)
4. [Current status](#4-current-status)
5. [Results so far](#5-results-so-far)
6. [Environments](#6-environments)
7. [Repository layout](#7-repository-layout)
8. [Setup and quickstart](#8-setup-and-quickstart)
9. [Running on Kaggle](#9-running-on-kaggle)
10. [Configuration](#10-configuration)
11. [Research rules](#11-research-rules)
12. [Next steps](#12-next-steps)
13. [Known limitations and open issues](#13-known-limitations-and-open-issues)
14. [Documentation index](#14-documentation-index)
15. [Phase history](#15-phase-history)

---

## 0. Audit and GPU runner (2026-10-10)

- **Errors and fixes:** `../errors_found.md` (every error with evidence; most reproduced by `../audit/verify_errors.py`);
  regression tests in `tests/unit/test_audit_fixes.py`. Corrected behaviour is the default; legacy behaviour
  stays behind config switches and all pre-registered gates load `configs/frozen/d68_d94.yaml`, so earlier
  reports still reproduce (verified on the D-94 gate).
- **Most important finding:** at the GPU plan's batch shape (32 prompts per round) a status threshold tuned
  at 80 prompts per round pinned Curator near Uniform (E-28); with it fixed and every method re-tuned at
  that shape, Curator leads in the large-cost-spread scenario and is second to the LP curriculum otherwise
  (tuning seeds; confirmation pre-registered as D-97). Spurious-arm detection by calibration does not
  reach useful power at the Level-1 GPU budget.
- **Run on Kaggle:** `python scripts/make_kaggle_gpu_bundle.py`, upload `dist/curator-gpu-kaggle.zip`, follow
  `kaggle/README_KAGGLE.md` (`run.py setup`, `smoke`, `compare`, `sysid`).

## 1. The problem

RL with verifiable rewards (RLVR) trains LLM reasoning like this: the model answers, a program
checks the answer, and GRPO updates the model. Real runs train on many environments at once,
and these environments differ in three ways:

- **Cost.** Long math solutions cost more tokens, and code answers need a sandboxed test run.
- **Usefulness over time.** An environment teaches the model while the model is partly right. It
  stops teaching once the model is always right (saturated) or always wrong (no gradient).
- **Reward reliability.** A reward can rise without any real benchmark improving (spurious reward).

Most runs mix environments uniformly or with hand-picked fixed weights. That spends budget on
saturated, too-hard or misleading environments.

**Problem statement.** You have a fixed GPU budget `B`, a base model, and `N` training
environments whose learning value is unknown and drifts, and whose measured costs differ. Learn a
policy that picks mixture weights `w_t` every round so that the final held-out benchmark score at
budget `B` is as high as possible, and that flags environments whose reward is spurious.

The objective is **fixed-budget final score** (`docs/SPEC.md` §2). Gain per dollar is the online
signal used to reach it, and a reported metric, but never the objective itself.

## 2. How CURATOR works

### Two loops

| Loop | Agent | Action | Reward | Status |
|---|---|---|---|---|
| Inner | the LLM (GRPO, TRL 1.15, LoRA) | tokens | verifier score 0/1 | unchanged off-the-shelf trainer |
| **Outer (this project)** | the CURATOR scheduler | mixture weights over environments | learning gain per measured dollar, checked against a held-out benchmark | built and tested on the simulator |

### The outer MDP

- **State.** Kept per environment:
  - discounted pass-rate posterior and its interval;
  - learning progress `LP` (LP-B: the slope of the pass rate over the last 10 rounds, with a z-score);
  - signal richness `SR` (the share of prompt groups with mixed outcomes, i.e. non-zero GRPO gradient);
  - measured unit cost;
  - status S1–S5;
  - bandit counters and calibration accumulators.

  Plus the round number and the remaining budget. The model's weights are hidden, so the problem is
  partially observed and is solved as a non-stationary bandit over these statistics.
- **Action.** A point on the simplex with a floor, `w_i ≥ ε/N` (ε = 0.10). It is turned into exact
  integer prompt counts by the largest-remainder method. Rounds 1–6 are a uniform warm-up.
- **Reward.** Computed per environment and per round in three stages:
  1. Learning signal: `x_i = α·clip(LP_i/σ_LP, ±3) + β·SR_i`, with α = β = 0.5.
  2. Divide by normalised measured cost: `r̃_i = x_i / c̃_i`. Then map to [0, 1] with running
     5th/95th percentiles.
  3. Every K = 10 rounds, a held-out calibration check. Its gain credit comes from a WLS
     regression of each environment's own-slice gain on its compute share. An environment is
     flagged when its benchmark gain is more than 2 SE below what its proxy implies; two flagged
     windows put it in S5.
- **Episode and terminal condition.** One training run. It ends at the first optimiser step where
  the charged cost (training + calibration + overhead) reaches `B`. The MDP discount is 1.

### The algorithm: Discounted UCB with a status layer

```
U_i = S̃_i/Ñ_i + κ·sqrt( ln(max(Σ_j Ñ_j, e)) / Ñ_i )        discounted UCB score   (γ = 0.95, κ = 0.5)
w   = (1−ε)·softmax( zscore(U) / τ ) + ε/N                  soft mixture + floor   (τ = 0.3, ε = 0.10)
w   ← status constraints                                    S3 saturated: ×0.5, ≤ 2ε/N
                                                            S4 too hard:  ×0.5, ≤ ε/N
                                                            S5 spurious:  ×0.5
                                                            S1 new: guaranteed share 0.30 (0.60 early)
Ñ_i ← γ·Ñ_i + u_i ;  S̃_i ← γ·S̃_i + u_i·r̄_i                 exposure-weighted discounted update
```

**Why a bandit and not deep RL.** One episode is a full LLM training run (≈ 1.4 GPU-hours), so
only tens of episodes are affordable. A bandit learns within one episode. Discounting tracks
drifting value. Dividing by cost is the bang-per-buck rule of bandits with knapsacks.

The full specification (state table, reward stages, pseudocode, hyperparameter provenance, GRPO
formulas) is in `docs/ROADMAP_v3.md`, `docs/SPEC.md` and the review brief.

### Code path of one round

```
RoundObservation (per env: prompts, successes, mixed groups, tokens, seconds, dollars)
  → signals/engine.py   pass rate → LP-B → richness → unit cost → proxy reward → status
  → scheduler/ducb.py   discounted counters, UCB scores
  → scheduler/mixture.py  z-score → softmax → floor → status constraints
  → core/quotas.py      integer prompt quotas
every K rounds: calibration/calibrator.py (targeted, paired) → S5 flags → status machine
```

## 3. What is new compared with prior work

| Method | Signal | Cost-aware | Held-out calibration | Validated attribution |
|---|---|---|---|---|
| Graves 2017 / TSCL 2020 | learning progress | ✗ | ✗ | ✗ |
| Online Data Mixing (2023) | loss, EXP3 | ✗ | ✗ | ✗ |
| DUMP (2025) | mean \|advantage\|, UCB | ✗ | ✗ | ✗ |
| SEC (COLM 2026) | mean \|advantage\|, TD(0) | ✗ | ✗ | ✗ |
| TAC (ICML 2026) | learnability + gradient transfer | ✗ | ✗ | ✗ |
| HarnessBandit (2026) | \|advantage\| + gradient sketch | ✗ | ✗ | ✗ |
| **CURATOR** | LP + richness ÷ measured cost, D-UCB | ✓ | ✓ | ✓ (planned: ROI vs leave-one-out) |

DataFlex-RL (Sept 2026, 12 seeds, Qwen2.5-7B) found that **no** published selection or mixing
policy beat uniform sampling with a significant paired interval. So the claim here is deliberately
narrower (`docs/ROADMAP_v3.md` §3):

> Under a fixed budget and a cost-heterogeneous portfolio, CURATOR reaches a higher held-out score
> per GPU-dollar than uniform, static, learning-progress and SEC/DUMP-style allocation. It detects
> spurious-reward environments online, and its ROI ranking agrees with leave-one-out. When costs are
> homogeneous it matches uniform.

Pre-registered hypotheses H1–H7, with the plan for negative results, are in
`docs/ROADMAP_v3.md` §3.3.

## 4. Current status

| Area | State |
|---|---|
| Spec, config, seeding, atomic I/O, run metadata, CLI | ✅ |
| Environments: GSM8K, Countdown, noisy, **MATH35, MBPP** (hash splits, sealed tests, manifests) | ✅ |
| Simulator with oracles (DP, static, myopic), scenarios S-A … S-J | ✅ |
| Signal engine (pass rate, LP-B, richness, proxy, S1–S5 status) | ✅ |
| Scheduler (D-UCB + mixture + status) and baselines (Uniform, Static, LP, Std UCB, SEC-style, DUMP-style) | ✅ |
| Fair tuning of every adaptive method (8 configs each, one-SE rule) | ✅ |
| Calibration (paired items, claim-targeted, own-slice credit, S5) | ✅ built; spurious-arm detection 78–86% after D-95 (Gate 6-sim ❌ as originally defined) |
| Calibrated Reward Engine (CRE) | ✅ built, ❌ **not adopted** (two pre-registered nulls) |
| Kaggle pilot: pass rates and costs, all 5 environments, 0.5B and 1.5B | ✅ (HF timing) |
| GRPO step time with vLLM on T4 | ✅ 42.2 s per step (16 prompts × 8, 512 tokens) |
| Per-environment cost under vLLM | ✅ spread 2.67x (MATH35 is the expensive arm), `reports/pilot/phase_e_cost_probe.md` |
| TRL GRPO smoke test; HF vs vLLM step time | ✅ passes; 83.5 s (HF) vs 42.2 s (vLLM) per step, so vLLM is the training backend |
| Phase F runner: TRL adapter, prompt stream, reward router, calibration on GPU, cost attribution + ledger, crash-safe checkpoints, sysid | ✅ built and tested on CPU (real TRL path with a tiny model; fake backend end to end) — not yet run on a GPU |
| Kaggle GPU package (`kaggle/run.py`: setup, smoke, compare, sysid) | ✅ built; P100 supported via a PyTorch log-prob path (TRL's Triton kernel needs compute capability >= 7.0) |
| ROI leaderboard, LOO, visualisation | ⬜ empty packages |
| S-I / S-J refit from measured data, pre-registered re-evaluation (seeds 400–449) | ✅ primary criteria pass; Curator ranks 3rd behind LP and Standard UCB (D-88, D-89) |
| Tests / lint | ✅ 404 passed, 1 skipped (POSIX-only; 23 contract tests need the datasets downloaded) / ruff clean |

### Gates (`docs/GATES.md`)

| Gate | Meaning | State |
|---|---|---|
| 0 | Spec and scope frozen | ⏳ owner sign-off pending (tag `spec-v1` not yet created) |
| 1 | Interface + 3 environments | ✅ |
| 1B' | MATH, MBPP, too-hard arm | ✅ portfolio rule passes on 0.5B (D-87) |
| 2 | Simulator recovery | 🟡 criteria a–e pass on 50 evaluation seeds; dev target (oracle-gap closure ≥ 75%) not met (43%) |
| 2'' | Beats SEC/DUMP-style bandits | ❌ on the placeholder S-I (passes on S-C and S-J) |
| 2''-R | Gate 2'' on scenarios refit from measured data | ✅ passes in both cost brackets; caveats in D-89 |
| H-γ1 | Curator without discounting | ❌ not adopted: large gains on the refit scenarios, but the no-regression check fails narrowly on S-A and S-D (D-92) |
| H-claim | Calibration targets by proxy claim + staleness, 2 slices | ✅ adopted: spurious-arm detection 78–86% vs 25–32%, false flags ≤ 2.5%, cost 3.4% (D-95) |
| 2''-CRE, 2''-κ | Calibrated reward engine variants | ❌ both pre-registered evaluations fail; line stopped |
| 3–5 | Static GRPO works; Curator does not alter GRPO; cost verified | ⬜ need Phase F |
| 6 | Calibration works | ❌ simulator part not met |
| 7–10 | ROI, baselines, LOO, experiment freeze | ⬜ |

## 5. Results so far

### Simulator (mean final score, 50 evaluation seeds 100–149, tuned configs)

| Scenario | Uniform | Static | LP | Std UCB | SEC-style | DUMP-style | **CURATOR** |
|---|---|---|---|---|---|---|---|
| S-A | 0.579 | 0.608 | 0.610 | 0.624 | 0.578 | 0.572 | **0.640** |
| S-B | 0.405 | 0.418 | 0.417 | **0.421** | 0.413 | 0.413 | 0.416 |
| S-C (costs vary independently of value) | 0.342 | 0.344 | 0.346 | 0.359 | 0.342 | 0.344 | **0.403** |
| S-I (long horizon; valuable arms are expensive) | 0.356 | 0.352 | 0.369 | **0.372** | 0.365 | 0.364 | 0.356 |
| S-J (S-I with equal costs) | 0.355 | 0.346 | 0.372 | **0.384** | 0.367 | 0.367 | 0.365 |

**Reading:**
- **S-C.** Cost-awareness is a large lever when cost and value are unrelated: +0.06 over
  SEC/DUMP-style (95% CI excludes 0).
- **S-I.** When the valuable environments are also the expensive ones, CURATOR loses to the
  learnability bandits (−0.008), and Standard UCB is the strongest method.
- **Oracle gap.** CURATOR closes 43% of the gap between Uniform and the oracle in S-A. The
  structural ceiling with a 6-round warm-up in a 25-round episode is 75.9% (D-65).

### Negative results (reported, not hidden)

- **Calibrated Reward Engine** (Gaussian posterior per arm, D-76). Evaluated on fresh seeds
  200–249: it fails Gate 2'' (i) on S-I and is statistically indistinguishable from targeted
  CURATOR (D-79).
- **CRE with retuned exploration** (H-κ, D-80). Pre-registered and evaluated on fresh seeds
  300–349: it also fails (D-82). The CRE line is stopped.
- **Calibration power study** (D-74/D-75). Targeted calibration detects the noisy arm in up to 87%
  of seeds within 10% of the budget, but with 7.3% false flags. The configuration selected by the
  pre-registered rule detects it in 33% (S-I) and 10% (S-J) of seeds.
- **Honest conclusion so far.** Feeding sparse measured gains back into the bandit does not
  improve allocation in this simulator.

### Simulator refit from measured data (D-88, D-89)

S-I and S-J were refit from the Kaggle measurements by `experiments/analysis/refit_scenarios.py`
(`reports/analysis/refit_scenarios.md`):
- **What the measurements set:**
  - difficulties come from pass@1;
  - unit costs in GPU-hours come from generation + pooled verification + the policy update. The
    update's attribution is bracketed: R1 = fully token-proportional (3.46× spread), R2 = half
    fixed (2.06×);
  - the benchmark is a macro-average over domains with a test set.
- **The group-polarisation parameter changed most.** The share of mixed groups fits a Beta-binomial
  concentration of **κ = 2.14**, while the placeholder scenarios assumed 25. Real prompt groups are
  far more often all-right or all-wrong.
- **What is still assumed:** learning speeds and the GSM8K↔MATH35 transfer.

Pre-registered evaluation, fresh seeds 400–449, frozen configs (`reports/gate2/refit_gate.md`):

| Scenario | Uniform | Static | LP | Std UCB | SEC-style | DUMP-style | **CURATOR** | CURATOR, no calib. |
|---|---|---|---|---|---|---|---|---|
| S-I-R1 (cost spread 3.46×) | 0.291 | 0.290 | 0.330 | **0.330** | 0.292 | 0.292 | 0.319 | 0.323 |
| S-I-R2 (cost spread 2.06×) | 0.290 | 0.290 | 0.331 | **0.335** | 0.292 | 0.291 | 0.320 | 0.316 |
| S-J-R (equal costs) | 0.290 | 0.297 | **0.336** | 0.335 | 0.290 | 0.294 | 0.323 | 0.326 |

- **The primary criteria pass.** CURATOR beats SEC-style and DUMP-style by +0.027 to +0.029 in
  both cost brackets (CIs exclude 0). It does not underperform Uniform in the equal-cost twin.
- **Why the learnability bandits lose.** They spend 35–36% of the budget on the noisy arm. With
  realistic polarisation, the real environments have few mixed groups, while a random reward
  always produces them. A mean-|advantage| signal therefore ranks the spurious arm highest.
- **But CURATOR is not the best method.** LP and Standard UCB beat it by 0.011–0.014 in every
  refit scenario.
- **The gain is not cost-driven.** CURATOR gains as much over Uniform with equal costs.
- **Calibration has no measurable effect.** The noisy arm reaches S5 in only 20–34% of seeds.
- **These are reported as findings, not tuned away.** Any fix needs a new pre-registration (D-89).

### Diagnosis and the γ = 1 test (D-90 … D-92)

**Diagnosis.** A 2⁵ factorial on tuning seeds 0–49 (`reports/analysis/refit_diagnosis.md`)
switched each of CURATOR's components on and off.
- **Discounting is what costs CURATOR.** Forgetting with γ = 0.95 loses 0.013–0.020. With
  polarised groups the signals are noisy, and a 20-round memory throws information away.
- **Cost normalisation pays only without discounting:** +0.016 when the cost spread is 3.46×, 0
  when costs are equal.
- **The proxy is partly fooled by the noisy arm.** Learning progress is barely detectable within 70
  rounds, so the proxy runs mostly on richness, and the noisy arm has the highest richness.
- **Calibration rarely looks at the noisy arm.** Targeted calibration evaluates the most-funded arm,
  so the noisy slice is re-checked under once per run.

**The γ = 1 test** (pre-registered, fresh seeds 500–549, `reports/gate2/gamma_gate.md`) changed
only the discount.

| Scenario | CURATOR (γ 0.95) | CURATOR (γ 1) | Std UCB | LP | SEC-style |
|---|---|---|---|---|---|
| S-I-R1 | 0.320 | **0.354** | 0.338 | 0.329 | 0.291 |
| S-I-R2 | 0.324 | **0.343** | 0.339 | 0.334 | 0.290 |
| S-J-R | 0.324 | **0.340** | 0.338 | 0.335 | 0.297 |

- **What passed:** γ = 1 improves CURATOR by +0.017 to +0.034, beats SEC/DUMP-style by +0.05 to
  +0.06, and beats Standard UCB in S-I-R1 (+0.016, CI > 0). It ties Standard UCB elsewhere.
- **Why it was not adopted:** the pre-registered no-regression check on the original scenarios
  failed narrowly. S-A and S-D show −0.003 and −0.004, with CIs reaching −0.012 against a −0.01
  margin. 50 seeds were too few for that test, which is a design flaw in the pre-registration. The
  rule still stands, so **γ stays 0.95**.
- **Concentration caveat.** γ = 1 also concentrates allocation (max weight 0.67 vs 0.38), which
  the simulator does not penalise.

### Calibration targeting fix (D-93 … D-95)

**Cause.** Targeted calibration evaluated the *most-funded* arm, usually GSM8K, so the noisy arm
(high richness, flat learning progress) was re-checked under once per run.

**Fix.** Two new targeting rules were designed, tuned on seeds 0–49 under a selection rule written
beforehand, and confirmed once on fresh seeds 600–699 (100 seeds, `reports/gate2/claim_gate.md`):
- `claim` ranks slices by the budget-weighted proxy claim (share × proxy signal);
- `claim_stale` multiplies that by (1 + windows since the slice was last checked);
- adopted: `claim_stale`, 2 slices per window.

| Scenario | Noisy arm caught (S5), before → after | False-flag rate | Calibration cost | Score vs before |
|---|---|---|---|---|
| S-I-R1 | 0.25 → **0.86** | 0.020 | 3.4% | +0.006 [+0.002, +0.010] |
| S-I-R2 | 0.32 → **0.81** | 0.025 | 3.3% | +0.004 [−0.000, +0.007] |
| S-J-R | 0.31 → **0.78** | 0.022 | 3.3% | +0.006 [+0.002, +0.010] |

- **What this supports:** detection of a spurious-reward environment (H4) works in the simulator,
  with few false flags, and Gate 2'' still holds.
- **What it does not:** the noisy arm's budget share falls only from 0.23 to 0.19, because S5
  halves an arm's weight rather than removing it. The score gain is small. CURATOR is still not
  better than Standard UCB or LP (−0.005 to 0 against UCB, −0.011 to −0.003 against LP).
- **Exploratory:** with γ = 1 as well, CURATOR beats Standard UCB in S-I-R1 (+0.023) and ties
  elsewhere, but false flags rise to 3–6.5%. No decision was attached.
- **Reproducibility note:** `configs/base.yaml` now uses the adopted targeting. Earlier reports
  (e.g. `refit_gate.md`, `cre_gate.md`) were produced with `exposure`/1 target and are not
  regenerated; the `exposure` rule remains available as an ablation.

### Real-model measurements (Kaggle Tesla T4, fp16)

Cost probe, Qwen2.5-0.5B-Instruct. vLLM generation (the training backend), 64 dev prompts × 8
samples, each environment at its own token cap, verifier through 4 threads
(`reports/pilot/phase_e_cost_probe.md`; the earlier HF-timed pilot is in `phase_e_pilot.md`):

| env | cap | pass@1 | pass@8 | mixed groups | truncated | gen s/prompt | verify s/prompt (pool) | rel. cost | rel. tokens |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 512 | 0.350 | 0.797 | 0.766 | 0.100 | 0.524 | 0.002 | 1.00× | 1.00× |
| math35 | 1024 | 0.168 | 0.500 | 0.500 | 0.125 | 1.403 | 0.002 | **2.67×** | 1.88× |
| mbpp | 512 | 0.194 | 0.483 | 0.467 | 0.000 | 0.317 | 0.328 | 1.23× | 0.43× |
| countdown | 512 | 0.000 | 0.000 | 0.000 | 0.352 | 0.535 | 0.001 | 1.02× | 0.84× |
| noisy | 512 | 0.373 | 1.000 | 1.000 | 0.064 | 0.526 | 0.002 | 1.00× | 0.90× |

- **Portfolio accepted (D-87).** 3 learnable environments (pass@8 in [0.15, 0.85]); Countdown at
  0/1,024 samples; cost spread **2.67×** (≥ 2× required). Level 1 stays on 0.5B; 1.5B costs about
  2× more per prompt.
- **MATH35 is the expensive arm** (long answers at a 1024 cap). MBPP is cheap to generate but
  verifier-bound: 1.3 s per prompt serially, 0.33 s through 4 threads. The reward wrapper must
  verify in parallel.
- **The HF-timed pilot hid this.** HF `generate` runs each batch until its longest answer
  finishes, which compressed the spread to 1.45× (D-86).
- **Sampling bug confirmed.** GSM8K pass@1 rose from 0.20 to 0.34 once Qwen's default
  `top_k=20` and `repetition_penalty=1.05` were disabled (D-84).
- **GRPO step time** (16 prompts × 8, 512 tokens): **42.2 s with vLLM colocate vs 83.5 s with
  HF**, so vLLM is the training backend. In a vLLM step the policy update and weight sync take about
  34 of the 42 s, so per-environment *training* cost is bracketed at MATH35 1.7–2.0× and MBPP
  0.6–0.8× of GSM8K until the cost meter measures it (Gate 5).
- **Run cost.** About 1.4–2.8 GPU-hours per run (60 rounds, R = 2), depending on how much
  MATH35 a method buys.
- **TRL GRPO smoke test passes.** The reward function receives `completion_ids` and every dataset
  column, and each prompt's G completions arrive contiguously.

## 6. Environments

| id | Domain | Data | Splits (train / calib / dev) | Sealed test | Verifier | Role |
|---|---|---|---|---|---|---|
| `gsm8k` | grade-school math | openai/gsm8k | hash split | GSM8K test | numeric answer match | learnable, cheap |
| `math35` | competition math | MATH levels 3–5, 5,582 problems | 4,982 / 300 / 300 | MATH-500 | conservative `\boxed{}` normaliser, no optional libraries | learnable, harder, long answers (cap 1024 tokens) |
| `mbpp` | Python code | google-research-datasets/mbpp | 314 / 90 (official val.) / 60 | MBPP test (500) | sandboxed execution of **all** assertions (one shown in the prompt) | learnable, cheap to generate, verifier-bound |
| `countdown` | arithmetic puzzle | procedural | seed ranges | seed range | expression check | zero-signal (too hard) arm for 0.5B |
| `noisy` | GSM8K prompts | GSM8K train | train only | — | Bernoulli(0.35) reward, independent of the answer | spurious-reward probe |

- **Gold and garbage checks.** Gold answers verify 100% and garbage answers 0%, on both MATH35
  (5,582 rows) and MBPP (464 rows).
- **The MBPP sandbox:**
  - runs a separate interpreter with a timeout, CPU, memory and file-size limits (POSIX), plus
    network and process-spawn guards;
  - decides success by a random per-run sentinel printed after the last assertion, so early
    `exit()` and forged success markers fail (tested);
  - is thread-safe (limits are set inside the child, D-86).

  **It is not a security boundary.** Run it only on disposable machines.
- **Splits.** Hash-based and salted (`curator-gate0-2026-10-d20`). Manifests are checksummed in
  `data/manifests/MANIFEST.sha256` and rebuild byte-identically.

## 7. Repository layout

```
curator-rl/
├── configs/
│   ├── base.yaml              every scientific hyperparameter (no Python defaults)
│   ├── experiment/smoke.yaml
│   └── sim/scenario_s{a..j}.yaml  simulator scenarios S-A … S-J
├── src/curator_rl/
│   ├── core/          config (pydantic), types, seeding, atomic I/O, JSONL, quotas, advantage, run metadata
│   ├── envs/          base interface, gsm8k, countdown, noisy, math35, mbpp, sandbox, splits, registry
│   ├── signals/       pass rate, LP estimators, richness, proxy reward, status machine, engine
│   ├── scheduler/     D-UCB, mixture map, curator; baselines/ (static, lp, ucb, sec, dump)
│   ├── calibration/   credit (C1/C2), calibrator (targeting, S5 flags), cre (not adopted)
│   ├── simulator/     world, scenarios, oracles, harness, replay
│   ├── evaluation/    sealed-test guard
│   ├── cost/ trainer/ roi/ viz/   empty: Phase F onward
│   └── cli.py         `curator init-run | show-config | env-info`
├── experiments/
│   ├── run_sim.py, gate2_check.py, gate2b_check.py, cre_gate.py, cre_kappa_gate.py
│   ├── sweeps/        tune_ducb_sim.py, tune_all.py (fair tuning)
│   └── analysis/      LP study, status thresholds, calibration power, CRE constants/diagnosis, curator diagnosis
├── scripts/
│   ├── download_data*.py, build_splits*.py, env_profile*.py      data pipeline
│   ├── kaggle_pilot.py, kaggle_trl_probe.py, kaggle_step_probe.py, kaggle_cost_probe.py
│   └── make_kaggle_bundle.py   builds dist/curator-rl-bundle.zip (refuses sealed/raw data)
├── notebooks/         kaggle_pilot, kaggle_step_probe, kaggle_cost_probe (.ipynb)
├── reports/           sim/, gate2/, sweeps/, analysis/, env_profiles/, pilot/
├── tests/             unit/, contract/, integration/, architecture/ (import-layer rules)
├── data/              raw/ (gitignored), processed/, manifests/, test_sealed/ (never bundled)
└── docs/              spec, decisions, gates, protocol, cost model, roadmaps, Kaggle, phase explanations
```

**Layering rule.** The scheduler, signals and simulator are CPU-only and import no GPU libraries.
`tests/architecture/test_import_rules.py` enforces this.

## 8. Setup and quickstart

Requires Python ≥ 3.11.

```bash
pip install -e ".[dev]"      # or: make install
make lint                    # ruff check .
make test                    # pytest (404 pass with data, 1 POSIX-only skip on Windows)
make smoke                   # init-run on configs/experiment/smoke.yaml
```

### Data

```bash
python scripts/download_data.py && python scripts/build_splits.py                    # GSM8K (+ Countdown, noisy)
python scripts/download_data_phase_e.py && python scripts/build_splits_phase_e.py    # MATH35, MBPP
python scripts/env_profile.py && python scripts/env_profile_phase_e.py               # profiles -> reports/env_profiles/
```

### Simulator and gates

```bash
python experiments/run_sim.py --scenario configs/sim/scenario_sc.yaml \
    --methods uniform,static,lp,ucb,sec,dump,curator --seeds 20     # any method set, any scenario
python experiments/sweeps/tune_all.py --seeds 50                    # fair tuning (tuning seeds only)
python experiments/gate2_check.py --seeds 50                        # Gate 2   (evaluation seeds 100-149)
python experiments/gate2b_check.py --seeds 50                       # Gate 2''
python experiments/analysis/calibration_power.py                    # calibration power study
```

Reports are written to `reports/`. The seed ranges 100–149, 200–249 and 300–349 have already been
used for evaluation. Any new pre-registered evaluation must use fresh seeds.

## 9. Running on Kaggle

The GPU work runs on Kaggle (T4 × 2, 16 GB each, fp16 only, internet on).

1. Run `python scripts/make_kaggle_bundle.py`. It writes `dist/curator-rl-bundle.zip` (126 files,
   no sealed or raw data).
2. Upload the zip as a new version of the Kaggle dataset `curator-rl-bundle`.
3. Attach the dataset to a notebook, choose **GPU T4 × 2**, and run the cells in order.

| Notebook | Measures | State |
|---|---|---|
| `kaggle_pilot.ipynb` | pass rates, advantage, tokens and HF-timed cost per environment, for 0.5B and 1.5B in parallel | ✅ done (`reports/pilot/`) |
| `kaggle_step_probe.ipynb` | GRPO step time, HF backend and vLLM colocate | ✅ vLLM 42.2 s, HF 83.5 s |
| `kaggle_cost_probe.ipynb` | per-environment cost under vLLM at each environment's own token cap; verifier serial vs pool; plus the TRL smoke test and HF step time | ✅ done (`reports/pilot/phase_e_cost_probe.md`) |

**Known pitfalls** (all handled in the notebooks):
- **vLLM** pins its own torch. Install it in a separate environment with
  `pip install virtualenv && python -m virtualenv …`; `python -m venv` fails on the Python 3.13
  image.
- **torchao.** Kaggle ships torchao 0.10, which PEFT 0.20 rejects. Run
  `pip uninstall -y torchao` before installing TRL.
- **Qwen sampling defaults.** Qwen's `generation_config` sets `top_k=20` and
  `repetition_penalty=1.05`. HF sampling must override both; TRL and explicit vLLM
  `SamplingParams` already do.
- **Hardware.** T4 has no native bf16, so use fp16. Never mix P100 and T4 within one comparison.

Platform plan and quota notes: `docs/KAGGLE.md`. Verified library versions: `docs/TRL_SPIKE.md`.

## 10. Configuration

Everything scientific lives in `configs/base.yaml`. Key frozen values:

| Group | Values |
|---|---|
| Scheduler | γ 0.95, κ 0.5, τ 0.3, ε 0.10, warm-up 6 rounds, `score_norm: zscore`, `cost_exponent: 1.0`, `status_control: hard`, S1 quota 0.30 |
| Baselines (tuned, D-68) | UCB κ 0.5 / τ 0.3 · LP τ 0.75 · SEC α 0.1 / τ 0.4 · DUMP c 2.0 / τ 0.4 |
| Signals | λ 0.9, LP-B with W = 10, richness `mixed`, Beta(1,1) prior, status n_min 64, p_sat 0.7, p_hard 0.05, z_up 1.5, h 4, dwell 3 |
| Proxy | α 0.5, β 0.5, LP clip ±3 |
| Calibration | enabled, K 10, k_min 2, z_mis 2.0, `targeting: claim_stale` (D-95), 2 targets, 100 paired items per slice |
| CRE | `enabled: false` (not adopted) |
| Batch shape | R (steps/round) 5 in the simulator, 2 planned on GPU · P 16 · G 8 |

Use `curator show-config` to print the resolved config. Configs are hashed into every run's
metadata.

## 11. Research rules

These rules are the reviewer in a one-person project. Every change is logged in
`docs/DECISIONS.md` (D-1 … D-95).

- **Sealed test sets** (`data/test_sealed/`) are used exactly once, at the end. They are never
  bundled, uploaded or opened by any script; `evaluation/guard.py` and the bundle builder enforce
  this.
- **Seeds.** Tuning uses tuning seeds only (0–49). Evaluation seed ranges are used once each:
  100–149, 200–249 and 300–349 are spent.
- **Pre-registration.** Estimators, thresholds and acceptance criteria are written down before an
  evaluation runs. They are never changed after seeing its outcome, and nulls are reported.
- **Fair baselines.** Every adaptive method gets the same tuning budget (8 configurations) and the
  same selection rule (one-SE, least concentrated).
- **Matched budget.** Every method stops at the same `B`. Calibration and scheduler overhead are
  charged only to the methods that use them.

## 12. Next steps

| Step | Work | Output |
|---|---|---|
| ✅ | Cost probe, TRL smoke test, HF step time; portfolio frozen (D-87) | done |
| ✅ | Refit S-I / S-J from measured data; pre-registered re-evaluation on seeds 400–449 (D-88, D-89) | Gate 2''-R passes; Curator 3rd behind LP and Std UCB |
| ✅ | Diagnosis on tuning seeds (D-90); pre-registered γ = 1 test on seeds 500–549 (D-91) | not adopted (D-92); γ stays 0.95 |
| ✅ | Calibration targeting by proxy claim + staleness (D-93 … D-95) | adopted |
| 1 | Optional, new pre-registrations on fresh seeds (≥ 700): re-test γ = 1 with a powered non-inferiority design; a harder S5 response than halving the weight; or let the GPU runs decide both | Optional |
| 2 | **Phase F**: TRL trainer adapter, reward wrapper with parallel verifier pool, span-based cost meter, checkpoint and resume across Kaggle sessions | CURATOR driving real GRPO; Gates 3–5 |
| 3 | Calibration on GPU, ROI leaderboard, leave-one-out | Gates 6, 7, 9 |
| 4 | Tier-1 matrix: 5 paired seeds × methods at matched budget (≈ 125 GPU-h); then one sealed-test evaluation | Tests of H1–H6 |

Owner actions still open:
- Confirm the proposed decisions D-52 … D-62 and sign off Gate 0 (tag `spec-v1`).
- Commit the current work when ready.

## 13. Known limitations and open issues

- **Simulator.** The refit scenarios use measured pass rates, costs and group polarisation. They
  still assume the learning speeds and the transfer, and the simulator has no forgetting or
  between-round interference (D-67).
- **CURATOR vs its simpler relatives.** On the refit scenarios, LP and Standard UCB beat full
  CURATOR. The main cause is discounting (D-90). Removing it was not adopted because of an
  underpowered no-regression test (D-92).
- **Calibration and the noisy arm.** Calibration does not yet improve allocation. Targeting
  rarely re-checks the noisy arm, S5 catches it in only 20–34% of seeds, and the richness term
  rewards the noisy arm's always-mixed groups (D-89, D-90).
- **α/β refit.** Refitting the proxy weights from calibration (with a trust region) is designed but
  not wired in. α = β = 0.5.
- **Cost.**
  - The share of the policy update in each environment's training cost is unmeasured (generation
    and verification are measured). The cost meter in Phase F measures it (Gate 5).
  - The pilot's batch-to-batch CV (0.11–0.18 on some environments) mixes prompt content with
    noise. Repeatability must be checked on identical batches.
- **Kaggle quota.** The weekly quota and whether 2 × T4 counts double are unverified.
- **Noisy arm.** Spurious-reward effects are Qwen-specific in the literature, so the noisy-arm
  result needs a second model family (Llama-3.2-1B planned).
- **Countdown.** Its pass@16 was not measured directly, and its samples have not been audited by
  hand.

## 14. Documentation index

| Document | Content |
|---|---|
| `docs/SPEC.md` | frozen contract: objective, splits, budget rule, notation, required hyperparameters, interfaces |
| `docs/DECISIONS.md` | decision log D-1 … D-95 (plus proposed D-52 … D-62) |
| `docs/GATES.md` | every gate, its criteria, evidence and state |
| `docs/EXPERIMENT_PROTOCOL.md` | seeds, margins (δ_S = 0.01), statistics |
| `docs/COST_MODEL.md` | cost accounting and measured values |
| `docs/ROADMAP_v3.md` | forward plan: claims, hypotheses, portfolio, phases, ladder L1–L3, risks, sources |
| `docs/CURATOR_Implementation_Roadmap.md` | v2 technical reference (contracts, maths, ledger, LOO design) |
| `docs/KAGGLE.md`, `docs/TRL_SPIKE.md` | platform plan; TRL facts and pinned versions |
| `docs/phase_*_explanation.md`, `PHASE*_README.md` | what each phase built, why, and how it was verified |
| `docs/proposal/` | the original project proposal |

## 15. Phase history

| Phase | Built | Key outcome |
|---|---|---|
| 1 | config, seeding, atomic I/O, run metadata, CLI | reproducible run directories |
| 2 | GSM8K, Countdown, noisy; hash splits; sealed guard | Gate 1 ✅ |
| 3 | simulator, DP/static/myopic oracles, S-A … S-H | oracle beats Uniform by 11–159% |
| 4 | signal engine; LP estimator and status-threshold studies | LP-B selected (D-40), thresholds tuned (D-42) |
| 5 | D-UCB, mixture map, Static/LP/UCB baselines, sweep | Gate 2 passes on 50 seeds |
| A | fixes after review (status priority, hard status control, Static baseline) | stale-report and S1-pump theories corrected (D-63 … D-66) |
| B | Kaggle pilot, TRL probe, vLLM step probe | step time 42.2 s; sampling bug found (D-83, D-84) |
| C | SEC/DUMP-style baselines, fair tuning, S-I/S-J, Gate 2'' | wins S-C, loses S-I (D-68 … D-71) |
| D | calibrator, targeted calibration, power study, CRE | CRE not adopted after two pre-registered nulls (D-72 … D-82) |
| E | MATH35, MBPP, sandbox; second pilot; vLLM cost probe | portfolio accepted on 0.5B, spread 2.67x, vLLM backend (D-85 … D-87) |
| E+ | S-I/S-J refit from measured data; pre-registered re-evaluation | Gate 2''-R passes; Curator 3rd behind LP and Std UCB; κ = 2.14 polarisation (D-88, D-89) |
| E++ | factorial diagnosis; pre-registered γ = 1 test | discounting is the main cost; γ = 1 not adopted (power) (D-90 … D-92) |
| E+++ | calibration targeting by proxy claim; pre-registered confirmation | adopted: spurious-arm detection 25–32% → 78–86% (D-93 … D-95) |
