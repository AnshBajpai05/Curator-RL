# CURATOR (`curator-rl`)

**Cost-aware, calibrated environment scheduling for RL training of LLMs.** CURATOR is the
scheduling layer around an unmodified GRPO trainer. While a language model is trained with
reinforcement learning on several task environments (math, code, puzzles), it decides every
round which share of the GPU budget each environment gets. The aim is the highest held-out score
when the budget runs out.

> **Status (2026-11-03, branch `audit-fixes-and-gpu-run`):** the Level-1 GPU runs are done and
> analysed (Qwen2.5-0.5B on Kaggle T4, plus one OLMo-2-1B pair). Design, simulator, scheduler,
> baselines, calibration, the ROI leaderboard and all five environments are built and tested; a full
> audit found and fixed 30+ errors (`../errors_found.md`), and D-106 fixed a scheduler-checkpoint bug
> found after a merge. Runs **survive a lost Kaggle session** (every round checkpointed, mirrored to a
> Hugging Face dataset, resumed by rerunning the same command).
>
> **Real-GPU result (see [section 5](#5-results-so-far)):** every method gains 4.6-5.7 points of
> sampled accuracy over the untrained model, and **no method beats Uniform** (Curator -0.1 points, 95%
> interval [-1.2, +1.0], 3 seeds). The pre-registered **D-103 confirmation** (3 seeds per method,
> analysed 2026-11-02) decided the story: R1 and R2 fail (SEC -1.0 and DUMP -0.8 points against
> Uniform, every seed negative but below the 2-SE bar), R3 and R4 hold. The result is therefore a
> rigorous negative result on accuracy with its mechanism: learning-signal curricula (SEC, DUMP) put 34%
> and 31% of their questions into a random-reward environment in every seed (Uniform 20%), while
> Curator's label-free reward audit flags it in all 10 Curator runs (never a real environment) and keeps
> it to 6-10%. Calibration can be dropped (R4).
> **D-108 compute test (2026-11-03):** on fresh seeds 3-4 (six runs on three machines; seed 5 incomplete) Curator-Q ended above Uniform in 6 of 6 runs (+1.36 points final accuracy; Curator-lite +0.86, SEC +0.15, DUMP -0.57), starting behind and overtaking Uniform at about 0.8 GPU-hours; at matched compute the averages are small (Curator-Q +0.21, Curator-lite +0.44, SEC -0.14, DUMP -0.42). The pre-registered primary test is PENDING (seed 5), and Curator-Q tied Uniform on seeds 0-2, so this is a signal, not a result.
> **Next:** finish D-108 seed 5; optionally pre-register the final-accuracy signal on fresh seeds; then the
> phases of `docs/ROADMAP_v4.md` (proposed, D-107).
> Tests: 521 passed, 1 skipped (POSIX-only) with the datasets downloaded (run 2026-11-01); CI rebuilds
> the datasets and enforces 85% coverage.

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
| **Outer (this project)** | the CURATOR scheduler | mixture weights over environments | learning gain per measured dollar, checked against a held-out benchmark | built; tested in the simulator and run on Kaggle T4 (Qwen2.5-0.5B, OLMo-2-1B) |

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
  3. Every K = 10 rounds, a held-out calibration check on 2 targeted slices (`claim_stale`). Gain
     credit is a weighted regression of each slice's gain on compute share, fitted jointly for arms
     that share a slice (`calib.credit: transfer`, audit E-03; the older own-slice credit is a legacy
     switch). An environment is flagged when its benchmark gain is more than 2 SE below what its
     proxy implies; 2 consecutive flagged windows (3 in the GPU preset) put it in S5.
  4. Every round, label-free reward-health checks (see *Reward-health checks* below). The reward
     audit can put an arm in S5 at any round boundary.
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

**Settings used on the GPU (`kaggle/run.py --preset gpu_tuned`, D-98).** The values above are the
simulator defaults in `configs/base.yaml`. Every Curator run on Kaggle used the preset re-tuned on the
realism-v2 scenarios (tuning seeds only, 8 configurations per method): γ = 1 (no forgetting), κ = 0.25,
τ = 1.0, reward audit on, S5 capped at the floor ε/N (`s5_cap: floor`, 2% per arm with 5 arms) and
3 consecutive mismatch windows for a calibration flag. `n_min` is rescaled to the GPU batch shape
(32 prompts per round, R = 2; `rescale_for_batch_shape`, audit E-28). The preset stays exploratory
until a pre-registered confirmation (D-99).

**Why a bandit and not deep RL.** One episode is a full LLM training run (≈ 1.4 GPU-hours), so
only tens of episodes are affordable. A bandit learns within one episode. Discounting tracks
drifting value. Dividing by cost is the bang-per-buck rule of bandits with knapsacks.

The full specification (state table, reward stages, pseudocode, hyperparameter provenance, GRPO
formulas) is in `docs/ROADMAP_v3.md`, `docs/SPEC.md` and the review brief.

### Reward-health checks

Three checks act on the mixture, never on the reward, so the policy is never trained to evade them.

| Check | Code | When | Catches | Misses |
|---|---|---|---|---|
| Pre-flight verifier check | `signals/verifier_fuzz.py` | once before training (seconds, no GPU) | a checker that pays for junk: empty or echoed answers, empty or wrong boxes, number spray, code that exits early or prints its globals | anything that only shows up during training |
| Reward audit (label-free) | `signals/audit.py` | every round, from moments every observation already carries (zero extra compute) | a reward that ignores the prompt. The dispersion ratio D of group pass rates is 1 for such a reward and 3.0–4.4 for the real verifiers; the arm is flagged when the 99% upper bound of D is below 1.5 (after 40 degrees of freedom) | a reward that depends on the prompt but rewards the wrong thing |
| Held-out calibration | `calibration/calibrator.py` | every 10 rounds, charged to the budget | an arm whose benchmark gain falls short of its proxy claim | weak at the Level-1 budget (section 5) |

- **On the GPU the audit did the work.** Every S5 in every Curator run came from the audit, and it
  never flagged a real arm. Curator-lite (no calibration) caught the noisy arm just as fast.
- **Pre-flight results on our arms:** gsm8k, math35, mbpp and countdown pay 0.00 for junk; noisy
  pays 0.30 and the stress-only `hackable` arm 1.00, so both are flagged. With
  `signals.reward_audit.preflight_gate` (used by `curator_q`) a flagged arm starts in S5 at round 1.

### Question filtering (`_q` methods, D-102)

35–50% of GRPO groups carry no reward signal in every method (all answers right or all wrong). A `_q`
method draws ⌈1.75·P⌉ questions per step, generates and scores all of them (every token is charged),
and trains on P groups, the ones with a signal first (`trainer/group_filter.py`; online difficulty
filtering and DAPO-style dynamic sampling). `uniform_q` isolates the filter; `curator_q` is Curator +
the filter + the pre-flight gate. The filter changes which questions train, not which environments
are chosen.

### Code path of one round

```
RoundObservation (per env: prompts, successes, mixed groups, tokens, seconds, dollars)
  → signals/engine.py   pass rate → LP-B → richness → unit cost → proxy reward → status
  → scheduler/ducb.py   discounted counters, UCB scores
  → scheduler/mixture.py  z-score → softmax → floor → status constraints
  → core/quotas.py      integer prompt quotas
before round 1: signals/verifier_fuzz.py (pre-flight check) → optional S5 from the start
every round:    signals/audit.py (reward audit, label-free) → S5 flags → status machine
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
| **CURATOR** | LP + richness ÷ measured cost, D-UCB | ✓ | ✓ | ✓ built (ROI vs leave-one-out); not yet run on GPU |

**Also not in the compared work:** a label-free reward audit and a pre-flight verifier check that act
on the mixture (section 2). On the GPU these, not the cost term or calibration, are what changed
Curator's behaviour.

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
| Environments: GSM8K, Countdown, noisy, **MATH35, MBPP** (hash splits, sealed tests, manifests); `hackable` stress arm (D-102) | ✅ |
| Simulator with oracles (DP, static, myopic), scenarios S-A … S-J | ✅ |
| Signal engine (pass rate, LP-B, richness, proxy, S1–S5 status) | ✅ |
| Scheduler (D-UCB + mixture + status) and baselines (Uniform, Static, LP, Std UCB, SEC-style, DUMP-style; `pilot_uniform` and `clean_uniform` static mixes) | ✅ (`dirichlet` is the system-identification design, not a competing method) |
| Fair tuning of every adaptive method (8 configs each, one-SE rule) | ✅ |
| Calibration (paired items, claim-targeted, joint `transfer` credit, S5) | ✅ built. Simulator detection 78–86% after D-95, but **not at the GPU shape**: 10–28% after the audit fixes (D-96), 0.07 with a shared noisy slice (D-104). On GPU it raised no flag; the reward audit flagged the noisy arm |
| Reward audit (label-free dispersion test, D-98) | ✅ built; on GPU it flagged the noisy arm in every Curator run and never a real arm |
| Pre-flight verifier check (D-102) | ✅ built; flags noisy (0.30) and hackable (1.00), passes the 4 real arms (0.00) |
| Question filtering, `_q` methods (D-102) | ✅ built; run on GPU for 1 seed (`curator_q`, `uniform_q`) |
| Drift-triggered discounting, α/β fit, weight cap, smoothing (B2, B4, B7) | ✅ built and tested, **off**: no measurable benefit in the simulator (D-98) |
| Calibrated Reward Engine (CRE) | ✅ built, ❌ **not adopted** (two pre-registered nulls) |
| Kaggle pilot: pass rates and costs, all 5 environments, 0.5B and 1.5B | ✅ (HF timing) |
| GRPO step time with vLLM on T4 | ✅ 42.2 s per step (16 prompts × 8, 512 tokens) |
| Per-environment cost under vLLM | ✅ generation-only spread 2.67x (MATH35 is the expensive arm), `reports/pilot/phase_e_cost_probe.md`; full GRPO steps measured later by the sysid runs: only 1.2–1.6x (section 5) |
| TRL GRPO smoke test; HF vs vLLM step time | ✅ passes; 83.5 s (HF) vs 42.2 s (vLLM) per step, so vLLM is the training backend |
| Phase F runner: TRL adapter, prompt stream, reward router, calibration on GPU, cost attribution + ledger, crash-safe checkpoints, sysid | ✅ built, tested on CPU, and **run on Kaggle T4** (Qwen2.5-0.5B, 1.5 charged GPU-h per run; OLMo-2-1B pair, D-103) |
| Kaggle GPU package (`kaggle/run.py`: setup, smoke, compare, sysid) | ✅ built; P100 supported via a PyTorch log-prob path (TRL's Triton kernel needs compute capability >= 7.0) |
| ROI leaderboard, LOO, visualisation | ✅ built (D-100); on GPU the in-run ROI is descriptive only (3 calibration evaluations give intervals that include zero); leave-one-out not yet run on GPU |
| S-I / S-J refit from measured data, pre-registered re-evaluation (seeds 400–449) | ✅ primary criteria pass; Curator ranks 3rd behind LP and Standard UCB (D-88, D-89) |
| Tests / lint | ✅ 521 passed, 1 skipped (POSIX-only) with the datasets downloaded, 2026-11-01; without the datasets the contract tests skip / ruff clean |

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
| H-claim | Calibration targets by proxy claim + staleness, 2 slices | ✅ adopted: spurious-arm detection 78–86% vs 25–32%, false flags ≤ 2.5%, cost 3.4% (D-95); does not hold at the GPU shape (D-96) |
| 2''-CRE, 2''-κ | Calibrated reward engine variants | ❌ both pre-registered evaluations fail; line stopped |
| 3–5 | Static GRPO works; Curator does not alter GRPO; cost verified | ⬜ need Phase F |
| 6 | Calibration works | ❌ simulator part not met |
| 7–10 | ROI, baselines, LOO, experiment freeze | ⬜ |

## 5. Results so far

### Real-GPU results (Kaggle T4, Qwen2.5-0.5B-Instruct, 1.5 charged GPU-h per run)

Source: `../curator_results_november/comparison.md` (every seed; earlier downloads in `../curator_results_new/`,
`../curator_results/`, `../curator_results_olmo/`).
Primary measure is evaluation protocol 2 (D-101): sampled accuracy, mean of 4 samples per item,
macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method. The sealed
test is not used. Greedy accuracy is reported alongside.

| method | seeds | sampled macro | greedy macro | ever S5 |
|---|---|---|---|---|
| base (untrained) | - | 0.1785 | 0.3000 | - |
| **curator** | 3 | 0.2326 | 0.3086 | noisy |
| curator_nocal (Curator-lite) | 3 | 0.2341 | 0.3125 | noisy |
| curator_q | 3 | 0.2324 | 0.3100 | noisy |
| uniform | 3 | 0.2339 | 0.3061 | - |
| uniform_q | 3 | 0.2260 | 0.3078 | - |
| lp | 3 | 0.2354 | 0.3069 | - |
| dirichlet (sysid design, not a method) | 2 | 0.2277 | 0.3096 | - |
| dump | 3 | 0.2264 | 0.3106 | - |
| sec | 3 | 0.2243 | 0.3131 | - |

- **Training works.** Every method gains +4.2 to +6.8 points of sampled accuracy over the untrained
  model, mostly on GSM8K (about +13 to +16 points). Countdown stays at 0 for all methods.
- **Curator ties Uniform.** Over 3 seeds the paired difference is -0.0013, 95% t-interval
  [-0.012, +0.010], sign-flip p 0.75. LP is the same (+0.0015, p 0.75). The claim "Curator beats
  Uniform" is dropped (D-103).
- **The random-reward arm is caught, by the reward audit.** Every S5 in every Curator run came from
  the label-free audit, and no real arm was ever flagged. First flag: rounds 8, 23 and 8 (Curator
  seeds 0–2), 9, 14 and 8 (Curator-lite, which has no calibration), 1, 1 and 1 (`curator_q`, pre-flight
  gate) and 7 (OLMo). The first calibration runs at round 10, so calibration did not catch it. The pre-flight
  verifier check flags it too (junk reward rate 0.30; the other four arms 0.00, limit 0.05).
- **D-103 confirmation (3 seeds each; `../Curator-Nov-results/analysis/d103_report.md`).** SEC and
  DUMP score below Uniform in every seed (SEC -2.3, -0.3, -0.2; DUMP -1.5, -0.6, -0.2 points; means
  -1.0 and -0.8) but neither clears the pre-registered 2-SE bar (1.4 and 1.9 SE): **R2 fails**. Curator
  + question filter against Uniform + filter: +1.8, -0.2, +0.3, **R1 fails**. The mechanism holds in every
  seed (**R3**): SEC 33-34% and DUMP 31% of questions on the random-reward arm against Uniform's 20%;
  Curator variants 6-10% on average (single seeds up to 16% when the audit flags late). Curator-lite
  matches Curator (+0.15 points, **R4 holds**). By D-103's own rule the story is the fallback: a rigorous
  negative result with its mechanism, and the pre-flight check and in-run audit as the contribution.
  Greedy accuracy points the other way for SEC and DUMP (+0.7 and +0.4).
- **D-108 compute test (3 Nov; `../Curator-Nov-results/compute_analysis/`).** on fresh seeds 3-4 (six runs on three machines; seed 5 incomplete) Curator-Q ended above Uniform in 6 of 6 runs (+1.36 points final accuracy; Curator-lite +0.86, SEC +0.15, DUMP -0.57), starting behind and overtaking Uniform at about 0.8 GPU-hours; at matched compute the averages are small (Curator-Q +0.21, Curator-lite +0.44, SEC -0.14, DUMP -0.42). The pre-registered primary test is PENDING (seed 5), and Curator-Q tied Uniform on seeds 0-2, so this is a signal, not a result.
- **D-103 session B (2 Nov).** SEC at the authors' default temperature still over-funds the broken reward (25%, simulator predicted 26%; tuned SEC 34%, Uniform 20%) and scores -1.1 against Uniform (1 seed); perfect junk removal (`clean_uniform`) ties Uniform (-0.00, 2 seeds); with a realistic buggy checker the pre-flight check flags it before training (junk reward 0.20) and Curator-Q caps it at 5.7% from round 1 against Uniform's 20%, but accuracy is +0.65 points (0.86 SE): D-102 H-C fails.
- **Budget moves off the useless arms, but accuracy does not follow.** Mean delivered share under
  Curator: Countdown (no learning signal at 0.5B) 5% against Uniform's 20%, noisy 6–16% against 20%.
  The freed budget goes to GSM8K, MATH35 and MBPP, but the mix differs a lot between seeds (MBPP 53% on
  seed 0, 9–10% on seeds 1–2) and the score stays level with Uniform.
- **Measured training cost per arm (sysid runs, non-negative least squares over 62–66 steps).** A
  fixed 56–66 s per step dominates, so full GRPO steps differ far less across arms than generation
  alone: MATH35 1.22x and 1.58x GSM8K, MBPP 0.84–0.89x, fit R² 0.45–0.54. The 2.67x spread of the
  generation-only probe overstates the cost lever in real training at this scale.
- **Spurious reward on this model.** Training on the noisy arm moved GSM8K by +0.150 (SD 0.415) and
  +0.053 (SD 0.486) in the two sysid runs: no measurable effect either way. Group polarisation during
  training: κ 1.5–2.6 for the real arms, about 50 for the noisy arm.
- **Compute to reach Uniform's final level** is mixed on Qwen (Curator +3% on seed 0, not reached on
  seeds 1-2). The ROI leaderboard has too few calibration evaluations to rank arms.
- **OLMo-2-1B (1 seed each, descriptive):** base 0.1765, Curator 0.1781, Uniform 0.1758 sampled macro;
  Curator reached Uniform's final level with 23% less GPU time. About 14-15 rounds fit in 1.5 GPU-h.
  Noisy arm flagged S5 again.
- **Caveats.** 1-3 seeds per method, so intervals are wide. The simulator gain of +11-12% over Uniform
  did not transfer to the GPU: at 0.5B scale the data mix barely moves accuracy (`../self_critic.md`),
  and 35-50% of GRPO groups carry no reward signal in every method. Not run on GPU yet:
  leave-one-out, the hackable-checker stress test (D-102 H-C) and `clean_uniform`.

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

> **Superseded at the GPU shape (D-96, D-104).** The rates below were measured before the audit
> fixes, with own-slice credit and the noisy arm on its own slice. At the GPU batch shape, detection
> by calibration is 0% before the fixes and 10–28% after; with the noisy arm sharing GSM8K's slice it
> falls to 0.07. Spurious-arm detection is now the reward audit's job (section 2).

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
  0.6–0.8× of GSM8K until the cost meter measures it (Gate 5). The sysid runs later measured
  full-step costs of MATH35 1.22–1.58× and MBPP 0.84–0.89× (section 5).
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
| `hackable` | GSM8K prompts | the noisy arm's GSM8K partition | train only | — | pays when the reference number appears anywhere in the response (a deliberate false-positive bug) | stress test for the pre-flight check; never used with `noisy`; not run on GPU yet |

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
│   ├── envs/          base interface, gsm8k, countdown, noisy, hackable, math35, mbpp, finite, sandbox, splits, registry
│   ├── signals/       pass rate, LP estimators, richness, proxy reward, status machine, engine,
│   │                  reward audit, pre-flight verifier check, drift detector
│   ├── scheduler/     D-UCB, mixture map, curator; baselines/ (uniform, static, lp, ucb, sec, dump, random)
│   ├── calibration/   credit (C1/C2), calibrator (targeting, S5 flags), cre (not adopted)
│   ├── simulator/     world, scenarios, oracles, harness, replay
│   ├── evaluation/    sealed-test guard, item-paired comparisons, leave-one-out
│   ├── cost/          per-arm step attribution, NNLS cost model, matched-budget ledger
│   ├── trainer/       TRL adapter, round controller, prompt stream, reward router, question filter,
│   │                  calibration and report evaluation, checkpoints, sysid, results
│   ├── roi/           ROI leaderboard
│   ├── viz/           mixture and learning-curve plots
│   └── cli.py         `curator init-run | show-config | env-info`
├── experiments/
│   ├── run_sim.py, gate2_check.py, gate2b_check.py, cre_gate.py, cre_kappa_gate.py
│   ├── sweeps/        tune_ducb_sim.py, tune_all.py (fair tuning)
│   └── analysis/      LP study, status thresholds, calibration power, CRE constants/diagnosis, curator diagnosis
├── scripts/
│   ├── download_data*.py, build_splits*.py, env_profile*.py      data pipeline
│   ├── kaggle_pilot.py, kaggle_trl_probe.py, kaggle_step_probe.py, kaggle_cost_probe.py
│   ├── make_kaggle_bundle.py   builds dist/curator-rl-bundle.zip (refuses sealed/raw data)
│   └── make_kaggle_gpu_bundle.py  builds dist/curator-gpu-kaggle.zip for the GPU runs
├── kaggle/            run.py (setup, smoke, compare, sysid, loo; presets gpu_tuned / gpu_tuned_v1),
│                      README_KAGGLE.md, LOCAL_GPU_INSTRUCTIONS.md
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
make test                    # pytest (521 pass with data, 1 POSIX-only skip on Windows)
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
| Scheduler | γ 0.95, κ 0.5, τ 0.3, ε 0.10, warm-up 6 rounds, `score_norm: zscore`, `cost_exponent: 1.0`, `status_control: hard`, S1 quota 0.30, `pull_unit: delivered`, `s5_cap: none`; drift, weight cap and smoothing off |
| **GPU preset `gpu_tuned` (D-98)** | Curator: γ 1.0, κ 0.25, τ 1.0, reward audit on, `s5_cap: floor`, 3 mismatch windows; `curator_q` adds `preflight_gate`. DUMP on GPU: window 10. Runs: LoRA r 16, learning rate 1e-5, completion cap 1024, 1.5 charged GPU-h, R 2 · P 16 · G 8 |
| Baselines (tuned, D-68) | UCB κ 0.5 / τ 0.3 · LP τ 0.75 · SEC α 0.1 / τ 0.4 · DUMP c 2.0 / τ 0.4 |
| Signals | λ 0.9, LP-B with W = 10, richness `mixed`, Beta(1,1) prior, status n_min 64, p_sat 0.7, p_hard 0.05, z_up 1.5, h 4, dwell 3 |
| Proxy | α 0.5, β 0.5, LP clip ±3; α/β fit built, off |
| Reward audit | off in `base.yaml`, on in the GPU preset: `d_max` 1.5, z 2.33, `min_df` 40 |
| Calibration | enabled, K 10, k_min 2, z_mis 2.0, 2 mismatch windows, `credit: transfer`, `flag_scope: evaluated`, `targeting: claim_stale` (D-95), 2 targets, 100 paired items per slice |
| CRE | `enabled: false` (not adopted) |
| Batch shape | R (steps/round) 5 in the simulator, 2 on GPU (`n_min` rescaled, audit E-28) · P 16 · G 8 |

Use `curator show-config` to print the resolved config. Configs are hashed into every run's
metadata.

## 11. Research rules

These rules are the reviewer in a one-person project. Every change is logged in
`docs/DECISIONS.md` (D-1 … D-107).

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
| ✅ | Phase F runner built and run on Kaggle T4 (Qwen2.5-0.5B, 1.5 GPU-h per run); evaluation protocol 2 (D-101); pilots (D-102) | Curator ties Uniform; every method +4-7 points over the untrained model |
| ✅ | Thesis changed to robustness; Curator-lite and confirmation runs pre-registered (D-103) | claim "Curator beats Uniform" dropped |
| ✅ | **D-103 session A** (`confirm`, `lite`: sec, dump, uniform_q, curator_q, curator_nocal on seeds 1-2), analysed 2026-11-02 with the analysis fixed before the runs (`experiments/analysis/d103_confirm.py`) | R1, R2 fail; R3, R4 hold; fallback story by the pre-registered rule |
| ✅ | **D-103 session B**: `clean`, the declared `sec_tau1` check and `stress` (D-102 H-C) | clean ties Uniform; SEC at the authors' default 25% on the broken reward; H-C fails (pre-flight catches the bug, accuracy +0.65, 0.86 SE) |
| ✅ | **D-108 compute test**, seeds 3-4 on three machines (analysed 3 Nov) | primary PENDING (seed 5); Curator-Q +1.36 final accuracy in 6 of 6 replicate runs (descriptive) |
| 1 | **Finish D-108 seed 5** on account `compute_s5` (rerun the same step) | completes the pre-registered primary test |
| 2 | Optional: pre-register Curator-Q final accuracy against Uniform on fresh seeds (6+) | turns the D-108 signal into a result or a null |
| 2 | **Roadmap v4** (proposed, D-107): fault zoo and key-consistency audit, health-gated question allocation, gradient-sketch transfer and TAC-lite baseline, seed count from measured SD, second model, multi-turn tool-call arm, `curator-health` command | each phase gets its own pre-registration first |
| 3 | Optional, fresh simulator seeds (>= 700): re-test gamma = 1 with a powered non-inferiority design; a harder S5 response than halving the weight | optional |

Owner actions still open:
- Confirm the proposed decisions D-52 … D-62 and D-107 (roadmap v4), and sign off Gate 0 (tag `spec-v1`).
- Run the D-103 confirmation runs; commit results and the `Diagram_&_Documentation/` folder when ready.

## 13. Known limitations and open issues

- **Simulator.** The refit scenarios use measured pass rates, costs and group polarisation. They
  still assume the learning speeds and the transfer, and the simulator has no forgetting or
  between-round interference (D-67).
- **CURATOR vs its simpler relatives.** On the refit scenarios, LP and Standard UCB beat full
  CURATOR. The main cause is discounting (D-90). Removing it was not adopted because of an
  underpowered no-regression test (D-92).
- **Calibration and the noisy arm.** Calibration does not yet improve allocation. Targeting
  rarely re-checks the noisy arm, S5 catches it in only 20–34% of seeds, and the richness term
  rewards the noisy arm's always-mixed groups (D-89, D-90). At the GPU shape the reward audit
  replaced calibration as the detector (D-96, D-98).
- **α/β fit, drift detection, weight cap, smoothing.** Built and tested (B2, B4, B7) but off: no
  measurable benefit in the simulator (D-98). α = β = 0.5.
- **Reward audit blind spot.** It cannot see a reward that depends on the prompt but rewards the
  wrong thing (a wrong answer key, a checker that pays for the reference number anywhere). The
  pre-flight check covers known junk patterns; calibration is weak at this budget.
- **Cost.**
  - Verification time is measured per arm; generation and update are split by token share. The
    sysid check of that split fits only R² 0.45–0.54 with a large fixed overhead per step, so
    per-arm training cost is approximate, and the real spread (1.2–1.6x) is much smaller than the
    generation-only 2.67x.
  - The pilot's batch-to-batch CV (0.11–0.18 on some environments) mixes prompt content with
    noise. Repeatability must be checked on identical batches.
- **Kaggle quota.** The weekly quota and whether 2 × T4 counts double are unverified.
- **Noisy arm.** Spurious-reward effects are Qwen-specific in the literature. The noisy arm was
  flagged S5 on Qwen2.5-0.5B (3 seeds) and on one OLMo-2-1B run; Llama-3.2-1B is still planned.
- **Real-GPU evidence is thin.** 1-3 seeds per method, so every interval is wide. Curator ties
  Uniform, and the simulator gain did not transfer (D-103). SEC and DUMP below Uniform holds in every seed
  but not at the pre-registered 2-SE bar (D-103 R2).
- **No reward signal in many groups.** 35-50% of GRPO groups have all-equal rewards in every method
  (D-102), so the mixture has little to steer at 0.5B and 62-72 optimizer steps per run (31-36
  rounds of R = 2; OLMo 14-15 rounds).
- **Calibration on GPU.** With 3 calibration evaluations per run, ROI intervals include zero, so the
  leaderboard is descriptive only; calibration cost 2-3 rounds without a measurable score gain
  (Curator-lite, D-103).
- **Countdown.** Its pass@16 was not measured directly, and its samples have not been audited by
  hand.

## 14. Documentation index

| Document | Content |
|---|---|
| `docs/SPEC.md` | frozen contract: objective, splits, budget rule, notation, required hyperparameters, interfaces |
| `docs/DECISIONS.md` | decision log D-1 … D-107 (plus proposed D-52 … D-62) |
| `docs/GATES.md` | every gate, its criteria, evidence and state |
| `docs/EXPERIMENT_PROTOCOL.md` | seeds, margins (δ_S = 0.01), statistics |
| `docs/COST_MODEL.md` | cost accounting and measured values |
| `docs/ROADMAP_v4.md` | proposed plan (D-107): closing the gaps without changing the core; phases 0-6, each pre-registered |
| `../curator_results*/` | real-GPU outputs: `comparison.md`, `status.md`, per-run logs and plots (Qwen runs, D-102 pilots, OLMo pair) |
| `reports/analysis/sec_dump_fidelity.md` | our SEC and DUMP against the authors' code; simulator check of their default settings |
| `../Curator-Nov-results/` | the D-103 confirmation runs: steps, raw downloads, and the analysis (`experiments/analysis/d103_confirm.py`) |
| `../Diagram_&_Documentation/` | presentation diagrams: overview, system design, one scheduling round, status lifecycle, reward-health checks, a real run round by round (seed 0); `figures/` (all report figures, regenerated by `experiments/analysis/report_figures.py`), `reports/` (frozen copies of the analyses the report cites), `analysis/` (D-103 pre-run snapshot); index in its `README.md` |
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
| E+++ | calibration targeting by proxy claim; pre-registered confirmation | adopted: spurious-arm detection 25–32% → 78–86% (D-93 … D-95); not at the GPU shape (D-96) |
| F | GRPO runner, cost ledger, crash-safe checkpoints, Kaggle package; audit fixes (30+ errors); reward audit; ROI leaderboard | runs on Kaggle T4; every method +4-7 points over the untrained model; Curator ties Uniform (D-96 … D-102) |
| G | evaluation protocol 2, pilots, robustness thesis, Curator-lite, OLMo pair, scheduler checkpoint fix, roadmap v4 | claim "beats Uniform" dropped; confirmation runs pending (D-103 … D-107) |
