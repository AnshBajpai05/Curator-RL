# CURATOR — viva notes

Cost-aware, calibrated environment scheduling for RL training of LLMs.
Repository: `curator-rl/`. Every number below comes from the code, `configs/base.yaml`, or a committed report. Decision numbers (D-xx) refer to `curator-rl/docs/DECISIONS.md`.

---

## 0. The project in one paragraph

CURATOR is the outer control loop of an LLM reinforcement-learning run. A model (Qwen2.5-0.5B-Instruct) is trained with GRPO on several task environments at once: GSM8K, MATH levels 3–5, MBPP, Countdown, and a deliberately broken "noisy" environment. Every round, CURATOR decides what fraction of the next round's prompts each environment gets. It decides from four measured signals per environment: pass rate, learning progress, signal richness and measured cost. It also periodically checks those decisions against a held-out benchmark, and it flags environments whose reward does not translate into real benchmark gains. The objective is the highest held-out score when a fixed GPU budget `B` is spent. The GRPO trainer itself is not modified; CURATOR only controls the prompt mixture.

---

## 1. The problem, stated exactly

### 1.1 Setting

- **RLVR (RL with verifiable rewards).** The model writes a completion. A program verifies it and returns reward 0 or 1, and GRPO updates the model.
- **GRPO advantage.** For one prompt, `G = 8` completions are sampled with rewards `r_1 … r_G`. The advantage of completion `i` is `A_i = (r_i − mean(r)) / (std(r) + 1e-4)`.
  - If all `G` rewards are equal (all correct or all wrong), `std = 0` and every `A_i = 0`. That prompt contributes no gradient.
  - Only *mixed* groups (`0 < k < G` successes) train the model.
- **Environments differ in three ways:**
  1. **Cost per prompt.** MATH answers are long; MBPP answers need sandboxed test execution.
  2. **Teaching value over time.** An environment stops teaching once the model always solves it (saturated, all-1 groups) or never solves it (too hard, all-0 groups).
  3. **Reward reliability.** A reward can be produced without any real skill improving (spurious reward).

### 1.2 Formal objective (`docs/SPEC.md` §2)

```
max_π  E[ S_val(θ_T) ]                         final held-out benchmark score
s.t.   Σ_t cost(w_t) ≤ B                        charged cost (training + calibration + overhead)
```

- `w_t` is the mixture (a probability vector over environments) chosen at round `t`.
- The **objective is fixed-budget final score**. Gain-per-dollar is used as the online signal and reported as a metric, but it is never the objective. This was a deliberate fix of an ambiguity in the proposal (spec defect S-10).

### 1.3 The outer decision problem as an MDP

| Element | Exact definition |
|---|---|
| State | Per environment: discounted pass-rate posterior and interval, LP slope and its z-score, richness, measured unit cost, status S1–S5, bandit counters `Ñ_i, S̃_i`, calibration accumulators. Plus round number and remaining budget. Model weights are not observed, so the problem is partially observed. |
| Action | `w ∈ simplex` with `w_i ≥ ε/N` (ε = 0.10), converted to integer prompt counts. |
| Reward | Per environment per round: proxy reward `r̄_i ∈ [0,1]` (§4.6). Every K = 10 rounds, a held-out calibration check that can move the environment to S5. |
| Transition | One round of GRPO training: R optimizer steps × P = 16 prompts × G = 8 completions. |
| Terminal | First round at which charged cost ≥ B. |
| Discount | MDP discount = 1 (only the final score counts). `γ = 0.95` in the scheduler is a *bandit forgetting factor*, not the MDP discount (SPEC §6). |

### 1.4 Why a bandit, not deep RL, supervised learning or grid search

- One episode of the outer problem is one complete LLM training run, about 1.4–2.8 GPU-hours (60 rounds, R = 2). Only tens of episodes are affordable, so a policy that must learn *across* episodes (deep RL) cannot be trained.
- There are no labelled "good mixtures" to learn from, so supervised learning has no training data.
- Grid search over mixtures needs many full training runs, which defeats the purpose of saving compute.
- A bandit learns *within* one run from partial feedback, because only environments that receive prompts produce observations.
- Discounting handles the drift in value as environments saturate.
- Dividing value by cost is the bang-per-buck rule of single-resource bandits-with-knapsacks (Badanidiyuru et al.), which is optimal for the fractional knapsack.

---

## 2. Architecture

### 2.1 Two loops

| Loop | Agent | Action | Reward | Built? |
|---|---|---|---|---|
| Inner | the LLM, GRPO via TRL 1.15 with LoRA | tokens | verifier score 0/1 | off-the-shelf trainer, unchanged |
| Outer | CURATOR scheduler | mixture weights over environments | proxy reward per measured cost, checked against held-out benchmark | built and tested in simulation; not yet connected to the trainer |

### 2.2 Code layers (enforced by `tests/architecture/test_import_rules.py`)

| Layer | Packages | May import |
|---|---|---|
| L0 | `core` (config, types, seeding, atomic I/O, quotas, advantage, run metadata) | nothing from the project |
| L1 pure | `signals`, `scheduler`, `calibration`, `roi` | `core`, numpy, scipy only. No torch, transformers, TRL or datasets |
| L2 adapters | `envs`, `cost`, `trainer`, `simulator` | `core`; only `trainer/trl_adapter.py` may import TRL |
| L3 | `evaluation`, `experiments` | everything; nothing may import the sealed-test loader except the final evaluation |

The point of this rule: the scheduler is a pure function of `RoundObservation`s. The simulator and the real trainer emit the same observation type, so the scheduler code validated in simulation is exactly the code that will run on the GPU.

### 2.3 What exists and what is empty

- **Built and tested (326 tests pass, 24 skip without downloaded data):** config, all 5 environments, splits and the sealed-test guard, simulator and oracles, signal engine, D-UCB scheduler, 6 baselines, calibrator, CRE (built, not adopted), and the Kaggle measurement scripts.
- **Empty packages (one-line `__init__.py`):** `cost/` (cost meter), `trainer/` (TRL adapter), `roi/` (ROI leaderboard), `viz/`. These are Phase F and later.

---

## 3. Data contracts (`src/curator_rl/core/types.py`)

What crosses the boundary between trainer or simulator and scheduler, every round:

`EnvRoundObs` (one per environment per round):
- `n_prompts`, `n_rollouts` (= n_prompts × G), `k_success` (number of correct rollouts)
- `n_groups_mixed`: number of prompt groups with `0 < k < G`
- `sum_score`, `sum_score_sq`: sum of per-group pass rates and their squares
- `prompt_tokens`, `completion_tokens`, `verifier_seconds`, `gpu_seconds`, `cost_usd`
- `sum_abs_adv`: sum over groups of the group mean |GRPO advantage|. Only SEC/DUMP-style baselines read it.

`RoundObservation`: `round`, `steps`, `per_env` (dict of the above), `weights_used`, `round_cost_usd`, `overhead_usd`, `budget_remaining_usd`. It carries **only measured quantities**; hidden simulator variables (true skill, true pass rate) are never in it.

`CalibrationObservation`: per-domain benchmark scores and standard errors for the evaluated slices, the paired standard error of each slice's change, `exposure_by_env` (prompt mass each environment received in the window), and the window's training and evaluation cost.

`SignalVector`: the signal engine's per-environment output (pass rate and interval, LP and z, richness, unit cost, proxy values, status).

`MixtureDecision`: weights, UCB components (`mu_hat`, `bonus`), statuses, intents, integer quotas, and a state hash for auditability.

---

## 4. Exact flow of one round

Order of operations inside the simulator harness `simulator/harness.py::run_episode`. The real trainer callback will follow the same order.

```
for round_t in 0 .. max_rounds-1:
  1. w      = scheduler.select_mixture(prev_obs)          # §4.8–4.10
  2. w      = apply_scenario_constraints(w)               # only for late-arriving envs (S-H)
  3. obs    = world.step_round(w)                         # train one round, emit RoundObservation
  4. spend += obs.round_cost_usd
  5. exposure[e] += w_e × M                               # M = R × P prompts per round
  6. scheduler.update_observation(obs)                    # signal engine + bandit counters (§4.1–4.7)
  7. if (round_t+1) % K == 0 and scheduler uses calibration:
        request = scheduler.calibration_request()         # which slices, how many items (§5.2)
        calib   = world.calibration_observation(request)  # paired held-out evaluation
        spend  += eval_cost                               # charged to this method only
        scheduler.update_calibration(calib)               # credit, z-test, S5 flags (§5)
        reset exposure and window cost
  8. every report interval: uncharged reporting evaluation (same for every method)
  9. if spend ≥ B: stop                                   # overshoot at most one round
final score = benchmark evaluation (uncharged)
```

Inside `scheduler.update_observation` (`scheduler/ducb.py`), the signal engine runs first, then the bandit counters are updated. The engine updates per environment in **sorted environment-id order**, so results do not depend on dict ordering. The steps:

### 4.1 Pass-rate posterior (`signals/passrate.py`)

- Discounted counts, λ = 0.9: `s ← λ·s + k_success`, `n ← λ·n + n_rollouts`, `groups ← λ·groups + n_prompts`.
- The decay is applied **even in rounds where the environment got no prompts**, so evidence goes stale.
- Posterior under a Beta(1, 1) prior:
  ```
  p̂ = (s + α0) / (n + α0 + β0)
  sd = sqrt( p̂(1−p̂) / (n + α0 + β0 + 1) )
  interval [p_lo, p_hi] = p̂ ± sd                         (68% interval)
  ```
- Group-level standard error: `sqrt(p̂(1−p̂) / groups)`. The prompt group, not the rollout, is the unit, because the 8 rollouts of one prompt are correlated.
- `n_groups_eff` = discounted group count. It drives S1 exit.
- Two extra discounted windows (λ_fast = 0.7, λ_slow = 0.95) are kept for the alternative LP-A estimator.

### 4.2 Learning progress, LP-B (`signals/progress.py`)

- For each round in which the environment got prompts, the engine stores `(raw pass rate of that round, n_prompts, group SE)`. History is capped at 2W.
- LP-B is a **two-stage weighted least-squares slope** of the raw per-round pass rate over the last W = 10 such rounds (x = 0 … k−1):
  1. Stage 1: WLS with weights = prompt counts. The residual variance is computed with denominator `k − 2` (degrees of freedom).
  2. Stage 2: refit with weights `1 / (se_group² + residual variance)`. The residual term stands in for the autocorrelation between rounds.
  3. `SE(slope) = sqrt(1 / Σ w_i (x_i − x̄)²)`, and `z = slope / SE`.
- With fewer than 3 points it returns (0, 0, 0): no claim of progress without evidence.
- **Why LP-B (D-40).** A pre-registered study on simulator tuning seeds with known true skill velocity required: best rank correlation subject to false-positive rate ≤ 5% and lag ≤ 5.
  - LP-B: ρ 0.19–0.20, FPR 0.000, lag 3.
  - LP-A (fast minus slow EMA): ρ 0.23–0.27, but FPR 0.10, which breaks the constraint.
  - LP-C (difference of two window means): ρ 0.13–0.16.
- `lp_use: signed` (Curator). The LP baseline uses `abs`.

### 4.3 Signal richness (`signals/richness.py`, mode `mixed`)

- Raw value per round: `n_groups_mixed / n_prompts`, the fraction of prompt groups that produce a non-zero GRPO gradient.
- Smoothed: `SR ← λ·SR + (1−λ)·raw` with λ = 0.9. Rounds with no prompts leave SR unchanged.
- With G = 8, the proposal's "pass rate in [0.1, 0.9]" band is exactly `0 < k < 8`, so `band` mode collapses to `mixed` (spec defect S-8).

### 4.4 Measured unit cost (`signals/engine.py`)

- `unit_cost_i ← λ·unit_cost_i + (1−λ)·(cost_usd / n_prompts)`, with λ = 0.9; the first observation initialises it.
- Normalised: `c̃_i = unit_cost_i / mean(unit_cost over environments with a known cost)`.
- An environment with no measured cost yet gets proxy reward 0, so it is treated as expensive until measured.
- In the simulator, cost comes from the world's log-normal per-prompt cost. On the GPU it will come from the cost meter (Phase F).

### 4.5 Proxy reward (`signals/proxy.py`)

```
σ_LP  = discounted (λ=0.9) running std of LP values across ALL environments
LP'_i = clip( LP_i / σ_LP , −3, +3 )                    (0 if σ_LP ≤ 1e-6)
x_i   = α·LP'_i + β·SR_i                                α = β = 0.5
r̃_i   = x_i / c̃_i^θc                                    θc = cost_exponent = 1  (0 = ablation, no cost)
r̄_i   = clip( (r̃_i − q05) / (q95 − q05), 0, 1 )
```

- `q05`/`q95` are the 5th/95th percentiles of the last 200 `r̃` values (all environments pooled). Before 20 values exist, the priors `[0, 1]` are used.
- `x_i` (called `proxy_raw`) is the cost-free learning signal. Calibration uses this value.
- `r̄_i` (called `proxy_reward`) is what the bandit consumes.

### 4.6 Status machine S1–S5 (`signals/status.py`, one classifier per environment)

| Status | Meaning | Entry condition | Exit condition |
|---|---|---|---|
| S1 unexplored | not enough evidence | start state; re-entry only if `n_groups_eff < 32` (½·n_min) and the arm is not an established S3/S4 | `n_groups_eff ≥ 64` (n_min) and `rounds_seen ≥ 3` |
| S2 learning | default | anything not otherwise classified | — |
| S3 saturated | model reliably solves it | `p_lo ≥ 0.70` and `z_LP < 1.5` for 4 consecutive rounds | after 3 rounds dwell: `p̂ < 0.65` or `z_LP ≤ −2` |
| S4 too hard | no learning signal | `p_hi ≤ 0.05` and `SR ≤ 0.10` and `z_LP < 1.5` for 4 consecutive rounds | after 3 rounds dwell: `p̂ ≥ 0.10` or `z_LP ≥ 1.5` or `SR > 0.15` |
| S5 unreliable | proxy does not match benchmark | calibration mismatch flagged in 2 consecutive calibration windows | 2 consecutive clean windows |

- Priority when classifying from scratch: S1 > S5 > S3 > S4 > S2.
- S2 carries a sub-label: `S2a_progressing` if `z_LP > 0`, else `S2b_plateau`.
- **Tuning (D-42).** Thresholds were tuned over a 243-point grid to maximise macro-F1 against truth labels from the simulator, subject to ≤ 5 status flips per 100 rounds. Result: F1 0.97, flip rate 0.37 on tuning seeds; 0.967 on evaluation seeds.
- **D-51.** Evidence decay alone never demotes an established S3/S4 arm back to S1. Otherwise a starved too-hard arm would loop S4 → S1 → exploration quota → S4.

### 4.7 Discounted-UCB counters (`scheduler/ducb.py::update_observation`)

For every environment, every round (including the 6 warm-up rounds):

```
u_i  = weights_used_i                      (pull_unit: round → the compute share)
Ñ_i ← γ·Ñ_i + u_i                          γ = 0.95
S̃_i ← γ·S̃_i + u_i·r̄_i
```

- Effective memory is `1/(1−γ) = 20` rounds.
- The scheduler also accumulates, per calibration window, `Σ share·x_i` (`_win_x`), `Σ share` (`_win_u`) and dollars charged (`_win_usd`). Calibration reads these.

### 4.8 Scores (`compute_scores`)

```
U_i = S̃_i / Ñ_i  +  κ · sqrt( ln(max(Σ_j Ñ_j, e)) / Ñ_i )          κ = 0.5
```

- Rounds 1–6 are a uniform warm-up (`warmup_rounds: 6`). Scores are not used until round 7.

### 4.9 Mixture map (`scheduler/mixture.py::softmax_floor`)

```
z_i = (U_i − mean(U)) / (std(U) + 1e-12)             score_norm: zscore
p   = softmax( clip(z / τ, −50, 50) )                τ = 0.3
w_i = (1 − ε)·p_i + ε/N                              ε = 0.10 → floor 0.02 with N = 5
renormalise; assert Σw = 1 within 1e-12 and every w_i ≥ ε/N
```

- The z-score makes τ independent of the reward scale.
- The side effect is near one-hot allocation: mean max-weight 0.84–0.89 in S-A and S-C (D-67). The simulator does not penalise this.

### 4.10 Status constraints (`apply_status_constraints`, `status_control: hard`)

1. Multipliers: S3, S4 and S5 weights × 0.5.
2. Hard caps: S3 ≤ 2·ε/N = 0.04, S4 ≤ ε/N = 0.02.
3. Every weight is raised to at least the floor ε/N.
4. S1 exploration quota: `w_i ≥ 0.30`, doubled to 0.60 while `n_groups_eff < 32`. If several S1 quotas together exceed the mass the floors leave free, they are scaled down proportionally (D-47).
5. Redistribution until Σw = 1. Excess mass is taken proportionally from arms above their floor or quota. Released mass goes only to **uncapped, non-quota** arms, proportionally to their weight, so a capped arm never absorbs its own release.
6. Renormalise, then re-assert sum = 1 and the floor.

Consequences:
- S5 only **halves** an arm's weight; it is not capped.
- A Countdown arm in S4 is held at exactly 0.02. The measured Countdown share in the refit scenarios is 0.020.

### 4.11 Integer prompt counts (`core/quotas.py`)

- `m_i = floor(M·w_i)`. The leftover prompts go to the largest fractional remainders, with a seeded tie-break, so `Σm_i = M` exactly and deterministically.
- M = R·P: 5 × 16 = 80 per round in the simulator; the planned GPU setting is 2 × 16 = 32.

---

## 5. Calibration (`calibration/calibrator.py`, `calibration/credit.py`)

**Purpose.** The proxy reward is cheap but can be wrong; a random reward produces mixed groups and therefore high richness. Every K = 10 rounds CURATOR measures real held-out gains on a few environments and tests whether each environment's measured gain is consistent with what its proxy claims.

### 5.1 Paired evaluation items

- Each environment has its own calibration slice (the `calib` split): 100 fixed items per slice.
- The same items are re-evaluated every time. The real system will use greedy decoding on a fixed set; the simulator models an item as solved iff its fixed latent `u < p(skill)`, and redraws 2% of latents per evaluation ("churn").
- The SE of a slice's change is the paired McNemar SE, computed from discordant items only and floored at one discordant item:
  ```
  SE(Δ) = sqrt( n01 + n10 − (n01 − n10)² / n ) / n
  ```
- **Why paired (D-72).** With independent re-sampling, the SE of a window's gain was about 0.05 while true gains were about 0.01, so gains were unmeasurable.

### 5.2 Which slices to evaluate (targeting, D-93 … D-95)

- **First calibration (round 10):** every slice is evaluated. This is the paired baseline; no gain exists yet.
- **Afterwards, `claim_stale` with `max_targets = 2`:**
  ```
  key_j = max(Σ_window share_j · x_j , 0) × (1 + windows since slice j was last evaluated)
  evaluate the 2 slices with the highest key (ties broken by env id)
  ```
  - The first factor is the budget-weighted **proxy claim**: where the proxy promises the most is where a wrong proxy wastes the most budget.
  - The staleness factor guarantees that every funded arm is re-checked eventually.
- **Cost:** each evaluated item is charged at (mean training unit cost / G), i.e. one rollout's worth. With 2 × 100 items per window this is 3.3–3.4% of the budget.

### 5.3 Credit: gain per unit compute share (C2, own-slice WLS)

When slice j is evaluated, one observation is formed over the span since j was last evaluated:
- `gain_j = score_now − score_at_last_evaluation`, with its paired SE.
- `W_j` = the compute share j received over that span (sum of window shares).

Per environment, weighted least squares across all such observations:
```
b̂_j    = Σ (W·gain / SE²) / Σ (W² / SE²)               gain on own slice per unit of full compute share
SE(b̂_j) = 1 / sqrt( Σ W² / SE² )
```
- C1 credit (`ΔS × share`, the proposal's original rule) is computed and logged only. It cannot say which environment caused a gain.

### 5.4 Mismatch test → S5

```
x̄_j     = exposure-weighted mean of the proxy signal x_j over the run
s       = max( median_j ( b̂_j / x̄_j ), 0 )      over arms with evidence and x̄_j > 1e-9
implied_j = s · x̄_j                              proxy-implied gain
z_j     = ( b̂_j − implied_j ) / SE(b̂_j)
flag_j  = z_j < −2.0                              only once ≥ k_min = 2 gain-producing windows exist
```

- **Median, not mean:** a single junk arm with zero gain cannot drag the scale to zero and hide itself.
- **One-sided:** an arm that gains more than its proxy claims is never flagged.
- Flags go to `SignalEngine.set_calibration_mismatch`. The status machine enters S5 after 2 consecutive flags and leaves after 2 clean windows.
- **Timeline in a 70-round run:** calibrations at rounds 10, 20, …, 70. Round 10 is the baseline, round 20 gives the first gains, and decisions are possible from round 30. The earliest S5 entry is therefore after the round-40 calibration. The diagnosis measured about 6.3 calibrations and 4.3 decision-ready windows per episode (`reports/analysis/refit_diagnosis.md`).

---

## 6. The simulator (`simulator/world.py`, `scenarios.py`, `oracle.py`)

**Purpose.**
- To evaluate schedulers where the optimum is known.
- To run thousands of episodes on a CPU before spending GPU-hours.
- To force the scheduler to use only measured quantities: true skill exists only inside the world.

### 6.1 World dynamics

```
true pass rate      p_i = sigmoid( a_i · (s_i − d_i) )                   a = slope, d = difficulty
learnability        g_i = 4·p_i·(1 − p_i)                                1 at p = 0.5, 0 at p ∈ {0,1}
skill update        s_j ← s_j + R·η_j·gate_j·( w_j·g_j + Σ_i T_ji·w_i·g_i ) + N(0, 0.003²)
```

- `T_ji` is the transfer from training on i to skill on j. Negative values model interference.
- `gate_j` delays learning until a prerequisite skill passes a threshold.
- Learning per round depends on the **compute share** `w`, not on dollars. A more expensive arm gives the same learning per round at the same share but uses the budget faster, so the episode has fewer rounds. That is exactly the cost-versus-value trade-off.

### 6.2 Observations the world emits

- **Rollouts:** each prompt's own pass probability `q ~ Beta(κp, κ(1−p))`, then `k ~ Binomial(G, q)`.
  - κ (`rollout_concentration`) controls polarisation: small κ means most groups are all-right or all-wrong.
  - Placeholder scenarios used κ = 25; the value fitted to Kaggle data is **κ = 2.14** (§8.3).
- **Noisy arm:** `k ~ Binomial(G, 0.35)` regardless of skill. Its skill never changes, it transfers nothing, and its benchmark weight is 0.
- **Cost:** per prompt ~ LogNormal with mean = the arm's unit cost and σ = 0.10.
- **Benchmark:** `S = Σ_d π_d · sigmoid(a_d (s_d − d_d))`, observed with Binomial noise over 200 items per domain.

### 6.3 Oracles (reference policies that may read hidden state)

- **O1 DP:** exact dynamic programming over a discretised skill × budget grid (N ≤ 3).
- **O2 static:** best fixed mixture, by simplex grid (N ≤ 4) or Dirichlet random search with hill-climbing (N > 4).
- **O3 myopic:** weight proportional to the true marginal benchmark gain per dollar.
- **Metric:** oracle-gap closure = `(S_method − S_uniform) / (S_oracle − S_uniform)`. In Phase 3 the oracles beat Uniform by 11–159%, which proved the scenarios can discriminate between policies.

### 6.4 Scenarios (`configs/sim/`)

| Scenario | Purpose |
|---|---|
| S-A | easy/saturating + valuable-slow + noisy: does the allocator drop the noisy and saturated arms? |
| S-B | 8-arm portfolio with noisy, too-hard, gated and interfering arms |
| S-C | 4 identical arms, costs 1×, 2×, 4×, 8×: pure test of cost normalisation |
| S-D | positive transfer from a helper arm, negative transfer from an interferer |
| S-E | abrupt difficulty shift at round 12: tests discounting |
| S-F | too-hard arm unlocked by transfer: tests starvation and the exploration floor |
| S-G | noisy benchmark slices: base for calibration size/interval sweeps |
| S-H | valuable arm arrives at round 8: cold start |
| S-I / S-J | placeholder analogues of the real portfolio (≈ 70 rounds); S-J has equal costs |
| **S-I-R1, S-I-R2, S-J-R** | **refit from Kaggle measurements** (D-88), the most realistic scenarios |

**Refit scenarios (`experiments/analysis/refit_scenarios.py`):**
- Measured inputs: difficulty from pass@1; κ = 2.14 fitted to measured mixed-group shares; unit costs in GPU-hours from generation + pooled verification + the policy update.
- The policy update's share of cost is unmeasured, so it is bracketed:
  - R1: update fully token-proportional, learnable-arm cost spread 3.46×;
  - R2: half fixed, spread 2.06×;
  - S-J-R: R1 dynamics with every arm at R1's mean cost.
- Benchmark: macro-average with weight 0.25 each for gsm8k, math35, mbpp and countdown; noisy weight 0.
- **Assumed, not measured:** learning speeds η, and the 0.3 GSM8K ↔ MATH35 transfer.

---

## 7. Baselines (`scheduler/baselines/`)

Every method uses the same 6-round uniform warm-up, the same 0.10 exploration floor, and the same matched budget.

| Method | Exact rule | Tuned values |
|---|---|---|
| Uniform | `w_i = 1/N` | — |
| Static | fixed size-proportional weights from declared dataset sizes (D-63). In the refit: gsm8k 0.286, math35 0.207, mbpp 0.013, countdown 0.208, noisy 0.286 | — |
| LP curriculum (Graves/TSCL style) | softmax over `abs(LP)` from the same signal engine; cost-blind, no statuses, no calibration | τ 0.75 |
| Standard UCB | Curator's own proxy and softmax map with γ = 1, `cost_exponent = 0`, statuses off, calibration off | κ 0.5, τ 0.3 |
| SEC-style (Chen et al.) | `Q_i ← α·r + (1−α)·Q_i` for arms that got prompts, with `r` = mean \|A\| per prompt; `w = softmax(Q/τ)` + floor | α 0.1, τ 0.4 |
| DUMP-style (Wang et al.) | UCB1 on mean \|A\| with undiscounted prompt counts; softmax + floor | c 2.0, τ 0.4 |

- SEC and DUMP are re-implementations from the papers' descriptions ("style"), not the authors' code.
- **Mean |A| from pass counts alone (`core/advantage.py`).** For a group with k of G successes, `p = k/G`, unbiased `std = sqrt(G/(G−1)·p(1−p))`:
  ```
  mean|A| = 2p(1−p) / (std + 1e-4)          (0 for k = 0 or k = G)
  ```
  No trainer internals are needed, which is why D-54 promoted these baselines to the main comparison.
- **Fair tuning (`experiments/sweeps/tune_all.py`, D-55/D-68):**
  - Each adaptive method gets exactly 8 configurations, evaluated on tuning seeds and scenarios.
  - Objective: mean relative improvement over Uniform on the same seed.
  - Selection: the best configuration and every configuration within one standard error of it are candidates; the least concentrated candidate wins (highest mean allocation entropy).
  - S-J is never used for tuning, so it stays out-of-sample.

---

## 8. Environments and data (`src/curator_rl/envs/`)

### 8.1 The five arms (frozen Level-1 portfolio, D-87)

| id | Data and splits | Verifier, exactly | Role |
|---|---|---|---|
| `gsm8k` | openai/gsm8k; train pool hash-split into calib 300 / dev 300 / rest train (6,873); official test sealed | last `\boxed{…}`, numeric normalisation (Decimal/Fraction), equality with the gold number | learnable, cheap |
| `math35` | MATH levels 3–5: 4,982 train / 300 calib / 300 dev; sealed test = MATH-500 | last *balanced* `\boxed{…}` (handles nested braces); conservative normaliser that can reject an unusual correct spelling but never accepts two different values (no false positives); no optional libraries, so rewards are identical on every machine | learnable, **expensive** (1,024-token cap) |
| `mbpp` | google-research mbpp: official train 374 → 314 train + 60 dev; official validation 90 = calib; official test 500 sealed | last fenced code block is run in the sandbox against **all** assertions, though the prompt shows only one; pass iff a random sentinel prints after the last assertion | learnable, cheap to generate, verifier-bound |
| `countdown` | procedural: 4 numbers in 1–99, target 10–999, all numbers used; disjoint seed ranges per split | AST evaluator with a node whitelist (never `eval`), exact Fraction arithmetic, bounded expression size | zero-signal "too hard" arm (0 of 1,024 samples correct on 0.5B) |
| `noisy` | GSM8K train prompts | honest GSM8K verification is recorded for diagnosis only; reward ~ Bernoulli(0.35) drawn from `sha256(salt \| prompt_id \| response)`, so it is reproducible yet independent of correctness | spurious-reward probe |

### 8.2 Leakage and integrity controls

- **Hash splits (`envs/splits.py`):**
  - `u = int(sha256(salt + dataset + problem_id)[:8], 16) / 2³²`, with salt `curator-gate0-2026-10-d20`.
  - Items are sorted by `u`; the first 300 go to calib, the next 300 to dev, and the rest to train.
  - Manifests of ids are committed with SHA-256 checksums (`data/manifests/MANIFEST.sha256`) and rebuild byte-identically.
- **Sealed test guard (`evaluation/guard.py`).** The test set opens only if all of these hold:
  1. `CURATOR_FINAL_EVAL=1` is set;
  2. a frozen-config hash matches;
  3. the access is appended to `reports/test_access_log.jsonl`;
  4. a second opening for the same (checkpoint, split) is refused.
  The Kaggle bundle builder refuses to include sealed or raw data.
- **MBPP sandbox (`envs/sandbox.py`):**
  - Runs in a separate isolated interpreter (`python -I`) with a timeout.
  - On POSIX it sets CPU, address-space and file-size limits inside the child, which is thread-safe, unlike `preexec_fn`.
  - Socket, subprocess and fork/exec functions are disabled.
  - Success requires a random 24-hex-character sentinel printed after the last assertion, so `exit(0)` before the tests and forged success messages both fail.
  - It is **not** a security boundary and is used only on disposable machines.
- **Gold/garbage checks:** gold answers verify 100% and garbage 0% on MATH35 (5,582 rows) and MBPP (464 rows).

### 8.3 Real-hardware measurements (Kaggle T4, fp16, Qwen2.5-0.5B-Instruct)

Scripts in `scripts/`, results in `reports/pilot/phase_e_cost_probe.md`. 64 dev prompts × 8 samples per environment, each at its own token cap; verifier run serially and through a 4-thread pool.

| env | cap | pass@1 | pass@8 | mixed groups | gen s/prompt | verify s/prompt (pool) | relative cost |
|---|---|---|---|---|---|---|---|
| gsm8k | 512 | 0.350 | 0.797 | 0.766 | 0.524 | 0.002 | 1.00× |
| math35 | 1024 | 0.168 | 0.500 | 0.500 | 1.403 | 0.002 | 2.67× |
| mbpp | 512 | 0.194 | 0.483 | 0.467 | 0.317 | 0.328 | 1.23× |
| countdown | 512 | 0.000 | 0.000 | 0.000 | 0.535 | 0.001 | 1.02× |
| noisy | 512 | 0.373 | 1.000 | 1.000 | 0.526 | 0.002 | 1.00× |

Findings:
1. **vLLM is the training backend.** One GRPO step (P = 16, G = 8, 512 tokens, LoRA) takes 42.2 s with vLLM colocate vs 83.5 s with HF generate. In the vLLM step, the policy update and weight sync take about 34 of the 42 s.
2. **HF generate hides cost differences.** It runs each batch until its longest completion finishes, which compressed the measured spread to 1.45×. vLLM schedules sequences continuously, giving the real 2.67×.
3. **MATH35 is the expensive arm, not MBPP.** The roadmap expected MBPP. MBPP generates short answers but its verifier takes 1.3 s per prompt serially (0.33 s with 4 threads), so the reward wrapper must verify in parallel.
4. **Sampling bug (D-84).** Qwen's `generation_config` sets `top_k = 20` and `repetition_penalty = 1.05`. With those, GSM8K pass@1 measured 0.20; with both disabled (as TRL samples) it is 0.35.
5. **Polarisation κ = 2.14.** A Beta-binomial fit to the measured mixed-group shares (per arm 1.5–2.6) replaced the assumed 25: real prompt groups are far more often all-right or all-wrong.
6. **Portfolio acceptance rule passes (§4.1 of Roadmap v3):**
   - ≥ 2 arms with pass@8 in [0.15, 0.85]: three qualify;
   - a too-hard arm: Countdown;
   - cost spread ≥ 2×: 2.67×.
7. **Run cost:** 1.4–2.8 GPU-hours per 60-round run at R = 2, depending on how much MATH35 a method buys.
8. **TRL smoke test passes.** The reward function receives `completion_ids` and every dataset column (`env_id`, `prompt_id`, `reference`, `meta_json`), and the G completions of a prompt arrive contiguously.

---

## 9. Experimental protocol (`docs/EXPERIMENT_PROTOCOL.md`)

- **Pre-registration.** Estimators, thresholds and pass/fail criteria are written in `DECISIONS.md` *before* the run they govern, and never changed after seeing the outcome. A failed test is reported; a new idea gets a new pre-registration on fresh seeds.
- **Seeds:**
  - Tuning seeds 0–49.
  - Each evaluation range is used exactly once: 100–149 (Gate 2/2''), 200–249 (CRE), 300–349 (H-κ), 400–449 (refit), 500–549 (γ = 1), 600–699 (claim targeting).
- **Statistics:**
  - Paired differences per seed, 95% percentile bootstrap with 10,000 draws, seeds as the resampling unit.
  - Holm correction on the co-primary H1/H2.
  - Equivalence is claimed only inside a registered margin (δ_S = 1 benchmark point); non-significance is never treated as equivalence.
- **Matched budget.** Every method stops at the same B. Calibration and overhead are charged only to methods that use them; reporting evaluations are uncharged and identical for all methods.

---

## 10. Results, in the order they happened

### 10.1 Gate 2: simulator recovery (seeds 100–149)

| Criterion | Result | Threshold |
|---|---|---|
| (a) S-A, Curator − Uniform | +0.0618 [0.0506, 0.0730] | CI excludes 0 |
| (b) S-B, Curator − Uniform | +0.0110 [0.0069, 0.0149] | lower bound > −0.01 |
| (c) too-hard share, last 40% of rounds | 50/50 seeds ≤ 1.5 × floor | ≥ 90% of seeds |
| (d) status macro-F1 / flips per 100 rounds | 0.932 / 0.0 | ≥ 0.85 / ≤ 5 |
| (e) S-C compute to reach target vs Uniform | +26.2% [22.2%, 30.0%] | ≥ 15% |

- Oracle-gap closure in S-A was 43% against a 75% development target.
- D-65 shows the structural ceiling: playing Uniform for 6 warm-up rounds of a 25-round episode and the oracle afterwards closes at most 75.9%.

### 10.2 Gate 2'': against SEC/DUMP-style (seeds 100–149)

- **S-C (cost independent of value):** Curator beats SEC-style by +0.061 and DUMP-style by +0.059. Cost normalisation is a large lever when cost and value are unrelated.
- **S-I (placeholder; valuable arms are expensive):** Curator loses by −0.008 to both. **Fail.**
- **Diagnosis (D-69):**
  - The noisy arm took 20–25% of Curator's compute.
  - Cost normalisation undervalued the expensive but valuable arms.
  - A privileged (perfect) S5 flag raised Curator only to the SEC/DUMP level, not to Standard UCB.

### 10.3 Calibration power study (D-74, D-75)

- Evaluating every slice at a realistic cost detected the noisy arm in ≤ 10% of seeds within a 10% cost cap.
- Targeting the most-funded slices was 3–8× more powerful at equal cost.
- The rule's choice (top-1 by exposure, 100 items, K = 10) detected 33% (S-I) and 10% (S-J) at 2.4% of the budget.

### 10.4 Calibrated Reward Engine: two pre-registered failures (D-76 … D-82)

- **Design:**
  - Each arm's reward was replaced by a Gaussian posterior over `b_j` = own-slice gain per dollar charged to j.
  - Likelihood: `gain = b_j·usd + noise(paired SE)`.
  - Prior: `N(s·x_j/c_j, (ρ·|m0|)²)`.
  - Reward: `clip(m_j·B/R_max, 0, 1)`.
  - Constants were frozen from tuning seeds: s = 0.0001963, ρ = 1.232, R_max = 1.5105.
- **Result 1 (seeds 200–249):** fails on S-I (−0.008 vs SEC and DUMP), and is statistically indistinguishable from targeted Curator.
- **Why:** the reward channel had little leverage.
  - The cross-arm spread of the exploration bonus (0.33) was twice that of the mean reward (0.17).
  - The new rewards correlated 0.79 with the old ones.
  - The weights moved by a total-variation distance of only 0.024.
- **Result 2, H-κ (seeds 300–349):** re-tuning κ and τ for the CRE (κ 0.2, τ 1.0) still failed (−0.010 / −0.012), with a CRE-minus-control difference of +0.001. The line was stopped.

### 10.5 Refit from real measurements: pre-registered (D-88, D-89; seeds 400–449)

| Scenario | Uniform | Static | LP | Std UCB | SEC | DUMP | Curator | Curator, no calib |
|---|---|---|---|---|---|---|---|---|
| S-I-R1 (3.46×) | 0.291 | 0.290 | 0.330 | 0.330 | 0.292 | 0.292 | 0.319 | 0.323 |
| S-I-R2 (2.06×) | 0.290 | 0.290 | 0.331 | 0.335 | 0.292 | 0.291 | 0.320 | 0.316 |
| S-J-R (equal) | 0.290 | 0.297 | 0.336 | 0.335 | 0.290 | 0.294 | 0.323 | 0.326 |

- **Primary P1 passes:** Curator beats SEC-style and DUMP-style by +0.027 to +0.029 in both cost brackets, all CIs > 0.
- **Primary P2 passes:** in S-J-R, Curator − Uniform = +0.033.
- **Why SEC/DUMP lose.** They put 35–36% of the budget on the noisy arm. With G = 8 and reward Bernoulli(0.35), the chance a group is all-equal is `0.65⁸ + 0.35⁸ ≈ 0.032`, so about 97% of noisy groups are mixed and mean |A| is near its maximum. With κ = 2.14, the real arms have only 47–77% mixed groups. A mean-|advantage| signal therefore ranks the spurious arm first.
- **Curator is third.** LP and Standard UCB beat it by 0.011–0.014, CIs excluding 0.
- **Its gain is not cost-driven:** the margin over Uniform is the same with equal costs. It comes from starving Countdown (2% vs 20%) and spreading budget over the learnable arms.
- **Calibration has no measurable effect yet:** noisy arm in S5 in only 20–34% of seeds.

### 10.6 Factorial diagnosis (D-90; tuning seeds 0–49)

A 2⁵ design switched discounting, cost normalisation, status layer, calibration and the richness term on and off. The all-off cell reproduces Standard UCB exactly (0.3322 = 0.3322).

- **Discounting is the main loss:** main effect of γ = 0.95 vs γ = 1 is −0.020 (R1), −0.014 (R2), −0.013 (S-J-R). With polarised groups each round's signal is noisy, and a 20-round memory discards information that the slowly drifting world does not make stale.
- **Cost normalisation pays only without discounting:** +0.016 at 3.46× spread, 0 at equal cost.
- **The proxy is partly fooled.** Learning progress is barely detectable in 70 rounds (mean LP z 0.06–0.15 on learnable arms), so the proxy runs mostly on richness. The noisy arm has the highest richness (SR 0.906); its proxy `x` (0.466) exceeds MATH35's (0.438).
- **Calibration rarely looked at the noisy arm:** top-1 exposure targeting usually picked GSM8K.

### 10.7 γ = 1 test (D-91, D-92; seeds 500–549): not adopted

- γ = 1 improved Curator by +0.017 to +0.034, beat SEC/DUMP by +0.048 to +0.063, and beat Standard UCB in S-I-R1 by +0.016.
- The pre-registered no-regression check on the original scenarios failed narrowly: S-A −0.003 [−0.012, +0.006] and S-D −0.004 [−0.012, +0.003] against a −0.01 margin. 50 seeds were too few to resolve that margin, which is recorded as a design flaw in the test.
- The rule was applied as written: **γ stays 0.95**.

### 10.8 Claim-plus-staleness targeting (D-93 … D-95; 100 fresh seeds 600–699): adopted

| Scenario | Noisy arm caught (S5), before → after | False-flag rate | Calibration cost | Score vs before |
|---|---|---|---|---|
| S-I-R1 | 0.25 → 0.86 | 0.020 | 3.4% | +0.006 [+0.002, +0.010] |
| S-I-R2 | 0.32 → 0.81 | 0.025 | 3.3% | +0.004 [−0.000, +0.007] |
| S-J-R | 0.31 → 0.78 | 0.022 | 3.3% | +0.006 [+0.002, +0.010] |

- All four pre-registered criteria passed: detection, safety, non-inferiority, and Gate 2'' retained.
- The noisy arm's budget share fell only from 0.23 to 0.19, because S5 halves weight rather than removing it.
- Curator is still level with or below Standard UCB (−0.005 to 0) and LP (−0.011 to −0.003).

### 10.9 Hypotheses (Roadmap v3 §3.3) and where they stand (simulator only)

| ID | Hypothesis | Status |
|---|---|---|
| H1 | Curator > Uniform, heterogeneous costs | supported in sim (+0.028 to +0.030); GPU test pending |
| H2 | Curator > SEC/DUMP-style | supported in refit sim (+0.027 to +0.033); GPU test pending |
| H3 | cost normalisation drives H1/H2 | **not supported** on refit scenarios (same gain at equal cost); strongly supported in S-C |
| H4 | calibration detects the noisy arm and cuts its share | detection supported (78–86%); share cut is small (0.23 → 0.19) |
| H5 | ROI ranking agrees with leave-one-out | not built |
| H6 | equal costs: Curator ≈ Uniform | Curator is *above* Uniform (+0.033); the non-inferiority check passes |
| H7 | discounting helps under drift | evidence against: γ = 1 shows no significant loss even in drift scenario S-E; formally open |

---

## 11. What is new, precisely

| Method | Arm signal | Cost-aware | Reward sanity | Held-out calibration | Validated attribution |
|---|---|---|---|---|---|
| Graves 2017 / TSCL 2020 | learning progress | no | no | no | no |
| DUMP 2025 | mean \|advantage\|, UCB | no | no | no | no |
| SEC 2025/26 | mean \|advantage\|, TD(0) | no | no | no | no |
| TAC 2026, HarnessBandit 2026 | learnability + gradient transfer | no | no | no | no |
| **CURATOR** | LP + richness ÷ measured cost, D-UCB | yes | **reward audit** (prompt dependence, zero cost) | yes (paired items, transfer-aware credit, S5) | ROI leaderboard vs leave-one-out (built; H5) |

Concrete contributions with evidence:
1. **Measured-cost allocation.** Arms are whole environments, rewards are divided by measured GPU cost, and spend runs through a matched-budget ledger that charges calibration only to the method using it.
2. **A failure mode of learnability bandits, demonstrated.** Mean-|advantage| allocators are drawn to a random-reward environment, because random rewards always produce mixed groups. It appears only with measured group polarisation (κ = 2.14), and it is why SEC/DUMP-style lose in the refit.
3. **A zero-cost reward audit that removes that failure mode.** A verifier's reward depends on the prompt (prompts differ in difficulty), so group pass rates are over-dispersed: D = (G + κ)/(κ + 1). A reward that ignores the prompt has D = 1. On the real pilot every verifier arm has D = 3.0–4.4 and the random-reward arm 1.0–1.09. Curator flags D's upper confidence bound below 1.5 and caps that arm at the exploration floor. At the GPU shape (tuning seeds) the random arm is caught in 90–100% of runs by round ~14–21; its budget share falls from about 0.3 to 0.06; Curator's gain over Uniform rises from +4.5% to +11–12%, about twice the best baseline (LP, +6%). No real arm was ever flagged by the audit.
4. **Two layers, honestly scoped.** The audit catches rewards that ignore the prompt; benchmark calibration is the layer for rewards that depend on the prompt but reward the wrong thing. At the Level-1 budget (1,920 prompts) calibration detects such an arm only ~10% of the time, so the claim about it is restricted to measurement, not detection.
5. **Protocol.** A simulator with known optima and realism switches (forgetting, finite datasets, κ-consistent learning), pre-registered gates on fresh seed ranges, equal tuning budgets, bit-exact reproduction of older results, and negative results reported (CRE ×2, γ = 1, α/β fit, drift discounting, longer LP window: no gain).

DataFlex-RL (Sept 2026) found that no published selection or mixing policy beat uniform sampling with a significant paired interval. CURATOR's claim is therefore deliberately narrow: an advantage under a fixed budget with heterogeneous costs and a spurious arm, plus detection and attribution. It is not "beats uniform everywhere", and the GPU runs have not happened yet.

---

## 12. Limitations

1. **No real-GPU training under CURATOR yet.** The Phase F runner drives a real TRL GRPO trainer and is verified on CPU; every scheduler result is simulated.
2. **Assumed simulator parameters:** learning speeds η, the GSM8K ↔ MATH35 transfer, the value of repeated passes (repeat_decay 0.5) and forgetting (0.002 per untrained round). The sysid run is designed to measure the first two on hardware.
3. **Unmeasured update cost.** Bracketed (R1/R2) until the cost meter runs on GPU (Gate 5).
4. **All advantages are on tuning seeds.** The confirmatory run (D-99, fresh seeds 700–799) needs the owner's sign-off.
5. **Audit blind spot:** a reward that depends on the prompt but is wrong (a wrong answer key) passes the audit; at Level-1 budget calibration rarely catches it either.
6. **Audit assumption:** real arms differ in prompt difficulty (κ ≲ 5). An arm with nearly uniform difficulty (κ = 10) would be flagged in up to 3% of runs; at the old placeholder κ = 25, in about half.
7. **ROI from one run is weak:** pooled over seeds the in-run ROI ranks arms like leave-one-out (τ 0.6–1.0), but one run gives a median τ of only 0.4–0.6.
8. **Spurious rewards are model-specific.** The noisy-arm finding needs a second model family (Llama-3.2-1B planned).
9. **Compute:** Level 1 needs about 125 GPU-hours; the Kaggle plan is staged in about 27 GPU-hour steps.

---

## 13. What remains

| Step | Work | Gate |
|---|---|---|
| GPU, Level 1 | `kaggle/run.py`: smoke → compare (Curator, Uniform, LP; seed 0) → sysid (P100) → ablations (`curator_nocal`, `pilot_uniform`) → seeds 1–2 → leave-one-out | 3, 4, 5 on hardware; first real comparison |
| Simulator | refit η, transfer, κ, costs from the sysid run; run the pre-registered D-99 confirmation on seeds 700–799 | 6, then freeze (10) |
| Main matrix | all methods × 5 paired seeds at matched budget; equal-cost regime; second model family | 8 |
| Attribution | ROI vs leave-one-out on several seeds (H5); one sealed-test evaluation per final checkpoint; paper | 7, 9 |
| Level 2 | tool-call arm with a ≥ 5× cost spread (B11) | after Level 1 |

---

## 14. Likely viva questions, with exact answers

**Q1. What does CURATOR control, and what does it not touch?**
It controls only the per-round prompt mixture across environments, i.e. the integer quota of prompts drawn from each environment. It does not change the GRPO loss, the optimiser, sampling, LoRA or the verifiers.

**Q2. Why is the objective final score and not gain per dollar?**
A ratio objective can be maximised by spending little. The question is what to do with a fixed budget. Gain per dollar is the online signal (the knapsack rule), and it is reported as a metric (SPEC §2).

**Q3. What is the difference between γ and the MDP discount?**
The MDP discount is 1, because only the final score matters. γ = 0.95 is a bandit forgetting factor applied to the counters `Ñ, S̃`, so old evidence about a drifting environment fades (memory ≈ 20 rounds).

**Q4. Why does richness matter for GRPO specifically?**
A group whose G rewards are all equal has std 0, so every advantage is 0 and the prompt gives no gradient. Richness is the fraction of groups with 0 < k < G, i.e. the fraction of compute that produces gradient.

**Q5. Why divide by measured cost and not by a prior?**
Cost priors were wrong in this project: MBPP was expected to be the expensive arm and MATH35 is (2.67×). HF-generate timing also understated the spread (1.45× vs 2.67×). The scheduler uses an EMA of measured cost per prompt; priors are cold-start only and never treated as measurements (Roadmap H.5).

**Q6. Why z-score the UCB scores before the softmax?**
It makes τ independent of the reward scale. The cost is near one-hot mixtures, which the project measures as a standing diagnostic (D-67).

**Q7. Why an exploration floor?**
Every arm must keep producing observations so that drift, revival (S-F) and saturation can be detected. The floor is ε/N = 0.02 with 5 arms.

**Q8. Exactly how does an environment become S5?**
1. Every 10 rounds, two targeted slices are re-evaluated on fixed paired items.
2. Gains are regressed on compute share across windows (WLS) to give `b̂_j` with an SE.
3. The proxy-implied gain is `median(b̂/x̄)·x̄_j`.
4. If `(b̂_j − implied_j)/SE < −2` in two consecutive decision-ready windows, the environment enters S5 and its weight is halved.

**Q9. Why the median for the scale?**
With a mean, a junk arm with zero gain and positive proxy pulls the scale toward 0, which lowers every implied gain and hides the junk arm. The median is robust to one such arm.

**Q10. Why paired calibration items?**
With independent re-sampling, the SE of a slice's change (~0.05) was five times the true gain (~0.01). Re-evaluating fixed items leaves variance only from items that flipped (McNemar SE).

**Q11. Why do SEC/DUMP-style lose in the realistic simulator?**
Their signal is mean |advantage|, which is maximal for mixed groups. A Bernoulli(0.35) reward produces mixed groups 97% of the time, while real arms at κ = 2.14 produce 47–77%. They give the noisy arm 35–36% of the budget.

**Q12. Then why is CURATOR not the best method?**
On the refit scenarios, LP and Standard UCB beat it by 0.011–0.014. The factorial diagnosis attributes most of this to discounting: γ = 0.95 throws away evidence that polarised, noisy signals need. Removing it would put CURATOR ahead of UCB in S-I-R1 (+0.016), but the pre-registered non-inferiority check failed narrowly, so it was not adopted.

**Q13. Is cost-awareness doing anything?**
In S-C, where cost is unrelated to value, yes: +0.06 over SEC/DUMP-style and +26% compute-to-target over Uniform. In the refit scenarios it adds nothing while γ = 0.95, and +0.016 at 3.46× spread when γ = 1. The paper must state that the gain there comes from other components.

**Q14. Why did the Calibrated Reward Engine fail?**
Replacing the reward did not move the mixture. The exploration bonus varied twice as much across arms as the reward did, so even a better reward changed the weights by a total-variation distance of 0.024. Re-tuning κ/τ for it also failed. Both were pre-registered, so it was dropped.

**Q15. How do you prevent overfitting the evaluation?**
- Tuning happens on seeds 0–49 only.
- Each evaluation seed range is used once.
- Criteria are written before the run.
- The real sealed test sets open once, through a guard that requires an environment flag and a frozen config hash, logs the access, and refuses repeats.

**Q16. How is the comparison fair to baselines?**
Equal tuning budget (8 configurations each), the same selection rule (one-SE, then least concentrated), the same warm-up and floor, the same matched budget, and calibration charged only to the method that uses it.

**Q17. Why is the noisy arm's benchmark weight 0?**
It has no test set: its prompts are GSM8K, but its reward is random. Any budget spent on it should be counted as waste. In real training a random reward may still move a Qwen model (the Spurious Rewards result), which is why the probe must be repeated on a second model family.

**Q18. How is the MBPP verifier protected against reward hacking?**
- All assertions run, not just the one shown in the prompt.
- Success needs a random per-run sentinel printed after the last assertion, so early `exit()` or a printed fake marker fails.
- Network and process spawning are disabled, with CPU, memory and file-size limits on POSIX.

**Q19. What did the Kaggle measurements change in the design?**
- The backend: vLLM, 2× faster.
- Which arm is expensive: MATH35.
- The need for a parallel verifier: MBPP.
- A sampling-config bug: Qwen defaults lowered GSM8K pass@1 from 0.35 to 0.20.
- Group polarisation: κ 25 → 2.14, which reversed the S-I result against SEC/DUMP.

**Q20. What exactly is left before the claims are real?**
Phase F (trainer adapter, reward wrapper, cost meter, resume), then Gates 3–5, calibration on GPU, the main paired-seed matrix (~125 GPU-h at Level 1), the ROI leaderboard and its leave-one-out validation, and one sealed-test evaluation.

---

## 15. Where each thing lives

| Topic | File |
|---|---|
| All hyperparameters | `curator-rl/configs/base.yaml` |
| Observation types | `src/curator_rl/core/types.py` |
| Integer quotas | `src/curator_rl/core/quotas.py` |
| GRPO mean \|A\| | `src/curator_rl/core/advantage.py` |
| Signal pipeline order | `src/curator_rl/signals/engine.py` |
| Pass rate / LP / richness / proxy / status | `signals/passrate.py`, `progress.py`, `richness.py`, `proxy.py`, `status.py` |
| D-UCB, counters, calibration hooks | `src/curator_rl/scheduler/ducb.py` |
| Softmax, floor, status constraints | `src/curator_rl/scheduler/mixture.py` |
| Baselines | `src/curator_rl/scheduler/baselines/` |
| Calibration targeting, credit, z-test | `calibration/calibrator.py`, `calibration/credit.py` |
| CRE (not adopted) | `calibration/cre.py` |
| Simulator world, harness, oracles | `simulator/world.py`, `harness.py`, `oracle.py` |
| Environments, sandbox, splits | `src/curator_rl/envs/` |
| Sealed-test guard | `src/curator_rl/evaluation/guard.py` |
| Gate scripts | `experiments/gate2_check.py`, `gate2b_check.py`, `refit_gate.py`, `gamma_gate.py`, `claim_gate.py` |
| Gate results | `reports/gate2/*.md` |
| Kaggle probes | `scripts/kaggle_*.py`, `reports/pilot/*.md` |
| Every decision and its reason | `docs/DECISIONS.md` (D-1 … D-95) |
| Gate criteria and states | `docs/GATES.md` |
