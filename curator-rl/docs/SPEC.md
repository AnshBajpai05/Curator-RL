# CURATOR SPEC (v2.0) — frozen specification

> Status: skeleton. This file is the implementation contract. Code follows this doc, not the other way round.
> Every change after `spec-v1` is tagged requires a `DECISIONS.md` entry.

## 1. Scope freeze (from Roadmap v2.0, Part A0)

**MVP boundary (hard stopping line):** Curator scheduler + GRPO training + matched-compute budget ledger + held-out calibration + ROI engine + Tier-1 comparison of four allocation methods (Uniform, Static mixture, Learning-progress, Full Curator) + 3 paired seeds + Tier-1 leave-one-out validation, ending with one sealed untouched test evaluation and the Part S figures/tables.

**MVP research question:** Can adaptive, cost-aware environment allocation improve benchmark gain per unit compute compared with fixed and learning-progress-based allocation under a matched compute budget?

Component scope table (MVP ✓ / Stretch) and phase mapping: see `Roadmap v2.0` Part A0.2 and Part P.

## 2. Objective (resolves defect S-10)

Primary objective is **fixed-budget**: maximise final held-out benchmark score at budget B.

```
max_π  E[ S_val(θ_T) ]
subject to: Σ_t cost(w_t) ≤ B        (accounting rule: Roadmap H.6)
```

Ratio quantities (benchmark gain per GPU-dollar) are reported metrics and
leaderboard values, never the optimisation objective.

## 3. Split rule (resolves defect S-11)

Four splits: `train`, `calib` (drives scheduler, α/β fit, S5 status),
`dev` (threshold/hyperparameter tuning, validation curves), `test` (sealed,
used once, guarded by `evaluation.guard`). Assignment by
`u = int(sha256(salt + dataset + problem_id)[:8], 16) / 2**32` thresholded
into splits after official test items are set aside; salt frozen in
`configs/base.yaml` at Gate 0.

## 4. Budget accounting rule (resolves defect S-13)

Stop at the first optimizer step where cumulative charged cost ≥ B.
Charged: training, scheduler-owned calibration evals, Curator overhead.
Uncharged (identical protocol for every method): reporting evals.
Overshoot is logged and is at most one step.

## 5. Data contracts (Roadmap B.3)

Types live in `curator_rl/core/types.py` once Phase 2 starts:
`Prompt`, `Verdict`, `RolloutGroup`, `EnvRoundObs`, `RoundObservation`,
`CalibrationObservation`, plus Phase 4's `EnvStatus` (S1–S5 str enum) and
`SignalVector` (per-env signals + status; the Signal Engine output).
`MixtureDecision` is deferred to v1 Phase 5 (D-26); `RoiRecord` to the ROI
phase (Part I.6).

## 6. Notation table (Roadmap E.1 — frozen names)

| Math | Config / code name | Meaning |
|---|---|---|
| i, N | `env_id`, `n_envs` | Environment index and count |
| t, max_rounds | `round_t`, `max_rounds` | Current round (R optimizer steps) and its limit |
| R, P, G, M | `steps_per_round`, `prompts_per_step`, `group_size` | Round and batch shape; M = R·P |
| K | `calib.interval_rounds` | Rounds between calibration evaluations |
| p̂_i, LP_i, SR_i | `pass_rate`, `lp`, `richness` | Signals |
| ĉ_i, c̃_i | `unit_cost`, `unit_cost_norm` | Measured dollars per prompt; divided by mean across envs |
| r̃_i, r̄_i | `proxy_raw`, `proxy_unit` | Proxy reward before/after mapping to [0,1] |
| γ | `scheduler.gamma` | Bandit forgetting factor (the MDP discount is 1) |
| κ | `scheduler.exploration_coef` | UCB exploration coefficient (Part 2 called it c) |
| τ, ε | `scheduler.tau`, `scheduler.epsilon` | Softmax temperature; exploration floor ε/N |
| α, β | `proxy.alpha`, `proxy.beta` | Proxy weights, initialised from a prior then fitted |

The MDP discount is 1; γ is documented as a bandit forgetting factor.

## 7. Required scientific hyperparameters (no Python defaults)

Every key below MUST exist in `configs/base.yaml`; a missing key fails
`load_config` validation (Rule 5). `tests/unit/test_config.py::test_spec_lint_keys_exist`
enforces the correspondence between this list and `base.yaml`.

<!-- BEGIN_HYPERPARAM_LIST -->
scheduler.gamma
scheduler.exploration_coef
scheduler.tau
scheduler.epsilon
scheduler.warmup_rounds
scheduler.score_norm
scheduler.cost_exponent
scheduler.status_control
scheduler.pull_unit
scheduler.s3_multiplier
scheduler.s4_multiplier
scheduler.s5_shrink
scheduler.s5_cap
scheduler.s1_quota
baselines.ucb.exploration_coef
baselines.ucb.tau
baselines.lp.tau
baselines.sec.alpha
baselines.sec.tau
baselines.dump.exploration_coef
baselines.dump.tau
baselines.dump.window_rounds
signals.lambda
signals.lambda_fast
signals.lambda_slow
signals.lp_method
signals.lp_se
signals.lp_use
signals.window_rounds
signals.richness.mode
signals.richness.band_lo
signals.richness.band_hi
signals.richness.variance_min
signals.prior.alpha0
signals.prior.beta0
signals.status.n_min
signals.status.r_min
signals.status.p_sat
signals.status.p_hard
signals.status.sr_hard
signals.status.z_up
signals.status.z_neg
signals.status.hysteresis_p
signals.status.hysteresis_sr
signals.status.dwell_min
signals.status.consecutive_rounds
signals.status.s1_entry_ratio
proxy.alpha
proxy.beta
proxy.clip_l
proxy.quantile_prior_lo
proxy.quantile_prior_hi
cre.enabled
cre.mode
cre.proxy_scale
cre.prior_rel_sd
cre.roi_scale
cre.discount
calib.enabled
calib.interval_rounds
calib.k_min
calib.trust_region
calib.z_mis
calib.mismatch_windows
calib.mismatch_clear_windows
calib.flag_scope
calib.credit
calib.targeting
calib.max_targets
calib.items_per_slice
steps_per_round
prompts_per_step
group_size
data.split_salt
data.calib_size
data.dev_size
data.envs.gsm8k.max_prompt_tokens
data.envs.gsm8k.max_completion_tokens
data.envs.gsm8k.verifier_timeout_s
data.envs.gsm8k.prior_usd_per_prompt
data.envs.countdown.n_numbers
data.envs.countdown.numbers_min
data.envs.countdown.numbers_max
data.envs.countdown.target_min
data.envs.countdown.target_max
data.envs.countdown.use_all_numbers
data.envs.countdown.exact_division
data.envs.countdown.max_prompt_tokens
data.envs.countdown.max_completion_tokens
data.envs.countdown.verifier_timeout_s
data.envs.countdown.prior_usd_per_prompt
data.envs.math35.level_min
data.envs.math35.level_max
data.envs.math35.max_prompt_tokens
data.envs.math35.max_completion_tokens
data.envs.math35.verifier_timeout_s
data.envs.math35.prior_usd_per_prompt
data.envs.mbpp.memory_limit_mb
data.envs.mbpp.dev_size
data.envs.mbpp.max_prompt_tokens
data.envs.mbpp.max_completion_tokens
data.envs.mbpp.verifier_timeout_s
data.envs.mbpp.prior_usd_per_prompt
data.envs.noisy.mode
data.envs.noisy.q
data.envs.noisy.flip_p
data.envs.noisy.max_prompt_tokens
data.envs.noisy.max_completion_tokens
data.envs.noisy.verifier_timeout_s
data.envs.noisy.prior_usd_per_prompt
<!-- END_HYPERPARAM_LIST -->

## 8. Environment interface contract (Roadmap D.1 — frozen in Phase 2)

| Method | Contract |
|---|---|
| `generate_prompt(split, rng) -> Prompt` | Deterministic given (split, rng state); never returns another split's item |
| `generate_batch(split, n, rng, exclude=None) -> list[Prompt]` | Without replacement within a pass; fresh instances for procedural envs |
| `evaluate_response(prompt, response) -> Verdict` | Pure, thread-safe, timeout, times itself (`verifier_seconds`) |
| `compute_reward(verdict) -> float` | Reward shaping separate from verification; [0,1] unless spec says otherwise |
| `estimate_cost() -> CostPrior` | Cold-start prior only; never used as a measured cost |
| `metadata() -> dict` | Dataset version, split sizes, manifest hash, difficulty parameters |
| `split_ids(split) -> list[str]` | Used by leakage tests |

## 9. Layering rules (Roadmap B.2)

- L0 `core`: imports nothing from the project.
- L1 pure (`signals`, `scheduler`, `roi`, `calibration.regression/credit`): only `core`, numpy, scipy.
- L2 adapters (`envs`, `cost`, `trainer`, `simulator`): import `core`; only `trainer/trl_adapter.py` imports TRL.
- L3 (`evaluation`, `experiments`): everything except nothing imports `evaluation.final_test` / sealed loader.

Enforced by `tests/architecture/test_import_rules.py`. Scope: phase 1 guards the
torch/transformers/trl/datasets bans and the trl-adapter rule; deeper guards are
added when the packages get content.

## 10. Open decisions

None. Any new decision → `docs/DECISIONS.md`.
