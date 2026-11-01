# DECISIONS LOG

One entry per decision. Teammates append; entries are never deleted.

Format:

```
## D-<number> — <short title>
- Date:
- Owner:
- Decision:
- Reason:
- Affects:
```

---

## D-1 … D-15 — Specification defects S-1 … S-15 (Gate 0)

Status at repo creation: **resolved by the Roadmap v2.0 spec-review table (top of the roadmap document)**. At Gate 0 each teammate confirms their read-through of the affected section; confirmations are recorded here as entries D-1 … D-15 with date + initials. Any *change* to a resolution gets a new entry.

## D-16 — Dashboard wording aligned with v2.0 scope

- Date: (Gate 0)
- Owner: team
- Decision: proposal says "offline experimental dashboard" (figures + HTML leaderboard); live weight/cost monitoring moves to stretch.
- Reason: MVP scope freeze (Roadmap v2.0 A0.5); avoids a proposal/implementation mismatch during evaluation.

## D-17 — verl moved from promised plug-in to planned extension

- Date: (Gate 0)
- Owner: team
- Decision: proposal says "initial implementation through TRL, with verl support planned as an extension".
- Reason: TRL API is still moving and integrations dominate risk budget; verl adapter is Roadmap stretch S-15.

## D-18 — Gate 2 oracle-gap closure is a development target, not a pass threshold

- Date: (Gate 0)
- Owner: team
- Decision: Primary gate criteria are paired improvement over Uniform (S-A), no underperformance (S-B), junk share, status metrics; ≥75% oracle-gap closure is a target. Results in 60–75% with tight intervals and seed-consistent behaviour pass with a documented analysis here.
- Reason: a hard threshold on a noisy stochastic simulator would fail good implementations on noise.

## D-19 — Gate 4 equivalence via pre-registered margins δ, not KS p-values

- Date: (Gate 0)
- Owner: team
- Decision: register equivalence margins (final score |Δ| < δ_S, per-step reward |Δ| < δ_R) before the passthrough runs; KS checks are diagnostics only.
- Reason: "p > 0.05" is absence of evidence, not evidence of equivalence.

## D-20 — Split salt frozen

- Date: 2026-10-02 (Phase 2)
- Owner: P2/P3
- Decision: `data.split_salt = curator-gate0-2026-10-d20` in `configs/base.yaml`. Hash formula exactly as SPEC s3: `sha256(salt + dataset + problem_id)`.
- Reason: the salt must be frozen before any split is built; manifests are committed anyway, but freezing the salt at Gate 0 makes pre-Gate-0 reruns reproducible.

## D-21 — Environment parameters centralized in `configs/base.yaml` (data.envs.*)

- Date: 2026-10-02 (Phase 2)
- Owner: P2
- Decision: per-environment parameters (token limits, verifier timeout, Countdown generator ranges, noisy mode/q/flip_p, cost priors) live in `configs/base.yaml` under `data.envs.<env>` instead of the Roadmap's `configs/env/*.yaml` files. No `configs/env/` files are created.
- Reason: `data.envs.*` is part of the pydantic-validated `RootConfig`, so every environment parameter is inside the config hash and unknown-key/mis-typed validation applies. Separate per-env YAML files would duplicate keys and risk drift between the two sources.
- Affects: Roadmap Phase 2 "Files to create" list (configs/env/*) — deviation is intentional and documented here per the SHOULD-override rule.

## D-22 — Countdown split seed ranges

- Date: 2026-10-02 (Phase 2)
- Owner: P3
- Decision: procedural Countdown instance seeds are drawn from disjoint ranges train [0, 1e7), calib [1e7, 1.5e7), dev [1.5e7, 2e7), test [2e7, 3e7); constants in `envs/countdown.py::SPLIT_SEED_RANGES`, disjointness checked by leakage tests.
- Reason: Roadmap D.2 fixes the boundaries at 1e7/2e7 but leaves the calib/dev boundary open; 1.5e7 keeps ranges equal-sized.

## D-23 — DP oracle (O1) implemented as a vectorised skill-grid DP with conservative cost rounding

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `simulator/oracle.py::DPOracle` is exact backward induction over a per-env skill grid (default 80 levels, sized from the maximum reachable skill) × budget units (default 100), numpy-vectorised. Restrictions enforced at construction: N ≤ 3, no drift, no late-start envs, `cost_skill_slope == 0`. Costs round UP to budget units. Per-round skill gains quantise to whole grid levels (numpy `rint`), which is pessimistic for sub-level gains. `select_mixture` replans from the live world state each round (receding horizon, shared value table).
- Reason: a first recursive float-key DP memoised on (skills, units) exploded combinatorially (> 5 min/episode); a coarse level grid *stalls* (sub-quantum per-round gains never accumulate because states are re-decoded each round). The grid DP is feasible and, with up-rounded costs, provably dominates every fixed mixture under the same discretised model (unit-tested via `fixed_policy_value`). Roadmap A0.2 marks the DP oracle optional; the restrictions keep its state space Markov.
- Affects: Phase 3 item 10 unit test semantics (dominance is stated within the discretised model, not against continuous dynamics).

## D-24 — Static oracle (O2) uses random search + hill-climb for N > 4

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: N ≤ 4 exhausts the simplex grid (step 0.1); N > 4 draws 1500 seeded Dirichlet candidates and hill-climbs 300 pairwise perturbations.
- Reason: the full grid at N = 8 has C(17,7) = 19448 points per scenario-seed; random search + refinement reaches comparable values for a fraction of the work. CMA-ES was rejected as an extra dependency.

## D-25 — Myopic oracle (O3) is a first-order equimarginal rule

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `MyopicOracle` sets w_i proportional to max(0, dS/ds_i · eta_i_eff · g_i / c_i); zero for noisy and not-yet-available envs.
- Reason: it implements the equimarginal rule to first order; second-order and cross-env transfer effects are ignored. Validated as a strong reference (42–74% gap closure across S-A..S-H), not an exact optimum.

## D-26 — `MixtureDecision` deferred to Phase 5; Phase 3 schedulers return plain weight dicts

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `BaseScheduler.select_mixture` returns `dict[str, float]` with a `validate_weights` helper (sum = 1, non-negative, keys = env ids). The `MixtureDecision` type (ucb_scores, intents, statuses…) is added in v1 Phase 5 when D-UCB produces those fields.
- Reason: nothing in Phase 3 consumes the richer type; adding it now would be an unused abstraction.

## D-27 — Random baseline module named `random_baseline.py`, not `random.py`

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `scheduler/baselines/random_baseline.py`.
- Reason: avoids any ambiguity with the stdlib `random` module in tooling that resolves bare module names.

## D-28 — Scenario configuration is a standalone strict schema, not part of `RootConfig`

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `configs/sim/*.yaml` files are validated by pydantic models in `simulator/scenarios.py` (`ScenarioCfg`, extra="forbid") and are NOT merged into `configs/base.yaml` / `RootConfig`. Simulator-only knobs (budget, R/P/G, benchmark items, rollout concentration, oracle resolution) live there.
- Reason: scenario worlds are experiment fixtures, not scientific hyperparameters of the real system; mixing them into `RootConfig` would bloat the frozen config hash of every run and create unused-key noise for GPU runs. Same unknown-key-fails guarantee as the main config.
- Affects: Roadmap Phase 3 "Files to create" list (configs/sim/*.yaml kept, but schema location deviates deliberately).

## D-29 — Import-rule test fixed to recognise `curator_rl.<pkg>` submodule paths

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `tests/architecture/test_import_rules.py` now records the first TWO components for `curator_rl.*` imports; L1 modules may import `curator_rl.core` and their own package `curator_rl.<self>`.
- Reason: the Phase 1 parser kept only the top-level component, so `from curator_rl.core.types import X` looked like a bare `curator_rl` import and the L1-only-core exception could never fire. Latent until Phase 3 added the first L1 code (scheduler).

## D-30 — `EnvStatus` + `SignalVector` added in Phase 4; `MixtureDecision` still deferred

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `core/types.py` gains `EnvStatus` (StrEnum: S1_unexplored … S5_unreliable) and the frozen `SignalVector` dataclass (posterior pass rate + CI, lp/lp_se/lp_z, lp_raw_fast/slow, richness, mean score, unit cost / cost-norm, proxy_raw/proxy_unit/proxy_reward, status + S2 sub-label note). `MixtureDecision` stays deferred to v1 Phase 5 per D-26.
- Reason: these are the first types with a Phase 4 producer and consumer; adding MixtureDecision now would still be an unused abstraction.

## D-31 — Richness smoothing: per-round fraction EW with `signals.lambda`

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `RichnessEstimator` smooths the per-round informative fraction with the signal lambda (stored value unchanged on zero-prompt rounds); it does not pool counts Beta-style.
- Reason: round fractions carry the heterogeneity directly; counts would need a separate Beta prior parameterisation that is not in the roadmap spec for SR.

## D-32 — Band/variance richness fidelity under the round aggregate

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: the observation aggregate carries only `n_groups_mixed` (0<k<G). Band membership is estimated from the posterior [p_lo, p_hi] overlap with the band; the variance mode decides on the discounted per-round group-rate std and emits 0/1. `RichnessEstimator.update_groups` provides the exact per-group path (used by unit tests and wherever raw groups exist later).
- Reason: Roadmap S-8 collapses the 0.1–0.9 band to `mixed` for G = 8, so the aggregate path is exact for the MVP settings; the per-group entry point keeps fidelity possible without schema churn.

## D-33 — Interim unit-cost EMA inside the Signal Engine

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `c_i` for the proxy is an EMA of `EnvRoundObs.cost_usd / n_prompts` computed by the engine (updated only on prompt-bearing rounds; unknown unit cost => c_norm = 0 => proxy reward 0). The real Cost Meter (Phase 6) replaces the source behind the same interface.
- Reason: cost-normalised rewards are needed from Phase 5 on; the interim source keeps the proxy testable off-GPU. Never `estimate_cost()` priors.

## D-34 — S5 mismatch input API; full wiring deferred to Phase 9

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `StatusClassifier.set_mismatch(flag)` + `SignalEngine.set_calibration_mismatch(env_id, flag)` carry the streak counters and S5 enter/leave (q, q' from `calib.mismatch_windows` / `mismatch_clear_windows`). Phase 4 never feeds real flags — the α/β fitter that determines mismatch does not exist yet.
- Reason: the state machine is built and unit-tested now; the informational dependency is genuinely Phase 9 calibration.

## D-35 — Proxy discount reuses `signals.lambda`

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: σ_LP and the r̃ reservoir are discounted with `signals.lambda`; no separate `proxy.lam` field.
- Reason: one less unconfigurable scale; the roadmap leaves the proxy discount unspecified and a distinct value would need its own tuning grid.

## D-36 — Dwell semantics: gates S3/S4 exits; S2 exits ungated

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: leaving S3/S4 requires the hysteresis leave-rule AND `dwell_min` completed rounds; leaving S2 is ungated (entries into S3/S4 already require h consecutive rounds); leaving S1 is exempt and entering S5 is exempt (Roadmap E.5); leaving S5 uses q' clean windows.
- Reason: reading "D_min before leaving ANY status" literally would make S2's exit the slowest path in the machine with no benefit — the residual class needs no protection. The S3-exit classification falls through to _from_s3_exit (S2 by default, S4 only if its entry rule is already mid-run) so a forgetting spike cannot re-enter S3.

## D-37 — LP-B uses two-stage WLS with residual-variance inflation; LP-C runs on raw per-round rates

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: LP-B's SE divides the weighted residual spread by (k − 2) and inflates each round's variance by that spread (stage-2 weights = 1/(se_i² + resid_var)); LP-C's window means use the RAW per-round rate (k/n of that round alone), not the discounted pooled posterior.
- Reason: the first LP-B SE divided by Σw·(k−2), shrinking the SE by the prompt count and giving a measured FPR of 1.0 (flat sequences flagged significant 100% of the time). LP-C on pooled windows overlaps mechanically (both windows share EW mass) and measured FPR 0.40; on raw rates ≈ 0.06.

## D-38 — Studies collect episodes in-process via a recording scheduler

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `experiments/analysis/common.py` wraps Uniform/Random in a `RecordingScheduler`, runs `run_episode` directly and keeps the full `RoundObservation` stream plus the harness log's `true_skills`/`true_pass_rates` in memory. No change to the harness, world or `run_sim.py`.
- Reason: `run_sim.py` does not persist round logs; rebuilding worlds/episodes deterministically keeps the studies bit-reproducible without new plumbing.

## D-39 — Truth status labels mirror the prediction-side S1 evidence band

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: ground-truth S1 = the same discounted group-evidence state machine the predictor uses (n_groups_eff below the entry floor / above n_min), computed from allocations only (bookkeeping, not a hidden signal). S2/S3/S4 truth uses the true pass rate and true skill velocity; the S3 truth velocity anchor is 1e-2 skill/round; SR truth = informative-band heuristic.
- Reason: an undiscounted `rounds_seen`-only truth mismatched the prediction semantics for starved arms (1031 bogus errors in validation) and would have made every F1 number meaningless.

## D-40 — LP estimator study result: LP-B chosen; rule applied as pre-registered

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: LP-B (weighted slope, W=10) is adopted (`signals.lp_method: lp_b`). Study numbers (S-A/S-B/S-E × {uniform, random}; tuning seeds 0–19, evaluation 100–129): LP-B rho 0.19–0.20 vs true skill velocity, FPR 0.000, lag 3, sign accuracy 0.70–0.72. LP-A had better rho (0.23–0.27) but FPR 0.103–0.104 > the 5% constraint; LP-C rho 0.13–0.16 and sign accuracy 0.31. Both phases picked LP-B.
- Reason: the roadmap's pre-registered selection rule (best rho subject to FPR ≤ 5% and lag ≤ 5; fallback LP-A) selects LP-B, not the expected fallback. LP-A remains implemented (config), and the FPR miss is documented rather than silently allowed.

## D-41 — Study flip-rate estimator: pooled post-first-exit rate

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `flip_rate_pooled` = total transitions after each env's first S1 exit × 100 / total post-exit intervals, pooled across envs and captures. Used instead of a mean of per-env rates (which a late-S1-exit arm with a 2-interval tail can blow up to 100).
- Reason: the roadmap's "transitions per environment per 100 rounds" is a rate whose per-env mean is not a stable estimator on short tails; the pooled version measures exactly the flapping quantity with full precision.

## D-42 — Tuned status thresholds replace the roadmap starting values

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `configs/base.yaml` status block becomes p_sat 0.7, p_hard 0.05, z_up 1.5, dwell_min 3, consecutive_rounds 4 (h); sr_hard 0.10, z_neg 2.0, hysteresis 0.05 unchanged. NEW config key `signals.status.s1_entry_ratio: 0.5` — S1 is re-entered only below half of n_min evidence (exit still needs n_min), a roadmap-native anti-flap constant (the S1 exploration quota already references n_min/2).
- Reason: tuning grid (243 points over p_sat/p_hard/z_up/dwell/h × S-A/S-B/S-E/S-G) maximised truth-anchored macro-F1 subject to pooled flip ≤ 5: winner F1 0.9705/flip 0.37 on tuning seeds 0–9; validated on evaluation seeds 100–119: F1 0.9672/flip 0.38 (eval's own argmax differs only in h=2 vs 4 with near-identical F1; h=4 kept as the tuning-phase pick). Exit criteria met: F1 ≥ 0.85, flip ≤ 5/100. The pre-flip numbers (per-env-rate flip 6–7.5, bound-driven S1 re-entries) motivated D-41's estimator and the S1-entry hysteresis.

## D-43 — Truth anchor frozen as constants inside the tuning study

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `experiments/analysis/tune_status_thresholds.py` labels ground truth with hardcoded TRUTH_ANCHOR constants (the pre-tuning roadmap defaults), NOT the current base.yaml values; the grid varies prediction thresholds only.
- Reason: anchoring truth to mutable base.yaml values would make any post-hoc re-run circular (truth moving with the tuned prediction).


---

# Phase 5 decisions (D-44 … D-51; written retroactively at Phase A close-out, 2026-10-09)

These were cited in code comments while Phase 5 was built but not logged. Dates are the implementation dates (2026-10-05/06).

## D-44 — Phase 5 methods are built from `base.yaml` overrides, not per-method YAML

- Date: 2026-10-05 (Phase 5)
- Owner: P1
- Decision: `experiments/run_sim.py::make_scheduler` builds `static`, `lp`, `ucb`, `curator` from the single `configs/base.yaml`. `StandardUCB` is `Curator` with `gamma=1`, `cost_exponent=0`, `status_control=off`; `LPCurriculum` runs its own `SignalEngine` with `lp_use=abs`. No `configs/scheduler/*.yaml` files are created.
- Reason: same argument as D-21 — one validated, hashed config; ablation flags are config values, so an ablation differs from the full method by an inspectable override.
- Affects: Roadmap Phase 5 "Files to create" (`configs/scheduler/*.yaml`) — deliberate deviation under the SHOULD-override rule.

## D-45 — Static baseline weights = benchmark slice weights π_d (superseded in part by D-63)

- Date: 2026-10-05 (Phase 5)
- Owner: P1
- Decision: the Static mixture used the benchmark slice weights π_d, which are visible before training and never a hidden training signal.
- Reason: Roadmap J.1 #2 requires fixed weights set before training. Defect found at Phase A: S-A/S-B/S-C set no π_d, so Static was identical to Uniform. See D-63.

## D-46 — Uncharged reporting evaluations in the harness

- Date: 2026-10-05 (Phase 5)
- Owner: P1
- Decision: `ScenarioCfg.report_eval.interval_rounds` (default 0 = off) makes the harness record an uncharged benchmark evaluation every N rounds (`EpisodeResult.report_logs`), identical for every method.
- Reason: Roadmap L.4 / H.6 — the score-vs-cost curve is needed for compute-to-target (Gate 2e) and costs nothing under the ledger rule. Gate 2e enables it (interval 2) on S-C only.

## D-47 — S1 quota doubling is structural, with a quota mass budget

- Date: 2026-10-05 (Phase 5)
- Owner: P1
- Decision: the S1 exploration quota `s1_quota` doubles while `n_groups_eff < n_min/2` (the factor 2 is the Roadmap E.5 rule, not a tuned value). Boosted quotas never sum above the mass left by the other arms' floors; they are rescaled proportionally.
- Reason: without the budget, several S1 arms at once would demand more than 100% of the mixture.

## D-48 — `MixtureDecision` added; quotas moved to `core`; L1 may import other L1 packages

- Date: 2026-10-05 (Phase 5)
- Owner: P1
- Decision: (1) `core/types.py::MixtureDecision` ends the D-26 deferral. (2) `largest_remainder_quotas` moves to `core/quotas.py` (`simulator/quotas.py` re-exports it) so the L1 scheduler can compute `MixtureDecision.quotas` without importing the simulator. (3) `tests/architecture/test_import_rules.py` now lets an L1 package import `core` and any other L1 package; L2 imports from L1 stay banned.
- Reason: Roadmap A.2 decision 1 — the scheduler is a pure function of observations, so it must own a `SignalEngine` (another L1 package). The original rule (L1 may import only `core` and itself) could not express that. Heavy-library and L2 bans are unchanged.

## D-49 — D-UCB hyperparameters tuned on simulator tuning seeds

- Date: 2026-10-06 (Phase 5; values superseded by D-68)
- Owner: P1
- Decision: `gamma 0.95→0.90`, `exploration_coef 0.5→0.25`, `tau 1.0→0.3`, `epsilon` stays 0.10. New keys `s3_multiplier 0.5`, `s4_multiplier 0.5`, `s5_shrink 0.5`, `s1_quota 0.30` (Roadmap E.5 values). Evidence: `reports/sweeps/tune_ducb.{json,md}` (grid γ{.90,.95,.98} × κ{.25,.5,1} × τ{.3,1,2} × ε{.05,.10}, tuning seeds 0–9, S-A + S-C, objective mean relative improvement over Uniform; winner +12.65%).
- Reason: Roadmap E.7 — hyperparameters are chosen on tuning seeds only; evaluation seeds start at 100.
- Caveat (see D-67): the objective is score only; the winner is close to winner-take-all per round.

## D-50 — Gate 2(c) "junk" = the zero-signal arm; window = last 40% of the episode

- Date: 2026-10-06 (Phase 5)
- Owner: P1
- Decision: the junk arm is S-B's `too_hard`. The noisy arm is *not* a Gate 2 junk arm: pre-calibration it is indistinguishable from a plateau (E.5, H3) and is the target of Phase 9 calibration. The share is measured over the last 40% of rounds, not "after round 30", because S-B episodes are about 25 rounds.
- Reason: Roadmap Q Gate 2(c) was written for longer episodes; the pre-registered intent (junk compute share ≤ 1.5× floor late in the run) is kept.

## D-51 — An established S3/S4 arm is not demoted to S1 by evidence decay

- Date: 2026-10-06 (Phase 5)
- Owner: P1
- Decision: in `StatusClassifier._target`, an arm whose current status is S3 or S4 is exempt from the S1 test; S1 keeps its E.5 priority over S5 for every other arm; S5 can still override an S3/S4 incumbent. (Phase A restored the S1-over-S5 order after an intermediate edit had swapped it; regression tests added.)
- Reason: a starved too-hard arm lost evidence, re-entered S1, received the 30–60% exploration quota, re-learned it was too hard, and repeated. The measured residual pump (S2 arms re-entering S1) is below 0.75% of the budget (D-66).

---

# Phase A decisions (2026-10-09)

## D-63 — Static baseline: size-proportional on a declared `nominal_size`

- Date: 2026-10-09 (Phase A)
- Owner: P1
- Decision: `SimEnvCfg.nominal_size` (optional, scheduler-visible, ignored by the world and oracles). `_static_weights` uses size-proportional weights when every arm declares one, else π_d (D-45), else uniform. The eight scenario files declare sizes by env position from the list [6873, 7500, 974, 6873, 3000, 374, 5000, 2000] (real dataset sizes: GSM8K, MATH, MBPP, …), blind to every arm's quality.
- Reason: Roadmap J.1 #2 / v3 D-61. Sensitivity check (scratch, eval seeds 100–119, 20 random size permutations): Static mean 0.5845 in S-A (range 0.539–0.609) and 0.4094 in S-B (0.394–0.427). The declared assignment (S-A 0.608) is near the top of Static's range, so it is a conservative comparison for Curator, which beats Static in 20/20 (S-A) and 18/20 (S-B) permutations.
- Affects: S-H previously used its π_d for Static; it now uses nominal sizes like the rest.

## D-64 — `status_control: hard` becomes the default

- Date: 2026-10-09 (Phase A)
- Owner: P1
- Decision: `configs/base.yaml: scheduler.status_control: soft → hard` (S3 capped at 2× floor, S4 at the floor; Roadmap E.5).
- Reason: selected on tuning seeds 0–49 (not on the evaluation seeds): S-B junk-share passes in 49/50 seeds (soft: 38/50); mean improvement over Uniform S-A +0.0697 (soft +0.0624), S-B +0.0120 (soft +0.0147), S-C +0.0532 (same). Confirmed on evaluation seeds 100–149: Gate 2(c) passes (48/50 seeds, mean share 0.0144 vs 0.01875 threshold). The only cost is −0.0027 on S-B on tuning seeds, well inside the Gate 2(b) tolerance.
- Caveat: the D-49 sweep ran with `soft`; Phase C re-tunes with `status_control` in the grid.

## D-65 — Gate 2 passed at 50 seeds; oracle-gap shortfall analysed (D-18 procedure)

- Date: 2026-10-09 (Phase A)
- Owner: P1
- Decision: Gate 2 is passed (evidence: `reports/gate2/gate2.{json,md}`, seeds 100–149, criteria a–e all pass). The development target (oracle-gap closure ≥ 75% in S-A) is **not** met: 46% in the Gate report (best-static oracle on the first 20 seeds), 48.0% on all 50 seeds. The shortfall is analysed here before leaving the simulator phase, per D-18.
- Analysis: S-A episodes are 25 rounds and Curator spends the first `warmup_rounds` = 6 as a uniform mixture. A scheduler that plays Uniform for k rounds and the privileged static-oracle mixture afterward closes at most **75.9%** (k=6), 64.8% (k=9), 49.1% (k=12). Curator reaches 48.0%, i.e. 63% of its own ceiling. About 24 of the 52 missing points are therefore the unavoidable exploration prefix; the remaining ~28 are the cost of estimating from noisy proxies and of the near one-hot allocation (D-67).
- Consequence: the short horizon is a property of the scenario, not only of the scheduler. Roadmap v3 §4.2 requires at least 60 rounds in real runs, and Phase C re-parameterises the scenarios to the measured horizon and re-states the oracle-gap target there.

## D-66 — S1 re-entry left as is; measured residual under 0.75% of the budget

- Date: 2026-10-09 (Phase A)
- Owner: P1
- Decision: no further change to S1 re-entry beyond D-51.
- Reason: the suspected "S1 re-entry pump" was traced and quantified on tuning seeds 0–49: mass spent on S1-re-entered arms is 0.00% (S-A), 0.72% (S-B), 0.68% (S-C) of total budget. The earlier Gate 2(c) failure on file was produced by code from before D-51. Redesigning S1 would also move the Phase 4 truth-label mirror (D-39), so it is not worth the risk.

## D-67 — Allocation concentration is measured and added to Phase C tuning

- Date: 2026-10-09 (Phase A)
- Owner: P1
- Decision: report mean max-weight and mean weight entropy per method as standing diagnostics. Phase C's fair-tuning grid adds a concentration cap as a tuning constraint, and a mixture-smoothing option is allowed if it costs no more than the pre-registered tolerance.
- Reason: after warm-up, in 61% (S-A), 42% (S-B) and 56% (S-C) of rounds Curator puts more than half of the mixture on one arm (tuning seeds 0–49, measured under `soft`; under the new default `hard` it is 97% / 61% / 96%, mean max-weight 0.89 / 0.56 / 0.87, normalised entropy 0.33 / 0.60 / 0.33), because z-scored scores at τ=0.3 give a near one-hot softmax and a D-UCB bonus surge when a forgotten arm's Ñ_i collapses. The simulator has no forgetting or interference between rounds, so it cannot penalise this; a real GRPO run could. Roadmap Phase 5 item 13 lists "weights collapse onto one environment" as a failure condition.


---

# Phase C decisions (2026-10-09)

## D-68 — SEC-style and DUMP-style baselines; per-method baseline config; fair tuning

- Date: 2026-10-09 (Phase C)
- Owner: P1
- Decision: (1) `EnvRoundObs.sum_abs_adv` (optional trailing field, default 0.0) carries the sum over prompt groups of the group mean |GRPO advantage|; `core/advantage.py::group_mean_abs_advantage` computes it from pass counts alone with TRL's convention (unbiased group std, eps 1e-4): `2p(1-p)/(std+eps)`, zero for all-pass and all-fail groups. (2) `SECStyleBandit` (TD(0) on mean |A|, Boltzmann) and `DUMPStyleUCB` (UCB on mean |A|, softmax) are added as cost-blind, uncalibrated baselines. They are "style" re-implementations from the papers' descriptions, not reproductions of the authors' code. (3) A validated `baselines:` section in `base.yaml` gives every baseline its own hyperparameters (`ucb.{exploration_coef,tau}`, `lp.tau`, `sec.{alpha,tau}`, `dump.{exploration_coef,tau}`); before this, Standard UCB and LP silently reused Curator's tuned κ and τ. (4) All adaptive methods share the exploration floor ε = 0.10 and the warm-up length as protocol constants (not tuned). (5) `experiments/sweeps/tune_all.py`: 8 pre-registered configurations per adaptive method on tuning seeds 0–49 over S-A, S-B, S-C, S-I (S-J held out for H6). Objective: mean relative improvement over Uniform per seed. Selection rule, fixed in advance and identical for all methods: the best configuration, then every configuration within one standard error of it, then the least concentrated (highest normalised allocation entropy, D-67).
- Selected (evidence `reports/sweeps/tune_all.{json,md}`): Curator γ 0.95, κ 0.5, τ 0.3 (was γ 0.90, κ 0.25; objective +0.080 vs +0.081 for the old point, chosen for lower concentration); Standard UCB κ 0.5, τ 0.3; LP τ 0.75; SEC α 0.1, τ 0.4; DUMP c 2.0, τ 0.4. `base.yaml` carries these values.
- Reason: Roadmap v3 §4.4 and Part J ("equal tuning effort per method"); D-67 asks for a concentration-aware tie-break.
- Caveat: the Curator grid held `status_control` and `cost_exponent` fixed (method definition, D-64); the grid for SEC/DUMP spans their main knobs only. Mean |A| for binary GRPO rewards is `2·sqrt(p(1-p))` times a G-dependent constant, so the bandits' signal is the same quantity SEC and DUMP use.
- Effect on earlier results: Gate 2 re-run with the new Curator values passes all five criteria (see GATES.md); the oracle-gap closure in S-A moves from 46–48% (D-65) to 43%.

## D-69 — Gate 2'' fails on S-I before calibration; diagnosis and disposition; Gate 2(b) unit bug fixed

- Date: 2026-10-09 (Phase C)
- Owner: P1
- Decision: (1) Bug fix: `gate2_check.py` compared score differences in 0–1 units with `DELTA_GAP = 1.0`, a 100-point tolerance, so criterion (b) was vacuous. It is now `0.01` (1 benchmark point). The criterion still passes (lower bound +0.0069). (2) Gate 2'' (`experiments/gate2b_check.py`, evaluation seeds 100–149, S-J not tuned on): (i) fails, (ii) passes, (iii) and (iv) pass. Criterion (i): Curator beats SEC-style and DUMP-style in **S-C** (+0.061 [+0.050, +0.072] and +0.059 [+0.048, +0.070]) but not in **S-I** (−0.008 [−0.013, −0.004] and −0.008 [−0.013, −0.003]). In S-I Curator ties Uniform (+0.000 [−0.005, +0.005]) while the other adaptive methods beat it by 0.8–1.6 points. In the homogeneous twin S-J, Curator is above Uniform (+0.010 [+0.005, +0.015]), as H6 expects. (3) Disposition: the stop rule fired, so the cause was diagnosed before any trainer work (`reports/analysis/curator_diagnosis.{json,md}`, tuning seeds). The criterion is not changed. It is to be re-run with the calibrated Curator (Phase D) and must pass there with no privileged information.
- Diagnosis: (a) the noisy arm takes 20–25% of compute (cheap, rich in signal, and indistinguishable from a plateau before calibration, E.5/H3). A privileged S5 flag on it (an upper bound, not a method) lifts Curator from +0.011 to +0.030/+0.036 in S-I, from +0.034 to +0.078/+0.084 in S-J and from +0.121 to +0.158/+0.167 in S-A. (b) Cost normalisation costs 2.9 points in S-I (+0.053 for the UCB form vs +0.024 with cost) because the valuable arms (MATH, MBPP) are the expensive ones, yet it is what wins S-C (+0.18 vs +0.05). (c) relative to `soft`, `hard` statuses cost 1.3 points in S-I and 2.9 points in S-J and gain 1.3 points in S-A (the junk-share criterion needs the S4 cap, D-64). (d) The static oracle's mixture in S-I (48% MATH, 27% GSM8K, 25% Countdown, 0% MBPP) depends on the GSM8K↔MATH transfer, which learning-progress and richness signals cannot see; the first-order myopic oracle reaches only +6.5% vs +12.5% for the best static mixture.
- Reason: the project's own thesis is that a cheap proxy needs held-out calibration. The simulator now measures how much of the S-I gap a correct S5 detection can recover; Phase D has to deliver it, and Gate 2'' (i) on S-I is the acceptance test.
- Caveat: S-I's costs and pass rates are placeholders (D-70), so the sign and size of the cost effect in S-I will be re-measured after the Phase B pilot.

## D-70 — S-I and S-J scenarios (long horizon; heterogeneous and homogeneous cost)

- Date: 2026-10-09 (Phase C)
- Owner: P1
- Decision: `configs/sim/scenario_si.yaml` is the simulator analogue of the Level-1 portfolio (GSM8K, MATH35, MBPP, Countdown, noisy, too-hard) with unit costs 1×–3×, a GSM8K↔MATH positive transfer of 0.3, budget 9.5 and about 70 Uniform rounds (the v3 horizon rule is at least 60). `scenario_sj.yaml` has identical dynamics, sizes and budget with all unit costs equal to S-I's mean (0.0017). `nominal_size` follows real dataset sizes (6873, 4700, 974) with declared values for procedural arms. Tests enforce rounds ≥ 60, a ≥ 2× cost spread in S-I, equal costs in S-J and identical dynamics between the two.
- Reason: Roadmap v3 §4.1/§4.2 and H6. The static oracle beats Uniform by 12.5% in S-I and 23.8% in S-J, so both discriminate (Phase 3 rule: > 10%).
- Caveat: PLACEHOLDER costs, pass rates and sizes; refit from the Phase B pilot (Roadmap F.7). S-J is never used for tuning.

## D-71 — `hard_s4` status mode tried and rejected

- Date: 2026-10-09 (Phase C)
- Owner: P1
- Decision: not adopted. Hypothesis: the S3 hard cap hurts moderately saturated cheap arms in S-I, so capping only S4 hard (junk-share criterion) would recover S-I while keeping Gate 2(c). Tuning seeds 0–49, Curator relative improvement S-A/S-B/S-C/S-I/S-J: off +0.094/+0.019/+0.177/+0.018/+0.046; soft +0.096/+0.025/+0.170/+0.018/+0.054; hard_s4 +0.096/+0.026/+0.170/+0.007/+0.037; hard +0.116/+0.026/+0.170/+0.009/+0.029; S-B junk-share seeds passing: 1 / 11 / 49 / 50 of 50. `hard_s4` did not recover S-I, and S-A prefers the S3 cap, so `hard` stays (D-64) and the code was reverted.
- Reason: keep the config surface minimal; differences of 1–2 points in S-I and S-J are close to their standard errors (0.7 points).


---

# Phase D decisions (2026-10-09, increment D1)

## D-72 — Paired calibration in the simulator; calibration charged only to methods that use it

- Date: 2026-10-09 (Phase D)
- Owner: P1
- Decision: (1) `SimWorld.evaluate_benchmark_paired`: each slice holds fixed items with a latent u ~ U(0,1), solved iff u < p(skill); a `churn` fraction of latents is redrawn per evaluation; the SE of a slice's change is the paired (McNemar) SE, floored at one discordant item, and travels in the new optional field `CalibrationObservation.delta_se_by_domain`. A separate RNG stream (`sim_calib`) keeps training draws identical with or without calibration. Scenario options: `calib.paired`, `calib.churn`, `calib.items_per_slice`, `calib.cost_per_item_usd`. (2) `BaseScheduler.uses_calibration` (default False): the harness runs and charges calibration evaluations only for schedulers that set it (Curator when `calib.enabled`).
- Reason: (1) Roadmap v3 §4.3 — independent re-sampling made window gains unmeasurable (SE ≈ 0.05 vs gains ≈ 0.01); real calibration re-evaluates the same items. (2) Roadmap H.6: "calibration and Curator overhead charged to Curator only". Before this, Uniform paid for calibration on S-G.
- Affects: re-running the Phase 4 status studies on S-G would now give different (longer) Uniform/Random episodes; the committed Phase 4 reports remain valid for the code they were generated with.

## D-73 — Calibrator v1: own-slice credit, robust proxy scale, S5 mismatch test; `calib.enabled`

- Date: 2026-10-09 (Phase D)
- Owner: P1
- Decision: `calibration/credit.py` (window gains with paired or independent SE, C1 share credit, `OwnSliceCredit` = C2 diagonal form) and `calibration/calibrator.py`. Per environment j: b_j = weighted least-squares gain on its own slice per unit compute share across windows; proxy-implied gain = s · x̄_j with x̄_j the exposure-weighted window mean of α·LP′ + β·SR and s = median_j(b_j / x̄_j) (robust: one junk arm cannot drag the scale to zero); mismatch flag when (b_j − s·x̄_j)/se(b_j) < −z_mis after `k_min` windows (Roadmap I.5; underestimation never flagged). Flags feed the existing S5 state machine. New config key `calib.enabled` (Roadmap ablation "no calibration"); Standard UCB forces it off.
- Reason: Roadmap Part I; the S5 path was built in Phase 4 (D-34) and waited for this data source. The α/β refit, C2d (full transfer matrix) and calibrated-reward feedback into the bandit are not part of v1.

## D-74 — Calibration Power Study: under a realistic evaluation cost, calibration v1 does not pay

- Date: 2026-10-09 (Phase D)
- Owner: P1
- Decision: no calibration configuration is adopted yet; `calib.enabled: true` stays in `base.yaml`, but the simulator scenarios do not offer calibration until a design passes Gate 6-sim.
- Evidence: `experiments/analysis/calibration_power.py` → `reports/analysis/calibration_power.{json,md}` (tuning seeds 0–29, S-I and S-J, paired items, churn 0.02, eval cost = mean training unit cost / G per item). Grid: items/slice {25, 50, 100} × K {5, 10} × k_min {2, 3} × S5 shrink {0.5, 0.1}. Within the 10% cost cap the noisy arm is detected in at most 10% of seeds (50 items, K=10); with 100 items and K=5 detection rises to 37% (S-I) but calibration costs 16% of the budget and the score falls below Curator without calibration. The pre-registered rule picked 25 items / K=10, which detects nothing (0%) and is statistically indistinguishable from no calibration. A free-calibration run with 200 items (scratch, tuning seeds) detected the noisy arm in 15–21 of 30 seeds and lifted S-I from +1.5% to +2.7%, so the mechanism works but is underpowered at a realistic cost.
- Consequence: Gate 6-sim (noisy arm flagged S5 in ≥ 90% of seeds) is NOT met, and Gate 2'' (i) on S-I stays failed. The bottleneck is statistical power per unit of evaluation cost, not the detector logic. Options are listed in the Phase D status report to the owner; none is adopted without a decision.


## D-75 — Targeted calibration (owner's choice) and its power-study result

- Date: 2026-10-09 (Phase D, increment D2)
- Owner: P1 (direction chosen by the owner after D-74)
- Decision: (1) `Calibrator.select_targets`: the first calibration evaluates every slice (the paired baseline); afterwards, with `calib.targeting: exposure`, only the slices of the `calib.max_targets` environments that received the most compute in the window are evaluated, `calib.items_per_slice` paired items each. Each evaluated slice yields one gain over the span since it was last evaluated, regressed on the compute share it received over that span. (2) Interface: `scheduler.calibration_request() -> {env: items}`; the harness passes it to the world and charges `cost_per_item_usd × items`. A real trainer callback asks the same way. (3) Defaults adopted from the pre-registered rule of the power study: `targeting: exposure`, `max_targets: 1`, `items_per_slice: 100`, `interval_rounds: 10`, `k_min: 2` (S5 shrink unchanged at 0.5).
- Evidence (`reports/analysis/calibration_power.md`, tuning seeds 0–29, S-I/S-J, realistic per-item cost): at equal cost, targeting detects the noisy arm 3–8× more often than evaluating every slice. Top-3 / 200 items / K=10 detects it in 87% (S-I) and 80% (S-J) of seeds at 9.3% cost, but its false-flag rate (7.3%) breaks the 5% limit. The rule's choice (top-1 / 100 / K=10) detects 33% / 10%, costs 2.4%, and reaches +2.91% vs +2.58% for Curator without calibration: statistically indistinguishable (paired S-I +0.0026, se 0.0087; S-J +0.0040, se 0.0111).
- Bug found and fixed while running it: the study's `S5Recorder` wrapper did not forward `calibration_request`, so the first targeted run silently evaluated every slice; the results above are from the corrected run.
- Consequence: Gate 6-sim (≥ 90% detection) is still not met, but targeting makes it reachable at under 10% cost once the false-flag rate is controlled. More importantly, S5 detection alone cannot pass Gate 2'' (i) on S-I even when perfect: the privileged upper bound (D-69) reaches +3.0–3.6% on S-I versus about +4.2% for SEC-style and +5.5% for Standard UCB on tuning seeds. The remaining gap is cost normalisation undervaluing expensive-but-valuable arms, which is what the calibrated reward (measured gain ÷ cost fed back into the bandit, Roadmap I.2) is for. That is the next increment.


---

# Calibrated Reward Engine (CRE) — pre-registration (2026-10-09)

## D-76 — CRE design, frozen BEFORE any CRE result exists

- Date: 2026-10-09 (Phase D, increment D3)
- Owner: P1 (design reviewed and tightened by the owner's three clarifications A/B/C)
- Latent quantity (A): **expected change of environment j's own calibration slice per training dollar charged to j, under the mixtures actually run** (`b_j`, benchmark-fraction per dollar). The regressor is the dollars the ledger charged to j over the span since its slice was last evaluated, NOT prompt share, so cost enters as a measured regressor and no noisy cost is ever a denominator of the evidence. Attribution caveat: an own-slice change also contains transfer from other arms and re-evaluation churn; it is an observational estimate under the run's mixtures, not a causal effect. It does not isolate an arm whose slice is shared with another arm (the real noisy arm shares GSM8K inputs); that needs the full transfer-matrix credit (C2d), a later phase.
- Estimator (candidates A and B unified; one Gaussian posterior per arm): likelihood `gain_k = b_j · usd_k + noise(se_k)` from paired span gains (se = paired McNemar SE); prior `b_j ~ N(m0_j, v0_j)` with `m0_j = s · x_j / c_j` where x_j is the arm's CURRENT proxy signal (α·LP′ + β·SR), c_j its measured unit cost, and `s` a robust scale `median_j(b̂_j · c_j / x̄_j)` over arms that have evidence (s ≥ 0); `v0_j = (ρ · max(|m0_j|, floor))²` with `floor = 0.05 · R_max / B`. Posterior mean combines the two by precision; arms without calibration evidence sit exactly at their proxy prior, so no separate fallback path exists. Until the first ready calibration produces `s`, the existing proxy reward is used unchanged (one documented switch).
- Constants frozen from TUNING seeds only (`experiments/analysis/cre_constants.py`, seeds 0–49, scenarios S-A, S-B, S-C, S-I, never S-J, never evaluation seeds): `ρ = cre.prior_rel_sd` = robust SD (1.4826·MAD, clipped to [0.3, 3]) of the relative residual `(b̂_j − m0_j)/max(|m0_j|, floor)` under free, high-power calibration; `R_max = cre.roi_scale` = 95th percentile of `b̂_j · B` over arms with evidence, floored at 0.02. Recorded in D-77 before the evaluation run.
- Reward transform (B): `r_j = clip(m_j · B / R_max, 0, 1)` where `B` is the run's total budget (first observed budget_remaining + first round cost). A fixed, per-arm transform: no per-round scaling, no dependence on other arms; negative gains clip to 0; ties resolve by env id downstream. The new transform applies to every arm once active.
- Integration: the CRE reward REPLACES the proxy reward the D-UCB consumes, for every arm, from the first ready calibration on. The count-based bonus, softmax map, exploration floor ε/N, status constraints and sum-to-1 are unchanged. The posterior SD is logged and is NOT added as a second exploration bonus (no double counting). Calibration targeting unchanged (D-75). Calibration is charged by the harness exactly once, only to schedulers with `uses_calibration`.
- Primary configuration: `cre.mode: full`, `cre.discount: 1.0` (discounting off), targeting as D-75.
- Out-of-sample protocol (C): the evaluation run uses FRESH seeds **200–249**. Seeds 100–149 are historical (already conditioned the decision to build CRE, via the Gate 2'' failure) and are not used for any CRE statement.
- Primary comparison: Gate 2'' criterion (i), unchanged: CRE-Curator minus SEC-style and minus DUMP-style in S-C and in S-I, paired 95% bootstrap CI excluding 0 on seeds 200–249. (ii) S-J: CRE minus Uniform lower bound > −0.01. Gate 2 criteria (a), (b), (c), (e) are re-evaluated with the CRE variant on the same seeds; (d) (status macro-F1) is not re-evaluated because the classifier code is unchanged.
- Pre-registered ablations (reported, never used to select): `no_uncertainty` (use the likelihood-only point estimate, no shrinkage), `exact_cost` (privileged: the scenario's true unit cost replaces the measured one, to test whether cost-estimation noise matters), `discount 0.7` (candidate C as an ablation), existing targeted Curator without CRE, Curator without calibration. References: Uniform, Standard UCB, SEC-style, DUMP-style.
- Conduct rules: no estimator, transform, constant, targeting or criterion change after seeing seed 200–249 outcomes. Diagnosis on tuning seeds is allowed. A new idea after a failure is logged as a NEW hypothesis with its own pre-registration and fresh seeds, never as a rescue of this run. If Gate 2'' fails, the failure is reported as is.
- Caveats carried into every report: S-I and S-J costs and pass rates are placeholders; MATH and MBPP are not built, so even a pilot refit leaves them ungrounded (the pilot covers GSM8K, Countdown, noisy); a pass is evidence about behaviour in the simulator, not about the real system; estimated ROI is not leave-one-out validated contribution.

## D-77 — CRE constants frozen from tuning seeds (before any evaluation-seed run)

- Date: 2026-10-09 (Phase D, increment D3)
- Owner: P1
- Decision: `cre.prior_rel_sd = 0.997`, `cre.roi_scale = 1.8128`, applied exactly as the D-76 rules specify. Evidence: `reports/analysis/cre_constants.{json,md}` (tuning seeds 0–49; free, every-slice, 400-item paired calibration). `cre.enabled` stays `false` in `base.yaml` until the D-76 evaluation run decides.
- Observations made while freezing (tuning seeds only, no rule changed): (1) the constants rest on S-I alone, because S-A, S-B and S-C episodes are about 25 rounds and with K = 10 the engine needs three calibrations (about round 30) before it activates. In those three scenarios the CRE is inert by construction, so criterion results there measure that CRE changes nothing, not that it helps. (2) R_max = 1.81 exceeds 1 because `b·B` is a marginal rate extrapolated to the whole budget, not a predicted gain; the tail is the cheap, fast-learning GSM8K arm (median `b·B` per arm in S-I: gsm8k 1.43, mbpp 0.42, math35 0.26, countdown 0.11, noisy 0.02, too_hard 0.00 with the largest SE). The rule was not bent to make R_max smaller. (3) The measured ROI ordering GSM8K ≫ MBPP > MATH35 > Countdown > noisy differs from the best static mixture of D-69 (48% MATH35, 0% MBPP): own-slice gain per dollar ignores transfer and saturation, which the discount ablation and later C2d credit address.

## D-78 — Amendment of D-76/D-77 BEFORE any evaluation-seed run: frozen proxy scale, passive constant estimation

- Date: 2026-10-09 (Phase D, increment D3)
- Owner: P1
- Status of the evaluation seeds: seeds 200–249 have NOT been run. The evaluation script was smoke-tested on tuning seeds 0–1 only (its output was discarded). D-76's no-change rule starts at the first evaluation-seed outcome, so this amendment is permitted; it is logged here with its reason so it cannot look like a rescue.
- What the tuning-seed diagnosis found (seed 0, S-I, `cre_diag`): the online prior scale `s = median_j(b̂_j·c_j/x̄_j)` over arms with evidence was fragile. Targeted calibration (top-1) leaves evidence on 1–2 arms; there it was math35 (b̂ = 0) and the noisy arm, so `s ≈ 2·10⁻⁵`, every proxy prior ≈ 0, every CRE reward fell to 0–0.07 and the count bonus dominated, so allocations matched the existing targeted Curator. The online scale also departed from clarification A, which asks for unit normalisation defined from training/development data only.
- Amendment: (1) the proxy-to-gain scale becomes a constant `cre.proxy_scale`, frozen from tuning seeds; the online median is kept only as a diagnostic (`online_scale`). (2) Constants are now estimated PASSIVELY (the engine records evidence but never steers allocation), so they come from the CRE-free trajectory and do not depend on the placeholder values being frozen. The first freeze (D-77) was circular in this respect and is superseded: its values (ρ 0.997, R_max 1.8128) came from a run where the engine was steering with placeholder constants.
- Frozen values (`reports/analysis/cre_constants.{json,md}`, tuning seeds 0–49, S-A/S-B/S-C/S-I, 840 arm estimates, free every-slice 400-item calibration): `cre.proxy_scale = 0.0001963`, `cre.prior_rel_sd = 1.232`, `cre.roi_scale = 1.5105`. Everything else in D-76 stands: estimator, transform form, integration point, discount off, targeting D-75, Gate 2'' criteria, ablations, seeds 200–249.
- Known limitations (stated now, not discovered later): the per-scenario median ratio spans a factor of about 3 (S-A 2.3e-4, S-B 2.6e-4, S-C 3.9e-4, S-I 1.3e-4), so one pooled scale is a compromise and the prior will be mis-scaled in some scenarios; S-A/S-B/S-C episodes are about 25 rounds, so with K = 10 the engine activates only after round 30 and is inert there; top-1 targeting leaves most arms without evidence, so their rewards are proxy-prior only for long stretches.

## D-79 — CRE evaluation result: Gate 2'' (i) FAILS on S-I; CRE not adopted; diagnosis

- Date: 2026-10-09 (Phase D, increment D3)
- Owner: P1
- Result (`reports/gate2/cre_gate.{json,md}`, fresh seeds 200–249, run once, design frozen by D-76/D-78): criterion (i) FAIL: CRE − SEC-style = −0.0080 [−0.0120, −0.0041] and CRE − DUMP-style = −0.0082 [−0.0126, −0.0038] in S-I (S-C passes, but see below). (ii) PASS: S-J CRE − Uniform = +0.0098 [+0.0055, +0.0142]. (iii) PASS: Gate 2 (a) +0.0602, (b) +0.0087 [+0.0037, +0.0136], (c) 50/50 seeds, (e) +18.0% [13.5%, 22.3%]; (d) not re-evaluated.
- What the result does and does not show: (1) The S-A/S-B/S-C numbers are identical to the existing targeted Curator to the last digit, because the engine is inert there by construction (episodes of about 25 rounds, K = 10, activation after round 30). Those passes are the existing Curator's, not evidence for the CRE. (2) In S-I and S-J the CRE is statistically indistinguishable from targeted Curator (S-I −0.002 [−0.006, +0.002]; S-J −0.001 [−0.005, +0.003]), and the ablations are indistinguishable from it: no-uncertainty, discount 0.7, and exact cost (the privileged test of the cost-noise hypothesis: cost-estimation noise is not what limits the result). (3) The calibration machinery itself behaves as built: noisy-arm detection 26–28% with 0% false flags at 2.4% of spend, unchanged by the CRE.
- Diagnosis (`reports/analysis/cre_diagnosis.{json,md}`, tuning seeds 0–19, S-I, rounds after activation): the reward channel has little leverage on the mixture. The std across arms of the exploration bonus (0.33) is twice that of μ̂ (0.17); μ̂ under CRE rewards correlates 0.79 with μ̂ under proxy rewards (the discounted history still carries the proxy's view for about 20 rounds, and the engine is active for only about half of a 65-round run); the mean total-variation distance between CRE and targeted weights is 0.024. Changing the reward, even to a better one, cannot move this scheduler's mixture much.
- Disposition: Gate 2'' (i) on S-I remains failed. `cre.enabled` stays `false` in `base.yaml`; the CRE code stays as an optional, tested component. No rescue variant is run against seeds 200–249 (D-76).
- Open hypothesis (NEW, not a rescue; needs its own pre-registration and fresh seeds, proposed 300–349): the bottleneck is the scheduler's reward-to-allocation leverage, not the reward's content. A candidate to pre-register is to tune the exploration coefficient κ and the activation horizon jointly for CRE on tuning seeds (κ was tuned for the proxy reward scale in D-49/D-68, so the bonus may be mis-scaled relative to the new reward), then evaluate once on fresh seeds. A second, independent candidate is value-of-information targeting (D-75's deferred item), which only makes sense once the posterior moves allocation.
- Caveats unchanged: S-I/S-J use placeholder costs and pass rates; MATH and MBPP are not built; estimated ROI is not leave-one-out validated contribution; a null here says nothing about the real system.

## D-80 — NEW hypothesis H-κ (pre-registration, not a rescue of D-76): reward leverage is limited by the exploration scale

- Date: 2026-10-09 (Phase D, increment D4)
- Owner: P1 (direction chosen by the owner's delegation: "choose what is appropriate")
- Hypothesis: the CRE's null result (D-79) came from the scheduler, not the reward: κ and τ were tuned (D-49/D-68) against the proxy reward's scale, so the count bonus (std across arms 0.33) swamps the reward spread (0.17). With κ and τ tuned for the CRE, calibrated feedback should move the mixture and the S-I score.
- Design (frozen now): grid of 8 = κ {0.02, 0.05, 0.1, 0.2} × τ {0.3, 1.0}, applied to TWO methods with identical grids: `cre` (CRE enabled, D-76/D-78 constants unchanged) and the CONTROL `curator_targeted` (same calibration, CRE off). The control separates "a smaller κ helps any Curator" from "the CRE benefits". Tuning: seeds 0–49, scenario S-I only (the only scenario where the engine activates; S-J is held out). Objective: mean relative improvement over Uniform per seed. Selection rule, same as D-55: best, then everything within one SE, then the highest normalised allocation entropy; each method selects its own point. All other settings frozen as in D-76 (K = 10, k_min = 2, targeting top-1, 100 items, discount 1.0, constants of D-78, ε = 0.10, γ = 0.95, status_control hard).
- Evaluation (once, fresh seeds 300–349; seeds 100–149 and 200–249 are spent): scenarios S-A, S-B, S-C, S-I, S-J. Primary: `cre_tuned` minus SEC-style and minus DUMP-style in S-I, paired 95% CI excluding 0. Attribution (secondary): `cre_tuned` minus `control_tuned` in S-I and S-J. No-regression: `cre_tuned` minus default-κ targeted Curator, lower bound > −0.01, in S-A, S-B, S-C, S-J. References: Uniform, Standard UCB, SEC, DUMP, default-κ targeted Curator, default-κ CRE.
- Interpretation rules (fixed in advance): a primary PASS with a positive `cre_tuned − control_tuned` supports "calibrated reward adds value once κ is matched"; a primary PASS with `cre_tuned ≈ control_tuned` means the gain is from κ, not the CRE, and the CRE is still not adopted; a primary FAIL closes the κ explanation. No further variants are run against 300–349.
- Caveats: S-I/S-J placeholders; the engine is inert in S-A/B/C (a changed κ still changes them, hence the no-regression check); a pass is about the simulator only.

## D-81 — H-κ tuning result, frozen before the evaluation seeds

- Date: 2026-10-09 (Phase D, increment D4)
- Owner: P1
- Result of the pre-registered tuning (`reports/analysis/cre_kappa_tune.{json,md}`, seeds 0–49, S-I): both methods select the same point under the D-55 rule, κ = 0.2, τ = 1.0. For the CRE the objective surface is flat to mildly positive (best +0.0124 at κ 0.05/τ 1.0, chosen +0.0098); for the control targeted Curator, lowering κ hurts at nearly every point (best +0.0029, mostly negative). At the chosen point the CRE-minus-control objective gap is +0.0069, about one standard error (0.0067), so a close result is expected.
- Frozen for evaluation: `cre_tuned` and `control_tuned` use κ = 0.2, τ = 1.0; everything else as D-76/D-78. Evaluation: `experiments/cre_kappa_gate.py` on seeds 300–349, run once, interpretation rules as fixed in D-80.

## D-82 — H-κ result: PRIMARY FAIL; the κ explanation is closed; CRE line stopped

- Date: 2026-10-09 (Phase D, increment D4)
- Owner: P1
- Result (`reports/gate2/cre_kappa_gate.{json,md}`, fresh seeds 300–349, run once, rules of D-80/D-81): primary FAIL: `cre_tuned` − SEC-style = −0.0103 [−0.0145, −0.0059], − DUMP-style = −0.0122 [−0.0169, −0.0076] in S-I. Attribution: `cre_tuned` − `control_tuned` = +0.0008 [−0.0039, +0.0057] in S-I and −0.0062 [−0.0113, −0.0008] in S-J, i.e. no CRE effect even with κ and τ tuned for it. No-regression FAILS in S-A (−0.0252 [−0.0346, −0.0152]) and S-C (−0.0289 [−0.0359, −0.0220]): κ = 0.2 starves exploration in the short 25-round scenarios where the engine is inert, so the tuned point is not adoptable even apart from the CRE. In S-J both tuned variants beat default targeted Curator (+0.013, +0.009) but not Standard UCB, which stays the strongest method there.
- Interpretation (pre-fixed in D-80): a primary fail closes the κ explanation. Together with D-79 the evidence is consistent: in this simulator, with sparse targeted calibration, feeding measured own-slice gain per dollar back into the bandit does not improve allocation, whether or not the exploration scale is matched.
- Disposition: `cre.enabled` stays false; the CRE, calibrator and targeted calibration remain as tested optional components and the report-ready negative result. No further CRE variants are run on seeds 100–149, 200–249 or 300–349. Remaining untested directions are listed, not started: value-of-information targeting, transfer-aware credit (C2d), and re-evaluation on a pilot-grounded S-I once real MATH/MBPP costs exist (placeholders limit any conclusion about the real system).

## D-83 — Phase B pilot and TRL spike results; model choice; horizon consequence

- Date: 2026-10-09 (Phase B)
- Owner: P1
- Evidence: `reports/pilot/phase_b_pilot.md` (Kaggle T4, Qwen2.5-0.5B and 1.5B, GSM8K/Countdown/noisy), `docs/TRL_SPIKE.md`.
- Decisions: (1) **Level 1 uses Qwen2.5-0.5B-Instruct** (v3 §4.6 rule). GSM8K is learnable (pass@1 0.20, pass@8 0.64, 64% mixed groups), memory is 1.8 GB, and it costs 0.54× the 1.5B per prompt. 1.5B stays the Level-2 model. (2) The wall-clock cost model is adopted: batch CV 0.00–0.07 on shared Kaggle T4s is under the 10% target, so the token-based fallback in `COST_MODEL.md` is not needed. (3) Countdown (pass@8 = 0 for both models) is the too-hard arm for now; its verifier is unchanged and its parse rate (11–13%) means format compliance is part of the failure. (4) TRL is pinned at 1.15.0 with the versions listed in `TRL_SPIKE.md`; GRPO runs must use at least 512 completion tokens.
- Consequences: (a) the three BUILT environments have no cost spread (0.94–1.05×), so the realistic current portfolio is the equal-cost regime (S-J's analogue) and the cost lever cannot be tested on real runs until MATH35 and MBPP exist (Phase E); S-I stays a placeholder for those two arms. (b) A GRPO step at the planned shape (P = 16, G = 8, 512 tokens) is estimated at 45–90 s with plain HF generation, so the v3 horizon rule (at least 60 rounds) at R = 5 would cost about 4–7.5 GPU-hours per run, far above the planning figure of about 1. The next measurement (real batch shape, vLLM availability) must precede any change to R, P or B; the likely lever is fewer prompts per round, not fewer rounds.
- Caveats: 64 prompts per environment (pass rates ±0.06); one seed; the Countdown failure was not audited at sample level.

## D-84 — vLLM works on T4; measured step time; compute plan; pilot sampling discrepancy

- Date: 2026-10-09 (Phase B)
- Owner: P1
- Evidence: second Kaggle run (`docs/TRL_SPIKE.md`, "Step-time and vLLM probe").
- Findings: (1) vLLM colocate runs on the T4 from a virtualenv; one optimizer step at P = 16, G = 8, 512 tokens takes 42.2 s with 6.1 GB peak memory. (2) 512 completion tokens give healthy training signal (3–11% truncated, about 80% of groups with gradient). (3) Discrepancy: the GRPO mean reward in the first steps (0.25–0.51, untrained model) is well above the pilot's GSM8K pass@1 of 0.20 for the same model. Hypothesis, not yet confirmed: the pilot's HF `generate` inherited Qwen's generation_config (`top_k = 20`, `repetition_penalty = 1.05`) while TRL samples with top-k off and no penalty. Both scripts now set `top_k = 0` and `repetition_penalty = 1.0`; the pilot's pass rates (and so the learnability numbers in D-83) are provisional until re-run.
- Compute plan consequence (estimate from the measured step): with R = 2 steps per round (32 prompts per round) a round is about 84 s plus calibration, so 60 rounds take about 1.4 GPU-hours; with R = 3, about 2.1. A 35-run main matrix at R = 2 is then about 49 GPU-hours, which two parallel T4 sessions can cover in roughly one week of quota (weekly quota still unverified). The v3 planning figure of about 1 GPU-hour per run was optimistic by 40–110%, not by the 4–7× feared in D-83. The calibration cost cap (10%) must be re-checked with a measured per-item evaluation time.
- Open: HF-backend step time (decides whether vLLM is needed at all), update-versus-generation split, per-item calibration time.

## D-85 — Phase E: MATH35 and MBPP environments

- Date: 2026-10-09 (Phase E)
- Owner: P2/P3 track, built by P1
- Decisions: (1) `math35`: MATH (DigitalLearningGmbH/MATH-lighteval) train levels 3–5 (5,582 problems with a gold `\boxed` answer; 4 of the 5,586 level 3–5 rows have none and are dropped), hash-split into train 4,982 / calib 300 / dev 300; sealed test = MATH-500 (500 items, all levels, so partly out-of-distribution by level). (2) `mbpp`: google-research-datasets/mbpp `full`: official train (374) → dev 60 (hash-ranked) + train 314; calib = official validation (90, small: pool it into the aggregate); sealed = official test (500); the 10 `prompt` rows are unused. (3) Answer checking for MATH is a self-contained conservative normaliser (no optional library) so rewards are identical on every machine; it never equates two different values (tested) and may reject exotic correct spellings. (4) MBPP prompt shows ONE assertion; the verifier runs all assertions (shown + hidden + challenge), so special-casing the shown test fails. (5) Sandbox success is decided by a random per-run sentinel printed after the last assertion, never by exit code; early `exit()` / `os._exit(0)` and forged sentinels fail (tested). Setup code runs after the solution (standard MBPP order). (6) `FiniteJsonlEnv` is a shared base for the new environments; `Gsm8kEnv` is deliberately untouched. (7) New config sections `data.envs.math35` and `data.envs.mbpp` (D-21 pattern); `rebuild_manifest_index` makes MANIFEST.sha256 cover every environment so builders cannot clobber each other.
- Evidence (`reports/env_profiles/phase_e_profile.json`, full data): MATH35 gold 100% over 5,582 rows, garbage 0%, 0 exact-text overlaps with the sealed set, verifier 0.4 ms; MBPP gold 100% over 464 rows, garbage 0%, verifier 113 ms mean (p95 130 ms) on Windows. Manifests are byte-identical on rebuild.
- Bugs caught by the tests and fixed in this phase: `\$12.50` left a stray backslash; `\text{yes}` was deleted as if it were a unit; MBPP setup code ran before the solution (mbpp-927 failed until reordered).
- Consequences: (a) MBPP's verifier cost (about 0.11 s per completion here, probably 30–50 ms on Linux) is a real cost driver: 128 completions per step serially add roughly 4–14 s to a 42 s step, so the Phase F reward wrapper must evaluate in parallel and the cost meter must attribute verifier time (it counts as GPU time by default, `COST_MODEL.md`). This is the first real source of cost spread other than completion length. (b) The sandbox is NOT a security boundary; run it only on disposable machines. (c) Pass rates and costs of both environments for the base models are unmeasured; the next Kaggle pilot covers them. S-I and S-J stay placeholders until then.

## D-86 — Phase E pilot outcome: portfolio passes on 0.5B; cost spread must be measured under vLLM

- Date: 2026-10-09 (Phase E pilot)
- Owner: P1
- Evidence: `reports/pilot/phase_e_pilot.md` (Kaggle T4, both models, all five environments, corrected sampling).
- Findings: (1) Acceptance rule, Qwen2.5-0.5B: 3 arms with pass@8 in [0.15, 0.85] (gsm8k 0.688, math35 0.344, mbpp 0.500) -> pass; Countdown 0/512 samples -> consistent with the too-hard criterion (pass@16 not measured directly); cost spread 1.45x under HF generation -> below the 2x target. (2) The D-84 diagnosis is confirmed: GSM8K pass@1 0.199 -> 0.344 after removing the inherited top_k / repetition penalty. (3) MATH35 truncates 61% at 512 tokens; its configured cap is 1024. (4) The TRL HF-backend smoke test failed only because Kaggle's torchao 0.10 is rejected by PEFT 0.20.
- Decisions: (a) **Level 1 stays on Qwen2.5-0.5B-Instruct**; the 1.5B escalation rule is not triggered (1.5B costs 1.7x more per prompt). (b) **The pilot's HF timings are not used as unit costs.** HF `generate` runs each batch until its longest completion ends, which compresses cost differences (3.4x token difference, 1.07x vs 1.32x measured cost). Unit costs for the portfolio rule and for S-I/S-J come from `scripts/kaggle_cost_probe.py`: vLLM generation (the training backend) in GRPO-step-sized chunks (16 prompts x 8), each environment at its own token cap, verifier time both serial and through a 4-thread pool. (c) The batch-CV column is not used for the H.7 repeatability gate (it mixes prompt content with noise); repeatability is judged on identical batches. (d) The sandbox applies its resource limits inside the child interpreter instead of `preexec_fn`, so verifiers can run from a thread pool (Phase F parallel verifier; `preexec_fn` is unsafe with threads); new test `test_concurrent_calls_from_threads_are_independent`. (e) Notebooks uninstall torchao before installing TRL; the step-probe notebook uses `virtualenv` (Python 3.13 image). (f) `notebooks/` is excluded from ruff (runner notebooks, not library code).
- Consequence: if the vLLM cost spread is still < 2x, the portfolio rule's cost criterion fails on real costs, and the heterogeneous regime needs a costlier arm (MATH35 at a larger cap, or the Level 2 multi-turn tool arm) — to be decided from the probe numbers, before any Tier-1 run.

## D-87 — Portfolio accepted on real costs; vLLM is the training backend; MATH35 is the expensive arm

- Date: 2026-10-09 (Phase E cost probe)
- Owner: P1
- Evidence: `reports/pilot/phase_e_cost_probe.md` (vLLM 0.31 cost probe for 0.5B and 1.5B, TRL smoke test, HF step probe).
- Findings: (1) Acceptance rule on Qwen2.5-0.5B passes: 3 learnable arms (pass@8 gsm8k 0.797, math35 0.500 at its 1024 cap, mbpp 0.483); Countdown 0/1,024 samples; cost spread 2.67x on generation + pooled verification. (2) HF backend: 83.5 s per GRPO step at P = 16, G = 8, 512 tokens, of which 52.0 s generation; vLLM colocate: 42.2 s (D-83). (3) TRL GRPO smoke test passes with torchao removed; reward function receives completion_ids and all dataset columns; groups are contiguous. (4) MBPP's verifier is 0.16 s per completion on Kaggle CPUs (1.3 s per prompt serially, 0.33 s with 4 threads).
- Decisions: (a) **Level 1 portfolio frozen**: `gsm8k` (learnable, cheap), `math35` (learnable, **expensive**: long answers, cap 1024), `mbpp` (learnable, cheap to generate, verifier-bound), `countdown` (zero-signal arm; it fills the roadmap's `too_hard` role, so no separate `too_hard` variant is built for Level 1), `noisy` (spurious-reward probe). The roadmap's expectation that MBPP is the expensive arm is corrected: MATH35 is. (b) **vLLM colocate is the training generation backend** (2x faster step); cost attribution uses the token-share model with the Gate 5 cross-check (Roadmap v3 §4.7). (c) The Phase F reward wrapper verifies through a thread pool (>= 4 workers; the sandbox is thread-safe since D-86 d). (d) Unit costs for the S-I/S-J refit come from this probe (gen + pooled verify), with the policy-update share treated as unknown: the refit is run at both ends of the bracket (update fully token-proportional, and half fixed), giving relative training costs MATH35 1.7-2.0x and MBPP 0.6-0.8x of GSM8K. The cost meter replaces the bracket once Gate 5 passes. (e) Per-run cost estimate updated to 1.4-2.8 GPU-h (60 rounds, R = 2), depending on how much MATH35 a method buys.
- Gate 1B' is marked passed: learnable arms, too-hard arm and cost spread all meet §4.1, with the cost spread measured on generation + verification and the update share pending Gate 5.

## D-88 — Refit of S-I / S-J from measured data, and its pre-registered evaluation

- Date: 2026-10-10 (written BEFORE any evaluation-seed run of the refit scenarios)
- Owner: P1
- Refit (`experiments/analysis/refit_scenarios.py` -> `configs/sim/scenario_si_r1.yaml`, `scenario_si_r2.yaml`, `scenario_sj_r.yaml`, `reports/analysis/refit_scenarios.md`): five arms matching the frozen Level 1 portfolio (D-87; no separate `too_hard`, Countdown is the zero-signal arm). Measured inputs: difficulty from pass@1 (0.5B, vLLM); rollout concentration kappa = 2.14, least-squares fit to the measured mixed-group shares (per-arm fits 1.5-2.6; the placeholder value was 25); unit costs in GPU-hours per prompt from generation + pooled verification + the policy update (42.2 s step), with the update either fully token-proportional (R1, learnable-arm spread 3.46x) or half fixed (R2, 2.06x); S-J-R = R1 dynamics at R1's mean cost; benchmark = macro over domains with a test set (gsm8k, math35, mbpp, countdown 0.25 each; noisy 0, Roadmap v3 4.8); Static uses real train sizes; noisy_q 0.35 (design value). Assumed, not measured: eta and the gsm8k<->math35 transfer 0.3 (kept from the placeholder S-I). Budget = ~70 rounds under Uniform (smoke check on tuning seeds 0-2 only: 70-71 rounds).
- Pre-registered evaluation (`experiments/refit_gate.py`, run ONCE): fresh evaluation seeds **400-449**; configurations frozen as in `configs/base.yaml` (D-68 tuning, **no retuning** on the refit scenarios, equal for all methods); calibration offered as in D-75 (paired items, churn 0.02, 100 items per slice, K = 10, charged per item at mean unit cost / G, to methods that use it only).
  - Methods: Uniform, Static, LP, Standard UCB, SEC-style, DUMP-style, Curator (base config: targeted calibration on), Curator without calibration.
  - **Primary P1 (Gate 2'' (i) on real costs):** Curator minus SEC-style and minus DUMP-style, paired 95% bootstrap CI excluding 0, in S-I-R1 and in S-I-R2 (4 comparisons, all must pass).
  - **Primary P2 (Gate 2'' (ii)):** S-J-R, Curator minus Uniform, lower CI bound > -0.01.
  - Secondary (reported, never used to select or change anything): Curator minus Uniform, Static, LP and Standard UCB per scenario; calibration on vs off; noisy-arm S5 detection and false-flag rates; mean allocation per arm after round 8; calibration cost share; concentration.
  - Interpretation written now: (a) P1 passes in both brackets -> the S-I loss was an artefact of placeholder costs/dynamics; (b) passes in R1 only -> the advantage depends on how the update cost is attributed, and Gate 5 (cost meter) decides; (c) fails in both -> the S-I result is robust to realistic costs, and the paper's claim narrows to the regime where cost and value are unrelated (S-C) plus calibration/attribution (Roadmap v3 3.3 negative-result plan). Whatever the outcome, no estimator, threshold, config or scenario parameter is changed in response; a new idea needs a new pre-registration and fresh seeds.

## D-89 — Outcome of the refit evaluation (D-88): primary criteria pass; Curator is not the best method

- Date: 2026-10-10. Run once on seeds 400-449 as pre-registered; nothing was changed after seeing the outcome.
- Evidence: `reports/gate2/refit_gate.{json,md}`; inputs `reports/analysis/refit_scenarios.md`.
- **P1 passes** in both cost brackets: Curator minus SEC-style +0.027 [+0.021, +0.033] (R1), +0.029 [+0.022, +0.035] (R2); minus DUMP-style +0.027 [+0.021, +0.032] (R1), +0.029 [+0.022, +0.036] (R2). **P2 passes**: S-J-R Curator minus Uniform +0.033 [+0.027, +0.039]. Pre-registered reading (a): the earlier S-I loss (D-69) was an artefact of the placeholder scenario.
- Secondary findings, reported with the same weight as the primary result:
  1. **Why SEC/DUMP-style lose:** they put 35-36% of the budget on the noisy arm (Uniform 20%). With the measured polarisation (kappa 2.14), real arms have few mixed groups, while a Bernoulli(0.35) reward always produces mixed groups, so a mean-|advantage| signal ranks the spurious arm highest. This is the spurious-reward failure mode the project targets, and it appears only once the simulator uses measured group statistics.
  2. **Curator is not the best method in these scenarios:** LP curriculum and Standard UCB (Curator's proxy without discounting or the status layer's constraints) beat Curator by 0.011-0.014 in all three scenarios, CIs excluding 0. Curator ranks third, ahead of Uniform, Static, SEC-style and DUMP-style.
  3. **Cost-awareness is not what produces Curator's gain here:** Curator's margin over Uniform is as large in the equal-cost twin S-J-R (+0.033) as in S-I-R1/R2 (+0.028/+0.030); the gain comes from starving Countdown (2% vs 20%) and spreading over the learnable arms.
  4. **Calibration still has no measurable effect:** Curator with vs without calibration differs by -0.004 [-0.010, +0.002] (R1), +0.004 [-0.002, +0.011] (R2), -0.002 [-0.008, +0.003] (S-J-R); the noisy arm reaches S5 in only 20-34% of seeds (false-flag rate 2.5-3.5%), and Curator still gives the noisy arm about 22% of its budget.
- Consequences: (a) Gate 2'' is met on the measured-cost scenarios (new row 2''-R in GATES.md); the original S-I row stays as it was, now explained. (b) The claim "Curator beats learnability bandits" holds in the simulator, but the mechanism is spurious-reward robustness of the LP/success-rate proxy, not cost normalisation, and simpler members of the same family (LP, Standard UCB) do better than full Curator. The paper must report this. (c) Open questions for a NEW pre-registration on tuning seeds only (not acted on here): why discounting/status cost Curator ~0.012 against Standard UCB under polarised groups; why S5 detection stays low with paired calibration; whether a different status or richness rule helps. (d) All conclusions still rest on assumed learning speeds (eta) and transfer; the Kaggle training runs (Phase F onward) are the real test.

## D-90 — Diagnosis of D-89 on tuning seeds: discounting is the main cost; the proxy and targeting have weaknesses

- Date: 2026-10-10. Tuning seeds 0-49 only. Evidence: `reports/analysis/refit_diagnosis.{json,md}` (2^5 factorial over discounting, cost normalisation, status layer, calibration and the richness weight, on the three refit scenarios; Standard UCB is reproduced exactly by the all-off cell), `reports/analysis/gamma_robustness.{json,md}` (gamma 1 vs 0.95 on the original suite).
- Findings:
  1. **Discounting is the main cost.** Main effect of gamma 0.95 vs 1: -0.020 (S-I-R1), -0.014 (S-I-R2), -0.013 (S-J-R). Curator with only gamma changed to 1 scores +0.041 / +0.019 / +0.013 above full Curator. With measured polarisation (kappa 2.14) per-round signals are noisy, and a ~20-round memory discards information that the slowly drifting refit worlds do not make stale. On the original suite gamma 1 shows no significant loss anywhere (S-A..S-J; the drift scenario S-E +0.001), and +0.012 in S-J.
  2. **Cost normalisation pays only once discounting is off:** main effect +0.016 (R1, 3.46x spread), +0.003 (R2), +0.000 (S-J-R, equal costs), as expected. The best R1 cell (gamma 1, cost on, status off, calibration off) beats Standard UCB by +0.043 by moving budget to MBPP, the cheapest learnable arm. This relies on the assumed (equal) learning speeds.
  3. **The proxy is partly fooled by the spurious arm.** Learning progress is hardly detectable within ~70 rounds (mean LP z 0.06-0.15 on learnable arms), so the proxy runs mostly on richness, and the noisy arm has the highest richness (SR 0.91). Its proxy x (0.46) exceeds MATH35's (0.44). Removing richness (beta 0) is not a uniform improvement (main effect +0.010 / +0.004 / 0.000 for keeping it), because LP alone is too noisy.
  4. **Targeted calibration rarely looks at the noisy arm.** With top-1 exposure targeting, the noisy slice is re-evaluated only ~0.6-0.7 times per episode after the baseline (GSM8K is usually the most-funded arm), ~4 ready windows per episode, median noisy z 0, S5 reached in 20-34% of episodes. Calibration's net effect is slightly negative (-0.002 to -0.004): it costs ~2% of the budget and rarely acts.
  5. **Concentration caveat.** gamma 1 raises Curator's mean max-weight on the refit scenarios from ~0.38 to ~0.67. The simulator has no forgetting or interference to punish concentration (D-67), so this part of any gain is optimistic for real GRPO.
- Not acted on here (each would need its own pre-registration): a calibration targeting rule that also covers arms whose richness is high but whose LP is flat; a richness term that discounts groups whose rewards are independent of the answer; a concentration cap.

## D-91 — Pre-registration H-gamma1: Curator without discounting

- Date: 2026-10-10, written BEFORE the evaluation run. Candidate: `configs/base.yaml` with only `scheduler.gamma` changed from 0.95 to 1.0 (selected in D-90 on tuning seeds; one change, nothing else retuned).
- Evaluation (`experiments/gamma_gate.py`, run ONCE): fresh seeds **500-549**; refit scenarios with calibration offered as in D-88; original suite as in Gate 2/2''.
- **Adopt gamma = 1 iff all pass:** A1 Curator-g1 minus Curator(gamma 0.95), paired 95% CI excluding 0, in S-I-R1, S-I-R2 and S-J-R; A2 Curator-g1 minus SEC-style and minus DUMP-style, CI excluding 0, in S-I-R1 and S-I-R2; A3 S-J-R Curator-g1 minus Uniform, lower bound > -0.01; A4 no regression on the original suite (S-A, S-B, S-C, S-D, S-E, S-F, S-H, S-I, S-J): Curator-g1 minus Curator lower bound > -0.01 everywhere.
- Secondary, reported only: Curator-g1 minus Standard UCB, LP and Uniform per refit scenario; allocations, concentration, noisy-arm S5 rate.
- If adopted: `scheduler.gamma` becomes 1.0, the scheduler is then a cost-normalised UCB with status layer and calibration, and H7 ("discounting helps under drift") is reported as not supported in the simulator. If not adopted: gamma stays 0.95 and the D-89 caveats stand. In both cases the concentration caveat (D-90 item 5) is carried into the GPU experiments.

## D-92 — Outcome of H-gamma1 (D-91): NOT adopted (A4 non-inferiority fails); gamma stays 0.95

- Date: 2026-10-10. Run once on seeds 500-549 as pre-registered. Evidence: `reports/gate2/gamma_gate.{json,md}`.
- A1 passes: Curator-g1 minus Curator +0.034 [+0.026, +0.042] (S-I-R1), +0.019 [+0.011, +0.027] (S-I-R2), +0.017 [+0.009, +0.024] (S-J-R). A2 passes: minus SEC-style / DUMP-style +0.048 to +0.063, all CIs > 0. A3 passes: S-J-R minus Uniform +0.052 [+0.044, +0.059].
- **A4 fails**: no-regression bound -0.01 is crossed in S-A (-0.003 [-0.012, +0.006]) and S-D (-0.004 [-0.012, +0.003]); all other original scenarios pass (S-E +0.008, S-H +0.010, S-J +0.006). The point estimates are small; the failure is a non-inferiority test without enough seeds to exclude a loss of 0.01 (CI half-width ~0.008-0.009 at 50 seeds). That is a power flaw in the D-91 design, recorded as such; the rule is applied as written.
- Decision: **gamma stays 0.95** in `configs/base.yaml`. Reported findings: removing discounting improves Curator substantially on the measured-data scenarios, and with it Curator would beat Standard UCB in S-I-R1 (+0.016 [+0.008, +0.025]) and tie it in S-I-R2 and S-J-R; it concentrates allocation (max-weight ~0.67 vs ~0.38).
- A follow-up is allowed only as a NEW pre-registration with a power analysis for the non-inferiority criterion (enough fresh seeds to resolve a 0.01 margin) and fresh seeds; it must not reuse seeds 500-549. Whether to do that, or to settle the question with real GPU runs, is an owner decision.

## D-93 — Calibration targeting by proxy claim (option 3 of D-92): design and selection rule, fixed BEFORE tuning

- Date: 2026-10-10. Motivation (D-90): targeted calibration evaluates the most-funded arm, usually GSM8K, so the noisy arm, whose proxy is carried by richness (SR 0.91) with flat learning progress, is re-checked less than once per run and S5 catches it in only 20-34% of episodes.
- New rules (`calib.targeting`, code in `calibration/calibrator.py`): `claim` ranks slices by the budget-weighted proxy claim of the window (sum of share x proxy signal, which the scheduler already accumulates as `_win_x`): where the proxy promises the most gain per window is where a wrong proxy wastes the most; `claim_stale` multiplies the claim by (1 + windows since that slice was last evaluated), so no funded arm stays unchecked. `exposure` is unchanged and remains the default until a rule is adopted. Unit tests added (`tests/unit/test_calibration.py`).
- Tuning experiment (`experiments/analysis/claim_targeting_tune.py`, tuning seeds 0-49, refit scenarios S-I-R1, S-I-R2, S-J-R with calibration offered as in D-88/D-75): five variants: `exposure/1` (current), `claim/1`, `claim/2`, `claim_stale/1`, `claim_stale/2` (`/n` = max_targets). Calibration cost per item is charged as before, so two targets cost twice.
- **Selection rule (written before the run):** among variants with noisy-arm false-flag rate <= 5% (mean fraction of the other arms ever in S5) AND calibration cost <= 10% of spend AND mean score over the three scenarios not below `exposure/1` by more than 0.005, pick the one with the highest noisy-arm S5 detection rate (mean over the three scenarios); ties within 0.02 go to the cheaper variant. If none beats `exposure/1` on detection by at least 0.10 absolute, no change is proposed.
- The chosen variant (if any) is then evaluated once on fresh seeds under its own pre-registration (D-94). Calibration targeting is the only thing varied; gamma stays 0.95.

## D-94 — Tuning outcome (D-93) and pre-registration H-claim: calibration targeting `claim_stale` with 2 targets

- Date: 2026-10-10, written BEFORE the evaluation run. Tuning (seeds 0-49, `reports/analysis/claim_targeting_tune.md`): by the D-93 rule the chosen variant is `claim_stale/2`: noisy-arm S5 detection 0.78 vs 0.27 for `exposure/1`, false-flag rate 0.017, calibration cost 3.4% of spend (2.3% before), mean score +0.005 on tuning seeds. `claim/1` alone barely changes detection (0.28): staleness and the second target do the work.
- Candidate: `configs/base.yaml` with `calib.targeting: claim_stale`, `calib.max_targets: 2`; nothing else changes (gamma stays 0.95). The original scenario suite offers no calibration, so the change is inert there by construction (no no-regression test is needed; D-92's power flaw does not arise).
- Evaluation (`experiments/claim_gate.py`, run ONCE): fresh seeds **600-699** (100 seeds, so a 0.01 margin can be resolved: refit CI half-widths are ~0.0065 at 50 seeds), refit scenarios S-I-R1, S-I-R2, S-J-R with calibration offered as in D-88.
- **Adopt iff all pass:** C1 detection: noisy-arm S5 rate of `claim_stale/2` minus `exposure/1`, paired over seeds (difference in the binary indicator), 95% bootstrap CI lower bound > 0, in each of the three scenarios; C2 safety: mean false-flag rate <= 5% and calibration cost <= 10% of spend, in each scenario; C3 non-inferiority: Curator score with `claim_stale/2` minus with `exposure/1`, lower CI bound > -0.01 in each scenario; C4 Gate 2'' retained: `claim_stale/2` Curator minus SEC-style and minus DUMP-style, CI lower bound > 0 in S-I-R1 and S-I-R2, and S-J-R minus Uniform lower bound > -0.01.
- Secondary, reported only: score difference (not required to be positive), noisy-arm budget share, Curator minus Standard UCB and LP, and the same candidate combined with gamma = 1 (`curator_g1_cs2`), reported as exploratory with no decision attached.
- If adopted, `configs/base.yaml` takes the new targeting and calibration's cost share is re-reported; if not, `exposure/1` stays and the D-90 finding stands. Either way the claim in the paper is limited to detection and safety; any effect on score is reported as measured.

## D-95 — Outcome of H-claim (D-94): ADOPTED. Calibration targets by proxy claim with staleness, 2 slices per window

- Date: 2026-10-10. Run once on seeds 600-699 as pre-registered. Evidence: `reports/gate2/claim_gate.{json,md}`.
- **All four criteria pass.** C1 detection: noisy-arm S5 rate 0.86 / 0.81 / 0.78 vs 0.25 / 0.32 / 0.31 with `exposure/1` (differences +0.61 [+0.51, +0.71], +0.49 [+0.38, +0.60], +0.47 [+0.35, +0.58]). C2: false-flag rate 0.020 / 0.025 / 0.022 (<= 0.05), calibration cost 3.3-3.4% of spend (<= 10%). C3: score +0.006 [+0.002, +0.010], +0.004 [-0.000, +0.007], +0.006 [+0.002, +0.010] (non-inferiority margin -0.01 met with room to spare). C4: Gate 2'' retained (Curator minus SEC/DUMP-style +0.029 to +0.033, S-J-R minus Uniform +0.035).
- Decision: `configs/base.yaml` now has `calib.targeting: claim_stale`, `calib.max_targets: 2`. The `Curator` default is the adopted configuration; `exposure/1` remains available as an ablation. gamma stays 0.95.
- What this does and does not show: S5 now catches the spurious arm in about 80% of episodes with few false flags (D-90 root cause 'targeting rarely looks at the noisy arm' is fixed). The effect on the noisy arm's budget share is modest (0.23 -> 0.19) because S5 only halves an arm's weight (E.5 soft shrink), and the effect on score is small (+0.004 to +0.006). The spurious-reward detection claim (H4) is therefore supported in the simulator, while 'calibration improves final score' is supported only weakly. Curator (cs2) is still not better than Standard UCB or LP (-0.005 to 0 against UCB, -0.011 to -0.003 against LP); it is level with UCB in S-I-R2 and S-J-R.
- Exploratory, no decision attached: with gamma = 1 as well (`curator_g1_cs2`) Curator beats Standard UCB in S-I-R1 (+0.023 [+0.016, +0.030]) and ties it elsewhere, but its false-flag rate rises to 0.03-0.065 and concentration rises (D-92); gamma = 1 stays a candidate for a powered pre-registration or a GPU arm.
- Open (not acted on): S5's response is only a halving of the arm's weight; whether a harder response to a calibration-confirmed spurious arm helps is a new pre-registration on fresh seeds (>= 700).

---

# Audit and Phase F runner (2026-10-10, branch `audit-fixes-and-gpu-run`) — PROPOSED, pending owner confirmation

## D-96 — Audit fixes adopted as defaults; legacy behaviour kept behind switches

- Date: 2026-10-10. Evidence: `../errors_found.md` (27 + 1 errors, most reproduced by `../audit/verify_errors.py`), `../CHANGES_AND_IMPACT.md`, `tests/unit/test_audit_fixes.py`.
- `configs/base.yaml` now defaults to the corrected behaviour: `signals.lp_se: design` (E-01), `calib.flag_scope: evaluated` (E-02), `calib.credit: transfer` (E-03), `scheduler.pull_unit: delivered` (E-09), `data.envs.countdown.exact_division: true` (E-19); new switches with legacy defaults: `scheduler.s5_cap: none`, `baselines.dump.window_rounds: 0`; dead keys removed (`calib.target`, `warmup.total_rounds`, E-23). Verifier fixes (E-06, E-07, E-08) have no switch: they were wrong.
- Every pre-registered gate/study script from D-68 to D-94 now loads `configs/frozen/d68_d94.yaml` (exact config of commit 30052e8 + legacy switches). Re-running the D-94 gate on seeds 600-699 reproduces its committed report exactly (all scores, criteria and the adoption verdict).
- E-28 (status evidence thresholds tied to the batch shape) is fixed by `core.config.rescale_for_batch_shape`, applied by the GPU runner; `base.yaml` stays at the simulator shape.
- Consequence for earlier decisions: D-40 and D-42 were taken with the inflated LP SE and must be re-studied; D-95's detection rates do not hold at the GPU shape (0% pre-audit; 10-28% after the fixes; `reports/analysis/audit_impact.md`). H4 is not achievable by benchmark calibration at the Level-1 budget.

## D-97 — Pre-registration draft (PROPOSED, not run): GPU-shape confirmation on fresh seeds 700-799

- Scenarios: `scenario_si_r1_gpu.yaml`, `scenario_si_r2_gpu.yaml`, `scenario_sj_r_gpu.yaml` (R = 2, P = 16, ~60 rounds, delivered learning, carry rounding, shared noisy slice, generation-only calibration cost).
- Methods, each at its GPU-shape-tuned point (`reports/sweeps/tune_gpu_shape.md`, tuning seeds 0-39, equal budget, D-55 rule), all with `rescale_for_batch_shape(cfg, 32)`: Curator (gamma 1, kappa 0.1, tau 0.3, K 10), LP (tau 0.5), Standard UCB (kappa 0.25, tau 0.3), SEC-style (alpha 1, tau 0.4), DUMP-style (c 0.3, tau 0.4, window 10), Uniform, `pilot_uniform` (B10).
- Primary metric: noiseless final score (E-10), paired by seed; 95% t-intervals (n = 100) and Holm across the co-primary comparisons.
- Primary: (P1) Curator - LP > 0 in S-I-R1-GPU (the large-spread regime); (P2) Curator - SEC-style and - DUMP-style > 0 in S-I-R1-GPU and S-I-R2-GPU; (P3) S-J-R-GPU Curator - Uniform lower bound > -0.01. Secondary: everything else, noisy-arm budget share, detection, false flags, calibration cost.
- Interpretation fixed now: P1 pass -> the cost lever is real at the GPU shape when the spread is large; P1 fail with P2 pass -> Curator is a learnability-robust bandit but not better than the LP curriculum, and the paper says so. Detection results are reported as measured; no claim of spurious-reward detection by calibration at this budget unless detection >= 0.5 with false flags <= 5%.
- Run once with `experiments/analysis/audit_impact.py`-style code on seeds 700-799 after the owner confirms this entry; nothing above may change after seeing results.

## D-98 — Reward audit, realism v2 and the GPU preset (exploratory; tuning seeds 0-39 only)

- Date: 2026-10-12. Evidence: `reports/analysis/component_study_gpu2.md`, `reports/sweeps/tune_gpu2.md`, `reports/analysis/heldout_gpu2.md`; tests `tests/unit/test_reward_audit.py`, `test_proxy_fit.py`, `test_drift_and_cap.py`, `test_sim_realism.py`.
- **Reward audit (new, `signals/audit.py`).** A label-free, zero-cost check of whether an arm's reward depends on the prompt: the dispersion ratio D of group pass rates (1 for a reward that ignores the prompt; (G + kappa)/(kappa + 1) under prompt heterogeneity), pooled over rounds from moments every observation already carries. Real data: on the Kaggle pilot every verifier arm has D = 3.0-4.4 and the random-reward arm 1.0-1.09 (both model sizes). Threshold `d_max` 1.5 chosen by Monte Carlo (flag rate 99.8-100% for a random reward at median round 10-11 of 60; 0-0.2% for kappa <= 5; 0.2-3% for kappa = 10). An audit flag puts the arm in S5 (priority over S1; leaves when the audit clears). Blind spot, stated: a reward that depends on the prompt but rewards the wrong thing; that remains calibration's job. Caveat: the placeholder scenarios' kappa = 25 (no real arm measured near it) would be flagged in about half the runs.
- **Realism v2 scenarios** (`configs/sim/*_gpu2.yaml`, generated by `refit_scenarios.py --shape gpu2`): GPU shape plus learning from the mixed-group share at the measured kappa (B8), repeated passes over a finite dataset teach less (E-14; repeat_decay 0.5 ASSUMED), forgetting 0.002 per untrained round (B7, ASSUMED), reports on a separate random stream and paired test items (B1/E9), vectorised draws (E10), pilot pass@8 per arm (B10). Stress variants: a misaligned arm (prompt-dependent reward, no benchmark value) and random-reward transfer of -0.2 / +0.2 to GSM8K (B9). Every switch defaults to the original world: 210 of 210 fingerprinted episodes (frozen and base configs, 7 methods, 5 scenarios, 3 seeds) are bit-identical to commit b259f9c.
- **Component study** (Curator at its earlier GPU-shape point, components added one at a time): the audit + S5 floor cap gains +0.007 to +0.024 absolute over Curator in all six scenarios (every 95% interval above 0) and cuts the noisy arm's budget from 0.27-0.35 to 0.06; the alpha/beta fit (B4), drift-triggered discounting (B2), weight cap and smoothing (B7) show no measurable benefit and stay OFF (built, tested, available). B2 on S-E: gamma 1 + drift minus gamma 0.95 = -0.002 [-0.006, +0.001]. All audit S5 entries on real arms: zero; the remaining false flags come from benchmark calibration (requiring 3 consecutive mismatch windows halves them).
- **Fair re-tuning on realism v2** (8 configurations per method, D-55 rule): Curator (gamma 1, kappa 0.25, tau 1.0, audit, `mismatch_windows` 3) +11.6% over Uniform; LP +5.9%; UCB +4.0%; SEC +1.2%; DUMP +1.2%. Held out (S-J-R-GPU2) and stress scenarios: Curator +5.4% to +12.3%, LP +3.4% to +6.5%, `pilot_uniform` +2.7% to +4.5%; noisy arm detected in 100% of runs, false flags 0-5%. Curator without calibration scores +0.5 to +0.9 points higher still: at the Level-1 budget calibration measures but does not pay for itself in score.
- **Adopted for the Kaggle preset `gpu_tuned` only** (exploratory GPU runs); `configs/base.yaml` keeps every new switch off until a pre-registered confirmation (D-99). The previous preset stays as `gpu_tuned_v1`.
- Run robustness (no scheduler effect): per-round checkpoints, background artifact sync with resume, verifier/evaluation failure isolation, supervised restarts, clean stop before the session limit (`kaggle/README_KAGGLE.md`).

## D-99 — Pre-registration draft (PROPOSED, not run): realism-v2 confirmation on fresh seeds 700-799 (supersedes the D-97 draft)

- Scenarios: `scenario_si_r1_gpu2.yaml`, `scenario_si_r2_gpu2.yaml`, `scenario_sj_r_gpu2.yaml` (primary); `_misaligned`, `_noisy_neg`, `_noisy_pos` (secondary).
- Methods at their `reports/sweeps/tune_gpu2.md` points, `rescale_for_batch_shape(cfg, 32)`: Curator (preset `gpu_tuned`), Curator without calibration, LP (tau 0.75), UCB (kappa 1.0, tau 0.3), SEC-style (alpha 0.1, tau 0.4), DUMP-style (c 2.0, tau 0.4, window 10), Uniform, `pilot_uniform`.
- Primary metric: noiseless final score (E-10), paired by seed, 95% t-intervals (n = 100), Holm over the primary family. Primary: (P1) Curator - LP > 0 on each of the three primary scenarios; (P2) Curator - `pilot_uniform` > 0 on each; (P3) noisy-arm detection >= 0.9 with real-arm false flags <= 5% (audit and calibration combined). Secondary: everything else, calibration on/off, budget shares, MBPP passes.
- Interpretation fixed now: P1 + P2 pass -> the audit-equipped Curator beats the strongest baseline and the practitioner's baseline at the GPU shape in simulation; the GPU runs then test transfer to a real model. P3 fail -> the audit's thresholds do not hold up and its claim is withdrawn. Calibration is reported as measured; no claim of misaligned-reward detection at this budget.
- Run once, after the owner confirms this entry; nothing above may change after seeing results. D-97 is withdrawn (its scenarios and methods are superseded; it was never run).

## D-100 — ROI leaderboard and leave-one-out (D3); LP window (B3); in-run ROI is descriptive at Level 1

- Date: 2026-10-12. Evidence: `reports/analysis/roi_loo_sim.md`, `reports/analysis/lp_window_gpu2.md`; tests `tests/unit/test_roi_loo.py`, `tests/integration/test_trainer_fake.py` (LOO end to end).
- ROI (`roi/leaderboard.py`): per-slice weighted ridge on calibration spans (exposure accumulated per slice between its evaluations, so targeted calibration is handled), benchmark-weighted, divided by measured GPU-hours per prompt. Default structure `members`: an arm is credited only on the slice it is scored on, jointly with arms sharing that slice. The `full` transfer matrix needs the sysid design (every slice, every window).
- Leave-one-out (`evaluation/loo.py`, `run.py loo`): retrain without an arm at the same budget; the benchmark is unchanged. Ground truth for an arm's value: LOO under Uniform; LOO under an adaptive method also measures how well that method uses the arm.
- Simulator result (Curator preset, 20 tuning seeds): `members` ROI vs LOO-under-Uniform, pooled over seeds, Kendall tau +0.60 (S-I-R1-GPU2) and +1.00 (S-J-R-GPU2, H5 pass); one run at a time, median tau 0.4 / 0.6 and H5 passes in 10% / 30% of seeds. `full` structure: tau -0.20 / +0.60 pooled, about 0 per run. On S-I-R1-GPU2, MATH35 at 2x cost is genuinely worth less than nothing in an equal mixture, so the "junk arms in the bottom two" rule fails for a real reason.
- Consequence: a one-seed GPU ROI table and H5 verdict are descriptive only (the report says so); H5 needs several seeds or the pooled sysid design.
- B3: LP window W in {10, 15, 20, 30} with the preset: no window beats W = 10; W = 30 is worse on S-I-R1-GPU2 (-0.0039 [-0.0074, -0.0003]). W stays 10.

## D-101 — Evaluation protocol 2: sampled accuracy, per-item pairing, base-model reference (GPU runs)

- Date: 2026-10-20. Owner approved the same day, BEFORE any protocol-2 number existed (this entry is the pre-registration of the GPU measure). Evidence: Kaggle seed 0 (curator, uniform, lp; protocol 1).
- Why. Seed 0 tied on greedy dev macro (0.308 / 0.302 / 0.301) while training clearly moved the policy: entropy 0.49 -> 0.33, and sampled training accuracy rose under Curator (GSM8K 0.46 -> 0.53, MBPP 0.26 -> 0.33 first vs second half) but not under Uniform. Early RL sharpens the sampling distribution; the greedy answer moves little. Greedy macro also had about +-0.018 evaluation SE per run (MBPP dev has ~60 items), and there was no untrained reference.
- Protocol 2 (reports only; training is unchanged, `test_sampled_evaluations_never_change_the_run_they_measure`):
  - every report scores greedy AND sampled accuracy: k = 4 samples per item at the training temperature (1.0) with the training sampler (top_p 1, top_k 0, min_p 0), one seed per item from the item alone (the same for every method, seed and model);
  - per-item results are logged; methods are compared item by item on the same items (paired SE), per seed, then across seeds (t-interval, sign-flip);
  - the untrained model (`base_s0`, `run.py compare` runs it first) is evaluated with the same engine (vLLM, the training dtype), giving the gain of every method;
  - periodic reports every 10 rounds on a fixed random 100 of the final 300 items per domain (previously the 100 lowest-index items, not a random sample), so learning curves and the final score share their items; the compute measure is GPU-hours to reach the reference's final level.
- **Primary GPU measure (H1/H2 on hardware): sampled macro, item-paired against Uniform, across seeds.** Greedy macro is reported alongside as secondary. Protocol-1 results are kept (`summary_v1.json`); finished runs are re-evaluated from their final checkpoint, without retraining.
- Fixed with it: a relaunched finished run rewound its logs to the last checkpoint and dropped its final-evaluation row (restored from the backup on the next fetch); the ROI table says when it is not a ranking (no unflagged arm's interval excludes zero) and marks S5 arms.

## D-102 — Research-informed changes and the pilot plan (pre-registered before any pilot run)

- Date: 2026-10-21. Owner request ("add all the research-based improvements"); written before any of the runs below exist. Evidence: the protocol-2 result (D-101: Curator vs Uniform, sampled, 3 seeds, -0.0013 [-0.0122, +0.0096]; every mixing method ties; 35-50% of GRPO groups per step carry no reward signal in every method) and the literature in `report.md` §6.
- Built (training of the existing methods unchanged; their runs stay valid):
  - **Question-level filtering, "_q" methods** (`trainer/group_filter.py`): a step draws ceil(1.75 P) questions, generates and scores them all (every token charged to the budget), and trains on P groups, informative ones first (online difficulty filtering, Bae et al. 2026; DAPO dynamic sampling). vLLM's concurrent-sequence cap is widened by the same factor for these runs only. `uniform_q` isolates filtering; `curator_q` = Curator + filtering + the pre-flight gate.
  - **Pre-flight verifier check** (`signals/verifier_fuzz.py`): each arm's reward path scores fixed junk answers (empty, echoed prompt, empty/wrong box, number spray, code that exits early or prints its globals) on 6 training prompts before training; > 5% junk reward, or one probe paid on half the prompts, flags the arm (fuzzing RLVR verifiers, 2026). Result on our arms: gsm8k, math35, mbpp, countdown 0.00 (MBPP's sandbox resists early exit and global leaks); noisy 0.30 and hackable 1.00 on number spray: flagged. With `signals.reward_audit.preflight_gate` (curator_q only) a flagged arm is S5 from the first round boundary, a sticky flag apart from the evidence-based audit flag.
  - **Hackable arm** (`envs/hackable.py`, stress portfolio only): GSM8K prompts whose checker pays for the reference number anywhere in the response; its reward still depends on the prompt, so the in-run audit cannot see it, the pre-flight check does.
  - **Second model family:** `run.py --namespace` gives another model or portfolio its own output, results and backup prefix (run names never collide). Default second family: OLMo-2-0425-1B-DPO (open data, ungated, before OLMo's own RLVR stage); Llama-3.2-1B-Instruct optional (licence approval needed).
  - **pass@k** (k = 4, from the stored samples) in the comparison: coverage vs sharpening (Yue et al. 2025).
  - Notebook steps for all of the above and for the SEC/DUMP baselines (built earlier, never run on GPU).
- Not built, and why: GRESO (skips questions using an earlier epoch's outcome; our runs see under half an epoch); TAC's transferability term (per-domain gradient projections inside TRL: heavy, later); a higher LoRA learning rate (1e-5 is in the recommended range; changing it would break comparability with the 13 finished runs); entropy bonus (not in TRL's GRPO).
- **Pilot session** (`STEPS = ["pilot_q", "baselines", "olmo", "stress"]`, one seed each, about 9 GPU-hours), primary metric as D-101 (sampled macro, item-paired, 1.5 GPU-hours), secondary GPU-hours to reach Uniform's final level:
  - H-A (pilot_q): curator_q and uniform_q vs uniform_s0 (Qwen). Pays off if either is >= +1.5 points AND >= 2 paired SE above Uniform, or reaches Uniform's final level with >= 20% less compute.
  - H-B (olmo): on OLMo-2-1B, Uniform (which trains on the random-reward arm) vs Curator. Pays off if Curator is >= 2 paired SE above Uniform. Also reported: each method's gain over the untrained OLMo.
  - H-C (stress): with the hackable checker, Curator-Q vs Uniform. Pays off if Curator-Q is >= 2 paired SE above Uniform; whether the bug hurts Uniform is read against uniform_s0 of the main portfolio.
  - Baselines: SEC and DUMP seed 0 against Uniform, descriptive.
- Decision (24 Oct, new quota): a hypothesis that pays off gets seeds 1 and 2 (then 3-5 if its interval still crosses zero); none does: leave-one-out and the negative-result write-up. One seed is never reported as a result on its own.

## D-103 — New thesis (robustness), Curator-lite, confirmation runs (pre-registered before any of them)

- Date: 2026-10-21. Owner request. Evidence: the D-102 pilots, seed 0, sampled accuracy item-paired against Uniform: SEC -2.3 (SE 0.6), DUMP -1.5 (0.7), uniform_q -2.1 (0.6); curator_q -0.3 (0.6), and curator_q vs uniform_q +1.8 (0.6). SEC put 34% and DUMP 31% of their questions into the random-reward arm (Uniform 20%, Curator 6%). Greedy scores disagree in direction for SEC and DUMP (+1.3, +2.0, SE about 1); greedy is reported alongside everywhere. H-A (filtering raises accuracy) failed its D-102 bar; the claim "Curator beats Uniform" is dropped (D-101: -0.1 [-1.2, +1.0] over 3 seeds).
- **Thesis:** at small scale the data mix does not move accuracy (explained in `self_critic.md`); selectors that steer on learning signals (SEC, DUMP, variance-based question filtering) are pulled into a broken reward source and lose accuracy; Curator checks whether a reward depends on the prompt and is not pulled in.
- **Curator-lite = `curator_nocal`** (existing method: Curator without calibration, which cost 2-3 rounds and returned nothing measurable). No new code; seed 0 exists. Calibration, the ROI leaderboard, the transfer matrix and the per-token cost model are reported as measured and not paying off at this scale; the cost model is not changed (that would be a new, untested method).
- **Runs** (session hours; two runs in parallel on T4 x2): `confirm` = sec, dump, uniform_q, curator_q seeds 1, 2 (~8 h); `lite` = curator_nocal seeds 1, 2 (~2 h); `olmo` = curator_nocal and uniform on OLMo-2-1B, seed 0 (~2.5 h; changed from D-102's curator before any OLMo training result existed, since both D-102 OLMo training runs crashed at round 0); `clean` = clean_uniform (equal mix without countdown and noisy: the ceiling of a perfect junk detector) seeds 0, 1 (~2 h). Nothing else is added.
- **Primary measure** (as D-101): sampled macro accuracy, item-paired against the comparison method, per seed; across seeds the mean of the per-seed differences with its SE (sd / sqrt(3)) and the 95% t-interval (df 2). Greedy, pass@k and the share of compute on the random-reward arm are reported for every run.
- **Hypotheses and success rules** (3 seeds: seed 0 from D-102 plus seeds 1, 2):
  - R1 (primary): curator_q > uniform_q. Holds if all 3 seeds have the same sign and the mean is at least 2 across-seed SE.
  - R2: SEC < Uniform and DUMP < Uniform, each by the same rule. (Seeds 1, 2 of Uniform exist.)
  - R3 (mechanism): SEC and DUMP spend more than Uniform's 20% of questions on the random-reward arm, Curator variants under 13%. Descriptive.
  - R4: Curator-lite vs Curator, 3 seeds each: the simplification holds if the difference is within +-1 point.
  - OLMo (1 seed) and clean_uniform (2 seeds): descriptive; OLMo is reported either way.
- With 3 seeds, "mean >= 2 SE" is evidence at roughly one-sided p 0.1, not proof; the t-interval is reported next to it and the write-up says so.
- Record (21 Oct): the OLMo run started from a notebook with D-102's step, so it is full Curator vs Uniform on OLMo-2-1B, seed 0 (D-102 H-B as pre-registered), not Curator-lite. With the vLLM compile-hook fix, OLMo trains; about 6.5 min per round, so about 14 rounds per 1.5 GPU-hour run. It is reported as H-B; no Curator-lite OLMo run is added.
- **Story decision on the analysis day** (after `confirm` and `lite`), not later: R1 and R2 hold, the robustness thesis leads; otherwise the fallback (rigorous negative result with its mechanism, plus the deterministic pre-flight check and the in-run audit as the contribution). No change of story after that day.

## D-104 — This branch adopted as the base; parallel local work archived (L-96 … L-101)

- Date: 2026-10-22. The owner chose this branch (PR #1 merged into `main`, commit 24f29bd) as the base over uncommitted local work done on top of `bd9099e` in parallel. Where both built the same thing (GRPO runner, ledger, checkpoints, Kaggle package, reward audit, batch-shape evidence), this branch's version is kept and the local one dropped. What is unique is archived in `docs/archive/local-2026-10/` with local numbers L-96 … L-101 (they collided with D-96 … D-103 here); the local code is kept outside the repository in `RL_project_uncommitted_backup_2026-10-22.zip`.
- Findings from the local work that agree with this branch (independent confirmations, nothing to change): status evidence cannot reach `n_min` at the GPU batch shape (local: discounted evidence tends to 6.4/(1-0.9) = 64 per arm at R = 2, P = 16, never reaching 64; here E-28 / `rescale_for_batch_shape`); D-95's detection does not survive a shared noisy slice (L-97: 0.78 -> 0.07 with own-slice credit, the learning GSM8K arm wrongly flagged in 11% of runs; joint credit 0.15 / 4%; here D-96); a learning rate of 1e-5 (L-101: the pre-registered pilot on tuning seed 99 could not separate 1e-5, 3e-5, 1e-4 in 18 steps, tie broken to 1e-5; here D-102 keeps 1e-5).

## D-105 — Pre-registered label-based reward audit (L-98): negative; superseded by the label-free audit (D-98)

- Date: 2026-10-22 (the test itself ran on 2026-10-11). Evidence: `docs/archive/local-2026-10/` (L-98 pre-registration, frozen before any run; L-99 outcome; reports).
- Design: an opt-in audit correlating each arm's own reward with the slice verifier on answers the calibration evaluation had already produced (phi over unique (item, answer) pairs; invalid iff the one-sided 95% upper bound < 0.3 with >= 100 pairs; operating characteristics fixed before any experiment, including repeated looks).
- Outcome (Stage T, tuning seeds 0-49, noisy arm on the shared GSM8K slice): the audit caught the noisy arm in 98% of runs, but neither candidate met the pre-registered false-flag limits (GSM8K wrongly flagged 0.10 / 0.17 against a 0.05 limit), so Stage E was never run and seeds 700-799 of that protocol were never used. A diagnosis on the same tuning seeds attributed every GSM8K false flag to benchmark-gain calibration, none to the audit; an audit-only arm detected 0.97 with no false flags, which motivated proposal L-100 (never frozen).
- Relation to this branch: the label-free dispersion audit of D-98 needs no verifier and no calibration answers and reaches the same target (noisy arm detected in all runs, real-arm audit flags zero), and D-103 already treats calibration as not paying off at this scale. L-100 is therefore not pursued; it stays archived in case a label-based check is wanted for rewards that depend on the prompt but reward the wrong thing (D-98's stated blind spot, now covered by the pre-flight check of D-102).

## D-106 — Scheduler checkpoints now include the latest signal vectors (bug fix)

- Date: 2026-10-22. Found by the local work (L-96) and confirmed on this branch before the fix.
- Bug: `DiscountedUCB.get_state` did not save `_latest_vectors`, which the status layer reads at the next decision. A scheduler restored from a checkpoint therefore skipped its status constraints for one round: in the replay test, Countdown (status S4, too hard) got weight 0.02004 instead of being capped at the floor 0.02. This branch restores from per-round checkpoints after every lost session (D-98 run robustness), so every resumed Curator run had one round without status constraints; effects on finished runs are at most one round per resume and are not re-run.
- Fix: the vectors are saved with the status as its string value and rebuilt on load (`scheduler/ducb.py`); old checkpoints without them load as before. New test `tests/unit/test_scheduler_checkpoint.py` replays recorded simulator observations through all seven schedulers, snapshots through JSON mid-run and requires identical mixtures after resume. Full suite: 478 passed, 44 skipped.

## D-107 — Roadmap v4 (PROPOSED, pending owner confirmation)

- Date: 2026-10-24. Document: `docs/ROADMAP_v4.md`. Written after the D-103 pre-registration and before any D-103 confirmation run; it does not change D-103's story rules.
- Core kept fixed: outer-loop mixture only; discounted UCB on (alpha LP' + beta SR) / measured cost with floor and S1-S5; reward-health layers act on the mixture, never on the reward; one budget ledger; pre-registration. Every addition is a proxy feature, a status trigger or a charged trainer step, off by default.
- Phases, each with its own pre-registration written before its first run: 0 finish D-103 (plus the clean-ceiling decision rule); 1 fault zoo F1-F6 and a key-consistency audit, measured first on a frozen completion bank (D-108, sim seeds 800-899; GPU D-109); 2 health-gated sequential question allocation (D-110); 3 per-arm gradient-sketch transfer feature and a TAC-lite baseline (D-111, sim seeds 900-999); 4 seed count from the measured across-seed SD, second model, budget dependence (D-112); 5 multi-turn tool-call arm for the cost lever (D-113); 6 `curator-health` command and adapters.
- Exploratory input (tuning-free, from the measured 0.5B mixed-group rates; Monte Carlo, not a GPU result): the D-102 filter lifts the random-reward arm from ~18% of drawn questions to ~30% of trained groups under Uniform. To be checked against `uniform_q_s0` logs in Phase 0.

---

# Roadmap v3 decisions (PROPOSED — pending owner confirmation; see `docs/ROADMAP_v3.md` §11)

Status: proposed, not yet adopted. Each becomes final when the owner confirms it here with a date.

- **D-52** Claim reframed to gain per GPU-dollar + calibration + validated ROI; two cost regimes (v3 §3).
- **D-53** Portfolio v3: six Level-1 arms (GSM8K, MATH35, MBPP, Countdown, noisy, too-hard); `kk` and `toolcall` in Level 2 (§4.1).
- **D-54** SEC-style and DUMP-style baselines promoted to MVP; the "needs trainer internals" rationale is withdrawn (§4.4).
- **D-55** Equal tuning budget for every adaptive method (§4.4).
- **D-56** Horizon rule: ≥ 60 rounds, ≥ 12 calibration windows, calibration ≤ 10% of budget (§4.2).
- **D-57** Calibration uses paired items and is sized by a power study (§4.3).
- **D-58** Five seeds for the main matrix; Holm correction on H1/H2 (§4.8, §8).
- **D-59** Second model family for the noisy probe is in the MVP (§4.6).
- **D-60** Generation backend chosen in the TRL spike; vLLM allowed before Gate 5 (§4.7, §4.9).
- **D-61** Size-proportional static mixture pre-registered (implemented in the simulator as D-63).
- **D-62** Single owner; dependency-ordered phases; milestone ladder L1/L2/L3 (§6).
