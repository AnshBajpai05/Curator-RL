# CURATOR — improvements required

> **Status (2026-10-12, branch `audit-fixes-and-gpu-run`): every code item is done except B11 and E12.**
> - Done: A1 (shared slice + transfer credit; random-reward detection now comes from the new reward audit, 100% at the GPU shape in simulation), A2, A3/D2 (attribution, ledger, NNLS cost model; Gate 5 itself needs the GPU), A4, B1 (noiseless score, separate report stream, paired test items), B2 (drift-triggered discounting), B3 (studied: W = 10 stays, `reports/analysis/lp_window_gpu2.md`), B4 (alpha/beta fit), B5, B6 (calibration thresholds studied with the audit on: `mismatch_windows` 3), B7 (forgetting in the world; weight cap and smoothing in the scheduler), B8, B9, B10 (`pilot_uniform`), C2, C3, D1, D3 (ROI leaderboard + leave-one-out + H5 check), D4 (= B4), D5 (figures), E1-E11.
> - Studied and kept OFF (no measurable benefit on tuning seeds): B2, B4, B7 cap/smoothing (`reports/analysis/component_study_gpu2.md`). Adopted for the Kaggle preset: the reward audit + S5 floor cap, stricter calibration flags (DECISIONS.md D-98).
> - Open: B11 (Level-2 tool-call arm: after the Level-1 GPU results, by design), E12 (tags and sign-offs: owner decisions), C1 (GPU speed measurements: need the GPU), every pre-registered confirmation (D-99, proposed).
>
> **Read `errors_found.md` first.** It lists 27 errors found in the completed phases and in the plan (most verified by running code). Several of them (E-01, E-02, E-04, E-05) change results that items below build on, so fix those before starting the research items in §B.

What needs to change, why, and exactly how. Every item gives:
- the evidence (a file, a decision entry, or a measurement);
- the change (files, functions, config keys);
- how to check it worked.

Paths are relative to `curator-rl/`. "D-xx" refers to `docs/DECISIONS.md`.

**Measurements made for this document (2026-10-10).** Each was run on tuning seeds only, so no evaluation seed range was spent:
- **Final-score noise:** 40 episodes of S-I-R1, Curator vs Uniform, comparing the observed final score with the noiseless expected score. Result: CI half-width 0.0062 vs 0.0035.
- **Rounding of prompt counts:** largest-remainder rounding at M = 80 and M = 32 over 60 rounds, with an arm requested at weight 0.02.
- **Coverage:** `pytest --cov` over all of `src/`. Result: 85% total.
- **Code scan:** duplicated statistics code (7 copies), report provenance (none), git tags (none).

**Rule for any change that alters scheduler behaviour or simulator results:**
1. Tune on seeds 0–49.
2. Pre-register the criteria in `DECISIONS.md`.
3. Evaluate once on a fresh seed range.

Seed ranges 100–149, 200–249, 300–349, 400–449, 500–549 and 600–699 are spent. Proposed next ranges: 700–799, 800–899, 900–999, one per pre-registration.

---

## Summary

Priority:
- **P0** — must be fixed before GPU results can be trusted.
- **P1** — improves research results.
- **P2** — code quality and engineering.

Effort: **S** = up to half a day, **M** = 1–3 days, **L** = a week or more. "Pre-reg?" marks changes that need a pre-registered evaluation.

| ID | Area | Problem | Fix | Priority | Effort | Pre-reg? |
|---|---|---|---|---|---|---|
| A1 | calibration | The real noisy arm has no calibration slice, so S5 can never fire on GPU | Map noisy → GSM8K slice; add transfer-aware credit (C2d) | P0 | L | yes |
| A2 | scheduling | Delivered prompt shares differ from requested weights (0.031 vs 0.020 at M = 32); bookkeeping uses requested | Deficit-carry rounding; use delivered shares everywhere | P0 | M | yes (sim) |
| A3 | cost | Update cost per environment unmeasured; costs bracketed 2.06–3.46× | Build the span cost meter (Gate 5) | P0 | L | — |
| A4 | resume | No versioned whole-run checkpoint; loaders silently fill missing keys | Versioned, strict checkpoint plus kill-and-resume test | P0 | M | — |
| B1 | sim statistics | 68% of paired-difference variance is final-evaluation noise | Noiseless or paired-item final score; separate RNG stream | P1 | S | applies to new pre-regs |
| B2 | scheduler | γ = 0.95 costs 0.013–0.020 on refit scenarios | Powered γ = 1 re-test, or drift-triggered discounting | P1 | M | yes |
| B3 | signals | Learning progress barely detectable (z 0.06–0.15) | Longer LP window; α/β fit (B4) | P1 | S | yes |
| B4 | signals | Richness term rewards the noisy arm | Wire the α/β fitter | P1 | M | yes |
| B5 | status | S5 only halves the weight (noisy share 0.23 → 0.19) | Hard S5 cap at the floor | P1 | S | yes |
| B6 | calibration | Earliest S5 is after the round-40 calibration | Power-study grid over K, k_min, items | P1 | S | yes |
| B7 | sim realism | No forgetting, so concentration is never penalised | Add forgetting; add a max-weight cap or smoothing | P1 | M | yes |
| B8 | sim realism | Learning ignores group polarisation κ | Learnability from the Beta-binomial mixed-group rate | P1 | S | yes |
| B9 | sim realism | Noisy arm has zero effect on training | Scenario variants with ± side effect | P1 | S | yes |
| B10 | baselines | No baseline does what a practitioner would do (drop unsolvable arms) | Add "pilot-filtered uniform" | P1 | S | yes |
| B11 | claim | Cost lever untested at large spread on real arms | Level-2 tool-call arm (5–10× spread) | P1 | L | yes |
| C1 | GPU speed | Update and weight sync take ~34 of 42 s per step | Measure spans; larger micro-batch; compare server vs colocate | P1 | M | Gate 4 |
| C2 | GPU speed | MBPP verification 1.3 s per prompt serially | Thread pool overlapped with generation | P0 | S | — |
| D1–D4 | missing | ROI, LOO, α/β fitter, trainer adapter, viz are empty | Build to the specs below | P0/P1 | L | — |
| E1–E12 | engineering | Makefile bug, no lock file, CI never runs data tests, duplicated statistics, path hacks, no provenance, stale docs, no tags | See §E | P2 | S–M | — |

---

## A. Problems that would make GPU results wrong (P0)

### A1. The real noisy arm cannot be flagged by calibration

**Evidence**
- In the real portfolio, `noisy` uses GSM8K train prompts and has **no calib or test split** (README §6, `envs/noisy.py`).
- `Calibrator` evaluates and flags only arms with their own slice. Its docstring says: "Environments without their own slice (`domain_of[j] is None`) are never evaluated or flagged."
- In the simulator, the noisy arm has its own independent slice whose skill never moves. Its gain is therefore exactly 0, which makes detection easy. The adopted 78–86% detection rate (D-95) depends on that, so it is optimistic for the real system.
- D-76 already notes that own-slice credit "does not isolate an arm whose slice is shared with another arm (the real noisy arm shares GSM8K inputs); that needs the full transfer-matrix credit (C2d)".

**Change**
1. **Shared slice.** Pass `domain_of={"noisy": "gsm8k"}` to `Calibrator`; the constructor already accepts `domain_of`. Mirror this in the simulator: add `bench_domain: str | None` to `SimEnvParams` (`simulator/world.py`) so an arm can be scored on another arm's slice. Add scenario variants `S-I-R1-shared` and `S-J-R-shared` in which the noisy arm's slice is GSM8K's.
2. **Transfer-aware credit (C2d)** in `calibration/credit.py`, as a new class `TransferCredit`. For each slice `d` and each span `k` since its last evaluation:
   ```
   gain_{d,k} = Σ_j W_{j,k} · B_{d j} + noise(SE_{d,k})
   ```
   - Estimate row `B_{d,·}` by weighted ridge regression, with weights 1/SE² and a prior centred on 0 for off-diagonal entries.
   - The calibrator's per-arm credit becomes `b̂_j = B̂_{d(j), j}`, the arm's own column on its mapped slice.
   - The `z`-test in `Calibrator.observe` is unchanged, but uses the C2d estimate and its SE from the ridge posterior covariance.
3. **Identifiability.** GSM8K and noisy both feed the GSM8K slice, so their effects can only be separated if their shares vary independently across spans. Before building anything else, measure the correlation of `(W_gsm8k, W_noisy)` across spans under Curator on tuning seeds. If it exceeds about 0.8, add a small random share perturbation to the mixture (e.g. ±0.02, logged) so the regression is identifiable.

**Validate**
- Unit tests:
  - on synthetic spans with known `B`, the ridge estimate recovers it within 2 SE in ≥ 95% of draws;
  - the shared-slice simulator variant produces zero gain from noisy training.
- Pre-registered evaluation on fresh seeds 700–799, using the H-claim criteria of D-94 on the shared-slice scenarios: detection CI lower bound > 0 versus a no-calibration control, false flags ≤ 5%, calibration cost ≤ 10%, score non-inferiority −0.01.
- Report the detection rate next to D-95's independent-slice number; it will be lower.

### A2. Requested weights ≠ delivered prompts; bookkeeping uses the requested ones

**Evidence (measured)**
- `core/quotas.py::largest_remainder_quotas` gives a 0.02-weight arm 1 prompt every round at M = 32, i.e. **realised share 0.031 vs requested 0.020**. At M = 80 it gives 2 prompts per round: 0.025 vs 0.020.
- Small-weight arms are systematically over-served, because their fractional remainders rank highest. The planned GPU shape is M = R·P = 2·16 = 32.
- The bookkeeping uses the requested weights:
  - `scheduler/ducb.py::update_observation`: `u = observation.weights_used[env]`;
  - `simulator/harness.py::run_episode`: `exposure[env] += w × M`;
  - `simulator/world.py::step_round`: the skill update uses `weights.get(j)`, while rollouts and **costs** use the integer quotas. Learning and cost are therefore computed from different allocations.

**Change**
1. **Deficit-carry rounding** in `core/quotas.py` (new function `carry_quotas(weights, M, state)`):
   ```
   D_i ← D_i + w_i · M                         cumulative requested prompts
   m_i ← floor(D_i − delivered_i)              guaranteed part
   distribute the remaining M − Σ m_i to the largest (D_i − delivered_i − m_i)
   delivered_i ← delivered_i + m_i
   ```
   - Long-run delivered share then matches the requested share within 1 prompt per arm in total, not per round.
   - The state `(D, delivered)` must be part of the scheduler checkpoint.
2. **Use delivered shares everywhere:**
   - `u_i = n_prompts_i / Σ n_prompts` in `DiscountedUCB.update_observation` (new `pull_unit: delivered`);
   - `exposure[env] += n_prompts` in the harness, which feeds calibration credit;
   - `quota_i / M` in the world's skill update.
   - Put the world change behind a scenario flag `world.learning_from: requested | delivered`, with the default `requested`, so committed reports still reproduce. New pre-registrations use `delivered`.

**Validate**
- Property test: over 60 rounds with arbitrary weights, `|delivered_i − Σ_t w_{i,t}·M| ≤ 1`.
- Gate 4 already requires realised share within ±2 percentage points of the requested weight over 100 rounds; add this as an automated check in the trainer callback.
- Re-run the refit comparison on new seeds with `delivered` and report the change.

### A3. Per-environment training cost is not measured

**Evidence**
- `reports/pilot/phase_e_cost_probe.md` measures generation and verification only.
- Under vLLM, the policy update and weight sync take about 34 of the 42.2 s step. Its split across environments is bracketed: R1 = fully token-proportional, 3.46× spread; R2 = half fixed, 2.06× spread.
- The scheduler's cost signal and the matched-budget ledger both depend on it.

**Change** — build `cost/` exactly as `docs/COST_MODEL.md` §2–4 specifies:
- `cost/meter.py`:
  - spans `GEN`, `SCORE`, `UPDATE`, `EVAL` (calibration and reporting kept separate), `OVERHEAD`;
  - timed with `time.perf_counter_ns()`, with `torch.cuda.synchronize()` before each boundary read;
  - an injectable clock for tests.
- `cost/attribution.py`:
  - `SCORE` split exactly by per-item verifier seconds;
  - `GEN` split by `a_g·completion_tokens + b_g·n_seqs + c_g·prompt_tokens`;
  - `UPDATE` split by `a_u·tokens + b_u·n_seqs`;
  - coefficients fitted from single-environment micro-benchmarks (`scripts/cost_calibration.py`); shares rescaled each step so the attributed total equals the measured total.
- `cost/ledger.py`: charged categories `train_gen`, `train_score`, `train_update`, `calib_eval`, `curator_overhead`; uncharged `report_eval`; the stop rule from SPEC §4.

**Validate (Gate 5)**
- Attributed per-environment cost within 10% of exact single-environment timing.
- Unattributed time < 3% of wall time.
- Then replace the R1/R2 bracket in `experiments/analysis/refit_scenarios.py` with the measured attribution and regenerate the refit scenarios. They are a new version, so committed reports remain valid for their inputs.

### A4. Whole-run checkpoint and resume

**Evidence**
- Scheduler, engine, calibrator and CRE checkpoints exist, and round-trip tests exist (`tests/unit/test_ducb.py::test_checkpoint_round_trip_reproduces_decisions`, `test_calibration.py`, `test_cre.py`).
- But the loaders silently replace missing keys with defaults. Examples in `scheduler/ducb.py::load_checkpoint`: `state.get("win_x", {}) … or {e: 0.0 …}`, `state.get("budget_total")`; same pattern in `calibration/calibrator.py`.
- There is no schema version and no config-hash check.
- Kaggle sessions end after 12 hours, and a 60-round run plus calibration must survive interruption.

**Change**
- A single `RunCheckpoint` (new `trainer/checkpoint.py`) containing:
  - `schema_version`, `config_hash` (from `core/runmeta.py`), `git_commit`;
  - scheduler state (including A2's rounding state), trainer/optimizer/LoRA state;
  - all RNG streams (`core/seeding.SeedManager` names) and prompt-stream positions per environment;
  - ledger totals.
- Written atomically with `core/atomic.py`.
- Loading is strict: an unknown or missing key, a different `config_hash`, or a different `schema_version` raises an error unless an explicit migration is registered.

**Validate**
- Kill-and-resume test (Phase F requirement): run N rounds, kill, resume, run to the end. Weights, quotas, ledger and final scheduler state must equal an uninterrupted run with the same seed. On CPU, use the simulator with a fake trainer.
- On GPU, compare one resumed run against an uninterrupted one at rounds 1–10.

---

## B. Research performance: why CURATOR is not yet the best method, and how to fix it (P1)

Context (D-89, D-90, D-95): on the refit scenarios, CURATOR beats SEC/DUMP-style and Uniform but is **third behind LP and Standard UCB** (−0.011 to −0.014). The factorial diagnosis gives the causes. Each item below targets one of them.

### B1. Make simulator comparisons more powerful: remove final-evaluation noise (do this first)

**Evidence (measured)**
- `run_episode` scores the final model with `world.evaluate_benchmark(observed=True)`: Binomial(200 items) noise per domain, drawn independently for each method.
- On 40 tuning seeds of S-I-R1, Curator − Uniform:

  | Final score used | Mean | 95% CI half-width | SD of paired differences |
  |---|---|---|---|
  | Observed (current) | +0.0308 | 0.0062 | 0.0202 |
  | Noiseless `expected_benchmark_score(final skills)` | +0.0310 | 0.0035 | 0.0115 |

- `1 − (0.0115/0.0202)² = 68%` of the variance of paired differences is evaluation noise.
- The noiseless score gives a 44% narrower interval, which is worth about **3.1× more seeds**.
- In the real system all methods are scored on the **same** test items, so their evaluation noise is positively correlated. The simulator's independent draws overstate it.
- `evaluate_benchmark` also draws from `world.rng`, the training stream. Uncharged reporting evaluations therefore shift the training random numbers that follow them.

**Change**
1. In `simulator/harness.py::run_episode`, record `final_score_expected = world.expected_benchmark_score()` next to `final_score`. This is exactly `S_val(θ_T)`, the quantity inside the objective's expectation.
2. Add paired test items for the observed score. Each domain gets fixed item latents drawn from a seed-only stream `SeedManager(seed).rng("sim_test_items")`, so every method on the same seed is scored on the same items, as on GPU.
3. Give reporting and final evaluations their own stream, `rng("sim_report")`, so evaluations never consume training draws.
4. New pre-registrations name `final_score_expected` (or the paired-item score) as the primary simulator metric. Past gates keep their own metric; nothing is re-scored.

**Validate**
- Unit test: running with and without reporting evaluations gives identical training trajectories (same `true_skills` per round).
- Re-run the measurement above in CI as a smoke test.

**Effect on open questions:** D-92's γ = 1 non-inferiority test failed because its CI half-width was ≈ 0.009 at 50 seeds against a −0.01 margin, with point estimates −0.003 and −0.004. With this change the same seed count gives a half-width of about 0.005. This makes B2 testable with 100 seeds.

### B2. Discounting costs CURATOR 0.013–0.020

**Evidence**
- D-90 factorial: main effect of γ = 0.95 vs γ = 1 is −0.020 (S-I-R1), −0.014 (S-I-R2), −0.013 (S-J-R).
- D-92: γ = 1 gains +0.017 to +0.034 and beats Standard UCB in S-I-R1 (+0.016), but failed the no-regression check through lack of power.
- With κ = 2.14, per-round signals are noisy (B3), and a 20-round memory discards evidence the slowly drifting world does not make stale.

**Change (two options; pre-register one)**
- **Option 1 — powered re-test of γ = 1.** Use the B1 metric and seeds 700–799 (100 seeds). Sample size: to keep the lower bound above −0.01 when the true difference is −0.004, the half-width must be < 0.006. With the B1 metric, 100 seeds give ≈ 0.0035. Criteria as in D-91 (A1–A4).
- **Option 2 — drift-triggered discounting** (keeps protection against abrupt change, H7, scenario S-E):
  - γ = 1 by default.
  - Per arm, run a two-sided Page–Hinkley test on the standardised per-round pass rate `(p_raw,t − p̂_{t−1}) / se_group`.
  - When it fires, decay that arm's counters once (`Ñ_i ← ρ·Ñ_i`, `S̃_i ← ρ·S̃_i`, e.g. ρ = 0.3) and reset its detector.
  - Code: a new `signals/drift.py`, called from `DiscountedUCB.update_observation`. New config keys `scheduler.drift.enabled`, `.threshold`, `.decay`.
  - Tune the threshold on seeds 0–49 so the false-alarm rate is ≤ 1 per 100 rounds on refit scenarios and detection lag ≤ 5 rounds on S-E.

**Validate:** pre-registered criteria — beats Curator(γ 0.95) on the refit scenarios with CI > 0; no regression on S-A … S-J with lower bound > −0.01, using the B1 metric; S-E score not below Curator(γ 0.95).

### B3. Learning progress is too noisy to steer allocation

**Evidence**
- D-90: mean LP z-score on learnable arms is 0.06–0.15 within about 70 rounds, so the proxy runs mostly on richness.
- Cause, in closed form. When each prompt's pass probability is `q ~ Beta(κp, κ(1−p))` and G samples are drawn, the variance of a group's pass rate is
  ```
  Var(k/G) = p(1−p) · (G + κ) / (G·(κ + 1))
  ```
  and the fraction of it that comes from **which prompts were drawn** (not from sampling) is `G / (G + κ)`.
  - At the measured κ = 2.14 that fraction is **79%**; at the placeholder κ = 25 it was 24%.
  - The per-prompt variance is 2.5× what the placeholder scenarios assumed.
- Most round-to-round movement in pass rate is prompt-difficulty noise, not learning.

**Change**
- **Longer LP window.** The OLS slope SE scales as `W^(−3/2)`: `Var(slope) = 12σ²/(W(W²−1))`. Raising `signals.window_rounds` from 10 to 20 cuts the slope SE by a factor of 2.8. This fits B2's finding that the refit worlds drift slowly.
- Tune W ∈ {10, 15, 20, 30} on seeds 0–49 using the D-40 study (`experiments/analysis/lp_estimator_study.py`) on the **refit** scenarios. Keep D-40's rule: best ρ subject to FPR ≤ 5% and lag ≤ 5, with lag measured on S-E.
- `LPEstimator` already keeps `2W` history; no structural change is needed.

**Validate:** the study report (ρ, FPR, lag for each W). Then one pre-registered evaluation, combined with B4 if both are adopted.

### B4. Richness rewards the noisy arm: wire the α/β fit

**Evidence**
- D-90: noisy arm SR = 0.906, the highest of any arm. Its proxy `x` = 0.466 is above MATH35's 0.438.
- A Bernoulli(0.35) reward with G = 8 gives a mixed group with probability `1 − 0.65⁸ − 0.35⁸ = 0.968`, while real arms give 0.47–0.77 (`phase_e_cost_probe.md`).
- α = β = 0.5 are fixed. The fitter is designed (Roadmap v2 Part I, `calib.trust_region: 0.25` exists in `base.yaml`) but not implemented.

**Change** — new `calibration/regression.py::ProxyWeightFitter`, called from `DiscountedUCB.update_calibration` when `report.ready`:
1. Data: one row per arm with evidence, `y_j = b̂_j` (from A1's C2d), features `(LP̄'_j, SR̄_j)`, the exposure-weighted window means already accumulated by the scheduler. Rows are weighted by `1/SE(b̂_j)²`.
2. Fit: non-negative least squares on `y = s·(α·LP' + β·SR)`, with a ridge penalty toward the prior (0.5, 0.5).
3. Guard: accept the fit only if it beats the prior on leave-one-window-out prediction error; otherwise keep the prior (null-model fallback).
4. Trust region: each update may change α and β by at most 25% relative.
5. Write the new α, β into `ProxyReward` through a public setter, and log every fit in `reward_log`.

The noisy arm's high SR with zero gain then pushes β down.

**Validate**
- Gate 6-sim criterion from the roadmap: R² ≥ 0.8 on planted α/β in a synthetic scenario, and ≥ 20% better than the naive fit.
- Pre-registered refit evaluation (with B3): beats Curator-current with CI > 0; false flags ≤ 5%.

### B5. S5 only halves the spurious arm's weight

**Evidence**
- D-95: after adoption, the noisy arm's budget share fell only from 0.23 to 0.19.
- `mixture.py::apply_status_constraints` multiplies S5 weights by `s5_shrink = 0.5` and applies no cap, while S4 is capped at the floor.

**Change**
- In `apply_status_constraints`, under `status_control == "hard"`, add `out[e] = min(out[e], floor)` for S5, behind a new config key `scheduler.s5_cap: floor | none` (default `none` to keep committed reports reproducible).
- Alternative to pre-register instead: escalating shrink, `0.5^n` after n consecutive flagged windows.

**Expected effect:** detection happens around 60% of the way through a run. Capping at 0.02 instead of about 0.19 frees roughly `0.17 × 0.4 ≈ 7%` of the total budget for learnable arms. The risk is a falsely flagged good arm (current false-flag rate 2.0–2.5%); the existing exit rule (2 clean windows) limits the damage.

**Validate:** pre-registered criteria — score improvement over Curator-current with CI > 0 on refit scenarios; false-flag rate ≤ 5%; on seeds where a learnable arm is falsely flagged, report its budget loss.

### B6. Detection comes late

**Evidence**
- `k_min = 2` gain-producing windows are needed. The first calibration (round 10) is only a baseline, so decisions start at round 30 and the earliest S5 entry is after the round-40 calibration.
- Measured: about 4.3 decision-ready windows per episode (`reports/analysis/refit_diagnosis.md`).

**Change**
- Re-run the calibration power study (`experiments/analysis/calibration_power.py`) on the refit scenarios with `claim_stale/2` targeting, over K ∈ {5, 10}, items per slice ∈ {50, 100}, k_min ∈ {1, 2}, z_mis ∈ {2.0, 2.5}.
- Selection rule written beforehand: maximise detection subject to false flags ≤ 5% and cost ≤ 10%.
- K = 5 with 50 items has the same per-round cost as K = 10 with 100 items.

**Validate:** pre-registered evaluation of the chosen setting on fresh seeds; criteria as in D-94.

### B7. The simulator never penalises concentrated allocation

**Evidence**
- D-67: Curator's mean max-weight is 0.84–0.89 in S-A and S-C.
- D-92: γ = 1 raises the refit max-weight from 0.38 to 0.67.
- `world.py` has no forgetting: an arm's skill never falls when it stops being trained. Real GRPO can forget. Any gain from concentrating is therefore optimistic.

**Change**
1. Forgetting in `SimWorld.step_round`, per arm per round:
   ```
   s_j ← s_j − φ_j · (1 − share_j) · (s_j − s_j0)
   ```
   - φ = 0 by default. Variants φ ∈ {0.002, 0.005} give `S-I-R1-forget`.
   - Fit φ from Phase F logs: the skill decay on an arm's calibration slice while its share is at the floor.
2. Mixture smoothing in `mixture.py`, as `scheduler.max_weight` (cap, e.g. 0.6) or `scheduler.smoothing` (`w_t = (1−β)·w_{t−1} + β·w_new`).
3. Include concentration in the tuning objective, as D-67 already proposes.

**Validate:** pre-registered evaluation on the forgetting variants. The smoothed or capped scheduler must not lose more than 0.01 on the φ = 0 scenarios and must gain on φ > 0.

### B8. Simulated learning ignores group polarisation

**Evidence**
- `world.py::step_round` uses learnability `g = 4p(1−p)`, which does not depend on κ.
- In GRPO only mixed groups produce gradient, and the observed signals do depend on κ. Learning and observation are therefore inconsistent.

**Change** — replace `g` with the expected mixed-group fraction under the same Beta-binomial model, in closed form:
```
E[q^G]     = Π_{i=0}^{G−1} (κp + i) / (κ + i)
E[(1−q)^G] = Π_{i=0}^{G−1} (κ(1−p) + i) / (κ + i)
g_κ(p)     = 1 − E[q^G] − E[(1−q)^G]
```
- Normalise `g_κ` to peak at 1, so η keeps its meaning.
- Put it behind `world.learnability: binary | mixed_groups`; refit η from Phase F uniform runs (Roadmap v2 F.7).

**Validate:** unit test that `g_κ → 1 − p^G − (1−p)^G` as κ → ∞. Report how the refit rankings change on tuning seeds before any pre-registration.

### B9. The noisy arm cannot affect training in the simulator

**Evidence**
- Noisy arms never transfer (`world.py`: "noisy envs never learn and never transfer").
- Random rewards measurably change Qwen math scores (Shao et al., Spurious Rewards, cited in Roadmap v3 §2), and can harm other models.

**Change**
- Allow `transfer_out` on noisy arms, behind `noisy_transfer_allowed: true`.
- Create variants with `T_{gsm8k, noisy}` ∈ {−0.2, 0, +0.2}.

**Validate:** report detection and score on all three variants. The noisy-arm claim (H4) must hold for the negative and zero cases. If the positive case occurs on GPU, report it as a finding, as the roadmap's risk table already plans.

### B10. Add the baseline a practitioner would actually run

**Evidence**
- In the refit, CURATOR's gain over Uniform "comes from starving Countdown (2% vs 20%) and spreading over the learnable arms" (D-89).
- A practitioner who ran the pilot would drop Countdown before training: pass@8 = 0, 0 of 1,024 samples correct.
- The current Static baseline is size-proportional and gives noisy 0.286 and MBPP 0.013 (`reports/gate2/refit_gate.md`), which makes it a weak reference.

**Change**
- Add `pilot_uniform`: uniform over arms with pilot pass@8 > 0, i.e. gsm8k, math35, mbpp and noisy at 0.25 each, with Countdown at 0.
- The noisy arm stays in, because the pilot shows it as the easiest arm (pass@8 = 1.0). Nothing in a pilot reveals that its reward is random.
- Implement as `StaticMixtureScheduler` with weights from a new scenario field `pilot_pass_at_g`. Add it to `make_scheduler` in `experiments/run_sim.py` and to the GPU matrix.

**Validate:** pre-register it as an extra baseline in the next evaluation. If CURATOR does not beat it, the honest claim narrows to detection and attribution (Roadmap v3 §3.3 negative-result plan).

### B11. Test the cost lever where it matters

**Evidence**
- In S-C, where cost is unrelated to value, cost normalisation gains +0.06 over SEC/DUMP-style.
- On the real single-turn portfolio the spread is only 2.06–3.46×, and cost normalisation adds 0 while γ = 0.95 (D-90).

**Change:** the planned Level-2 `toolcall` environment (Roadmap v3 Phase J): deterministic multi-turn function calling with tool execution charged to the `SCORE` span, target spread ≥ 5× vs GSM8K. Build it after Level 1 is tagged.

**Validate:** Gate 1B' criteria plus measured spread; then H3 (cost normalisation drives the gain) on that portfolio.

---

## C. GPU runtime performance (Phase F)

### C1. Step time is dominated by update and weight sync

**Evidence:** vLLM colocate step is 42.2 s, of which generation ≈ 8.4 s and update + sync ≈ 34 s. Peak memory is 6.1 GB of 16 GB on a T4 (`reports/pilot/phase_e_cost_probe.md`).

**Change (measure first with A3's span meter, then try one at a time)**
1. **Larger micro-batch.** `scripts/kaggle_step_probe.py` uses `per_device_train_batch_size = G = 8` with `gradient_accumulation_steps = P = 16`, i.e. 16 sequential forward/backward passes. With about 10 GB free, try micro-batch 16 or 32 with accumulation 8 or 4. This gives the same effective batch, so the gradients are mathematically equivalent and only numerically different.
2. **Gradient checkpointing.** If it is enabled, disable it; the memory headroom allows it.
3. **Server vs colocate.** Compare vLLM server mode on the second T4 (one seed per session) against two colocated seeds, one per GPU. Choose by throughput in seed-rounds per GPU-hour, not by step time.
4. **Weight sync.** Measure the share of `UPDATE` spent syncing LoRA weights to vLLM. Do not change sync frequency: syncing less often makes generation off-policy, which changes the algorithm.

**Validate:** Gate 4 equivalence for every change — final score within δ_S = 1 point, mean training reward within 0.01, on a short static run against the baseline configuration.

### C2. MBPP verification must run in parallel (P0)

**Evidence:** 1.298 s per prompt serially vs 0.328 s with a 4-thread pool. Serially, a 16-prompt step would gain about 21 s. The sandbox is thread-safe (D-86): resource limits are set inside the child process.

**Change**
- `trainer/reward_wrapper.py`: a `ThreadPoolExecutor(max_workers=4)` (≥ 4) shared across steps.
- Submit verification per completion as soon as generation returns. Record each item's verifier seconds for the exact `SCORE` attribution (A3).

**Validate:** a unit test with a fake slow verifier checks wall time ≈ total time / workers. On GPU, `SCORE` span time per MBPP prompt should be ≤ 0.35 s.

### C3. Calibration evaluation cost

**Change**
- Greedy, batched vLLM generation over the fixed calibration items: 2 slices × 100 items per window.
- Prompts are cached once per run, and the time is charged to the `EVAL/calib` span.
- Check the simulator's assumption (cost per item = unit cost / G) against the measured `EVAL` span.

---

## D. Missing components (empty packages)

| ID | Package | Exact content | Gate |
|---|---|---|---|
| D1 | `trainer/` | `trl_adapter.py` (the only TRL import); `reward_wrapper.py` (dispatch by the `env_id` dataset column to the environment's `evaluate_response`, C2 pool); `prompt_stream.py` (draws each round's quotas per environment from the train split without replacement within a pass, using A2's `carry_quotas`); `callback.py` (at each round boundary: build `RoundObservation` from reward logs and cost spans → `scheduler.update_observation` → `select_mixture` → next quotas; every K rounds, `calibration_request` → evaluate → `update_calibration`); `checkpoint.py` (A4) | 3, 4 |
| D2 | `cost/` | as A3 | 5 |
| D3 | `roi/` + `evaluation/loo.py` | Per arm: estimated marginal gain per dollar `b̂_j / unit_cost_j` from C2d credit, with block-bootstrap intervals over calibration windows; a ranked leaderboard. LOO: retrain with arm i removed at the same budget; contribution_i = S(full) − S(without i). Agreement: Kendall τ between ROI ranks and LOO ranks; H5 requires τ ≥ 0.4 with both junk arms in the bottom 2 | 7, 9 |
| D4 | `calibration/regression.py` | as B4 | 6 |
| D5 | `viz/` | Pure functions over run logs: score vs GPU-dollar per method; weights over rounds with a status timeline; ROI vs LOO scatter; S5 timeline. Developed on simulator logs first | — |

---

## E. Code quality and engineering (P2)

### E1. `make` fails on a fresh clone
- **Evidence:** `Makefile` line 1 is `include requirements.lock`, but the file does not exist. GNU make stops with "No such file or directory". CI does not use `make`, which is why this was never caught.
- **Change:** use `-include requirements.lock`, or create the file (E2). Add a Windows note: `make` is not installed by default, so the README should list the plain `python -m …` equivalents.

### E2. No dependency lock
- **Evidence:** `pyproject.toml` has lower bounds only (`numpy>=1.26` …); `train = []` is empty. The GPU stack (TRL 1.15.0, PEFT 0.20.0, vLLM 0.31.0, torch versions) is pinned only in prose (`docs/TRL_SPIKE.md`).
- **Change:**
  - Generate `requirements.lock` for the CPU stack (`pip-compile` or `uv pip compile pyproject.toml --extra dev`).
  - Fill `train` with exact pins: `trl==1.15.0`, `peft==0.20.0`, the transformers and torch versions recorded in TRL_SPIKE.md.
  - Add `requirements-vllm.lock` for the separate vLLM virtualenv.
  - Have `scripts/make_kaggle_bundle.py` include the lock files.

### E3. CI never runs the dataset tests; coverage is measured only on `core`
- **Evidence:**
  - `pyproject.toml` `[tool.coverage.run] source = ["curator_rl.core"]`.
  - Measured over all of `src/`: 85% total. Core logic is at 95–100%, but `envs/gsm8k.py` 27%, `envs/finite.py` 29%, `envs/noisy.py` 41%, `envs/registry.py` 0%, `simulator/replay.py` 0%.
  - 23 contract tests skip whenever the data is not downloaded — always the case in CI — so the "gold 100%, garbage 0%" verifier checks are never re-run automatically.
- **Change:**
  1. A CI job that runs `scripts/download_data.py`, `build_splits.py` and the `_phase_e` versions, with `actions/cache` keyed on `data/manifests/MANIFEST.sha256`, then `pytest tests/contract`.
  2. Set coverage `source = ["curator_rl"]` with `--cov-fail-under=85`.
  3. Add tests for `envs/registry.py` and `simulator/replay.py`, or delete `replay.py` if unused.

### E4. Statistics code duplicated in 7 gate scripts
- **Evidence:** `gate2_check.py`, `gate2b_check.py`, `cre_gate.py`, `cre_kappa_gate.py`, `refit_gate.py`, `gamma_gate.py` and `claim_gate.py` each define their own bootstrap. D-69 already found a unit bug in one gate's tolerance (`DELTA_GAP = 1.0` instead of `0.01`), which made a criterion vacuous.
- **Change:**
  - Create `src/curator_rl/evaluation/stats.py` with `paired_bootstrap(diffs, n=10_000, seed)`, `ci_excludes_zero`, `non_inferior(lower, margin)`, `holm(pvals)`. Margins must be passed in benchmark-fraction units, with a named constant `ONE_POINT = 0.01`.
  - Unit tests: coverage of a known-mean normal sample ≈ 95%, and fixed-seed reproducibility.
  - Replace the 7 copies. Results must not change: run each gate script on its tuning-seed smoke setting before and after and compare.

### E5. `sys.path.insert` in 30 scripts
- **Evidence:** `grep -rn "sys.path.insert" experiments scripts` returns 49 lines in 30 files (40 lines in `experiments/`).
- **Change:** add `experiments/__init__.py` and include `experiments*` in `[tool.setuptools.packages.find]` (or run scripts as `python -m experiments.gate2_check`), then delete the path inserts. `scripts/` only needs the installed package, so its inserts can simply be removed.

### E6. Reports have no provenance
- **Evidence:** report JSONs contain `generated_utc` but no git commit or config hash (checked `reports/gate2/*.json` and `reports/analysis/*.json`). D-95 notes that earlier reports were produced with a different targeting rule and "are not regenerated", and there is no automatic way to tell which reports are stale.
- **Change:**
  - Every report writer adds `{git_commit, git_dirty, config_hash, scenario_hashes, seeds, package_version}`, built with the existing `core/runmeta.py`.
  - A `make reports` target regenerates them.
  - A test (`tests/integration/test_report_provenance.py`) lists reports whose `config_hash` differs from the current `base.yaml` hash. It warns rather than fails, since old reports are legitimately frozen.

### E7. The scheduler reads another class's private state
- **Evidence:** `scheduler/ducb.py` reads `self._engine._signals_cfg.status.n_min` and `self._engine._unit_cost[e]` with `# noqa: SLF001`.
- **Change:** add read-only properties `SignalEngine.n_min` and `SignalEngine.unit_cost(env_id)`, and use them.

### E8. Untyped checkpoint state
- **Evidence:** every `load_checkpoint` carries `# type: ignore[...]` on each line.
- **Change:** define a `TypedDict` per state (`DUCBState`, `CalibratorState`, …). Add `pyright` (basic mode) for `src/` to CI. This pairs with A4's strict loader.

### E9. Simulator RNG stream sharing
Covered in B1: reporting and final evaluations must use `rng("sim_report")`, not `world.rng`.

### E10. Rollout draws are not vectorised (low priority)
- **Evidence:** `world.py::_rollout_env` calls `rng.binomial(G, q)` once per prompt in a Python loop.
- **Change:** `ks = rng.binomial(G, qs)`. This changes the random stream, so do it only together with a new world version flag and new seed ranges. Runs are already fast: 10 episodes × 5 methods took 4.5 s.

### E11. Stale or contradictory documentation
| File | Stale text | Fix |
|---|---|---|
| `README.md` line 9–12 | "335 tests pass", "Nothing after commit `2d3b40a` is committed yet" | Everything is committed (`bd9099e`); 326 pass / 24 skip without data, 349 with data |
| `README.md` §4 | "349 passed, 1 skipped" without saying it needs data | State both numbers and the reason |
| `docs/COST_MODEL.md` header | "Status: skeleton. Nothing below has been measured yet" | §6 contains measurements; update the header |
| `docs/GATES.md` phase log | "No commit yet", tags "not yet created" | Fill in commits; create the tags (E12) |
| `scheduler/curator.py` docstring | "the alpha/beta fit + S5 mismatch feed arrive with Phase 9" | S5 feed is wired (D-73); α/β is still pending |
| `Curator/` (repo root) | duplicate of `curator-rl/docs/CURATOR_Implementation_Roadmap.md` and both proposal PDFs | Delemakte the duplicates, keep `curator-rl/docs/` |
| `envs/math35.py` docstring | "5586 problems" | README and gold check say 5,582; correct one of them |

### E12. Git and review process
- **Evidence:**
  - No tags exist (`git tag` is empty), although the protocol relies on `spec-v1`, `phase-5`, `L1`.
  - Commit `9cec931` ("Update latest code") adds 25,690 lines in one commit.
  - Gate 0 sign-off and decisions D-52 … D-62 are still "proposed".
  - The roadmap names "single-owner blind spots (no second reviewer)" as a top risk.
- **Change:**
  1. Owner signs off D-1 … D-15 and D-52 … D-62 in `DECISIONS.md`, then `git tag spec-v1`.
  2. Tag the current state (`git tag phase-e3 bd9099e`).
  3. From now on: one branch and pull request per item in this document. A teammate reviews each PR before merge, using a checklist: tests, ruff, decision entry if behaviour changes, report provenance. That turns the project's biggest stated risk into a team process.

---

## F. Recommended order

Each step's dependencies come first.

1. **Hygiene (1–2 days, no results change):** E1, E2, E11, E12, E7, E5, E4 (verified to not change outputs), E6.
2. **Simulator fixes behind version flags (2–3 days):** B1/E9 first, because they make every later test about 3× cheaper; then A2 (world part), B8, B9, B7 (forgetting variants). Committed reports stay reproducible under the old flags.
3. **Phase F build (1–2 weeks, GPU):** D1, A3/D2, A4, A2 (trainer part), C2, then C1 measurements. Gates 3, 4, 5.
4. **Before GPU calibration:** A1 (shared slice + C2d) in the simulator, then pre-register on seeds 700–799.
5. **Scheduler improvements, one pre-registration on seeds 800–899.** Run a factorial over B5 (S5 cap), B3 (window), B2 (drift-triggered discount or γ = 1), with B10's `pilot_uniform` added as a baseline and B1's metric as primary. Adopt only components whose main effect passes; report the rest.
6. **B4 (α/β fit) and B6 (calibration timing):** pre-registered on seeds 900–999.
7. **Freeze and main runs:** D3 (ROI, LOO), D5, Gate 10 freeze, then the Level-1 GPU matrix (~125 GPU-hours).
8. **Level 2:** B11 (tool-call arm).
