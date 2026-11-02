# D-103 confirmation: results

Generated 2026-11-02 06:47 UTC by `experiments/analysis/d103_confirm.py` (written before the runs). Model: Qwen2.5-0.5B-Instruct. Primary measure: sampled macro accuracy, item-paired, in percentage points. Sources: `..\curator_results_november\runs`.

## Pre-registered rules

| Rule | Comparison | Per seed | Mean | SE | 95% t-interval | Sign-flip p | Verdict | Why |
|---|---|---|---|---|---|---|---|---|
| R1 (primary) | curator_q - uniform_q | s0 +1.83, s1 -0.15, s2 +0.25 | +0.65 | 0.60 | [-1.96, +3.25] | 0.500 | **FAILS** | seeds disagree in sign; |mean| = 1.1 SE < 2 |
| R2 | sec - uniform | s0 -2.31, s1 -0.33, s2 -0.23 | -0.96 | 0.68 | [-3.87, +1.96] | 0.250 | **FAILS** | |mean| = 1.4 SE < 2 |
| R2 | dump - uniform | s0 -1.48, s1 -0.62, s2 -0.15 | -0.75 | 0.39 | [-2.43, +0.93] | 0.250 | **FAILS** | |mean| = 1.9 SE < 2 |
| R4 | curator_nocal - curator | s0 -1.10, s1 +1.27, s2 +0.29 | +0.15 | 0.69 | [-2.81, +3.12] | 0.750 | **HOLDS** | mean +0.15 points (within +-1); t-interval [-2.81, +3.12] not inside +-1 |

**R2 overall: FAILS** (needs both SEC and DUMP).

**R3 (mechanism, descriptive): HOLDS.** Share of training questions on the random-reward arm (SEC and DUMP above 20%, Curator variants under 13%):

| Method | Seeds | Per seed | Mean |
|---|---|---|---|
| sec | 3 | s0 33.9%, s1 33.4%, s2 33.6% | 33.6% |
| dump | 3 | s0 30.6%, s1 31.1%, s2 30.8% | 30.8% |
| uniform | 3 | s0 20.0%, s1 19.9%, s2 20.0% | 20.0% |
| uniform_q | 3 | s0 20.0%, s1 20.0%, s2 20.0% | 20.0% |
| lp | 3 | s0 10.2%, s1 13.8%, s2 16.0% | 13.3% |
| curator | 3 | s0 6.1%, s1 15.8%, s2 6.7% | 9.5% |
| curator_nocal | 3 | s0 5.9%, s1 13.7%, s2 6.2% | 8.6% |
| curator_q | 3 | s0 6.0%, s1 6.0%, s2 5.9% | 6.0% |
| dirichlet | 2 | s0 17.0%, s1 16.1% | 16.6% |
| pilot_uniform | 1 | s0 25.0% | 25.0% |

With 3 seeds, "mean >= 2 SE" is evidence at roughly one-sided p 0.1, not proof (D-103).

## Every method against Uniform

| Method | Seeds | Sampled diff (mean) | 95% t-interval | Greedy diff (mean) | Random-reward share | Gain over untrained (sampled) | pass@k |
|---|---|---|---|---|---|---|---|
| curator | 3 | -0.13 | [-1.22, +0.96] | +0.25 | 9.5% | +5.40 | 0.388 |
| curator_nocal | 3 | +0.02 | [-2.44, +2.48] | +0.64 | 8.6% | +5.56 | 0.396 |
| curator_q | 3 | -0.15 | [-1.64, +1.35] | +0.39 | 6.0% | +5.39 | 0.387 |
| dirichlet | 2 | -0.36 | [-14.53, +13.80] | -0.12 | 16.6% | +4.77 | 0.382 |
| dump | 3 | -0.75 | [-2.43, +0.93] | +0.44 | 30.8% | +4.78 | 0.374 |
| lp | 3 | +0.15 | [-1.72, +2.03] | +0.08 | 13.3% | +5.69 | 0.391 |
| pilot_uniform | 1 | -0.56 | one seed (eval SE 0.64) | +0.83 | 25.0% | +5.92 | 0.388 |
| sec | 3 | -0.96 | [-3.87, +1.96] | +0.69 | 33.6% | +4.58 | 0.385 |
| uniform_q | 3 | -0.79 | [-4.20, +2.61] | +0.17 | 20.0% | +4.74 | 0.378 |

## Figures

- `figures/accuracy_vs_uniform.png`: every method against Uniform (filled = 3-seed interval).
- `figures/noisy_arm_share.png`: budget on the random-reward arm, seed dots.
- `figures/noisy_share_by_round.png`: the same over training rounds (the trap, as it happens).
- `figures/rules_per_seed.png`: R1 and R2 seed by seed.

## Caveats

- SEC and DUMP are re-implementations from the papers; SEC's tuned temperature (0.4) is sharper than the authors' default (1.0), which was outside the tuning grid (`reports/analysis/sec_dump_fidelity.md`).
- The portfolio contains a random-reward arm on purpose: the comparison tests robustness, not accuracy on clean data.
- Story decision (D-103): R1 and R2 hold -> the robustness thesis leads; otherwise the fallback (rigorous negative result with its mechanism, plus the pre-flight check and the in-run audit).
