# CURATOR GPU comparison

Sampled accuracy = mean over 4 samples per item at the training temperature (evaluation protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method (uncharged; the sealed test is not used).

| method | seeds | sampled macro (mean) | pass@k macro | greedy macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |
|---|---|---|---|---|---|---|---|---|
| curator_nocal | 2 | 0.2301 | 0.3800 | 0.3021 | 1.533 | 33.5 | noisy | 1 |
| curator_q | 2 | 0.2369 | 0.4008 | 0.3067 | 1.540 | 25.0 | noisy | 2 |
| dump | 2 | 0.2207 | 0.3775 | 0.3179 | 1.519 | 30.5 | - | 3 |
| sec | 2 | 0.2244 | 0.3808 | 0.3117 | 1.515 | 30.5 | - | 1 |
| uniform | 2 | 0.2208 | 0.3825 | 0.3133 | 1.554 | 30.0 | - | 2 |

## Per domain (sampled accuracy; greedy in brackets; mean over seeds)

| method | gsm8k | math35 | mbpp | countdown |
|---|---|---|---|---|
| curator_nocal | 0.470 (0.612) | 0.161 (0.257) | 0.290 (0.333) | 0.000 (0.007) |
| curator_q | 0.495 (0.600) | 0.167 (0.258) | 0.285 (0.358) | 0.000 (0.010) |
| dump | 0.482 (0.613) | 0.147 (0.268) | 0.252 (0.383) | 0.001 (0.007) |
| sec | 0.489 (0.615) | 0.157 (0.262) | 0.252 (0.367) | 0.000 (0.003) |
| uniform | 0.460 (0.607) | 0.160 (0.255) | 0.263 (0.383) | 0.000 (0.008) |

## Pre-flight verifier check (D-102)

Reward each arm paid for junk before any training (empty or echoed answers, wrong or sprayed numbers, code that exits early or prints its globals). Above 5% the checker can be gamed.

| arm | junk reward rate | flagged |
|---|---|---|
| countdown | 0.00 | - |
| gsm8k | 0.00 | - |
| math35 | 0.00 | - |
| mbpp | 0.00 | - |
| noisy | 0.30 | yes |

## Item by item against uniform (same seed, same items)

Paired over items: the SE is the evaluation's, for one seed. Whether a difference holds across training seeds is the across-seed interval below it.

| method | seed | sampled macro diff (SE) | greedy macro diff (SE) | sampled diff per domain |
|---|---|---|---|---|
| curator_nocal | 3 | +0.0098 (0.0098) | -0.0100 (0.0164) | gsm8k +0.023, math35 -0.004, mbpp +0.021, countdown -0.001 |
| curator_nocal | 4 | +0.0088 (0.0060) | -0.0125 (0.0130) | gsm8k -0.004, math35 +0.006, mbpp +0.033, countdown +0.000 |
| curator_q | 3 | +0.0225 (0.0081) | +0.0108 (0.0169) | gsm8k +0.041, math35 +0.013, mbpp +0.037, countdown -0.001 |
| curator_q | 4 | +0.0096 (0.0060) | -0.0242 (0.0123) | gsm8k +0.028, math35 +0.002, mbpp +0.008, countdown +0.000 |
| dump | 3 | +0.0050 (0.0069) | +0.0108 (0.0096) | gsm8k +0.028, math35 -0.016, mbpp +0.008, countdown -0.001 |
| dump | 4 | -0.0052 (0.0065) | -0.0017 (0.0134) | gsm8k +0.016, math35 -0.009, mbpp -0.029, countdown +0.002 |
| sec | 3 | +0.0048 (0.0072) | +0.0058 (0.0103) | gsm8k +0.023, math35 -0.012, mbpp +0.008, countdown -0.001 |
| sec | 4 | +0.0023 (0.0068) | -0.0092 (0.0118) | gsm8k +0.033, math35 +0.005, mbpp -0.029, countdown +0.000 |

curator_nocal vs uniform, sampled macro over 2 seeds: +0.0093, 95% t-interval [+0.0027, +0.0159], sign-flip p 0.500

curator_q vs uniform, sampled macro over 2 seeds: +0.0160, 95% t-interval [-0.0660, +0.0981], sign-flip p 0.500

dump vs uniform, sampled macro over 2 seeds: -0.0001, 95% t-interval [-0.0650, +0.0648], sign-flip p 1.000

sec vs uniform, sampled macro over 2 seeds: +0.0035, 95% t-interval [-0.0123, +0.0194], sign-flip p 0.500

## GPU-hours to reach uniform's final level (sampled, the same 100 items per domain)

Learning curves start at the untrained model (0 GPU-h). Fewer GPU-hours = the same accuracy for less compute.

| method | seed | uniform final (at GPU-h) | GPU-h to reach it | compute saved |
|---|---|---|---|---|
| curator_nocal | 3 | 0.2077 at 1.57 | 0.68 | +57% |
| curator_q | 3 | 0.2077 at 1.57 | 0.82 | +48% |
| dump | 3 | 0.2077 at 1.57 | 0.98 | +38% |
| sec | 3 | 0.2077 at 1.57 | 0.76 | +52% |
| curator_nocal | 4 | 0.2148 at 1.54 | 1.11 | +28% |
| curator_q | 4 | 0.2148 at 1.54 | 1.28 | +16% |
| dump | 4 | 0.2148 at 1.54 | not reached (best 0.2119) | - |
| sec | 4 | 0.2148 at 1.54 | not reached (best 0.2137) | - |

## Greedy, paired by seed against uniform (protocol 1)

| method | n | mean diff | 95% t-interval | sign-flip p |
|---|---|---|---|---|
| curator_nocal | 2 | -0.0113 | [-0.0271, +0.0046] | 0.500 |
| curator_q | 2 | -0.0067 | [-0.2290, +0.2157] | 1.000 |
| dump | 2 | +0.0046 | [-0.0748, +0.0840] | 1.000 |
| sec | 2 | -0.0017 | [-0.0970, +0.0936] | 1.000 |

With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only.

## ROI leaderboard: curator_q, seed 3 (2 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 2 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | gsm8k | +0.0342 (0.0641) | 1.26e-03 | +4.84 | [-12.94, +22.62] |
| 2 | noisy | +0.0023 (0.4989) | 1.12e-03 | +0.37 | [-156.08, +156.83] |
| 3 | countdown | +0.0000 (0.0000) | 1.59e-03 | +0.00 | [+0.00, +0.00] |
| 4 | math35 | +0.0000 (0.0000) | 2.48e-03 | +0.00 | [+0.00, +0.00] |
| 5 | mbpp | -0.0181 (0.0314) | 5.69e-04 | -5.69 | [-24.99, +13.62] |


## ROI leaderboard: curator_q, seed 4 (2 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 2 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | gsm8k | +0.0401 (0.0517) | 8.83e-04 | +8.12 | [-12.37, +28.60] |
| 2 | math35 | +0.0299 (0.0330) | 1.75e-03 | +3.04 | [-3.54, +9.63] |
| 3 | noisy | +0.0021 (0.4993) | 8.97e-04 | +0.42 | [-194.42, +195.26] |
| 4 | countdown | +0.0000 (0.0000) | 1.34e-03 | +0.00 | [+0.00, +0.00] |
| 5 | mbpp | +0.0000 (0.0000) | 5.14e-04 | +0.00 | [+0.00, +0.00] |

