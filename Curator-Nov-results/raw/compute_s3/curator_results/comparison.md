# CURATOR GPU comparison

Sampled accuracy = mean over 4 samples per item at the training temperature (evaluation protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method (uncharged; the sealed test is not used).

| method | seeds | sampled macro (mean) | pass@k macro | greedy macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |
|---|---|---|---|---|---|---|---|---|
| curator_nocal | 2 | 0.2301 | 0.3921 | 0.3063 | 1.516 | 35.0 | noisy | 8 |
| curator_q | 2 | 0.2315 | 0.3867 | 0.3104 | 1.525 | 24.5 | noisy | 4 |
| dump | 2 | 0.2174 | 0.3712 | 0.3104 | 1.517 | 33.0 | - | 0 |
| sec | 2 | 0.2231 | 0.3817 | 0.3121 | 1.534 | 33.5 | - | 1 |
| uniform | 2 | 0.2210 | 0.3754 | 0.3037 | 1.507 | 33.5 | - | 1 |

## Per domain (sampled accuracy; greedy in brackets; mean over seeds)

| method | gsm8k | math35 | mbpp | countdown |
|---|---|---|---|---|
| curator_nocal | 0.483 (0.605) | 0.153 (0.262) | 0.283 (0.350) | 0.001 (0.008) |
| curator_q | 0.478 (0.610) | 0.166 (0.258) | 0.281 (0.367) | 0.000 (0.007) |
| dump | 0.476 (0.603) | 0.154 (0.260) | 0.240 (0.375) | 0.000 (0.003) |
| sec | 0.466 (0.600) | 0.160 (0.258) | 0.267 (0.383) | 0.000 (0.007) |
| uniform | 0.481 (0.593) | 0.157 (0.248) | 0.246 (0.367) | 0.000 (0.007) |

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
| curator_nocal | 3 | +0.0213 (0.0083) | -0.0158 (0.0169) | gsm8k +0.029, math35 -0.007, mbpp +0.062, countdown +0.001 |
| curator_nocal | 4 | -0.0031 (0.0079) | +0.0208 (0.0129) | gsm8k -0.026, math35 +0.001, mbpp +0.013, countdown +0.000 |
| curator_q | 3 | +0.0173 (0.0080) | -0.0083 (0.0169) | gsm8k +0.010, math35 +0.005, mbpp +0.054, countdown +0.000 |
| curator_q | 4 | +0.0035 (0.0070) | +0.0217 (0.0134) | gsm8k -0.016, math35 +0.014, mbpp +0.017, countdown -0.001 |
| dump | 3 | -0.0004 (0.0071) | -0.0017 (0.0115) | gsm8k +0.002, math35 -0.016, mbpp +0.013, countdown +0.000 |
| dump | 4 | -0.0069 (0.0076) | +0.0150 (0.0132) | gsm8k -0.012, math35 +0.010, mbpp -0.025, countdown -0.001 |
| sec | 3 | +0.0140 (0.0067) | -0.0083 (0.0156) | gsm8k +0.007, math35 +0.007, mbpp +0.042, countdown +0.001 |
| sec | 4 | -0.0098 (0.0075) | +0.0250 (0.0132) | gsm8k -0.037, math35 -0.001, mbpp +0.000, countdown -0.001 |

curator_nocal vs uniform, sampled macro over 2 seeds: +0.0091, 95% t-interval [-0.1458, +0.1639], sign-flip p 1.000

curator_q vs uniform, sampled macro over 2 seeds: +0.0104, 95% t-interval [-0.0769, +0.0978], sign-flip p 0.500

dump vs uniform, sampled macro over 2 seeds: -0.0036, 95% t-interval [-0.0447, +0.0374], sign-flip p 0.500

sec vs uniform, sampled macro over 2 seeds: +0.0021, 95% t-interval [-0.1488, +0.1530], sign-flip p 1.000

## GPU-hours to reach uniform's final level (sampled, the same 100 items per domain)

Learning curves start at the untrained model (0 GPU-h). Fewer GPU-hours = the same accuracy for less compute.

| method | seed | uniform final (at GPU-h) | GPU-h to reach it | compute saved |
|---|---|---|---|---|
| curator_nocal | 3 | 0.2175 at 1.50 | 0.83 | +45% |
| curator_q | 3 | 0.2175 at 1.50 | 0.98 | +35% |
| dump | 3 | 0.2175 at 1.50 | not reached (best 0.2119) | - |
| sec | 3 | 0.2175 at 1.50 | 1.11 | +26% |
| curator_nocal | 4 | 0.2217 at 1.51 | not reached (best 0.2217) | - |
| curator_q | 4 | 0.2217 at 1.51 | not reached (best 0.2196) | - |
| dump | 4 | 0.2217 at 1.51 | not reached (best 0.2094) | - |
| sec | 4 | 0.2217 at 1.51 | not reached (best 0.2100) | - |

## Greedy, paired by seed against uniform (protocol 1)

| method | n | mean diff | 95% t-interval | sign-flip p |
|---|---|---|---|---|
| curator_nocal | 2 | +0.0025 | [-0.2304, +0.2354] | 1.000 |
| curator_q | 2 | +0.0067 | [-0.1839, +0.1973] | 1.000 |
| dump | 2 | +0.0067 | [-0.0992, +0.1126] | 1.000 |
| sec | 2 | +0.0083 | [-0.2034, +0.2201] | 1.000 |

With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only.

## ROI leaderboard: curator_q, seed 3 (2 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | gsm8k | +0.1373 (0.0697) | 1.25e-03 | +19.64 | [+0.11, +39.17] |
| 2 | noisy | +0.0108 (0.4985) | 1.09e-03 | +1.76 | [-157.93, +161.45] |
| 3 | countdown | +0.0000 (0.0000) | 1.60e-03 | +0.00 | [+0.00, +0.00] |
| 4 | math35 | +0.0000 (0.0000) | 2.43e-03 | +0.00 | [+0.00, +0.00] |
| 5 | mbpp | -0.0171 (0.0297) | 5.48e-04 | -5.58 | [-24.53, +13.37] |


## ROI leaderboard: curator_q, seed 4 (2 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 2 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | math35 | +0.0289 (0.0202) | 1.67e-03 | +3.09 | [-1.15, +7.34] |
| 2 | countdown | +0.0000 (0.0000) | 1.29e-03 | +0.00 | [+0.00, +0.00] |
| 3 | mbpp | +0.0000 (0.0000) | 5.09e-04 | +0.00 | [+0.00, +0.00] |
| 4 | noisy | -0.0050 (0.4975) | 8.77e-04 | -1.02 | [-199.48, +197.44] |
| 5 | gsm8k | -0.0496 (0.0938) | 8.69e-04 | -10.19 | [-47.99, +27.61] |

