# CURATOR GPU comparison

Sampled accuracy = mean over 4 samples per item at the training temperature (evaluation protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method (uncharged; the sealed test is not used).

| method | seeds | sampled macro (mean) | pass@k macro | greedy macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |
|---|---|---|---|---|---|---|---|---|
| curator_nocal | 2 | 0.2365 | 0.3883 | 0.3029 | 1.515 | 36.5 | noisy | 1 |
| curator_q | 2 | 0.2431 | 0.4046 | 0.3204 | 1.506 | 26.5 | noisy | 2 |
| dump | 2 | 0.2154 | 0.3742 | 0.3087 | 1.514 | 34.0 | - | 0 |
| sec | 2 | 0.2278 | 0.3908 | 0.3154 | 1.534 | 34.5 | - | 0 |
| uniform | 2 | 0.2289 | 0.3929 | 0.3000 | 1.515 | 33.0 | - | 1 |

## Per domain (sampled accuracy; greedy in brackets; mean over seeds)

| method | gsm8k | math35 | mbpp | countdown |
|---|---|---|---|---|
| curator_nocal | 0.485 (0.607) | 0.162 (0.257) | 0.298 (0.342) | 0.000 (0.007) |
| curator_q | 0.490 (0.610) | 0.173 (0.255) | 0.308 (0.400) | 0.000 (0.017) |
| dump | 0.463 (0.607) | 0.152 (0.262) | 0.246 (0.358) | 0.000 (0.008) |
| sec | 0.482 (0.608) | 0.160 (0.272) | 0.269 (0.375) | 0.000 (0.007) |
| uniform | 0.485 (0.595) | 0.165 (0.258) | 0.265 (0.342) | 0.000 (0.005) |

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
| curator_nocal | 3 | +0.0073 (0.0074) | +0.0042 (0.0160) | gsm8k -0.007, math35 -0.007, mbpp +0.042, countdown +0.001 |
| curator_nocal | 4 | +0.0079 (0.0080) | +0.0017 (0.0151) | gsm8k +0.007, math35 +0.000, mbpp +0.025, countdown +0.000 |
| curator_q | 3 | +0.0140 (0.0075) | +0.0242 (0.0136) | gsm8k -0.013, math35 +0.014, mbpp +0.054, countdown +0.000 |
| curator_q | 4 | +0.0146 (0.0081) | +0.0167 (0.0117) | gsm8k +0.022, math35 +0.002, mbpp +0.033, countdown +0.001 |
| dump | 3 | -0.0140 (0.0077) | +0.0092 (0.0147) | gsm8k -0.034, math35 -0.014, mbpp -0.008, countdown +0.001 |
| dump | 4 | -0.0129 (0.0071) | +0.0083 (0.0122) | gsm8k -0.010, math35 -0.013, mbpp -0.029, countdown +0.000 |
| sec | 3 | +0.0042 (0.0060) | +0.0175 (0.0093) | gsm8k -0.003, math35 +0.003, mbpp +0.017, countdown +0.000 |
| sec | 4 | -0.0062 (0.0066) | +0.0133 (0.0109) | gsm8k -0.003, math35 -0.014, mbpp -0.008, countdown +0.000 |

curator_nocal vs uniform, sampled macro over 2 seeds: +0.0076, 95% t-interval [+0.0036, +0.0116], sign-flip p 0.500

curator_q vs uniform, sampled macro over 2 seeds: +0.0143, 95% t-interval [+0.0103, +0.0182], sign-flip p 0.500

dump vs uniform, sampled macro over 2 seeds: -0.0134, 95% t-interval [-0.0201, -0.0068], sign-flip p 0.500

sec vs uniform, sampled macro over 2 seeds: -0.0010, 95% t-interval [-0.0672, +0.0651], sign-flip p 1.000

## GPU-hours to reach uniform's final level (sampled, the same 100 items per domain)

Learning curves start at the untrained model (0 GPU-h). Fewer GPU-hours = the same accuracy for less compute.

| method | seed | uniform final (at GPU-h) | GPU-h to reach it | compute saved |
|---|---|---|---|---|
| curator_nocal | 3 | 0.2185 at 1.53 | 0.72 | +53% |
| curator_q | 3 | 0.2185 at 1.53 | 1.02 | +33% |
| dump | 3 | 0.2185 at 1.53 | not reached (best 0.2123) | - |
| sec | 3 | 0.2185 at 1.53 | 1.45 | +5% |
| curator_nocal | 4 | 0.2213 at 1.50 | 0.92 | +39% |
| curator_q | 4 | 0.2213 at 1.50 | 1.04 | +31% |
| dump | 4 | 0.2213 at 1.50 | not reached (best 0.2137) | - |
| sec | 4 | 0.2213 at 1.50 | 1.46 | +3% |

## Greedy, paired by seed against uniform (protocol 1)

| method | n | mean diff | 95% t-interval | sign-flip p |
|---|---|---|---|---|
| curator_nocal | 2 | +0.0029 | [-0.0130, +0.0188] | 0.500 |
| curator_q | 2 | +0.0204 | [-0.0272, +0.0681] | 0.500 |
| dump | 2 | +0.0088 | [+0.0035, +0.0140] | 0.500 |
| sec | 2 | +0.0154 | [-0.0111, +0.0419] | 0.500 |

With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only.

## ROI leaderboard: curator_q, seed 3 (2 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 2 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | countdown | +0.0000 (0.0000) | 1.45e-03 | +0.00 | [+0.00, +0.00] |
| 2 | math35 | +0.0000 (0.0000) | 2.16e-03 | +0.00 | [+0.00, +0.00] |
| 3 | noisy | -0.0033 (0.4989) | 1.01e-03 | -0.58 | [-173.45, +172.29] |
| 4 | mbpp | -0.0093 (0.0307) | 5.40e-04 | -3.06 | [-22.96, +16.85] |
| 5 | gsm8k | -0.0497 (0.0684) | 1.13e-03 | -7.83 | [-28.98, +13.32] |


## ROI leaderboard: curator_q, seed 4 (2 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 2 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | gsm8k | +0.0132 (0.0547) | 8.25e-04 | +2.86 | [-20.32, +26.05] |
| 2 | math35 | +0.0209 (0.0392) | 1.64e-03 | +2.28 | [-6.08, +10.63] |
| 3 | noisy | +0.0007 (0.4993) | 8.21e-04 | +0.15 | [-212.58, +212.88] |
| 4 | countdown | +0.0000 (0.0000) | 1.17e-03 | +0.00 | [+0.00, +0.00] |
| 5 | mbpp | +0.0000 (0.0000) | 4.48e-04 | +0.00 | [+0.00, +0.00] |

