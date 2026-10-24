# CURATOR GPU comparison

Sampled accuracy = mean over 4 samples per item at the training temperature (evaluation protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method (uncharged; the sealed test is not used).

| method | seeds | sampled macro (mean) | greedy macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |
|---|---|---|---|---|---|---|---|
| base (untrained) | - | 0.1785 | 0.3000 | 0 | 0 | - | - |
| curator | 3 | 0.2326 | 0.3086 | 1.523 | 32.0 | noisy | 1 |
| curator_nocal | 1 | 0.2354 | 0.3142 | 1.522 | 35.0 | noisy | 1 |
| dirichlet | 2 | 0.2277 | 0.3096 | 1.524 | 32.0 | - | 0 |
| lp | 3 | 0.2354 | 0.3069 | 1.512 | 35.0 | - | 2 |
| pilot_uniform | 1 | 0.2377 | 0.3067 | 1.537 | 33.0 | - | 1 |
| uniform | 3 | 0.2339 | 0.3061 | 1.523 | 34.0 | - | 2 |

## Per domain (sampled accuracy; greedy in brackets; mean over seeds)

| method | gsm8k | math35 | mbpp | countdown |
|---|---|---|---|---|
| base (untrained) | 0.343 (0.567) | 0.133 (0.260) | 0.237 (0.367) | 0.000 (0.007) |
| curator | 0.487 (0.601) | 0.167 (0.273) | 0.276 (0.356) | 0.000 (0.004) |
| curator_nocal | 0.475 (0.613) | 0.166 (0.260) | 0.300 (0.383) | 0.001 (0.000) |
| dirichlet | 0.458 (0.590) | 0.201 (0.315) | 0.252 (0.333) | 0.000 (0.000) |
| lp | 0.496 (0.618) | 0.166 (0.261) | 0.279 (0.339) | 0.001 (0.010) |
| pilot_uniform | 0.489 (0.620) | 0.172 (0.287) | 0.287 (0.317) | 0.002 (0.003) |
| uniform | 0.492 (0.604) | 0.170 (0.260) | 0.274 (0.356) | 0.000 (0.004) |

## Item by item against uniform (same seed, same items)

Paired over items: the SE is the evaluation's, for one seed. Whether a difference holds across training seeds is the across-seed interval below it.

| method | seed | sampled macro diff (SE) | greedy macro diff (SE) | sampled diff per domain |
|---|---|---|---|---|
| curator | 0 | +0.0031 (0.0077) | +0.0058 (0.0110) | gsm8k -0.002, math35 -0.003, mbpp +0.017, countdown +0.000 |
| curator | 1 | -0.0015 (0.0060) | -0.0200 (0.0095) | gsm8k +0.001, math35 +0.006, mbpp -0.013, countdown +0.000 |
| curator | 2 | -0.0056 (0.0086) | +0.0217 (0.0125) | gsm8k -0.015, math35 -0.012, mbpp +0.004, countdown +0.000 |
| curator_nocal | 0 | -0.0079 (0.0080) | +0.0158 (0.0147) | gsm8k -0.028, math35 -0.005, mbpp +0.000, countdown +0.001 |
| dirichlet | 0 | -0.0148 (0.0107) | -0.0008 (0.0186) | gsm8k -0.045, math35 +0.015, mbpp -0.029, countdown +0.000 |
| dirichlet | 1 | +0.0075 (0.0100) | -0.0017 (0.0199) | gsm8k +0.037, math35 +0.018, mbpp -0.025, countdown +0.000 |
| lp | 0 | -0.0069 (0.0074) | +0.0000 (0.0139) | gsm8k -0.003, math35 +0.000, mbpp -0.025, countdown +0.000 |
| lp | 1 | +0.0038 (0.0060) | -0.0108 (0.0094) | gsm8k +0.005, math35 -0.003, mbpp +0.013, countdown +0.000 |
| lp | 2 | +0.0077 (0.0051) | +0.0133 (0.0082) | gsm8k +0.008, math35 -0.008, mbpp +0.029, countdown +0.002 |
| pilot_uniform | 0 | -0.0056 (0.0064) | +0.0083 (0.0121) | gsm8k -0.013, math35 +0.002, mbpp -0.013, countdown +0.002 |

curator vs uniform, sampled macro over 3 seeds: -0.0013, 95% t-interval [-0.0122, +0.0096], sign-flip p 0.750

dirichlet vs uniform, sampled macro over 2 seeds: -0.0036, 95% t-interval [-0.1453, +0.1380], sign-flip p 1.000

lp vs uniform, sampled macro over 3 seeds: +0.0015, 95% t-interval [-0.0172, +0.0203], sign-flip p 0.750

## Gain over the untrained model (item by item, sampled)

| method | seed | sampled macro gain (SE) | per domain |
|---|---|---|---|
| curator | 0 | +0.0679 (0.0090) | gsm8k +0.158, math35 +0.035, mbpp +0.079, countdown +0.000 |
| curator | 1 | +0.0467 (0.0079) | gsm8k +0.142, math35 +0.036, mbpp +0.008, countdown +0.000 |
| curator | 2 | +0.0475 (0.0080) | gsm8k +0.131, math35 +0.030, mbpp +0.029, countdown +0.000 |
| curator_nocal | 0 | +0.0569 (0.0088) | gsm8k +0.132, math35 +0.033, mbpp +0.062, countdown +0.001 |
| dirichlet | 0 | +0.0483 (0.0111) | gsm8k +0.105, math35 +0.055, mbpp +0.033, countdown +0.000 |
| dirichlet | 1 | +0.0471 (0.0099) | gsm8k +0.145, math35 +0.048, mbpp -0.004, countdown +0.000 |
| lp | 0 | +0.0579 (0.0096) | gsm8k +0.157, math35 +0.037, mbpp +0.037, countdown +0.000 |
| lp | 1 | +0.0519 (0.0091) | gsm8k +0.147, math35 +0.028, mbpp +0.033, countdown +0.000 |
| lp | 2 | +0.0608 (0.0100) | gsm8k +0.154, math35 +0.033, mbpp +0.054, countdown +0.002 |
| pilot_uniform | 0 | +0.0592 (0.0094) | gsm8k +0.146, math35 +0.039, mbpp +0.050, countdown +0.002 |
| uniform | 0 | +0.0648 (0.0096) | gsm8k +0.159, math35 +0.037, mbpp +0.062, countdown +0.000 |
| uniform | 1 | +0.0481 (0.0087) | gsm8k +0.142, math35 +0.030, mbpp +0.021, countdown +0.000 |
| uniform | 2 | +0.0531 (0.0098) | gsm8k +0.146, math35 +0.042, mbpp +0.025, countdown +0.000 |

## GPU-hours to reach uniform's final level (sampled, the same 100 items per domain)

Learning curves start at the untrained model (0 GPU-h). Fewer GPU-hours = the same accuracy for less compute.

| method | seed | uniform final (at GPU-h) | GPU-h to reach it | compute saved |
|---|---|---|---|---|
| curator | 0 | 0.2313 at 1.50 | 1.46 | +3% |
| curator_nocal | 0 | 0.2313 at 1.50 | not reached (best 0.2275) | - |
| dirichlet | 0 | 0.2313 at 1.50 | not reached (best 0.2283) | - |
| lp | 0 | 0.2313 at 1.50 | not reached (best 0.2244) | - |
| pilot_uniform | 0 | 0.2313 at 1.50 | 1.48 | +1% |
| curator | 1 | 0.2215 at 1.54 | not reached (best 0.2169) | - |
| dirichlet | 1 | 0.2215 at 1.54 | 1.34 | +13% |
| lp | 1 | 0.2215 at 1.54 | not reached (best 0.2190) | - |
| curator | 2 | 0.2219 at 1.52 | not reached (best 0.2167) | - |
| lp | 2 | 0.2219 at 1.52 | 1.08 | +29% |

## Greedy, paired by seed against uniform (protocol 1)

| method | n | mean diff | 95% t-interval | sign-flip p |
|---|---|---|---|---|
| curator | 3 | +0.0025 | [-0.0497, +0.0547] | 0.750 |
| curator_nocal | 1 | +0.0158 | one seed: no interval | - |
| dirichlet | 2 | -0.0004 | [-0.0057, +0.0049] | 1.000 |
| lp | 3 | +0.0008 | [-0.0292, +0.0309] | 1.000 |
| pilot_uniform | 1 | +0.0083 | one seed: no interval | - |

With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only.

## ROI leaderboard: curator, seed 0 (3 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 3 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | noisy | +0.0727 (0.4687) | 1.52e-03 | +14.93 | [-173.62, +203.48] |
| 2 | mbpp | +0.0102 (0.0147) | 6.04e-04 | +5.28 | [-9.58, +20.14] |
| 3 | gsm8k | +0.0251 (0.0655) | 1.78e-03 | +4.41 | [-18.12, +26.93] |
| 4 | countdown | +0.0000 (0.0000) | 1.86e-03 | +0.00 | [+0.00, +0.00] |
| 5 | math35 | +0.0000 (0.0000) | 3.72e-03 | +0.00 | [+0.00, +0.00] |


## ROI leaderboard: curator, seed 1 (3 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 3 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | gsm8k | +0.0372 (0.0247) | 1.20e-03 | +9.66 | [-2.90, +22.21] |
| 2 | countdown | +0.0000 (0.1569) | 1.88e-03 | +0.00 | [-51.01, +51.01] |
| 3 | math35 | +0.0000 (0.0232) | 2.49e-03 | +0.00 | [-5.69, +5.69] |
| 4 | mbpp | +0.0000 (0.0000) | 6.56e-04 | +0.00 | [+0.00, +0.00] |
| 5 | noisy | -0.0205 (0.0610) | 1.28e-03 | -5.02 | [-34.24, +24.20] |


## ROI leaderboard: curator, seed 2 (3 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): noisy. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 3 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | countdown | +0.0000 (0.0000) | 1.58e-03 | +0.00 | [+0.00, +0.00] |
| 2 | mbpp | +0.0000 (0.0000) | 6.14e-04 | +0.00 | [+0.00, +0.00] |
| 3 | math35 | -0.0162 (0.0256) | 2.14e-03 | -2.37 | [-9.70, +4.96] |
| 4 | gsm8k | -0.0120 (0.0275) | 1.09e-03 | -3.43 | [-18.89, +12.04] |
| 5 | noisy | -0.0690 (0.4963) | 1.11e-03 | -19.51 | [-294.38, +255.37] |


## ROI leaderboard: dirichlet, seed 0 (6 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

**Not a ranking:** with 6 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | noisy | +0.0683 (0.0410) | 1.15e-03 | +37.04 | [-6.52, +80.60] |
| 2 | mbpp | +0.0135 (0.0118) | 5.85e-04 | +14.42 | [-10.35, +39.20] |
| 3 | math35 | +0.0176 (0.0155) | 2.18e-03 | +5.04 | [-3.67, +13.74] |
| 4 | countdown | -0.0099 (0.0109) | 1.77e-03 | -3.50 | [-11.04, +4.04] |
| 5 | gsm8k | -0.0331 (0.0297) | 1.19e-03 | -17.33 | [-47.78, +13.11] |


## ROI leaderboard: dirichlet, seed 1 (6 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

**Not a ranking:** with 6 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | noisy | +0.0390 (0.0434) | 1.20e-03 | +20.34 | [-24.04, +64.72] |
| 2 | mbpp | -0.0008 (0.0106) | 6.76e-04 | -0.78 | [-19.94, +18.39] |
| 3 | countdown | -0.0051 (0.0101) | 1.98e-03 | -1.62 | [-7.90, +4.66] |
| 4 | math35 | -0.0069 (0.0213) | 2.22e-03 | -1.94 | [-13.72, +9.83] |
| 5 | gsm8k | -0.0247 (0.0379) | 1.18e-03 | -13.04 | [-52.33, +26.25] |

