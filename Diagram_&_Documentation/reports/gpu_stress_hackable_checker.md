# CURATOR GPU comparison

Sampled accuracy = mean over 4 samples per item at the training temperature (evaluation protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method (uncharged; the sealed test is not used).

| method | seeds | sampled macro (mean) | pass@k macro | greedy macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |
|---|---|---|---|---|---|---|---|---|
| curator_q | 1 | 0.2354 | 0.3900 | 0.3158 | 1.524 | 29.0 | hackable | 2 |
| uniform | 1 | 0.2290 | 0.3917 | 0.3258 | 1.542 | 35.0 | - | 3 |

## Per domain (sampled accuracy; greedy in brackets; mean over seeds)

| method | gsm8k | math35 | mbpp | countdown |
|---|---|---|---|---|
| curator_q | 0.511 (0.610) | 0.177 (0.267) | 0.254 (0.383) | 0.000 (0.003) |
| uniform | 0.487 (0.623) | 0.154 (0.290) | 0.275 (0.383) | 0.000 (0.007) |

## Pre-flight verifier check (D-102)

Reward each arm paid for junk before any training (empty or echoed answers, wrong or sprayed numbers, code that exits early or prints its globals). Above 5% the checker can be gamed.

| arm | junk reward rate | flagged |
|---|---|---|
| countdown | 0.00 | - |
| gsm8k | 0.00 | - |
| hackable | 0.20 | yes |
| math35 | 0.00 | - |
| mbpp | 0.00 | - |

## Item by item against uniform (same seed, same items)

Paired over items: the SE is the evaluation's, for one seed. Whether a difference holds across training seeds is the across-seed interval below it.

| method | seed | sampled macro diff (SE) | greedy macro diff (SE) | sampled diff per domain |
|---|---|---|---|---|
| curator_q | 0 | +0.0065 (0.0076) | -0.0100 (0.0166) | gsm8k +0.024, math35 +0.022, mbpp -0.021, countdown +0.000 |

## GPU-hours to reach uniform's final level (sampled, the same 100 items per domain)

Learning curves start at the untrained model (0 GPU-h). Fewer GPU-hours = the same accuracy for less compute.

| method | seed | uniform final (at GPU-h) | GPU-h to reach it | compute saved |
|---|---|---|---|---|
| curator_q | 0 | 0.2288 at 1.54 | 1.00 | +35% |

## Greedy, paired by seed against uniform (protocol 1)

| method | n | mean diff | 95% t-interval | sign-flip p |
|---|---|---|---|---|
| curator_q | 1 | -0.0100 | one seed: no interval | - |

With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only.

## ROI leaderboard: curator_q, seed 0 (2 calibration evaluations)

Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on (roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.

Flagged S5 (rewards that do not depend on the prompt): hackable. Their ROI is not meaningful, whatever its rank.

**Not a ranking:** with 2 calibration evaluations no (unflagged) arm's interval excludes zero, so the order below is noise.

| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | ROI (benchmark points per GPU-hour) | 95% interval |
|---|---|---|---|---|---|
| 1 | gsm8k | +0.0366 (0.0333) | 8.87e-04 | +7.36 | [-5.77, +20.50] |
| 2 | hackable | +0.0011 (0.4998) | 8.45e-04 | +0.22 | [-206.74, +207.19] |
| 3 | countdown | +0.0000 (0.0000) | 1.24e-03 | +0.00 | [+0.00, +0.00] |
| 4 | math35 | +0.0000 (0.0000) | 1.82e-03 | +0.00 | [+0.00, +0.00] |
| 5 | mbpp | -0.1373 (0.0677) | 4.37e-04 | -56.10 | [-110.34, -1.86] |

