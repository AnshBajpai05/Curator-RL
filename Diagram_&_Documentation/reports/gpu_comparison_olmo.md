# CURATOR GPU comparison

Sampled accuracy = mean over 4 samples per item at the training temperature (evaluation protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method (uncharged; the sealed test is not used).

| method | seeds | sampled macro (mean) | pass@k macro | greedy macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |
|---|---|---|---|---|---|---|---|---|
| base (untrained) | - | 0.1765 | 0.2967 | 0.2183 | 0 | 0 | - | - |
| curator | 1 | 0.1781 | 0.3025 | 0.2175 | 1.556 | 15.0 | noisy | 3 |
| uniform | 1 | 0.1758 | 0.2892 | 0.2175 | 1.597 | 14.0 | - | 0 |

## Per domain (sampled accuracy; greedy in brackets; mean over seeds)

| method | gsm8k | math35 | mbpp | countdown |
|---|---|---|---|---|
| base (untrained) | 0.553 (0.650) | 0.072 (0.087) | 0.079 (0.133) | 0.001 (0.003) |
| curator | 0.557 (0.667) | 0.072 (0.067) | 0.083 (0.133) | 0.000 (0.003) |
| uniform | 0.556 (0.653) | 0.068 (0.077) | 0.079 (0.133) | 0.000 (0.007) |

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
| curator | 0 | +0.0023 (0.0037) | +0.0000 (0.0073) | gsm8k +0.001, math35 +0.004, mbpp +0.004, countdown +0.000 |

## Gain over the untrained model (item by item, sampled)

| method | seed | sampled macro gain (SE) | per domain |
|---|---|---|---|
| curator | 0 | +0.0017 (0.0039) | gsm8k +0.003, math35 +0.000, mbpp +0.004, countdown -0.001 |
| uniform | 0 | -0.0006 (0.0031) | gsm8k +0.003, math35 -0.004, mbpp +0.000, countdown -0.001 |

## GPU-hours to reach uniform's final level (sampled, the same 100 items per domain)

Learning curves start at the untrained model (0 GPU-h). Fewer GPU-hours = the same accuracy for less compute.

| method | seed | uniform final (at GPU-h) | GPU-h to reach it | compute saved |
|---|---|---|---|---|
| curator | 0 | 0.1779 at 1.60 | 1.23 | +23% |

## Greedy, paired by seed against uniform (protocol 1)

| method | n | mean diff | 95% t-interval | sign-flip p |
|---|---|---|---|---|
| curator | 1 | +0.0000 | one seed: no interval | - |

With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only.
