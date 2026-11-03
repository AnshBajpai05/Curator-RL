# Compute efficiency against Uniform

Grid (GPU-hours): [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]. `delta` = mean accuracy difference at matched compute (percentage points); `saving` = share of Uniform's compute saved to reach Uniform's final smoothed accuracy (not reached = 0). A block is one seed on one account; every method is paired with Uniform of the same block.
Rule: mean >= 2 SE and at least 3 of 3 blocks positive (in the expected direction). Primary method: `curator_q`. Blocks kept: [('acct3', 3), ('acct4', 4), ('acct5', 5)].

| Method | Expected | Blocks | delta per block | delta mean (SE) | 95% t-interval | p | Verdict (delta) | saving per block | saving mean |
|---|---|---|---|---|---|---|---|---|---|
| curator_q | ahead | 2 | acct3:s3 -0.28, acct4:s4 +0.71 | +0.21 (0.49) | [-6.08, +6.50] | 1.000 | PENDING (2 of 3 seeds) | +32%, +28% | +30% |
| curator_nocal | ahead | 2 | acct3:s3 +0.35, acct4:s4 +0.76 | +0.56 (0.20) | [-2.04, +3.15] | 0.500 | PENDING (2 of 3 seeds) | +10%, +37% | +23% |
| sec | behind | 2 | acct3:s3 +0.15, acct4:s4 -0.12 | +0.02 (0.14) | [-1.70, +1.74] | 1.000 | PENDING (2 of 3 seeds) | +24%, not reached | +12% |
| dump | behind | 2 | acct3:s3 -0.51, acct4:s4 -0.69 | -0.60 (0.09) | [-1.81, +0.61] | 0.500 | PENDING (2 of 3 seeds) | not reached, not reached | +0% |

Points per learning curve (incl. the untrained model): [10, 11, 12, 13, 14].

## Final accuracy against Uniform (descriptive)

Item-paired sampled macro at the end of each run, same block; not part of any rule.

| Method | Blocks | per block | mean (SE) | 95% t-interval | blocks above Uniform |
|---|---|---|---|---|---|
| curator_q | 2 | acct3:s3 +1.73, acct4:s4 +1.46 | +1.59 (0.14) | [-0.13, +3.31] | 2 of 2 |
| curator_nocal | 2 | acct3:s3 +2.12, acct4:s4 +0.79 | +1.46 (0.67) | [-7.01, +9.93] | 2 of 2 |
| sec | 2 | acct3:s3 +1.40, acct4:s4 -0.62 | +0.39 (1.01) | [-12.45, +13.22] | 1 of 2 |
| dump | 2 | acct3:s3 -0.04, acct4:s4 -1.29 | -0.67 (0.62) | [-8.61, +7.27] | 0 of 2 |

## Estimated GPU-hours saved (descriptive)

Every run spends its full budget, so nothing is actually returned: this is the GPU Uniform needed minus the GPU the method needed to reach Uniform's final smoothed accuracy (not reached = 0).

| Method | Saved per block (GPU-h) | Mean saved |
|---|---|---|
| curator_q | acct3:s3 0.48, acct4:s4 0.42 | 0.45 GPU-h |
| curator_nocal | acct3:s3 0.15, acct4:s4 0.55 | 0.35 GPU-h |
| sec | acct3:s3 0.36, acct4:s4 0.00 | 0.18 GPU-h |
| dump | acct3:s3 0.00, acct4:s4 0.00 | 0.00 GPU-h |

## Where the GPU went (descriptive)

GPU-hours charged to each environment (generation + scoring + update, `cost/attribution.py`), mean over finished runs; the share is of the run's environment GPU-hours.

| Method | Runs | Broken reward (noisy) | Countdown (never solved) | Useful environments | Total |
|---|---|---|---|---|---|
| uniform | 2 | 0.24 GPU-h, 16% | 0.39 GPU-h, 26% | 0.87 GPU-h | 1.50 GPU-h |
| curator_q | 2 | 0.09 GPU-h, 6% | 0.13 GPU-h, 8% | 1.29 GPU-h | 1.50 GPU-h |
| curator_nocal | 2 | 0.22 GPU-h, 15% | 0.12 GPU-h, 8% | 1.17 GPU-h | 1.51 GPU-h |
| sec | 2 | 0.46 GPU-h, 30% | 0.23 GPU-h, 15% | 0.84 GPU-h | 1.53 GPU-h |
| dump | 2 | 0.41 GPU-h, 27% | 0.23 GPU-h, 15% | 0.87 GPU-h | 1.51 GPU-h |

Dollar figures appear with `--usd-per-gpu-hour <price>` (the Kaggle runs themselves were free).

## Reward checks (descriptive)

| Method | Run | Environments ever in S5 | First S5 round of the broken reward |
|---|---|---|---|
| curator_q | acct3:s3 | noisy | 1 |
| curator_q | acct4:s4 | noisy | 1 |
| curator_nocal | acct3:s3 | none | None |
| curator_nocal | acct4:s4 | noisy | 17 |
