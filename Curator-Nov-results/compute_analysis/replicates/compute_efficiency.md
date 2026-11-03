# Compute efficiency against Uniform

Grid (GPU-hours): [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]. `delta` = mean accuracy difference at matched compute (percentage points); `saving` = share of Uniform's compute saved to reach Uniform's final smoothed accuracy (not reached = 0). A block is one seed on one account; every method is paired with Uniform of the same block.
Rule: mean >= 2 SE and at least 3 of 3 blocks positive (in the expected direction). Primary method: `curator_q`.

| Method | Expected | Blocks | delta per block | delta mean (SE) | 95% t-interval | p | Verdict (delta) | saving per block | saving mean |
|---|---|---|---|---|---|---|---|---|---|
| curator_q | ahead | 6 | acct3:s3 -0.28, acct3:s4 -0.64, acct4:s3 +0.53, acct4:s4 +0.71, acct5:s3 +0.93, acct5:s4 -0.00 | +0.21 (0.25) | [-0.44, +0.85] | 0.438 | descriptive | +32%, not reached, +31%, +28%, +48%, +15% | +26% |
| curator_nocal | ahead | 6 | acct3:s3 +0.35, acct3:s4 -0.54, acct4:s3 +1.01, acct4:s4 +0.76, acct5:s3 +0.99, acct5:s4 +0.10 | +0.44 (0.25) | [-0.19, +1.08] | 0.156 | descriptive | +10%, not reached, +49%, +37%, +55%, +26% | +29% |
| sec | behind | 6 | acct3:s3 +0.15, acct3:s4 -1.28, acct4:s3 -0.16, acct4:s4 -0.12, acct5:s3 +0.65, acct5:s4 -0.05 | -0.14 (0.26) | [-0.80, +0.53] | 0.625 | descriptive | +24%, not reached, +3%, not reached, +34%, not reached | +10% |
| dump | behind | 6 | acct3:s3 -0.51, acct3:s4 -0.89, acct4:s3 -0.17, acct4:s4 -0.69, acct5:s3 +0.37, acct5:s4 -0.63 | -0.42 (0.19) | [-0.90, +0.06] | 0.094 | descriptive | not reached, not reached, not reached, not reached, +34%, not reached | +6% |

Points per learning curve (incl. the untrained model): [9, 10, 11, 12, 13, 14].

## Final accuracy against Uniform (descriptive)

Item-paired sampled macro at the end of each run, same block; not part of any rule.

| Method | Blocks | per block | mean (SE) | 95% t-interval | blocks above Uniform |
|---|---|---|---|---|---|
| curator_q | 6 | acct3:s3 +1.73, acct3:s4 +0.35, acct4:s3 +1.40, acct4:s4 +1.46, acct5:s3 +2.25, acct5:s4 +0.96 | +1.36 (0.27) | [+0.68, +2.04] | 6 of 6 |
| curator_nocal | 6 | acct3:s3 +2.12, acct3:s4 -0.31, acct4:s3 +0.73, acct4:s4 +0.79, acct5:s3 +0.98, acct5:s4 +0.88 | +0.86 (0.32) | [+0.05, +1.68] | 5 of 6 |
| sec | 6 | acct3:s3 +1.40, acct3:s4 -0.98, acct4:s3 +0.42, acct4:s4 -0.62, acct5:s3 +0.48, acct5:s4 +0.23 | +0.15 (0.35) | [-0.74, +1.04] | 4 of 6 |
| dump | 6 | acct3:s3 -0.04, acct3:s4 -0.69, acct4:s3 -1.40, acct4:s4 -1.29, acct5:s3 +0.50, acct5:s4 -0.52 | -0.57 (0.30) | [-1.34, +0.19] | 1 of 6 |

## Estimated GPU-hours saved (descriptive)

Every run spends its full budget, so nothing is actually returned: this is the GPU Uniform needed minus the GPU the method needed to reach Uniform's final smoothed accuracy (not reached = 0).

| Method | Saved per block (GPU-h) | Mean saved |
|---|---|---|
| curator_q | acct3:s3 0.48, acct3:s4 0.00, acct4:s3 0.47, acct4:s4 0.42, acct5:s3 0.75, acct5:s4 0.24 | 0.39 GPU-h |
| curator_nocal | acct3:s3 0.15, acct3:s4 0.00, acct4:s3 0.75, acct4:s4 0.55, acct5:s3 0.86, acct5:s4 0.39 | 0.45 GPU-h |
| sec | acct3:s3 0.36, acct3:s4 0.00, acct4:s3 0.05, acct4:s4 0.00, acct5:s3 0.53, acct5:s4 0.00 | 0.16 GPU-h |
| dump | acct3:s3 0.00, acct3:s4 0.00, acct4:s3 0.00, acct4:s4 0.00, acct5:s3 0.54, acct5:s4 0.00 | 0.09 GPU-h |

## Where the GPU went (descriptive)

GPU-hours charged to each environment (generation + scoring + update, `cost/attribution.py`), mean over finished runs; the share is of the run's environment GPU-hours.

| Method | Runs | Broken reward (noisy) | Countdown (never solved) | Useful environments | Total |
|---|---|---|---|---|---|
| uniform | 6 | 0.25 GPU-h, 16% | 0.39 GPU-h, 26% | 0.88 GPU-h | 1.53 GPU-h |
| curator_q | 6 | 0.09 GPU-h, 6% | 0.13 GPU-h, 8% | 1.29 GPU-h | 1.50 GPU-h |
| curator_nocal | 6 | 0.18 GPU-h, 12% | 0.12 GPU-h, 8% | 1.22 GPU-h | 1.52 GPU-h |
| sec | 6 | 0.45 GPU-h, 29% | 0.23 GPU-h, 15% | 0.85 GPU-h | 1.53 GPU-h |
| dump | 6 | 0.41 GPU-h, 27% | 0.24 GPU-h, 16% | 0.87 GPU-h | 1.52 GPU-h |

Dollar figures appear with `--usd-per-gpu-hour <price>` (the Kaggle runs themselves were free).

## Reward checks (descriptive)

| Method | Run | Environments ever in S5 | First S5 round of the broken reward |
|---|---|---|---|
| curator_q | acct3:s3 | noisy | 1 |
| curator_q | acct3:s4 | noisy | 1 |
| curator_q | acct4:s3 | noisy | 1 |
| curator_q | acct4:s4 | noisy | 1 |
| curator_q | acct5:s3 | noisy | 1 |
| curator_q | acct5:s4 | noisy | 1 |
| curator_nocal | acct3:s3 | none | None |
| curator_nocal | acct3:s4 | noisy | 16 |
| curator_nocal | acct4:s3 | noisy | 10 |
| curator_nocal | acct4:s4 | noisy | 17 |
| curator_nocal | acct5:s3 | noisy | 16 |
| curator_nocal | acct5:s4 | noisy | 17 |
