# Tuning at the GPU shape (tuning seeds; exploratory, nothing adopted)

Seeds 0..39; scenarios ['S-I-R1-GPU', 'S-I-R2-GPU'] (S-J-R-GPU held out); objective = mean relative noiseless-score gain over Uniform; rule = best within one SE, then least concentrated (D-55).

## A. Calibration design vs noisy-arm detection (post-audit Curator)

| K | items/slice | k_min | detected (S5) | false flags | calib cost | score |
|---|---|---|---|---|---|---|
| 10 | 100 | 2 | 0.14 | 0.006 | 0.023 | 0.2179 |
| 5 | 100 | 2 | 0.20 | 0.009 | 0.043 | 0.2176 |
| 5 | 200 | 2 | 0.24 | 0.031 | 0.083 | 0.2159 |
| 3 | 100 | 2 | 0.14 | 0.034 | 0.068 | 0.2156 |
| 3 | 200 | 1 | 0.28 | 0.053 | 0.128 | 0.2138 |
| 2 | 200 | 1 | 0.28 | 0.059 | 0.180 | 0.2114 |

## B. Fair re-tuning (8 configurations per method)

| method | chosen point | objective (rel. gain over Uniform) | SE | norm. entropy |
|---|---|---|---|---|
| curator | `{"gamma": 1.0, "exploration_coef": 0.1, "tau": 0.3}` | +0.0603 | 0.0177 | 0.46 |
| ucb | `{"exploration_coef": 0.25, "tau": 0.3}` | +0.0492 | 0.0222 | 0.43 |
| lp | `{"tau": 0.5}` | +0.0615 | 0.0058 | 0.49 |
| sec | `{"alpha": 1.0, "tau": 0.4}` | +0.0194 | 0.0041 | 0.90 |
| dump | `{"exploration_coef": 0.3, "tau": 0.4, "window_rounds": 10}` | +0.0146 | 0.0045 | 0.92 |

All points:

| method | point | objective | SE | entropy |
|---|---|---|---|---|
| curator | `{"gamma": 1.0, "exploration_coef": 0.1, "tau": 0.3}` | +0.0603 | 0.0177 | 0.46 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.1, "tau": 0.3}` | +0.0439 | 0.0146 | 0.43 |
| curator | `{"gamma": 1.0, "exploration_coef": 0.25, "tau": 0.3}` | +0.0423 | 0.0144 | 0.48 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.25, "tau": 0.3}` | +0.0414 | 0.0078 | 0.57 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.25, "tau": 1.0}` | +0.0383 | 0.0069 | 0.76 |
| curator | `{"gamma": 1.0, "exploration_coef": 0.1, "tau": 1.0}` | +0.0373 | 0.0099 | 0.76 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.1, "tau": 1.0}` | +0.0333 | 0.0085 | 0.72 |
| curator | `{"gamma": 1.0, "exploration_coef": 0.25, "tau": 1.0}` | +0.0326 | 0.0104 | 0.74 |
| dump | `{"exploration_coef": 0.3, "tau": 0.4, "window_rounds": 10}` | +0.0146 | 0.0045 | 0.92 |
| dump | `{"exploration_coef": 0.3, "tau": 0.4, "window_rounds": 0}` | +0.0135 | 0.0044 | 0.92 |
| dump | `{"exploration_coef": 2.0, "tau": 0.4, "window_rounds": 0}` | +0.0088 | 0.0040 | 0.94 |
| dump | `{"exploration_coef": 2.0, "tau": 0.4, "window_rounds": 10}` | +0.0088 | 0.0051 | 0.96 |
| dump | `{"exploration_coef": 2.0, "tau": 0.1, "window_rounds": 10}` | +0.0008 | 0.0041 | 0.86 |
| dump | `{"exploration_coef": 2.0, "tau": 0.1, "window_rounds": 0}` | -0.0147 | 0.0053 | 0.75 |
| dump | `{"exploration_coef": 0.3, "tau": 0.1, "window_rounds": 10}` | -0.0581 | 0.0043 | 0.57 |
| dump | `{"exploration_coef": 0.3, "tau": 0.1, "window_rounds": 0}` | -0.0685 | 0.0046 | 0.50 |
| lp | `{"tau": 0.2}` | +0.0645 | 0.0042 | 0.32 |
| lp | `{"tau": 0.3}` | +0.0630 | 0.0045 | 0.36 |
| lp | `{"tau": 0.1}` | +0.0623 | 0.0047 | 0.28 |
| lp | `{"tau": 0.5}` | +0.0615 | 0.0058 | 0.49 |
| lp | `{"tau": 0.75}` | +0.0588 | 0.0038 | 0.66 |
| lp | `{"tau": 1.0}` | +0.0556 | 0.0045 | 0.77 |
| lp | `{"tau": 1.5}` | +0.0478 | 0.0047 | 0.89 |
| lp | `{"tau": 2.0}` | +0.0324 | 0.0039 | 0.93 |
| sec | `{"alpha": 1.0, "tau": 0.4}` | +0.0194 | 0.0041 | 0.90 |
| sec | `{"alpha": 0.3, "tau": 0.4}` | +0.0150 | 0.0033 | 0.91 |
| sec | `{"alpha": 0.6, "tau": 0.4}` | +0.0132 | 0.0042 | 0.91 |
| sec | `{"alpha": 0.1, "tau": 0.4}` | +0.0072 | 0.0041 | 0.92 |
| sec | `{"alpha": 1.0, "tau": 0.1}` | -0.0168 | 0.0045 | 0.54 |
| sec | `{"alpha": 0.6, "tau": 0.1}` | -0.0565 | 0.0044 | 0.53 |
| sec | `{"alpha": 0.1, "tau": 0.1}` | -0.0667 | 0.0042 | 0.50 |
| sec | `{"alpha": 0.3, "tau": 0.1}` | -0.0712 | 0.0044 | 0.49 |
| ucb | `{"exploration_coef": 0.1, "tau": 0.3}` | +0.0718 | 0.0255 | 0.39 |
| ucb | `{"exploration_coef": 0.25, "tau": 0.3}` | +0.0492 | 0.0222 | 0.43 |
| ucb | `{"exploration_coef": 0.5, "tau": 0.3}` | +0.0425 | 0.0112 | 0.45 |
| ucb | `{"exploration_coef": 1.0, "tau": 0.3}` | +0.0290 | 0.0044 | 0.53 |
| ucb | `{"exploration_coef": 1.0, "tau": 1.0}` | +0.0273 | 0.0045 | 0.78 |
| ucb | `{"exploration_coef": 0.1, "tau": 1.0}` | +0.0271 | 0.0126 | 0.80 |
| ucb | `{"exploration_coef": 0.5, "tau": 1.0}` | +0.0215 | 0.0084 | 0.79 |
| ucb | `{"exploration_coef": 0.25, "tau": 1.0}` | +0.0215 | 0.0106 | 0.80 |
