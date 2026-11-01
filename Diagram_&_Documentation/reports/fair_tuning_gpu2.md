# Tuning at the GPU shape, realism v2 (tuning seeds; exploratory)

Seeds 0..39; scenarios ['S-I-R1-GPU2', 'S-I-R2-GPU2'] (the S-J-R twin held out); objective = mean relative noiseless-score gain over Uniform; rule = best within one SE, then least concentrated (D-55). Curator also carries {"signals.reward_audit.enabled": true, "scheduler.s5_cap": "floor", "calib.mismatch_windows": 3}.


## B. Fair re-tuning (8 configurations per method)

| method | chosen point | objective (rel. gain over Uniform) | SE | norm. entropy |
|---|---|---|---|---|
| curator | `{"gamma": 1.0, "exploration_coef": 0.25, "tau": 1.0}` | +0.1157 | 0.0076 | 0.68 |
| ucb | `{"exploration_coef": 1.0, "tau": 0.3}` | +0.0398 | 0.0044 | 0.54 |
| lp | `{"tau": 0.75}` | +0.0585 | 0.0034 | 0.66 |
| sec | `{"alpha": 0.1, "tau": 0.4}` | +0.0116 | 0.0039 | 0.92 |
| dump | `{"exploration_coef": 2.0, "tau": 0.4, "window_rounds": 10}` | +0.0120 | 0.0042 | 0.96 |

All points:

| method | point | objective | SE | entropy |
|---|---|---|---|---|
| curator | `{"gamma": 1.0, "exploration_coef": 0.1, "tau": 0.3}` | +0.1209 | 0.0152 | 0.43 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.1, "tau": 1.0}` | +0.1180 | 0.0070 | 0.67 |
| curator | `{"gamma": 1.0, "exploration_coef": 0.25, "tau": 0.3}` | +0.1174 | 0.0091 | 0.55 |
| curator | `{"gamma": 1.0, "exploration_coef": 0.25, "tau": 1.0}` | +0.1157 | 0.0076 | 0.68 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.1, "tau": 0.3}` | +0.1143 | 0.0102 | 0.52 |
| curator | `{"gamma": 1.0, "exploration_coef": 0.1, "tau": 1.0}` | +0.1131 | 0.0077 | 0.67 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.25, "tau": 1.0}` | +0.0979 | 0.0042 | 0.71 |
| curator | `{"gamma": 0.95, "exploration_coef": 0.25, "tau": 0.3}` | +0.0923 | 0.0052 | 0.64 |
| dump | `{"exploration_coef": 2.0, "tau": 0.4, "window_rounds": 0}` | +0.0134 | 0.0040 | 0.94 |
| dump | `{"exploration_coef": 0.3, "tau": 0.4, "window_rounds": 10}` | +0.0128 | 0.0041 | 0.92 |
| dump | `{"exploration_coef": 0.3, "tau": 0.4, "window_rounds": 0}` | +0.0125 | 0.0041 | 0.92 |
| dump | `{"exploration_coef": 2.0, "tau": 0.4, "window_rounds": 10}` | +0.0120 | 0.0042 | 0.96 |
| dump | `{"exploration_coef": 2.0, "tau": 0.1, "window_rounds": 10}` | +0.0003 | 0.0036 | 0.86 |
| dump | `{"exploration_coef": 2.0, "tau": 0.1, "window_rounds": 0}` | -0.0200 | 0.0045 | 0.75 |
| dump | `{"exploration_coef": 0.3, "tau": 0.1, "window_rounds": 10}` | -0.0597 | 0.0039 | 0.57 |
| dump | `{"exploration_coef": 0.3, "tau": 0.1, "window_rounds": 0}` | -0.0682 | 0.0040 | 0.50 |
| lp | `{"tau": 0.3}` | +0.0639 | 0.0056 | 0.36 |
| lp | `{"tau": 0.75}` | +0.0585 | 0.0034 | 0.66 |
| lp | `{"tau": 0.2}` | +0.0567 | 0.0045 | 0.32 |
| lp | `{"tau": 0.5}` | +0.0542 | 0.0043 | 0.48 |
| lp | `{"tau": 1.0}` | +0.0540 | 0.0043 | 0.77 |
| lp | `{"tau": 0.1}` | +0.0513 | 0.0042 | 0.28 |
| lp | `{"tau": 1.5}` | +0.0506 | 0.0049 | 0.89 |
| lp | `{"tau": 2.0}` | +0.0319 | 0.0038 | 0.93 |
| sec | `{"alpha": 1.0, "tau": 0.4}` | +0.0148 | 0.0037 | 0.90 |
| sec | `{"alpha": 0.1, "tau": 0.4}` | +0.0116 | 0.0039 | 0.92 |
| sec | `{"alpha": 0.3, "tau": 0.4}` | +0.0099 | 0.0040 | 0.91 |
| sec | `{"alpha": 0.6, "tau": 0.4}` | +0.0085 | 0.0037 | 0.91 |
| sec | `{"alpha": 1.0, "tau": 0.1}` | -0.0180 | 0.0042 | 0.53 |
| sec | `{"alpha": 0.6, "tau": 0.1}` | -0.0618 | 0.0039 | 0.53 |
| sec | `{"alpha": 0.1, "tau": 0.1}` | -0.0667 | 0.0043 | 0.51 |
| sec | `{"alpha": 0.3, "tau": 0.1}` | -0.0717 | 0.0038 | 0.48 |
| ucb | `{"exploration_coef": 0.1, "tau": 0.3}` | +0.0615 | 0.0255 | 0.40 |
| ucb | `{"exploration_coef": 0.25, "tau": 0.3}` | +0.0408 | 0.0231 | 0.41 |
| ucb | `{"exploration_coef": 1.0, "tau": 0.3}` | +0.0398 | 0.0044 | 0.54 |
| ucb | `{"exploration_coef": 0.5, "tau": 0.3}` | +0.0398 | 0.0104 | 0.44 |
| ucb | `{"exploration_coef": 1.0, "tau": 1.0}` | +0.0336 | 0.0049 | 0.78 |
| ucb | `{"exploration_coef": 0.1, "tau": 1.0}` | +0.0254 | 0.0117 | 0.81 |
| ucb | `{"exploration_coef": 0.5, "tau": 1.0}` | +0.0220 | 0.0084 | 0.79 |
| ucb | `{"exploration_coef": 0.25, "tau": 1.0}` | +0.0204 | 0.0099 | 0.80 |
