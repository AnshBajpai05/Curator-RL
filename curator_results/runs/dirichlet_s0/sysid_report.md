# System identification report (one randomized-mixture run)

## Per-arm training cost (NNLS on step time vs tokens)

Fixed overhead 65.57 s/step, R^2 0.536, 66 steps.

| arm | s / 1k tokens | tokens / prompt | s / prompt | relative cost |
|---|---|---|---|---|
| countdown | 0.275 | 3963 | 5.186 | 1.07x |
| gsm8k | 0.301 | 2543 | 4.864 | 1.00x |
| math35 | 0.364 | 5062 | 5.940 | 1.22x |
| mbpp | 0.000 | 844 | 4.098 | 0.84x |
| noisy | 0.049 | 2508 | 4.221 | 0.87x |

## Learning and transfer: slice gain per window of full compute share

Row = calibration slice, column = arm trained; mean (posterior SD).

| slice | windows | countdown | gsm8k | math35 | mbpp | noisy |
|---|---|---|---|---|---|---|
| countdown | 5 | +0.119 (0.420) | +0.096 (0.315) | -0.280 (0.354) | -0.002 (0.131) | +0.013 (0.326) |
| gsm8k | 5 | -0.145 (0.548) | -0.482 (0.472) | +0.477 (0.574) | +0.160 (0.156) | +0.150 (0.415) |
| math35 | 5 | -0.262 (0.605) | -0.523 (0.431) | +0.868 (0.509) | +0.167 (0.254) | -0.187 (0.450) |
| mbpp | 5 | +0.395 (0.386) | -0.124 (0.282) | -0.094 (0.330) | +0.093 (0.157) | -0.263 (0.288) |

**Random-reward (noisy) training -> GSM8K accuracy: +0.150 (SD 0.415).** Near zero means the spurious reward neither helps nor hurts this model; clearly positive reproduces the Qwen Spurious-Rewards effect; negative means it harms.

## Group polarisation kappa per arm (during training)

gsm8k: 1.50; math35: 2.55; mbpp: 1.70; countdown: -; noisy: 49.95;

