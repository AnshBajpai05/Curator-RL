# System identification report (one randomized-mixture run)

## Per-arm training cost (NNLS on step time vs tokens)

Fixed overhead 56.00 s/step, R^2 0.452, 62 steps.

| arm | s / 1k tokens | tokens / prompt | s / prompt | relative cost |
|---|---|---|---|---|
| countdown | 0.506 | 4429 | 5.742 | 1.28x |
| gsm8k | 0.401 | 2503 | 4.503 | 1.00x |
| math35 | 0.716 | 5060 | 7.120 | 1.58x |
| mbpp | 0.489 | 1006 | 3.992 | 0.89x |
| noisy | 0.346 | 2586 | 4.395 | 0.98x |

## Learning and transfer: slice gain per window of full compute share

Row = calibration slice, column = arm trained; mean (posterior SD).

| slice | windows | countdown | gsm8k | math35 | mbpp | noisy |
|---|---|---|---|---|---|---|
| countdown | 5 | -0.080 (0.310) | +0.008 (0.207) | -0.184 (0.577) | +0.162 (0.334) | +0.066 (0.422) |
| gsm8k | 5 | -0.422 (0.358) | -0.177 (0.296) | +0.017 (0.662) | +0.612 (0.409) | +0.053 (0.486) |
| math35 | 5 | +0.282 (0.330) | -0.136 (0.333) | -0.121 (0.705) | -0.037 (0.387) | -0.042 (0.511) |
| mbpp | 5 | +0.268 (0.232) | +0.287 (0.205) | -0.652 (0.564) | -0.276 (0.245) | +0.373 (0.404) |

**Random-reward (noisy) training -> GSM8K accuracy: +0.053 (SD 0.486).** Near zero means the spurious reward neither helps nor hurts this model; clearly positive reproduces the Qwen Spurious-Rewards effect; negative means it harms.

## Group polarisation kappa per arm (during training)

gsm8k: 2.60; math35: 2.50; mbpp: 1.80; countdown: -; noisy: 49.95;

