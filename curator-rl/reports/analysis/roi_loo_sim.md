# ROI vs leave-one-out in the simulator (GPU shape, realism v2; tuning seeds)

Curator at the Kaggle preset `gpu_tuned`; seeds 0..19. ROI = in-run estimate from Curator's own calibration (benchmark points per GPU-hour); LOO = noiseless score lost when the arm is left out at the same budget (95% t-interval over seeds); the truth is LOO under Uniform (the arm's value in an equal mixture); LOO under Curator also reflects how well Curator uses the arm. H5: tau >= 0.4 and the junk arms ['noisy', 'countdown'] in the bottom two of both rankings.

## S-I-R1-GPU2

| arm | ROI members (points/GPU-h) | ROI full matrix | LOO under Uniform (truth) | LOO under Curator |
|---|---|---|---|---|
| gsm8k | +7.03 | -3.18 | +0.0151 [+0.0125, +0.0177] | +0.0326 [+0.0287, +0.0364] |
| mbpp | +3.59 | +3.23 | +0.0004 [-0.0016, +0.0024] | -0.0020 [-0.0112, +0.0071] |
| countdown | +0.00 | +1.83 | -0.0068 [-0.0094, -0.0042] | +0.0001 [-0.0075, +0.0078] |
| noisy | -7.54 | -7.51 | -0.0070 [-0.0092, -0.0049] | +0.0054 [-0.0025, +0.0132] |
| math35 | +1.95 | +12.62 | -0.0082 [-0.0102, -0.0061] | -0.0088 [-0.0180, +0.0003] |

- ROI `members` vs truth, pooled over seeds: tau +0.60, junk in the bottom two of both False -> H5 FAIL; one seed at a time: median tau +0.40, H5 passes in 10% of seeds
- ROI `full` vs truth, pooled over seeds: tau -0.20, junk in the bottom two of both False -> H5 FAIL; one seed at a time: median tau +0.10, H5 passes in 0% of seeds

## S-J-R-GPU2

| arm | ROI members (points/GPU-h) | ROI full matrix | LOO under Uniform (truth) | LOO under Curator |
|---|---|---|---|---|
| gsm8k | +9.96 | +5.25 | +0.0147 [+0.0121, +0.0172] | +0.0245 [+0.0193, +0.0297] |
| math35 | +4.53 | +6.26 | -0.0002 [-0.0023, +0.0020] | -0.0058 [-0.0149, +0.0033] |
| mbpp | +1.30 | +5.70 | -0.0036 [-0.0058, -0.0015] | -0.0153 [-0.0223, -0.0083] |
| countdown | +0.00 | +4.99 | -0.0082 [-0.0108, -0.0057] | -0.0047 [-0.0107, +0.0014] |
| noisy | -1.48 | +3.47 | -0.0089 [-0.0112, -0.0067] | +0.0007 [-0.0066, +0.0080] |

- ROI `members` vs truth, pooled over seeds: tau +1.00, junk in the bottom two of both True -> H5 PASS; one seed at a time: median tau +0.60, H5 passes in 30% of seeds
- ROI `full` vs truth, pooled over seeds: tau +0.60, junk in the bottom two of both True -> H5 PASS; one seed at a time: median tau +0.10, H5 passes in 0% of seeds

