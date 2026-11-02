# CURATOR GPU comparison

Sampled accuracy = mean over 4 samples per item at the training temperature (evaluation protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the same dev items for every method (uncharged; the sealed test is not used).

| method | seeds | sampled macro (mean) | pass@k macro | greedy macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |
|---|---|---|---|---|---|---|---|---|
| sec | 1 | 0.2321 | 0.3875 | 0.3042 | 1.523 | 33.0 | - | 0 |

## Per domain (sampled accuracy; greedy in brackets; mean over seeds)

| method | gsm8k | math35 | mbpp | countdown |
|---|---|---|---|---|
| sec | 0.479 (0.620) | 0.166 (0.253) | 0.283 (0.333) | 0.000 (0.010) |

## Pre-flight verifier check (D-102)

Reward each arm paid for junk before any training (empty or echoed answers, wrong or sprayed numbers, code that exits early or prints its globals). Above 5% the checker can be gamed.

| arm | junk reward rate | flagged |
|---|---|---|
| countdown | 0.00 | - |
| gsm8k | 0.00 | - |
| math35 | 0.00 | - |
| mbpp | 0.00 | - |
| noisy | 0.30 | yes |
