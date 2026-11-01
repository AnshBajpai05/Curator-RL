# SEC and DUMP baselines: fidelity to the authors' code (exploratory, 2026-11-01)

Checked before the D-103 confirmation runs, to answer: are our SEC-style and DUMP-style baselines
handicapped compared with the authors' own implementations? Nothing here changes D-103: its runs use
the pre-registered tuned points. Simulator numbers use tuning seeds 0-39 only.

## Sources read

- SEC (Chen et al., COLM 2026): `github.com/ServiceNow/sec`, `verl/verl/utils/auto_curriculum/priority.py`
  and `dataloader.py`.
- DUMP (Wang et al., arXiv 2504.09710): `github.com/ZhentingWang/DUMP`, `verl/utils/curriculum_learning.py`.
- Ours: `src/curator_rl/scheduler/baselines/sec.py`, `dump.py`.

## SEC

| Aspect | Authors' code | Ours |
|---|---|---|
| Arm reward | per-sample mean \|advantage\| over response tokens, averaged per arm in the batch | per-group mean \|GRPO advantage\|, averaged per arm in the round (same quantity for sequence-level advantages) |
| Value update | EMA, `lr` 0.1 | EMA (`alpha`), tuned 0.1 |
| Unsampled arms | new value 0, so their Q decays toward 0 | Q unchanged |
| Selection | Boltzmann, temperature **1.0**, random draw per sample | softmax, temperature **0.4** (tuned), deterministic integer quotas |
| Floor / warm-up | none | shared protocol constants: 10% floor (eps/N) and 6 uniform rounds |

Our tuning grid was temperature {0.1, 0.4}, so the authors' default 1.0 was never tried.

## DUMP

| Aspect | Authors' code | Ours |
|---|---|---|
| Learnability | mean \|advantage\| over a sliding window of 300 samples | mean \|advantage\| over the last 10 rounds (`window_rounds` 10) |
| Exploration bonus | `1.0 * sqrt(2 ln(N+1) / (n+1))`, n counted in samples (≈ c 1.41 in our form) | `c * sqrt(ln N / n)`, c tuned 2.0, n counted in prompts |
| Selection | softmax, temperature **0.1**, random draw per sample | softmax, temperature **0.4** (tuned; 0.1 was in the grid and lost) |
| Floor / warm-up | none | shared 10% floor and 6 uniform rounds |

Side notes on the authors' file (not used by its sampling weights): a Beta success threshold of
reward >= 3 disagrees with its own comment (>= 1), and the docstring says Thompson sampling while the
code uses UCB plus softmax.

## Simulator check (realism v2, S-I-R1-GPU2 + S-I-R2-GPU2, tuning seeds 0-39)

`python experiments/analysis/sec_dump_defaults.py <repo root>`

| Method | Score vs Uniform | SE | Share on the noisy arm |
|---|---|---|---|
| Uniform | 0.0% | - | 20.0% |
| Curator (GPU preset) | +11.6% | 0.8% | 5.1% |
| SEC, our tuned point (alpha 0.1, tau 0.4) | +1.2% | 0.4% | 36.9% |
| SEC, authors' default (alpha 0.1, tau 1.0) | +1.2% | 0.4% | 26.4% |
| DUMP, our tuned point (c 2.0, tau 0.4, W 10) | +1.2% | 0.4% | 31.4% |
| DUMP, authors'-like (c 1.41, tau 0.1, W 10) | −0.6% | 0.4% | 48.7% |

The simulator reproduces the GPU budget shares (GPU seed 0: SEC 34%, DUMP 31%, Curator 6%).

## Verdict

- **DUMP is not handicapped.** The authors' sharper temperature puts more budget on the broken reward
  (49%) and scores lower; our tuned point is the kinder setting.
- **SEC is faithful in signal and update, but our temperature is sharper than the authors' default.**
  At their default it still over-funds the broken reward (26% against Uniform's 20%), less than at our
  point (37%), with the same simulator score. Our temperature choice therefore strengthens the
  budget-share effect; it is not shown to change SEC's score.
- **Disclosure for D-103:** report SEC at its tuned point as pre-registered, and state that the
  authors' default temperature (1.0) was outside the tuning grid. A seed-0 GPU run of SEC at
  temperature 1.0 would settle it; it must be declared in `docs/DECISIONS.md` before it runs.
