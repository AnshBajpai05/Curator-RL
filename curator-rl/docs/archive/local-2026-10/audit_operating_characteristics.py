"""Operating characteristics of the reward-validity decision rule (D-98). Pure statistics, no experiment data.

Rule: pooled 2x2 table of (arm reward, slice verifier) over unique (item, answer) pairs; phi coefficient;
one-sided upper bound via Fisher z with critical value z; flag 'invalid' iff n >= n_min, every margin
has >= min_class items in each class, and the upper bound < phi0.

Two settings:
- single look: one decision at n pairs;
- repeated looks: the decision is recomputed at every calibration window on pooled pairs that grow
  from n_min by `step` new pairs per window, for `looks` windows; a reward counts as flagged if it is
  flagged at ANY look (the relevant false-positive rate for an online rule).

Rewards: 'random' = Bernoulli(0.35) independent of correctness; a float e = the verifier's verdict
flipped with probability e (a valid but noisy reward). Base correctness 0.4.

    python experiments/analysis/audit_operating_characteristics.py
"""

from __future__ import annotations

import math

import numpy as np

KINDS = ("random", 0.0, 0.1, 0.2, 0.3, 0.4)


def decide(r, y, phi0, z, n_min=100, min_class=5):
    n = len(r)
    if n < n_min or min(r.sum(), n - r.sum(), y.sum(), n - y.sum()) < min_class:
        return "inconclusive"
    phi = float(np.corrcoef(r, y)[0, 1])
    up = math.tanh(math.atanh(max(min(phi, 0.999999), -0.999999)) + z / math.sqrt(n - 3))
    return "invalid" if up < phi0 else "valid"


def draw(kind, n, rng):
    y = (rng.random(n) < 0.4).astype(float)
    if kind == "random":
        r = (rng.random(n) < 0.35).astype(float)
    else:
        r = np.where(rng.random(n) < kind, 1 - y, y)
    return r, y


def rates(phi0, z, n_min, looks, step, reps, rng):
    out = {}
    for kind in KINDS:
        first, ever = 0, 0
        for _ in range(reps):
            r, y = draw(kind, n_min + step * (looks - 1), rng)
            flags = [decide(r[: n_min + step * k], y[: n_min + step * k], phi0, z, n_min) == "invalid"
                     for k in range(looks)]
            first += flags[0]
            ever += any(flags)
        out[kind] = (first / reps, ever / reps)
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    print("P(flag 'invalid'): first look / ever over 6 looks (+20 pairs per look); base correctness 0.4")
    for phi0, z in ((0.3, 1.645), (0.3, 2.326), (0.3, 2.576)):
        res = rates(phi0, z, 100, 6, 20, 3000, rng)
        print(f"phi0={phi0} z={z}: " + "  ".join(f"{k}: {a:.3f}/{b:.3f}" for k, (a, b) in res.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
