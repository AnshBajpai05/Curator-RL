"""Paired-comparison statistics for the GPU matrix and new simulator gates (audit E-11).

The protocol used a 95% percentile bootstrap over seeds. With the GPU matrix's 3-5 paired seeds
that interval is far too narrow: measured coverage 74.9% at n = 3 and 82.8% at n = 5 (93.9% at
n = 50). These functions give the intervals that hold at small n:

- `paired_t_interval`: Student-t interval of the mean paired difference (95.3% / 94.5% coverage
  at n = 3 / 5 under normal differences);
- `sign_flip_pvalue`: exact two-sided permutation test (all 2^n sign flips for n <= 16), no
  normality assumption;
- `tost_equivalent`: two one-sided t-tests for equivalence within +-margin (H6, Gate 4); a
  non-significant difference is never treated as equivalence;
- `holm`: Holm step-down adjustment for the co-primary hypotheses H1/H2;
- `paired_bootstrap`: kept for n >= 20 (the simulator gates), refuses smaller samples.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

MIN_BOOTSTRAP_N = 20


def paired_t_interval(diffs: Sequence[float], level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) of the mean paired difference; needs n >= 2."""
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        raise ValueError("a paired t-interval needs at least 2 paired differences")
    mean = float(d.mean())
    half = float(stats.t.ppf(0.5 + level / 2.0, d.size - 1) * d.std(ddof=1) / math.sqrt(d.size))
    return mean, mean - half, mean + half


def sign_flip_pvalue(diffs: Sequence[float], *, n_mc: int = 100_000, seed: int = 0) -> float:
    """Two-sided p-value of mean(d) = 0 by sign flips (exact for n <= 16, Monte Carlo above)."""
    d = np.asarray(diffs, dtype=float)
    observed = abs(d.mean())
    if d.size <= 16:
        flips = np.array(list(itertools.product((-1.0, 1.0), repeat=d.size)))
    else:
        flips = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_mc, d.size))
    means = np.abs((flips * d).mean(axis=1))
    return float(np.mean(means >= observed - 1e-15))


def tost_equivalent(diffs: Sequence[float], margin: float, alpha: float = 0.05) -> tuple[bool, float, float]:
    """Equivalence within +-margin by TOST: the (1 - 2 alpha) t-interval lies inside (-margin, margin)."""
    _, lo, hi = paired_t_interval(diffs, level=1.0 - 2.0 * alpha)
    return bool(lo > -margin and hi < margin), lo, hi


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (p.size - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def paired_bootstrap(diffs: Sequence[float], *, n_boot: int = 10_000, seed: int = 0,
                     level: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap (mean, lo, hi); refuses n < 20, where it undercovers badly."""
    d = np.asarray(diffs, dtype=float)
    if d.size < MIN_BOOTSTRAP_N:
        raise ValueError(f"percentile bootstrap needs n >= {MIN_BOOTSTRAP_N} (got {d.size}); "
                         "use paired_t_interval or sign_flip_pvalue")
    boots = np.random.default_rng(seed).choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [50.0 * (1 - level), 100.0 - 50.0 * (1 - level)])
    return float(d.mean()), float(lo), float(hi)
