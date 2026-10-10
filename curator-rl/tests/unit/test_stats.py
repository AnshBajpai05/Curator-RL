"""Small-sample paired statistics (audit E-11)."""

from __future__ import annotations

import numpy as np
import pytest

from curator_rl.core.stats import (
    holm,
    paired_bootstrap,
    paired_t_interval,
    sign_flip_pvalue,
    tost_equivalent,
)


@pytest.mark.parametrize("n", [3, 5])
def test_t_interval_has_nominal_coverage_where_the_bootstrap_does_not(n):
    rng = np.random.default_rng(n)
    hits = 0
    reps = 3000
    for _ in range(reps):
        _, lo, hi = paired_t_interval(rng.normal(0.0, 1.0, n))
        hits += lo <= 0.0 <= hi
    assert 0.93 < hits / reps < 0.97


def test_bootstrap_refuses_small_samples():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2, 0.3, 0.4, 0.5])
    mean, lo, hi = paired_bootstrap(np.linspace(-1, 1, 30) + 0.5)
    assert lo < mean < hi


def test_sign_flip_exact_pvalue():
    assert sign_flip_pvalue([1.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(2 / 32)   # only +all and -all
    assert sign_flip_pvalue([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0)


def test_tost_requires_the_interval_inside_the_margin():
    tight = [0.001, -0.002, 0.0015, 0.0, -0.001]
    loose = [0.02, -0.03, 0.015, -0.01, 0.025]
    assert tost_equivalent(tight, margin=0.01)[0]
    assert not tost_equivalent(loose, margin=0.01)[0]


def test_holm_adjustment():
    assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])
