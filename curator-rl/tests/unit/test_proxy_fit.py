"""Proxy weights fitted to calibrated gains (calibration/regression.py, improvement B4)."""

from __future__ import annotations

import numpy as np
import pytest

from curator_rl.calibration.regression import FitRow, ProxyWeightFitter


def _fitter(**kw):
    base = dict(ridge=0.3, trust_region=0.25, min_rows=3)
    base.update(kw)
    return ProxyWeightFitter(0.5, 0.5, **base)


def _problem(rng, alpha, n=5, scale=0.05):
    lp = rng.normal(0.0, 1.2, n).clip(-3, 3)
    sr = rng.uniform(0.2, 1.0, n)
    se = rng.uniform(0.002, 0.012, n)
    se[rng.integers(n)] = 0.08                                  # one barely-measured arm, as on GPU
    y = scale * (alpha * lp + (1 - alpha) * sr) + rng.normal(0.0, se)
    return [FitRow(f"e{i}", float(y[i]), float(se[i]), float(lp[i]), float(sr[i])) for i in range(n)]


def _naive_alpha(rows):
    x = np.array([[r.lp, r.sr] for r in rows])
    y = np.array([r.gain for r in rows])
    c, *_ = np.linalg.lstsq(x, y, rcond=None)
    c = np.clip(c, 0, None)
    return c[0] / c.sum() if c.sum() > 0 else 0.5


def test_recovers_planted_weights_and_beats_the_naive_fit():
    rng = np.random.default_rng(0)
    fitter = _fitter()
    planted, fitted, naive = [], [], []
    for _ in range(300):
        a = rng.uniform(0.1, 0.9)
        rows = _problem(rng, a)
        fit = fitter.fit(rows, (0.5, 0.5))
        if fit.proposed is None:
            continue
        planted.append(a)
        fitted.append(fit.proposed[0])
        naive.append(_naive_alpha(rows))
    planted, fitted, naive = map(np.array, (planted, fitted, naive))
    r2 = 1 - np.sum((fitted - planted) ** 2) / np.sum((planted - planted.mean()) ** 2)
    rmse_fit = np.sqrt(np.mean((fitted - planted) ** 2))
    rmse_naive = np.sqrt(np.mean((naive - planted) ** 2))
    assert len(planted) > 280
    assert r2 >= 0.8                                    # Gate 6-sim criterion
    assert rmse_fit <= 0.8 * rmse_naive                 # >= 20% better than unweighted least squares


def test_pure_noise_rarely_moves_the_proxy():
    rng = np.random.default_rng(1)
    fitter = _fitter()
    accepted = 0
    for _ in range(300):
        rows = [FitRow(f"e{i}", float(rng.normal(0, 0.01)), 0.01, float(rng.normal()), float(rng.uniform(0.2, 1)))
                for i in range(5)]
        accepted += fitter.fit(rows, (0.5, 0.5)).accepted
    assert accepted / 300 < 0.35          # the out-of-sample guard keeps the prior most of the time


def test_a_rich_arm_without_gain_pushes_beta_down():
    # four honest arms (gain follows LP' and SR) and a random-reward arm: highest SR, zero gain
    rows = [FitRow("gsm8k", 0.040, 0.004, 0.8, 0.55), FitRow("math35", 0.030, 0.005, 0.6, 0.35),
            FitRow("mbpp", 0.035, 0.004, 0.7, 0.45), FitRow("countdown", 0.000, 0.004, 0.0, 0.02),
            FitRow("noisy", 0.000, 0.004, 0.0, 0.97)]
    fit = _fitter().fit(rows, (0.5, 0.5))
    assert fit.accepted and fit.proposed[1] < 0.5 and fit.beta < 0.5
    assert fit.beta == pytest.approx(0.375)       # trust region: at most -25% per fit


def test_trust_region_limits_each_step():
    rows = [FitRow(f"e{i}", 0.05 * lp, 0.002, lp, sr) for i, (lp, sr) in
            enumerate([(1.0, 0.9), (0.5, 0.1), (2.0, 0.5), (0.2, 0.8), (1.5, 0.3)])]   # gains follow LP' only
    assert _fitter(ridge=0.0).fit(rows, (0.5, 0.5)).proposed[0] > 0.95    # unshrunk: LP' only
    fit = _fitter().fit(rows, (0.5, 0.5))
    assert fit.accepted and fit.proposed[0] > 0.7
    assert fit.alpha == pytest.approx(0.625) and fit.beta == pytest.approx(0.375)
    fit2 = _fitter().fit(rows, (fit.alpha, fit.beta))
    assert fit2.beta == pytest.approx(0.375 * 0.75)


def test_too_few_rows_keep_the_current_weights():
    rows = [FitRow("a", 0.01, 0.01, 0.5, 0.5), FitRow("b", 0.02, 0.01, 0.2, 0.9)]
    fit = _fitter().fit(rows, (0.6, 0.4))
    assert not fit.accepted and (fit.alpha, fit.beta) == pytest.approx((0.6, 0.4))


def test_rows_with_infinite_or_missing_evidence_are_ignored():
    rows = [FitRow("a", 0.01, float("inf"), 0.5, 0.5), FitRow("b", float("nan"), 0.01, 0.2, 0.9),
            FitRow("c", 0.02, 0.01, 0.2, 0.9)]
    assert _fitter().fit(rows, (0.5, 0.5)).n_rows == 1
