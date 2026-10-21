"""Drift-triggered discounting (B2) and concentration control (B7)."""

from __future__ import annotations

import numpy as np
import pytest

from curator_rl.scheduler.mixture import cap_weights
from curator_rl.signals.drift import PageHinkley, standardised_surprise

G = 8


def _round_surprise(rng, p_true, p_prev, n=6, kappa=2.14):
    q = rng.beta(kappa * p_true, kappa * (1 - p_true), size=n)
    k = rng.binomial(G, q)
    return standardised_surprise(int(k.sum()), n * G, n, G, p_prev, (G + kappa) / (kappa + 1))


def test_surprise_is_standardised_on_a_stationary_arm():
    rng = np.random.default_rng(0)
    for n in (3, 6, 12):
        xs = [_round_surprise(rng, 0.4, 0.4, n=n) for _ in range(4000)]
        assert abs(np.mean(xs)) < 0.06 and 0.9 < np.std(xs) < 1.1


def test_surprise_needs_a_dispersion_estimate():
    assert np.isnan(standardised_surprise(10, 48, 6, G, 0.4, float("nan")))


def test_page_hinkley_rarely_alarms_on_slow_learning_but_catches_an_abrupt_shift():
    rng = np.random.default_rng(1)
    false_alarms = 0
    for _ in range(200):                          # 60 rounds of slow learning: +0.004 per round
        ph, p = PageHinkley(0.5, 8.0), 0.30
        for _ in range(60):
            p_next = p + 0.004
            false_alarms += ph.update(_round_surprise(rng, p_next, p))
            p = p_next
    assert false_alarms / (200 * 60) < 0.01       # well under one false alarm per 100 rounds
    lags = []
    for _ in range(200):                          # abrupt drop from 0.6 to 0.3 (S-E style regime change)
        ph = PageHinkley(0.5, 8.0)
        for t in range(1, 30):
            if ph.update(_round_surprise(rng, 0.3, 0.6)):
                lags.append(t)
                break
    assert len(lags) >= 195 and np.median(lags) <= 5     # B2 target: detection lag <= 5 rounds


def test_page_hinkley_state_round_trip():
    a = PageHinkley(0.5, 8.0)
    for x in (1.0, 2.0, -0.3):
        a.update(x)
    b = PageHinkley(0.5, 8.0)
    b.load_checkpoint(a.get_state())
    assert a.update(3.0) == b.update(3.0) and a.get_state() == b.get_state()


def test_cap_redistributes_excess_and_keeps_a_valid_mixture():
    w = {"a": 0.7, "b": 0.2, "c": 0.06, "d": 0.04}
    out = cap_weights(w, 0.5)
    assert out["a"] == pytest.approx(0.5) and sum(out.values()) == pytest.approx(1.0)
    assert all(out[e] >= w[e] - 1e-12 for e in "bcd")


def test_cap_never_feeds_status_capped_arms():
    w = {"a": 0.8, "b": 0.1, "noisy": 0.02, "c": 0.08}
    out = cap_weights(w, 0.5, frozen={"noisy"})
    assert out["noisy"] == pytest.approx(0.02) and out["a"] == pytest.approx(0.5)


def test_cap_below_one_over_n_is_lifted_to_uniform():
    out = cap_weights({"a": 0.6, "b": 0.4}, 0.3)
    assert out == pytest.approx({"a": 0.5, "b": 0.5})
