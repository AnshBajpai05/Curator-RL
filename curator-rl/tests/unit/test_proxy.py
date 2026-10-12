"""Unit tests for the proxy reward (Roadmap E.4, Part M item 11-adjacent)."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from curator_rl.core.config import ProxyCfg
from curator_rl.signals.proxy import ProxyReward


def make_cfg(alpha=0.5, beta=0.5, clip_l=3.0, q_lo=0.0, q_hi=1.0) -> ProxyCfg:
    return ProxyCfg(
        alpha=alpha, beta=beta, clip_l=clip_l,
        quantile_prior_lo=q_lo, quantile_prior_hi=q_hi,
        fit={"enabled": False, "ridge": 1.0, "min_rows": 3},
    )


def test_hand_values_exact_cost_exponent_one():
    proxy = ProxyReward(make_cfg(), lam=0.9, cost_exponent=1.0, min_quantile_samples=10_000)
    proxy.update_lp_values({"e": 0.4})
    out = proxy.compute({"e": 0.4}, {"e": 0.2}, {"e": 2.0})
    r = out["e"]
    # sigma=0 on first update -> lp' = 0; x = 0.5*0 + 0.5*0.2 = 0.1; r = x/2 (priors 0..1)
    assert math.isclose(r.proxy_raw, 0.1)
    assert math.isclose(r.proxy_unit, 0.05)
    assert math.isclose(r.proxy_reward, 0.05)


def test_sigma_warmup_clip_zero():
    proxy = ProxyReward(make_cfg(), lam=0.9, cost_exponent=0.0, min_quantile_samples=10_000)
    proxy.update_lp_values({"e": 1.0})
    # after first value: var=0 -> sigma=0 -> lp' = 0 (guard), not overflow
    out = proxy.compute({"e": 1.0}, {"e": 0.0}, {"e": 1.0})
    assert out["e"].proxy_raw == 0.0


def test_sigma_scales_after_observations():
    proxy = ProxyReward(make_cfg(), lam=0.5, cost_exponent=0.0, min_quantile_samples=10_000)
    proxy.update_lp_values({"a": 1.0, "b": -1.0})
    # lam 0.5 EW std of {1,-1} = ~1.0-ish; lp' = lp/sigma within clip
    sigma = proxy.sigma_lp()
    assert 0.3 <= sigma <= 1.5
    lp_clipped = proxy.clip_lp(0.5)
    assert abs(lp_clipped - 0.5 / sigma) < 1e-9


def test_cost_exponent_zero_ablation():
    proxy = ProxyReward(make_cfg(), lam=0.9, cost_exponent=0.0, min_quantile_samples=10_000)
    proxy.update_lp_values({"e": 1.0, "f": -1.0})
    sigma = proxy.sigma_lp()
    out = proxy.compute({"e": 4.0, "f": 4.0}, {"e": 0.0, "f": 0.0}, {"e": 2.0, "f": 8.0})
    # cost disabled: identical rewards for identical lp regardless of cost;
    # 4/sigma > clip_l=3 -> lp' clips to 3 -> x = 1.5 for both
    assert out["e"].proxy_unit == out["f"].proxy_unit == 1.5
    assert abs(out["e"].proxy_unit - 0.5 * min(4.0 / sigma, 3.0)) < 1e-9


def test_unknown_unit_cost_yields_zero_reward():
    proxy = ProxyReward(make_cfg(), lam=0.9, cost_exponent=1.0, min_quantile_samples=10_000)
    proxy.update_lp_values({"a": 1.0, "b": -1.0})
    out = proxy.compute({"a": 1.0, "b": 1.0}, {"a": 0.9, "b": 0.9}, {"a": 0.0, "b": 2.0})
    assert out["a"].proxy_reward == 0.0
    assert out["b"].proxy_reward > 0.0


def test_warmup_maps_with_priors():
    proxy = ProxyReward(make_cfg(q_lo=0.0, q_hi=2.0), lam=0.9, cost_exponent=0.0,
                        min_quantile_samples=100)
    assert len(proxy._reservoir) == 0
    assert abs(proxy._map(0.05) - 0.05 / 2.0) < 1e-12


def test_reservoir_switches_after_min_samples():
    proxy = ProxyReward(make_cfg(), lam=0.9, cost_exponent=0.0, min_quantile_samples=5)
    proxy.update_lp_values({"a": 1.0, "b": -1.0})
    sigma = proxy.sigma_lp()
    proxy._min_samples = 2  # force the empirical path after two kept values
    rt1 = 0.5 * 0.1 / sigma
    rt2 = 0.5 * 0.9 / sigma
    proxy.compute({"e": 0.1}, {"e": 0.0}, {"e": 1.0})
    proxy.compute({"e": 0.9}, {"e": 0.0}, {"e": 1.0})
    # reservoir = [rt1, rt2]; percentile bounds from these four kept values
    proxy.compute({"e": 0.1}, {"e": 0.0}, {"e": 1.0})
    proxy.compute({"e": 0.9}, {"e": 0.0}, {"e": 1.0})
    mapped_mid = proxy.compute({"e": 0.5}, {"e": 0.0}, {"e": 1.0})["e"].proxy_reward
    expected = ((0.5 * 0.5 / sigma) - rt1) / (rt2 - rt1)
    assert abs(mapped_mid - expected) < 1e-9  # expected == 0.5
    assert 0.0 <= mapped_mid <= 1.0


def test_quantile_reservoir_percentiles_span_data():
    proxy = ProxyReward(make_cfg(), lam=0.9, cost_exponent=0.0,
                        min_quantile_samples=5, reservoir_size=100)
    proxy.update_lp_values({"a": 1.0, "b": -1.0})
    for v in [0.0, 1.0] * 10:
        proxy.compute({"e": v}, {"e": 0.0}, {"e": 1.0})
    ordered = sorted(proxy._reservoir)
    max_v = ordered[-1]
    lo = ordered[max(int(0.05 * (len(ordered) - 1)), 0)]
    hi = ordered[max(int(0.95 * (len(ordered) - 1)), 0)]
    assert lo <= 0.1 * max_v           # 5th percentile sits in the low mass
    assert hi >= 0.9 * max_v           # 95th percentile sits in the high mass


@given(r=st.floats(-100.0, 100.0), q_lo=st.floats(0.0, 5.0), span=st.floats(0.5, 20.0))
@settings(max_examples=25, deadline=None)
def test_reward_mapping_bounded_property(r, q_lo, span):
    proxy = ProxyReward(make_cfg(q_lo=q_lo, q_hi=q_lo + span), lam=0.9, cost_exponent=1.0,
                        min_quantile_samples=10_000)
    # current mapping bounds held by the properties: set them directly
    proxy._min_samples = 3
    proxy._reservoir.clear()
    proxy._reservoir.extend([q_lo, q_lo, q_lo + span, q_lo + span])
    assert 0.0 <= proxy._map(r) <= 1.0
