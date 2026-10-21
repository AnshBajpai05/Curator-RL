"""Unit tests for the mixture map (Roadmap E.6 + E.5, Part M items 6-8)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from curator_rl.core.config import SchedulerCfg
from curator_rl.core.types import EnvStatus, SignalVector
from curator_rl.scheduler.mixture import (
    apply_status_constraints,
    scores_to_weights,
)

ENVS = ["a", "b", "c"]


def make_cfg(**kwargs) -> SchedulerCfg:
    base = dict(
        gamma=0.95, exploration_coef=0.5, tau=1.0, epsilon=0.10,
        warmup_rounds=6, score_norm="zscore", cost_exponent=1.0,
        status_control="soft", pull_unit="round",
        s3_multiplier=0.5, s4_multiplier=0.5, s5_shrink=0.5, s5_cap="none", s1_quota=0.30,
        max_weight=1.0, smoothing=1.0,
        drift={"enabled": False, "delta": 0.5, "threshold": 8.0, "decay": 0.3},
    )
    base.update(kwargs)
    return SchedulerCfg(**base)


def make_sv(env_id: str, status: EnvStatus = EnvStatus.S2, *, n_groups_eff=1000.0) -> SignalVector:
    return SignalVector(
        env_id=env_id, round_t=10, n_rounds_seen=10, n_groups_eff=n_groups_eff,
        pass_rate=0.5, pass_lo=0.45, pass_hi=0.55, lp=0.0, lp_se=0.05, lp_z=0.0,
        lp_raw_fast=0.5, lp_raw_slow=0.5, richness=0.5, mean_score=0.5,
        unit_cost_usd=0.001, unit_cost_norm=1.0, proxy_raw=0.5, proxy_unit=0.5,
        proxy_reward=0.5, status=status, status_note=status.value,
    )


def svs(**statuses) -> dict[str, SignalVector]:
    out = {e: make_sv(e, EnvStatus.S2) for e in ENVS}
    for e, s in statuses.items():
        out[e] = make_sv(e, s)
    return out


# scores -> weights -------------------------------------------------------------

def test_softmax_sums_to_one_and_holds_floor():
    w = scores_to_weights({"a": 1.0, "b": 0.2, "c": -0.5}, tau=1.0, epsilon=0.10, env_ids=ENVS)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert all(v >= 0.10 / 3 - 1e-12 for v in w.values())
    assert w["a"] > w["b"] > w["c"]  # monotone in score


def test_softmax_no_overflow_at_extreme_scores():
    w = scores_to_weights(
        {"a": 1e6, "b": -1e6, "c": 0.0}, tau=1.0, epsilon=0.05, env_ids=ENVS,
        score_norm="none",  # z-score would shrink the extremes; raw scores clip at ±50/tau
    )
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["a"] > 0.9
    assert math.isfinite(w["a"])


def test_score_norm_modes():
    for mode in ("none", "zscore"):
        w = scores_to_weights({"a": 3.0, "b": 1.0, "c": 1.0}, tau=1.0, epsilon=0.0, env_ids=ENVS, score_norm=mode)
        assert abs(sum(w.values()) - 1.0) < 1e-12
        assert w["a"] > w["b"] == w["c"]  # tied scores stay tied
    # rank breaks ties by stable index order, so only assert the ordering
    w = scores_to_weights({"a": 3.0, "b": 1.0, "c": 1.0}, tau=1.0, epsilon=0.0, env_ids=ENVS, score_norm="rank")
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["a"] > w["b"] and w["a"] > w["c"]
    with pytest.raises(ValueError):
        scores_to_weights({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, 0.0, ENVS, score_norm="bogus")


def test_zscore_shift_invariance():
    base = scores_to_weights({"a": 1.0, "b": 2.0, "c": 3.0}, 1.0, 0.05, ENVS, score_norm="zscore")
    shifted = scores_to_weights({"a": 101.0, "b": 102.0, "c": 103.0}, 1.0, 0.05, ENVS, score_norm="zscore")
    for e in ENVS:
        assert abs(base[e] - shifted[e]) < 1e-12


# status constraints ------------------------------------------------------------

def test_status_control_off_is_passthrough():
    cfg = make_cfg(status_control="off")
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    out, intents = apply_status_constraints(weights, svs(a=EnvStatus.S3), cfg, n_min=64)
    for e in ENVS:
        assert abs(out[e] - weights[e]) < 1e-12
    assert set(intents.values()) == {"hold"}


def test_s3_soft_multiplier_redistributes_and_holds_floor():
    cfg = make_cfg(status_control="soft", epsilon=0.10, s3_multiplier=0.5)
    # a dominates: saturating it must free mass to b, c
    weights = {"a": 0.8, "b": 0.1, "c": 0.1}
    out, intents = apply_status_constraints(weights, svs(a=EnvStatus.S3), cfg, n_min=64)
    assert abs(sum(out.values()) - 1.0) < 1e-12
    assert out["a"] < 0.8
    assert out["b"] > 0.1 and out["c"] > 0.1
    assert all(v >= 0.10 / 3 - 1e-12 for v in out.values())
    assert intents["a"] == "shrink"


def test_s1_quota_holds_and_boosts_while_half_explored():
    cfg = make_cfg(status_control="soft", epsilon=0.0, s1_quota=0.30)
    weights = {"a": 0.9, "b": 0.05, "c": 0.05}
    # half-explored: n_groups_eff 20 < n_min/2 = 32 -> quota boosted to 0.6
    vectors = svs(b=EnvStatus.S1, c=EnvStatus.S2)
    vectors["b"] = make_sv("b", EnvStatus.S1, n_groups_eff=20.0)
    out, _ = apply_status_constraints(weights, vectors, cfg, n_min=64)
    assert out["b"] >= 0.6 - 1e-9
    assert out["a"] <= 0.4 + 1e-9
    # fully explored enough: n_groups_eff 40 >= 32 -> plain 0.30 quota
    vectors = svs(b=EnvStatus.S1)
    vectors["b"] = make_sv("b", EnvStatus.S1, n_groups_eff=40.0)
    out2, _ = apply_status_constraints(weights, vectors, cfg, n_min=64)
    assert 0.30 - 1e-9 <= out2["b"] < 0.6


def test_all_s1_quota_budget_never_exceeds_one():
    cfg = make_cfg(status_control="soft", epsilon=0.0, s1_quota=0.30)
    weights = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    out, _ = apply_status_constraints(
        weights, svs(a=EnvStatus.S1, b=EnvStatus.S1, c=EnvStatus.S1), cfg, n_min=64
    )
    assert abs(sum(out.values()) - 1.0) < 1e-12  # boosted quotas rescaled to budget
    assert all(v > 0 for v in out.values())


def test_hard_cap_limits_saturated_arms():
    cfg = make_cfg(status_control="hard", epsilon=0.10)
    weights = {"a": 0.9, "b": 0.05, "c": 0.05}
    out, _ = apply_status_constraints(weights, svs(a=EnvStatus.S3), cfg, n_min=64)
    assert out["a"] <= 2.0 * (0.10 / 3) + 1e-9


def test_s5_shrink_applies():
    cfg = make_cfg(status_control="soft")
    weights = {"a": 0.5, "b": 0.25, "c": 0.25}
    out, _ = apply_status_constraints(weights, svs(a=EnvStatus.S5), cfg, n_min=64)
    assert out["a"] < 0.5


@given(
    w_a=st.floats(0.05, 0.9), w_b=st.floats(0.05, 0.9), w_c=st.floats(0.05, 0.9),
    s_a=st.sampled_from(list(EnvStatus)), s_b=st.sampled_from(list(EnvStatus)),
    s_c=st.sampled_from(list(EnvStatus)),
)
@settings(max_examples=50, deadline=None)
def test_constraint_properties_hold_for_any_status_mix(w_a, w_b, w_c, s_a, s_b, s_c):
    cfg = make_cfg(status_control="soft", epsilon=0.10, s1_quota=0.30)
    total = w_a + w_b + w_c
    weights = {"a": w_a / total, "b": w_b / total, "c": w_c / total}
    vectors = svs(a=s_a, b=s_b, c=s_c)
    out, _ = apply_status_constraints(weights, vectors, cfg, n_min=64)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert all(v >= 0.10 / 3 - 1e-9 for v in out.values())
    for e, s in (("a", s_a), ("b", s_b), ("c", s_c)):
        if s == EnvStatus.S1:
            assert out[e] >= 0.30 - 1e-9  # plain quota (evidence above n_min/2)
