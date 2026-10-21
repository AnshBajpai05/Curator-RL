"""ROI leaderboard (roi/leaderboard.py) and leave-one-out (evaluation/loo.py), improvement D3."""

from __future__ import annotations

import math

import numpy as np
import pytest

from curator_rl.evaluation.loo import WithoutArm, h5_verdict, kendall_tau, loo_contributions
from curator_rl.roi.leaderboard import fit_slices, roi_table, span_rows
from curator_rl.scheduler.baselines.uniform import UniformScheduler
from curator_rl.simulator.harness import run_episode
from curator_rl.simulator.scenarios import load_scenario

ARMS = ["a", "b", "noisy"]


def _windows(n, rng, truth, *, targeted=False, se=0.002):
    """Calibration records: slice a gains b_a per full share of arm a; slice b likewise; noisy (scored on a) 0."""
    rows, acc = [], {"a": 0.5, "b": 0.5}
    for k in range(n):
        share = rng.dirichlet(np.ones(3))
        exposure = {e: float(100 * s) for e, s in zip(ARMS, share)}
        evaluated = ["a", "b"] if (not targeted or k == 0) else (["a"] if k % 2 else ["b"])
        for d in ("a", "b"):
            acc[d] += truth[d] * exposure[d] / 100.0       # every window trains both slices
        rows.append({"exposure": exposure, "scores": {d: acc[d] + rng.normal(0, se) for d in evaluated},
                     "delta_se": {d: se for d in evaluated}})
    return rows


def test_spans_accumulate_exposure_between_evaluations_of_a_slice():
    rows = [{"exposure": {"a": 10, "b": 0}, "scores": {"a": 0.5, "b": 0.5}},
            {"exposure": {"a": 0, "b": 10}, "scores": {"a": 0.6}},
            {"exposure": {"a": 10, "b": 10}, "scores": {"a": 0.7, "b": 0.9}}]
    spans = span_rows(rows, ["a", "b"])
    b_span = [s for s in spans if s.domain == "b"][0]
    assert b_span.shares.tolist() == pytest.approx([0.5, 1.5])      # windows 2-3 since b's baseline: 0+0.5 / 1+0.5
    assert b_span.gain == pytest.approx(0.4)
    assert len([s for s in spans if s.domain == "a"]) == 2


def test_members_roi_recovers_own_slice_effects_under_targeted_calibration():
    rng = np.random.default_rng(0)
    rows = _windows(12, rng, {"a": 0.04, "b": 0.02}, targeted=True)
    table = roi_table(rows, ARMS, {"a": 1e-3, "b": 1e-3, "noisy": 1e-3}, {"a": 0.5, "b": 0.5},
                      domain_of={"a": "a", "b": "b", "noisy": "a"})
    by = {r.env_id: r for r in table}
    assert [r.env_id for r in table] == ["a", "b", "noisy"]
    assert by["a"].gain_per_share == pytest.approx(0.5 * 0.04, abs=0.004)
    assert by["noisy"].roi_lo < 0 < by["noisy"].roi_hi                       # no value, interval covers 0


def test_full_structure_needs_more_spans_than_members():
    rng = np.random.default_rng(1)
    rows = _windows(4, rng, {"a": 0.04, "b": 0.02}, targeted=True)
    spans = span_rows(rows, ARMS)
    full = fit_slices(spans, ARMS)
    members = fit_slices(spans, ARMS, {"a": "a", "b": "b", "noisy": "a"})
    assert math.sqrt(full["a"]["cov"][0, 0]) > 2 * math.sqrt(members["a"]["cov"][0, 0])


def test_kendall_tau_and_h5():
    assert kendall_tau({"x": 1, "y": 2, "z": 3}, {"x": 10, "y": 20, "z": 30}) == pytest.approx(1.0)
    assert kendall_tau({"x": 1, "y": 2, "z": 3}, {"x": 30, "y": 20, "z": 10}) == pytest.approx(-1.0)
    roi = {"g": 5.0, "m": 3.0, "p": 2.0, "noisy": -1.0, "countdown": 0.0}
    loo = {"g": 0.03, "m": 0.01, "p": 0.02, "noisy": -0.01, "countdown": 0.0}
    v = h5_verdict(roi, loo, ["noisy", "countdown"])
    assert v["pass"] and v["tau"] >= 0.4
    loo_bad = dict(loo, countdown=0.05)
    assert not h5_verdict(roi, loo_bad, ["noisy", "countdown"])["junk_bottom"]


def test_loo_contributions_are_paired():
    m, lo, hi = loo_contributions([0.5, 0.6, 0.55], {"a": [0.4, 0.5, 0.45]})["a"]
    assert m == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_without_arm_keeps_the_world_and_hides_the_arm():
    sc = load_scenario("configs/sim/scenario_si_r1_gpu2.yaml")
    ids = [e.env_id for e in sc.envs]
    reduced = [e for e in ids if e != "mbpp"]
    r = run_episode(WithoutArm(UniformScheduler(reduced), "mbpp", ids), sc, seed=0)
    assert r.prompts_by_env["mbpp"] == 0
    assert all(log["weights"]["mbpp"] == 0.0 for log in r.round_logs)
    shares = {e: r.prompts_by_env[e] / sum(r.prompts_by_env.values()) for e in reduced}
    assert all(abs(s - 0.25) < 0.01 for s in shares.values())
