"""Integration tests: every scenario runs end to end (Roadmap F.5, Phase 3 item 11).

Budgets are shrunk (not the committed scenario values) to keep the suite CPU-
cheap; the full-budget scenario validation with oracles is run through
`experiments/run_sim.py` and its reports.
"""

from pathlib import Path

import pytest

from curator_rl.core.seeding import SeedManager
from curator_rl.scheduler.baselines.random_baseline import RandomScheduler
from curator_rl.scheduler.baselines.uniform import UniformScheduler
from curator_rl.simulator.harness import run_episode
from curator_rl.simulator.oracle import StaticOracle
from curator_rl.simulator.scenarios import load_scenario

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "configs" / "sim"
SCENARIOS = sorted(SCENARIO_DIR.glob("scenario_s*.yaml"))


def shrunk(scenario, budget_usd=0.6, max_rounds=60):
    return scenario.model_copy(update={"budget_usd": budget_usd, "max_rounds": max_rounds})


def test_all_scenarios_present():
    """S-A..S-J, the three refits of S-I/S-J from measured data (D-88), their GPU-shape versions (E-05) and the
    realism-v2 GPU-shape set (three refits + misaligned-arm and noisy-transfer stress variants)."""
    names = [p.name for p in SCENARIOS]
    assert len(SCENARIOS) == 22, f"expected S-A..S-J + 3 refits + 3 GPU-shape + 6 GPU2 scenarios, found {names}"
    assert {"scenario_si_r1.yaml", "scenario_si_r2.yaml", "scenario_sj_r.yaml", "scenario_si_r1_gpu.yaml",
            "scenario_si_r2_gpu.yaml", "scenario_sj_r_gpu.yaml", "scenario_si_r1_gpu2.yaml",
            "scenario_si_r2_gpu2.yaml", "scenario_sj_r_gpu2.yaml", "scenario_si_r1_gpu2_misaligned.yaml",
            "scenario_si_r1_gpu2_noisy_neg.yaml", "scenario_si_r1_gpu2_noisy_pos.yaml"} <= set(names)


def test_long_horizon_portfolio_scenarios_meet_the_horizon_rule():
    """Roadmap v3 4.2: S-I/S-J run at least 60 rounds under Uniform; S-J is S-I with equal costs."""
    si = load_scenario(SCENARIO_DIR / "scenario_si.yaml")
    sj = load_scenario(SCENARIO_DIR / "scenario_sj.yaml")
    ids = [e.env_id for e in si.envs]
    for sc in (si, sj):
        res = run_episode(UniformScheduler(ids), sc, seed=0, method="uniform")
        assert res.rounds >= 60, f"{sc.scenario_id}: only {res.rounds} rounds"
    costs_i = [e.cost_usd_per_prompt for e in si.envs]
    costs_j = [e.cost_usd_per_prompt for e in sj.envs]
    assert max(costs_i) / min(costs_i) >= 2.0              # real cost spread (H1 regime)
    assert len(set(costs_j)) == 1                          # homogeneous twin (H6 regime)
    for a, b in zip(si.envs, sj.envs):                      # identical dynamics, only cost differs
        assert (a.env_id, a.difficulty, a.eta, a.noisy_q, a.transfer_out) == (
            b.env_id, b.difficulty, b.eta, b.noisy_q, b.transfer_out
        )


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_runs_end_to_end(path):
    scenario = shrunk(load_scenario(path))
    for method, make in (
        ("uniform", lambda envs: UniformScheduler(envs)),
        ("random", lambda envs: RandomScheduler(envs, SeedManager(1).rng("scheduler"))),
    ):
        result = run_episode(make([e.env_id for e in scenario.envs]), scenario, seed=1,
                             method=method)
        assert result.rounds > 0
        assert 0.0 <= result.final_score <= 1.0
        assert result.within_budget_tolerance, (
            f"{scenario.scenario_id}/{method}: cost {result.total_cost_usd:.3f} "
            f"vs budget {result.budget_usd:.3f}"
        )
        assert len(result.round_logs) == result.rounds
        for log in result.round_logs:
            assert abs(sum(log["weights"].values()) - 1.0) < 1e-6


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_determinism_by_seed(path):
    scenario = shrunk(load_scenario(path), budget_usd=0.3, max_rounds=30)
    env_ids = [e.env_id for e in scenario.envs]
    r1 = run_episode(UniformScheduler(env_ids), scenario, seed=42, method="uniform")
    r2 = run_episode(UniformScheduler(env_ids), scenario, seed=42, method="uniform")
    assert r1.final_score == r2.final_score
    assert r1.total_cost_usd == r2.total_cost_usd
    assert r1.rounds == r2.rounds


class _CalibratingUniform(UniformScheduler):
    """Uniform mixture that consumes calibration observations (test double)."""

    uses_calibration = True

    def __init__(self, env_ids):
        super().__init__(env_ids)
        self.calibrations = []

    def update_calibration(self, observation):
        self.calibrations.append(observation)


def test_calibration_produced_and_charged_only_for_methods_that_use_it():
    """Roadmap H.6 / D-72: calibration is run and charged only for its users."""
    scenario = load_scenario(SCENARIO_DIR / "scenario_sg.yaml")  # calib enabled
    scenario = shrunk(scenario, budget_usd=0.5, max_rounds=60)
    env_ids = [e.env_id for e in scenario.envs]
    user = _CalibratingUniform(env_ids)
    charged = run_episode(user, scenario, seed=3, method="calibrating")
    assert len(charged.calib_logs) > 0 and len(user.calibrations) == len(charged.calib_logs)
    assert all(log["eval_cost_usd"] == scenario.calib.cost_usd for log in charged.calib_logs)
    plain = run_episode(UniformScheduler(env_ids), scenario, seed=3, method="uniform")
    assert plain.calib_logs == []
    assert plain.rounds >= charged.rounds          # the non-user buys at least as many rounds


def test_paired_calibration_reports_change_se_and_keeps_training_draws():
    """D-72: paired items give a change SE from the second evaluation on, and the
    calibration stream never perturbs the training draws."""
    base = shrunk(load_scenario(SCENARIO_DIR / "scenario_sg.yaml"), budget_usd=1.5, max_rounds=60)
    paired = base.model_copy(update={"calib": base.calib.model_copy(update={"paired": True, "churn": 0.02})})
    env_ids = [e.env_id for e in base.envs]
    a, b = _CalibratingUniform(env_ids), _CalibratingUniform(env_ids)
    ra = run_episode(a, paired, seed=5, method="paired")
    rb = run_episode(b, base, seed=5, method="unpaired")
    assert a.calibrations[0].delta_se_by_domain is None
    later = a.calibrations[1].delta_se_by_domain
    assert later is not None and set(later) == set(env_ids)
    n_b = base.world.benchmark_items
    assert all(1.0 / n_b - 1e-12 <= v <= 1.0 for v in later.values())   # floored at one flip
    assert all(c.delta_se_by_domain is None for c in b.calibrations)
    assert ra.round_logs[0]["per_env"] == rb.round_logs[0]["per_env"]  # same training draws


def test_scenario_sa_oracle_gap_above_ten_percent():
    """Exit criterion (Phase 3 item 12): oracle beats uniform by > 10% in S-A."""
    scenario = load_scenario(SCENARIO_DIR / "scenario_sa.yaml")
    env_ids = [e.env_id for e in scenario.envs]
    uniform_scores = []
    oracle_scores = []
    for seed in range(5):
        uniform_scores.append(
            run_episode(UniformScheduler(env_ids), scenario, seed=seed, method="uniform").final_score
        )
        oracle_scores.append(
            run_episode(StaticOracle(_world_of(scenario, seed), simplex_step=0.1, seed=seed),
                        scenario, seed=seed, method="static_oracle").final_score
        )
    diff = sum(oracle_scores) / 5 - sum(uniform_scores) / 5
    assert diff > 0.10, f"oracle improvement {diff:.3f} not above 10 points"


def _world_of(scenario, seed: int):
    from curator_rl.simulator.scenarios import build_world

    return build_world(scenario, seed)


class _TargetingUniform(_CalibratingUniform):
    """Calibration user that asks for one slice with 40 items after the first evaluation."""

    def calibration_request(self):
        if not self.calibrations:
            return None
        return {self.env_ids[0]: 40}


def test_targeted_calibration_evaluates_only_requested_slices_and_charges_by_item():
    """D-75: the harness passes the scheduler's request and charges per item."""
    base = shrunk(load_scenario(SCENARIO_DIR / "scenario_sg.yaml"), budget_usd=1.5, max_rounds=60)
    calib = base.calib.model_copy(update={"paired": True, "cost_per_item_usd": 0.0001, "items_per_slice": 40})
    sc = base.model_copy(update={"calib": calib})
    env_ids = [e.env_id for e in sc.envs]
    sched = _TargetingUniform(env_ids)
    res = run_episode(sched, sc, seed=2, method="targeting")
    first, second = sched.calibrations[0], sched.calibrations[1]
    assert set(first.score_by_domain) == set(env_ids)                    # no request -> every slice
    assert set(second.score_by_domain) == {env_ids[0]}                    # request honoured
    assert res.calib_logs[0]["eval_cost_usd"] == pytest.approx(0.0001 * 40 * len(env_ids))
    assert res.calib_logs[1]["eval_cost_usd"] == pytest.approx(0.0001 * 40)
    assert set(second.delta_se_by_domain) == {env_ids[0]}                 # paired with its baseline
