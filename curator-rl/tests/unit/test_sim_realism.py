"""Simulator realism switches (improvement plan B1, B7, B8, B9, E-14, E10); defaults are covered by
the golden reproduction of the original world."""

from __future__ import annotations

import math

import numpy as np
import pytest

from curator_rl.scheduler.baselines.uniform import UniformScheduler
from curator_rl.simulator.harness import run_episode
from curator_rl.simulator.scenarios import ScenarioCfg, build_world, load_scenario
from curator_rl.simulator.world import SimEnvParams, SimWorld, mixed_group_share


def test_mixed_group_share_reaches_the_binomial_limit():
    for p in (0.1, 0.35, 0.8):
        assert mixed_group_share(p, 1e7, 8) == pytest.approx(1 - p**8 - (1 - p) ** 8, abs=1e-5)
        assert mixed_group_share(p, math.inf, 8) == pytest.approx(1 - p**8 - (1 - p) ** 8)
    # polarised prompts (small kappa) give far fewer mixed groups (pilot: 0.67 at p = 0.34 for GSM8K)
    assert mixed_group_share(0.344, 1.65, 8) == pytest.approx(0.672, abs=0.01)


def _world(**kw):
    envs = kw.pop("envs", None) or [SimEnvParams("a", difficulty=0.0, eta=0.05),
                                    SimEnvParams("b", difficulty=0.0, eta=0.05)]
    base = dict(steps_per_round=2, prompts_per_round=32, group_size=8, budget_usd=10.0, skill_noise_std=0.0,
                rollout_concentration=2.14, rng=np.random.default_rng(0))
    base.update(kw)
    return SimWorld(envs, **base)


def test_mixed_groups_learnability_peaks_at_one_and_depends_on_kappa():
    w = _world(learnability_mode="mixed_groups")
    assert w.learnability_of("a") == pytest.approx(1.0)        # p = 0.5 at skill 0, difficulty 0
    w.skills["a"] = 1.0                                          # p = sigmoid(2) = 0.88
    assert w.learnability_of("a") < 4 * 0.88 * 0.12 + 0.2     # polarised groups learn less near the edges


def test_forgetting_pulls_an_untrained_arm_back_toward_its_start():
    w = _world(envs=[SimEnvParams("a", eta=0.05, forget_rate=0.05), SimEnvParams("b", eta=0.05, forget_rate=0.05)])
    for _ in range(30):
        w.step_round({"a": 1.0, "b": 0.0})
    a_peak = w.skills["a"]
    for _ in range(30):
        w.step_round({"a": 0.0, "b": 1.0})
    assert w.skills["a"] < a_peak * 0.3 and w.skills["b"] > 0.0


def test_repeated_passes_over_a_small_dataset_teach_less():
    def gain(train_items):
        w = _world(envs=[SimEnvParams("mbpp", eta=0.05, train_items=train_items)], repeat_decay=0.5)
        for _ in range(40):                         # 40 x 32 = 1,280 prompts
            w.step_round({"mbpp": 1.0})
        return w.skills["mbpp"], w.passes()["mbpp"]

    big, passes_big = gain(100_000)
    small, passes_small = gain(314)
    assert passes_small == pytest.approx(1280 / 314) and passes_big < 0.02
    assert small < 0.9 * big                        # later passes are worth less (~20% here)


def test_noisy_transfer_moves_its_target_only_when_allowed():
    def run(flag):
        envs = [SimEnvParams("gsm8k", eta=0.05),
                SimEnvParams("noisy", eta=0.05, noisy_q=0.35, transfer_out={"gsm8k": -0.5})]
        w = _world(envs=envs, noisy_transfer=flag)
        for _ in range(20):
            w.step_round({"gsm8k": 0.0, "noisy": 1.0})
        return w.skills["gsm8k"]

    assert run(False) == 0.0 and run(True) < -0.05


def _scenario(tmp_path, report_interval, **world):
    text = f"""
scenario_id: T
budget_usd: 1.0
max_rounds: 30
steps_per_round: 2
prompts_per_step: 16
report_eval: {{interval_rounds: {report_interval}}}
world: {{rollout_concentration: 2.14, report_stream: {world.get('report_stream', 'shared')},
        final_eval: {world.get('final_eval', 'binomial')}}}
envs:
  - {{env_id: a, eta: 0.01, cost_usd_per_prompt: 0.001}}
  - {{env_id: b, eta: 0.01, cost_usd_per_prompt: 0.001, difficulty: 0.5}}
"""
    path = tmp_path / f"s{report_interval}{world}.yaml".replace(" ", "").replace("'", "")
    path.write_text(text)
    return load_scenario(path)


def test_separate_report_stream_leaves_training_untouched(tmp_path):
    def skills(interval, stream):
        sc = _scenario(tmp_path, interval, report_stream=stream)
        r = run_episode(UniformScheduler(["a", "b"]), sc, 3)
        return [lg["true_skills"] for lg in r.round_logs]

    assert skills(5, "separate") == skills(0, "separate")     # B1: reports no longer consume training draws
    assert skills(5, "shared") != skills(0, "shared")         # (the legacy behaviour they fix)


def test_paired_final_items_are_shared_by_every_method_on_a_seed(tmp_path):
    sc = _scenario(tmp_path, 0, final_eval="paired_items")
    w1, w2 = build_world(sc, 7), build_world(sc, 7)
    assert w1.evaluate_report() == w2.evaluate_report()        # same skills, same items -> same score
    w3 = build_world(sc, 8)
    w3.skills = dict(w1.skills)
    assert w3.evaluate_report() != w1.evaluate_report()        # another seed draws other items


def test_vectorized_draws_have_the_same_distribution():
    rates = {}
    for mode in ("loop", "vectorized"):
        w = _world(rollout_draws=mode, rng=np.random.default_rng(1))
        k = n = 0
        for _ in range(200):
            obs = w._rollout_env("a", 32)
            k += obs.k_success
            n += obs.n_rollouts
        rates[mode] = k / n
    assert rates["loop"] == pytest.approx(0.5, abs=0.02) and rates["vectorized"] == pytest.approx(0.5, abs=0.02)


def test_noisy_transfer_out_without_the_flag_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("""
scenario_id: B
budget_usd: 1.0
envs:
  - {env_id: g, eta: 0.01, cost_usd_per_prompt: 0.001}
  - {env_id: n, eta: 0.01, cost_usd_per_prompt: 0.001, noisy_q: 0.35, transfer_out: {g: 0.2}}
""")
    with pytest.raises(ValueError, match="noisy_transfer"):
        load_scenario(p)
    assert isinstance(ScenarioCfg, type)


def test_episode_reports_passes_over_finite_datasets():
    sc = load_scenario("configs/sim/scenario_si_r1_gpu.yaml")
    r = run_episode(UniformScheduler([e.env_id for e in sc.envs]), sc, 0)
    assert sum(r.prompts_by_env.values()) == r.rounds * sc.prompts_per_round
    assert r.passes_by_env["mbpp"] is None          # this scenario declares no train_items (legacy)
