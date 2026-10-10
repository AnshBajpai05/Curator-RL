"""Regression tests for the 2026-10-10 audit (errors_found.md). One test (or more) per error ID.

Each test fails on the pre-audit behaviour and passes on the fix; where the old behaviour is kept
behind a config switch for reproducibility, the legacy path is pinned too.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
import pytest
from tests.conftest import BASE_CFG

from curator_rl.calibration.calibrator import Calibrator
from curator_rl.calibration.credit import OwnSliceCredit, SharedSliceCredit
from curator_rl.core.config import load_config
from curator_rl.core.quotas import CarryQuotas, largest_remainder_quotas
from curator_rl.core.types import CalibrationObservation, EnvStatus
from curator_rl.signals.passrate import Posterior
from curator_rl.signals.progress import LPEstimator
from curator_rl.signals.status import flip_rate

CFG = load_config(BASE_CFG)


def _post(rate: float) -> Posterior:
    return Posterior(env_id="e", pass_rate=rate, pass_lo=rate - 0.05, pass_hi=rate + 0.05,
                     se_group=0.05, n_groups_eff=16.0, n_rounds_seen=3, rate_fast=rate,
                     rate_slow=rate, se_fast=0.05, se_slow=0.05)


# --------------------------------------------------------------------------- E-01 LP-B SE

def _flat_lp_z(se_mode: str, seed: int, kappa: float = 2.14, p: float = 0.35, g: int = 8) -> float:
    """z of LP-B after 10 flat rounds of Beta-binomial groups (true slope exactly 0)."""
    rng = np.random.default_rng(seed)
    est = LPEstimator("lp_b", window_rounds=10, se_mode=se_mode)
    for _ in range(10):
        n = int(rng.integers(6, 30))                      # allocations vary round to round
        q = rng.beta(kappa * p, kappa * (1 - p), size=n)
        rates = rng.binomial(g, q) / g
        mean = float(rates.mean())
        var = max(float(rates.var(ddof=1)), p * (1 - p) / g)
        est.observe("e", _post(p), n, raw_rate=mean, raw_var=var)
    return est.compute("e", _post(p))[2]


def test_e01_design_se_gives_calibrated_z_on_a_flat_arm():
    zs = [_flat_lp_z("design", s) for s in range(1500)]
    assert 0.85 < statistics.stdev(zs) < 1.15
    assert 0.02 < np.mean(np.abs(zs) >= 2.0) < 0.09          # nominal two-sided 0.046


def test_e01_legacy_se_is_inflated_and_kept_for_reproducibility():
    zs = [_flat_lp_z("residual_legacy", s) for s in range(400)]
    assert statistics.stdev(zs) < 0.5                         # the bug: z deflated ~sqrt(n)


def test_e01_design_se_detects_a_real_slope():
    """Slope 0.015/round, 40 prompts/round, kappa 2.14: expected z = 0.015 / 0.0053 ~ 2.8."""
    zs = []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        est = LPEstimator("lp_b", window_rounds=10, se_mode="design")
        for t in range(10):
            p = 0.30 + 0.015 * t
            rates = rng.binomial(8, rng.beta(2.14 * p, 2.14 * (1 - p), size=40)) / 8
            est.observe("e", _post(p), 40, raw_rate=float(rates.mean()),
                        raw_var=max(float(rates.var(ddof=1)), p * (1 - p) / 8))
        zs.append(est.compute("e", _post(0.4))[2])
    assert 2.3 < float(np.mean(zs)) < 3.3 and np.mean(np.asarray(zs) > 2.0) > 0.6


def test_e01_old_checkpoints_without_variance_still_load():
    est = LPEstimator("lp_b", window_rounds=10)
    est.load_checkpoint({"history": {"e": [[0.3, 16.0, 0.05], [0.31, 16.0, 0.05], [0.32, 16.0, 0.05]]}})
    slope, se, z = est.compute("e", _post(0.32))
    assert slope == pytest.approx(0.01) and se > 0


# ------------------------------------------------------------- E-02 / E-03 calibration

def _cobs(k, scores, exposure, paired=None):
    return CalibrationObservation(
        window_k=k, round=10 * k, score_total=0.0, score_by_domain=dict(scores), se_total=0.01,
        se_by_domain={d: 0.01 for d in scores}, n_items=100, eval_cost_usd=0.0,
        exposure_by_env=dict(exposure), window_cost_usd=1.0, delta_se_by_domain=paired)


def test_e02_flags_only_reach_arms_that_were_reevaluated():
    from curator_rl.scheduler.ducb import DiscountedUCB

    calib = CFG.calib.model_copy(update={"flag_scope": "evaluated", "k_min": 1})
    sched = DiscountedUCB(["a", "b"], CFG.signals, CFG.proxy, CFG.scheduler, calib, 8)
    seen: list[str] = []
    sched._engine.set_calibration_mismatch = lambda env, flag: seen.append(env)  # noqa: SLF001

    class Rep:
        ready, evaluated, mismatch = True, ("a",), {"a": True, "b": True}

    sched._calibrator.observe = lambda *a, **k: Rep()  # noqa: SLF001
    sched.update_calibration(_cobs(1, {"a": 0.5}, {"a": 1.0, "b": 1.0}))
    assert seen == ["a"]                     # 'b' kept its streak: no new evidence about it

    legacy = CFG.calib.model_copy(update={"flag_scope": "all", "k_min": 1})
    sched2 = DiscountedUCB(["a", "b"], CFG.signals, CFG.proxy, CFG.scheduler, legacy, 8)
    seen.clear()
    sched2._engine.set_calibration_mismatch = lambda env, flag: seen.append(env)  # noqa: SLF001
    sched2._calibrator.observe = lambda *a, **k: Rep()  # noqa: SLF001
    sched2.update_calibration(_cobs(1, {"a": 0.5}, {"a": 1.0, "b": 1.0}))
    assert seen == ["a", "b"]                # legacy behaviour pinned


def test_e03_shared_slice_credit_separates_the_spurious_arm():
    """GSM8K and noisy share the GSM8K slice; only GSM8K training moves it."""
    rng = np.random.default_rng(3)
    cr = SharedSliceCredit({"gsm8k": ["gsm8k", "noisy"]})
    own = OwnSliceCredit(["gsm8k", "noisy"])
    for _ in range(8):
        w_g, w_n = rng.uniform(0.1, 0.5), rng.uniform(0.05, 0.4)
        gain = 0.08 * w_g + rng.normal(0, 0.004)            # noisy contributes nothing
        cr.add("gsm8k", {"gsm8k": w_g, "noisy": w_n}, gain, 0.004)
        own.add("gsm8k", w_g, gain, 0.004)
        own.add("noisy", w_n, gain, 0.004)
    b_noisy, sd_noisy = cr.estimate("noisy")
    b_gsm, _ = cr.estimate("gsm8k")
    assert abs(b_noisy) < 2.5 * sd_noisy and b_gsm == pytest.approx(0.08, abs=0.02)
    assert own.estimate("noisy")[0] > 0.05                 # own-slice credit hands it GSM8K's gains


def test_e03_unshared_transfer_credit_equals_own_slice_credit():
    cr, own = SharedSliceCredit({"a": ["a"]}), OwnSliceCredit(["a"])
    for w, y, se in ((0.5, 0.02, 0.01), (1.0, 0.05, 0.02)):
        cr.add("a", {"a": w}, y, se)
        own.add("a", w, y, se)
    assert cr.estimate("a") == pytest.approx(own.estimate("a"))


def test_e03_shared_slice_is_evaluated_once_and_credits_both_arms():
    cfg = CFG.calib.model_copy(update={"targeting": "all", "credit": "transfer"})
    cal = Calibrator(["gsm8k", "noisy", "math35"], cfg, domain_of={"noisy": "gsm8k"})
    assert set(cal.select_targets({}, {})) == {"gsm8k", "math35"}
    cal.observe(_cobs(1, {"gsm8k": 0.40, "math35": 0.20}, {"gsm8k": 40, "noisy": 40, "math35": 20}),
                {"gsm8k": 0.5, "noisy": 0.5, "math35": 0.4})
    rep = cal.observe(_cobs(2, {"gsm8k": 0.43, "math35": 0.21}, {"gsm8k": 60, "noisy": 20, "math35": 20}),
                      {"gsm8k": 0.5, "noisy": 0.5, "math35": 0.4})
    assert set(rep.evaluated) == {"gsm8k", "noisy", "math35"}
    assert rep.gain_se["noisy"] < float("inf")


# ------------------------------------------------------------------------ E-09 quotas

@pytest.mark.parametrize("total", [32, 80])
def test_e09_carry_quotas_track_requested_shares(total):
    rng = np.random.default_rng(total)
    carry = CarryQuotas()
    requested = {e: 0.0 for e in "abcde"}
    for _ in range(60):
        w = rng.dirichlet(np.ones(5)) * 0.9 + 0.02
        w = dict(zip("abcde", w / w.sum()))
        q = carry.next(w, total)
        assert sum(q.values()) == total and min(q.values()) >= 0
        for e in w:
            requested[e] += w[e] * total
        assert max(abs(carry.delivered[e] - requested[e]) for e in w) < 1.0 + 1e-9


def test_e09_largest_remainder_over_serves_small_arms_but_carry_does_not():
    w = {"gsm8k": 0.40, "math35": 0.25, "mbpp": 0.25, "countdown": 0.02, "noisy": 0.08}
    lr = sum(largest_remainder_quotas(w, 32, tie_break_seed=t)["countdown"] for t in range(60))
    carry = CarryQuotas()
    cq = sum(carry.next(w, 32)["countdown"] for _ in range(60))
    assert lr / (60 * 32) > 0.03                          # 0.031 delivered for 0.020 requested
    assert abs(cq / (60 * 32) - 0.02) < 1.0 / (60 * 32)


# ---------------------------------------------------------------------- E-06 / E-07 / E-08

@pytest.mark.parametrize("pred,gold,ok", [
    ("50.5", "10\\frac{1}{2}", False),        # was accepted: 10 1/2 read as 101/2
    ("\\frac{21}{2}", "10\\frac{1}{2}", True),  # was rejected
    ("23", "2(3)", False),                     # implicit product read as 23
    ("\\frac{5}{2}", "2\\frac{1}{2}", True),
    ("-\\frac{1}{2}", "\\frac{-1}{2}", True),
    ("0.5", "\\frac12", True),
])
def test_e06_math_checker_mixed_numbers_and_products(pred, gold, ok):
    from curator_rl.envs.math35 import answers_equal
    assert answers_equal(pred, gold) is ok


SBX_TESTS = ["assert add(2, 3) == 5", "assert add(10, 20) == 30"]


def test_e07_sandbox_refuses_token_forging_and_always_equal_objects():
    from curator_rl.envs.sandbox import run_python_tests
    forge = ('import re, os\nprint(re.search(r"SBX[0-9a-f]+", open(__file__).read()).group(0), flush=True)\n'
             "os._exit(0)\n")
    always_eq = "class A:\n    def __eq__(self, o):\n        return True\ndef add(a, b):\n    return A()\n"
    frame_walk = "import sys\ndef add(a, b):\n    return sys._getframe().f_globals\n"
    for code in (forge, always_eq, frame_walk):
        res = run_python_tests(code, SBX_TESTS)
        assert not res.passed and res.rejected is not None


def test_e07_strict_equality_still_accepts_honest_builtin_answers():
    from curator_rl.envs.sandbox import run_python_tests
    assert run_python_tests("def add(a, b):\n    return a + b\n", SBX_TESTS).passed
    assert run_python_tests("def f(input, k):\n    return input[:k]\n", ["assert f([1, 2, 3], 2) == [1, 2]"]).passed
    assert run_python_tests("def f():\n    return 2\n", ["assert f() == 2.0"]).passed   # int vs float ok
    assert not run_python_tests("def f():\n    return (1, 2)\n", ["assert f() == [1, 2]"]).passed


@pytest.mark.parametrize("resp,gold", [
    ("\\boxed{18}", "18"), ("\\boxed{\\$18}", "18"), ("\\boxed{18 \\text{ dollars}}", "18"),
    ("\\boxed{1{,}000}", "1000"), ("\\boxed{1,000}", "1000"), ("\\boxed{\\$1,250.50}", "2501/2"),
])
def test_e08_gsm8k_checker_accepts_common_answer_formats(resp, gold):
    from curator_rl.envs.gsm8k import _verify_boxed
    assert _verify_boxed(resp, gold)[1]


def test_e08_gsm8k_checker_still_rejects_wrong_or_nonnumeric_answers():
    from curator_rl.envs.gsm8k import _verify_boxed
    for resp in ("\\boxed{17}", "\\boxed{18 or 19}", "\\boxed{x}", "no box 18"):
        assert not _verify_boxed(resp, "18")[1]


# ------------------------------------------------------------------- E-13 / E-17 / E-25

def test_e13_jsonl_keeps_records_written_after_a_crash(tmp_path):
    from curator_rl.core.jsonl import JsonlWriter, read_jsonl
    path = tmp_path / "log.jsonl"
    with JsonlWriter(path, 1) as w:
        w.write({"round": 1})
        w.write({"round": 2})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"_schema": 1, "_seq": 3, "rou')            # killed mid-write
    with JsonlWriter(path, 1) as w:
        for r in (4, 5):
            w.write({"round": r})
    assert [r["round"] for r in read_jsonl(path)] == [1, 2, 4, 5]
    assert (tmp_path / "log.jsonl.corrupt-tail").exists()


def test_e25_guard_fails_closed_on_a_corrupted_access_log(tmp_path):
    from curator_rl.evaluation.guard import SealedTest, SealedTestError
    log = tmp_path / "reports" / "test_access_log.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"checkpoint_hash": "abc", "split": "test"}\n{garbage\n', encoding="utf-8")
    guard = SealedTest(tmp_path / "data", tmp_path, "test", "abc")
    with pytest.raises(SealedTestError):
        guard._register_access()                               # noqa: SLF001


def test_e17_replay_carries_mixed_groups_and_sums(tmp_path):
    import json as _json

    from curator_rl.scheduler.baselines.uniform import UniformScheduler
    from curator_rl.simulator.replay import TraceReplayer

    trace = tmp_path / "trace.jsonl"
    row = {"k_success": 40, "n_rollouts": 80, "n_prompts": 10, "n_groups_mixed": 6, "cost_usd": 1.0,
           "sum_score": 5.0, "sum_score_sq": 3.0, "sum_abs_adv": 4.0}
    trace.write_text("\n".join(_json.dumps({"round": t, "per_env": {"a": row, "b": row}}) for t in range(3)),
                     encoding="utf-8")
    seen = []

    class Spy(UniformScheduler):
        def update_observation(self, obs):
            seen.append(obs)

    TraceReplayer(["a", "b"], trace).run(Spy(["a", "b"]), prompts_per_round=7, budget_usd=100.0, max_rounds=2)
    obs = seen[0]
    assert sum(o.n_prompts for o in obs.per_env.values()) == 7
    assert all(o.n_groups_mixed > 0 and o.sum_abs_adv > 0 for o in obs.per_env.values())


# ------------------------------------------------------------------------------ E-21

def test_e21_flip_rate_counts_only_the_tail():
    assert flip_rate([EnvStatus.S1, EnvStatus.S2, EnvStatus.S2, EnvStatus.S3], warmup=2) == 100.0


# ---------------------------------------------------------------------- B5 / E-16 / E-10

def test_b5_s5_cap_floor_holds_spurious_arm_at_floor():
    from curator_rl.core.types import SignalVector
    from curator_rl.scheduler.mixture import apply_status_constraints

    def sv(env, status):
        return SignalVector(env_id=env, round_t=10, n_rounds_seen=10, n_groups_eff=200.0,
                            pass_rate=0.4, pass_lo=0.35, pass_hi=0.45, lp=0.0, lp_se=0.01, lp_z=0.0,
                            lp_raw_fast=0.4, lp_raw_slow=0.4, richness=0.9, mean_score=0.4,
                            unit_cost_usd=1.0, unit_cost_norm=1.0, proxy_raw=0.4, proxy_unit=0.4,
                            proxy_reward=0.5, status=status, status_note="")

    vectors = {"a": sv("a", EnvStatus.S2), "b": sv("b", EnvStatus.S2), "noisy": sv("noisy", EnvStatus.S5)}
    w = {"a": 0.3, "b": 0.3, "noisy": 0.4}
    capped, _ = apply_status_constraints(w, vectors, CFG.scheduler.model_copy(update={"s5_cap": "floor"}),
                                         n_min=64)
    halved, _ = apply_status_constraints(w, vectors, CFG.scheduler.model_copy(update={"s5_cap": "none"}),
                                         n_min=64)
    floor = CFG.scheduler.epsilon / 3
    assert capped["noisy"] == pytest.approx(floor) and halved["noisy"] > 2 * floor


def test_e16_dump_window_forgets_old_rounds():
    from curator_rl.core.types import EnvRoundObs, RoundObservation
    from curator_rl.scheduler.baselines.dump import DUMPStyleUCB

    def obs(adv_a):
        per = {e: EnvRoundObs(env_id=e, n_prompts=10, n_rollouts=80, k_success=40, n_groups_mixed=5,
                              sum_score=5.0, sum_score_sq=2.5, prompt_tokens=0, completion_tokens=0,
                              verifier_seconds=0.0, gpu_seconds=0.0, cost_usd=1.0,
                              sum_abs_adv=adv_a if e == "a" else 5.0) for e in ("a", "b")}
        return RoundObservation(round=1, steps=1, per_env=per, weights_used={"a": 0.5, "b": 0.5},
                                round_cost_usd=2.0, overhead_usd=0.0, budget_remaining_usd=10.0)

    win = DUMPStyleUCB(["a", "b"], CFG.baselines.dump.model_copy(update={"window_rounds": 2}), CFG.scheduler)
    for adv in (9.0, 9.0, 0.0, 0.0):                         # arm 'a' saturates after two rounds
        win.update_observation(obs(adv))
    assert win._sum["a"] == pytest.approx(0.0)               # noqa: SLF001 - only recent rounds count


def test_e10_episode_reports_noiseless_final_score():
    from pathlib import Path

    from experiments.run_sim import make_scheduler

    from curator_rl.simulator.harness import run_episode
    from curator_rl.simulator.scenarios import load_scenario

    sc = load_scenario(Path(BASE_CFG).parents[0] / "sim" / "scenario_sa.yaml")
    r = run_episode(make_scheduler("uniform", sc, 0), sc, 0, method="uniform")
    assert 0.0 < r.final_score_expected < 1.0 and not math.isnan(r.final_score_expected)
