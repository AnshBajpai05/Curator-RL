"""Reward audit (signals/audit.py): a reward that ignores the prompt is caught; real verifiers are not.

The thresholds are the configured ones (d_max 1.5, z 2.33, min_df 40); the populations are the ones
measured on the Kaggle pilot: verifier arms with prompt heterogeneity kappa ~ 1-2.5 (dispersion
D = 3.0-4.4) and a Bernoulli reward that ignores the prompt (D = 1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from curator_rl.core.config import RewardAuditCfg, StatusCfg
from curator_rl.core.types import EnvRoundObs, EnvStatus
from curator_rl.signals.audit import PromptDependenceAudit
from curator_rl.signals.status import StatusClassifier

G = 8
CFG = RewardAuditCfg(enabled=True, d_max=1.5, z=2.33, min_df=40)


def _round(rng, kappa, p, n):
    q = np.full(n, p) if kappa == math.inf else rng.beta(kappa * p, kappa * (1 - p), size=n)
    k = rng.binomial(G, q)
    r = k / G
    return EnvRoundObs("x", n, n * G, int(k.sum()), int(((k > 0) & (k < G)).sum()), float(r.sum()),
                       float((r * r).sum()), 0, 0, 0.0, 0.0, 0.0)


def _first_flag(rng, kappa, p, rounds=60):
    audit = PromptDependenceAudit(CFG, ["x"], G)
    for t in range(rounds):
        if audit.observe(_round(rng, kappa, p, int(rng.integers(3, 11)))).flag:
            return t + 1
    return None


def test_dispersion_estimates_the_beta_binomial_ratio():
    rng = np.random.default_rng(0)
    for kappa, expected in ((math.inf, 1.0), (2.14, (G + 2.14) / 3.14)):
        audit = PromptDependenceAudit(CFG, ["x"], G)
        for _ in range(400):
            res = audit.observe(_round(rng, kappa, 0.35, 8))
        assert res.dispersion == pytest.approx(expected, rel=0.08)


def test_a_reward_that_ignores_the_prompt_is_flagged_early():
    rng = np.random.default_rng(1)
    firsts = [_first_flag(rng, math.inf, 0.35) for _ in range(200)]
    assert sum(f is not None for f in firsts) >= 196                 # >= 98% of 60-round runs
    assert np.median([f for f in firsts if f]) <= 14                  # typically by round ~11


def test_real_verifier_heterogeneity_is_never_flagged():
    rng = np.random.default_rng(2)
    for p in (0.1, 0.35, 0.6):
        firsts = [_first_flag(rng, 2.14, p) for _ in range(200)]
        assert sum(f is not None for f in firsts) == 0


def test_no_decision_before_min_df_and_uninformative_rounds_are_ignored():
    audit = PromptDependenceAudit(CFG, ["x"], G)
    zeros = EnvRoundObs("x", 8, 64, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0)   # all-fail round: no information
    for _ in range(50):
        res = audit.observe(zeros)
    assert res.df == 0 and not res.flag and math.isnan(res.dispersion)
    rng = np.random.default_rng(3)
    for _ in range(4):                                                     # 4 x 7 = 28 df < 40
        res = audit.observe(_round(rng, math.inf, 0.35, 8))
    assert res.df == 28 and not res.flag


def test_flag_clears_only_on_strong_evidence_of_prompt_dependence():
    rng = np.random.default_rng(4)
    audit = PromptDependenceAudit(CFG, ["x"], G)
    for _ in range(10):
        res = audit.observe(_round(rng, math.inf, 0.35, 8))
    assert res.flag
    for _ in range(3):                                   # a little real-looking data does not clear it
        res = audit.observe(_round(rng, 1.0, 0.35, 8))
    assert res.flag
    for _ in range(80):                                  # a reward that became prompt-dependent does
        res = audit.observe(_round(rng, 1.0, 0.35, 8))
    assert not res.flag


def test_checkpoint_round_trip():
    rng = np.random.default_rng(5)
    a = PromptDependenceAudit(CFG, ["x"], G)
    for _ in range(7):
        a.observe(_round(rng, math.inf, 0.35, 8))
    b = PromptDependenceAudit(CFG, ["x"], G)
    b.load_checkpoint(a.get_state())
    obs = _round(rng, math.inf, 0.35, 8)
    assert a.observe(obs) == b.observe(obs)


def test_a_preflight_flag_holds_for_the_whole_run_and_survives_a_checkpoint():
    rng = np.random.default_rng(6)
    audit = PromptDependenceAudit(CFG, ["x", "y"], G)
    audit.preflag(["x", "not-an-arm"])
    assert audit.result("x").flag and not audit.result("y").flag
    for _ in range(80):                       # strong evidence of a real verifier never clears it
        audit.observe(_round(rng, 2.14, 0.35, 8))
    assert audit.result("x").flag
    clone = PromptDependenceAudit(CFG, ["x", "y"], G)
    clone.load_checkpoint(audit.get_state())
    assert clone.result("x").flag
    older = audit.get_state()
    del older["preflight"]                    # checkpoints from before D-102
    clone.load_checkpoint(older)
    assert not clone.result("x").flag


def _classifier():
    status = StatusCfg(n_min=64, r_min=3, p_sat=0.7, p_hard=0.05, sr_hard=0.1, z_up=1.5, z_neg=2.0,
                       hysteresis_p=0.05, hysteresis_sr=0.05, dwell_min=3, consecutive_rounds=4,
                       s1_entry_ratio=0.5)
    return StatusClassifier(status, mismatch_windows=2, clear_windows=2)


def test_audit_s5_takes_priority_over_s1_and_releases_when_cleared():
    c = _classifier()
    args = dict(n_groups_eff=10.0, rounds_seen=2, p_hat=0.35, p_lo=0.3, p_hi=0.4, z_lp=0.0, sr=0.9)
    assert c.step(**args)[0] == EnvStatus.S1                          # little evidence: exploring
    c.set_audit(True)
    assert c.step(**args)[0] == EnvStatus.S5                          # the audit overrides S1
    args.update(n_groups_eff=200.0, rounds_seen=20)
    assert c.step(**args)[0] == EnvStatus.S5
    c.set_audit(False)
    assert c.step(**args)[0] == EnvStatus.S2                          # no calibration streak to wait for


def test_audit_off_leaves_calibration_s5_rules_unchanged():
    c = _classifier()
    args = dict(n_groups_eff=200.0, rounds_seen=20, p_hat=0.35, p_lo=0.3, p_hi=0.4, z_lp=0.0, sr=0.9)
    c.step(**args)
    c.set_mismatch(True)
    c.set_mismatch(True)
    assert c.step(**args)[0] == EnvStatus.S5
    c.set_mismatch(False)
    assert c.step(**args)[0] == EnvStatus.S5                          # needs q' = 2 clean windows
    c.set_mismatch(False)
    assert c.step(**args)[0] == EnvStatus.S2
