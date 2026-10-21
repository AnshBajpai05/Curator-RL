"""Reward audit: does an arm's reward depend on the prompt at all? (label-free, zero extra compute)

A verifier scores a completion against ITS prompt, and prompts differ in difficulty, so the pass
rates of a round's prompt groups spread out more than coin flips with the same mean would. A reward
that ignores the prompt (a random reward, a constant, a format-only check whose outcome is unrelated
to the question) shows no such excess. Under a Beta-binomial model with concentration kappa, the
group pass rate k/G has variance

    Var(k/G) = p (1 - p) / G * D,      D = (G + kappa) / (kappa + 1)

so the dispersion ratio D is 1 for a prompt-independent reward and grows as prompts differ.

Measured on the Kaggle pilot (Qwen2.5 0.5B and 1.5B, 64 prompts x G = 8 per arm; the implied kappa
from the mixed-group share): GSM8K D = 3.6 / 3.0, MATH35 3.4 / 4.4, MBPP 3.5 / 4.1, while the noisy
arm (Bernoulli(0.35) reward) is 1.0 / 1.09. `d_max` sits between the two populations.

Estimator. Per round, from the moments every observation already carries (sum and sum of squares of
the group rates): SS_t = sum_g (r_g - rbar)^2 with n_t - 1 degrees of freedom, and its expectation
under prompt independence E0_t = (n_t - 1) v_t / G, where v_t = pbar (1 - pbar) N / (N - 1) is the
unbiased Bernoulli variance of the round's N = n_t G rollouts. Using the round's own mean makes the
test immune to learning (the pass rate may drift between rounds). Pooled over rounds,

    D_hat = sum SS_t / sum E0_t,    df = sum (n_t - 1),    log D_hat ~ N(log D, 2 / df).

Decision (hysteresis through the confidence interval):
- flag (reward does not depend on the prompt) when the UPPER bound log D_hat + z sqrt(2/df) is below
  log d_max and df >= min_df;
- clear a flag only when the LOWER bound log D_hat - z sqrt(2/df) is above log d_max.

What it can and cannot catch: random, constant and prompt-unrelated rewards. It does NOT catch a
reward that depends on the prompt but rewards the wrong thing (e.g. a wrong answer key); that is the
benchmark calibration's job (calibration/calibrator.py). Pure L1: no RNG, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NotRequired, TypedDict

from curator_rl.core.config import RewardAuditCfg
from curator_rl.core.types import EnvRoundObs


class AuditState(TypedDict):
    ss: dict[str, float]
    e0: dict[str, float]
    df: dict[str, int]
    flag: dict[str, bool]
    preflight: NotRequired[list[str]]      # arms flagged before training (D-102); absent in older checkpoints


@dataclass(frozen=True)
class AuditResult:
    dispersion: float        # D_hat (nan before any informative round)
    df: int
    flag: bool               # True: the reward shows no dependence on the prompt


class PromptDependenceAudit:
    """Pooled dispersion test per environment (see module docstring)."""

    def __init__(self, cfg: RewardAuditCfg, env_ids, group_size: int) -> None:
        self.cfg = cfg
        self.group_size = int(group_size)
        self._ss = {e: 0.0 for e in env_ids}
        self._e0 = {e: 0.0 for e in env_ids}
        self._df = {e: 0 for e in env_ids}
        self._flag = {e: False for e in env_ids}
        self._preflight: set[str] = set()

    def preflag(self, env_ids) -> None:
        """Arms whose reward paid for junk in the pre-flight verifier check: flagged for the whole run.

        Kept apart from the evidence-based flag, which later rounds may clear; this one never clears.
        """
        self._preflight |= {e for e in env_ids if e in self._flag}

    def observe(self, obs: EnvRoundObs) -> AuditResult:
        e = obs.env_id
        n = int(obs.n_prompts)
        if n >= 2 and obs.n_rollouts > n:
            g = obs.n_rollouts / n                     # completions per prompt (G)
            p = obs.k_success / obs.n_rollouts
            big_n = obs.n_rollouts
            v = p * (1.0 - p) * big_n / (big_n - 1)
            if v > 0.0:                                 # all-0 / all-1 rounds carry no information
                mean = obs.sum_score / n
                ss = max(obs.sum_score_sq - n * mean * mean, 0.0)
                self._ss[e] += ss
                self._e0[e] += (n - 1) * v / g
                self._df[e] += n - 1
        return self._decide(e)

    def _decide(self, e: str) -> AuditResult:
        df, e0 = self._df[e], self._e0[e]
        pre = e in self._preflight
        if df == 0 or e0 <= 0.0:
            return AuditResult(float("nan"), df, self._flag[e] or pre)
        d_hat = max(self._ss[e] / e0, 1e-9)
        half = self.cfg.z * math.sqrt(2.0 / df)
        log_d, log_max = math.log(d_hat), math.log(self.cfg.d_max)
        if df >= self.cfg.min_df:
            if not self._flag[e] and log_d + half < log_max:
                self._flag[e] = True
            elif self._flag[e] and log_d - half > log_max:
                self._flag[e] = False
        return AuditResult(d_hat, df, self._flag[e] or pre)

    def result(self, e: str) -> AuditResult:
        return self._decide(e)

    # ------------------------------------------------------------------ checkpoints

    def get_state(self) -> AuditState:
        return {"ss": dict(self._ss), "e0": dict(self._e0), "df": dict(self._df), "flag": dict(self._flag),
                "preflight": sorted(self._preflight)}

    def load_checkpoint(self, state: AuditState) -> None:
        self._ss = {e: float(v) for e, v in state["ss"].items()}
        self._e0 = {e: float(v) for e, v in state["e0"].items()}
        self._df = {e: int(v) for e, v in state["df"].items()}
        self._flag = {e: bool(v) for e, v in state["flag"].items()}
        self._preflight = set(state.get("preflight", []))
