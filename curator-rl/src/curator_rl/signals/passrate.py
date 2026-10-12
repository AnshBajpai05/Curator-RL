"""Discounted pooled pass-rate estimation (Roadmap E.2, Signal Engine Phase 4).

Each environment keeps three discounted count pairs (main λ, fast λ_f, slow
λ_s) so the LP-A estimator can reuse the same state, plus a discounted
prompt-group count for the group-level standard error (rollouts within a
prompt group are correlated, so the group -- not the rollout -- is the unit;
Roadmap E.2). The posterior uses a Beta(a0, b0) prior and the 68% interval is
the posterior mean +/- one posterior standard deviation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from curator_rl.core.types import EnvRoundObs


@dataclass(frozen=True)
class Posterior:
    """Everything the engine, the LP estimator and the status classifier need."""

    env_id: str
    pass_rate: float
    pass_lo: float
    pass_hi: float
    se_group: float
    n_groups_eff: float
    n_rounds_seen: int
    rate_fast: float
    rate_slow: float
    se_fast: float
    se_slow: float


@dataclass
class _Counts:
    s: float = 0.0
    n: float = 0.0
    groups: float = 0.0

    def decay_tick(self, k: float, n: float, groups: float, lam: float) -> None:
        self.s = lam * self.s + k
        self.n = lam * self.n + n
        self.groups = lam * self.groups + groups


@dataclass
class _EnvState:
    main: _Counts = field(default_factory=_Counts)
    fast: _Counts = field(default_factory=_Counts)
    slow: _Counts = field(default_factory=_Counts)
    rounds_seen: int = 0


class PassRateTracker:
    """Discounted pooled per-environment pass rates (Roadmap E.2)."""

    def __init__(
        self,
        lam: float,
        lambda_fast: float,
        lambda_slow: float,
        alpha0: float,
        beta0: float,
    ) -> None:
        self._lam = lam
        self._lambda_fast = lambda_fast
        self._lambda_slow = lambda_slow
        self._a0 = alpha0
        self._b0 = beta0
        self._envs: dict[str, _EnvState] = {}

    # ------------------------------------------------------------------ update

    def update(self, obs: EnvRoundObs) -> Posterior:
        """Fold one round's counts in (decay applies even on a zero-prompt round)."""
        state = self._envs.setdefault(obs.env_id, _EnvState())
        k = float(obs.k_success)
        n = float(obs.n_rollouts)
        groups = float(obs.n_prompts)
        state.main.decay_tick(k, n, groups, self._lam)
        state.fast.decay_tick(k, n, groups, self._lambda_fast)
        state.slow.decay_tick(k, n, groups, self._lambda_slow)
        if obs.n_prompts > 0:
            state.rounds_seen += 1

        rate, lo, hi = self._interval(state.main)
        rate_fast, _, _ = self._interval(state.fast)
        rate_slow, _, _ = self._interval(state.slow)

        # group-level SE on the main window (max with 1 guard against /0)
        se_group = math.sqrt(rate * (1.0 - rate) / max(state.main.groups, 1.0))
        se_fast = self._group_se(state.fast)
        se_slow = self._group_se(state.slow)

        return Posterior(
            env_id=obs.env_id,
            pass_rate=rate,
            pass_lo=lo,
            pass_hi=hi,
            se_group=se_group,
            n_groups_eff=state.main.groups,
            n_rounds_seen=state.rounds_seen,
            rate_fast=rate_fast,
            rate_slow=rate_slow,
            se_fast=se_fast,
            se_slow=se_slow,
        )

    def rounds_seen(self, env_id: str) -> int:
        return self._envs.get(env_id, _EnvState()).rounds_seen

    # ----------------------------------------------------------------- helpers

    def _posterior_stats(self, counts: _Counts) -> tuple[float, float]:
        strength = counts.n + self._a0 + self._b0
        rate = (counts.s + self._a0) / strength
        variance = rate * (1.0 - rate) / (strength + 1.0)
        return rate, math.sqrt(max(variance, 0.0))

    def _interval(self, counts: _Counts) -> tuple[float, float, float]:
        rate, std = self._posterior_stats(counts)
        return rate, rate - std, rate + std

    def _group_se(self, counts: _Counts) -> float:
        rate, std = self._posterior_stats(counts)
        return math.sqrt(rate * (1.0 - rate) / max(counts.groups, 1.0)) if counts.groups > 0 else std

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for env_id, st in self._envs.items():
            out[env_id] = {
                "main": [st.main.s, st.main.n, st.main.groups],
                "fast": [st.fast.s, st.fast.n, st.fast.groups],
                "slow": [st.slow.s, st.slow.n, st.slow.groups],
                "rounds_seen": st.rounds_seen,
            }
        return out

    def load_checkpoint(self, state: Mapping[str, Mapping[str, Any]]) -> None:
        self._envs = {}
        for env_id, raw in state.items():
            st = _EnvState(
                main=_Counts(*raw["main"]),
                fast=_Counts(*raw["fast"]),
                slow=_Counts(*raw["slow"]),
                rounds_seen=int(raw["rounds_seen"]),
            )
            self._envs[env_id] = st
