"""Proxy reward with cost normalisation and quantile mapping (Roadmap E.4, S-5).

Pipeline per round, per environment (the engine drives the ordering via
`update_lp_values`, sorted by env id, so the pooled sigma statistic stays
permutation-invariant):

    sigma_LP  discounted std of the transformed LP values seen across envs
    LP'       clip(lp / sigma_LP, -L, L)
    x         alpha * LP' + beta * SR
    r_tilde   x / c_norm^theta_c      (theta_c = 0 => no cost normalisation)
    r_bar     clip((r_tilde - q_lo) / (q_hi - q_lo), 0, 1)

q_lo / q_hi are the running 5th / 95th percentiles of r_tilde, estimated from
a bounded reservoir; before `min_quantile_samples` values exist the config
priors are used. The proxy is deterministic end to end (no RNG).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NotRequired, TypedDict

from curator_rl.core.config import ProxyCfg


class ProxyState(TypedDict):
    mean: float
    var: float
    count: int
    reservoir: list[float]
    alpha: NotRequired[float]           # absent before the alpha/beta fit existed
    beta: NotRequired[float]


@dataclass(frozen=True)
class ProxyResult:
    """x (no cost map), r_tilde (cost-normalised), r_bar (mapped to [0, 1]); LP' as used in x."""

    proxy_raw: float
    proxy_unit: float
    proxy_reward: float
    lp_scaled: float = 0.0


class ProxyReward:
    """Proxy reward E.4: sigma-scaled LP, cost normalisation, quantile map."""

    def __init__(
        self,
        cfg: ProxyCfg,
        lam: float,
        cost_exponent: float,
        *,
        min_quantile_samples: int = 20,
        reservoir_size: int = 200,
    ) -> None:
        self._cfg = cfg
        self.alpha, self.beta = float(cfg.alpha), float(cfg.beta)   # fitted values replace these (B4)
        self._lam = lam
        self._cost_exponent = cost_exponent
        self._min_samples = min_quantile_samples
        self._lp_mean = 0.0
        self._lp_var = 0.0
        self._lp_count = 0
        self._reservoir: deque[float] = deque(maxlen=reservoir_size)

    # ---------------------------------------------------------------- update

    def update_lp_values(self, lp_values: Mapping[str, float]) -> None:
        """Track the discounted std of transformed LP across envs; call before compute."""
        # env dict order is not guaranteed: iterate sorted for permutation invariance
        for env_id in sorted(lp_values):
            value = float(lp_values[env_id])
            self._lp_count += 1
            if self._lp_count == 1:
                self._lp_mean = value
                self._lp_var = 0.0
                continue
            lam = self._lam
            self._lp_mean = lam * self._lp_mean + (1.0 - lam) * value
            self._lp_var = lam * self._lp_var + (1.0 - lam) * (value - self._lp_mean) ** 2

    def sigma_lp(self) -> float:
        return math.sqrt(max(self._lp_var, 0.0))

    def clip_lp(self, lp: float) -> float:
        sigma = self.sigma_lp()
        if sigma <= 1e-6:
            return 0.0
        return max(min(lp / sigma, self._cfg.clip_l), -self._cfg.clip_l)

    # ---------------------------------------------------------------- compute

    def compute(
        self,
        lp_transformed: Mapping[str, float],
        richness: Mapping[str, float],
        unit_cost_norm: Mapping[str, float],
    ) -> dict[str, ProxyResult]:
        out: dict[str, ProxyResult] = {}
        raws: list[float] = []
        for env_id in sorted(lp_transformed):
            sigma = self.sigma_lp()
            lp = lp_transformed[env_id]
            lp_scaled = 0.0 if sigma <= 1e-6 else max(min(lp / sigma, self._cfg.clip_l), -self._cfg.clip_l)
            x = self.alpha * lp_scaled + self.beta * richness.get(env_id, 0.0)
            cost = unit_cost_norm.get(env_id, 0.0)
            if self._cost_exponent == 0.0 or cost <= 0.0:
                # unknown unit cost (never observed): assume expensive => reward 0
                r_tilde = x if cost > 0.0 else 0.0
            else:
                r_tilde = x / (cost**self._cost_exponent)
            out[env_id] = ProxyResult(proxy_raw=x, proxy_unit=r_tilde, proxy_reward=self._map(r_tilde),
                                      lp_scaled=lp_scaled)
            raws.append(r_tilde)
        self._reservoir.extend(raws)
        return out

    def _map(self, r_tilde: float) -> float:
        if len(self._reservoir) < self._min_samples:
            lo, hi = self._cfg.quantile_prior_lo, self._cfg.quantile_prior_hi
        else:
            values = sorted(self._reservoir)
            n = len(values)
            lo = values[max(int(0.05 * (n - 1)), 0)]
            hi = values[max(int(0.95 * (n - 1)), 0)]
        span = max(hi - lo, 1e-12)
        return max(min((r_tilde - lo) / span, 1.0), 0.0)

    def set_weights(self, alpha: float, beta: float) -> None:
        """Replace (alpha, beta), e.g. with the calibrated fit (improvement B4)."""
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError(f"invalid proxy weights alpha={alpha}, beta={beta}")
        self.alpha, self.beta = float(alpha), float(beta)

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> ProxyState:
        return {
            "mean": self._lp_mean,
            "var": self._lp_var,
            "count": self._lp_count,
            "reservoir": list(self._reservoir),
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def load_checkpoint(self, state: ProxyState) -> None:
        self._lp_mean = float(state["mean"])
        self._lp_var = float(state["var"])
        self._lp_count = int(state["count"])
        self._reservoir = deque((float(v) for v in state["reservoir"]), maxlen=self._reservoir.maxlen)
        # checkpoints from before the alpha/beta fit carry no weights: the configured ones apply
        self.alpha = float(state.get("alpha", self._cfg.alpha))
        self.beta = float(state.get("beta", self._cfg.beta))
