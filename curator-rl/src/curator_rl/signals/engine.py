"""SignalEngine facade (Roadmap E.2-E.5, Phase 4).

Consumes the same `RoundObservation` stream the real trainer and the
simulator produce, and yields a per-environment `SignalVector` each round.
Pure L1: no torch, no datasets, no simulator imports, deterministic (no RNG).

Per-round update order (deterministic, permutation-invariant over env ids):

1. discounted pooled pass-rate counts (decay even on zero-prompt rounds);
2. LP history observation + (lp, se, z) via the configured estimator;
3. lp_use transformation (signed | positive | abs);
4. proxy sigma tracking over the transformed values (sorted iteration);
5. unit-cost EMA per environment (interim D-33: from `EnvRoundObs.cost_usd`;
   the real Cost Meter arrives in Phase 6 with the same interface);
6. richness (mode per config);
7. proxy rewards (sigma scale, cost normalisation, quantile map);
8. status classifier per environment.
"""

from __future__ import annotations

from collections.abc import Mapping

from curator_rl.core.config import CalibCfg, ProxyCfg, SignalsCfg
from curator_rl.core.types import (
    CalibrationObservation,
    EnvRoundObs,
    EnvStatus,
    RoundObservation,
    SignalVector,
)
from curator_rl.signals.passrate import PassRateTracker, Posterior
from curator_rl.signals.progress import LPEstimator
from curator_rl.signals.proxy import ProxyReward
from curator_rl.signals.richness import RichnessEstimator
from curator_rl.signals.status import StatusClassifier


def _zero_obs(env_id: str) -> EnvRoundObs:
    return EnvRoundObs(
        env_id=env_id, n_prompts=0, n_rollouts=0, k_success=0, n_groups_mixed=0,
        sum_score=0.0, sum_score_sq=0.0, prompt_tokens=0, completion_tokens=0,
        verifier_seconds=0.0, gpu_seconds=0.0, cost_usd=0.0,
    )


class SignalEngine:
    """Turns `RoundObservation`s into per-environment signals and statuses."""

    def __init__(
        self,
        signals_cfg: SignalsCfg,
        proxy_cfg: ProxyCfg,
        calib_cfg: CalibCfg,
        cost_exponent: float,
        group_size: int,
        env_ids: list[str] | tuple[str, ...],
    ) -> None:
        env_ids_t = tuple(env_ids)
        if not env_ids_t:
            raise ValueError("SignalEngine needs at least one environment")
        if len(set(env_ids_t)) != len(env_ids_t):
            raise ValueError("SignalEngine env_ids must be unique")
        self.env_ids = env_ids_t
        self._tracker = PassRateTracker(
            signals_cfg.lam, signals_cfg.lambda_fast, signals_cfg.lambda_slow,
            signals_cfg.prior.alpha0, signals_cfg.prior.beta0,
        )
        self._lp = LPEstimator(signals_cfg.lp_method, signals_cfg.window_rounds, signals_cfg.lp_se)
        self._group_size = int(group_size)
        self._richness = RichnessEstimator(signals_cfg.richness, signals_cfg.lam, group_size)
        self._proxy = ProxyReward(proxy_cfg, signals_cfg.lam, cost_exponent)
        self._classifiers: dict[str, StatusClassifier] = {
            e: StatusClassifier(
                signals_cfg.status,
                mismatch_windows=calib_cfg.mismatch_windows,
                clear_windows=calib_cfg.mismatch_clear_windows,
            )
            for e in self.env_ids
        }
        self._signals_cfg = signals_cfg
        self._unit_cost: dict[str, float] = {e: 0.0 for e in self.env_ids}
        self._round_t = 0

    # ------------------------------------------------------------------ update

    def update(self, observation: RoundObservation) -> dict[str, SignalVector]:
        """Process one round; returns per-env SignalVector (statuses included)."""
        unknown = set(observation.per_env) - set(self.env_ids)
        if unknown:
            raise ValueError(f"SignalEngine received unknown env ids: {sorted(unknown)}")
        self._round_t = observation.round

        sorted_envs = sorted(self.env_ids)
        posteriors: dict[str, Posterior] = {}
        means: dict[str, float] = {}
        for env_id in sorted_envs:
            obs = observation.per_env.get(env_id, _zero_obs(env_id))
            posterior = self._tracker.update(obs)
            posteriors[env_id] = posterior
            raw_rate = obs.k_success / obs.n_rollouts if obs.n_rollouts > 0 else None
            self._lp.observe(env_id, posterior, obs.n_prompts, raw_rate,
                             self._within_round_variance(obs, posterior.pass_rate))
            means[env_id] = obs.sum_score / obs.n_prompts if obs.n_prompts > 0 else 0.0
            if obs.n_prompts > 0 and obs.cost_usd > 0:
                ema = self._unit_cost[env_id]
                lam = self._signals_cfg.lam
                unit = obs.cost_usd / obs.n_prompts
                self._unit_cost[env_id] = unit if ema == 0.0 else lam * ema + (1.0 - lam) * unit

        lp_used: dict[str, float] = {}
        lp_info: dict[str, tuple[float, float, float]] = {}
        for env_id in sorted_envs:
            posterior = posteriors[env_id]
            lp, se, z = self._lp.compute(env_id, posterior)
            lp_info[env_id] = (lp, se, z)
            match self._signals_cfg.lp_use:
                case "signed":
                    value = lp
                case "positive":
                    value = max(lp, 0.0)
                case _:  # "abs"
                    value = abs(lp)
            lp_used[env_id] = value

        self._proxy.update_lp_values(lp_used)

        richness: dict[str, float] = {}
        for env_id in sorted_envs:
            obs = observation.per_env.get(env_id, _zero_obs(env_id))
            richness[env_id] = self._richness.update(env_id, obs, posteriors[env_id])

        mean_cost = 0.0
        known = [c for e in sorted_envs if (c := self._unit_cost[e]) > 0]
        if known:
            mean_cost = sum(known) / len(known)
        unit_cost_norm: dict[str, float] = {
            e: (self._unit_cost[e] / mean_cost if mean_cost > 0 and self._unit_cost[e] > 0 else 0.0)
            for e in sorted_envs
        }

        rewards = self._proxy.compute(lp_used, richness, unit_cost_norm)

        out: dict[str, SignalVector] = {}
        for env_id in sorted_envs:
            posterior = posteriors[env_id]
            status, note = self._classifiers[env_id].step(
                n_groups_eff=posterior.n_groups_eff,
                rounds_seen=posterior.n_rounds_seen,
                p_hat=posterior.pass_rate,
                p_lo=posterior.pass_lo,
                p_hi=posterior.pass_hi,
                z_lp=lp_info[env_id][2],
                sr=richness[env_id],
            )
            lp, se, z = lp_info[env_id]
            result = rewards[env_id]
            out[env_id] = SignalVector(
                env_id=env_id,
                round_t=observation.round,
                n_rounds_seen=posterior.n_rounds_seen,
                n_groups_eff=posterior.n_groups_eff,
                pass_rate=posterior.pass_rate,
                pass_lo=posterior.pass_lo,
                pass_hi=posterior.pass_hi,
                lp=lp,
                lp_se=se,
                lp_z=z,
                lp_raw_fast=posterior.rate_fast,
                lp_raw_slow=posterior.rate_slow,
                richness=richness[env_id],
                mean_score=means[env_id],
                unit_cost_usd=self._unit_cost[env_id],
                unit_cost_norm=unit_cost_norm[env_id],
                proxy_raw=result.proxy_raw,
                proxy_unit=result.proxy_unit,
                proxy_reward=result.proxy_reward,
                status=status,
                status_note=note,
            )
        return out

    def _within_round_variance(self, obs: EnvRoundObs, p_pooled: float) -> float | None:
        """Between-group variance of pass rates inside one round (LP-B `design` SE, E-01).

        Unbiased sample variance of the n per-prompt group rates from the round's
        moments, floored at p(1-p)/G: the variance a group rate has even when every
        prompt is equally hard (pure binomial sampling). One prompt: p(1-p), the
        largest variance a rate in [0, 1] with mean p can have.
        """
        n = obs.n_prompts
        if n <= 0:
            return None
        p = min(max(p_pooled, 1e-6), 1.0 - 1e-6)
        floor = p * (1.0 - p) / max(self._group_size, 1)
        if n == 1:
            return p * (1.0 - p)
        mean = obs.sum_score / n
        var = max(obs.sum_score_sq / n - mean * mean, 0.0) * n / (n - 1)
        return max(var, floor)

    # ------------------------------------------------------------- calibration

    def update_calibration(self, obs: CalibrationObservation) -> None:
        """Book a calibration window (mismatch flags arrive via set_calibration_mismatch)."""

    def set_calibration_mismatch(self, env_id: str, flag: bool) -> None:
        """Forward a proxy-vs-benchmark mismatch flag (wired in Phase 9)."""
        self._classifiers[env_id].set_mismatch(flag)

    def statuses(self, env_id: str) -> EnvStatus:
        return self._classifiers[env_id].status

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> dict[str, object]:
        return {
            "round_t": self._round_t,
            "unit_cost": dict(self._unit_cost),
            "tracker": self._tracker.get_state(),
            "lp": self._lp.get_state(),
            "richness": self._richness.get_state(),
            "proxy": self._proxy.get_state(),
            "classifiers": {e: c.get_state() for e, c in self._classifiers.items()},
        }

    def load_checkpoint(self, state: Mapping[str, object]) -> None:
        self._round_t = int(state["round_t"])  # type: ignore[arg-type]
        self._unit_cost = {e: float(v) for e, v in state["unit_cost"].items()}  # type: ignore[union-attr,attr-defined]
        for env_id, raw in state["classifiers"].items():  # type: ignore[union-attr]
            self._classifiers[env_id].load_checkpoint(raw)  # type: ignore[arg-type]
        self._tracker.load_checkpoint(state["tracker"])  # type: ignore[arg-type]
        self._lp.load_checkpoint(state["lp"])  # type: ignore[arg-type]
        self._richness.load_checkpoint(state["richness"])  # type: ignore[arg-type]
        self._proxy.load_checkpoint(state["proxy"])  # type: ignore[arg-type]
