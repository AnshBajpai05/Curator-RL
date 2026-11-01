"""Discounted-UCB scheduler backend (Roadmap E.6, v1 Phase 5).

State per environment: discounted pull mass `Ñ_i` and discounted reward mass
`Σ̃_i`, updated from the round's realised exposure (u = weights_used) and the
SignalEngine's proxy reward r̄. Scores = μ̂ + κ·sqrt(ln(max(Ñ(t), e))/Ñ_i),
then the E.6 mixture map (`mixture.scores_to_weights`) and the E.5 status
constraints (`mixture.apply_status_constraints`).

The scheduler owns its own `SignalEngine` (same L1 code Phase 4 validated) —
signals are computed from the RoundObservations the trainer/harness delivers,
never from hidden state. Deterministic; checkpointable (Part M item 13).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any, NotRequired, TypedDict

from curator_rl.calibration.calibrator import CalibrationReport, Calibrator, CalibratorState
from curator_rl.calibration.cre import CalibratedRewardEngine
from curator_rl.calibration.regression import FitRow, ProxyFit, ProxyWeightFitter
from curator_rl.core.config import CalibCfg, CreCfg, ProxyCfg, SchedulerCfg, SignalsCfg
from curator_rl.core.types import (
    CalibrationObservation,
    EnvStatus,
    MixtureDecision,
    RoundObservation,
    SignalVector,
)
from curator_rl.scheduler.base import BaseScheduler, validate_weights
from curator_rl.scheduler.mixture import apply_status_constraints, cap_weights, scores_to_weights
from curator_rl.signals.drift import PageHinkley, standardised_surprise
from curator_rl.signals.engine import EngineState, SignalEngine


class DUCBState(TypedDict):
    round: int
    steps: int
    n_hat: dict[str, float]
    s_hat: dict[str, float]
    last_weights: dict[str, float]
    engine: EngineState
    # keys below are absent in checkpoints made before the feature that added them
    win_x: NotRequired[dict[str, float]]
    win_u: NotRequired[dict[str, float]]
    calibrator: NotRequired[CalibratorState | None]
    win_usd: NotRequired[dict[str, float]]
    budget_total: NotRequired[float | None]
    cre: NotRequired[Any]
    fit_lp: NotRequired[dict[str, float]]
    fit_sr: NotRequired[dict[str, float]]
    fit_w: NotRequired[dict[str, float]]
    drift: NotRequired[dict[str, dict[str, float]] | None]
    prev_pass: NotRequired[dict[str, float]]
    latest_vectors: NotRequired[dict[str, dict[str, Any]]]


class DiscountedUCB(BaseScheduler):
    """D-UCB scoring + E.6 mixture map + E.5 status constraints (config-gated)."""

    def __init__(
        self,
        env_ids,
        signals_cfg: SignalsCfg,
        proxy_cfg: ProxyCfg,
        scheduler_cfg: SchedulerCfg,
        calib_cfg: CalibCfg,
        group_size: int,
        *,
        prompts_per_round: int | None = None,
        cre_cfg: CreCfg | None = None,
        domain_of: dict[str, str | None] | None = None,
    ) -> None:
        super().__init__(env_ids)
        self.cfg = scheduler_cfg
        self._calib_cfg = calib_cfg
        self._engine = SignalEngine(
            signals_cfg, proxy_cfg, calib_cfg,
            scheduler_cfg.cost_exponent, group_size, env_ids,
        )
        self._n_hat = {e: 0.0 for e in self.env_ids}
        self._s_hat = {e: 0.0 for e in self.env_ids}
        self._round = 0
        self._steps = 0
        self._last_weights = {e: 1.0 / len(self.env_ids) for e in self.env_ids}
        self._prompts_per_round = prompts_per_round
        self.last_decision: MixtureDecision | None = None
        self._latest_vectors: dict[str, SignalVector] = {}
        # calibration (Roadmap Part I, D-73): proxy signal accumulated per window
        self.uses_calibration = bool(calib_cfg.enabled)
        # domain_of maps an arm to the calibration slice it is scored on (None: no slice);
        # e.g. {"noisy": "gsm8k"} on real hardware, where the noisy arm has no slice of its own
        self._calibrator = (
            Calibrator(self.env_ids, calib_cfg, domain_of) if calib_cfg.enabled else None
        )
        self._win_x = {e: 0.0 for e in self.env_ids}
        self._win_u = {e: 0.0 for e in self.env_ids}
        self.last_calibration: CalibrationReport | None = None
        # Calibrated Reward Engine (D-76): only meaningful with calibration on
        self._cre = (
            CalibratedRewardEngine(self.env_ids, cre_cfg)
            if (cre_cfg is not None and cre_cfg.enabled and calib_cfg.enabled)
            else None
        )
        # alpha/beta fit (improvement B4): run-long exposure-weighted means of LP' and SR per arm
        self._fitter = (
            ProxyWeightFitter(proxy_cfg.alpha, proxy_cfg.beta, ridge=proxy_cfg.fit.ridge,
                              trust_region=calib_cfg.trust_region, min_rows=proxy_cfg.fit.min_rows)
            if (proxy_cfg.fit.enabled and calib_cfg.enabled) else None
        )
        self._fit_lp = {e: 0.0 for e in self.env_ids}
        self._fit_sr = {e: 0.0 for e in self.env_ids}
        self._fit_w = {e: 0.0 for e in self.env_ids}
        self.last_proxy_fit: ProxyFit | None = None
        self._win_usd = {e: 0.0 for e in self.env_ids}   # dollars charged since last calibration
        # drift-triggered discounting (improvement B2): per-arm change detector on pass-rate surprises
        drift = scheduler_cfg.drift
        self._drift = ({e: PageHinkley(drift.delta, drift.threshold) for e in self.env_ids}
                       if drift.enabled else None)
        self._group_size = int(group_size)
        self._prev_pass: dict[str, float] = {}
        self.drift_alarms: list[tuple[int, str]] = []
        self._budget_total: float | None = None
        self.cost_override: dict[str, float] | None = None  # privileged ablation (exact cost)
        self.reward_log: list[dict[str, object]] = []        # per round: reward and its source

    # ---------------------------------------------------------------- mixture

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        self._round += 1
        if self._round <= self.cfg.warmup_rounds or observation is None:
            weights = {e: 1.0 / len(self.env_ids) for e in self.env_ids}
            self._last_weights = dict(weights)
            self.last_decision = None
            return weights

        scores = self.compute_scores()
        weights = scores_to_weights(
            scores, self.cfg.tau, self.cfg.epsilon, list(self.env_ids),
            score_norm=self.cfg.score_norm,
        )
        if self.cfg.smoothing < 1.0:          # improvement B7: w = (1 - s) w_prev + s w_new
            s = self.cfg.smoothing
            weights = {e: (1.0 - s) * self._last_weights[e] + s * weights[e] for e in self.env_ids}
        vectors = self._latest_vectors
        if vectors and self.cfg.status_control != "off":
            weights, intents = apply_status_constraints(
                weights, vectors, self.cfg, n_min=self._n_min()
            )
        else:
            intents = {e: "hold" for e in self.env_ids}
        if self.cfg.max_weight < 1.0:          # improvement B7: no single arm above the cap
            weights = cap_weights(weights, self.cfg.max_weight,
                                  frozen={e for e, i in intents.items() if i == "shrink"})
        validate_weights(weights, self.env_ids)
        self._last_weights = dict(weights)
        self.last_decision = self._decision(weights, scores, vectors or {}, intents)
        return weights

    def _n_min(self) -> int:
        return self._engine.n_min

    def compute_scores(self) -> dict[str, float]:
        """μ̂ + bonus per environment (Roadmap E.6); exposed for debugging.

        The reward is already folded into Σ̃ (and thus μ̂) by
        `update_observation`; scores need only the current counters.
        """
        total_n = sum(self._n_hat.values())
        log_arg = math.log(max(total_n, math.e))
        scores: dict[str, float] = {}
        for e in self.env_ids:
            n_i = max(self._n_hat[e], 1e-9)
            mu_hat = self._s_hat[e] / n_i
            bonus = self.cfg.exploration_coef * math.sqrt(log_arg / n_i)
            scores[e] = mu_hat + bonus
        return scores

    def _decision(self, weights, scores, vectors, intents) -> MixtureDecision:
        envs = list(self.env_ids)
        total_n = sum(self._n_hat.values())
        log_arg = math.log(max(total_n, math.e))
        mu_hat, bonus = {}, {}
        for e in envs:
            n_i = max(self._n_hat[e], 1e-9)
            mu_hat[e] = self._s_hat[e] / n_i
            bonus[e] = self.cfg.exploration_coef * math.sqrt(log_arg / n_i)
        quotas = (
            _quotas(weights, self._prompts_per_round) if self._prompts_per_round else {}
        )
        return MixtureDecision(
            round=self._round,
            steps=self._steps,
            weights=dict(weights),
            ucb_scores={e: mu_hat[e] + bonus[e] for e in envs},
            mu_hat=mu_hat,
            bonus=bonus,
            statuses={
                e: getattr(vectors[e], "status_note", "warmup") if vectors else "warmup"
                for e in envs
            },
            intents=intents,
            quotas=quotas,
            rng_hash=self.state_hash(),
        )

    # ----------------------------------------------------------------- updates

    def update_observation(self, observation: RoundObservation) -> None:
        vectors = self._engine.update(observation)
        if self._drift is not None:
            for env_id, env_obs in observation.per_env.items():
                if env_id not in self._prev_pass:
                    continue
                x = standardised_surprise(env_obs.k_success, env_obs.n_rollouts, env_obs.n_prompts,
                                          self._group_size, self._prev_pass[env_id],
                                          float(vectors[env_id].reward_dispersion))
                if self._drift[env_id].update(x):
                    # a detected change: this arm's evidence is stale, the others' is not
                    self._n_hat[env_id] *= self.cfg.drift.decay
                    self._s_hat[env_id] *= self.cfg.drift.decay
                    self.drift_alarms.append((observation.round, env_id))
            # checkpointed (not the vectors): a resumed run computes exactly the same surprises
            self._prev_pass = {e: float(v.pass_rate) for e, v in vectors.items()}
        self._latest_vectors = vectors
        self._steps = observation.steps
        if self._budget_total is None:
            self._budget_total = observation.budget_remaining_usd + observation.round_cost_usd
        round_log: dict[str, object] = {"round": observation.round}
        total_prompts = sum(max(o.n_prompts, 0) for o in observation.per_env.values())
        delivered = self.cfg.pull_unit == "delivered"
        for env_id, env_obs in observation.per_env.items():
            u = observation.weights_used.get(env_id, 0.0)
            if self.cfg.pull_unit == "prompt":
                u = u * max(env_obs.n_prompts, 0)
            elif delivered:  # what was actually trained this round (audit E-09)
                u = max(env_obs.n_prompts, 0) / total_prompts if total_prompts > 0 else 0.0
            self._win_usd[env_id] += float(env_obs.cost_usd)
            r_bar = float(vectors[env_id].proxy_reward)
            source = "proxy"
            if self._cre is not None and self._cre.active:
                est = self._cre.estimate(
                    env_id, float(vectors[env_id].proxy_raw), self._unit_cost(vectors[env_id]),
                    self._budget_total or 0.0,
                )
                r_bar, source = est.reward, f"cre:{est.source}"
                round_log[env_id] = {"r": r_bar, "src": source, "b": est.mean_per_usd,
                                     "sd": est.sd_per_usd, "n": est.n_obs}
            else:
                round_log[env_id] = {"r": r_bar, "src": source}
            self._n_hat[env_id] = self.cfg.gamma * self._n_hat[env_id] + u
            self._s_hat[env_id] = self.cfg.gamma * self._s_hat[env_id] + u * r_bar
            share = u if delivered else observation.weights_used.get(env_id, 0.0)
            if share > 0.0:
                self._win_x[env_id] += share * float(vectors[env_id].proxy_raw)
                self._win_u[env_id] += share
                self._fit_lp[env_id] += share * float(vectors[env_id].lp_scaled)
                self._fit_sr[env_id] += share * float(vectors[env_id].richness)
                self._fit_w[env_id] += share
        self.reward_log.append(round_log)

    def _unit_cost(self, sv) -> float:
        if self.cost_override is not None and sv.env_id in self.cost_override:
            return float(self.cost_override[sv.env_id])
        return float(sv.unit_cost_usd)

    def calibration_request(self) -> dict[str, int] | None:
        """Slices (and items per slice) the next calibration should evaluate (D-75)."""
        if self._calibrator is None:
            return None
        return self._calibrator.select_targets(self._win_u, self._win_x)

    def preflag_unreliable(self, env_ids) -> None:
        """Arms whose reward paid for junk before training (pre-flight verifier check, D-102): S5 throughout."""
        self._engine.preflag_unreliable(env_ids)

    def update_calibration(self, observation: CalibrationObservation) -> None:
        """One calibration window: credit, proxy-vs-benchmark test, S5 flags (D-73)."""
        self._engine.update_calibration(observation)
        if self._calibrator is None:
            return
        proxy_x = {
            e: (self._win_x[e] / self._win_u[e] if self._win_u[e] > 0 else 0.0)
            for e in self.env_ids
        }
        report = self._calibrator.observe(observation, proxy_x, self._win_usd)
        self.last_calibration = report
        if self._cre is not None:
            costs = {e: (self.cost_override or {}).get(e, self._engine.unit_cost(e)) for e in self.env_ids}
            self._cre.ingest(report.span_gain, proxy_x, costs, ready=report.ready,
                             window_k=report.window_k)
        if report.ready:
            # E-02: with targeted calibration an arm that was not re-evaluated keeps its old
            # estimate; re-sending its old flag would count one evaluation as several
            updated = (self.env_ids if self._calib_cfg.flag_scope == "all"
                       else [e for e in self.env_ids if e in report.evaluated])
            for env_id in updated:
                self._engine.set_calibration_mismatch(env_id, report.mismatch.get(env_id, False))
            if self._fitter is not None:
                self._fit_proxy(report)
        self._win_x = {e: 0.0 for e in self.env_ids}
        self._win_u = {e: 0.0 for e in self.env_ids}
        self._win_usd = {e: 0.0 for e in self.env_ids}

    def _fit_proxy(self, report: CalibrationReport) -> None:
        rows = [FitRow(e, report.gain_hat[e], report.gain_se[e], self._fit_lp[e] / self._fit_w[e],
                       self._fit_sr[e] / self._fit_w[e])
                for e in self.env_ids
                if self._fit_w[e] > 0 and e in report.gain_hat and report.gain_se.get(e, float("inf")) < float("inf")]
        fit = self._fitter.fit(rows, self._engine.proxy_weights)  # type: ignore[union-attr]
        if fit.accepted:
            self._engine.set_proxy_weights(fit.alpha, fit.beta)
        self.last_proxy_fit = fit
        self.reward_log.append({"round": self._round, "proxy_fit": fit.as_log()})

    @property
    def proxy_weights(self) -> tuple[float, float]:
        return self._engine.proxy_weights

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> DUCBState:
        return {
            "round": self._round,
            "steps": self._steps,
            "n_hat": dict(self._n_hat),
            "s_hat": dict(self._s_hat),
            "last_weights": dict(self._last_weights),
            "engine": self._engine.get_state(),
            "win_x": dict(self._win_x),
            "win_u": dict(self._win_u),
            "calibrator": None if self._calibrator is None else self._calibrator.get_state(),
            "win_usd": dict(self._win_usd),
            "budget_total": self._budget_total,
            "cre": None if self._cre is None else self._cre.get_state(),
            "fit_lp": dict(self._fit_lp), "fit_sr": dict(self._fit_sr), "fit_w": dict(self._fit_w),
            "drift": None if self._drift is None else {e: d.get_state() for e, d in self._drift.items()},
            "prev_pass": dict(self._prev_pass),
            # the status layer reads these at the next decision; without them a restored scheduler
            # skips its status constraints for one round (replay test: test_scheduler_checkpoint.py)
            "latest_vectors": {e: {**asdict(sv), "status": sv.status.value}
                               for e, sv in self._latest_vectors.items()},
        }

    def load_checkpoint(self, state: DUCBState) -> None:  # type: ignore[override]  (narrower than Mapping)
        self._round = int(state["round"])
        self._steps = int(state["steps"])
        self._n_hat = {e: float(v) for e, v in state["n_hat"].items()}
        self._s_hat = {e: float(v) for e, v in state["s_hat"].items()}
        self._last_weights = {e: float(v) for e, v in state["last_weights"].items()}
        self._engine.load_checkpoint(state["engine"])
        self._win_x = {e: float(v) for e, v in state.get("win_x", {}).items()} or {e: 0.0 for e in self.env_ids}
        self._win_u = {e: float(v) for e, v in state.get("win_u", {}).items()} or {e: 0.0 for e in self.env_ids}
        calibrator_state = state.get("calibrator")
        if self._calibrator is not None and calibrator_state is not None:
            self._calibrator.load_checkpoint(calibrator_state)
        self._win_usd = {e: float(v) for e, v in state.get("win_usd", {}).items()} or {
            e: 0.0 for e in self.env_ids}
        self._budget_total = state.get("budget_total")
        cre_state, drift_state = state.get("cre"), state.get("drift")
        if self._cre is not None and cre_state is not None:
            self._cre.load_checkpoint(cre_state)
        if self._drift is not None and drift_state is not None:
            for e, d in drift_state.items():
                self._drift[e].load_checkpoint(d)
            self._prev_pass = {e: float(v) for e, v in state.get("prev_pass", {}).items()}
        if "fit_w" in state:                               # absent in checkpoints made before B4
            self._fit_lp = {e: float(v) for e, v in state.get("fit_lp", {}).items()}
            self._fit_sr = {e: float(v) for e, v in state.get("fit_sr", {}).items()}
            self._fit_w = {e: float(v) for e, v in state.get("fit_w", {}).items()}
        self._latest_vectors = {
            e: SignalVector(**{**v, "status": EnvStatus(v["status"])})
            for e, v in state.get("latest_vectors", {}).items()
        }

    def state_hash(self) -> str:
        payload = json.dumps(
            {"n": self._n_hat, "s": self._s_hat, "r": self._round}, sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def _quotas(weights: dict[str, float], m: int) -> dict[str, int]:
    from curator_rl.core.quotas import largest_remainder_quotas

    return largest_remainder_quotas(weights, m)
