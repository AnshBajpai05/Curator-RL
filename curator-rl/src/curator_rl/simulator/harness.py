"""Episode harness and oracle-gap metric (Roadmap F.5, v1 Phase 3).

`run_episode` drives one scheduler through one scenario world under the
budget-ledger stop rule (Roadmap H.6/S-13, at round granularity): the round
that crosses the budget is charged and the episode stops right after it, so
the overshoot is at most one round. Reporting evaluations (final benchmark)
are uncharged; scheduler-owned calibration evaluations (optional) are charged.

Seeds: `SeedManager` hands out independent named streams, so the world,
the scheduler and the oracle see independent randomness for the same seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol, cast

from curator_rl.core.types import CalibrationObservation, RoundObservation
from curator_rl.scheduler.base import BaseScheduler
from curator_rl.simulator.scenarios import ScenarioCfg, build_world
from curator_rl.simulator.world import SimWorld


class OracleScheduler(Protocol):
    """Duck type shared by oracles and scheduler adapters: what the harness calls."""

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]: ...

    def update_observation(self, observation: RoundObservation) -> None: ...

    def update_calibration(self, observation: CalibrationObservation) -> None: ...


SchedulerLike = BaseScheduler | OracleScheduler


@dataclass
class EpisodeResult:
    """Outcome of one simulated episode (Roadmap F.5 outputs)."""

    scenario_id: str
    method: str
    seed: int
    rounds: int
    final_score: float
    final_score_se: float
    total_cost_usd: float
    budget_usd: float
    benchmark_evals: int
    # noiseless S_val(theta_T) of the final skills (audit E-10): the objective itself, with no
    # Binomial evaluation noise drawn independently per method; use it as the primary metric
    final_score_expected: float = float("nan")
    # delivered training prompts per arm, and passes over each finite training set (audit E-14)
    prompts_by_env: dict[str, int] = field(default_factory=dict)
    passes_by_env: dict[str, float | None] = field(default_factory=dict)
    round_logs: list[dict] = field(default_factory=list)
    calib_logs: list[dict] = field(default_factory=list)
    report_logs: list[dict] = field(default_factory=list)

    @property
    def within_budget_tolerance(self) -> bool:
        """Overshoot is at most one round (Roadmap H.6)."""
        return self.total_cost_usd <= self.budget_usd * 1.5

    def concentration(self, *, skip_rounds: int = 0) -> dict[str, float]:
        """Allocation concentration over rounds after `skip_rounds` (D-67)."""
        return weight_concentration(self.round_logs, skip_rounds=skip_rounds)


def weight_concentration(round_logs: list[dict], *, skip_rounds: int = 0) -> dict[str, float]:
    """Concentration diagnostics of the per-round mixtures (D-67).

    - `mean_max_weight`: mean over rounds of the largest weight;
    - `norm_entropy`: mean Shannon entropy divided by ln(N) (1 = uniform, 0 = one-hot);
    - `frac_rounds_over_half`: share of rounds where one arm holds more than half.
    """
    rows = [log["weights"] for log in round_logs[skip_rounds:]]
    if not rows:
        return {"mean_max_weight": float("nan"), "norm_entropy": float("nan"),
                "frac_rounds_over_half": float("nan"), "n_rounds": 0}
    max_w, ent, over = [], [], 0
    for w in rows:
        vals = [max(v, 0.0) for v in w.values()]
        n = len(vals)
        max_w.append(max(vals))
        over += max(vals) > 0.5
        h = -sum(v * math.log(v) for v in vals if v > 0.0)
        ent.append(h / math.log(n) if n > 1 else 1.0)
    return {
        "mean_max_weight": sum(max_w) / len(rows),
        "norm_entropy": sum(ent) / len(rows),
        "frac_rounds_over_half": over / len(rows),
        "n_rounds": len(rows),
    }


def apply_scenario_constraints(
    weights: dict[str, float], world: SimWorld, round_t: int
) -> dict[str, float]:
    """Zero out not-yet-available (late-start) envs and renormalise (S-H)."""
    masked = {
        env_id: (0.0 if env.late_start_round > round_t else weights.get(env_id, 0.0))
        for env_id, env in world.envs.items()
    }
    total = sum(masked.values())
    if total <= 0:
        avail = [i for i, e in world.envs.items() if e.late_start_round <= round_t]
        n = max(len(avail), 1)
        return {i: (1.0 / n if i in avail else 0.0) for i in masked}
    return {i: v / total for i, v in masked.items()}


def run_episode(
    scheduler: SchedulerLike,
    scenario: ScenarioCfg,
    seed: int,
    *,
    method: str = "unknown",
) -> EpisodeResult:
    """Run one scheduler on a fresh (scenario, seed) world until the budget
    is exhausted or `max_rounds` is reached (Roadmap F.5)."""
    world = build_world(scenario, seed)

    calibration = scenario.calib
    report_eval_every = scenario.report_eval.interval_rounds
    spend = 0.0
    prev_obs: RoundObservation | None = None
    logs: list[dict] = []
    calib_logs: list[dict] = []
    report_logs: list[dict] = []
    exposure: dict[str, float] = {i: 0.0 for i in world._order}
    window_cost = 0.0

    for round_t in range(scenario.max_rounds):
        weights = scheduler.select_mixture(prev_obs)
        weights = apply_scenario_constraints(weights, world, round_t)
        obs = world.step_round(weights, steps=(round_t + 1) * scenario.steps_per_round)
        spend += obs.round_cost_usd
        window_cost += obs.round_cost_usd
        for env_id, w in weights.items():
            served = (obs.per_env[env_id].n_prompts if world.learning_from == "delivered"
                      else w * world.M)
            exposure[env_id] = exposure.get(env_id, 0.0) + served
        obs = replace(obs, budget_remaining_usd=max(world.budget_usd - spend, 0.0))
        scheduler.update_observation(obs)
        prev_obs = obs
        logs.append(
            {
                "round": obs.round,
                "weights": dict(weights),
                "round_cost_usd": obs.round_cost_usd,
                "spend_usd": spend,
                "per_env": {
                    env_id: {
                        "n_prompts": e.n_prompts,
                        "n_rollouts": e.n_rollouts,
                        "k_success": e.k_success,
                        "cost_usd": e.cost_usd,
                    }
                    for env_id, e in obs.per_env.items()
                },
                # diagnostics (never visible to the scheduler, Roadmap F.1)
                "true_skills": dict(world.skills),
                "true_pass_rates": world.true_pass_rates(),
            }
        )
        if (
            calibration.enabled
            and getattr(scheduler, "uses_calibration", False)
            and calibration.interval_rounds > 0
            and (round_t + 1) % calibration.interval_rounds == 0
        ):
            request_fn = getattr(scheduler, "calibration_request", None)
            request = cast("dict[str, int] | None", request_fn() if callable(request_fn) else None)
            default_items = calibration.items_per_slice or scenario.world.benchmark_items
            if request:
                eval_cost = calibration.eval_cost_items(sum(request.values()))
            else:
                eval_cost = calibration.eval_cost(len(world._order), scenario.world.benchmark_items)
            calib_obs: CalibrationObservation = world.calibration_observation(
                window_k=len(calib_logs) + 1,
                exposure_by_env=dict(exposure),
                window_cost_usd=window_cost,
                eval_cost_usd=eval_cost,
                paired=calibration.paired,
                churn=calibration.churn,
                n_items=default_items,
                request=request,
            )
            spend += eval_cost
            scheduler.update_calibration(calib_obs)
            calib_logs.append(
                {
                    "window_k": calib_obs.window_k,
                    "round": calib_obs.round,
                    "score_total": calib_obs.score_total,
                    "se_total": calib_obs.se_total,
                    "eval_cost_usd": calib_obs.eval_cost_usd,
                    "evaluated": sorted(calib_obs.score_by_domain),
                    "exposure_by_env": dict(calib_obs.exposure_by_env),
                    # per-slice detail for the ROI leaderboard (roi/leaderboard.py); same names as a GPU run's
                    "scores": dict(calib_obs.score_by_domain), "se": dict(calib_obs.se_by_domain),
                    "delta_se": dict(calib_obs.delta_se_by_domain or {}),
                    "exposure": dict(calib_obs.exposure_by_env),
                }
            )
            exposure = {i: 0.0 for i in world._order}
            window_cost = 0.0
        if (
            report_eval_every
            and (round_t + 1) % report_eval_every == 0
        ):
            # uncharged reporting evaluation, identical protocol for every
            # method (Roadmap L.4) — gives the score-vs-cost curve
            score_r, se_r = world.evaluate_report()
            report_logs.append(
                {
                    "round": round_t + 1,
                    "spend_usd": spend,
                    "score": score_r,
                    "se": se_r,
                }
            )
        if spend >= world.budget_usd:
            break

    final_score, final_se = world.evaluate_report()                       # uncharged
    final_expected = world.expected_benchmark_score()                     # noiseless, no RNG
    return EpisodeResult(
        scenario_id=scenario.scenario_id,
        method=method,
        seed=seed,
        rounds=world.rounds_done,
        final_score=final_score,
        final_score_se=final_se,
        total_cost_usd=spend,
        budget_usd=world.budget_usd,
        benchmark_evals=world.benchmark_evals,
        final_score_expected=final_expected,
        prompts_by_env=dict(world.prompts_seen),
        passes_by_env=world.passes(),
        round_logs=logs,
        calib_logs=calib_logs,
        report_logs=report_logs,
    )


def oracle_gap_closure(
    score_method: float, score_uniform: float, score_oracle: float
) -> float:
    """(S_method - S_uniform) / (S_oracle - S_uniform) at equal budget (F.4/F.5)."""
    denom = score_oracle - score_uniform
    if abs(denom) < 1e-12:
        raise ValueError("oracle and uniform scores coincide; scenario cannot discriminate")
    return (score_method - score_uniform) / denom


def mean_or(values: list[float], default: float = float("nan")) -> float:
    return sum(values) / len(values) if values else default


def scores_summary(scores: list[float]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    return {"mean": mean, "std": math.sqrt(var), "n": n}
