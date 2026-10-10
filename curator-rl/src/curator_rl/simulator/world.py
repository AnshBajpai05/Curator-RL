"""Synthetic world with known dynamics (Roadmap F.1/F.2, v1 Phase 3).

Each environment has a latent skill s_i; the true pass probability is
p_i = sigmoid(a_i * (s_i - d_i)), learnability g_i = 4 p_i (1 - p_i) peaks at
p = 0.5 like the GRPO signal, and skills move by

    s_j(t+1) = s_j(t) + R * eta_j * sum_i T_ji * w_i * g_i + xi

with T_ii = 1 and T_ji < 0 modelling interference. Costs are drawn per prompt
with log-normal noise. The world exposes ONLY measured quantities through
`RoundObservation`; true skills and true pass rates stay hidden from the
scheduler (F.1) and exist solely for oracles, diagnostics and tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from curator_rl.core.advantage import group_mean_abs_advantage
from curator_rl.core.quotas import CarryQuotas
from curator_rl.core.types import CalibrationObservation, EnvRoundObs, RoundObservation
from curator_rl.simulator.quotas import largest_remainder_quotas

_EPS = 1e-6


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def learnability(p: float) -> float:
    """g = 4 p (1 - p): 1 at p = 0.5, 0 at p in {0, 1} (Roadmap F.2)."""
    return 4.0 * p * (1.0 - p)


@dataclass
class SimEnvParams:
    """One simulated environment (Roadmap F.2 environment types).

    `transfer_out` maps OTHER env ids to T_{other, self}: the effect training
    on this env has on their skill (T_ii = 1 is implicit). `noisy_q` turns the
    env into the noisy type (reported success ~ Bernoulli(q) independent of
    skill, eta forced to 0, no transfer in or out). `gate` delays learning
    until a prerequisite env's skill passes a threshold; `drift` shifts the
    difficulty abruptly at a round; `late_start_round` keeps the env
    unavailable (weight forced to 0 and renormalised by the harness) before
    that round; `cost_skill_slope` makes unit cost rise with skill.
    """

    env_id: str
    skill0: float = 0.0
    slope: float = 2.0
    difficulty: float = 0.0
    eta: float = 0.01
    cost_usd_per_prompt: float = 0.001
    transfer_out: dict[str, float] = field(default_factory=dict)
    noisy_q: float | None = None
    gate_prereq: str | None = None
    gate_theta: float = 0.0
    gate_slope: float = 4.0
    drift_round: int | None = None
    drift_shift: float = 0.0
    late_start_round: int = 0
    cost_skill_slope: float = 0.0
    # benchmark slice of this env's domain (defaults mirror the training env)
    bench_slope: float | None = None
    bench_difficulty: float | None = None
    bench_weight: float | None = None  # pi_d; None -> uniform over domains
    # calibration slice this arm is scored on when it has none of its own (audit E-03):
    # e.g. the real noisy arm uses GSM8K prompts and is scored on the GSM8K slice
    bench_domain: str | None = None

    def __post_init__(self) -> None:
        if self.noisy_q is not None and not (0.0 < self.noisy_q < 1.0):
            raise ValueError(f"{self.env_id}: noisy_q must be in (0, 1)")
        if self.cost_usd_per_prompt <= 0:
            raise ValueError(f"{self.env_id}: cost_usd_per_prompt must be > 0")
        self.slope = float(self.slope)
        self.bench_slope = self.slope if self.bench_slope is None else float(self.bench_slope)
        self.bench_difficulty = (
            self.difficulty if self.bench_difficulty is None else float(self.bench_difficulty)
        )


class SimWorld:
    """The synthetic world: skills, rollouts, benchmark, costs (Roadmap F.2)."""

    def __init__(
        self,
        envs: list[SimEnvParams],
        *,
        steps_per_round: int,
        prompts_per_round: int,
        group_size: int,
        budget_usd: float,
        skill_noise_std: float = 0.003,
        cost_lognormal_sigma: float = 0.10,
        rollout_concentration: float = 25.0,
        usd_per_gpu_hour: float = 1.0,
        rng: np.random.Generator | None = None,
        calib_rng: np.random.Generator | None = None,
        learning_from: str = "requested",
        rounding: str = "largest_remainder",
    ) -> None:
        if learning_from not in ("requested", "delivered"):
            raise ValueError(f"learning_from must be requested|delivered, got {learning_from!r}")
        if rounding not in ("largest_remainder", "carry"):
            raise ValueError(f"rounding must be largest_remainder|carry, got {rounding!r}")
        # audit E-09: `requested` (legacy) trains on the requested weights while rollouts and
        # costs follow the integer quotas; `delivered` trains on the quotas actually served
        self.learning_from = learning_from
        self._carry = CarryQuotas() if rounding == "carry" else None
        if not envs:
            raise ValueError("SimWorld needs at least one environment")
        ids = [e.env_id for e in envs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate env ids: {ids}")
        self.envs: dict[str, SimEnvParams] = {e.env_id: e for e in envs}
        self._order: tuple[str, ...] = tuple(ids)
        self.R = int(steps_per_round)
        self.M = int(prompts_per_round)
        self.G = int(group_size)
        self.budget_usd = float(budget_usd)
        self.skill_noise_std = float(skill_noise_std)
        self.cost_sigma = float(cost_lognormal_sigma)
        self.kappa_b = float(rollout_concentration)
        self.usd_per_gpu_hour = float(usd_per_gpu_hour)
        self.rng = rng if rng is not None else np.random.default_rng()
        # separate stream so paired calibration never perturbs the training draws
        self.calib_rng = calib_rng if calib_rng is not None else np.random.default_rng(0)
        self._calib_items: dict[str, np.ndarray] = {}
        self._calib_prev: dict[str, np.ndarray] = {}

        self.skills: dict[str, float] = {e.env_id: float(e.skill0) for e in envs}
        # noisy envs never learn and never transfer (Roadmap F.2 noisy type)
        self.noisy = {e.env_id for e in envs if e.noisy_q is not None}
        for env_id in self.noisy:
            self.envs[env_id].eta = 0.0
        # reverse transfer map: T[target][source] = T_{target, source}; T_ii = 1 implicit
        self.transfer_in: dict[str, dict[str, float]] = {i: {} for i in self._order}
        for e in envs:
            if e.env_id in self.noisy:
                continue
            for target, coeff in e.transfer_out.items():
                if target in self.noisy:
                    raise ValueError(f"noisy env '{target}' cannot receive transfer (F.2)")
                self.transfer_in[target][e.env_id] = float(coeff)
        self.rounds_done = 0
        self.benchmark_evals = 0
        self.bench_weight: dict[str, float] = self._resolve_bench_weights()

    # ------------------------------------------------------------------ world

    def _resolve_bench_weights(self) -> dict[str, float]:
        raw = {i: self.envs[i].bench_weight for i in self._order}
        if all(w is None for w in raw.values()):
            n = len(self._order)
            return {i: 1.0 / n for i in self._order}
        missing = [i for i, w in raw.items() if w is None]
        if missing:
            raise ValueError(f"benchmark weights must be all set or all None; unset: {missing}")
        total = sum(float(w) for w in raw.values())  # type: ignore[arg-type]
        if total <= 0:
            raise ValueError("benchmark weights must sum to a positive value")
        return {i: float(w) / total for i, w in raw.items()}  # type: ignore[arg-type]

    def true_pass(self, env_id: str) -> float:
        """p_i = sigmoid(a_i (s_i - d_i)) — hidden from the scheduler (F.1)."""
        e = self.envs[env_id]
        return sigmoid(e.slope * (self.skills[env_id] - e.difficulty))

    def true_pass_rates(self) -> dict[str, float]:
        return {i: self.true_pass(i) for i in self._order}

    def gate_factor(self, env_id: str) -> float:
        """Delayed type: eta gated by sigmoid(slope * (s_prereq - theta))."""
        e = self.envs[env_id]
        if e.gate_prereq is None:
            return 1.0
        s = self.skills[e.gate_prereq]
        return sigmoid(e.gate_slope * (s - e.gate_theta))

    def unit_cost(self, env_id: str) -> float:
        """Mean unit cost now (cost-drifting type rises with skill)."""
        e = self.envs[env_id]
        factor = 1.0 + e.cost_skill_slope * (self.skills[env_id] - e.skill0)
        return e.cost_usd_per_prompt * max(factor, 1e-3)

    def expected_benchmark_score(self, skills: dict[str, float] | None = None) -> float:
        """Noiseless benchmark value (used by the oracles, Roadmap F.4)."""
        s = self.skills if skills is None else skills
        total = 0.0
        for env_id in self._order:
            e = self.envs[env_id]
            p = sigmoid(e.bench_slope * (s[env_id] - e.bench_difficulty))
            total += self.bench_weight[env_id] * p
        return total

    # ----------------------------------------------------------------- rounds

    def step_round(self, weights: dict[str, float], *, steps: int | None = None) -> RoundObservation:
        """Advance one round under mixture `weights` and return the observation."""
        validate = sum(weights.values())
        if abs(validate - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1 (got {validate})")

        round_t = self.rounds_done
        if self._carry is not None:
            quotas = self._carry.next(weights, self.M)
        else:
            quotas = largest_remainder_quotas(weights, self.M, tie_break_seed=round_t)
        g = {i: learnability(self.true_pass(i)) for i in self._order if i not in self.noisy}
        if self.learning_from == "delivered":
            trained = {i: quotas.get(i, 0) / self.M for i in self._order}
        else:
            trained = weights

        # skill update for every env j: R * eta_j_eff * sum_i T_ji w_i g_i + xi
        # (T_jj = 1; each source i contributes its own allocation w_i * g_i)
        for j in self._order:
            if j in self.noisy:
                continue  # noisy envs never change skill (Roadmap F.2)
            e = self.envs[j]
            inflow = trained.get(j, 0.0) * g.get(j, 0.0)  # T_jj = 1
            for i, t_ji in self.transfer_in[j].items():
                if i in self.noisy:
                    continue
                inflow += t_ji * trained.get(i, 0.0) * g.get(i, 0.0)
            delta = self.R * e.eta * self.gate_factor(j) * inflow
            xi = self.rng.normal(0.0, self.skill_noise_std) if self.skill_noise_std > 0 else 0.0
            self.skills[j] += delta + xi

        self.rounds_done += 1
        # drift applies from the round AFTER the shift was scheduled
        for e in self.envs.values():
            if e.drift_round is not None and self.rounds_done >= e.drift_round:
                e.difficulty += e.drift_shift
                e.drift_round = None  # one-shot

        per_env: dict[str, EnvRoundObs] = {}
        round_cost = 0.0
        for env_id in self._order:
            m_i = quotas.get(env_id, 0)
            obs = self._rollout_env(env_id, m_i)
            per_env[env_id] = obs
            round_cost += obs.cost_usd

        steps_now = self.R * (round_t + 1) if steps is None else steps
        return RoundObservation(
            round=round_t + 1,
            steps=steps_now,
            per_env=per_env,
            weights_used=dict(weights),
            round_cost_usd=round_cost,
            overhead_usd=0.0,
            budget_remaining_usd=max(self.budget_usd - round_cost, 0.0),  # caller refines
        )

    def _rollout_env(self, env_id: str, m_i: int) -> EnvRoundObs:
        """Draw m_i prompts' rollout outcomes for one env (Roadmap F.2)."""
        e = self.envs[env_id]
        if m_i <= 0:
            return EnvRoundObs(
                env_id=env_id, n_prompts=0, n_rollouts=0, k_success=0, n_groups_mixed=0,
                sum_score=0.0, sum_score_sq=0.0, prompt_tokens=0, completion_tokens=0,
                verifier_seconds=0.0, gpu_seconds=0.0, cost_usd=0.0,
            )
        k_success = 0
        n_groups_mixed = 0
        sum_score = 0.0
        sum_score_sq = 0.0
        sum_abs_adv = 0.0
        if env_id in self.noisy:
            q = float(e.noisy_q or 0.0)
            ks = self.rng.binomial(self.G, q, size=m_i)  # Bernoulli(q) per rollout
        else:
            p = self.true_pass(env_id)
            p = min(max(p, _EPS), 1.0 - _EPS)
            a = self.kappa_b * p
            b = self.kappa_b * (1.0 - p)
            qs = self.rng.beta(a, b, size=m_i)  # per-prompt pass probability
            ks = np.asarray(
                [self.rng.binomial(self.G, float(q)) for q in qs], dtype=np.int64
            )
        for k in ks:
            rate = k / self.G
            k_success += int(k)
            n_groups_mixed += int(0 < k < self.G)
            sum_score += rate
            sum_score_sq += rate * rate
            sum_abs_adv += group_mean_abs_advantage(int(k), self.G)
        n_rollouts = m_i * self.G
        # costs: log-normal noise around the current mean unit cost
        mean_cost = self.unit_cost(env_id)
        mu = math.log(mean_cost) - 0.5 * self.cost_sigma**2
        costs = self.rng.lognormal(mu, self.cost_sigma, size=m_i) if self.cost_sigma > 0 else np.full(m_i, mean_cost)
        env_cost = float(costs.sum())
        verifier_seconds = 1e-4 * n_rollouts  # synthetic; sim has no real verifiers
        gpu_seconds = env_cost * 3600.0 / self.usd_per_gpu_hour
        return EnvRoundObs(
            env_id=env_id,
            n_prompts=m_i,
            n_rollouts=n_rollouts,
            k_success=k_success,
            n_groups_mixed=n_groups_mixed,
            sum_score=sum_score,
            sum_score_sq=sum_score_sq,
            prompt_tokens=m_i * 256,        # synthetic token counts; real values
            completion_tokens=n_rollouts * 128,  # come from the trainer in Phase 7
            verifier_seconds=verifier_seconds,
            gpu_seconds=gpu_seconds,
            cost_usd=env_cost,
            sum_abs_adv=sum_abs_adv,
        )

    # ------------------------------------------------------------- benchmark

    _n_b_default: int = 200

    def set_benchmark_items(self, n_items: int) -> None:
        self._n_b_default = int(n_items)

    def evaluate_benchmark(self, *, observed: bool = True) -> tuple[float, float, dict[str, float], dict[str, float]]:
        """Evaluate the benchmark: returns (S, SE, S_by_domain, SE_by_domain).

        With `observed=False` the noiseless expected value is returned with
        SE 0 (oracle use only, Roadmap F.4). Observed slices use Binomial
        noise over n_b items per domain (Roadmap F.2).
        """
        n_b = self._n_b_default
        s_by_domain: dict[str, float] = {}
        se_by_domain: dict[str, float] = {}
        total = 0.0
        var_total = 0.0
        for env_id in self._order:
            e = self.envs[env_id]
            p = sigmoid(e.bench_slope * (self.skills[env_id] - e.bench_difficulty))
            pi_d = self.bench_weight[env_id]
            if observed:
                s_d = float(self.rng.binomial(n_b, p)) / n_b
                se_d = math.sqrt(max(p * (1.0 - p), _EPS) / n_b)
            else:
                s_d, se_d = p, 0.0
            s_by_domain[env_id] = s_d
            se_by_domain[env_id] = se_d
            total += pi_d * s_d
            var_total += (pi_d * se_d) ** 2
        self.benchmark_evals += 1
        return total, math.sqrt(var_total), s_by_domain, se_by_domain

    def evaluate_benchmark_paired(
        self, churn: float, n_items: int | None = None, request: dict[str, int] | None = None
    ) -> tuple[float, float, dict[str, float], dict[str, float], dict[str, float] | None]:
        """Paired evaluation on fixed items (Roadmap v3 4.3, D-72).

        Each slice holds n_b items with a fixed latent u ~ U(0,1); an item is
        solved iff u < p(skill). A fraction `churn` of latents is redrawn per
        evaluation. Returns (S, SE, S_d, SE_d, SE of the change in S_d since the
        previous paired evaluation, or None the first time). The change SE is the
        paired (McNemar) one: sqrt(n01 + n10 - (n01 - n10)^2 / n) / n, floored at
        one discordant item.
        """
        default_n = int(n_items or self._n_b_default)
        targets = [e for e in self._order if request is None or e in request]
        s_by: dict[str, float] = {}
        se_by: dict[str, float] = {}
        dse_by: dict[str, float] = {}
        total, var_total, wsum = 0.0, 0.0, 0.0
        for env_id in targets:
            n_b = int(request[env_id]) if request is not None else default_n
            e = self.envs[env_id]
            p = sigmoid(e.bench_slope * (self.skills[env_id] - e.bench_difficulty))
            u = self._calib_items.get(env_id)
            if u is None or len(u) != n_b:
                u = self.calib_rng.random(n_b)
            elif churn > 0.0:
                u = u.copy()
                flip = self.calib_rng.random(n_b) < churn
                u[flip] = self.calib_rng.random(int(flip.sum()))
            self._calib_items[env_id] = u
            correct = u < p
            s_d = float(correct.mean())
            se_d = math.sqrt(max(s_d * (1.0 - s_d), _EPS) / n_b)
            prev = self._calib_prev.get(env_id)
            if prev is not None and len(prev) == n_b:
                n01 = int(np.sum(~prev & correct))
                n10 = int(np.sum(prev & ~correct))
                disc, diff = n01 + n10, n01 - n10
                dse_by[env_id] = math.sqrt(max(disc - diff * diff / n_b, 1.0)) / n_b
            self._calib_prev[env_id] = correct
            pi_d = self.bench_weight[env_id]
            s_by[env_id], se_by[env_id] = s_d, se_d
            total += pi_d * s_d
            var_total += (pi_d * se_d) ** 2
            wsum += pi_d
        self.benchmark_evals += 1
        norm = wsum if wsum > 0 else 1.0  # score over the evaluated slices only
        return total / norm, math.sqrt(var_total) / norm, s_by, se_by, (dse_by or None)

    def calibration_observation(
        self,
        *,
        window_k: int,
        exposure_by_env: dict[str, float],
        window_cost_usd: float,
        eval_cost_usd: float,
        paired: bool = False,
        churn: float = 0.0,
        n_items: int | None = None,
        request: dict[str, int] | None = None,
    ) -> CalibrationObservation:
        """Produce a CalibrationObservation in the real-system schema (B.3).

        `request` (paired mode only) names the slices to evaluate and their item
        counts (targeted calibration, D-75); None evaluates every slice.
        """
        delta_se = None
        n_slice = int(n_items or self._n_b_default)
        if paired:
            total, se_total, by_domain, se_by_domain, delta_se = self.evaluate_benchmark_paired(
                churn, n_slice, request)
        else:
            total, se_total, by_domain, se_by_domain = self.evaluate_benchmark(observed=True)
        return CalibrationObservation(
            window_k=window_k,
            round=self.rounds_done,
            score_total=total,
            score_by_domain=by_domain,
            se_total=se_total,
            se_by_domain=se_by_domain,
            n_items=(sum(request.values()) if (paired and request) else n_slice * len(self._order)),
            eval_cost_usd=eval_cost_usd,
            exposure_by_env=dict(exposure_by_env),
            window_cost_usd=window_cost_usd,
            delta_se_by_domain=delta_se,
        )
