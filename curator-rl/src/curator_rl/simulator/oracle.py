"""Oracles with known optima (Roadmap F.4, v1 Phase 3).

- **O1 DPOracle** (N <= 3, no drift/late-start): exact DP over discretised
  skills and budget. Deterministic mean dynamics; value = final benchmark.
- **O2 StaticOracle**: best fixed mixture, by simplex-grid search (N <= 4) or
  seeded Dirichlet random search with local hill-climb refinement (N > 4).
- **O3 MyopicOracle**: first-order equimarginal rule — weight proportional to
  the true marginal benchmark gain per dollar of each environment.

Oracles MAY read hidden world state by design (F.4); they are references, not
schedulers. The reported metric is oracle-gap closure
`(S_method - S_uniform) / (S_oracle - S_uniform)` at equal budget.

All oracle evaluation runs on a *copy* of the environment parameters so the
live world (and the caller's scenario) is never mutated.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from itertools import combinations_with_replacement

import numpy as np

from curator_rl.core.seeding import SeedManager
from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import Weights
from curator_rl.simulator.world import SimEnvParams, SimWorld, learnability, sigmoid


def sigmoid_arr(x: np.ndarray) -> np.ndarray:
    """Vectorised logistic; stable at extreme inputs."""
    x = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------- helpers


def _mean_unit_cost(e: SimEnvParams, skill: float) -> float:
    factor = 1.0 + e.cost_skill_slope * (skill - e.skill0)
    return e.cost_usd_per_prompt * max(factor, 1e-3)


def _copy_envs(envs: list[SimEnvParams]) -> list[SimEnvParams]:
    """Shallow copies with fresh transfer dicts, so drift mutation is local."""
    return [dataclasses.replace(e, transfer_out=dict(e.transfer_out)) for e in envs]


def _effective_weights(
    envs: list[SimEnvParams], weights: dict[str, float], round_t: int
) -> dict[str, float]:
    """Mask late-start envs and renormalise (mirrors the harness constraint)."""
    masked = {
        e.env_id: (0.0 if e.late_start_round > round_t else weights.get(e.env_id, 0.0))
        for e in envs
    }
    total = sum(masked.values())
    if total <= 0:
        avail = [e.env_id for e in envs if e.late_start_round <= round_t]
        n = max(len(avail), 1)
        return {i: (1.0 / n if i in avail else 0.0) for i in masked}
    return {i: v / total for i, v in masked.items()}


def _transfer_in(envs: list[SimEnvParams]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {e.env_id: {} for e in envs}
    noisy = {e.env_id for e in envs if e.noisy_q is not None}
    for e in envs:
        if e.env_id in noisy:
            continue
        for target, coeff in e.transfer_out.items():
            if target not in noisy:
                out[target][e.env_id] = float(coeff)
    return out


def _resolve_bench_weights(envs: list[SimEnvParams]) -> dict[str, float]:
    raw = [e.bench_weight for e in envs]
    if all(w is None for w in raw):
        n = len(envs)
        return {e.env_id: 1.0 / n for e in envs}
    vals = [float(w) for w in raw if w is not None]
    if len(vals) != len(raw):
        raise ValueError("benchmark weights must be all set or all None")
    total = sum(vals)
    return {e.env_id: w / total for e, w in zip(envs, vals)}


def max_rounds_cap(envs: list[SimEnvParams], *, prompts_per_round: int, budget_usd: float) -> int:
    min_cost = min(_mean_unit_cost(e, e.skill0) for e in envs)
    return max(200, math.ceil(budget_usd / (prompts_per_round * min_cost)) + 2)


def deterministic_final_score(
    envs: list[SimEnvParams],
    weights_fn: Callable[[dict[str, float], int], dict[str, float]],
    *,
    steps_per_round: int,
    prompts_per_round: int,
    budget_usd: float,
    max_rounds: int,
) -> float:
    """Mean-dynamics episode under a mixture policy; returns the expected
    noiseless benchmark score (O2 valuation; same semantics as the DP model).

    Works on copies of `envs`; the caller's parameters are never mutated.
    """
    envs = _copy_envs(envs)
    skills = {e.env_id: float(e.skill0) for e in envs}
    difficulty = {e.env_id: float(e.difficulty) for e in envs}
    noisy = {e.env_id for e in envs if e.noisy_q is not None}
    transfer = _transfer_in(envs)
    spend = 0.0
    for round_t in range(max_rounds):
        weights = _effective_weights(envs, weights_fn(skills, round_t), round_t)
        g = {
            e.env_id: learnability(sigmoid(e.slope * (skills[e.env_id] - difficulty[e.env_id])))
            for e in envs
            if e.env_id not in noisy
        }
        for e in envs:
            if e.env_id in noisy:
                continue
            inflow = weights.get(e.env_id, 0.0) * g.get(e.env_id, 0.0)
            for src, t_ji in transfer[e.env_id].items():
                inflow += t_ji * weights.get(src, 0.0) * g.get(src, 0.0)
            gate = 1.0
            if e.gate_prereq is not None:
                gate = sigmoid(e.gate_slope * (skills[e.gate_prereq] - e.gate_theta))
            skills[e.env_id] += steps_per_round * e.eta * gate * inflow
        spend += prompts_per_round * sum(
            weights.get(e.env_id, 0.0) * _mean_unit_cost(e, skills[e.env_id]) for e in envs
        )
        for e in envs:
            if e.drift_round is not None and round_t + 1 >= e.drift_round:
                difficulty[e.env_id] += e.drift_shift
                e.drift_round = None
        if spend >= budget_usd:
            break
    bench = _resolve_bench_weights(envs)
    return sum(
        bench[e.env_id] * sigmoid(e.bench_slope * (skills[e.env_id] - e.bench_difficulty))
        for e in envs
    )


def simplex_grid(n: int, step: float) -> list[tuple[float, ...]]:
    """All mixtures on a simplex grid with the given step (each sums to 1)."""
    k = int(round(1.0 / step))
    if k <= 0 or abs(k * step - 1.0) > 1e-9:
        raise ValueError("simplex_step must divide 1.0")
    combos: list[tuple[float, ...]] = []
    for c in combinations_with_replacement(range(n), k):
        w = [0.0] * n
        for idx in c:
            w[idx] += 1.0 / k
        combos.append(tuple(w))
    return combos


# ------------------------------------------------------------------- oracles


class _OracleHooks:
    """No-op harness hooks: oracles are stateless across observations."""

    def update_observation(self, observation: RoundObservation) -> None:
        return None

    def update_calibration(self, observation) -> None:  # noqa: ANN001 - CalibrationObservation
        return None


class StaticOracle(_OracleHooks):
    """O2: best fixed mixture (Roadmap F.4).

    N <= 4: exhaustive simplex-grid search. N > 4: seeded Dirichlet random
    search followed by local hill-climb refinement (documented in
    DECISIONS.md; CMA-ES would add a dependency for little gain here).
    """

    def __init__(
        self,
        world: SimWorld,
        *,
        simplex_step: float = 0.1,
        random_search_draws: int = 1500,
        refine_rounds: int = 300,
        seed: int = 0,
    ) -> None:
        self.env_ids = list(world._order)
        self._R = world.R
        self._M = world.M
        self._budget = world.budget_usd
        self._max_rounds = max_rounds_cap(
            list(world.envs.values()), prompts_per_round=world.M, budget_usd=world.budget_usd
        )
        self._rng = SeedManager(seed).rng("oracle")
        best_w, best_v = self._search(world, simplex_step, random_search_draws, refine_rounds)
        self.best_weights: dict[str, float] = dict(zip(self.env_ids, best_w))
        self.best_value: float = best_v

    def _score(self, world: SimWorld, mixture: tuple[float, ...]) -> float:
        return deterministic_final_score(
            list(world.envs.values()),
            lambda _s, _t, m=mixture: dict(zip(self.env_ids, m)),
            steps_per_round=self._R,
            prompts_per_round=self._M,
            budget_usd=self._budget,
            max_rounds=self._max_rounds,
        )

    def _search(
        self, world: SimWorld, step: float, draws: int, refine_rounds: int
    ) -> tuple[tuple[float, ...], float]:
        n = len(self.env_ids)
        if n <= 4:
            candidates = simplex_grid(n, step)
        else:
            candidates = [tuple(map(float, d)) for d in self._rng.dirichlet(np.ones(n), size=draws)]
        best_w = max(candidates, key=lambda m: self._score(world, m))
        best_v = self._score(world, best_w)
        if n > 4:
            w = np.array(best_w)
            for _ in range(refine_rounds):
                i, j = self._rng.integers(0, n, size=2)
                if i == j:
                    continue
                amount = min(float(self._rng.uniform(0, 0.2)), float(w[j]))
                trial = w.copy()
                trial[i] += amount
                trial[j] -= amount
                v = self._score(world, tuple(trial))
                if v > best_v:
                    best_v, w = v, trial
            best_w = tuple(float(x) / float(w.sum()) for x in w)
        return best_w, best_v

    def select_mixture(self, observation: RoundObservation | None) -> Weights:
        return dict(self.best_weights)


class MyopicOracle(_OracleHooks):
    """O3: allocate by true marginal benchmark gain per dollar (F.4).

    First-order equimarginal approximation:
        r_i = (dS/ds_i) * eta_i_eff * g_i / c_i,   w_i proportional to r_i+
    Second-order and cross-env transfer effects are ignored; documented as
    the O3 implementation choice in DECISIONS.md.
    """

    def __init__(self, world: SimWorld) -> None:
        self.world = world

    def select_mixture(self, observation: RoundObservation | None) -> Weights:
        world = self.world
        raw: dict[str, float] = {}
        for env_id in world._order:
            e = world.envs[env_id]
            if env_id in world.noisy or e.late_start_round > world.rounds_done:
                raw[env_id] = 0.0
                continue
            p_b = sigmoid(e.bench_slope * (world.skills[env_id] - e.bench_difficulty))
            ds_ds = world.bench_weight[env_id] * e.bench_slope * p_b * (1.0 - p_b)
            g_i = learnability(world.true_pass(env_id))
            r_i = ds_ds * e.eta * world.gate_factor(env_id) * g_i / world.unit_cost(env_id)
            raw[env_id] = max(r_i, 0.0)
        total = sum(raw.values())
        if total <= 0:
            n = len(world._order)
            return {i: 1.0 / n for i in world._order}
        return {i: v / total for i, v in raw.items()}


class DPOracle(_OracleHooks):
    """O1: exact DP over a discretised skill grid and budget (Roadmap F.4).

    Supported for N <= 3, no drift, no late-start envs and no cost-skill
    coupling (those make the dynamics time- or cost-state-dependent, which
    the unit-indexed backward induction cannot represent).

    Dynamics are deterministic, so value iteration runs backward over budget
    units on a per-env skill grid, vectorised with numpy:

        V(s, u)   = max(bench(s), max_w V(step(s, w), u - cost_units(w)))
        V(s, 0)   = bench(s)

    Costs are rounded UP to budget units, so the DP value is conservative
    and dominates every fixed mixture evaluated under the same discretised
    model (`fixed_policy_value`). Skill quantisation is fine enough that
    per-round gains near the initial full-weight learning rate move several
    grid levels per round; far from that rate (e.g. saturated arms) gains
    quantise to zero, which only makes the DP pessimistic about allocations
    that are nearly worthless anyway. The policy replans from the live world
    state each round; V is computed once per episode and queried.
    """

    _MAX_LEVELS = 80

    def __init__(
        self,
        world: SimWorld,
        *,
        skill_levels: int = 60,
        budget_units: int = 40,
        simplex_step: float = 0.1,
    ) -> None:
        n = len(world._order)
        if n > 3:
            raise ValueError("DPOracle supports N <= 3 (Roadmap F.4)")
        for e in world.envs.values():
            if e.drift_round is not None or e.late_start_round > 0:
                raise ValueError("DPOracle requires no drift and no late-start envs")
            if e.cost_skill_slope != 0.0:
                raise ValueError("DPOracle requires cost_skill_slope == 0 (state-independent costs)")
        self.world = world
        self.order: tuple[str, ...] = world._order
        self.n_levels = max(4, min(int(skill_levels), self._MAX_LEVELS))
        self.budget_units = int(budget_units)
        self.unit_usd = world.budget_usd / self.budget_units
        self.actions = simplex_grid(n, simplex_step)
        self.action_costs = [
            int(math.ceil(world.M * sum(w_i * world.envs[env_id].cost_usd_per_prompt
                for w_i, env_id in zip(action, self.order)) / self.unit_usd - 1e-12))
            for action in self.actions
        ]
        self.learnable = [i for i in world._order if i not in world.noisy]
        self.frozen = {i: world.skills[i] for i in world._order if i in world.noisy}
        self._build_grid()
        self._gain_tables = self._gain_tables_per_action()
        self._V: np.ndarray | None = None

    # -- grid and dynamics ----------------------------------------------------

    def _build_grid(self) -> None:
        world = self.world
        min_cost = min(e.cost_usd_per_prompt for e in world.envs.values())
        est_rounds = world.budget_usd / (world.M * min_cost)
        self._lo: dict[str, float] = {}
        self._delta: dict[str, float] = {}
        self._grids: dict[str, np.ndarray] = {}
        for env_id in self.learnable:
            e = world.envs[env_id]
            max_inflow = e.eta * (1.0 + sum(abs(t) for t in world.transfer_in[env_id].values()))
            span = min(4.0, world.R * max_inflow * est_rounds)
            lo = e.skill0 - 0.25
            self._lo[env_id] = lo
            self._delta[env_id] = (span + 0.25) / (self.n_levels - 1)
            self._grids[env_id] = lo + self._delta[env_id] * np.arange(self.n_levels)
        self._mesh = np.meshgrid(
            *[self._grids[i] for i in self.learnable], indexing="ij"
        ) if self.learnable else []
        self._skill_at = {
            dim: {idx: float(self._grids[dim][idx]) for idx in range(self.n_levels)}
            for dim in self.learnable
        }

    def _levels_of(self, skills: dict[str, float]) -> tuple[int, ...]:
        idx = []
        for dim in self.learnable:
            pos = (skills[dim] - self._lo[dim]) / self._delta[dim]
            idx.append(int(np.clip(round(pos), 0, self.n_levels - 1)))
        return tuple(idx)

    def _skills_at(self, levels: tuple[int, ...]) -> dict[str, float]:
        skills = dict(self.frozen)
        skills.update(
            {dim: float(self._grids[dim][lv]) for dim, lv in zip(self.learnable, levels)}
        )
        return skills

    def _gain_tables_per_action(self) -> list[list[np.ndarray]]:
        """Per-action, per-learnable-env skill gain of one round in grid levels.

        Vectorised over the whole skill grid. Mirrors SimWorld.step_round:
        ds_j = R * eta_j * gate_j * sum_i T_ji w_i g_i, with T_jj = 1.
        """
        world = self.world
        g: dict[str, np.ndarray | float] = {}
        for dim, mesh in zip(self.learnable, self._mesh):
            e = world.envs[dim]
            g[dim] = learnability(sigmoid_arr(e.slope * (mesh - e.difficulty)))
        for dim in self.frozen:
            e = world.envs[dim]
            g[dim] = learnability(sigmoid(e.slope * (self.frozen[dim] - e.difficulty)))
        out = []
        for action, _cost in zip(self.actions, self.action_costs):
            w = dict(zip(world._order, action))
            gains = []
            for j in self.learnable:
                e = world.envs[j]
                inflow = w.get(j, 0.0) * g[j]
                for src, t_ji in world.transfer_in[j].items():
                    inflow = inflow + t_ji * w.get(src, 0.0) * g[src]
                gain = world.R * e.eta * inflow
                if e.gate_prereq is not None:
                    if e.gate_prereq in self.frozen:
                        gate = sigmoid(e.gate_slope * (self.frozen[e.gate_prereq] - e.gate_theta))
                        gain = gain * gate
                    else:
                        # gate varies along the PREREQ's axis, broadcast elsewhere
                        axis = self.learnable.index(e.gate_prereq)
                        shape = [1] * len(self.learnable)
                        shape[axis] = self.n_levels
                        gate = sigmoid_arr(e.gate_slope * (self._grids[e.gate_prereq] - e.gate_theta))
                        gain = gain * np.asarray(gate).reshape(shape)
                gains.append(np.rint(np.asarray(gain) / self._delta[j]))
            out.append(gains)
        return out

    def _shifted_indices(self, gains: list[np.ndarray]) -> tuple[np.ndarray, ...]:
        """Grid indices after one round, one broadcast-ready array per dim."""
        idx = []
        for axis, gain in enumerate(gains):
            base = np.arange(self.n_levels, dtype=np.int64).reshape(
                [self.n_levels if a == axis else 1 for a in range(len(self.learnable))]
            )
            new = base + gain.astype(np.int64)
            idx.append(np.clip(new, 0, self.n_levels - 1))
        return tuple(idx)

    def _gather_at_state(
        self, idx: tuple[np.ndarray, ...], levels: tuple[int, ...], units: int, V: np.ndarray
    ) -> float:
        """V at the shifted state reached from `levels` with `units` left.

        The shifted index arrays are full-mesh when transfers couple dims, so
        the entry is selected with the full current-state level tuple.
        """
        sel = tuple(int(arr[levels]) for arr in idx)
        return float(V[sel + (units,)])

    # -- value iteration --------------------------------------------------------

    def _benchmark_grid(self) -> np.ndarray:
        world = self.world
        total = np.zeros([self.n_levels] * len(self.learnable))
        for env_id in world._order:
            e = world.envs[env_id]
            if env_id in self.frozen:
                p = sigmoid(e.bench_slope * (self.frozen[env_id] - e.bench_difficulty))
                total = total + world.bench_weight[env_id] * p
            else:
                mesh = self._mesh[self.learnable.index(env_id)]
                total = total + world.bench_weight[env_id] * sigmoid_arr(
                    e.bench_slope * (mesh - e.bench_difficulty)
                )
        return total

    def solve(self) -> np.ndarray:
        """Backward induction over budget units; returns the value table V."""
        if self._V is not None:
            return self._V
        bench = self._benchmark_grid()
        shifted = [self._shifted_indices(g) for g in self._gain_tables]
        shape = [self.n_levels] * len(self.learnable) + [self.budget_units + 1]
        V = np.zeros(shape)
        V[..., 0] = bench
        for u in range(1, self.budget_units + 1):
            best = bench.copy()
            for idx, cost in zip(shifted, self.action_costs):
                if cost > u:
                    continue
                candidate = V[idx + (u - cost,)]
                np.maximum(best, candidate, out=best)
            V[..., u] = best
        self._V = V
        return V

    def value(self, units: int | None = None) -> float:
        """Optimal value from the live world state with `units` budget left."""
        if units is None:
            units = self.budget_units
        V = self.solve()
        levels = self._levels_of(self.world.skills)
        return float(V[levels + (max(0, int(units)),)])

    def fixed_policy_value(self, mixture: dict[str, float], *, units: int | None = None) -> float:
        """Value of always playing `mixture` under the same discretised model.

        Rolls the quantised one-round transitions until the budget units are
        exhausted, then evaluates the benchmark on the decoded skills. (V
        itself cannot be used here: it embeds the *optimal* continuation.)
        """
        if units is None:
            units = self.budget_units
        a_idx = next(
            (i for i, action in enumerate(self.actions)
             if all(abs(w - mixture.get(env_id, 0.0)) < 1e-9
                    for w, env_id in zip(action, self.order))),
            None,
        )
        if a_idx is None:
            raise ValueError(f"mixture {mixture} is not on the simplex grid")
        cost = self.action_costs[a_idx]
        gains = self._gain_tables[a_idx]
        levels = self._levels_of(self.world.skills)
        units_left = units
        while cost <= units_left:
            units_left -= cost
            idx = self._shifted_indices(gains)
            levels = tuple(int(arr[levels]) for arr in idx)
        skills = dict(self.frozen)
        skills.update(
            {dim: float(self._grids[dim][lv]) for dim, lv in zip(self.learnable, levels)}
        )
        total = 0.0
        for env_id in self.order:
            e = self.world.envs[env_id]
            total += self.world.bench_weight[env_id] * sigmoid(
                e.bench_slope * (skills[env_id] - e.bench_difficulty)
            )
        return total

    # -- policy ------------------------------------------------------------------

    def best_mixture_now(self, units: int | None = None) -> Weights:
        """Optimal first action from the live world state (receding horizon)."""
        if units is None:
            units = self.budget_units
        V = self.solve()
        levels = self._levels_of(self.world.skills)
        best_w, best_v = None, -math.inf
        for a_idx, (action, cost) in enumerate(zip(self.actions, self.action_costs)):
            if cost > units:
                continue
            idx = self._shifted_indices(self._gain_tables[a_idx])
            v = self._gather_at_state(idx, levels, units - cost, V)
            if v > best_v:
                best_v, best_w = v, dict(zip(self.world._order, action))
        if best_w is None:
            return {i: 1.0 / len(self.world._order) for i in self.world._order}
        return best_w

    def select_mixture(self, observation: RoundObservation | None) -> Weights:
        units = self.budget_units
        if observation is not None:
            units = max(0, int(math.floor(observation.budget_remaining_usd / self.unit_usd)))
        return self.best_mixture_now(units)
