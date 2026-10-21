"""Trace replay skeleton, Tier 1R (Roadmap F.6; stretch component).

Replays per-environment traces from real runs against candidate schedulers
under the **arm-clock validity assumption**: an environment's outcome depends
only on its own number of pulls (no transfer/interference modelling). Replay
is valid for scheduler logic and hyperparameters ONLY — never for transfer
claims or leaderboard claims (Roadmap F.6).

Trace format: JSONL, one record per logged round::

    {"round": 1, "per_env": {"gsm8k": {"k_success": 12, "n_rollouts": 64,
                                       "n_prompts": 8, "cost_usd": 0.01}, ...}}

When the scheduler pulls an environment more often than the trace did, the
last logged chunk repeats (documented extrapolation; coverage of the logged
mixture must be wide enough, F.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from curator_rl.core.jsonl import read_jsonl
from curator_rl.core.quotas import largest_remainder_quotas
from curator_rl.core.types import EnvRoundObs, RoundObservation
from curator_rl.scheduler.base import BaseScheduler


@dataclass
class ReplayEnvState:
    """Per-environment replay queue under the arm-clock assumption."""

    env_id: str
    chunks: list[dict] = field(default_factory=list)
    cursor: int = 0

    def take(self, n_prompts: int) -> dict:
        if not self.chunks:
            raise ValueError(f"env '{self.env_id}' has no trace chunks")
        chunk = self.chunks[min(self.cursor, len(self.chunks) - 1)]
        self.cursor += 1
        scale = n_prompts / max(chunk.get("n_prompts", 1), 1)
        # every additive per-prompt statistic scales with the prompts taken (audit E-17: mixed
        # groups were dropped, so richness was 0, and score sums were not sums)
        out: dict = {"n_prompts": n_prompts}
        for key in ("k_success", "n_rollouts", "n_groups_mixed"):
            out[key] = int(round(chunk.get(key, 0) * scale))
        for key in ("cost_usd", "sum_score", "sum_score_sq", "sum_abs_adv"):
            out[key] = float(chunk.get(key, 0.0)) * scale
        return out


@dataclass
class ReplayResult:
    rounds: int
    final_pass_rates: dict[str, float]
    total_cost_usd: float


class TraceReplayer:
    """Replay a logged run against a candidate scheduler (skeleton, F.6)."""

    def __init__(self, env_ids: list[str], trace_path: Path) -> None:
        self.env_ids = list(env_ids)
        self.states: dict[str, ReplayEnvState] = {i: ReplayEnvState(i) for i in self.env_ids}
        self._load(trace_path)

    def _load(self, trace_path: Path) -> None:
        for record in read_jsonl(Path(trace_path)):
            for env_id, chunk in record.get("per_env", {}).items():
                if env_id in self.states:
                    self.states[env_id].chunks.append(chunk)

    def run(self, scheduler: BaseScheduler, *, prompts_per_round: int, budget_usd: float,
            max_rounds: int = 10_000) -> ReplayResult:
        spend = 0.0
        rounds = 0
        prev_obs: RoundObservation | None = None
        for round_t in range(max_rounds):
            weights = scheduler.select_mixture(prev_obs)
            per_env: dict[str, EnvRoundObs] = {}
            round_cost = 0.0
            quotas = largest_remainder_quotas(weights, prompts_per_round, tie_break_seed=round_t)
            for env_id in self.env_ids:
                m_i = quotas.get(env_id, 0)
                chunk = self.states[env_id].take(m_i) if m_i > 0 else {
                    "k_success": 0, "n_rollouts": 0, "n_prompts": 0, "n_groups_mixed": 0,
                    "cost_usd": 0.0, "sum_score": 0.0, "sum_score_sq": 0.0, "sum_abs_adv": 0.0,
                }
                n_roll = chunk["n_rollouts"]
                k = chunk["k_success"]
                n_p = chunk["n_prompts"]
                rate = k / max(n_roll, 1)
                per_env[env_id] = EnvRoundObs(
                    env_id=env_id, n_prompts=n_p, n_rollouts=n_roll, k_success=k,
                    n_groups_mixed=chunk["n_groups_mixed"],
                    # traces without per-prompt moments fall back to the pooled rate per prompt
                    sum_score=chunk["sum_score"] or rate * n_p,
                    sum_score_sq=chunk["sum_score_sq"] or rate * rate * n_p,
                    prompt_tokens=0, completion_tokens=0, verifier_seconds=0.0,
                    gpu_seconds=0.0, cost_usd=chunk["cost_usd"], sum_abs_adv=chunk["sum_abs_adv"],
                )
                round_cost += chunk["cost_usd"]
            spend += round_cost
            rounds += 1
            prev_obs = RoundObservation(
                round=round_t + 1, steps=round_t + 1, per_env=per_env,
                weights_used=dict(weights), round_cost_usd=round_cost, overhead_usd=0.0,
                budget_remaining_usd=max(budget_usd - spend, 0.0),
            )
            scheduler.update_observation(prev_obs)
            if spend >= budget_usd:
                break
        final_rates = {
            env_id: (sum(c.get("k_success", 0) for c in st.chunks) / max(
                sum(c.get("n_rollouts", 0) for c in st.chunks), 1))
            for env_id, st in self.states.items()
        }
        return ReplayResult(rounds=rounds, final_pass_rates=final_rates, total_cost_usd=spend)
