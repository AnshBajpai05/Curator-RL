"""DUMP-style bandit baseline (Roadmap v3 section 4.4; Wang et al., DUMP).

UCB over distributions (environments) with the mean absolute GRPO advantage as
the arm reward: `score_i = mean_i + c * sqrt(ln(max(N, e)) / n_i)`, then
`softmax(score / tau)` with the shared exploration floor. A re-implementation from
the paper's description ("style"). `window_rounds = 0` keeps all-time counts (D-68);
`window_rounds = W > 0` uses only each arm's last W rounds with prompts, as DUMP
estimates learnability from recent samples (audit E-16: the all-time mean reacts
slowly once an arm saturates, which handicapped this baseline under drift).
Cost-blind and uncalibrated by design.
"""

from __future__ import annotations

import math
from collections import deque

from curator_rl.core.config import DumpBaselineCfg, SchedulerCfg
from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import BaseScheduler, validate_weights
from curator_rl.scheduler.mixture import softmax_floor


class DUMPStyleUCB(BaseScheduler):
    """UCB1 on mean |advantage| per prompt, softmax action map."""

    def __init__(self, env_ids, dump_cfg: DumpBaselineCfg, scheduler_cfg: SchedulerCfg) -> None:
        super().__init__(env_ids)
        self._cfg = dump_cfg
        self._sched = scheduler_cfg
        self._n = {e: 0.0 for e in self.env_ids}      # prompts seen (in the window if W > 0)
        self._sum = {e: 0.0 for e in self.env_ids}    # sum of per-prompt mean |A|
        self._window = int(dump_cfg.window_rounds)
        self._recent: dict[str, deque] = {e: deque() for e in self.env_ids}
        self._round = 0

    def scores(self) -> dict[str, float]:
        total = sum(self._n.values())
        log_arg = math.log(max(total, math.e))
        out: dict[str, float] = {}
        for e in self.env_ids:
            n = max(self._n[e], 1e-9)
            out[e] = self._sum[e] / n + self._cfg.exploration_coef * math.sqrt(log_arg / n)
        return out

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        self._round += 1
        if self._round <= self._sched.warmup_rounds or observation is None:
            return {e: 1.0 / len(self.env_ids) for e in self.env_ids}
        weights = softmax_floor(
            self.scores(), self._cfg.tau, self._sched.epsilon, list(self.env_ids), score_norm="none"
        )
        validate_weights(weights, self.env_ids)
        return weights

    def update_observation(self, observation: RoundObservation) -> None:
        for env_id, obs in observation.per_env.items():
            if obs.n_prompts > 0:
                self._n[env_id] += obs.n_prompts
                self._sum[env_id] += obs.sum_abs_adv
                if self._window > 0:
                    recent = self._recent[env_id]
                    recent.append((float(obs.n_prompts), float(obs.sum_abs_adv)))
                    if len(recent) > self._window:
                        n_old, s_old = recent.popleft()
                        self._n[env_id] -= n_old
                        self._sum[env_id] -= s_old

    def get_state(self) -> dict[str, object]:
        return {"round": self._round, "n": dict(self._n), "sum": dict(self._sum),
                "recent": {e: list(d) for e, d in self._recent.items()}}

    def load_checkpoint(self, state) -> None:
        self._round = int(state["round"])  # type: ignore[arg-type]
        self._n = {e: float(v) for e, v in state["n"].items()}  # type: ignore[union-attr]
        self._sum = {e: float(v) for e, v in state["sum"].items()}  # type: ignore[union-attr]
        self._recent = {e: deque(tuple(x) for x in state.get("recent", {}).get(e, []))  # type: ignore[union-attr]
                        for e in self.env_ids}
