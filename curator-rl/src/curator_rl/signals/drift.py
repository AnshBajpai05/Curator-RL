"""Change detection on an arm's pass rate (improvement B2: drift-triggered discounting).

Discounting every arm's bandit evidence by gamma each round (D-UCB) costs CURATOR 0.013-0.020 on the
refit scenarios (D-90), and the fair GPU-shape tuning chose gamma = 1. With gamma = 1 nothing protects
against an abrupt change (scenario S-E). The alternative: keep all evidence until a change is
DETECTED, then decay only that arm's evidence once.

Statistic: the standardised surprise of the round's pass rate,

    x_t = (p_raw,t - p_hat_{t-1}) / se_t,     se_t = sqrt(p (1 - p) D / (G n_t)),

where p_hat_{t-1} is the arm's discounted pass-rate estimate before this round, n_t the round's prompt
count and D the arm's pooled dispersion ratio from the reward audit (signals/audit.py): under the
Beta-binomial model a group pass rate has variance p (1 - p) D / G. The pooled D is used instead of the
round's own sample variance, which at 3-12 prompts per round is so noisy that it inflated the
statistic's SD to 1.26-1.48. A two-sided Page-Hinkley / CUSUM test accumulates

    up_t   = max(0, up_{t-1}   + x_t - delta)
    down_t = max(0, down_{t-1} - x_t - delta)

and alarms when either exceeds `threshold`; both sums then restart. Slow learning moves x_t by a
fraction of a standard error per round and is absorbed by `delta`; an abrupt shift of several
standard errors accumulates within a few rounds. Deterministic, no RNG.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


class PageHinkley:
    """Two-sided CUSUM on standardised residuals."""

    def __init__(self, delta: float, threshold: float) -> None:
        self.delta, self.threshold = float(delta), float(threshold)
        self.up = 0.0
        self.down = 0.0

    def update(self, x: float) -> bool:
        if not math.isfinite(x):
            return False
        self.up = max(0.0, self.up + x - self.delta)
        self.down = max(0.0, self.down - x - self.delta)
        if self.up > self.threshold or self.down > self.threshold:
            self.up = self.down = 0.0
            return True
        return False

    def get_state(self) -> dict[str, float]:
        return {"up": self.up, "down": self.down}

    def load_checkpoint(self, state: Mapping[str, float]) -> None:
        self.up, self.down = float(state["up"]), float(state["down"])


def standardised_surprise(k_success: int, n_rollouts: int, n_prompts: int, group_size: int,
                          p_prev: float, dispersion: float) -> float:
    """x_t of the module docstring; nan when the round or the dispersion carries no information."""
    if n_prompts <= 0 or n_rollouts <= 0 or not math.isfinite(dispersion):
        return float("nan")
    p_raw = k_success / n_rollouts
    p = min(max(p_prev, 1e-3), 1.0 - 1e-3)
    var = p * (1.0 - p) * max(dispersion, 1.0) / max(group_size, 1)
    return (p_raw - p_prev) / math.sqrt(var / n_prompts)
