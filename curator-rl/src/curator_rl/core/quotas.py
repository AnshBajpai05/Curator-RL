"""Weight-to-quota conversion (Roadmap D.4): largest-remainder and deficit-carry rounding.

`m_i = round(M * w_i)` with the residual distributed by largest remainder and
a seeded tie-break, so `sum(m_i) == M` exactly and the mapping is
deterministic. `stochastic` rounding is left as a Phase 8 option.

Lives in `core` (L0, no scientific logic) so the L1 scheduler can compute
`MixtureDecision.quotas` without importing the simulator (layering rule B.2);
`simulator.quotas` re-exports this implementation for backwards compatibility.
"""

from __future__ import annotations

import numpy as np


def largest_remainder_quotas(
    weights: dict[str, float],
    total: int,
    *,
    tie_break_seed: int = 0,
) -> dict[str, int]:
    """Integer quotas summing exactly to `total` (Roadmap D.4)."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return {}
    keys = sorted(weights)
    exact = np.array([weights[k] * total for k in keys], dtype=float)
    floors = np.floor(exact).astype(int)
    remainder = total - int(floors.sum())
    if remainder > 0:
        fracs = exact - floors
        # deterministic tie-break: jitter by hash order, then stable sort
        rng = np.random.default_rng(abs(tie_break_seed) + 1)
        jitter = rng.uniform(0, 1e-9, size=len(keys))
        order = np.argsort(-(fracs + jitter), kind="stable")
        for idx in order[:remainder]:
            floors[idx] += 1
    return {k: int(v) for k, v in zip(keys, floors)}


class CarryQuotas:
    """Deficit-carry rounding (audit E-09): long-run delivered shares match requested ones.

    Per-round largest-remainder rounding gives every small arm its leftover prompt every
    round: at M = 32 an arm requested at 0.020 receives 1 prompt (0.031) every round. Here
    each arm accumulates requested prompts D_i += w_i M and receives floor(D_i - delivered_i)
    plus the round's leftover prompts by largest outstanding deficit, so
    |delivered_i - sum_t w_{i,t} M| < 1 for every arm at every round (exact sum M per round).
    Deterministic (ties broken by env id); the state is two dicts, checkpointed with the run.
    """

    def __init__(self) -> None:
        self.requested: dict[str, float] = {}
        self.delivered: dict[str, int] = {}

    def next(self, weights: dict[str, float], total: int) -> dict[str, int]:
        if total < 0:
            raise ValueError("total must be non-negative")
        keys = sorted(weights)
        for k in keys:
            self.requested[k] = self.requested.get(k, 0.0) + max(float(weights[k]), 0.0) * total
            self.delivered.setdefault(k, 0)
        owed = {k: self.requested[k] - self.delivered[k] for k in keys}
        quota = {k: max(int(np.floor(owed[k])), 0) for k in keys}
        excess = sum(quota.values()) - total
        if excess > 0:                       # can only happen after weights moved sharply
            for k in sorted(keys, key=lambda k: (owed[k] - quota[k], k)):
                take = min(excess, quota[k])
                quota[k] -= take
                excess -= take
                if excess == 0:
                    break
        left = total - sum(quota.values())
        for k in sorted(keys, key=lambda k: (-(owed[k] - quota[k]), k))[:max(left, 0)]:
            quota[k] += 1
        for k in keys:
            self.delivered[k] += quota[k]
        return quota

    def get_state(self) -> dict[str, dict]:
        return {"requested": dict(self.requested), "delivered": dict(self.delivered)}

    def load_state(self, state: dict[str, dict]) -> None:
        self.requested = {k: float(v) for k, v in state["requested"].items()}
        self.delivered = {k: int(v) for k, v in state["delivered"].items()}
