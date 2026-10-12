"""Matched-budget ledger (SPEC section 4, COST_MODEL section 4).

Charged to the budget B: training steps (generation + scoring + update) and scheduler-owned
calibration evaluations. Not charged (identical for every method): reporting evaluations. The
run stops at the first round boundary where the charged total reaches B, so the overshoot is at
most one round (R optimizer steps). Units are GPU-hours (`usd_per_gpu_hour = 1`, COST_MODEL 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHARGED = ("train", "calib")
UNCHARGED = ("report", "setup")


@dataclass
class Ledger:
    budget_gpu_hours: float
    n_gpus: int = 1
    seconds: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in CHARGED + UNCHARGED})

    def charge(self, category: str, wall_seconds: float) -> None:
        if category not in self.seconds:
            raise ValueError(f"unknown ledger category '{category}'")
        self.seconds[category] += max(float(wall_seconds), 0.0) * self.n_gpus

    @property
    def charged_gpu_hours(self) -> float:
        return sum(self.seconds[k] for k in CHARGED) / 3600.0

    @property
    def remaining_gpu_hours(self) -> float:
        return max(self.budget_gpu_hours - self.charged_gpu_hours, 0.0)

    def exhausted(self) -> bool:
        return self.charged_gpu_hours >= self.budget_gpu_hours

    def get_state(self) -> dict:
        return {"budget_gpu_hours": self.budget_gpu_hours, "n_gpus": self.n_gpus, "seconds": dict(self.seconds)}

    def load_state(self, state: dict) -> None:
        self.seconds = {k: float(v) for k, v in state["seconds"].items()}
