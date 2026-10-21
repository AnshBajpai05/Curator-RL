"""Base scheduler interface (Roadmap A.2 decision 1, Phase 3 task 3).

The scheduler is a pure function of observations: `RoundObservation` in,
a mixture (weights over environments) out. It never imports torch, TRL or a
dataset, and it MUST NOT read hidden simulator variables — only what arrives
through `RoundObservation` / `CalibrationObservation`.

Phase 3 adds the interface plus the Uniform and Random stubs needed to
validate the harness; the D-UCB Curator and the remaining baselines arrive in
Phase 5 (Roadmap Part P, v1 Phase 5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from curator_rl.core.types import CalibrationObservation, RoundObservation

Weights = dict[str, float]


class BaseScheduler(ABC):
    """Common interface for every allocation method (Roadmap Part J)."""

    # Roadmap H.6: calibration evaluations are run and charged ONLY for methods
    # that use them (D-72). Subclasses that consume CalibrationObservation set it.
    uses_calibration: bool = False

    def __init__(self, env_ids: Sequence[str]) -> None:
        if not env_ids:
            raise ValueError("a scheduler needs at least one environment")
        self.env_ids: tuple[str, ...] = tuple(env_ids)

    @abstractmethod
    def select_mixture(self, observation: RoundObservation | None) -> Weights:
        """Return the mixture w_t for the next round.

        `observation` is None before the first round. Weights must be
        non-negative and sum to 1 (within 1e-12).
        """

    def update_observation(self, observation: RoundObservation) -> None:  # noqa: B027
        """Consume the round observation (default: stateless methods ignore it)."""

    def update_calibration(self, observation: CalibrationObservation) -> None:  # noqa: B027
        """Consume a calibration observation (default: no-op until Phase 9)."""

    def get_state(self) -> Mapping[str, object]:
        """Serializable scheduler state for checkpointing (Roadmap Part N); stateless methods: {}."""
        return {}

    def save_checkpoint(self) -> Mapping[str, object]:
        return self.get_state()

    def load_checkpoint(self, state: Mapping[str, object]) -> None:  # noqa: B027
        """Restore from `get_state()` output (default: stateless methods ignore it)."""


def validate_weights(weights: Mapping[str, float], env_ids: Sequence[str]) -> None:
    """Raise if weights are not a valid mixture over exactly `env_ids`."""
    keys = set(weights)
    if keys != set(env_ids):
        raise ValueError(f"weights keys {sorted(keys)} != env ids {sorted(env_ids)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1 (got {total!r})")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
