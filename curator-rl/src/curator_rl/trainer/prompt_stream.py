"""Mixture-controlled prompt stream for real GRPO steps (Phase F; audit E-09, E-18).

Each optimizer step asks for P prompts. The stream turns the scheduler's current mixture into
integer per-environment quotas with deficit-carry rounding (`core.quotas.CarryQuotas`), so the
delivered share of every arm tracks the requested share within one prompt over the whole run,
even at P = 16 (largest-remainder rounding gave a 0.02 arm 0.031). Inside each finite
environment it walks a seeded permutation of the training split, one pass after another: no
prompt repeats within a pass (SPEC section 8). Procedural environments (Countdown) draw fresh
instances. Fully deterministic given the seed; the state is checkpointed with the run.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from curator_rl.core.quotas import CarryQuotas
from curator_rl.core.seeding import SeedManager
from curator_rl.core.types import Prompt
from curator_rl.envs.base import Environment


class _PassSampler:
    """Seeded permutation of one finite split; a new permutation per pass."""

    def __init__(self, size: int, seed: int) -> None:
        self.size, self.seed = size, seed
        self.pass_index, self.position = 0, 0
        self._order = self._permutation()

    def _permutation(self) -> np.ndarray:
        return np.random.default_rng([self.seed, self.pass_index]).permutation(self.size)

    def next_index(self) -> int:
        if self.position >= self.size:
            self.pass_index, self.position = self.pass_index + 1, 0
            self._order = self._permutation()
        idx = int(self._order[self.position])
        self.position += 1
        return idx

    def get_state(self) -> dict:
        return {"pass": self.pass_index, "position": self.position}

    def load_state(self, state: Mapping) -> None:
        self.pass_index, self.position = int(state["pass"]), int(state["position"])
        self._order = self._permutation()


class PromptStream:
    """Turns mixtures into the prompts of each GRPO step."""

    def __init__(self, envs: Mapping[str, Environment], *, seed: int, split: str = "train") -> None:
        self.envs = dict(envs)
        self.env_ids = tuple(sorted(self.envs))
        self.split = split
        self.seed = int(seed)
        self.weights = {e: 1.0 / len(self.env_ids) for e in self.env_ids}
        self.carry = CarryQuotas()
        seeds = SeedManager(self.seed)
        self._samplers: dict[str, _PassSampler] = {}
        self._procedural: dict[str, np.random.Generator] = {}
        for e in self.env_ids:
            size = self.envs[e].split_size(split)
            if size is None:
                self._procedural[e] = seeds.rng(f"stream_{e}")
            else:
                self._samplers[e] = _PassSampler(size, seeds.int_seed(f"stream_{e}"))
        self.delivered_round: dict[str, int] = {e: 0 for e in self.env_ids}

    def set_weights(self, weights: Mapping[str, float]) -> None:
        total = sum(max(float(weights.get(e, 0.0)), 0.0) for e in self.env_ids)
        if total <= 0:
            raise ValueError("mixture has no positive weight")
        self.weights = {e: max(float(weights.get(e, 0.0)), 0.0) / total for e in self.env_ids}
        self.delivered_round = {e: 0 for e in self.env_ids}

    def next_step(self, n_prompts: int) -> list[Prompt]:
        """The prompts of one optimizer step, grouped by environment (sorted env order)."""
        quotas = self.carry.next(self.weights, n_prompts)
        out: list[Prompt] = []
        for e in self.env_ids:
            for _ in range(quotas[e]):
                out.append(self._draw(e))
            self.delivered_round[e] += quotas[e]
        return out

    def _draw(self, env_id: str) -> Prompt:
        env = self.envs[env_id]
        if env_id in self._samplers:
            return env.prompt_at(self.split, self._samplers[env_id].next_index())
        return env.generate_prompt(self.split, self._procedural[env_id])

    # ------------------------------------------------------------------ checkpoints

    def get_state(self) -> dict:
        return {
            "weights": dict(self.weights), "carry": self.carry.get_state(),
            "samplers": {e: s.get_state() for e, s in self._samplers.items()},
            "procedural": {e: r.bit_generator.state for e, r in self._procedural.items()},
            "delivered_round": dict(self.delivered_round),
        }

    def load_state(self, state: Mapping) -> None:
        self.weights = {e: float(v) for e, v in state["weights"].items()}
        self.carry.load_state(state["carry"])
        for e, s in state["samplers"].items():
            self._samplers[e].load_state(s)
        for e, s in state["procedural"].items():
            self._procedural[e].bit_generator.state = s
        self.delivered_round = {e: int(v) for e, v in state["delivered_round"].items()}
