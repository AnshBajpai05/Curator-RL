"""Environment abstraction (Roadmap D.1, frozen interface).

An Environment turns data (or a generator) into `Prompt`s and turns
(model, prompt, response) triples into verified rewards. It never computes
learning progress and never reads a clock except its own verifier timing.

Ownership summary (Roadmap D.1):
- success / reward / verifier time / parse failure  -> Environment (Verdict)
- prompt tokens, completion tokens, rollout count    -> trainer adapter
- GPU-seconds / dollars                              -> Cost Meter (Part H)
- pass rate, LP, richness, status, proxy reward      -> Signal Engine
- weights, UCB scores                                -> Scheduler
"""

from __future__ import annotations

import abc
import threading
from typing import Any

from curator_rl.core.types import CostPrior, Prompt, Verdict


class VerifierTimeoutError(RuntimeError):
    """The verifier did not finish within its configured timeout."""


def run_with_timeout(fn, timeout_s: float) -> Any:
    """Run fn() with a wall-clock timeout.

    Cross-platform (Roadmap O.1): a daemon worker thread is joined with the
    timeout; on timeout the calling context proceeds while the daemon thread
    winds down in the background. Verifiers used in Phase 2 are bounded by
    design, so orphaned threads are a safety net, not a leak mechanism
    (unbounded loops live in the Phase-11 code sandbox, S-11).
    """
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - forwarded to caller
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise VerifierTimeoutError(f"verifier exceeded {timeout_s:.2f}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class Environment(abc.ABC):
    """Contract (SPEC section 8):

    - generate_prompt  deterministic given (split, rng state); never serves
      another split's items.
    - generate_batch   sampling without replacement within a pass over a
      finite split; fresh instances for procedural environments.
    - evaluate_response pure, thread-safe, timeout, times itself
      (verifier_seconds). Verification only - no reward shaping.
    - compute_reward   reward shaping, separate from verification; range
      [0, 1] unless the spec says otherwise.
    - estimate_cost    cold-start prior only, never a measured cost.
    - metadata         dataset version, split sizes, manifest hash, difficulty
      parameters.
    - split_ids        ids of a split, used by leakage tests.
    """

    cfg: Any                      # the environment's validated config section (token limits, timeouts, ...)

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id

    @abc.abstractmethod
    def generate_prompt(self, split: str, rng) -> Prompt: ...

    @abc.abstractmethod
    def generate_batch(self, split: str, n: int, rng, exclude: set[str] | None = None) -> list[Prompt]: ...

    @abc.abstractmethod
    def evaluate_response(self, prompt: Prompt, response: str) -> Verdict: ...

    @abc.abstractmethod
    def compute_reward(self, verdict: Verdict) -> float: ...

    @abc.abstractmethod
    def estimate_cost(self) -> CostPrior: ...

    @abc.abstractmethod
    def metadata(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def split_ids(self, split: str) -> list[str]: ...

    # Pass-based sampling (audit E-18: SPEC section 8 asks for "without replacement within a
    # pass"). Finite environments expose their split size and an item by position, so the
    # trainer's prompt stream can walk a seeded permutation; procedural ones return None.
    def split_size(self, split: str) -> int | None:
        return None

    def prompt_at(self, split: str, index: int) -> Prompt:
        raise NotImplementedError(f"{self.env_id} is procedural: use generate_prompt")
