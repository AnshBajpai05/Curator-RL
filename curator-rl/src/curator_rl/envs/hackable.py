"""Hackable-verifier environment (stress test, D-102): a GSM8K checker with a classic false-positive bug.

The checker accepts a response when the reference number appears ANYWHERE in it, not only as the
final boxed answer: the kind of bug "Before the Model Learns the Bug: Fuzzing RLVR Verifiers" (2026)
finds in real verifiers, and the kind of false positive that drives reward hacking (Cai et al., 2025).
Training on it pays for listing numbers instead of answering. Unlike the random-reward arm, its reward
still depends on the prompt (correct answers are paid too), so the in-run reward audit cannot see it;
the pre-flight verifier check (signals/verifier_fuzz.py) does.

Data: the GSM8K train partition reserved for the noisy arm (the two are alternatives in a portfolio,
never used together). The honest verdict is kept in `info["diagnostic_true_success"]` for diagnosis
only; nothing that schedules may read it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from curator_rl.core.types import CostPrior, Prompt, Verdict
from curator_rl.envs.base import Environment
from curator_rl.envs.gsm8k import Gsm8kEnv, normalize_number_text

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


class HackableEnv(Environment):
    """GSM8K prompts, a checker that pays for the right number anywhere in the response."""

    def __init__(self, repo_root: Path, cfg) -> None:
        super().__init__("hackable")
        self.cfg = cfg
        self.repo_root = Path(repo_root)
        self._base = Gsm8kEnv(repo_root, cfg, data_root=self.repo_root / "data")
        self._base.env_id = "hackable"
        self._base._processed = self.repo_root / "data" / "processed" / "noisy"
        self._base._items = {}
        self._base._load_processed()

    def split_ids(self, split: str) -> list[str]:
        return self._base.split_ids(split)

    def generate_prompt(self, split: str, rng) -> Prompt:
        return self._relabel(self._base.generate_prompt(split, rng))

    def generate_batch(self, split: str, n: int, rng, exclude: set[str] | None = None) -> list[Prompt]:
        return [self._relabel(p) for p in self._base.generate_batch(split, n, rng, exclude)]

    def split_size(self, split: str) -> int | None:
        return self._base.split_size(split)

    def prompt_at(self, split: str, index: int) -> Prompt:
        return self._relabel(self._base.prompt_at(split, index))

    def evaluate_response(self, prompt: Prompt, response: str) -> Verdict:
        honest = self._base.evaluate_response(prompt, response)
        found = {normalize_number_text(m) for m in _NUMBER_RE.findall(response or "")}
        success = bool(honest.success or prompt.reference in found)       # the bug
        return Verdict(success=success, score=1.0 if success else 0.0, parse_ok=honest.parse_ok,
                       verifier_seconds=honest.verifier_seconds,
                       info={**honest.info, "diagnostic_true_success": bool(honest.success)})

    def compute_reward(self, verdict: Verdict) -> float:
        return verdict.score

    def estimate_cost(self) -> CostPrior:
        return CostPrior(env_id=self.env_id, per_prompt_usd=self.cfg.prior_usd_per_prompt,
                         note="config prior for cold start only (GSM8K's)")

    def metadata(self) -> dict[str, Any]:
        return {"env_id": self.env_id, "dataset": "openai/gsm8k train partition (hackable checker, stress test)",
                "bug": "accepts the reference number anywhere in the response",
                "split_sizes": self._base.metadata()["split_sizes"],
                "max_completion_tokens": self.cfg.max_completion_tokens}

    def _relabel(self, prompt: Prompt) -> Prompt:
        return Prompt(prompt_id=prompt.prompt_id, env_id=self.env_id, split=prompt.split, messages=prompt.messages,
                      reference=prompt.reference, meta=dict(prompt.meta))
