"""Noisy-reward environment (sanity arm, Roadmap D.2).

A wrapper over the GSM8K train partition. Verification is the honest GSM8K
check; noise is applied ONLY in `compute_reward` (reward shaping lives here,
separate from verification).

Modes (config `data.envs.noisy`):
- `random`: reward ~ Bernoulli(q), statistically independent of the true
  outcome - the misalignment case the calibration step must catch.
- `flip`: the true reward is flipped with probability flip_p.

Draws are seeded by sha256 of (mode salt | prompt_id | rollout index | response),
so rewards are reproducible while remaining statistically independent of
correctness for `random` mode (a uniform hash is independent of the true
outcome) and independent across the rollouts of a group (audit E-24).

The true success is recorded in `verdict.info["diagnostic_true_success"]`
for diagnosis; the Signal Engine must never read it (Roadmap D.2).
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

from curator_rl.core.types import CostPrior, Prompt, Verdict
from curator_rl.envs.base import Environment
from curator_rl.envs.gsm8k import Gsm8kEnv

_MODE_SALTS = {
    "random": "curator-noisy-random-v1",
    "flip": "curator-noisy-flip-v1",
}


def _uniform_unit(salt: str, key: str) -> float:
    digest = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    return struct.unpack("<I", digest[:4])[0] / 2**32


class NoisyRewardEnv(Environment):
    """Noisy sanity wrapper: verification honest, reward corrupted by design."""

    def __init__(self, repo_root: Path, cfg) -> None:
        if cfg.mode not in _MODE_SALTS:
            raise ValueError(f"noisy mode '{cfg.mode}' must be one of {sorted(_MODE_SALTS)}")
        super().__init__("noisy")
        self.cfg = cfg
        self.mode = cfg.mode
        self.repo_root = Path(repo_root)
        self._base = Gsm8kEnv(repo_root, cfg, data_root=self.repo_root / "data")
        self._base.env_id = "noisy"
        self._base._processed = self.repo_root / "data" / "processed" / "noisy"
        self._base._items = {}
        self._base._load_processed()

    # --- serving: identical to the base data (train partition only) --------
    def split_ids(self, split: str) -> list[str]:
        return self._base.split_ids(split)

    def generate_prompt(self, split: str, rng) -> Prompt:
        prompt = self._base.generate_prompt(split, rng)
        return self._relabel(prompt)

    def generate_batch(self, split: str, n: int, rng, exclude: set[str] | None = None) -> list[Prompt]:
        return [self._relabel(p) for p in self._base.generate_batch(split, n, rng, exclude)]

    def split_size(self, split: str) -> int | None:
        return self._base.split_size(split)

    def prompt_at(self, split: str, index: int) -> Prompt:
        return self._relabel(self._base.prompt_at(split, index))

    def evaluate_response(self, prompt: Prompt, response: str) -> Verdict:
        verdict = self._base.evaluate_response(prompt, response)
        # Key by (prompt_id, rollout index, response): reproducible, and independent per rollout.
        # Without the index, identical completions of one prompt got identical rewards (audit E-24):
        # as a trained policy collapses onto one answer per prompt, the "random" reward became a
        # fixed per-prompt value. The marginal probability stays q for every response (Roadmap D.2).
        # Callers that pass no rollout index (pre-E-24) keep the old key.
        index = prompt.meta.get("rollout_index")
        key = (f"{prompt.prompt_id}|{response}" if index is None
               else f"{prompt.prompt_id}|r{int(index)}|{response}")
        return Verdict(
            success=verdict.success,
            score=verdict.score,
            parse_ok=verdict.parse_ok,
            verifier_seconds=verdict.verifier_seconds,
            info={
                **verdict.info,
                "diagnostic_true_success": bool(verdict.success),
                "noise_key": key,
            },
        )

    # --- the actual noise, reward shaping only -----------------------------
    def compute_reward(self, verdict: Verdict) -> float:
        true_success = bool(verdict.info.get("diagnostic_true_success", verdict.success))
        key = verdict.info.get("noise_key", str(verdict.success))
        if self.mode == "random":
            noisy = _uniform_unit(_MODE_SALTS["random"], key) < self.cfg.q
        else:  # flip
            flip = _uniform_unit(_MODE_SALTS["flip"], key) < self.cfg.flip_p
            noisy = (not true_success) if flip else true_success
        return 1.0 if noisy else 0.0

    def estimate_cost(self) -> CostPrior:
        return CostPrior(
            env_id=self.env_id,
            per_prompt_usd=self.cfg.prior_usd_per_prompt,
            note="config prior for cold start only (Roadmap H.5)",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "dataset": "openai/gsm8k train partition (noisy reward wrapper)",
            "mode": self.mode,
            "q": self.cfg.q,
            "flip_p": self.cfg.flip_p,
            "split_sizes": self._base.metadata()["split_sizes"],
            "max_completion_tokens": self.cfg.max_completion_tokens,
            "max_prompt_tokens": self.cfg.max_prompt_tokens,
            "verifier_timeout_s": self.cfg.verifier_timeout_s,
        }

    def _relabel(self, prompt: Prompt) -> Prompt:
        return Prompt(
            prompt_id=prompt.prompt_id,
            env_id=self.env_id,
            split=prompt.split,
            messages=prompt.messages,
            reference=prompt.reference,
            meta={**prompt.meta, "noise_key": f"{prompt.prompt_id}"},
        )
