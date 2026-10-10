"""Shared base for finite, JSONL-backed environments (Roadmap v3 Phase E).

Same contract as `Gsm8kEnv` (SPEC section 8): deterministic draws given the rng, distinct
ids within a batch, sealed test served only through `evaluation.guard`. Subclasses
implement `_to_prompt` and `evaluate_response`. `Gsm8kEnv` is deliberately left as it is:
it is committed, frozen and covered by its own contract tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from curator_rl.core.types import CostPrior, Prompt
from curator_rl.envs.base import Environment
from curator_rl.envs.splits import manifest_dir, read_manifest_ids


class FiniteJsonlEnv(Environment):
    """Finite dataset of processed JSONL rows (one file per split)."""

    dataset_label = ""

    def __init__(self, env_id: str, repo_root: Path, cfg, data_root: Path | None = None) -> None:
        super().__init__(env_id)
        self.cfg = cfg
        self.repo_root = Path(repo_root)
        self.data_root = data_root if data_root is not None else self.repo_root / "data"
        self._processed = self.data_root / "processed" / env_id
        self._items: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(self._processed.glob("*.jsonl")):
            self._items[path.stem] = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]

    # ----------------------------------------------------------------- contract
    def split_ids(self, split: str) -> list[str]:
        ids = [item["problem_id"] for item in self._items.get(split, [])]
        if split == "test" and not ids:
            ids = read_manifest_ids(self.data_root, self.env_id, "test")
        return ids

    def generate_prompt(self, split: str, rng) -> Prompt:
        self._require_split(split)
        items = self._items[split]
        return self._to_prompt(items[int(rng.integers(len(items)))], split)

    def split_size(self, split: str) -> int | None:
        self._require_split(split)
        return len(self._items[split])

    def prompt_at(self, split: str, index: int) -> Prompt:
        self._require_split(split)
        return self._to_prompt(self._items[split][index], split)

    def generate_batch(self, split: str, n: int, rng, exclude: set[str] | None = None) -> list[Prompt]:
        self._require_split(split)
        out: list[Prompt] = []
        exclude = set(exclude or ())
        seen: set[str] = set()
        tries = 0
        while len(out) < n and tries < 10 * n:
            tries += 1
            prompt = self.generate_prompt(split, rng)
            if prompt.prompt_id in exclude or prompt.prompt_id in seen:
                continue
            seen.add(prompt.prompt_id)
            out.append(prompt)
        if len(out) < n:
            raise ValueError(f"could not draw {n} distinct prompts from split '{split}'")
        return out

    def compute_reward(self, verdict) -> float:
        reward = verdict.score
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward {reward} outside [0, 1]")
        return reward

    def estimate_cost(self) -> CostPrior:
        return CostPrior(env_id=self.env_id, per_prompt_usd=self.cfg.prior_usd_per_prompt,
                         note="config prior for cold start only (Roadmap H.5)")

    def metadata(self) -> dict[str, Any]:
        hashes: dict[str, str] = {}
        mdir = manifest_dir(self.data_root)
        if mdir.exists():
            for path in sorted(mdir.glob(f"{self.env_id}_*.txt")):
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "env_id": self.env_id,
            "dataset": self.dataset_label,
            "split_sizes": {s: len(v) for s, v in sorted(self._items.items())},
            "manifest_hashes": hashes,
            "max_completion_tokens": self.cfg.max_completion_tokens,
            "max_prompt_tokens": self.cfg.max_prompt_tokens,
            "verifier_timeout_s": self.cfg.verifier_timeout_s,
        }

    # ---------------------------------------------------------------- internals
    def _require_split(self, split: str) -> None:
        if split == "test":
            raise ValueError("the sealed test split is served only through evaluation.guard")
        if not self._items.get(split):
            raise ValueError(f"split '{split}' has no processed data")

    def _to_prompt(self, item: dict[str, Any], split: str) -> Prompt:  # pragma: no cover - abstract
        raise NotImplementedError
