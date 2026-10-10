"""Paired calibration and reporting evaluations on real models (Phase F/G; D-72 on hardware).

Calibration (charged to the methods that use it): each slice holds a FIXED set of items from the
`calib` split, chosen with a run-independent seed so every method and every seed is scored on the
same items. The model answers greedily; the change of a slice between two evaluations has the
paired (McNemar) standard error, computed from the items whose verdict flipped.

Reporting (uncharged, identical for every method): the same mechanism on the `dev` split, giving
the score-vs-GPU-hours curve and the final score (macro-average over BENCH_DOMAINS). The sealed
test split is never opened here; it is read once, at the very end, through `evaluation.guard`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from curator_rl.core.types import Prompt

ITEM_SEED = 20261010          # fixed: all methods and seeds are scored on the same items
GenerateFn = Callable[[Sequence[Prompt], int], list[str]]


def fixed_items(env, split: str, n: int, tag: str) -> list[Prompt]:
    """n deterministic items of a split (all of them when the split is smaller)."""
    key = int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16)
    size = env.split_size(split)
    if size is None:                                   # procedural (Countdown): fixed seed stream
        return env.generate_batch(split, n, np.random.default_rng([ITEM_SEED, key]))
    order = np.random.default_rng([ITEM_SEED, key]).permutation(size)[: min(n, size)]
    return [env.prompt_at(split, int(i)) for i in sorted(order)]


def paired_se(prev: Sequence[bool], cur: Sequence[bool]) -> float:
    """McNemar SE of the change in accuracy on the same items, floored at one discordant item."""
    n = len(cur)
    n01 = sum(1 for a, b in zip(prev, cur) if not a and b)
    n10 = sum(1 for a, b in zip(prev, cur) if a and not b)
    return math.sqrt(max(n01 + n10 - (n01 - n10) ** 2 / n, 1.0)) / n


class EvalSuite:
    """Fixed items per domain for one split; remembers the last verdicts for paired SEs."""

    def __init__(self, envs: Mapping[str, object], domains: Sequence[str], *, split: str,
                 items_per_domain: int, caps: Mapping[str, int], workers: int = 4) -> None:
        self.envs = dict(envs)
        self.split = split
        self.caps = dict(caps)
        self.items = {d: fixed_items(self.envs[d], split, items_per_domain, f"{split}:{d}") for d in domains}
        self._last: dict[str, list[bool]] = {}
        self._pool = ThreadPoolExecutor(max_workers=max(int(workers), 1))

    def evaluate(self, request: Mapping[str, int], generate: GenerateFn) -> dict:
        """Score the requested domains: {domain: n_items}. Returns scores, SEs and paired SEs."""
        scores, ses, dses, n_total = {}, {}, {}, 0
        for d, n in request.items():
            items = self.items[d][: max(int(n), 1)]
            texts = generate(items, self.caps.get(d, 512))
            env = self.envs[d]
            verdicts = list(self._pool.map(lambda pt, env=env: env.evaluate_response(pt[0], pt[1]),
                                           zip(items, texts)))
            correct = [bool(v.success) for v in verdicts]
            s = sum(correct) / len(correct)
            scores[d], ses[d] = s, math.sqrt(max(s * (1 - s), 1e-6) / len(correct))
            prev = self._last.get(d)
            if prev is not None and len(prev) == len(correct):
                dses[d] = paired_se(prev, correct)
            self._last[d] = correct
            n_total += len(correct)
        return {"score_by_domain": scores, "se_by_domain": ses, "delta_se_by_domain": dses or None,
                "n_items": n_total}

    def get_state(self) -> dict:
        return {"last": {d: list(v) for d, v in self._last.items()}}

    def load_state(self, state: Mapping) -> None:
        self._last = {d: [bool(x) for x in v] for d, v in state.get("last", {}).items()}
