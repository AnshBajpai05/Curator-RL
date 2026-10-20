"""Paired calibration and reporting evaluations on real models (Phase F/G; D-72 on hardware).

Calibration (charged to the methods that use it): each slice holds a FIXED set of items from the
`calib` split, chosen with a run-independent seed so every method and every seed is scored on the
same items. The model answers greedily; the change of a slice between two evaluations has the
paired (McNemar) standard error, computed from the items whose verdict flipped.

Reporting (uncharged, identical for every method): the same mechanism on the `dev` split, giving
the score-vs-GPU-hours curve and the final score (macro-average over BENCH_DOMAINS). The sealed
test split is never opened here; it is read once, at the very end, through `evaluation.guard`.

Evaluation protocol 2 (D-101) adds, for reports: SAMPLED accuracy (the mean over k samples per item
at the training temperature; what RL improves first, while greedy accuracy moves little early on),
one random seed per item derived from the item alone (every method and the base model draw with the
same seeds), and per-item results, so methods are compared item by item (paired) instead of through
two noisy averages. Periodic reports use a fixed random subset of the final items (`spread`), so the
learning curve and the final evaluation are on the same items.
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
# (items, token cap, samples per item, one seed per item) -> k completions per item
SampledFn = Callable[[Sequence[Prompt], int, int, Sequence[int]], list[list[str]]]


def item_seed(domain: str, prompt_id: str) -> int:
    """The sampling seed of one evaluation item: the same for every method, run and model."""
    return int(hashlib.sha256(f"{ITEM_SEED}:{domain}:{prompt_id}".encode()).hexdigest()[:8], 16)


def fixed_items(env, split: str, n: int, tag: str) -> list[Prompt]:
    """n deterministic items of a split (all of them when the split is smaller)."""
    key = int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16)
    size = env.split_size(split)
    if size is None:                                   # procedural (Countdown): fixed seed stream
        return env.generate_batch(split, n, np.random.default_rng([ITEM_SEED, key]))
    order = np.random.default_rng([ITEM_SEED, key]).permutation(size)[: min(n, size)]
    return [env.prompt_at(split, int(i)) for i in sorted(order)]


def spread_indices(n_total: int, n: int, split: str, domain: str) -> list[int]:
    """Positions of a fixed random n of a domain's n_total items (sorted): the periodic-report subset."""
    key = int(hashlib.sha256(f"spread:{split}:{domain}".encode()).hexdigest()[:8], 16)
    return sorted(int(i) for i in np.random.default_rng([ITEM_SEED, key]).permutation(n_total)[:n])


def paired_se(prev: Sequence[bool], cur: Sequence[bool]) -> float:
    """McNemar SE of the change in accuracy on the same items, floored at one discordant item."""
    n = len(cur)
    n01 = sum(1 for a, b in zip(prev, cur) if not a and b)
    n10 = sum(1 for a, b in zip(prev, cur) if a and not b)
    return math.sqrt(max(n01 + n10 - (n01 - n10) ** 2 / n, 1.0)) / n


def _safe_verdict(env, prompt: Prompt, text: str) -> bool:
    """One item's correctness; a verifier that raises counts the answer as wrong."""
    try:
        return bool(env.evaluate_response(prompt, text).success)
    except Exception:  # noqa: BLE001
        return False


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

    def evaluate(self, request: Mapping[str, int], generate: GenerateFn, *, retries: int = 1) -> dict:
        """Score the requested domains: {domain: n_items}. Returns scores, SEs and paired SEs.

        A domain whose generation or scoring raises is retried `retries` times, then left out and
        listed under "errors" (the caller decides what a missing domain means). Paired-item state
        is committed only for domains that succeeded, so a failed evaluation never corrupts the
        next paired SE.
        """
        scores, ses, dses, n_total, errors = {}, {}, {}, 0, {}
        for d, n in request.items():
            items = self.items[d][: max(int(n), 1)]
            env = self.envs[d]
            correct = None
            for attempt in range(retries + 1):
                try:
                    texts = generate(items, self.caps.get(d, 512))
                    verdicts = list(self._pool.map(lambda pt, env=env: _safe_verdict(env, pt[0], pt[1]),
                                                   zip(items, texts)))
                    correct = [bool(v) for v in verdicts]
                    break
                except Exception as exc:  # noqa: BLE001 - reported; the run goes on
                    errors[d] = f"attempt {attempt + 1}: {exc!r}"[:300]
                    print(f"[eval] {self.split}/{d} failed ({errors[d]})", flush=True)
            if correct is None:
                continue
            errors.pop(d, None)
            s = sum(correct) / len(correct)
            scores[d], ses[d] = s, math.sqrt(max(s * (1 - s), 1e-6) / len(correct))
            prev = self._last.get(d)
            if prev is not None and len(prev) == len(correct):
                dses[d] = paired_se(prev, correct)
            self._last[d] = correct
            n_total += len(correct)
        return {"score_by_domain": scores, "se_by_domain": ses, "delta_se_by_domain": dses or None,
                "n_items": n_total, "errors": errors}

    def subset(self, domain: str, n: int, *, spread: bool = False) -> list[Prompt]:
        """The first n items, or (spread) a fixed random n of them, in item order.

        The item lists are sorted by dataset index, so the first n are not a random sample of the n
        a final evaluation uses (datasets can be ordered, e.g. by difficulty); `spread` is.
        """
        items = self.items[domain]
        n = min(max(int(n), 1), len(items))
        if not spread or n == len(items):
            return items[:n]
        return [items[i] for i in spread_indices(len(items), n, self.split, domain)]

    def evaluate_items(self, request: Mapping[str, int], generate: GenerateFn, *,
                       sampled: SampledFn | None = None, k: int = 0, spread: bool = False,
                       retries: int = 1) -> dict:
        """Protocol 2: greedy and (if `sampled` and k > 0) sampled accuracy, with per-item results.

        Same contract as `evaluate` for the greedy part (scores, SEs, paired state). Adds
        "sampled_by_domain" / "sampled_se_by_domain" (mean over items of the per-item mean of k
        samples; SE over items) and "items": {domain: {"ids", "greedy", "sampled"}}.
        """
        scores, ses, dses, n_total, errors = {}, {}, {}, 0, {}
        s_scores, s_ses, items_out = {}, {}, {}
        for d, n in request.items():
            items = self.subset(d, n, spread=spread)
            env = self.envs[d]
            cap = self.caps.get(d, 512)
            correct = None
            for attempt in range(retries + 1):
                try:
                    texts = generate(items, cap)
                    correct = [bool(v) for v in self._pool.map(
                        lambda pt, env=env: _safe_verdict(env, pt[0], pt[1]), zip(items, texts))]
                    break
                except Exception as exc:  # noqa: BLE001 - reported; the run goes on
                    errors[d] = f"attempt {attempt + 1}: {exc!r}"[:300]
                    print(f"[eval] {self.split}/{d} failed ({errors[d]})", flush=True)
            if correct is None:
                continue
            errors.pop(d, None)
            # sampled accuracy separately: a failure there costs the sampled score, never the greedy one
            means = (self._sampled_means(d, items, sampled, k, retries, errors)
                     if sampled is not None and k > 0 else None)
            s = sum(correct) / len(correct)
            scores[d], ses[d] = s, math.sqrt(max(s * (1 - s), 1e-6) / len(correct))
            prev = self._last.get(d)
            if prev is not None and len(prev) == len(correct):
                dses[d] = paired_se(prev, correct)
            self._last[d] = correct
            n_total += len(correct)
            items_out[d] = {"ids": [it.prompt_id for it in items], "greedy": [int(c) for c in correct]}
            if means is not None:
                m = sum(means) / len(means)
                var = sum((x - m) ** 2 for x in means) / max(len(means) - 1, 1)
                s_scores[d], s_ses[d] = m, math.sqrt(var / len(means))
                items_out[d]["sampled"] = [round(x, 4) for x in means]
        return {"score_by_domain": scores, "se_by_domain": ses, "delta_se_by_domain": dses or None,
                "n_items": n_total, "errors": errors, "sampled_by_domain": s_scores,
                "sampled_se_by_domain": s_ses, "items": items_out}

    def _sampled_means(self, d: str, items: list[Prompt], sampled: SampledFn, k: int, retries: int,
                       errors: dict) -> list[float] | None:
        """Per-item mean correctness over k samples (each item with its own seed); None after failures."""
        env = self.envs[d]
        for attempt in range(retries + 1):
            try:
                groups = sampled(items, self.caps.get(d, 512), k, [item_seed(d, it.prompt_id) for it in items])
                if len(groups) != len(items) or any(len(g) != k for g in groups):
                    raise RuntimeError(f"sampled generation returned {len(groups)} groups for {len(items)} items "
                                       f"(k = {k})")
                flat = [(it, t) for it, g in zip(items, groups) for t in g]
                ok = list(self._pool.map(lambda pt, env=env: _safe_verdict(env, pt[0], pt[1]), flat))
                errors.pop(f"{d}:sampled", None)
                return [sum(ok[i * k:(i + 1) * k]) / k for i in range(len(items))]
            except Exception as exc:  # noqa: BLE001 - reported; the run goes on
                errors[f"{d}:sampled"] = f"attempt {attempt + 1}: {exc!r}"[:300]
                print(f"[eval] {self.split}/{d} sampled failed ({errors[f'{d}:sampled']})", flush=True)
        return None

    def get_state(self) -> dict:
        return {"last": {d: list(v) for d, v in self._last.items()}}

    def load_state(self, state: Mapping) -> None:
        self._last = {d: [bool(x) for x in v] for d, v in state.get("last", {}).items()}
