"""Evaluation protocol 2 (D-101): sampled accuracy, per-item results, item-paired comparisons, curves."""

from __future__ import annotations

import math
import statistics

import pytest

from curator_rl.core.types import Prompt, Verdict
from curator_rl.evaluation.paired_items import (
    hours_to_reach,
    learning_curve,
    paired_item_diff,
    subset_ids,
)
from curator_rl.trainer.calibration_eval import EvalSuite, item_seed
from curator_rl.trainer.run_config import BENCH_DOMAINS


class _Env:
    def __init__(self, n, tag):
        self.items = [Prompt(f"{tag}{k}", tag, "dev", (), str(k)) for k in range(n)]

    def split_size(self, split):
        return len(self.items)

    def prompt_at(self, split, i):
        return self.items[i]

    def evaluate_response(self, prompt, text):
        return Verdict(success=text == prompt.reference, score=0.0, parse_ok=True, verifier_seconds=0.0)


def _greedy(items, cap):
    return [p.reference if int(p.reference) % 2 == 0 else "x" for p in items]


def test_greedy_and_sampled_accuracy_per_item_with_one_seed_per_item():
    env = _Env(20, "a")
    suite = EvalSuite({"a": env}, ["a"], split="dev", items_per_domain=20, caps={})
    seeds_seen = []

    def sampled(items, cap, k, seeds):
        seeds_seen.append(list(seeds))
        return [[p.reference if j < int(p.reference) % 3 else "x" for j in range(k)] for p in items]

    r = suite.evaluate_items({"a": 20}, _greedy, sampled=sampled, k=4)
    rec = r["items"]["a"]
    assert rec["ids"] == [p.prompt_id for p in env.items]
    assert rec["greedy"] == [1 - k % 2 for k in range(20)]
    assert rec["sampled"] == [(k % 3) / 4 for k in range(20)]
    assert r["sampled_by_domain"]["a"] == pytest.approx(sum((k % 3) / 4 for k in range(20)) / 20)
    assert r["score_by_domain"]["a"] == 0.5
    # the seed belongs to the item: every method, run and model draws its samples with the same seeds
    assert seeds_seen[0] == [item_seed("a", p.prompt_id) for p in env.items]
    assert len(set(seeds_seen[0])) == 20


def test_a_failing_sampler_costs_only_the_sampled_score():
    suite = EvalSuite({"a": _Env(6, "a")}, ["a"], split="dev", items_per_domain=6, caps={})

    def broken(items, cap, k, seeds):
        raise RuntimeError("vLLM died")

    r = suite.evaluate_items({"a": 6}, _greedy, sampled=broken, k=4, retries=1)
    assert r["score_by_domain"]["a"] == 0.5 and "a" not in r["sampled_by_domain"]
    assert "a:sampled" in r["errors"] and "sampled" not in r["items"]["a"]


def test_the_periodic_subset_is_a_fixed_random_part_of_the_final_items():
    env = _Env(300, "a")
    one, two = (EvalSuite({"a": env}, ["a"], split="dev", items_per_domain=300, caps={}) for _ in range(2))
    sub = one.subset("a", 100, spread=True)
    assert [p.prompt_id for p in sub] == [p.prompt_id for p in two.subset("a", 100, spread=True)]
    refs = [int(p.reference) for p in sub]
    assert len(sub) == 100 and refs != list(range(100))          # not the first 100 (ordered datasets)
    assert min(refs) < 30 and max(refs) > 270
    # the analysis rebuilds the same subset from a final report's item list
    final = {"items": {"a": {"ids": [p.prompt_id for p in one.items["a"]]}}}
    assert subset_ids(final, 100)["a"] == {p.prompt_id for p in sub}


def _report(values: dict[str, list[float]], kind: str = "sampled", **extra) -> dict:
    return {"items": {d: {"ids": [f"{d}{i}" for i in range(len(v))], kind: v} for d, v in values.items()}, **extra}


def test_paired_item_differences_and_their_standard_error():
    a = _report({"x": [1, 1, 0, 1], "y": [0.5, 0.5, 0.5, 0.5]})
    b = _report({"x": [1, 0, 0, 1], "y": [0.25, 0.5, 0.75, 0.5]})
    macro, se, per = paired_item_diff(a, b, "sampled", domains=("x", "y"))
    assert per == {"x": 0.25, "y": 0.0} and macro == 0.125
    se_x = statistics.stdev([0, 1, 0, 0]) / 2
    se_y = statistics.stdev([0.25, 0, -0.25, 0]) / 2
    assert se == pytest.approx(math.sqrt(se_x ** 2 + se_y ** 2) / 2)
    assert paired_item_diff(a, _report({"x": [1, 0, 0, 1]}), "sampled", domains=("x", "y")) is None


def test_learning_curve_is_on_the_same_items_and_starts_at_the_untrained_model():
    n, n_sub = 10, 4
    full = {d: [f"{d}{i}" for i in range(n)] for d in BENCH_DOMAINS}
    probe = {"items": {d: {"ids": ids} for d, ids in full.items()}}
    sub = subset_ids(probe, n_sub)

    def row(on_subset, off_subset, **extra):
        items = {d: {"ids": ids, "sampled": [on_subset if i in sub[d] else off_subset for i in ids]}
                 for d, ids in full.items()}
        return {"protocol": 2, "items": items, **extra}

    base = row(0.1, 0.9, tag="final")
    periodic = row(0.3, 0.3, tag="periodic", subset="spread", charged_gpu_hours=0.5)
    periodic["items"] = {d: {"ids": sorted(sub[d]), "sampled": [0.3] * n_sub} for d in full}
    final = row(0.5, 0.0, tag="final", charged_gpu_hours=1.0)
    curve = learning_curve([periodic, final], n_subset=n_sub, base_final=base)
    assert curve == [(0.0, pytest.approx(0.1)), (0.5, pytest.approx(0.3)), (1.0, pytest.approx(0.5))]
    assert learning_curve([periodic], n_subset=n_sub) == []          # no protocol-2 final yet


def test_hours_to_reach_a_target():
    curve = [(0.0, 0.2), (1.0, 0.3), (2.0, 0.5)]
    assert hours_to_reach(curve, 0.4) == pytest.approx(1.5)
    assert hours_to_reach(curve, 0.1) == 0.0
    assert hours_to_reach(curve, 0.6) is None
