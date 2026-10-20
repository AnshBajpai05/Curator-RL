"""Item-level comparisons of evaluation protocol 2 (D-101): paired differences and learning curves.

Two runs evaluated on the same items differ item by item; the SE of the mean of those differences
removes the item-difficulty variance that two independent averages carry (the dominant noise of a
300-item benchmark), so a single seed can resolve gaps of a point or two. A run's training noise
still needs several seeds: these SEs are about the evaluation, per seed.

Learning curves use the periodic reports' fixed random subset (`spread`) and the same subset of the
final evaluation, so every point of a curve, and the untrained model at zero GPU-hours, is on the
same items.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from curator_rl.trainer.calibration_eval import spread_indices
from curator_rl.trainer.run_config import BENCH_DOMAINS


def item_values(report: Mapping, kind: str) -> dict[str, dict[str, float]]:
    """{domain: {item id: value}} of a protocol-2 report row; kind is "sampled" or "greedy"."""
    out: dict[str, dict[str, float]] = {}
    for d, rec in (report.get("items") or {}).items():
        if rec.get(kind) is not None:
            out[d] = {i: float(v) for i, v in zip(rec["ids"], rec[kind])}
    return out


def paired_item_diff(a: Mapping, b: Mapping, kind: str = "sampled",
                     domains: Sequence[str] = BENCH_DOMAINS) -> tuple[float, float, dict[str, float]] | None:
    """(macro difference a - b, its SE, per-domain differences) over the items both evaluated.

    None unless both reports have per-item values for every domain. The macro SE treats domains as
    independent (different items): sqrt(sum of squared domain SEs) / number of domains.
    """
    va, vb = item_values(a, kind), item_values(b, kind)
    if any(d not in va or d not in vb for d in domains):
        return None
    per, ses = {}, []
    for d in domains:
        diffs = [va[d][i] - vb[d][i] for i in va[d] if i in vb[d]]
        if not diffs:
            return None
        m = sum(diffs) / len(diffs)
        var = sum((x - m) ** 2 for x in diffs) / max(len(diffs) - 1, 1)
        per[d] = m
        ses.append(math.sqrt(var / len(diffs)))
    return sum(per.values()) / len(per), math.sqrt(sum(x * x for x in ses)) / len(per), per


def pass_at_k(report: Mapping, domains: Sequence[str] = BENCH_DOMAINS) -> float | None:
    """Macro share of items solved at least once in the k samples.

    Coverage rather than reliability: RL raises 1-sample accuracy by sharpening onto answers the model
    could already produce, and can shrink coverage at large k (Yue et al., 2025). From the stored
    per-item sample means; None without them.
    """
    vals = item_values(report, "sampled")
    if any(d not in vals or not vals[d] for d in domains):
        return None
    return sum(sum(1.0 for v in vals[d].values() if v > 0) / len(vals[d]) for d in domains) / len(domains)


def subset_ids(final: Mapping, n: int, split: str = "dev") -> dict[str, set[str]]:
    """The item ids of the periodic subset, rebuilt from a final report (whose items are the full list)."""
    out = {}
    for d, rec in (final.get("items") or {}).items():
        ids = list(rec["ids"])
        out[d] = {ids[i] for i in spread_indices(len(ids), min(n, len(ids)), split, d)}
    return out


def _macro_on(report: Mapping, ids: Mapping[str, set[str]], kind: str,
              domains: Sequence[str] = BENCH_DOMAINS) -> float | None:
    vals = item_values(report, kind)
    means = []
    for d in domains:
        chosen = [v for i, v in vals.get(d, {}).items() if i in ids.get(d, ())]
        if not chosen:
            return None
        means.append(sum(chosen) / len(chosen))
    return sum(means) / len(means)


def learning_curve(reports: Sequence[Mapping], *, n_subset: int, base_final: Mapping | None = None,
                   kind: str = "sampled") -> list[tuple[float, float]]:
    """(charged GPU-hours, macro on the periodic subset) for a run's protocol-2 reports, starting with
    the untrained model at 0 GPU-hours when `base_final` is given. Empty without a protocol-2 final."""
    rows = [r for r in reports if r.get("protocol", 1) >= 2 and r.get("items")]
    final = next((r for r in reversed(rows) if r.get("tag") == "final"), None)
    if final is None:
        return []
    ids = subset_ids(final, n_subset)
    pts = []
    if base_final is not None and (v := _macro_on(base_final, ids, kind)) is not None:
        pts.append((0.0, v))
    for r in rows:
        if r.get("tag") != "final" and r.get("subset") != "spread":
            continue
        v = _macro_on(r, ids, kind)
        if v is not None:
            pts.append((float(r["charged_gpu_hours"]), v))
    return pts


def hours_to_reach(curve: Sequence[tuple[float, float]], target: float) -> float | None:
    """First GPU-hours at which a curve reaches `target` (linear between points); None if never."""
    for (h0, v0), (h1, v1) in zip(curve, curve[1:]):
        if v0 >= target:
            return h0
        if v1 >= target:
            return h0 + (h1 - h0) * (target - v0) / (v1 - v0)
    if curve and curve[-1][1] >= target:
        return curve[-1][0]
    return None
