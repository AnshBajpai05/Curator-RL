"""ROI leaderboard: benchmark gain per GPU-hour of each training arm (Roadmap Part I/J, improvement D3).

From a run's calibration evaluations (paired items, possibly targeted: a slice need not be evaluated
every window) the per-slice gain model is fitted

    gain_{d,k} = sum_j W_{j,k} B_{dj} + noise(SE_{d,k}),

where gain_{d,k} is slice d's change since ITS previous evaluation and W_{j,k} the share of the
training prompts arm j received over that same span (exposure accumulated per slice, so targeted
calibration is handled exactly). Weighted ridge regression per slice (weights 1/SE^2, prior
B ~ N(0, TRANSFER_PRIOR_SD^2) as in calibration/credit.py) gives B_{dj} with posterior SDs.

Two structures. `full` estimates the whole transfer matrix (every arm on every slice): it needs every
slice evaluated often under varied mixtures, i.e. the sysid design. `members` (default for an
in-run leaderboard) credits each arm only on the slice it is scored on, jointly with the arms that
share that slice (the calibrator's own C2d credit): 1-2 coefficients per slice instead of N. In the
simulator at the Level-1 budget the full matrix ranked arms almost backwards (too few spans per
slice for N unknowns; reports/analysis/roi_loo_sim.md), which is why `members` is the default.

An arm's value is its effect on the BENCHMARK (macro-average over slices with weights pi_d):
g_j = sum_d pi_d B_{dj} per window of full compute share, so transfer counts. Dividing by what a
window of full share costs when spent on arm j (prompts per window x GPU-hours per prompt of j)
gives ROI_j in benchmark points per GPU-hour. Intervals: the analytic posterior (primary), plus an
optional bootstrap over calibration spans (flagged when there are fewer than 20 spans, where a
percentile bootstrap undercovers; audit E-11).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from curator_rl.calibration.credit import TRANSFER_PRIOR_SD

_MIN_SE = 1e-6
Z95 = 1.959963984540054


@dataclass(frozen=True)
class SpanRow:
    domain: str
    shares: np.ndarray          # per arm: sum over the span's windows of that window's share (W_{j,k})
    gain: float
    se: float


def span_rows(calib_rows: Sequence[Mapping], env_ids: Sequence[str]) -> list[SpanRow]:
    """Turn calibration records into regression rows (exposure accumulated per slice between evaluations).

    Each record needs `scores` {slice: accuracy}, `exposure` {arm: prompts since the previous
    calibration} and optionally `delta_se` {slice: paired SE of the change}. Records without scores
    (logged failures) only add their exposure.
    """
    env_ids = list(env_ids)
    pending: dict[str, np.ndarray] = {}
    last: dict[str, float] = {}
    last_se: dict[str, float] = {}
    out: list[SpanRow] = []
    for row in calib_rows:
        exposure = np.array([float((row.get("exposure") or {}).get(e, 0.0)) for e in env_ids])
        # a window's compute shares; a span of several windows sums them (as the calibrator's credit does),
        # so the regressor grows with the span's length exactly as the gain does
        window_share = exposure / exposure.sum() if exposure.sum() > 0 else np.zeros(len(env_ids))
        for d in pending:
            pending[d] = pending[d] + window_share
        scores = row.get("scores") or {}
        ses = row.get("se") or {}
        paired = row.get("delta_se") or {}
        for d, s in scores.items():
            span = pending.get(d, window_share.copy())
            if d in last and span.sum() > 0:
                se = paired.get(d)
                if se is None:
                    se = math.hypot(float(ses.get(d, 0.0)), last_se.get(d, 0.0))
                out.append(SpanRow(d, span.copy(), float(s) - last[d], max(float(se), _MIN_SE)))
            last[d], last_se[d] = float(s), float(ses.get(d, 0.0))
            pending[d] = np.zeros(len(env_ids))
        for d in list(pending):
            if d not in scores and d not in last:
                pending.pop(d)          # never evaluated yet: its first evaluation is the baseline
    return out


def fit_slices(rows: Sequence[SpanRow], env_ids: Sequence[str],
               domain_of: Mapping[str, str | None] | None = None) -> dict[str, dict]:
    """Per-slice weighted ridge: {slice: {"mean": (n_arms,), "cov": (n_arms, n_arms), "spans": k}}.

    With `domain_of` (structure `members`) slice d is regressed only on the arms scored on d; every
    other coefficient of that slice is fixed at 0 with zero variance.
    """
    env_ids = list(env_ids)
    n = len(env_ids)
    out = {}
    for d in sorted({r.domain for r in rows}):
        cols = (list(range(n)) if domain_of is None
                else [i for i, e in enumerate(env_ids) if domain_of.get(e, e) == d])
        mean, cov = np.zeros(n), np.zeros((n, n))
        rs = [r for r in rows if r.domain == d]
        if cols:
            x = np.array([r.shares[cols] for r in rs])
            y = np.array([r.gain for r in rs])
            w = np.array([1.0 / r.se**2 for r in rs])
            precision = (x * w[:, None]).T @ x + np.eye(len(cols)) / TRANSFER_PRIOR_SD**2
            c = np.linalg.inv(precision)
            mean[cols] = c @ ((x * w[:, None]).T @ y)
            cov[np.ix_(cols, cols)] = c
        out[d] = {"mean": mean, "cov": cov, "spans": len(rs)}
    return out


@dataclass(frozen=True)
class RoiRow:
    env_id: str
    gain_per_share: float       # g_j: benchmark gain per window of full compute share
    gain_sd: float
    roi: float                  # benchmark points (fraction) per GPU-hour spent on the arm
    roi_lo: float
    roi_hi: float
    gpu_hours_per_prompt: float


def roi_table(calib_rows: Sequence[Mapping], env_ids: Sequence[str], gpu_hours_per_prompt: Mapping[str, float],
              bench_weights: Mapping[str, float], *, domain_of: Mapping[str, str | None] | None = None,
              structure: str = "members") -> list[RoiRow]:
    """ROI per arm, ranked best first (see module docstring). `members` uses `domain_of` (default: own id)."""
    env_ids = list(env_ids)
    rows = span_rows(calib_rows, env_ids)
    fits = _fit(rows, env_ids, domain_of, structure)
    exposure_total = sum(sum(float(v) for v in (r.get("exposure") or {}).values()) for r in calib_rows)
    n_windows = max(sum(1 for r in calib_rows if r.get("scores")), 1)
    prompts_per_window = exposure_total / n_windows if exposure_total > 0 else 1.0
    total_w = sum(bench_weights.get(d, 0.0) for d in fits) or 1.0
    table = []
    for i, e in enumerate(env_ids):
        g = sum(bench_weights.get(d, 0.0) / total_w * f["mean"][i] for d, f in fits.items())
        var = sum((bench_weights.get(d, 0.0) / total_w) ** 2 * f["cov"][i, i] for d, f in fits.items())
        sd = math.sqrt(max(var, 0.0))
        c = float(gpu_hours_per_prompt.get(e) or float("nan"))
        scale = prompts_per_window * c
        if not math.isfinite(scale) or scale <= 0:
            table.append(RoiRow(e, g, sd, float("nan"), float("nan"), float("nan"), c))
            continue
        table.append(RoiRow(e, g, sd, g / scale, (g - Z95 * sd) / scale, (g + Z95 * sd) / scale, c))
    return sorted(table, key=lambda r: (-(r.roi if math.isfinite(r.roi) else -math.inf), r.env_id))


def _fit(rows, env_ids, domain_of, structure):
    if structure == "members":
        return fit_slices(rows, env_ids, domain_of or {e: e for e in env_ids})
    if structure == "full":
        return fit_slices(rows, env_ids)
    raise ValueError(f"structure must be members|full, got {structure!r}")


def bootstrap_roi(calib_rows: Sequence[Mapping], env_ids: Sequence[str], gpu_hours_per_prompt: Mapping[str, float],
                  bench_weights: Mapping[str, float], *, domain_of: Mapping[str, str | None] | None = None,
                  structure: str = "members", n_boot: int = 2000, seed: int = 0) -> dict[str, tuple]:
    """Percentile intervals from resampling calibration spans; returns {arm: (lo, hi, n_spans)}."""
    env_ids = list(env_ids)
    rows = span_rows(calib_rows, env_ids)
    if not rows:
        return {e: (float("nan"), float("nan"), 0) for e in env_ids}
    exposure_total = sum(sum(float(v) for v in (r.get("exposure") or {}).values()) for r in calib_rows)
    n_windows = max(sum(1 for r in calib_rows if r.get("scores")), 1)
    ppw = exposure_total / n_windows
    rng = np.random.default_rng(seed)
    draws = {e: [] for e in env_ids}
    for _ in range(n_boot):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        fits = _fit(sample, env_ids, domain_of, structure)
        total_w = sum(bench_weights.get(d, 0.0) for d in fits) or 1.0
        for i, e in enumerate(env_ids):
            g = sum(bench_weights.get(d, 0.0) / total_w * f["mean"][i] for d, f in fits.items())
            c = gpu_hours_per_prompt.get(e)
            if c:
                draws[e].append(g / (ppw * c))
    return {e: ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(rows)) if v
                else (float("nan"), float("nan"), len(rows))) for e, v in draws.items()}


def render(table: Sequence[RoiRow], *, bootstrap: Mapping[str, tuple] | None = None) -> str:
    lines = ["| rank | arm | benchmark gain per window of full share (SD) | GPU-h / prompt | "
             "ROI (benchmark points per GPU-hour) | 95% interval |" + (" bootstrap 95% |" if bootstrap else ""),
             "|---|---|---|---|---|---|" + ("---|" if bootstrap else "")]
    for k, r in enumerate(table, 1):
        roi = "-" if not math.isfinite(r.roi) else f"{100 * r.roi:+.2f}"
        ci = "-" if not math.isfinite(r.roi) else f"[{100 * r.roi_lo:+.2f}, {100 * r.roi_hi:+.2f}]"
        extra = ""
        if bootstrap:
            lo, hi, n = bootstrap.get(r.env_id, (float("nan"), float("nan"), 0))
            extra = f" [{100 * lo:+.2f}, {100 * hi:+.2f}]{' (n<20: undercovers)' if n < 20 else ''} |"
        lines.append(f"| {k} | {r.env_id} | {r.gain_per_share:+.4f} ({r.gain_sd:.4f}) | {r.gpu_hours_per_prompt:.2e} | "
                     f"{roi} | {ci} |{extra}")
    return "\n".join(lines) + "\n"
