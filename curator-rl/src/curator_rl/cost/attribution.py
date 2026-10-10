"""Per-environment cost attribution of a mixed GRPO step (COST_MODEL section 3; audit A3).

A step's wall time has two parts:
- SCORE (verification): measured exactly per completion, split by each arm's serial verifier
  seconds (the thread pool's wall time is shared in proportion to the work it did);
- GEN + UPDATE (everything else): split by each arm's share of the step's tokens (prompt +
  completions). This is the token-proportional assumption (refit bracket R1).

The assumption is checked, not trusted: `fit_step_cost_model` regresses the GEN + UPDATE seconds
of many steps on per-arm token counts plus a fixed per-step overhead (non-negative least
squares). With mixtures that vary from step to step, which the system-identification run makes
them do on purpose, the per-arm coefficients are identified from ONE run, without timing every
environment alone (Gate 5's "exact single-environment timing").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import nnls


def attribute_step(step_seconds: float, score_wall_seconds: float,
                   verifier_seconds: Mapping[str, float], tokens: Mapping[str, float]) -> dict[str, float]:
    """Seconds of one step charged to each arm (sums to `step_seconds`)."""
    envs = sorted(set(verifier_seconds) | set(tokens))
    score = min(max(score_wall_seconds, 0.0), max(step_seconds, 0.0))
    rest = max(step_seconds - score, 0.0)
    v_total = sum(max(verifier_seconds.get(e, 0.0), 0.0) for e in envs)
    t_total = sum(max(tokens.get(e, 0.0), 0.0) for e in envs)
    out = {}
    for e in envs:
        s = score * (verifier_seconds.get(e, 0.0) / v_total if v_total > 0 else 0.0)
        g = rest * (tokens.get(e, 0.0) / t_total if t_total > 0 else 1.0 / len(envs))
        out[e] = s + g
    if v_total <= 0 and score > 0:                      # nothing verified: spread SCORE by tokens
        for e in envs:
            out[e] += score * (tokens.get(e, 0.0) / t_total if t_total > 0 else 1.0 / len(envs))
    return out


def fit_step_cost_model(steps: Sequence[Mapping]) -> dict:
    """NNLS fit of GEN+UPDATE seconds = c0 + sum_e a_e * tokens_e over logged steps.

    Each step dict needs `gen_update_seconds` and `tokens` ({env: tokens}). Returns per-arm
    seconds per 1k tokens, the fixed per-step overhead, R^2 and the number of steps used.
    """
    envs = sorted({e for s in steps for e in s["tokens"]})
    if len(steps) < len(envs) + 2:
        raise ValueError("too few steps to identify a per-arm cost model")
    x = np.array([[1.0] + [s["tokens"].get(e, 0.0) / 1000.0 for e in envs] for s in steps])
    y = np.array([s["gen_update_seconds"] for s in steps], dtype=float)
    coef, _ = nnls(x, y)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    return {"overhead_seconds_per_step": float(coef[0]),
            "seconds_per_1k_tokens": {e: float(c) for e, c in zip(envs, coef[1:])},
            "r2": 1.0 - ss_res / ss_tot, "n_steps": len(steps)}
