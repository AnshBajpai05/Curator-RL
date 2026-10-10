"""Learning-progress estimators (Roadmap E.3, selection protocol on the simulator).

Three estimators share one observation feed (per-round posterior pass rate and
prompt count):

- LP-A: fast discounted rate minus slow discounted rate (conservative SE --
  the windows overlap, so se_f^2 + se_s^2 overstates the SE, never hides a
  real signal).
- LP-B: weighted-least-squares slope of the RAW per-round pass rate over the
  last W rounds; z = slope / SE. Two SE modes (`signals.lp_se`):
  - `design` (default since the 2026-10-10 audit, E-01): each round's rate is
    a mean over n_i prompt groups, so its variance is s_i^2 / n_i, with s_i^2
    the between-group variance of pass rates measured INSIDE that round (from
    `sum_score`/`sum_score_sq`), floored at the binomial minimum p(1-p)/G.
    Weights 1/Var, SE = 1/sqrt(sum w (x - xbar)^2). z is ~N(0, 1) on a flat
    arm (verified: SD 1.0 vs 0.28 before).
  - `residual_legacy`: the D-37 two-stage fit. Its residual variance is the
    per-PROMPT variance but was used as the per-ROUND variance, inflating the
    SE by about sqrt(n_i) once the input became raw rates. Kept only to
    reproduce decisions taken with it (D-40, D-42).
- LP-C: mean of the last W rounds minus the mean of the previous W rounds,
  independent-samples z-test. LP-C runs on the RAW per-round rate (k/n of
  that round alone), not the discounted pooled posterior: pooled windows
  overlap mechanically, which would invalidate the independent-samples
  assumption entirely (measured FPR 0.4 in validation vs <= 0.10 on raw).

Each round the engine calls `observe(env_id, posterior, n_prompts, raw_rate,
raw_var)` (which records history only when the environment actually got
prompts in that round) and then `compute(env_id, posterior)` for the
(lp, se, z) triple.
"""

from __future__ import annotations

import math

from curator_rl.signals.passrate import Posterior

SE_MODES = ("design", "residual_legacy")


class LPEstimator:
    """Selected-method LP estimator (lp_method: lp_a | lp_b | lp_c)."""

    def __init__(self, lp_method: str, window_rounds: int, se_mode: str = "design") -> None:
        if lp_method not in ("lp_a", "lp_b", "lp_c"):
            raise ValueError(f"unknown lp_method '{lp_method}'")
        if window_rounds < 1:
            raise ValueError("window_rounds must be >= 1")
        if se_mode not in SE_MODES:
            raise ValueError(f"unknown lp se_mode '{se_mode}'")
        self._method = lp_method
        self._window = window_rounds
        self._se_mode = se_mode
        # per-env history of (raw rate, n_prompts, pooled group SE, within-round
        # between-group variance or None), bounded at 2W (LP-C needs the window pair)
        self._history: dict[str, list[tuple[float, ...]]] = {}

    # ------------------------------------------------------------------ update

    def observe(self, env_id: str, posterior: Posterior, n_prompts: int,
                raw_rate: float | None = None, raw_var: float | None = None) -> None:
        if n_prompts <= 0:
            return
        if raw_rate is None:
            raw_rate = posterior.pass_rate
        history = self._history.setdefault(env_id, [])
        history.append((raw_rate, float(n_prompts), posterior.se_group,
                        -1.0 if raw_var is None else float(raw_var)))
        if len(history) > 2 * self._window:
            del history[: len(history) - 2 * self._window]

    def compute(self, env_id: str, posterior: Posterior) -> tuple[float, float, float]:
        """Return (lp, se, z) for one environment.

        A guard at insufficient history returns (0.0, 0.0, 0.0): no claim of
        progress is made while the estimator cannot decide anything.
        """
        if self._method == "lp_a":
            lp = posterior.rate_fast - posterior.rate_slow
            se = math.sqrt(posterior.se_fast**2 + posterior.se_slow**2)
            return lp, se, self._z(lp, se)

        history = self._history.get(env_id, [])
        if self._method == "lp_b":
            if len(history) < 3:
                return 0.0, 0.0, 0.0
            if self._se_mode == "residual_legacy":
                return self._slope_lp(history[-self._window :])
            return self._slope_design(history[-self._window :])
        # lp_c: non-overlapping windows, each of width W
        need = 2 * self._window
        if len(history) < need:
            return 0.0, 0.0, 0.0
        return self._windowed_means(history[-need:])

    # ------------------------------------------------------------- estimators

    @staticmethod
    def _z(lp: float, se: float) -> float:
        return lp / se if se > 0.0 else 0.0

    @staticmethod
    def _slope_design(points: list[tuple[float, ...]]) -> tuple[float, float, float]:
        """WLS slope with design-based per-round variances, x = 0..k-1 (E-01 fix).

        Var(rate_i) = s_i^2 / n_i, where s_i^2 is the round's between-group
        variance of pass rates (already floored at p(1-p)/G by the engine). A
        point without a recorded variance falls back to rate(1-rate), the
        largest variance a group pass rate can have (conservative).
        """
        x = range(len(points))
        weights = []
        for p in points:
            rate, n = p[0], max(p[1], 1.0)
            var = p[3] if len(p) > 3 and p[3] >= 0.0 else rate * (1.0 - rate)
            weights.append(n / max(var, 1e-6))
        total = sum(weights)
        mx = sum(w * xi for w, xi in zip(weights, x)) / total
        my = sum(w * p[0] for w, p in zip(weights, points)) / total
        sxx = sum(w * (xi - mx) ** 2 for w, xi in zip(weights, x))
        if sxx <= 0.0:
            return 0.0, 0.0, 0.0
        slope = sum(w * (xi - mx) * (p[0] - my) for w, xi, p in zip(weights, x, points)) / sxx
        se = 1.0 / math.sqrt(sxx)
        return slope, se, slope / se

    @staticmethod
    def _slope_lp(points: list[tuple[float, ...]]) -> tuple[float, float, float]:
        """LEGACY (`residual_legacy`, D-37): two-stage WLS over (rate, n_prompts, se) points.

        Kept to reproduce D-40/D-42. Its residual variance is the per-prompt
        variance, used as the per-round variance: the SE is ~sqrt(n) too large
        on raw per-round rates (audit E-01).

        Stage 1 fits with denominator weights to get the residual spread; stage 2
        re-weights each round by its combined variance (posterior se^2 inflated by
        the residual variance, which stands in for the cross-round autocorrelation
        of pooled discounted counts). The residual variance is divided by the
        degrees of freedom (k - 2), NOT by the weight total (which would shrink
        the SE by the prompt count and break the false-positive rate).
        """
        k = len(points)
        w = [max(p[1], 0.0) for p in points]
        total_w = sum(w)
        if total_w <= 0.0:
            return 0.0, 0.0, 0.0
        x = list(range(k))
        mean_x = sum(wi * xi for wi, xi in zip(w, x)) / total_w
        mean_y = sum(wi * p[0] for wi, p in zip(w, points)) / total_w
        sxx = sum(wi * (xi - mean_x) ** 2 for wi, xi in zip(w, x))
        if sxx <= 0.0:
            return 0.0, 0.0, 0.0

        def fit(weights: list[float]) -> tuple[float, float]:
            tw = sum(weights)
            mx = sum(wi * xi for wi, xi in zip(weights, x)) / tw
            my = sum(wi * p[0] for wi, p in zip(weights, points)) / tw
            sxx_w = sum(wi * (xi - mx) ** 2 for wi, xi in zip(weights, x))
            if sxx_w <= 0.0:
                return 0.0, mx
            slope_w = (
                sum(wi * (xi - mx) * (p[0] - my) for wi, xi, p in zip(weights, x, points)) / sxx_w
            )
            return slope_w, mx

        slope_stage1, _ = fit(w)
        resid_var = (
            sum(wi * (p[0] - (mean_y + slope_stage1 * (xi - mean_x))) ** 2
                for wi, xi, p in zip(w, x, points)) / max(k - 2, 1)
        )
        eff_var = [max(p[2], 1e-12) ** 2 + resid_var for p in points]  # se^2 + residual inflation

        slope, mx2 = fit([1.0 / v for v in eff_var])
        sxx_eff = sum((1.0 / v) * (xi - mx2) ** 2 for v, xi in zip(eff_var, x))
        se_slope = math.sqrt(1.0 / sxx_eff) if sxx_eff > 0.0 else 0.0
        z = slope / se_slope if se_slope > 0.0 else 0.0
        return slope, se_slope, z

    @staticmethod
    def _windowed_means(history: list[tuple[float, ...]]) -> tuple[float, float, float]:
        w = len(history) // 2
        recent = [p[0] for p in history[w:]]
        previous = [p[0] for p in history[:w]]
        mean_r = sum(recent) / w
        mean_p = sum(previous) / w
        var_r = sum((v - mean_r) ** 2 for v in recent) / (w - 1)
        var_p = sum((v - mean_p) ** 2 for v in previous) / (w - 1)
        se = math.sqrt(max(var_r / w + var_p / w, 0.0))
        lp = mean_r - mean_p
        return lp, se, (lp / se if se > 0.0 else 0.0)

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> dict[str, object]:
        return {"method": self._method, "se_mode": self._se_mode,
                "history": {e: list(h) for e, h in self._history.items()}}

    def load_checkpoint(self, state: dict[str, object]) -> None:
        # checkpoints written before the audit hold 3-tuples (no variance): -1.0 marks "unknown"
        self._history = {e: [tuple(p) if len(p) > 3 else (*p, -1.0) for p in h]  # type: ignore[misc]
                         for e, h in state["history"].items()}  # type: ignore[union-attr]
