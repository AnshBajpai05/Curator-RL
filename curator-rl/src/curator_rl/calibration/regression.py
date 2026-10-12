"""Proxy weights fitted to calibrated gains (Roadmap Part I, improvement B4).

The proxy is x_j = alpha * LP'_j + beta * SR_j. Its two weights were fixed at 0.5 by hand, and the
richness term SR rewards any arm with many mixed groups, including a random-reward arm (its SR is
the highest of all, while its benchmark gain is zero). Calibration measures each arm's gain per
window of compute share, b_j, with a standard error. This module fits

    b_j = a * LP'_j + b * SR_j + noise(SE_j),     a, b >= 0

by weighted least squares (weights 1/SE^2), with a ridge penalty pulling (a, b) toward the prior
direction (alpha0, beta0) at the scale that direction fits best. Only the DIRECTION matters (the
proxy is quantile-mapped downstream), so the result is reported as alpha + beta = alpha0 + beta0.

Guards, in order:
1. at least `min_rows` arms with calibrated evidence;
2. the fit must beat the prior direction on leave-one-arm-out weighted prediction error, else the
   prior (or the current weights) are kept: a null-model fallback, so noise cannot move the proxy;
3. trust region: alpha and beta each move by at most `trust_region` (relative) per fit.

Pure numpy, deterministic, no state beyond its configuration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_MIN_SE = 1e-9


@dataclass(frozen=True)
class FitRow:
    env_id: str
    gain: float      # b_j: calibrated gain per window of full compute share
    se: float        # its standard error
    lp: float        # exposure-weighted mean LP' (clipped, sigma-scaled) over the run so far
    sr: float        # exposure-weighted mean richness


@dataclass(frozen=True)
class ProxyFit:
    alpha: float                 # weights to use from now on
    beta: float
    proposed: tuple[float, float] | None   # the unconstrained fitted direction (None: no fit)
    accepted: bool
    reason: str
    n_rows: int
    loo_fit: float = float("nan")
    loo_prior: float = float("nan")

    def as_log(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "proposed": self.proposed, "accepted": self.accepted,
                "reason": self.reason, "n_rows": self.n_rows, "loo_fit": self.loo_fit, "loo_prior": self.loo_prior}


def _solve(x: np.ndarray, y: np.ndarray, w: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    """argmin_c sum w (y - x c)^2 + lam |c - target|^2 subject to c >= 0 (2 coefficients)."""
    def ridge(cols: list[int]) -> np.ndarray:
        xs = x[:, cols]
        a = (xs * w[:, None]).T @ xs + lam * np.eye(len(cols))
        rhs = (xs * w[:, None]).T @ y + lam * target[cols]
        c = np.zeros(x.shape[1])
        c[cols] = np.linalg.solve(a, rhs) if np.linalg.cond(a) < 1e12 else 0.0
        return c

    c = ridge([0, 1])
    if c[0] >= 0 and c[1] >= 0:
        return c
    best, best_loss = np.zeros(2), float("inf")
    for cols in ([0], [1]):
        cand = ridge(cols)
        if np.all(cand >= 0):
            loss = float(np.sum(w * (y - x @ cand) ** 2) + lam * np.sum((cand - target) ** 2))
            if loss < best_loss:
                best, best_loss = cand, loss
    return best


class ProxyWeightFitter:
    """Weighted, ridge-regularised, guarded fit of the proxy direction (see module docstring)."""

    def __init__(self, prior_alpha: float, prior_beta: float, *, ridge: float, trust_region: float,
                 min_rows: int) -> None:
        if prior_alpha < 0 or prior_beta < 0 or prior_alpha + prior_beta <= 0:
            raise ValueError("prior weights must be non-negative and not both zero")
        self.total = prior_alpha + prior_beta
        self.prior = np.array([prior_alpha, prior_beta]) / self.total
        self.ridge = float(ridge)
        self.trust_region = float(trust_region)
        self.min_rows = int(min_rows)

    # ------------------------------------------------------------------ pieces

    def _design(self, rows: Sequence[FitRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.array([[r.lp, r.sr] for r in rows], dtype=float)
        y = np.array([r.gain for r in rows], dtype=float)
        w = np.array([1.0 / max(r.se, _MIN_SE) ** 2 for r in rows], dtype=float)
        return x, y, w

    def _prior_scale(self, x, y, w) -> float:
        x0 = x @ self.prior
        den = float(np.sum(w * x0 * x0))
        return max(float(np.sum(w * x0 * y)) / den, 0.0) if den > 0 else 0.0

    def _fit_coefs(self, x, y, w) -> np.ndarray:
        s0 = self._prior_scale(x, y, w)
        diag = np.einsum("ij,i,ij->j", x, w, x)
        lam = self.ridge * float(diag.sum()) / (2.0 * max(len(y), 1))    # prior worth `ridge` average arms
        return _solve(x, y, w, s0 * self.prior, lam)

    def _loo(self, x, y, w) -> tuple[float, float]:
        err_fit = err_prior = 0.0
        n = len(y)
        for i in range(n):
            keep = np.arange(n) != i
            c = self._fit_coefs(x[keep], y[keep], w[keep])
            s0 = self._prior_scale(x[keep], y[keep], w[keep])
            err_fit += w[i] * (y[i] - x[i] @ c) ** 2
            err_prior += w[i] * (y[i] - s0 * (x[i] @ self.prior)) ** 2
        return err_fit / n, err_prior / n

    def _trust(self, proposed: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Move alpha (and so beta = 1 - alpha) at most `trust_region` relative to each current value."""
        a_cur, b_cur = current
        tr = self.trust_region
        lo = max(a_cur * (1 - tr), 1.0 - b_cur * (1 + tr), 0.0)
        hi = min(a_cur * (1 + tr), 1.0 - b_cur * (1 - tr), 1.0)
        a = min(max(proposed[0], lo), hi)
        return np.array([a, 1.0 - a])

    # ------------------------------------------------------------------ public

    def fit(self, rows: Sequence[FitRow], current: tuple[float, float]) -> ProxyFit:
        cur = np.array(current, dtype=float)
        cur = cur / cur.sum() if cur.sum() > 0 else self.prior.copy()
        keep = [r for r in rows if math.isfinite(r.gain) and math.isfinite(r.se) and r.se > 0
                and math.isfinite(r.lp) and math.isfinite(r.sr)]
        alpha, beta = (cur * self.total).tolist()
        if len(keep) < self.min_rows:
            return ProxyFit(alpha, beta, None, False, f"{len(keep)} rows < min_rows {self.min_rows}", len(keep))
        x, y, w = self._design(keep)
        coefs = self._fit_coefs(x, y, w)
        if coefs.sum() <= 0:
            return ProxyFit(alpha, beta, None, False, "no positive association with gains", len(keep))
        direction = coefs / coefs.sum()
        loo_fit, loo_prior = self._loo(x, y, w)
        proposed = tuple((direction * self.total).tolist())
        if not loo_fit < loo_prior:
            return ProxyFit(alpha, beta, proposed, False, "does not beat the prior out of sample", len(keep),
                            loo_fit, loo_prior)
        new = self._trust(direction, cur) * self.total
        return ProxyFit(float(new[0]), float(new[1]), proposed, True, "accepted", len(keep), loo_fit, loo_prior)
