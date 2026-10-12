"""Window gains and credit rules (Roadmap Part I.2-I.3, D-73).

Pure functions and small accumulators; no RNG, no scheduler state.

- `window_deltas`: per-slice change of the calibration benchmark between two
  evaluations, with its standard error (the paired SE when the observation
  carries one, else the independent-samples SE).
- `share_credit` (C1): the proposal's credit, delta_S x compute share. Kept as the
  baseline rule and for logging; it carries no information about WHICH
  environment caused the gain (I.2).
- `OwnSliceCredit` (C2, diagonal form): for environment j with its own benchmark
  slice d(j), the model is delta_S_{d(j),k} = W_{j,k} * b_j + noise(SE_k), with
  W_{j,k} the share of window k's compute spent on j. Weighted least squares
  across windows with an optional Gaussian prior on b_j. Transfer from other
  environments into j's slice is not modelled here (C2d is the full matrix form).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict

import numpy as np

from curator_rl.core.types import CalibrationObservation

_MIN_SE = 1e-6


def window_deltas(
    prev: CalibrationObservation, cur: CalibrationObservation
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-domain change cur - prev and its SE (Roadmap I.3)."""
    delta: dict[str, float] = {}
    se: dict[str, float] = {}
    paired = cur.delta_se_by_domain or {}
    for d, s in cur.score_by_domain.items():
        if d not in prev.score_by_domain:
            continue
        delta[d] = float(s) - float(prev.score_by_domain[d])
        if d in paired:
            se[d] = max(float(paired[d]), _MIN_SE)
        else:
            se[d] = max(math.hypot(float(cur.se_by_domain.get(d, 0.0)),
                                   float(prev.se_by_domain.get(d, 0.0))), _MIN_SE)
    return delta, se


def exposure_shares(exposure: Mapping[str, float]) -> dict[str, float]:
    """W_{j,k}: share of the window's compute mass spent on each environment."""
    total = sum(max(float(v), 0.0) for v in exposure.values())
    if total <= 0:
        return {e: 0.0 for e in exposure}
    return {e: max(float(v), 0.0) / total for e, v in exposure.items()}


def share_credit(delta_total: float, exposure: Mapping[str, float]) -> dict[str, float]:
    """C1: credit_j = delta_S * share_j (Roadmap I.2; baseline rule only)."""
    return {e: delta_total * w for e, w in exposure_shares(exposure).items()}


class OwnSliceState(TypedDict):
    sww: dict[str, float]
    swy: dict[str, float]
    n: dict[str, int]


class SharedSliceState(TypedDict):
    members: dict[str, list[str]]
    n: dict[str, int]
    xtx: dict[str, list[list[float]]]
    xty: dict[str, list[float]]


class OwnSliceCredit:
    """Weighted least squares of own-slice gains on compute share, per environment."""

    def __init__(self, env_ids) -> None:
        self.env_ids = tuple(env_ids)
        self._sww = {e: 0.0 for e in self.env_ids}   # sum W^2 / SE^2
        self._swy = {e: 0.0 for e in self.env_ids}   # sum W * y / SE^2
        self._n = {e: 0 for e in self.env_ids}       # windows with exposure

    def add(self, env_id: str, share: float, gain: float, se: float) -> None:
        if share <= 0.0:
            return
        inv = 1.0 / max(se, _MIN_SE) ** 2
        self._sww[env_id] += share * share * inv
        self._swy[env_id] += share * gain * inv
        self._n[env_id] += 1

    def estimate(self, env_id: str, *, prior_mean: float = 0.0,
                 prior_sd: float | None = None) -> tuple[float, float]:
        """Posterior (mean, sd) of b_j; `prior_sd=None` is the likelihood-only estimate."""
        precision = self._sww[env_id]
        weighted = self._swy[env_id]
        if prior_sd is not None and prior_sd > 0:
            precision += 1.0 / prior_sd**2
            weighted += prior_mean / prior_sd**2
        if precision <= 0.0:
            return prior_mean, float("inf")
        return weighted / precision, 1.0 / math.sqrt(precision)

    def windows(self, env_id: str) -> int:
        return self._n[env_id]

    def get_state(self) -> OwnSliceState:
        return {"sww": dict(self._sww), "swy": dict(self._swy), "n": dict(self._n)}

    def load_checkpoint(self, state: OwnSliceState) -> None:
        self._sww = {e: float(v) for e, v in state["sww"].items()}
        self._swy = {e: float(v) for e, v in state["swy"].items()}
        self._n = {e: int(v) for e, v in state["n"].items()}


# Weak ridge prior on every transfer coefficient: b ~ N(0, PRIOR_SD^2) in "gain per window of
# full compute share". Observed coefficients are 0.0-0.15 (refit diagnosis), so a prior SD of 1.0
# only matters when two arms' shares are collinear, where it keeps the system solvable and lets
# the posterior SD (not a point estimate) carry the "cannot tell them apart" verdict.
TRANSFER_PRIOR_SD = 1.0


class SharedSliceCredit:
    """C2d per slice (audit E-03): arms scored on the SAME slice are regressed jointly.

    For slice d with member arms j in M(d), every evaluation of d gives one span gain
        gain_{d,k} = sum_{j in M(d)} W_{j,k} * b_j + noise(SE_k),
    W_{j,k} = compute share arm j received over the span since d was last evaluated.
    Weighted ridge (prior b_j ~ N(0, TRANSFER_PRIOR_SD^2)) gives the posterior mean and SD of
    every b_j. A slice with one member reduces to `OwnSliceCredit` (up to the weak prior).
    Why it matters: on real hardware the noisy arm has no slice of its own and is scored on the
    GSM8K slice. Own-slice credit would hand it the gains the GSM8K arm produced; joint credit
    separates them whenever their shares vary independently across spans.
    """

    def __init__(self, members: Mapping[str, list[str]]) -> None:
        self.members = {d: list(m) for d, m in members.items()}
        self._xtx = {d: np.zeros((len(m), len(m))) for d, m in self.members.items()}
        self._xty = {d: np.zeros(len(m)) for d, m in self.members.items()}
        self._n = {e: 0 for m in self.members.values() for e in m}

    def add(self, domain: str, shares: Mapping[str, float], gain: float, se: float) -> None:
        m = self.members[domain]
        x = np.array([max(float(shares.get(e, 0.0)), 0.0) for e in m])
        if not np.any(x > 0.0):
            return
        inv = 1.0 / max(se, _MIN_SE) ** 2
        self._xtx[domain] += inv * np.outer(x, x)
        self._xty[domain] += inv * gain * x
        for e, xi in zip(m, x):
            if xi > 0.0:
                self._n[e] += 1

    def estimate(self, env_id: str) -> tuple[float, float]:
        """Posterior (mean, sd) of b_j; (0, inf) before the arm has any evidence."""
        if self._n.get(env_id, 0) == 0:
            return 0.0, float("inf")
        domain = next(d for d, m in self.members.items() if env_id in m)
        m = self.members[domain]
        if len(m) == 1:  # unshared slice: exactly the own-slice WLS estimate, no prior needed
            sww, swy = float(self._xtx[domain][0, 0]), float(self._xty[domain][0])
            return swy / sww, 1.0 / math.sqrt(sww)
        precision = self._xtx[domain] + np.eye(len(m)) / TRANSFER_PRIOR_SD**2
        cov = np.linalg.inv(precision)
        mean = cov @ self._xty[domain]
        i = m.index(env_id)
        return float(mean[i]), float(math.sqrt(max(cov[i, i], 0.0)))

    def windows(self, env_id: str) -> int:
        return self._n.get(env_id, 0)

    def get_state(self) -> SharedSliceState:
        return {"members": self.members, "n": dict(self._n),
                "xtx": {d: v.tolist() for d, v in self._xtx.items()},
                "xty": {d: v.tolist() for d, v in self._xty.items()}}

    def load_checkpoint(self, state: SharedSliceState) -> None:
        self._n = {e: int(v) for e, v in state["n"].items()}  # type: ignore[union-attr]
        self._xtx = {d: np.asarray(v, dtype=float) for d, v in state["xtx"].items()}  # type: ignore[union-attr]
        self._xty = {d: np.asarray(v, dtype=float) for d, v in state["xty"].items()}  # type: ignore[union-attr]
