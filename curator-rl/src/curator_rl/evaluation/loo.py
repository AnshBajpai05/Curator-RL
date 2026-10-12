"""Leave-one-out contributions and the ROI-vs-LOO agreement test (Roadmap Gate 7/9, H5; improvement D3).

LOO: retrain with arm i removed at the same budget; contribution_i = S(full) - S(without i), paired by
seed. The benchmark itself is NOT changed (arm i's slice is still scored; it just is not trained).
H5 asks whether the cheap, in-run ROI estimate (roi/leaderboard.py) ranks arms like the expensive LOO
does: Kendall tau >= 0.4 between the two rankings, with both junk arms (no benchmark value) in the
bottom two of both.

`WithoutArm` runs any scheduler on a world that still contains the arm: the arm gets weight 0 and is
removed from what the scheduler observes, so the scheduler behaves exactly as it would on a portfolio
without that arm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from curator_rl.core.stats import paired_t_interval
from curator_rl.core.types import CalibrationObservation, RoundObservation


class WithoutArm:
    """Scheduler adapter: `inner` was built for the portfolio without `excluded`."""

    def __init__(self, inner, excluded: str, all_env_ids: Sequence[str]) -> None:
        self.inner, self.excluded = inner, excluded
        self.env_ids = tuple(all_env_ids)
        self.uses_calibration = getattr(inner, "uses_calibration", False)

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        weights = dict(self.inner.select_mixture(self._strip(observation)))
        weights[self.excluded] = 0.0
        return weights

    def update_observation(self, observation: RoundObservation) -> None:
        self.inner.update_observation(self._strip(observation))

    def update_calibration(self, observation: CalibrationObservation) -> None:
        exposure = {e: v for e, v in observation.exposure_by_env.items() if e != self.excluded}
        self.inner.update_calibration(replace(observation, exposure_by_env=exposure))

    def calibration_request(self):
        fn = getattr(self.inner, "calibration_request", None)
        return fn() if callable(fn) else None

    def _strip(self, obs: RoundObservation | None) -> RoundObservation | None:
        if obs is None:
            return None
        return replace(obs, per_env={e: o for e, o in obs.per_env.items() if e != self.excluded},
                       weights_used={e: w for e, w in obs.weights_used.items() if e != self.excluded})

    def __getattr__(self, name):          # diagnostics (e.g. _latest_vectors) pass through
        return getattr(self.inner, name)


def loo_contributions(full: Sequence[float], without: Mapping[str, Sequence[float]]) -> dict[str, tuple]:
    """{arm: (mean contribution, lo, hi)} with paired 95% t-intervals over seeds."""
    out = {}
    for arm, scores in without.items():
        diffs = [f - w for f, w in zip(full, scores)]
        out[arm] = paired_t_interval(diffs) if len(diffs) >= 2 else ((diffs[0],) * 3 if diffs else (math.nan,) * 3)
    return out


def kendall_tau(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Kendall tau-b between two scorings of the same keys (ties count as neither)."""
    keys = sorted(set(a) & set(b))
    conc = disc = ties_a = ties_b = 0
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            da, db = a[x] - a[y], b[x] - b[y]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + ties_a) * (conc + disc + ties_b))
    return (conc - disc) / denom if denom > 0 else math.nan


def h5_verdict(roi: Mapping[str, float], loo: Mapping[str, float], junk: Sequence[str],
               *, tau_min: float = 0.4) -> dict:
    """H5: tau(ROI, LOO) >= tau_min and every junk arm in the bottom len(junk) of both rankings."""
    tau = kendall_tau(roi, loo)
    k = len(junk)
    bottom_roi = set(sorted(roi, key=lambda e: roi[e])[:k])
    bottom_loo = set(sorted(loo, key=lambda e: loo[e])[:k])
    junk_bottom = set(junk) <= bottom_roi and set(junk) <= bottom_loo
    return {"tau": tau, "tau_ok": bool(tau >= tau_min), "junk_bottom": junk_bottom,
            "bottom_roi": sorted(bottom_roi), "bottom_loo": sorted(bottom_loo),
            "pass": bool(tau >= tau_min and junk_bottom)}
