"""Calibrator: the second, slow speed of Curator's reward (Roadmap Part I, D-73/D-75).

Every K rounds the scheduler hands over one `CalibrationObservation` (benchmark
scores for the slices that were evaluated, with SEs) and the exposure-weighted
mean proxy signal x_j = alpha*LP' + beta*SR each environment showed during the
window. The calibrator:

1. turns each slice's change since ITS previous evaluation into one gain
   observation, paired with the compute share the environment received over
   that span (slices need not be evaluated every window: targeted calibration,
   D-75);
2. accumulates C1 share credit (baseline, logged) and C2 own-slice credit:
   b_j = gain on j's own slice per window of full compute share;
3. maps proxy units to gain units with a ROBUST scale s = median_j(b_j / x_j)
   over environments with a positive proxy (a single junk arm cannot drag the
   scale to zero and hide itself), giving the proxy-implied gain s * x_j;
4. flags a proxy-vs-benchmark mismatch for j when the likelihood-only estimate
   lies more than z_mis standard errors BELOW the proxy-implied gain (Roadmap
   I.5). Underestimation is logged, never flagged. No flags before `k_min`
   windows.

Targeting (`select_targets`): the first calibration evaluates every slice (the
paired baseline); afterwards, with `targeting: exposure`, only the slices of the
`max_targets` environments that received the most compute in the window — where
a wrong proxy wastes the most budget — are evaluated. `targeting: all` evaluates
every slice every time (calibration v1, D-73).

The flags feed the S5 state machine (q consecutive windows to enter, q' to
leave). Environments without any slice (`domain_of[j] is None`) are never
evaluated or flagged by this rule.

Shared slices (audit E-03). `domain_of` may map several arms to ONE slice (on real
hardware the noisy arm is scored on the GSM8K slice). Targeting then ranks SLICES
(the summed claims of their arms), a shared slice is evaluated once, and with
`calib.credit: transfer` the arms of a slice are credited jointly
(`SharedSliceCredit`), so an arm cannot inherit the gains a sibling produced.
With `credit: own_slice` and one arm per slice the behaviour is exactly the
pre-audit one.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from curator_rl.calibration.credit import (
    OwnSliceCredit,
    SharedSliceCredit,
    exposure_shares,
    share_credit,
)
from curator_rl.core.config import CalibCfg
from curator_rl.core.types import CalibrationObservation

_MIN_SE = 1e-6


class CalibratorState(TypedDict):
    credit_mode: NotRequired[str]       # absent before shared-slice credit (then own_slice)
    credit: Any                         # OwnSliceState | SharedSliceState, per credit_mode
    last_score: dict[str, float]
    last_se: dict[str, float]
    span_w: dict[str, float]
    span_usd: NotRequired[dict[str, float]]
    x_sum: dict[str, float]
    x_w: dict[str, float]
    windows: int
    c1_total: dict[str, float]
    age: NotRequired[dict[str, int]]


@dataclass(frozen=True)
class CalibrationReport:
    """What one calibration window concluded (logged; flags are acted upon)."""

    window_k: int
    ready: bool
    evaluated: tuple[str, ...] = ()
    delta_by_domain: dict[str, float] = field(default_factory=dict)
    delta_se_by_domain: dict[str, float] = field(default_factory=dict)
    share_credit: dict[str, float] = field(default_factory=dict)
    gain_hat: dict[str, float] = field(default_factory=dict)
    gain_se: dict[str, float] = field(default_factory=dict)
    proxy_mean: dict[str, float] = field(default_factory=dict)
    proxy_scale: float = 0.0
    proxy_implied: dict[str, float] = field(default_factory=dict)
    z: dict[str, float] = field(default_factory=dict)
    mismatch: dict[str, bool] = field(default_factory=dict)
    # per evaluated arm: {"gain", "se", "usd", "share"} over the span since its slice
    # was last evaluated (the CRE's likelihood inputs, D-76)
    span_gain: dict[str, dict[str, float]] = field(default_factory=dict)


class Calibrator:
    """Slice credit + robust proxy scale + S5 mismatch flags (pure, deterministic)."""

    def __init__(self, env_ids, calib_cfg: CalibCfg,
                 domain_of: Mapping[str, str | None] | None = None) -> None:
        self.env_ids = tuple(sorted(env_ids))
        self._cfg = calib_cfg
        self._domain = {e: (domain_of or {}).get(e, e) for e in self.env_ids}
        self._members: dict[str, list[str]] = {}
        for e in self.env_ids:
            d = self._domain[e]
            if d is not None:
                self._members.setdefault(d, []).append(e)
        self._domains = tuple(sorted(self._members))
        self._credit_mode = calib_cfg.credit
        self._credit: OwnSliceCredit | SharedSliceCredit = (
            SharedSliceCredit(self._members) if self._credit_mode == "transfer"
            else OwnSliceCredit(self.env_ids))
        self._last_score: dict[str, float] = {}       # per slice, at its last evaluation
        self._last_se: dict[str, float] = {}
        self._span_w = {e: 0.0 for e in self.env_ids}  # compute share since j's slice was last evaluated
        self._span_usd = {e: 0.0 for e in self.env_ids}  # dollars charged to j over the same span
        self._x_sum = {e: 0.0 for e in self.env_ids}
        self._x_w = {e: 0.0 for e in self.env_ids}
        self._age = {d: 0 for d in self._domains}       # windows since slice d was last evaluated
        self._windows = 0                                # calibration windows observed
        self._c1_total = {e: 0.0 for e in self.env_ids}

    # ---------------------------------------------------------------- targeting

    def select_targets(self, window_shares: Mapping[str, float],
                       window_claims: Mapping[str, float] | None = None) -> dict[str, int]:
        """Slices to evaluate at this calibration, with items per slice (keyed by slice id).

        `window_claims[e]` is the budget-weighted proxy claim of the window (sum over rounds of
        share x proxy signal). `claim` ranks by it: where the proxy promises the most gain per
        window is where a wrong proxy wastes the most, even if that arm is not the most funded.
        `claim_stale` multiplies it by (1 + windows since the slice was last evaluated), so no
        funded arm stays unchecked indefinitely. A shared slice's key is the sum over its arms,
        and a shared slice is evaluated once for all of them.
        """
        n = self._cfg.items_per_slice
        mode = self._cfg.targeting
        if mode == "all" or not self._last_score:
            return {d: n for d in self._domains}
        if mode == "exposure" or window_claims is None:
            per_env = {e: float(window_shares.get(e, 0.0)) for e in self.env_ids}
        else:
            per_env = {e: max(float(window_claims.get(e, 0.0)), 0.0) for e in self.env_ids}
        key = {d: sum(per_env[e] for e in self._members[d]) for d in self._domains}
        if mode == "claim_stale":
            key = {d: v * (1.0 + self._age[d]) for d, v in key.items()}
        ranked = sorted(self._domains, key=lambda d: (-key[d], d))
        return {d: n for d in ranked[: self._cfg.max_targets]}

    # ------------------------------------------------------------------ observe

    def observe(self, obs: CalibrationObservation, proxy_x: Mapping[str, float],
                window_usd: Mapping[str, float] | None = None) -> CalibrationReport:
        shares = exposure_shares({e: obs.exposure_by_env.get(e, 0.0) for e in self.env_ids})
        for d in self._domains:
            self._age[d] += 1
        paired = obs.delta_se_by_domain or {}
        delta, dse, evaluated = {}, {}, []
        span_gain: dict[str, dict[str, float]] = {}
        for e in self.env_ids:
            w = shares[e]
            self._span_w[e] += w
            self._span_usd[e] += float((window_usd or {}).get(e, 0.0))
            if w > 0.0:
                self._x_sum[e] += w * float(proxy_x.get(e, 0.0))
                self._x_w[e] += w
        new_scores: dict[str, tuple[float, float]] = {}
        for d in self._domains:
            if d not in obs.score_by_domain:
                continue
            members = self._members[d]
            evaluated.extend(members)
            s_now = float(obs.score_by_domain[d])
            se_now = float(obs.se_by_domain.get(d, 0.0))
            if d in self._last_score:
                gain = s_now - self._last_score[d]
                se = float(paired[d]) if d in paired else math.hypot(se_now, self._last_se[d])
                se = max(se, _MIN_SE)
                delta[d], dse[d] = gain, se
                if isinstance(self._credit, SharedSliceCredit):
                    self._credit.add(d, {e: self._span_w[e] for e in members}, gain, se)
                else:
                    for e in members:
                        self._credit.add(e, self._span_w[e], gain, se)
                for e in members:
                    span_gain[e] = {"gain": gain, "se": se, "usd": self._span_usd[e],
                                    "share": self._span_w[e]}
            for e in members:
                self._span_w[e] = 0.0
                self._span_usd[e] = 0.0
            self._age[d] = 0
            new_scores[d] = (s_now, se_now)
        for d, (s, se) in new_scores.items():
            self._last_score[d], self._last_se[d] = s, se
        if not delta:  # baseline evaluation (or nothing comparable yet): no gain, no decision
            return CalibrationReport(window_k=obs.window_k, ready=False, evaluated=tuple(evaluated))
        self._windows += 1  # k_min counts windows that produced gains

        total_gain = sum(delta.values()) / len(delta) if delta else 0.0
        c1 = share_credit(total_gain, shares)
        for e in self.env_ids:
            self._c1_total[e] += c1[e]

        gain_hat, gain_se, xbar = {}, {}, {}
        for e in self.env_ids:
            gain_hat[e], gain_se[e] = self._credit.estimate(e)
            xbar[e] = self._x_sum[e] / self._x_w[e] if self._x_w[e] > 0 else 0.0

        ratios = [gain_hat[e] / xbar[e] for e in self.env_ids
                  if self._domain[e] is not None and xbar[e] > 1e-9
                  and self._credit.windows(e) > 0 and gain_se[e] < float("inf")]
        scale = max(statistics.median(ratios), 0.0) if ratios else 0.0

        ready = self._windows >= self._cfg.k_min
        implied, z, mismatch = {}, {}, {}
        for e in self.env_ids:
            implied[e] = scale * xbar[e]
            if self._domain[e] is None or gain_se[e] == float("inf") or not ready:
                z[e], mismatch[e] = 0.0, False
                continue
            z[e] = (gain_hat[e] - implied[e]) / gain_se[e]
            mismatch[e] = bool(z[e] < -self._cfg.z_mis)

        return CalibrationReport(
            window_k=obs.window_k, ready=ready, evaluated=tuple(evaluated),
            delta_by_domain=delta, delta_se_by_domain=dse, share_credit=c1,
            gain_hat=gain_hat, gain_se=gain_se, proxy_mean=xbar,
            proxy_scale=scale, proxy_implied=implied, z=z, mismatch=mismatch,
            span_gain=span_gain,
        )

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> CalibratorState:
        return {
            "credit_mode": self._credit_mode,
            "credit": self._credit.get_state(),
            "last_score": dict(self._last_score), "last_se": dict(self._last_se),
            "span_w": dict(self._span_w), "span_usd": dict(self._span_usd),
            "x_sum": dict(self._x_sum), "x_w": dict(self._x_w),
            "windows": self._windows, "c1_total": dict(self._c1_total), "age": dict(self._age),
        }

    def load_checkpoint(self, state: CalibratorState) -> None:
        mode = state.get("credit_mode", "own_slice")
        if mode != self._credit_mode:
            raise ValueError(f"checkpoint credit mode {mode!r} != configured {self._credit_mode!r}")
        self._credit.load_checkpoint(state["credit"])
        self._last_score = {d: float(v) for d, v in state["last_score"].items()}
        self._last_se = {d: float(v) for d, v in state["last_se"].items()}
        self._span_w = {e: float(v) for e, v in state["span_w"].items()}
        self._span_usd = {e: float(v) for e, v in state.get("span_usd", {}).items()} or {
            e: 0.0 for e in self.env_ids}
        self._x_sum = {e: float(v) for e, v in state["x_sum"].items()}
        self._x_w = {e: float(v) for e, v in state["x_w"].items()}
        self._windows = int(state["windows"])
        self._c1_total = {e: float(v) for e, v in state["c1_total"].items()}
        self._age = {d: int(v) for d, v in state.get("age", {}).items()} or {
            d: 0 for d in self._domains}
