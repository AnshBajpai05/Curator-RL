"""Mixture map: UCB scores -> weights -> status constraints (Roadmap E.6).

Pure functions, no scheduler state:

- `scores_to_weights`: score_norm (none | zscore | rank) -> clip(z/τ, ±50) ->
  log-sum-exp softmax -> exploration floor w_i >= ε/N. Sum = 1 within 1e-12.
- `apply_status_constraints`: the E.5 table as multipliers/quotas AFTER the
  base weights, then proportional redistribution, then re-impose floors
  (exploration floor AND S1 quota) and renormalise. Gated by
  `status_control: off | soft | hard`.

Constants: the doubling of the S1 quota while n_groups_eff < n_min/2 is the
roadmap's own rule (E.5, "larger while n_groups_eff < n_min/2"); the 2x factor
is structural, not a tuned quantity (D-47).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from curator_rl.core.config import SchedulerCfg
from curator_rl.core.types import EnvStatus, SignalVector

_Z_CLIP = 50.0
_EPS_NORM = 1e-12
_QUOTA_BOOST = 2.0  # roadmap E.5: S1 quota doubles while evidence < n_min/2


def scores_to_weights(  # noqa: C901  (public alias kept explicit)
    scores: Mapping[str, float],
    tau: float,
    epsilon: float,
    env_ids: list[str],
    *,
    score_norm: str = "zscore",
) -> dict[str, float]:
    return softmax_floor(scores, tau, epsilon, env_ids, score_norm=score_norm)


def softmax_floor(
    scores: Mapping[str, float],
    tau: float,
    epsilon: float,
    env_ids: list[str],
    *,
    score_norm: str = "zscore",
) -> dict[str, float]:
    """Turn raw scores into the E.6 mixture: normalise -> softmax -> floor."""
    if not env_ids:
        raise ValueError("no environments")
    values = [float(scores[e]) for e in env_ids]
    arr = np.asarray(values, dtype=float)

    if score_norm == "zscore":
        std = float(arr.std(ddof=0))
        arr = (arr - arr.mean()) / (std + 1e-12)
    elif score_norm == "rank":
        order = arr.argsort().argsort().astype(float)  # 0..n-1, ties broken by order
        n = len(env_ids)
        arr = order / max(n - 1, 1) * 2.0 - 1.0  # map to [-1, 1]
    elif score_norm != "none":
        raise ValueError(f"unknown score_norm '{score_norm}'")

    scaled = np.clip(arr / tau, -_Z_CLIP, _Z_CLIP)
    shifted = scaled - scaled.max()
    exps = np.exp(shifted)
    probs = exps / exps.sum()

    n = len(env_ids)
    floor = epsilon / n
    weights = (1.0 - epsilon) * probs + floor
    total = float(weights.sum())
    weights = weights / total  # renormalise the floor addition exactly
    out = {e: float(w) for e, w in zip(env_ids, weights)}
    if abs(sum(out.values()) - 1.0) > _EPS_NORM:  # pragma: no cover - numeric guard
        raise ArithmeticError("softmax weights do not sum to 1")
    if any(w < floor - _EPS_NORM for w in out.values()):
        raise ArithmeticError("exploration floor violated")
    return out


def cap_weights(weights: Mapping[str, float], cap: float, *, frozen: set[str] | None = None) -> dict[str, float]:
    """Cap every weight at `cap` (improvement B7), handing the excess to the arms below it.

    Arms in `frozen` (e.g. capped by their status) never receive any of the excess. The effective cap
    is at least 1/N (a mixture over N arms cannot have all weights below 1/N). Sum stays 1; weights
    only fall at capped arms and only rise elsewhere, so floors are preserved.
    """
    out = {e: float(w) for e, w in weights.items()}
    n = len(out)
    if n == 0 or cap >= 1.0:
        return out
    cap = max(cap, 1.0 / n)
    frozen = frozen or set()
    for _ in range(n + 1):
        excess = sum(max(w - cap, 0.0) for w in out.values())
        if excess <= _EPS_NORM:
            break
        out = {e: min(w, cap) for e, w in out.items()}
        takers = {e: w for e, w in out.items() if w < cap - _EPS_NORM and e not in frozen}
        if not takers:
            takers = {e: w for e, w in out.items() if w < cap - _EPS_NORM}
        if not takers:
            break
        room = {e: cap - w for e, w in takers.items()}
        total_room = sum(room.values())
        give = min(excess, total_room)
        for e in takers:
            out[e] += give * room[e] / total_room
    total = sum(out.values())
    return {e: w / total for e, w in out.items()}


def status_multiplier(sv: SignalVector, cfg: SchedulerCfg) -> float:
    """Soft-constraint multiplier for one environment from its status (E.5)."""
    if sv.status == EnvStatus.S3:
        return cfg.s3_multiplier
    if sv.status == EnvStatus.S4:
        return cfg.s4_multiplier
    if sv.status == EnvStatus.S5:
        return cfg.s5_shrink
    return 1.0


def _quota_floor(sv: SignalVector, cfg: SchedulerCfg, n_min: int, n_envs: int) -> float | None:
    """S1 exploration quota for one env (E.5): q_explore, 2x while half-explored."""
    if sv.status != EnvStatus.S1:
        return None
    quota = cfg.s1_quota
    if sv.n_groups_eff < n_min / 2.0:
        quota *= _QUOTA_BOOST
    return min(quota, 1.0 - (n_envs - 1) * _EPS_NORM)


def apply_status_constraints(
    weights: Mapping[str, float],
    vectors: Mapping[str, SignalVector],
    cfg: SchedulerCfg,
    *,
    n_min: int,
) -> tuple[dict[str, float], dict[str, str]]:
    """Apply E.5 status constraints and return (weights, intents).

    `soft` applies multiplicative caps with proportional redistribution;
    `hard` additionally caps each non-quota arm at 2 x the exploration floor
    (releasing everything above it). Sum stays 1; floors/quotas hold.
    """
    env_ids = sorted(weights)
    n = len(env_ids)
    floor = cfg.epsilon / n
    out = {e: float(weights[e]) for e in env_ids}
    intents: dict[str, str] = {}

    if cfg.status_control == "off":
        return dict(out), {e: "hold" for e in env_ids}

    # 1. multiplicative caps (soft + hard), quotas are floors not caps
    capped: set[str] = set()
    for e in env_ids:
        sv = vectors[e]
        if sv.status in (EnvStatus.S3, EnvStatus.S4, EnvStatus.S5):
            out[e] = out[e] * status_multiplier(sv, cfg)
            capped.add(e)
        if cfg.status_control == "hard" and sv.status == EnvStatus.S3:
            out[e] = min(out[e], 2.0 * floor)   # E.5: S3 hard = cap at 2 x floor
        if cfg.status_control == "hard" and sv.status == EnvStatus.S4:
            out[e] = min(out[e], floor)         # E.5: S4 hard = cap at the floor
        if cfg.status_control == "hard" and sv.status == EnvStatus.S5 and cfg.s5_cap == "floor":
            out[e] = min(out[e], floor)         # B5: a calibration-confirmed spurious arm keeps only the floor
        if sv.status == EnvStatus.S1:
            intents[e] = "explore"
        elif sv.status in (EnvStatus.S3, EnvStatus.S4, EnvStatus.S5):
            intents[e] = "shrink"
        else:
            intents[e] = "hold"

    # 2. floors, then S1 quotas (above the floor), with a quota mass budget:
    # boosted quotas must never sum above what the non-quota floors leave free
    for e in env_ids:
        out[e] = max(out[e], floor)
    quota_of: dict[str, float] = {}
    for e in env_ids:
        q = _quota_floor(vectors[e], cfg, n_min, n)
        if q is not None:
            quota_of[e] = max(q, floor)
    if quota_of:
        budget = max(1.0 - floor * (n - len(quota_of)), 0.0)
        qsum = sum(quota_of.values())
        if qsum > budget:
            quota_of = {e: q * budget / qsum for e, q in quota_of.items()}
    for e, q in quota_of.items():
        out[e] = max(out[e], q)

    # 3. proportional redistribution until sum == 1 with all floors/quotas held
    for _ in range(100):
        total = sum(out.values())
        if total <= 0:
            out = {e: 1.0 / n for e in env_ids}
            break
        if abs(total - 1.0) <= _EPS_NORM:
            break
        # arms free to shrink: everything strictly above its own floor/quota
        donor = {
            e: out[e] - max(floor, quota_of.get(e, floor))
            for e in env_ids
        }
        free_total = sum(v for v in donor.values() if v > 0)
        if total > 1.0 and free_total > _EPS_NORM:
            scale_free = (total - 1.0) / free_total
            for e in env_ids:
                if donor[e] > 0:
                    out[e] -= donor[e] * scale_free
        elif total < 1.0:
            # released weight goes to the UNCAPPED, non-quota arms
            # proportionally to their current mass (E.5: "redistribution is
            # automatic"; capped arms never absorb their own release)
            fillers = {e: out[e] for e in env_ids if e not in quota_of and e not in capped}
            free_fill = sum(fillers.values())
            if free_fill > _EPS_NORM:
                for e in env_ids:
                    if e in fillers:
                        out[e] += (1.0 - total) * fillers[e] / free_fill
            else:
                # everything is capped or at a quota: divide the remainder
                # over the uncapped arms, else uniformly (defensive)
                targets = [e for e in env_ids if e not in capped] or env_ids
                share = (1.0 - total) / len(targets)
                for e in targets:
                    out[e] += share
        else:
            break

    out = {e: max(w, floor if e not in quota_of else max(floor, quota_of[e])) for e, w in out.items()}
    total = sum(out.values())
    out = {e: w / total for e, w in out.items()}

    # 4. intent labels: increase / decrease relative to the input weights
    for e in env_ids:
        if intents[e] == "hold":
            delta = out[e] - weights[e]
            if delta > 1e-4:
                intents[e] = "increase"
            elif delta < -1e-4:
                intents[e] = "decrease"

    if abs(sum(out.values()) - 1.0) > _EPS_NORM:  # pragma: no cover - numeric guard
        raise ArithmeticError("status-constrained weights do not sum to 1")
    if any(out[e] < floor - _EPS_NORM for e in env_ids):
        raise ArithmeticError("exploration floor lost in redistribution")
    return out, intents
