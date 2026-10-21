"""Core data contracts (Roadmap B.3, frozen in Phase 2).

These types are owned by `core` (L0) so every layer (scheduler, simulator,
trainer adapter) can share them: the simulator, the trace replayer and the
real trainer all produce the same observation types.

Phase 2 introduces `Prompt`, `Verdict`, `RolloutGroup` and `CostPrior`.
Phase 3 adds `EnvRoundObs`, `RoundObservation` and `CalibrationObservation`
(Roadmap B.3): the simulator, the trace replayer and the real trainer all
produce these same observation types. Phase 4 adds `EnvStatus` and
`SignalVector` (the Signal Engine's per-environment output). `MixtureDecision`
and `RoiRecord` are added in the phases that first need them (Roadmap Part
P, decision D-26).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

SPLITS = ("train", "calib", "dev", "test")


@dataclass(frozen=True)
class Prompt:
    """A single training or evaluation item.

    `reference` is verifier-only: it MUST never be shown to the model and is
    not carried into the message list. `meta` holds environment-specific
    parameters (e.g. Countdown numbers and target) that the verifier needs.
    """

    prompt_id: str
    env_id: str
    split: str
    messages: tuple[dict[str, str], ...]
    reference: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(m.get("content", "") for m in self.messages)


@dataclass(frozen=True)
class Verdict:
    """Outcome of verifying one completion against one prompt.

    Invariant: 0.0 <= score <= 1.0. `parse_ok` is False when no parseable
    answer could be extracted (a different failure than a wrong answer).
    """

    success: bool
    score: float
    parse_ok: bool
    verifier_seconds: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutGroup:
    """One prompt with its G rollouts. Produced by the reward wrapper /
    simulator, consumed by the Signal Engine through the round observation."""

    env_id: str
    prompt_id: str
    n: int
    prompt_tokens: int
    completion_tokens: tuple[int, ...]
    scores: tuple[float, ...]
    successes: tuple[bool, ...]
    verifier_seconds: float


@dataclass(frozen=True)
class CostPrior:
    """Cold-start cost estimate. NEVER used as a measured cost (Roadmap H)."""

    env_id: str
    per_prompt_usd: float
    note: str


@dataclass(frozen=True)
class EnvRoundObs:
    """Per-environment statistics for one round (Roadmap B.3).

    `sum_score`/`sum_score_sq` are over all rollout scores of the round;
    `n_groups_mixed` counts prompt groups with an informative pass rate
    (0 < k < G). `prompt_tokens`/`completion_tokens` are round totals.
    """

    env_id: str
    n_prompts: int
    n_rollouts: int
    k_success: int
    n_groups_mixed: int
    sum_score: float
    sum_score_sq: float
    prompt_tokens: int
    completion_tokens: int
    verifier_seconds: float
    gpu_seconds: float
    cost_usd: float
    # Sum over the round's prompt groups of the group mean |GRPO advantage|
    # (`core.advantage.group_mean_abs_advantage`). SEC/DUMP-style allocators
    # read it; Curator ignores it. Default keeps older constructors valid.
    sum_abs_adv: float = 0.0


@dataclass(frozen=True)
class RoundObservation:
    """Everything the scheduler may see at a round boundary (Roadmap B.3/F.1).

    The scheduler MUST NOT be able to read any hidden simulator variable, so
    this type carries only measured quantities. `budget_remaining_usd` is
    tracked by the caller (harness or trainer callback) under the ledger rule
    (Roadmap H.6).
    """

    round: int
    steps: int
    per_env: dict[str, EnvRoundObs]
    weights_used: dict[str, float]
    round_cost_usd: float
    overhead_usd: float
    budget_remaining_usd: float


@dataclass(frozen=True)
class CalibrationObservation:
    """One calibration-benchmark evaluation of a window (Roadmap B.3, I.1/I.3).

    `exposure_by_env` is the compute mass W_{j,k} (prompt-equivalents) each
    environment received since the previous calibration; `window_cost_usd`
    is the charged training cost of that window.
    """

    window_k: int
    round: int
    score_total: float
    score_by_domain: dict[str, float]
    se_total: float
    se_by_domain: dict[str, float]
    n_items: int
    eval_cost_usd: float
    exposure_by_env: dict[str, float]
    window_cost_usd: float
    # SE of the change in each slice since the previous evaluation, from PAIRED
    # items (only items that flipped add variance; Roadmap v3 4.3). None when the
    # items are not paired or this is the first evaluation (D-72).
    delta_se_by_domain: dict[str, float] | None = None


class EnvStatus(StrEnum):
    """Per-environment status classes (Roadmap E.5, resolves S-1)."""

    S1 = "S1_unexplored"
    S2 = "S2_learning"
    S3 = "S3_saturated"
    S4 = "S4_too_hard"
    S5 = "S5_unreliable"


@dataclass(frozen=True)
class SignalVector:
    """Per-environment signals for one round (Roadmap E.2-E.5, Phase 4).

    Produced by the Signal Engine from a `RoundObservation`; consumed by the
    scheduler (Phase 5) and the study scripts. `status_note` carries the S2
    sub-label ("S2a_progressing" / "S2b_plateau") and free-form context.
    """

    env_id: str
    round_t: int
    n_rounds_seen: int
    n_groups_eff: float
    pass_rate: float
    pass_lo: float
    pass_hi: float
    lp: float
    lp_se: float
    lp_z: float
    lp_raw_fast: float
    lp_raw_slow: float
    richness: float
    mean_score: float
    unit_cost_usd: float
    unit_cost_norm: float
    proxy_raw: float
    proxy_unit: float
    proxy_reward: float
    status: EnvStatus
    status_note: str
    # reward audit (signals/audit.py): dispersion ratio D of group pass rates (1 = reward ignores the
    # prompt) and whether the audit flags the arm; LP' as used in the proxy (for the alpha/beta fit);
    # whether the drift detector fired this round (improvement B2)
    reward_dispersion: float = float("nan")
    audit_flag: bool = False
    lp_scaled: float = 0.0
    drift_alarm: bool = False


@dataclass(frozen=True)
class MixtureDecision:
    """One scheduler decision (Roadmap B.3; Phase 5, ends the D-26 deferral).

    `weights` is the mixture actually returned; `ucb_scores`/`mu_hat`/`bonus`
    are the pre-mixture score components; `statuses`/`intents` are the
    SignalVector statuses and the (status, Δw)-derived action labels; `quotas`
    are the largest-remainder prompt counts for the round; `rng_hash` pins
    the decision to a scheduler state hash for auditability.
    """

    round: int
    steps: int
    weights: dict[str, float]
    ucb_scores: dict[str, float]
    mu_hat: dict[str, float]
    bonus: dict[str, float]
    statuses: dict[str, str]
    intents: dict[str, str]
    quotas: dict[str, int]
    rng_hash: str
