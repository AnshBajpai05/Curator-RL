"""Validated configuration system.

Scientific hyperparameters have NO Python defaults: a config file (or
override) must supply them, otherwise validation fails. Unknown keys fail.
Everything is extra="forbid", so the config hash covers the whole experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class CuratorConfigError(Exception):
    """Raised for any user-facing configuration problem."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(_StrictModel):
    name: str
    tier: Literal[0, "1R", 1, 2]
    method: str
    seed: int = Field(ge=0)


class BudgetCfg(_StrictModel):
    total_usd: float = Field(gt=0)
    usd_per_gpu_hour: float = Field(gt=0)
    n_gpus: int = Field(ge=1)


class SchedulerDriftCfg(_StrictModel):
    """Drift-triggered discounting (improvement B2): gamma stays 1, an arm forgets only on a detected change."""

    enabled: bool
    delta: float = Field(ge=0)          # Page-Hinkley tolerance, in standardised-residual units
    threshold: float = Field(gt=0)      # Page-Hinkley alarm level (lambda)
    decay: float = Field(gt=0, le=1)    # rho: the arm's bandit counters are multiplied by it on an alarm


class SchedulerCfg(_StrictModel):
    gamma: float = Field(gt=0, le=1)
    exploration_coef: float = Field(ge=0)
    tau: float = Field(gt=0)
    epsilon: float = Field(ge=0, lt=1)
    warmup_rounds: int = Field(ge=0)
    score_norm: Literal["none", "zscore", "rank"]
    cost_exponent: float = Field(ge=0)
    status_control: Literal["off", "soft", "hard"]
    # bandit counter increment per round: round = requested weight; prompt = weight x prompts
    # (legacy); delivered = realised prompt share n_i / sum n (audit E-09, matches what was trained)
    pull_unit: Literal["round", "prompt", "delivered"]
    s3_multiplier: float = Field(ge=0, le=1)   # S3 weight multiplier (applied to the WEIGHT)
    s4_multiplier: float = Field(ge=0, le=1)   # S4 weight multiplier
    s5_shrink: float = Field(ge=0, le=1)       # S5 weight multiplier
    # S5 hard response (improvement B5): none = multiplier only; floor = also cap at eps/N under `hard`
    s5_cap: Literal["none", "floor"]
    s1_quota: float = Field(ge=0)
    # concentration control (improvement B7): cap on any single weight (1 = off) and exponential
    # smoothing of the mixture, w_t = (1 - s) w_{t-1} + s w_new (smoothing 1 = off)
    max_weight: float = Field(gt=0, le=1)
    smoothing: float = Field(gt=0, le=1)
    drift: SchedulerDriftCfg


class UcbBaselineCfg(_StrictModel):
    """Standard UCB baseline (J.1 #5): its own kappa/tau, tuned independently (D-68)."""

    exploration_coef: float = Field(ge=0)
    tau: float = Field(gt=0)


class LpBaselineCfg(_StrictModel):
    """Learning-progress curriculum baseline (J.1 #4)."""

    tau: float = Field(gt=0)


class SecBaselineCfg(_StrictModel):
    """SEC-style bandit (J.1 #9, D-68): TD(0) step `alpha`, Boltzmann temperature `tau`."""

    alpha: float = Field(gt=0, le=1)
    tau: float = Field(gt=0)


class DumpBaselineCfg(_StrictModel):
    """DUMP-style UCB over mean |advantage| (D-68)."""

    exploration_coef: float = Field(ge=0)
    tau: float = Field(gt=0)
    # 0 = all-time mean (D-68 legacy); W > 0 = mean over the last W rounds with prompts, as
    # DUMP estimates learnability from recent samples (audit E-16)
    window_rounds: int = Field(ge=0)


class BaselinesCfg(_StrictModel):
    ucb: UcbBaselineCfg
    lp: LpBaselineCfg
    sec: SecBaselineCfg
    dump: DumpBaselineCfg


class RichnessCfg(_StrictModel):
    mode: Literal["mixed", "band", "variance"]
    band_lo: float = Field(ge=0, le=1)
    band_hi: float = Field(ge=0, le=1)
    variance_min: float = Field(gt=0)

    @field_validator("band_hi")
    @classmethod
    def _band_ordered(cls, hi: float, info):  # noqa: ANN001
        lo = info.data.get("band_lo")
        if lo is not None and hi < lo:
            raise ValueError("richness.band_hi must be >= richness.band_lo")
        return hi


class StatusCfg(_StrictModel):
    n_min: int = Field(gt=0)
    r_min: int = Field(gt=0)
    p_sat: float = Field(ge=0, le=1)
    p_hard: float = Field(ge=0, le=1)
    sr_hard: float = Field(ge=0, le=1)
    z_up: float = Field(gt=0)
    z_neg: float = Field(gt=0)
    hysteresis_p: float = Field(ge=0, le=1)
    hysteresis_sr: float = Field(ge=0, le=1)
    dwell_min: int = Field(ge=1)
    consecutive_rounds: int = Field(ge=1)
    s1_entry_ratio: float = Field(ge=0, le=1)


class RewardAuditCfg(_StrictModel):
    """Prompt-dependence audit of each arm's reward (signals/audit.py): flags rewards that ignore the prompt."""

    enabled: bool
    d_max: float = Field(gt=1)   # dispersion ratio separating prompt-independent (D=1) from real (D~3-4) rewards
    z: float = Field(gt=0)       # one-sided confidence multiplier on log D (enter below, clear above)
    min_df: int = Field(ge=2)    # pooled degrees of freedom before any decision
    preflight_gate: bool = False  # S5 from the start for an arm whose reward paid for junk before training (D-102)


class PriorCfg(_StrictModel):
    alpha0: float = Field(gt=0)
    beta0: float = Field(gt=0)


class SignalsCfg(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lam: float = Field(alias="lambda", gt=0, le=1)
    lambda_fast: float = Field(gt=0, le=1)
    lambda_slow: float = Field(gt=0, le=1)
    window_rounds: int = Field(gt=0)
    lp_method: Literal["lp_a", "lp_b", "lp_c"]
    # LP-B standard error (audit E-01): design = per-round variance from within-round
    # group spread (calibrated z); residual_legacy = D-37 fit (SE ~sqrt(n) too large)
    lp_se: Literal["design", "residual_legacy"]
    lp_use: Literal["signed", "positive", "abs"]
    richness: RichnessCfg
    prior: PriorCfg
    status: StatusCfg
    reward_audit: RewardAuditCfg


class ProxyFitCfg(_StrictModel):
    """alpha/beta fitted to calibrated gains (improvement B4, calibration/regression.py)."""

    enabled: bool
    ridge: float = Field(ge=0)     # penalty (in units of prediction precision) toward the prior direction
    min_rows: int = Field(ge=2)    # arms with calibrated evidence needed before any fit


class ProxyCfg(_StrictModel):
    alpha: float = Field(ge=0)
    beta: float = Field(ge=0)
    clip_l: float = Field(gt=0)
    quantile_prior_lo: float = Field(ge=0)
    quantile_prior_hi: float = Field(gt=0)
    fit: ProxyFitCfg

    @field_validator("quantile_prior_hi")
    @classmethod
    def _quantile_ordered(cls, hi: float, info):  # noqa: ANN001
        lo = info.data.get("quantile_prior_lo")
        if lo is not None and hi <= lo:
            raise ValueError("proxy.quantile_prior_hi must be > proxy.quantile_prior_lo")
        return hi


class CalibCfg(_StrictModel):
    enabled: bool  # Curator consumes calibration (ablation `calib.enabled: false`, Roadmap 12)
    interval_rounds: int = Field(ge=1)
    k_min: int = Field(ge=1)
    trust_region: float = Field(gt=0, le=1)  # max relative change of alpha and beta per fit (proxy.fit)
    z_mis: float = Field(gt=0)
    mismatch_windows: int = Field(ge=1)
    mismatch_clear_windows: int = Field(ge=1)
    # which arms receive a mismatch/clear update after a ready window (audit E-02):
    # evaluated = only arms whose slice was re-evaluated (each flag is new evidence);
    # all = every arm with past evidence (legacy: stale estimates re-flagged every window)
    flag_scope: Literal["evaluated", "all"]
    # gain credit (audit E-03): own_slice = each arm alone on its slice (C2 diagonal, legacy);
    # transfer = arms SHARING a slice are regressed jointly on their shares (C2d per slice)
    credit: Literal["own_slice", "transfer"]
    # which slices each calibration evaluates (D-75; claim rules D-93): exposure = most compute;
    # claim = largest budget-weighted proxy claim (share x proxy signal); claim_stale = claim x (1 + age)
    targeting: Literal["all", "exposure", "claim", "claim_stale"]
    max_targets: int = Field(ge=1)         # slices per calibration when targeting is not "all"
    items_per_slice: int = Field(ge=1)     # paired items evaluated per targeted slice


class CreCfg(_StrictModel):
    """Calibrated Reward Engine (D-76). `prior_rel_sd` and `roi_scale` are frozen from tuning seeds."""

    enabled: bool
    mode: Literal["full", "no_uncertainty"]
    proxy_scale: float = Field(gt=0)    # s: proxy units -> gain per dollar x cost, frozen from tuning seeds
    prior_rel_sd: float = Field(gt=0)   # rho: relative sd of the proxy-implied prior
    roi_scale: float = Field(gt=0)      # R_max: benchmark gain at which the reward saturates
    discount: float = Field(gt=0, le=1)  # per-window forgetting of evidence; 1 = off


class CostCfg(_StrictModel):
    count_cpu_verifier_as_gpu_time: bool
    warmup_steps: int = Field(ge=0)


class TokenLimitsCfg(_StrictModel):
    max_prompt_tokens: int = Field(gt=0)
    max_completion_tokens: int = Field(gt=0)
    verifier_timeout_s: float = Field(gt=0)
    prior_usd_per_prompt: float = Field(gt=0)


class Gsm8kEnvCfg(TokenLimitsCfg):
    pass


class CountdownEnvCfg(TokenLimitsCfg):
    n_numbers: int = Field(ge=2, le=5)  # le=5 bounds the reachable-target search
    numbers_min: int = Field(ge=1)
    numbers_max: int = Field(ge=2)
    target_min: int = Field(ge=1)
    target_max: int = Field(ge=2)
    use_all_numbers: bool
    # the prompt allows only exact division; true enforces it in generator and verifier (E-19)
    exact_division: bool

    @field_validator("numbers_max")
    @classmethod
    def _numbers_ordered(cls, hi: int, info):  # noqa: ANN001
        lo = info.data.get("numbers_min")
        if lo is not None and hi < lo:
            raise ValueError("countdown.numbers_max must be >= countdown.numbers_min")
        return hi

    @field_validator("target_max")
    @classmethod
    def _target_ordered(cls, hi: int, info):  # noqa: ANN001
        lo = info.data.get("target_min")
        if lo is not None and hi < lo:
            raise ValueError("countdown.target_max must be >= countdown.target_min")
        return hi


class NoisyEnvCfg(TokenLimitsCfg):
    mode: Literal["random", "flip"]
    q: float = Field(gt=0, lt=1)      # Bernoulli(q) success rate in 'random' mode
    flip_p: float = Field(gt=0, lt=1)  # flip probability in 'flip' mode


class Math35EnvCfg(TokenLimitsCfg):
    level_min: int = Field(ge=1, le=5)
    level_max: int = Field(ge=1, le=5)

    @field_validator("level_max")
    @classmethod
    def _levels_ordered(cls, hi: int, info):  # noqa: ANN001
        lo = info.data.get("level_min")
        if lo is not None and hi < lo:
            raise ValueError("math35.level_max must be >= math35.level_min")
        return hi


class MbppEnvCfg(TokenLimitsCfg):
    memory_limit_mb: int = Field(ge=64)   # sandbox address-space limit (POSIX)
    dev_size: int = Field(ge=1)           # dev items carved from the official train split


class DataEnvsCfg(_StrictModel):
    gsm8k: Gsm8kEnvCfg
    countdown: CountdownEnvCfg
    noisy: NoisyEnvCfg
    math35: Math35EnvCfg
    mbpp: MbppEnvCfg


class DataCfg(_StrictModel):
    root: str
    split_salt: str
    calib_size: int = Field(ge=1)
    dev_size: int = Field(ge=1)
    envs: DataEnvsCfg


class PathsCfg(_StrictModel):
    runs_root: str


class LoggingCfg(_StrictModel):
    fsync: bool
    schema_version: int = Field(ge=1)


class RootConfig(_StrictModel):
    experiment: ExperimentCfg
    budget: BudgetCfg
    scheduler: SchedulerCfg
    baselines: BaselinesCfg
    signals: SignalsCfg
    proxy: ProxyCfg
    calib: CalibCfg
    cre: CreCfg
    cost: CostCfg
    data: DataCfg
    paths: PathsCfg
    logging_cfg: LoggingCfg = Field(alias="logging")
    steps_per_round: int = Field(ge=1)
    prompts_per_step: int = Field(ge=1)
    group_size: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _set_by_path(payload: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = payload
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise CuratorConfigError(
                f"override path '{dotted}' does not exist in the config (missing '{part}')"
            )
        node = node[part]
    if not isinstance(node, dict):
        raise CuratorConfigError(f"override path '{dotted}' traverses a non-mapping")
    node[parts[-1]] = value


def _apply_overrides(payload: dict, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise CuratorConfigError(f"override '{override}' must be of the form a.b=c (missing '=')")
        dotted, raw = override.split("=", 1)
        dotted = dotted.strip()
        if not dotted:
            raise CuratorConfigError(f"override '{override}' has an empty key")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise CuratorConfigError(f"override '{override}' value is not valid YAML: {exc}") from exc
        _set_by_path(payload, dotted, value)


def _deep_merge(base: dict, child: dict) -> dict:
    out = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_payload(path: Path) -> dict:
    """Load a YAML file, resolving `include: [files]` relative to the file, child wins."""
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CuratorConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise CuratorConfigError(f"config file {path} is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise CuratorConfigError(f"config file {path} must contain a YAML mapping")
    include = payload.pop("include", None)
    if include is not None:
        if isinstance(include, str):
            include = [include]
        if not isinstance(include, list) or not all(isinstance(i, str) for i in include):
            raise CuratorConfigError(f"'include' in {path} must be a path or a list of paths")
        base: dict = {}
        for inc in include:
            base = _deep_merge(base, _load_payload(path.parent / inc))
        payload = _deep_merge(base, payload)
    return payload


REFERENCE_PROMPTS_PER_ROUND = 80   # the simulator shape (R = 5 x P = 16) the status thresholds were tuned at


def rescale_for_batch_shape(cfg: RootConfig, prompts_per_round: int) -> RootConfig:
    """Scale evidence thresholds counted in prompt groups to another batch shape (audit E-28).

    `signals.status.n_min` (S1 exit) was tuned at 80 prompts per round, where an arm at a uniform
    share accumulates ~133 discounted groups. At the GPU plan's 32 prompts per round the steady state
    is ~60 < 64, so 77% of arm-rounds stayed in S1 and its exploration quota pinned the mixture near
    uniform. Scaling keeps the threshold's meaning (the same fraction of steady-state evidence).
    """
    if prompts_per_round == REFERENCE_PROMPTS_PER_ROUND:
        return cfg
    n_min = max(8, round(cfg.signals.status.n_min * prompts_per_round / REFERENCE_PROMPTS_PER_ROUND))
    status = cfg.signals.status.model_copy(update={"n_min": n_min})
    return cfg.model_copy(update={"signals": cfg.signals.model_copy(update={"status": status})})


def config_dict_to_canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: RootConfig) -> str:
    """SHA-256 (64 hex chars) of the canonical (sorted-key) JSON form of the config."""
    return hashlib.sha256(config_dict_to_canonical(cfg.model_dump(by_alias=True, mode="json")).encode("utf-8")).hexdigest()


def dump_config(cfg: RootConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(by_alias=True, mode="json"), sort_keys=False), encoding="utf-8")


def load_config(path: Path, overrides: list[str] | tuple[str, ...] = ()) -> RootConfig:
    """Load a YAML config (with include support), apply 'a.b=c' overrides, and validate."""
    payload = _load_payload(Path(path))
    _apply_overrides(payload, list(overrides))
    try:
        return RootConfig.model_validate(payload)
    except ValidationError as exc:
        raise CuratorConfigError(str(exc)) from exc
