"""Scenario specifications and builders (Roadmap F.2/F.3, v1 Phase 3).

A scenario is a parameter setting of the world model, not separate code: the
scenario suite S-A..S-H is one YAML file each under `configs/sim/`. Parameters
are validated with pydantic (strict, extra=forbid) so a typo in a scenario
file fails loudly. `build_world` draws the concrete world for a seed; the
same (scenario, seed) pair always yields the same world.

Scenario suite (Roadmap F.3):

| ID | N | Contents                                   | Primary question |
|----|---|--------------------------------------------|------------------|
| S-A | 3 | easy, valuable-slow, noisy                 | drop noisy/saturated? |
| S-B | 8 | full portfolio analogue incl. noisy, too-hard | realistic comparison |
| S-C | 4 | cost heterogeneity 1x..8x, equal gains     | cost normalisation (H2) |
| S-D | 4 | transfer and interference matrix           | credit rules (H5) |
| S-E | 3 | abrupt regime change in one env            | discounting (H4) |
| S-F | 3 | too-hard unlockable via transfer           | starvation/revival |
| S-G | 8 | evaluation-noise sweep (n_b, K)            | choice of K, set size |
| S-H | 5 | late-arriving environment                  | cold start |
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from curator_rl.core.seeding import SeedManager
from curator_rl.simulator.world import SimEnvParams, SimWorld


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateCfg(_StrictModel):
    prereq: str
    theta: float = 0.0
    slope: float = 4.0


class DriftCfg(_StrictModel):
    round: int = Field(ge=1)
    shift: float


class SimEnvCfg(_StrictModel):
    env_id: str
    skill0: float = 0.0
    slope: float = 2.0
    difficulty: float = 0.0
    eta: float = Field(gt=0)
    cost_usd_per_prompt: float = Field(gt=0)
    transfer_out: dict[str, float] = Field(default_factory=dict)
    noisy_q: float | None = None
    gate: GateCfg | None = None
    drift: DriftCfg | None = None
    late_start_round: int = Field(default=0, ge=0)
    cost_skill_slope: float = 0.0
    bench_slope: float | None = None
    bench_difficulty: float | None = None
    bench_weight: float | None = None
    # calibration slice of ANOTHER arm this arm is scored on (audit E-03), e.g. noisy -> gsm8k
    bench_domain: str | None = None
    # Declared dataset size (D-63): the a-priori, scheduler-visible quantity the
    # Static baseline is proportional to. Never used by the world or oracles.
    nominal_size: float | None = Field(default=None, gt=0)
    # B7: per-round skill decay toward skill0 while the arm is not trained (0 = no forgetting)
    forget_rate: float = Field(default=0.0, ge=0, le=1)
    # E-14: finite training set the world counts passes over (None = inexhaustible)
    train_items: int | None = Field(default=None, ge=1)
    # B10: pass@G measured in the pilot, visible before training (pilot_uniform drops arms at 0)
    pilot_pass_at_g: float | None = Field(default=None, ge=0, le=1)


class CalibSimCfg(_StrictModel):
    """Calibration evaluations offered to schedulers that use them (charged to them only, D-72).

    `paired`: evaluate the same fixed items every time (an item is solved iff its
    latent u < p), so a window's change has only flip variance, as with greedy
    decoding on a fixed calibration set. `churn`: fraction of items whose latent
    is redrawn at each evaluation (random flips unrelated to skill).
    """

    enabled: bool = False
    interval_rounds: int = Field(default=5, ge=1)
    cost_usd: float = Field(default=0.01, ge=0)
    paired: bool = False
    churn: float = Field(default=0.02, ge=0, le=1)
    # Calibration slice size (None = world.benchmark_items, the final-eval size) and,
    # when set, a per-item cost so that cost and statistical power stay consistent:
    # eval cost = cost_per_item_usd * items_per_slice * n_slices (D-74).
    items_per_slice: int | None = Field(default=None, ge=1)
    cost_per_item_usd: float | None = Field(default=None, ge=0)

    def eval_cost(self, n_slices: int, default_items: int) -> float:
        if self.cost_per_item_usd is None:
            return self.cost_usd
        return self.cost_per_item_usd * (self.items_per_slice or default_items) * n_slices

    def eval_cost_items(self, n_items_total: int) -> float:
        """Cost of a targeted evaluation of `n_items_total` items (D-75)."""
        if self.cost_per_item_usd is None:
            return self.cost_usd
        return self.cost_per_item_usd * n_items_total


class ReportEvalCfg(_StrictModel):
    """Uncharged reporting evaluations (Roadmap L.4); 0 disables (D-46)."""

    interval_rounds: int = Field(default=0, ge=0)


class SimWorldCfg(_StrictModel):
    skill_noise_std: float = Field(default=0.003, ge=0)
    cost_lognormal_sigma: float = Field(default=0.10, ge=0)
    rollout_concentration: float = Field(default=25.0, gt=0)
    benchmark_items: int = Field(default=200, ge=1)
    # audit E-09: legacy scenarios train on requested weights with per-round largest-remainder
    # rounding; `delivered` + `carry` train on the prompts actually served, as a real trainer does
    learning_from: Literal["requested", "delivered"] = "requested"
    rounding: Literal["largest_remainder", "carry"] = "largest_remainder"
    # realism switches (world.py docstring); defaults reproduce the original world exactly
    learnability: Literal["binary", "mixed_groups"] = "binary"           # B8
    repeat_decay: float = Field(default=1.0, gt=0, le=1)                  # E-14 (needs train_items)
    noisy_transfer: bool = False                                          # B9
    report_stream: Literal["shared", "separate"] = "shared"               # B1 / E9
    final_eval: Literal["binomial", "paired_items"] = "binomial"          # B1
    rollout_draws: Literal["loop", "vectorized"] = "loop"                 # E10


class OracleCfg(_StrictModel):
    budget_units: int = Field(default=40, ge=5)
    simplex_step: float = Field(default=0.1, gt=0, le=0.5)
    random_search_draws: int = Field(default=1500, ge=1)


class ScenarioCfg(_StrictModel):
    scenario_id: str
    description: str = ""
    budget_usd: float = Field(gt=0)
    max_rounds: int = Field(default=200, ge=1)
    steps_per_round: int = Field(default=5, ge=1)
    prompts_per_step: int = Field(default=16, ge=1)
    group_size: int = Field(default=8, ge=2)
    calib: CalibSimCfg = CalibSimCfg()
    report_eval: ReportEvalCfg = ReportEvalCfg()
    world: SimWorldCfg = SimWorldCfg()
    oracle: OracleCfg = OracleCfg()
    envs: list[SimEnvCfg] = Field(min_length=1)

    @property
    def prompts_per_round(self) -> int:
        return self.steps_per_round * self.prompts_per_step


def load_scenario(path: Path) -> ScenarioCfg:
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"scenario file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"scenario file {path} is not valid YAML: {exc}") from exc
    try:
        cfg = ScenarioCfg.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surfaced with the file name
        raise ValueError(f"invalid scenario file {path}: {exc}") from exc
    ids = [e.env_id for e in cfg.envs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate env ids in {path}: {ids}")
    for env in cfg.envs:
        if env.gate is not None and env.gate.prereq not in ids:
            raise ValueError(f"{path}: gate prereq '{env.gate.prereq}' is not a scenario env")
        for target in env.transfer_out:
            if target not in ids:
                raise ValueError(f"{path}: transfer target '{target}' is not a scenario env")
            if target == env.env_id:
                raise ValueError(f"{path}: T_ii = 1 is implicit; remove self-transfer on '{env.env_id}'")
        if env.noisy_q is not None and env.transfer_out and not cfg.world.noisy_transfer:
            raise ValueError(f"{path}: transfer_out on noisy arm '{env.env_id}' needs world.noisy_transfer: true")
        if env.bench_domain is not None:
            owner = next((o for o in cfg.envs if o.env_id == env.bench_domain), None)
            if owner is None or owner.env_id == env.env_id or owner.bench_domain is not None:
                raise ValueError(f"{path}: bench_domain of '{env.env_id}' must name another arm "
                                 "that has its own slice")
    return cfg


def domain_map(cfg: ScenarioCfg) -> dict[str, str | None]:
    """Calibration slice of every arm (its own id unless `bench_domain` says otherwise)."""
    return {e.env_id: (e.bench_domain or e.env_id) for e in cfg.envs}


def scenario_env_params(cfg: ScenarioCfg) -> list[SimEnvParams]:
    envs: list[SimEnvParams] = []
    for e in cfg.envs:
        envs.append(
            SimEnvParams(
                env_id=e.env_id,
                skill0=e.skill0,
                slope=e.slope,
                difficulty=e.difficulty,
                eta=e.eta,
                cost_usd_per_prompt=e.cost_usd_per_prompt,
                transfer_out=dict(e.transfer_out),
                noisy_q=e.noisy_q,
                gate_prereq=e.gate.prereq if e.gate else None,
                gate_theta=e.gate.theta if e.gate else 0.0,
                gate_slope=e.gate.slope if e.gate else 4.0,
                drift_round=e.drift.round if e.drift else None,
                drift_shift=e.drift.shift if e.drift else 0.0,
                late_start_round=e.late_start_round,
                cost_skill_slope=e.cost_skill_slope,
                bench_slope=math.nan if e.bench_slope is None else e.bench_slope,
                bench_difficulty=math.nan if e.bench_difficulty is None else e.bench_difficulty,
                bench_weight=e.bench_weight,
                bench_domain=e.bench_domain,
                forget_rate=e.forget_rate,
                train_items=e.train_items,
            )
        )
    return envs


def build_world(cfg: ScenarioCfg, seed: int) -> SimWorld:
    """Concrete world for (scenario, seed); deterministic in both (F.5)."""
    seeds = SeedManager(seed)
    world = SimWorld(
        scenario_env_params(cfg),
        steps_per_round=cfg.steps_per_round,
        prompts_per_round=cfg.prompts_per_round,
        group_size=cfg.group_size,
        budget_usd=cfg.budget_usd,
        skill_noise_std=cfg.world.skill_noise_std,
        cost_lognormal_sigma=cfg.world.cost_lognormal_sigma,
        rollout_concentration=cfg.world.rollout_concentration,
        rng=seeds.rng("sim_world"),
        calib_rng=seeds.rng("sim_calib"),
        learning_from=cfg.world.learning_from,
        rounding=cfg.world.rounding,
        learnability_mode=cfg.world.learnability,
        repeat_decay=cfg.world.repeat_decay,
        noisy_transfer=cfg.world.noisy_transfer,
        report_rng=seeds.rng("sim_report") if cfg.world.report_stream == "separate" else None,
        test_item_rng=seeds.rng("sim_test_items"),
        final_eval=cfg.world.final_eval,
        rollout_draws=cfg.world.rollout_draws,
    )
    world.set_benchmark_items(cfg.world.benchmark_items)
    return world
