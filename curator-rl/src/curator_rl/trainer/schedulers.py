"""Schedulers for real runs (the same objects the simulator validated) plus the sysid design.

`make_run_scheduler` builds any method of the comparison from `configs/base.yaml` exactly as
`experiments/run_sim.make_scheduler` does in the simulator, so a method's GPU behaviour is the
behaviour its simulator gates measured. `DirichletDesign` is not a method: it is the
system-identification run's experimental design (a fresh random mixture every round, floor eps/N),
which makes per-arm costs and learning rates identifiable from one run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from curator_rl.core.seeding import SeedManager
from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import BaseScheduler


class DirichletDesign(BaseScheduler):
    """Random mixtures w ~ (1 - eps) Dirichlet(1) + eps/N, redrawn every round (sysid only)."""

    def __init__(self, env_ids: Sequence[str], seed: int, epsilon: float) -> None:
        super().__init__(env_ids)
        self._rng = SeedManager(seed).rng("dirichlet_design")
        self._eps = float(epsilon)

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        n = len(self.env_ids)
        draw = self._rng.dirichlet(np.ones(n))
        w = (1.0 - self._eps) * draw + self._eps / n
        return {e: float(v) for e, v in zip(self.env_ids, w / w.sum())}

    def get_state(self) -> dict:
        return {"rng": self._rng.bit_generator.state}

    def load_checkpoint(self, state) -> None:
        self._rng.bit_generator.state = state["rng"]  # pyright: ignore[reportAttributeAccessIssue]


def make_run_scheduler(method: str, cfg, env_ids: Sequence[str], *, seed: int,
                       prompts_per_round: int, domain_of: Mapping[str, str | None],
                       nominal_sizes: Mapping[str, float]):
    """Construct a scheduler for a real run from a validated RootConfig."""
    from curator_rl.scheduler.baselines.dump import DUMPStyleUCB
    from curator_rl.scheduler.baselines.lp import LPCurriculum
    from curator_rl.scheduler.baselines.sec import SECStyleBandit
    from curator_rl.scheduler.baselines.static import StaticMixtureScheduler
    from curator_rl.scheduler.baselines.ucb import StandardUCB
    from curator_rl.scheduler.baselines.uniform import UniformScheduler
    from curator_rl.scheduler.curator import Curator

    env_ids = list(env_ids)
    from curator_rl.trainer.group_filter import split_method

    method, _ = split_method(method)          # "_q" = the same scheduler plus question filtering (D-102)
    if method == "uniform":
        return UniformScheduler(env_ids)
    if method == "dirichlet":
        return DirichletDesign(env_ids, seed, cfg.scheduler.epsilon)
    if method == "static":                      # size-proportional (D-61/D-63)
        total = sum(float(nominal_sizes[e]) for e in env_ids)
        return StaticMixtureScheduler(env_ids, {e: float(nominal_sizes[e]) / total for e in env_ids})
    if method == "clean_uniform":               # D-103: equal mix of the arms that are neither junk nor unsolvable:
        keep = [e for e in env_ids if e not in ("countdown", "noisy", "hackable")]   # a perfect junk detector
        return StaticMixtureScheduler(env_ids, {e: (1.0 / len(keep) if e in keep else 0.0) for e in env_ids})
    if method == "pilot_uniform":               # improvement B10: drop arms the pilot could not solve
        keep = [e for e in env_ids if e != "countdown"]
        return StaticMixtureScheduler(env_ids, {e: (1.0 / len(keep) if e in keep else 0.0) for e in env_ids})
    if method == "lp":
        sched = cfg.scheduler.model_copy(update={"tau": cfg.baselines.lp.tau})
        return LPCurriculum(env_ids, cfg.signals, cfg.proxy, sched, cfg.calib, cfg.group_size)
    if method == "ucb":
        sched = cfg.scheduler.model_copy(update={"exploration_coef": cfg.baselines.ucb.exploration_coef,
                                                 "tau": cfg.baselines.ucb.tau})
        return StandardUCB(env_ids, cfg.signals, cfg.proxy, sched, cfg.calib, cfg.group_size,
                           prompts_per_round=prompts_per_round)
    if method == "sec":
        return SECStyleBandit(env_ids, cfg.baselines.sec, cfg.scheduler)
    if method == "dump":
        return DUMPStyleUCB(env_ids, cfg.baselines.dump, cfg.scheduler)
    if method == "curator" or method.startswith("curator_"):   # variants differ only in config (e.g. curator_nocal)
        return Curator(env_ids, cfg.signals, cfg.proxy, cfg.scheduler, cfg.calib, cfg.group_size,
                       prompts_per_round=prompts_per_round, cre_cfg=cfg.cre, domain_of=dict(domain_of))
    raise ValueError(f"unknown method '{method}'")
