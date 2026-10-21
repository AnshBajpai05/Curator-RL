"""Shared helpers for the Phase 4 offline studies (LP estimator selection,
status threshold tuning). All episode collection happens in-process: the
recording scheduler captures the full `RoundObservation` stream and the
harness's own round logs carry the ground-truth diagnostics, so the studies
need no change to the harness, the simulator, or `run_sim.py` (D-38).

Seeding convention (Roadmap E.7): tuning seeds 0..19, evaluation seeds
100..149 — disjoint sets. Truth status labels are anchored to the FROZEN
base.yaml default thresholds (independent of any tuned grid point, D-39).
Every script is deterministic: no RNG besides the seeded sim/streams.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.core.seeding import SeedManager  # noqa: E402
from curator_rl.core.types import EnvStatus, RoundObservation, SignalVector  # noqa: E402
from curator_rl.scheduler.base import BaseScheduler  # noqa: E402
from curator_rl.scheduler.baselines.random_baseline import RandomScheduler  # noqa: E402
from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.signals import SignalEngine  # noqa: E402
from curator_rl.signals.status import flip_rate  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import ScenarioCfg, load_scenario  # noqa: E402

TUNING_SEEDS = list(range(20))
EVAL_SEEDS = list(range(100, 150))
VELOCITY_S3_ANCHOR = 1e-2  # truth: |skill velocity|/round below this => not learning
SR_TRUE_BAND = (0.05, 0.95)  # group-rate truth band considered informative
Z_FPR_THRESHOLD = 2.0  # "significant LP" z-score boundary


@dataclass(frozen=True)
class EpisodeCapture:
    """One episode's full observation stream plus ground-truth diagnostics."""

    scenario_id: str
    seed: int
    scheduler: str
    env_ids: tuple[str, ...]
    observations: tuple[RoundObservation, ...]
    true_skills: tuple[dict[str, float], ...]
    true_pass_rates: tuple[dict[str, float], ...]

    @property
    def rounds(self) -> int:
        return len(self.observations)

    def velocities(self) -> dict[str, list[float]]:
        """Per-env true skill velocity v_t = s_t - s_{t-1}, anchored at s_0."""
        out: dict[str, list[float]] = {}
        for env_id in self.env_ids:
            series = [self.true_skills[i][env_id] for i in range(self.rounds)]
            anchor = [series[0]] + series
            out[env_id] = [series[t] - anchor[t] for t in range(self.rounds)]
        return out


class RecordingScheduler(BaseScheduler):
    """Wraps a real scheduler and records every RoundObservation it sees."""

    def __init__(self, inner, *, tag: str, env_ids: tuple[str, ...]):
        super().__init__(env_ids)
        self.inner = inner
        self.tag = tag
        self.observations: list[RoundObservation] = []

    def select_mixture(self, observation):
        return self.inner.select_mixture(observation)

    def update_observation(self, obs) -> None:
        self.observations.append(obs)
        self.inner.update_observation(obs)


def _make_wrapper(kind: str, scenario: ScenarioCfg, seed: int) -> RecordingScheduler:
    env_ids = tuple(e.env_id for e in scenario.envs)
    if kind == "uniform":
        return RecordingScheduler(UniformScheduler(env_ids), tag=kind, env_ids=env_ids)
    if kind == "random":
        return RecordingScheduler(
            RandomScheduler(env_ids, SeedManager(seed).rng("scheduler")), tag=kind, env_ids=env_ids
        )
    raise ValueError(f"unknown scheduler kind '{kind}'")


def collect_episodes(
    scenario_paths: str | Path | list[str | Path],
    seeds: list[int],
    schedulers: tuple[str, ...] = ("uniform", "random"),
    *,
    budget_cap: float | None = None,
    max_rounds_cap: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[EpisodeCapture]:
    """Run every (scenario, scheduler kind, seed) combination once each."""
    paths = [scenario_paths] if isinstance(scenario_paths, (str, Path)) else list(scenario_paths)
    captures: list[EpisodeCapture] = []
    for path in paths:
        scenario = load_scenario(Path(path))
        if budget_cap is not None:
            scenario = scenario.model_copy(update={"budget_usd": min(scenario.budget_usd, budget_cap)})
        if max_rounds_cap is not None:
            scenario = scenario.model_copy(
                update={"max_rounds": min(scenario.max_rounds, max_rounds_cap)}
            )
        env_ids = tuple(e.env_id for e in scenario.envs)
        for kind in schedulers:
            for seed in seeds:
                sched = _make_wrapper(kind, scenario, seed)
                result = run_episode(sched, scenario, seed)
                captures.append(
                    EpisodeCapture(
                        scenario_id=scenario.scenario_id,
                        seed=seed,
                        scheduler=kind,
                        env_ids=env_ids,
                        observations=tuple(sched.observations),
                        true_skills=tuple(log["true_skills"] for log in result.round_logs),
                        true_pass_rates=tuple(log["true_pass_rates"] for log in result.round_logs),
                    )
                )
        if progress:
            progress(f"scenario {scenario.scenario_id} done ({len(captures)} captures)")
    return captures


def run_engine(
    capture: EpisodeCapture,
    signals_cfg,
    proxy_cfg,
    calib_cfg,
    cost_exponent: float,
    group_size: int,
    *,
    status_cfg=None,
) -> list[dict[str, SignalVector]]:
    """Run a SignalEngine (configurable lp_method / status cfg) over one capture."""
    cfg = signals_cfg
    if status_cfg is not None:
        payload = signals_cfg.model_dump()
        payload["status"] = status_cfg.model_dump()
        cfg = type(signals_cfg).model_validate(payload)
    engine = SignalEngine(cfg, proxy_cfg, calib_cfg, cost_exponent, group_size, capture.env_ids)
    return [engine.update(obs) for obs in capture.observations]


def precompute_status_inputs(capture: EpisodeCapture, signals_cfg, group_size: int):
    """Classifier inputs per (round, env), status-independent (grid-point reuse).

    The status classifier consumes only (n_groups_eff, rounds_seen, p_hat,
    p_lo, p_hi, z_lp, sr) — none of which depend on the status thresholds —
    so the study runs this pipeline once per capture and replays the cheap
    prototype classifier per grid point.

    Returns list of per-round dicts: env_id -> {"n_groups_eff", "rounds_seen",
    "p_hat", "p_lo", "p_hi", "z_lp", "sr", "raw_rate"}.
    """
    from curator_rl.signals.passrate import PassRateTracker
    from curator_rl.signals.progress import LPEstimator
    from curator_rl.signals.richness import RichnessEstimator

    tracker = PassRateTracker(
        signals_cfg.lam, signals_cfg.lambda_fast, signals_cfg.lambda_slow,
        signals_cfg.prior.alpha0, signals_cfg.prior.beta0,
    )
    lp = LPEstimator(signals_cfg.lp_method, signals_cfg.window_rounds)
    richness = RichnessEstimator(signals_cfg.richness, signals_cfg.lam, group_size)

    rounds_out: list[dict[str, dict[str, float]]] = []
    for obs in capture.observations:
        row: dict[str, dict[str, float]] = {}
        for env_id in sorted(capture.env_ids):
            env_obs = obs.per_env.get(env_id, _zero_env_obs(env_id))
            posterior = tracker.update(env_obs)
            raw_rate = env_obs.k_success / env_obs.n_rollouts if env_obs.n_rollouts > 0 else None
            lp.observe(env_id, posterior, env_obs.n_prompts, raw_rate)
            _, _, z = lp.compute(env_id, posterior)
            sr = richness.update(env_id, env_obs, posterior)
            row[env_id] = {
                "n_groups_eff": posterior.n_groups_eff,
                "rounds_seen": posterior.n_rounds_seen,
                "p_hat": posterior.pass_rate,
                "p_lo": posterior.pass_lo,
                "p_hi": posterior.pass_hi,
                "z_lp": z,
                "sr": sr,
                "raw_rate": raw_rate if raw_rate is not None else float("nan"),
            }
        rounds_out.append(row)
    return rounds_out


def _zero_env_obs(env_id: str):
    from curator_rl.core.types import EnvRoundObs

    return EnvRoundObs(
        env_id=env_id, n_prompts=0, n_rollouts=0, k_success=0, n_groups_mixed=0,
        sum_score=0.0, sum_score_sq=0.0, prompt_tokens=0, completion_tokens=0,
        verifier_seconds=0.0, gpu_seconds=0.0, cost_usd=0.0,
    )


def classify_with_status(status_cfg, precomputed: list[dict[str, dict[str, float]]]):
    """Per-round statuses from one grid point's StatusCfg (deterministic replay)."""
    from curator_rl.core.config import CalibCfg  # noqa: F401  (streaks come as 0 here)

    envs = sorted(precomputed[0]) if precomputed else []
    traces: dict[str, list[str]] = {e: [] for e in envs}

    for env_id in envs:
        from curator_rl.signals.status import StatusClassifier

        cl = StatusClassifier(status_cfg, mismatch_windows=1, clear_windows=10)
        for round_row in precomputed:
            fields = round_row[env_id]
            status, _ = cl.step(
                n_groups_eff=fields["n_groups_eff"],
                rounds_seen=int(fields["rounds_seen"]),
                p_hat=fields["p_hat"],
                p_lo=fields["p_lo"],
                p_hi=fields["p_hi"],
                z_lp=fields["z_lp"],
                sr=fields["sr"],
            )
            traces[env_id].append(status_name(status))
    return traces


# --------------------------------------------------------------- truth labels

def sr_true_of(p_true: float) -> float:
    """Oracle richness: informative bands carry mostly informative groups."""
    return 1.0 if SR_TRUE_BAND[0] < p_true < SR_TRUE_BAND[1] else 0.05


def true_status(p_true: float, velocity: float, cfg_anchor) -> EnvStatus:
    """Dwell-free oracle status from ground truth (anchored thresholds, D-39)."""
    if p_true >= cfg_anchor.p_sat and abs(velocity) < VELOCITY_S3_ANCHOR:
        return EnvStatus.S3
    if p_true <= cfg_anchor.p_hard and sr_true_of(p_true) <= cfg_anchor.sr_hard:
        return EnvStatus.S4
    return EnvStatus.S2


def truth_labels(capture: EpisodeCapture, cfg_anchor, *, lam: float = 0.9) -> dict[str, list[EnvStatus]]:
    """Sequential (replayed) truth status per env (D-39 / D-40).

    S1 at truth mirrors prediction-side semantics: discounted group evidence
    (allocation bookkeeping, not a hidden signal) below the S1 entry floor
    (`s1_entry_ratio * n_min`, or `n_min` while already in S1), or fewer than
    r_min observed prompt rounds, means 'no evidence yet'. The learning labels
    (S2/S3/S4) come from the true pass rate and true skill velocity.
    """
    velocities = capture.velocities()
    out: dict[str, list[EnvStatus]] = {}
    for env_id in capture.env_ids:
        groups_eff = 0.0
        seen = 0
        in_s1 = True
        labels: list[EnvStatus] = []
        for t in range(capture.rounds):
            m = capture.observations[t].per_env[env_id].n_prompts
            groups_eff = lam * groups_eff + m
            if m > 0:
                seen += 1
            floor = cfg_anchor.n_min * cfg_anchor.s1_entry_ratio if not in_s1 else cfg_anchor.n_min
            if seen < cfg_anchor.r_min or groups_eff < floor:
                labels.append(EnvStatus.S1)
                in_s1 = True
            else:
                in_s1 = False
                labels.append(
                    true_status(capture.true_pass_rates[t][env_id], velocities[env_id][t], cfg_anchor)
                )
        out[env_id] = labels
    return out


# -------------------------------------------------------------------- metrics

def macro_f1(true: list[str], pred: list[str]) -> float:
    """Macro-F1 over the classes present in `true`."""
    if not true:
        return 0.0
    labels = sorted(set(true))
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom else 1.0)
    return sum(f1s) / len(f1s)


def status_name(status: object) -> str:
    return status.value if hasattr(status, "value") else str(status)


def status_traces(vectors_by_round: list[dict[str, SignalVector]]) -> dict[str, list[str]]:
    """Per-env status-name traces from one engine run."""
    envs = sorted(vectors_by_round[0])
    out: dict[str, list[str]] = {e: [] for e in envs}
    for per_round in vectors_by_round:
        for env_id in envs:
            out[env_id].append(status_name(per_round[env_id].status))
    return out


def flip_rate_after_first_exit(trace: list[str]) -> float:
    """Flip rate measured strictly after the env's first S1 exit (the designed
    warm-up handover); per-100-rounds conversions via the status module."""
    idx = next((i for i, s in enumerate(trace) if s != "S1_unexplored"), None)
    if idx is None:
        return 0.0
    return flip_rate([EnvStatus(s) for s in trace], warmup=idx + 1)


def flip_rate_pooled(traces: dict[str, list[str]]) -> float:
    """Pooled flip rate: total post-first-exit transitions per 100 intervals.

    Pooled because a per-env mean of rates is dominated by short tails (an
    arm that exits S1 near the end of a run gets a 2-interval denominator and
    every single transition becomes flip=100).
    """
    transitions = 0
    intervals = 0
    for trace in traces.values():
        idx = next((i for i, s in enumerate(trace) if s != "S1_unexplored"), None)
        if idx is None:
            continue
        tail = trace[idx + 1:]
        transitions += sum(1 for a, b in zip(tail, tail[1:]) if a != b)
        intervals += max(len(tail), 1)
    return transitions * 100.0 / intervals if intervals else 0.0


def mean_flip_rate(traces: dict[str, list[str]], *, warmup: int = 0) -> float:
    """Flip-rate metric for the studies.

    warmup < 0 => pooled post-first-exit rate (D-41). warmup k >= 0 => plain
    per-100-rounds rate after k rounds.
    """
    if warmup < 0:
        return flip_rate_pooled(traces)
    values = [flip_rate([EnvStatus(s) for s in trace], warmup=warmup) for trace in traces.values()]
    return sum(values) / max(len(values), 1)


def default_cfg(root: Path | None = None):
    """(signals, proxy, calib, cost_exponent, group_size, status_anchor) from base.yaml."""
    path = (root or REPO_ROOT) / "configs" / "base.yaml"
    cfg = load_config(path)
    return (
        cfg.signals,
        cfg.proxy,
        cfg.calib,
        cfg.scheduler.cost_exponent,
        cfg.group_size,
        cfg.signals.status,
    )
