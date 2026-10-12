"""Simulator experiment runner (Roadmap F.5, v1 Phase 3).

Runs a scenario with any number of methods and seeds, then reports final
benchmark scores and oracle-gap closure:

    python experiments/run_sim.py --scenario configs/sim/scenario_sa.yaml \
        --methods uniform,random,static_oracle,myopic_oracle,dp_oracle --seeds 20

Writes a JSON report to reports/sim/ and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from curator_rl.core.seeding import SeedManager  # noqa: E402
from curator_rl.scheduler.baselines.random_baseline import RandomScheduler  # noqa: E402
from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.simulator.harness import (  # noqa: E402
    EpisodeResult,
    run_episode,
    scores_summary,
)
from curator_rl.simulator.oracle import DPOracle, MyopicOracle, StaticOracle  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402

DEFAULT_SEEDS = 20
DEFAULT_METHODS = ("uniform", "random", "static_oracle", "myopic_oracle", "dp_oracle")
_SCHEDULER_METHODS = ("static", "pilot_uniform", "lp", "ucb", "sec", "dump", "curator")
_BASE_CFG: dict | None = None
# Configuration of every pre-registered gate and study from D-68 to D-94 (audit E-04): their
# scripts load this frozen file, never the live base.yaml, so they keep reproducing.
FROZEN_D68_D94 = REPO_ROOT / "configs" / "frozen" / "d68_d94.yaml"


def _load_base_config():
    """Lazily load configs/base.yaml once (D-44: methods via overrides, no YAML per method)."""
    global _BASE_CFG
    if _BASE_CFG is None:
        from curator_rl.core.config import load_config

        _BASE_CFG = load_config(REPO_ROOT / "configs" / "base.yaml")
    return _BASE_CFG


def _static_weights(scenario) -> dict[str, float]:
    """Pre-registered static mixture (Roadmap J.1 #2, D-63).

    Size-proportional on the declared `nominal_size` when every arm has one;
    otherwise the benchmark slice weights pi_d (D-45), else uniform. All three
    are visible before training and never a hidden training signal.
    """
    envs = scenario.envs
    if all(e.nominal_size is not None for e in envs):
        raw = [float(e.nominal_size) for e in envs]
    else:
        raw = [float(e.bench_weight) if e.bench_weight is not None else 1.0 for e in envs]
    total = sum(raw)
    return {e.env_id: r / total for e, r in zip(envs, raw)}


def _pilot_uniform_weights(scenario) -> dict[str, float]:
    """Improvement B10: uniform over the arms the pilot could solve (pass@G > 0), zero elsewhere.

    What a practitioner does after the pilot: drop arms with no signal (Countdown: 0 of 1,024
    samples). The noisy arm stays in: the pilot shows it as the easiest arm (pass@8 = 0.98), and
    nothing in a pilot reveals that its reward is random.
    """
    if any(e.pilot_pass_at_g is None for e in scenario.envs):
        raise ValueError("pilot_uniform needs pilot_pass_at_g on every arm of the scenario")
    keep = [e.env_id for e in scenario.envs if e.pilot_pass_at_g > 0]
    if not keep:
        raise ValueError("no arm has pilot pass@G > 0")
    return {e.env_id: (1.0 / len(keep) if e.env_id in keep else 0.0) for e in scenario.envs}


def make_scheduler(
    method: str, scenario, seed: int, *, dp_levels: int = 80, dp_units: int | None = None, cfg=None
):
    """Construct a scheduler-like object for one episode (Roadmap F.5).

    `cfg` overrides the cached `configs/base.yaml` (tuning sweeps pass variants).
    """
    env_ids = [e.env_id for e in scenario.envs]
    if method == "uniform":
        return UniformScheduler(env_ids)
    if method == "random":
        return RandomScheduler(env_ids, SeedManager(seed).rng("scheduler"))
    if method in _SCHEDULER_METHODS:
        from curator_rl.scheduler.baselines.dump import DUMPStyleUCB
        from curator_rl.scheduler.baselines.lp import LPCurriculum
        from curator_rl.scheduler.baselines.sec import SECStyleBandit
        from curator_rl.scheduler.baselines.static import StaticMixtureScheduler
        from curator_rl.scheduler.baselines.ucb import StandardUCB
        from curator_rl.scheduler.curator import Curator

        cfg = cfg if cfg is not None else _load_base_config()
        prompts_per_round = scenario.steps_per_round * scenario.prompts_per_step
        if method == "static":
            return StaticMixtureScheduler(env_ids, _static_weights(scenario))
        if method == "pilot_uniform":
            return StaticMixtureScheduler(env_ids, _pilot_uniform_weights(scenario))
        if method == "lp":
            lp_sched = cfg.scheduler.model_copy(update={"tau": cfg.baselines.lp.tau})
            return LPCurriculum(env_ids, cfg.signals, cfg.proxy, lp_sched, cfg.calib, cfg.group_size)
        if method == "ucb":
            ucb_sched = cfg.scheduler.model_copy(
                update={
                    "exploration_coef": cfg.baselines.ucb.exploration_coef,
                    "tau": cfg.baselines.ucb.tau,
                }
            )
            return StandardUCB(
                env_ids, cfg.signals, cfg.proxy, ucb_sched, cfg.calib, cfg.group_size,
                prompts_per_round=prompts_per_round,
            )
        if method == "sec":
            return SECStyleBandit(env_ids, cfg.baselines.sec, cfg.scheduler)
        if method == "dump":
            return DUMPStyleUCB(env_ids, cfg.baselines.dump, cfg.scheduler)
        from curator_rl.simulator.scenarios import domain_map

        return Curator(
            env_ids, cfg.signals, cfg.proxy, cfg.scheduler, cfg.calib, cfg.group_size,
            prompts_per_round=prompts_per_round, cre_cfg=cfg.cre, domain_of=domain_map(scenario),
        )
    if method == "static_oracle":
        from curator_rl.simulator.scenarios import build_world

        world = build_world(scenario, seed)
        return StaticOracle(
            world,
            simplex_step=scenario.oracle.simplex_step,
            random_search_draws=scenario.oracle.random_search_draws,
            seed=seed,
        )
    if method == "myopic_oracle":
        from curator_rl.simulator.scenarios import build_world

        return MyopicOracle(build_world(scenario, seed))
    if method == "dp_oracle":
        from curator_rl.simulator.scenarios import build_world

        return DPOracle(
            build_world(scenario, seed),
            skill_levels=dp_levels,
            budget_units=dp_units or scenario.oracle.budget_units,
            simplex_step=scenario.oracle.simplex_step,
        )
    raise ValueError(f"unknown method '{method}'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run simulator episodes (Roadmap F.5)")
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed0", type=int, default=0, help="first seed; seeds are consecutive")
    parser.add_argument(
        "--dp-levels", type=int, default=80,
        help="DP oracle skill grid resolution (3 learnable dims are O(levels^3) — lower it there)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))

    results: dict[str, list[EpisodeResult]] = {}
    timings: dict[str, float] = {}
    for method in methods:
        t0 = time.perf_counter()
        episodes: list[EpisodeResult] = []
        for seed in seeds:
            try:
                scheduler = make_scheduler(method, scenario, seed, dp_levels=args.dp_levels)
            except ValueError as exc:
                print(f"[skip] {method}: {exc}")
                episodes = []
                break
            episodes.append(run_episode(scheduler, scenario, seed, method=method))
        results[method] = episodes
        timings[method] = time.perf_counter() - t0

    summary: dict[str, dict] = {}
    for method, eps in results.items():
        if not eps:
            summary[method] = {"error": "not runnable on this scenario"}
            continue
        summary[method] = {
            **scores_summary([e.final_score for e in eps]),
            "mean_rounds": scores_summary([float(e.rounds) for e in eps])["mean"],
            "mean_cost_usd": scores_summary([e.total_cost_usd for e in eps])["mean"],
            "seconds": round(timings[method], 2),
        }

    # oracle-gap closure of every method against the best available oracle
    ref_uniform = summary.get("uniform", {}).get("mean")
    oracle_score = max(
        (summary[m]["mean"] for m in ("dp_oracle", "static_oracle") if m in summary and "mean" in summary[m]),
        default=None,
    )
    if ref_uniform is not None and oracle_score is not None:
        for _method, stats in summary.items():
            if "mean" not in stats:
                continue
            denom = oracle_score - ref_uniform
            stats["oracle_gap_closure"] = (
                (stats["mean"] - ref_uniform) / denom if abs(denom) > 1e-12 else None
            )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (REPO_ROOT / "reports" / "sim" / f"{scenario.scenario_id}_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": scenario.scenario_id,
        "scenario_file": str(args.scenario),
        "seeds": seeds,
        "methods": methods,
        "summary": summary,
        "generated_utc": stamp,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nScenario {scenario.scenario_id} — final benchmark score (mean ± std, n seeds)")
    for method, stats in summary.items():
        if "mean" not in stats:
            print(f"  {method:<16} ERROR: {stats.get('error')}")
            continue
        gap = stats.get("oracle_gap_closure")
        gap_s = f"  gap={gap * 100:5.1f}%" if gap is not None else ""
        print(
            f"  {method:<16} {stats['mean']:.4f} ± {stats['std']:.4f}"
            f"  rounds={stats['mean_rounds']:.1f}  cost={stats['mean_cost_usd']:.3f}"
            f"  ({stats['seconds']}s){gap_s}"
        )
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
