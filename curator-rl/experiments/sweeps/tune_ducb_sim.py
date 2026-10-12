"""D-UCB hyperparameter tuning sweep (Roadmap E.7, v1 Phase 6 = v2 Phase 6).

Grid over gamma / kappa / tau / epsilon on simulator tuning seeds (0-9) over
S-A (drop saturated + noisy discrimination) and S-C (cost heterogeneity, where
cost normalisation pays). Objective: mean relative final-score improvement over
Uniform; the per-point junk (noisy-arm, post-warm-up) share is recorded as a
diagnostic (H3: the noisy arm is not reliably droppable pre-calibration).

    python experiments/sweeps/tune_ducb_sim.py --seeds 10

Writes reports/sweeps/tune_ducb.{json,md}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import UTC, datetime

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.scheduler.curator import Curator  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.run_sim import FROZEN_D68_D94  # noqa: E402  (audit E-04: pre-registered config)

REPORT_DIR = REPO_ROOT / "reports" / "sweeps"
GRID = {
    "gamma": [0.90, 0.95, 0.98],
    "exploration_coef": [0.25, 0.5, 1.0],
    "tau": [0.3, 1.0, 2.0],
    "epsilon": [0.05, 0.10],
}
WARMUP_SHARE_FROM = 16  # post-warm-up window for the junk-share diagnostic


def run_point(point, scenario_paths, seeds, base_cfg):
    scores = {}
    junk_shares = {}
    for path in scenario_paths:
        scenario = load_scenario(str(path))
        env_ids = [e.env_id for e in scenario.envs]
        noisy = next((e.env_id for e in scenario.envs if e.noisy_q is not None), None)
        sched_cfg = base_cfg.scheduler.model_copy(update=point)
        runs = []
        shares = []
        for seed in seeds:
            scheduler = Curator(
                env_ids, base_cfg.signals, base_cfg.proxy, sched_cfg, base_cfg.calib,
                base_cfg.group_size,
                prompts_per_round=scenario.steps_per_round * scenario.prompts_per_step,
            )
            result = run_episode(scheduler, scenario, seed=seed, method="curator")
            runs.append(result.final_score)
            if noisy:
                late = [log for log in result.round_logs if log["round"] > WARMUP_SHARE_FROM]
                if late:
                    shares.append(sum(log["weights"][noisy] for log in late) / len(late))
        scores[scenario.scenario_id] = runs
        if noisy:
            junk_shares[scenario.scenario_id] = sum(shares) / max(len(shares), 1)
    return scores, junk_shares


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune gamma/kappa/tau/epsilon (Roadmap E.7)")
    parser.add_argument("--scenarios", default="S-A,S-C")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args(argv)

    sim_dir = REPO_ROOT / "configs" / "sim"
    wanted = {s.strip().replace("-", "").lower() for s in args.scenarios.split(",")}
    paths = []
    for p in sorted(sim_dir.glob("scenario_*.yaml")):
        sid = load_scenario(str(p)).scenario_id.replace("-", "").lower()
        if sid in wanted:
            paths.append(p)
    seeds = list(range(args.seeds))
    base_cfg = load_config(FROZEN_D68_D94)

    # uniform reference per scenario/seed
    uniform_scores: dict[str, list[float]] = {}
    for path in paths:
        scenario = load_scenario(str(path))
        scheduler = UniformScheduler([e.env_id for e in scenario.envs])
        uniform_scores[scenario.scenario_id] = [
            run_episode(scheduler, scenario, seed=seed, method="uniform").final_score for seed in seeds
        ]

    keys = list(GRID)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, values in enumerate(itertools.product(*GRID.values())):
        point = dict(zip(keys, values))
        scores, junk = run_point(point, paths, seeds, base_cfg)
        rel = {}
        for sid, runs in scores.items():
            ref = uniform_scores[sid]
            rel[sid] = sum((c - u) / u for c, u in zip(runs, ref)) / len(runs)
        rows.append({
            "point": point,
            "rel_improvement": rel,
            "mean_rel_improvement": sum(rel.values()) / len(rel),
            "junk_share": junk,
            "mean_final": {sid: sum(r) / len(r) for sid, r in scores.items()},
        })
        if i % 10 == 0:
            print(f"[{i + 1}/{len(rows) if rows else '?'}] {point} -> "
                  f"mean_rel={rows[-1]['mean_rel_improvement']:+.4f} ({time.perf_counter() - t0:.0f}s)")

    rows.sort(key=lambda r: -r["mean_rel_improvement"])
    best = rows[0]
    report = {
        "generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "grid": GRID,
        "seeds": seeds,
        "scenarios": [load_scenario(str(p)).scenario_id for p in paths],
        "uniform_mean": {k: sum(v) / len(v) for k, v in uniform_scores.items()},
        "best": best,
        "rows": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "tune_ducb.json"
    md_path = REPORT_DIR / "tune_ducb.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print("BEST:", json.dumps(best["point"]), "mean_rel", f"{best['mean_rel_improvement']:+.4f}",
          "junk", {k: round(v, 3) for k, v in best["junk_share"].items()})
    print(f"report: {json_path}")
    return 0


def render_md(report: dict) -> str:
    lines = ["# D-UCB tuning sweep (Roadmap E.7)", "",
             f"Seeds: {report['seeds']}; scenarios: {report['scenarios']}", "",
             "| rank | gamma | kappa | tau | epsilon | mean rel impr | per-scenario | junk share |",
             "|---|---|---|---|---|---|---|---|"]
    for i, row in enumerate(report["rows"][:15], 1):
        p = row["point"]
        per = ", ".join(f"{k}: {v:+.3f}" for k, v in row["rel_improvement"].items())
        junk = ", ".join(f"{k}: {v:.2f}" for k, v in row["junk_share"].items())
        lines.append(
            f"| {i} | {p['gamma']} | {p['exploration_coef']} | {p['tau']} | {p['epsilon']} |"
            f" {row['mean_rel_improvement']:+.4f} | {per} | {junk} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
