"""Fair tuning of every adaptive method (Roadmap v3 section 4.4, D-55/D-68).

Every adaptive method gets the SAME tuning budget: 8 pre-registered
configurations over the hyperparameters that matter for it, evaluated on the
SAME tuning seeds and tuning scenarios. Shared protocol constants (exploration
floor epsilon = 0.10 and warm-up length) are NOT tuned. The homogeneous twin S-J
is deliberately excluded from tuning so it stays out-of-sample evidence for H6.

Objective (per tuning seed): mean over the tuning scenarios of the relative
final-score improvement over Uniform on the same seed,
    o_s = mean_scenarios (score_method - score_uniform) / score_uniform.

Selection rule (pre-registered, identical for all methods): take the best
configuration by mean objective; every configuration whose mean is within one
standard error of the best (SE of the best configuration's per-seed objective)
is a candidate; among candidates pick the one with the highest mean normalised
allocation entropy (least concentrated mixtures, D-67); ties -> best objective.

    python experiments/sweeps/tune_all.py --seeds 30

Writes reports/sweeps/tune_all.{json,md}. Tuning seeds are 0..seeds-1
(evaluation seeds start at 100).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.run_sim import (
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

REPORT_DIR = REPO_ROOT / "reports" / "sweeps"
TUNING_SCENARIOS = ("S-A", "S-B", "S-C", "S-I")      # S-J is held out on purpose
SKIP_ROUNDS = 8                                      # concentration is measured post warm-up/exploration

# 8 pre-registered configurations per method (equal tuning budget).
GRIDS: dict[str, list[dict]] = {
    "curator": [
        {"gamma": g, "exploration_coef": k, "tau": t}
        for g, k, t in itertools.product((0.90, 0.95), (0.25, 0.5), (0.3, 1.0))
    ],
    "ucb": [
        {"exploration_coef": k, "tau": t}
        for k, t in itertools.product((0.25, 0.5, 1.0, 2.0), (0.3, 1.0))
    ],
    "lp": [{"tau": t} for t in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)],
    "sec": [
        {"alpha": a, "tau": t}
        for a, t in itertools.product((0.1, 0.3, 0.6, 1.0), (0.1, 0.4))
    ],
    "dump": [
        {"exploration_coef": c, "tau": t}
        for c, t in itertools.product((0.1, 0.3, 1.0, 2.0), (0.1, 0.4))
    ],
}


def scenario_path(sid: str) -> Path:
    return REPO_ROOT / "configs" / "sim" / f"scenario_{sid.replace('-', '').lower()}.yaml"


def variant_cfg(base, method: str, point: dict):
    """A RootConfig with `point` applied to the method's own hyperparameters."""
    if method == "curator":
        return base.model_copy(update={"scheduler": base.scheduler.model_copy(update=point)})
    section = getattr(base.baselines, method).model_copy(update=point)
    return base.model_copy(update={"baselines": base.baselines.model_copy(update={method: section})})


def uniform_scores(seeds: list[int]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for sid in TUNING_SCENARIOS:
        scenario = load_scenario(str(scenario_path(sid)))
        ids = [e.env_id for e in scenario.envs]
        out[sid] = [
            run_episode(UniformScheduler(ids), scenario, seed=s, method="uniform").final_score
            for s in seeds
        ]
    return out


def evaluate_point(task: tuple) -> dict:
    """Run one (method, point) over all tuning scenarios and seeds (worker entry)."""
    method, index, point, seeds, uniform = task
    base = load_config(FROZEN_D68_D94)
    cfg = variant_cfg(base, method, point)
    per_scenario: dict[str, list[float]] = {}
    ent, maxw = [], []
    for sid in TUNING_SCENARIOS:
        scenario = load_scenario(str(scenario_path(sid)))
        rel = []
        for i, seed in enumerate(seeds):
            sched = make_scheduler(method, scenario, seed, cfg=cfg)
            res = run_episode(sched, scenario, seed=seed, method=method)
            u = uniform[sid][i]
            rel.append((res.final_score - u) / u)
            c = res.concentration(skip_rounds=SKIP_ROUNDS)
            ent.append(c["norm_entropy"])
            maxw.append(c["mean_max_weight"])
        per_scenario[sid] = rel
    per_seed = np.mean([per_scenario[sid] for sid in TUNING_SCENARIOS], axis=0)
    return {
        "method": method,
        "index": index,
        "point": point,
        "objective_mean": float(per_seed.mean()),
        "objective_se": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed))),
        "per_scenario_mean": {sid: float(np.mean(v)) for sid, v in per_scenario.items()},
        "norm_entropy": float(np.nanmean(ent)),
        "mean_max_weight": float(np.nanmean(maxw)),
    }


def select(points: list[dict]) -> dict:
    """Pre-registered one-SE + least-concentrated rule (module docstring)."""
    best = max(points, key=lambda p: p["objective_mean"])
    cutoff = best["objective_mean"] - best["objective_se"]
    candidates = [p for p in points if p["objective_mean"] >= cutoff]
    chosen = max(candidates, key=lambda p: (p["norm_entropy"], p["objective_mean"]))
    return {
        "best_index": best["index"],
        "cutoff": cutoff,
        "candidate_indices": [p["index"] for p in candidates],
        "chosen_index": chosen["index"],
        "chosen_point": chosen["point"],
    }


def render_md(report: dict) -> str:
    lines = [
        "# Fair tuning of all adaptive methods (D-55 / D-68)", "",
        f"Tuning seeds {report['seeds'][0]}..{report['seeds'][-1]}; scenarios {report['scenarios']} "
        "(S-J held out); 8 configurations per method; objective = mean relative improvement over "
        "Uniform; selection = best within one SE, then least concentrated.", "",
    ]
    for method, block in report["methods"].items():
        sel = block["selection"]
        lines += [f"## {method}", "",
                  "| # | params | objective | SE | norm. entropy | mean max-w | per scenario | |",
                  "|---|---|---|---|---|---|---|---|"]
        for p in block["points"]:
            mark = "**chosen**" if p["index"] == sel["chosen_index"] else (
                "cand." if p["index"] in sel["candidate_indices"] else "")
            per = ", ".join(f"{k[2:]}: {v:+.3f}" for k, v in p["per_scenario_mean"].items())
            lines.append(
                f"| {p['index']} | {json.dumps(p['point'])} | {p['objective_mean']:+.4f} | "
                f"{p['objective_se']:.4f} | {p['norm_entropy']:.2f} | {p['mean_max_weight']:.2f} | {per} | {mark} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fair tuning of all adaptive methods (D-55)")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--methods", default=",".join(GRIDS))
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count")
    args = parser.parse_args(argv)
    seeds = list(range(args.seeds))
    methods = [m for m in args.methods.split(",") if m]
    t0 = time.perf_counter()
    uniform = uniform_scores(seeds)
    tasks = [
        (m, i, point, seeds, uniform) for m in methods for i, point in enumerate(GRIDS[m])
    ]
    results: dict[str, list[dict]] = {m: [] for m in methods}
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        for res in pool.map(evaluate_point, tasks):
            results[res["method"]].append(res)
            print(f"  {res['method']:8} #{res['index']} {json.dumps(res['point'])}: "
                  f"obj {res['objective_mean']:+.4f} (se {res['objective_se']:.4f}) "
                  f"H {res['norm_entropy']:.2f}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    report = {
        "generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "seeds": seeds,
        "scenarios": list(TUNING_SCENARIOS),
        "methods": {
            m: {"points": sorted(results[m], key=lambda p: p["index"]),
                "selection": select(results[m])}
            for m in methods
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tune_all.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "tune_all.md").write_text(render_md(report), encoding="utf-8")
    print("\nselected:")
    for m in methods:
        print(f"  {m}: {json.dumps(report['methods'][m]['selection']['chosen_point'])}")
    print(f"report: {REPORT_DIR / 'tune_all.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
