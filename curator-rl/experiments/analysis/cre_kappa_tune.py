"""Tune kappa and tau for the CRE and for its control (D-80), tuning seeds only.

Grid of 8 (kappa {0.02,0.05,0.1,0.2} x tau {0.3,1.0}) for BOTH `cre` and the control
`curator_targeted` (same calibration, CRE off). Scenario S-I, seeds 0..n-1. Selection (D-55
rule): best objective, then within one SE, then highest normalised allocation entropy.

    python experiments/analysis/cre_kappa_tune.py --seeds 50

Writes reports/analysis/cre_kappa_tune.{json,md}.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np  # noqa: E402

from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.cre_gate import method_cfg, scenario_variant  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402
from experiments.sweeps.tune_all import select  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "analysis"
GRID = [{"exploration_coef": k, "tau": t} for k, t in itertools.product((0.02, 0.05, 0.1, 0.2), (0.3, 1.0))]
METHODS = ("cre", "curator_targeted")


def with_scheduler(cfg, point):
    return cfg.model_copy(update={"scheduler": cfg.scheduler.model_copy(update=point)})


def evaluate(task):
    method, index, point, seeds, uniform = task
    sc = scenario_variant("S-I", method_cfg(method).calib.interval_rounds)
    cfg = with_scheduler(method_cfg(method), point)
    rel, ent, maxw = [], [], []
    for i, s in enumerate(seeds):
        r = run_episode(make_scheduler("curator", sc, s, cfg=cfg), sc, seed=s, method=method)
        rel.append((r.final_score - uniform[i]) / uniform[i])
        c = r.concentration(skip_rounds=8)
        ent.append(c["norm_entropy"])
        maxw.append(c["mean_max_weight"])
    a = np.asarray(rel)
    return {"method": method, "index": index, "point": point, "objective_mean": float(a.mean()),
            "objective_se": float(a.std(ddof=1) / np.sqrt(len(a))), "norm_entropy": float(np.nanmean(ent)),
            "mean_max_weight": float(np.nanmean(maxw))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CRE kappa/tau tuning (D-80)")
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    sc = scenario_variant("S-I", 10)
    ids = [e.env_id for e in sc.envs]
    uniform = [run_episode(UniformScheduler(ids), sc, seed=s, method="u").final_score for s in seeds]
    tasks = [(m, i, p, seeds, uniform) for m in METHODS for i, p in enumerate(GRID)]
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(evaluate, tasks))
    out = {"seeds": [seeds[0], seeds[-1]], "methods": {}}
    md = ["# CRE kappa/tau tuning (D-80)", "", f"Tuning seeds {seeds[0]}..{seeds[-1]}, S-I; objective = mean relative improvement over Uniform.", ""]
    for m in METHODS:
        pts = sorted((r for r in res if r["method"] == m), key=lambda r: r["index"])
        sel = select(pts)
        out["methods"][m] = {"points": pts, "selection": sel}
        md += [f"## {m}", "", "| # | params | objective | SE | norm. entropy | mean max-w | |", "|---|---|---|---|---|---|---|"]
        for p in pts:
            mark = "**chosen**" if p["index"] == sel["chosen_index"] else ("cand." if p["index"] in sel["candidate_indices"] else "")
            md.append(f"| {p['index']} | {json.dumps(p['point'])} | {p['objective_mean']:+.4f} | {p['objective_se']:.4f} | "
                      f"{p['norm_entropy']:.2f} | {p['mean_max_weight']:.2f} | {mark} |")
        md.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cre_kappa_tune.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (REPORT_DIR / "cre_kappa_tune.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
