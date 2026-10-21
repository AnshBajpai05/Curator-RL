"""Does dropping discounting (gamma = 1) hurt Curator on the ORIGINAL scenario suite? (D-90 diagnosis)

Tuning seeds 0-49 only. Compares Curator as configured (gamma 0.95) with the same Curator at
gamma = 1, calibration off in both (the original scenarios offer no calibration), on S-A..S-J
(S-G excluded: it is an evaluation-noise sweep, not an allocation scenario). Reports paired
differences and allocation concentration, because gamma = 1 concentrates allocation and the
simulator has no forgetting to punish that (D-67).

    python experiments/analysis/gamma_robustness.py

Writes reports/analysis/gamma_robustness.{json,md}.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.analysis.refit_diagnosis import variant_cfg  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

SCENARIOS = ("sa", "sb", "sc", "sd", "se", "sf", "sh", "si", "sj")
VARIANTS = {"gamma_0.95": dict(gamma=True, cost=True, status=True, calib=False, richness=True),
            "gamma_1": dict(gamma=False, cost=True, status=True, calib=False, richness=True)}
REFS = ("ucb", "uniform")
SEEDS = list(range(50))


def cell(task):
    sid, name = task
    sc = load_scenario(REPO_ROOT / "configs" / "sim" / f"scenario_{sid}.yaml")
    cfg = variant_cfg(VARIANTS[name]) if name in VARIANTS else load_config(FROZEN_D68_D94)
    method = "curator" if name in VARIANTS else name
    rs = [run_episode(make_scheduler(method, sc, s, cfg=cfg), sc, seed=s, method=method) for s in SEEDS]
    return sid, name, [r.final_score for r in rs], float(np.mean(
        [r.concentration(skip_rounds=8)["mean_max_weight"] for r in rs]))


def main() -> int:
    tasks = [(s, n) for s in SCENARIOS for n in (*VARIANTS, *REFS)]
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(cell, tasks))
    res = {(s, n): (sc, mw) for s, n, sc, mw in out}
    rng = np.random.default_rng(6061)
    rows = {}
    for s in SCENARIOS:
        a, b = res[(s, "gamma_0.95")][0], res[(s, "gamma_1")][0]
        m, lo, hi = gate2_check.paired_bootstrap([y - x for x, y in zip(a, b)], rng)
        rows[s] = {"gamma_0.95": float(np.mean(a)), "gamma_1": float(np.mean(b)), "diff": [m, lo, hi],
                   "maxw_0.95": res[(s, "gamma_0.95")][1], "maxw_1": res[(s, "gamma_1")][1],
                   "ucb": float(np.mean(res[(s, "ucb")][0])), "uniform": float(np.mean(res[(s, "uniform")][0]))}
    dest = REPO_ROOT / "reports" / "analysis"
    (dest / "gamma_robustness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ["# Curator with gamma = 1 on the original scenario suite (tuning seeds 0-49)", "",
          "Calibration off in both variants; everything else as in `configs/base.yaml`.", "",
          "| scenario | gamma 0.95 | gamma 1 | gamma 1 minus 0.95 [95% CI] | max-weight 0.95 / 1 | Std UCB | Uniform |",
          "|---|---|---|---|---|---|---|"]
    for s, r in rows.items():
        m, lo, hi = r["diff"]
        md.append(f"| S-{s[1].upper()} | {r['gamma_0.95']:.4f} | {r['gamma_1']:.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                  f"{r['maxw_0.95']:.2f} / {r['maxw_1']:.2f} | {r['ucb']:.4f} | {r['uniform']:.4f} |")
    (dest / "gamma_robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
