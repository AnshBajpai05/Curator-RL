"""Exploratory (tuning seeds 0-39 only): SEC/DUMP at our tuned points vs the authors' code defaults.

Authors' defaults read from their repos (2026-11-01):
- SEC (ServiceNow/sec, verl/utils/auto_curriculum/priority.py): EMA lr 0.1, Boltzmann temperature 1.0.
- DUMP (ZhentingWang/DUMP, verl/utils/curriculum_learning.py): softmax temperature 0.1, UCB bonus
  1.0 * sqrt(2 ln N / n) (= c ~1.41 in our form), sliding window of recent samples.
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402
from experiments.sweeps.tune_gpu_shape import _base, _variant  # noqa: E402

SKIP = 6
SCEN = {"S-I-R1-GPU2": "scenario_si_r1_gpu2.yaml", "S-I-R2-GPU2": "scenario_si_r2_gpu2.yaml"}
SEEDS = range(40)
CELLS = [
    ("uniform", "uniform", None),
    ("curator (GPU preset)", "curator", {"gamma": 1.0, "exploration_coef": 0.25, "tau": 1.0}),
    ("SEC tuned (a 0.1, tau 0.4)", "sec", {"alpha": 0.1, "tau": 0.4}),
    ("SEC authors' default (a 0.1, tau 1.0)", "sec", {"alpha": 0.1, "tau": 1.0}),
    ("DUMP tuned (c 2.0, tau 0.4, W 10)", "dump", {"exploration_coef": 2.0, "tau": 0.4, "window_rounds": 10}),
    ("DUMP authors'-like (c 1.41, tau 0.1, W 10)", "dump", {"exploration_coef": 1.41, "tau": 0.1, "window_rounds": 10}),
]


def run(method, point, sc, seed):
    cfg = _base() if point is None else _variant(method, point, "gpu2")
    r = run_episode(make_scheduler(method, sc, seed, cfg=cfg), sc, seed=seed, method=method)
    shares = [lg["per_env"]["noisy"]["n_prompts"] / max(sum(v["n_prompts"] for v in lg["per_env"].values()), 1)
              for lg in r.round_logs[SKIP:]]
    return r.final_score_expected, float(np.mean(shares))


scs = {sid: load_scenario(ROOT / "configs" / "sim" / f) for sid, f in SCEN.items()}
uni = {sid: [run("uniform", None, sc, s)[0] for s in SEEDS] for sid, sc in scs.items()}
print(f"{'method':44s} {'rel. score vs Uniform':>22s} {'SE':>7s} {'noisy share':>12s}")
for label, method, point in CELLS:
    rel, share = [], []
    for i, s in enumerate(SEEDS):
        per = []
        for sid, sc in scs.items():
            score, sh = run(method, point, sc, s)
            per.append((score - uni[sid][i]) / uni[sid][i])
            share.append(sh)
        rel.append(float(np.mean(per)))
    se = statistics.stdev(rel) / len(rel) ** 0.5
    print(f"{label:44s} {100 * statistics.fmean(rel):+21.1f}% {100 * se:6.1f}% {100 * statistics.fmean(share):11.1f}%")
