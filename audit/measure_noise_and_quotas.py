"""Measure final-evaluation noise and quota rounding (improvement_required.md B1/A2; errors_found.md E-09/E-10).

Tuning seeds only; writes nothing. From the repository root:

    curator-rl/.venv/Scripts/python audit/measure_noise_and_quotas.py curator-rl
"""
import statistics
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))  # for experiments.run_sim

import numpy as np  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

from curator_rl.core.quotas import largest_remainder_quotas  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import build_world, load_scenario  # noqa: E402

scen = load_scenario(REPO / "configs/sim/scenario_si_r1.yaml")
SEEDS = range(0, 40)

# --- 1. evaluation noise: observed final score vs noiseless expected score -------------
obs = {"uniform": [], "curator": []}
exp = {"uniform": [], "curator": []}
for seed in SEEDS:
    for m in obs:
        r = run_episode(make_scheduler(m, scen, seed), scen, seed, method=m)
        skills = r.round_logs[-1]["true_skills"]
        w = build_world(scen, seed)
        obs[m].append(r.final_score)
        exp[m].append(w.expected_benchmark_score(skills))

def half_width(d):
    rng = np.random.default_rng(0)
    d = np.asarray(d)
    boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), (hi - lo) / 2

d_obs = [a - b for a, b in zip(obs["curator"], obs["uniform"])]
d_exp = [a - b for a, b in zip(exp["curator"], exp["uniform"])]
print("1) S-I-R1, seeds 0-39, curator - uniform")
print("   observed final score : mean %+.4f  CI half-width %.4f  sd %.4f" % (*half_width(d_obs), statistics.stdev(d_obs)))
print("   noiseless expected   : mean %+.4f  CI half-width %.4f  sd %.4f" % (*half_width(d_exp), statistics.stdev(d_exp)))
noise = [o - e for o, e in zip(obs["curator"], exp["curator"])]
print("   sd of (observed - expected) for one method: %.4f" % statistics.stdev(noise))

# --- 2. quota rounding at the planned GPU shape (M = R*P = 2*16 = 32) ------------------
print("2) floor arm (w = 0.02) prompt count over 60 rounds")
for M in (80, 32):
    zero = 0
    delivered = 0
    for t in range(60):
        w = {"gsm8k": 0.40, "math35": 0.25, "mbpp": 0.25, "countdown": 0.02, "noisy": 0.08}
        q = largest_remainder_quotas(w, M, tie_break_seed=t)
        zero += q["countdown"] == 0
        delivered += q["countdown"]
    print(f"   M={M}: rounds with 0 prompts {zero}/60; realised share {delivered/(60*M):.4f} vs requested 0.0200")
