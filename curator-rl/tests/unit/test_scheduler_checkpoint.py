"""Every scheduler must resume exactly: a checkpoint taken mid-run, loaded into a fresh object,
must produce the same mixtures as the uninterrupted run (Phase F, Gate 3 kill-and-resume).

The observations are recorded once from a simulator run, then replayed into three schedulers: A
sees all of them; B sees the first k and is snapshotted (through JSON, as on disk); C is a fresh
scheduler loaded from that snapshot and sees the rest.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_sim import make_scheduler  # noqa: E402

from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.simulator.scenarios import build_world, load_scenario  # noqa: E402

METHODS = ["uniform", "static", "lp", "ucb", "sec", "dump", "curator"]
ROUNDS, SPLIT_AT = 14, 9


def _observations(scenario):
    world = build_world(scenario, seed=3)
    sched = UniformScheduler([e.env_id for e in scenario.envs])
    obs, prev = [], None
    for t in range(ROUNDS):
        o = world.step_round(sched.select_mixture(prev), steps=(t + 1) * scenario.steps_per_round)
        obs.append(o)
        prev = o
    return obs


@pytest.mark.parametrize("method", METHODS)
def test_scheduler_resumes_exactly_from_a_json_checkpoint(method):
    scenario = load_scenario(REPO_ROOT / "configs" / "sim" / "scenario_si_r1.yaml")
    obs = _observations(scenario)

    def run(sched, lo, hi, prev):
        mixtures = []
        for t in range(lo, hi):
            mixtures.append(sched.select_mixture(prev))
            sched.update_observation(obs[t])
            prev = obs[t]
        return mixtures, prev

    a = make_scheduler(method, scenario, seed=3)
    full, _ = run(a, 0, ROUNDS, None)

    b = make_scheduler(method, scenario, seed=3)
    _, prev = run(b, 0, SPLIT_AT, None)
    snapshot = json.loads(json.dumps(dict(b.get_state() or {})))

    c = make_scheduler(method, scenario, seed=999)
    c.load_checkpoint(snapshot)
    rest, _ = run(c, SPLIT_AT, ROUNDS, prev)

    for t, (got, want) in enumerate(zip(rest, full[SPLIT_AT:]), start=SPLIT_AT):
        for e in want:
            assert got[e] == pytest.approx(want[e], abs=1e-12), (method, t, e)
