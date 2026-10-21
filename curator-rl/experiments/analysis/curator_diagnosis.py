"""Diagnosis of Curator v0 in the long-horizon portfolio scenarios (D-69).

Gate 2'' criterion (i) failed in S-I. Two questions, answered on TUNING seeds
(never on evaluation seeds) with committed, regenerable output:

1. Component ablation: which Curator component costs score in S-I / S-J / S-A / S-C?
   (statuses, discounting, cost normalisation; the last row is Standard UCB.)
2. Spurious-reward upper bound: how much would a CORRECT S5 flag on the noisy arm
   recover? This uses privileged knowledge (the noisy arm is flagged from round 6),
   so it is an upper bound on what calibration (Phase D) can deliver, not a method.

    python experiments/analysis/curator_diagnosis.py --seeds 40

Writes reports/analysis/curator_diagnosis.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402
from experiments.sweeps.tune_all import variant_cfg  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "analysis"
ABLATION_SCENARIOS = ("S-I", "S-J", "S-A", "S-C")
S5_SCENARIOS = ("S-I", "S-J", "S-A")
S5_START_ROUND = 6
NOISY_ARM = "noisy"


def _path(sid: str) -> Path:
    return REPO_ROOT / "configs" / "sim" / f"scenario_{sid.replace('-', '').lower()}.yaml"


class PrivilegedS5:
    """DIAGNOSTIC ONLY: reports the noisy arm as a proxy-benchmark mismatch from `start` on."""

    def __init__(self, inner, env: str = NOISY_ARM, start: int = S5_START_ROUND) -> None:
        self.inner, self.env, self.start, self.t = inner, env, start, 0
        self.env_ids = inner.env_ids

    def select_mixture(self, obs):
        self.t += 1
        if self.t >= self.start:
            self.inner._engine.set_calibration_mismatch(self.env, True)  # noqa: SLF001
        return self.inner.select_mixture(obs)

    def update_observation(self, obs):
        return self.inner.update_observation(obs)

    def update_calibration(self, obs):
        return self.inner.update_calibration(obs)


def _uniform(scenario, seeds):
    ids = [e.env_id for e in scenario.envs]
    return np.array([run_episode(UniformScheduler(ids), scenario, seed=s, method="u").final_score for s in seeds])


def _summ(scores, uni) -> dict:
    rel = (np.asarray(scores) - uni) / uni
    return {"mean": float(rel.mean()), "se": float(rel.std(ddof=1) / np.sqrt(len(rel)))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curator v0 diagnosis (D-69)")
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args(argv)
    seeds = list(range(args.seeds))                       # tuning seeds only
    base = load_config(REPO_ROOT / "configs" / "base.yaml")

    variants = {
        "curator (as configured)": ("curator", base),
        "status soft": ("curator", variant_cfg(base, "curator", {"status_control": "soft"})),
        "status off": ("curator", variant_cfg(base, "curator", {"status_control": "off"})),
        "status off, gamma 1.0": ("curator", variant_cfg(base, "curator", {"status_control": "off", "gamma": 1.0})),
        "status off, gamma 1.0, no cost (= Standard UCB form)": (
            "curator", variant_cfg(base, "curator", {"status_control": "off", "gamma": 1.0, "cost_exponent": 0.0})),
        "Standard UCB (tuned baseline)": ("ucb", base),
    }
    ablation: dict[str, dict] = {}
    s5: dict[str, dict] = {}
    for sid in ABLATION_SCENARIOS:
        scenario = load_scenario(str(_path(sid)))
        uni = _uniform(scenario, seeds)
        ablation[sid] = {}
        for name, (method, cfg) in variants.items():
            scores = [run_episode(make_scheduler(method, scenario, s, cfg=cfg), scenario, seed=s, method=method).final_score
                      for s in seeds]
            ablation[sid][name] = _summ(scores, uni)
        if sid in S5_SCENARIOS:
            s5[sid] = {}
            for label, shrink, privileged in (("no S5 information", 0.5, False),
                                              ("privileged S5, shrink 0.5", 0.5, True),
                                              ("privileged S5, shrink 0.1", 0.1, True)):
                cfg = base.model_copy(update={"scheduler": base.scheduler.model_copy(update={"s5_shrink": shrink})})
                scores, shares = [], []
                for s in seeds:
                    sched = make_scheduler("curator", scenario, s, cfg=cfg)
                    if privileged:
                        sched = PrivilegedS5(sched)
                    r = run_episode(sched, scenario, seed=s, method="curator")
                    scores.append(r.final_score)
                    shares.append(float(np.mean([log["weights"][NOISY_ARM] for log in r.round_logs[8:]])))
                s5[sid][label] = {**_summ(scores, uni), "noisy_share": float(np.mean(shares))}

    report = {"seeds": seeds, "ablation": ablation, "privileged_s5_upper_bound": s5,
              "note": "tuning seeds only; privileged S5 is an upper bound, not a method"}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "curator_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "curator_diagnosis.md").write_text(render_md(report), encoding="utf-8")
    print((REPORT_DIR / "curator_diagnosis.md").read_text(encoding="utf-8"))
    return 0


def render_md(report: dict) -> str:
    lines = ["# Curator v0 diagnosis (D-69)", "",
             f"Tuning seeds {report['seeds'][0]}..{report['seeds'][-1]}. Values are mean relative improvement over "
             "Uniform (paired by seed) with the standard error.", "",
             "## 1. Component ablation", "",
             "| variant | " + " | ".join(report["ablation"]) + " |", "|---|" + "---|" * len(report["ablation"])]
    names = list(next(iter(report["ablation"].values())))
    for n in names:
        lines.append(f"| {n} | " + " | ".join(
            f"{report['ablation'][sid][n]['mean']:+.4f} ({report['ablation'][sid][n]['se']:.4f})"
            for sid in report["ablation"]) + " |")
    lines += ["", "## 2. Spurious-reward upper bound (privileged S5 on the noisy arm, from round "
              f"{S5_START_ROUND}; NOT a method)", "",
              "| variant | " + " | ".join(report["privileged_s5_upper_bound"]) + " |",
              "|---|" + "---|" * len(report["privileged_s5_upper_bound"])]
    labels = list(next(iter(report["privileged_s5_upper_bound"].values())))
    for n in labels:
        lines.append(f"| {n} | " + " | ".join(
            f"{v[n]['mean']:+.4f} ({v[n]['se']:.4f}), noisy share {v[n]['noisy_share']:.2f}"
            for v in report["privileged_s5_upper_bound"].values()) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
