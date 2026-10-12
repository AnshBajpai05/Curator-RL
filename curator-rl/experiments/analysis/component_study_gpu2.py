"""Which of the new components help, at the GPU shape with realism v2? (TUNING seeds only)

Exploratory study behind the Kaggle preset: nothing here is a confirmatory claim (that needs a
pre-registration on fresh seeds >= 700, docs/DECISIONS.md). Starting from Curator at its GPU-shape
tuned point (gamma 1, kappa 0.1, tau 0.3; reports/sweeps/tune_gpu_shape.md), components are added one
at a time:

    A  reward audit (signals/audit.py) + S5 capped at the exploration floor (B5)
    F  alpha/beta fitted to calibrated gains (B4)
    D  drift-triggered discounting (B2)
    M  weight cap 0.6 (B7)        S  mixture smoothing 0.5 (B7)

Every method runs on the six GPU2 scenarios (configs/sim/*gpu2*.yaml). Metric: relative gain of the
noiseless final score over Uniform on the same seed (E-10), mean and 95% t-interval over seeds; plus
the budget share the noisy arm received, detection (ever in S5), false flags, MBPP passes.
Part B checks the B2 criterion on S-E (abrupt regime change): drift-triggered discounting with
gamma 1 must not score below gamma 0.95.

    python experiments/analysis/component_study_gpu2.py --seeds 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config, rescale_for_batch_shape  # noqa: E402
from curator_rl.core.stats import paired_t_interval  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

SCENARIOS = {
    "S-I-R1-GPU2": "scenario_si_r1_gpu2.yaml", "S-I-R2-GPU2": "scenario_si_r2_gpu2.yaml",
    "S-J-R-GPU2": "scenario_sj_r_gpu2.yaml", "S-I-R1-GPU2-MIS": "scenario_si_r1_gpu2_misaligned.yaml",
    "S-I-R1-GPU2-NN": "scenario_si_r1_gpu2_noisy_neg.yaml", "S-I-R1-GPU2-NP": "scenario_si_r1_gpu2_noisy_pos.yaml",
}
JUNK = {"noisy", "format"}           # arms with no benchmark value (format: only in the MIS scenario)
TUNED_CURATOR = {"scheduler.gamma": 1.0, "scheduler.exploration_coef": 0.1, "scheduler.tau": 0.3}
COMPONENTS = {
    "A": {"signals.reward_audit.enabled": True, "scheduler.s5_cap": "floor"},
    "F": {"proxy.fit.enabled": True},
    "D": {"scheduler.drift.enabled": True},
    "M": {"scheduler.max_weight": 0.6},
    "S": {"scheduler.smoothing": 0.5},
}
CURATOR_VARIANTS = ["", "A", "AF", "AD", "AFD", "AFDM", "AFDS", "F"]
BASELINES = {"uniform": {}, "pilot_uniform": {}, "lp": {"baselines.lp.tau": 0.5},
             "ucb": {"baselines.ucb.exploration_coef": 0.25, "baselines.ucb.tau": 0.3},
             "sec": {"baselines.sec.alpha": 1.0, "baselines.sec.tau": 0.4},
             "dump": {"baselines.dump.exploration_coef": 0.3, "baselines.dump.tau": 0.4,
                      "baselines.dump.window_rounds": 10}}


def _cfg(overrides: dict, prompts_per_round: int):
    ov = [f"{k}={json.dumps(v) if isinstance(v, str) else v}" for k, v in overrides.items()]
    return rescale_for_batch_shape(load_config(REPO_ROOT / "configs" / "base.yaml", ov), prompts_per_round)


def methods() -> dict[str, tuple[str, dict]]:
    out = {name: (name, ov) for name, ov in BASELINES.items()}
    for v in CURATOR_VARIANTS:
        ov = dict(TUNED_CURATOR)
        for c in v:
            ov.update(COMPONENTS[c])
        out["curator" + (f"+{v}" if v else "")] = ("curator", ov)
    return out


class Watch:
    """Delegating wrapper: first round each arm was in S5."""

    def __init__(self, inner) -> None:
        self.inner, self.env_ids = inner, inner.env_ids
        self.uses_calibration = getattr(inner, "uses_calibration", False)
        self.first_s5: dict[str, int] = {}
        self._round = 0

    def select_mixture(self, obs):
        self._round += 1
        w = self.inner.select_mixture(obs)
        for e, sv in (getattr(self.inner, "_latest_vectors", None) or {}).items():
            if sv.status.value.startswith("S5") and e not in self.first_s5:
                self.first_s5[e] = self._round
        return w

    def update_observation(self, obs):
        self.inner.update_observation(obs)

    def update_calibration(self, obs):
        self.inner.update_calibration(obs)

    def calibration_request(self):
        return self.inner.calibration_request()


def run_cell(task) -> dict:
    name, sid, seeds = task
    factory, overrides = methods()[name]
    sc = load_scenario(REPO_ROOT / "configs" / "sim" / SCENARIOS[sid])
    cfg = _cfg(overrides, sc.prompts_per_round)
    ids = [e.env_id for e in sc.envs]
    rows = []
    for s in seeds:
        w = Watch(make_scheduler(factory, sc, s, cfg=cfg))
        r = run_episode(w, sc, seed=s, method=name)
        total = sum(r.prompts_by_env.values()) or 1
        junk_here = [e for e in ids if e in JUNK]
        rows.append({
            "seed": s, "score": r.final_score_expected,
            "junk_share": sum(r.prompts_by_env[e] for e in junk_here) / total,
            "noisy_share": r.prompts_by_env.get("noisy", 0) / total,
            "detected": {e: w.first_s5.get(e) for e in junk_here},
            "false_flags": sum(1 for e in ids if e not in JUNK and e in w.first_s5) / len([e for e in ids if e not in JUNK]),
            "mbpp_passes": r.passes_by_env.get("mbpp"),
            "max_weight": r.concentration(skip_rounds=6)["mean_max_weight"],
            "calib_cost": sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12),
            "proxy_weights": list(getattr(w.inner, "proxy_weights", (None, None))),
        })
    return {"name": name, "sid": sid, "rows": rows}


def run_se(task) -> dict:
    name, overrides, seeds = task
    sc = load_scenario(REPO_ROOT / "configs" / "sim" / "scenario_se.yaml")
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      [f"{k}={v}" for k, v in overrides.items()])
    return {"name": name, "scores": [run_episode(make_scheduler("curator", sc, s, cfg=cfg), sc, seed=s).final_score_expected
                                     for s in seeds]}


def _fmt_ci(vals) -> str:
    m, lo, hi = paired_t_interval(vals)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    t0 = time.perf_counter()
    tasks = [(m, sid, seeds) for m in methods() for sid in SCENARIOS]
    se_tasks = [("gamma 0.95", {"scheduler.gamma": 0.95}, seeds), ("gamma 1", {"scheduler.gamma": 1.0}, seeds),
                ("gamma 1 + drift", {"scheduler.gamma": 1.0, "scheduler.drift.enabled": True}, seeds)]
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(run_cell, tasks))
        se = list(pool.map(run_se, se_tasks))
    res = {(c["name"], c["sid"]): c["rows"] for c in cells}

    def rel(name, sid):
        return [(a["score"] - u["score"]) / u["score"] for a, u in zip(res[(name, sid)], res[("uniform", sid)])]

    summary = {}
    for name in methods():
        for sid in SCENARIOS:
            rows = res[(name, sid)]
            det = {e: [r["detected"][e] for r in rows] for e in rows[0]["detected"]}
            summary[f"{name}|{sid}"] = {
                "rel_gain": paired_t_interval(rel(name, sid)) if name != "uniform" else (0.0, 0.0, 0.0),
                "noisy_share": statistics.fmean(r["noisy_share"] for r in rows),
                "junk_share": statistics.fmean(r["junk_share"] for r in rows),
                "detect_rate": {e: sum(v is not None for v in vs) / len(vs) for e, vs in det.items()},
                "detect_round": {e: (float(np.median([v for v in vs if v is not None])) if any(vs) else None)
                                 for e, vs in det.items()},
                "false_flags": statistics.fmean(r["false_flags"] for r in rows),
                "mbpp_passes": statistics.fmean(r["mbpp_passes"] for r in rows if r["mbpp_passes"] is not None),
                "max_weight": statistics.fmean(r["max_weight"] for r in rows),
                "calib_cost": statistics.fmean(r["calib_cost"] for r in rows),
            }
    vs_c0 = {}
    for name in methods():
        if name.startswith("curator+"):
            for sid in SCENARIOS:
                d = [a["score"] - b["score"] for a, b in zip(res[(name, sid)], res[("curator", sid)])]
                vs_c0[f"{name}|{sid}"] = paired_t_interval(d)
    se_scores = {r["name"]: r["scores"] for r in se}
    se_cmp = {"drift_minus_g095": paired_t_interval([a - b for a, b in zip(se_scores["gamma 1 + drift"], se_scores["gamma 0.95"])]),
              "g1_minus_g095": paired_t_interval([a - b for a, b in zip(se_scores["gamma 1"], se_scores["gamma 0.95"])]),
              "means": {k: statistics.fmean(v) for k, v in se_scores.items()}}

    rep = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": [0, args.seeds - 1],
           "summary": summary, "curator_variant_minus_curator": vs_c0, "s_e": se_cmp,
           "seconds": round(time.perf_counter() - t0, 1)}
    out = REPO_ROOT / "reports" / "analysis"
    (out / "component_study_gpu2.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    lines = ["# Component study at the GPU shape, realism v2 (tuning seeds; exploratory)", "",
             f"Seeds 0..{args.seeds - 1}. Relative gain = (noiseless final score - Uniform's) / Uniform's, same seed; "
             "95% t-interval. `A` audit + S5 cap, `F` alpha/beta fit, `D` drift-triggered discounting, "
             "`M` weight cap 0.6, `S` smoothing 0.5; Curator at its GPU-shape tuned point.", ""]
    for sid in SCENARIOS:
        lines += [f"## {sid}", "", "| method | rel. gain over Uniform | noisy share | junk share | junk detected "
                  "(median round) | false flags | MBPP passes | mean max w |", "|---|---|---|---|---|---|---|---|"]
        for name in methods():
            s = summary[f"{name}|{sid}"]
            m, lo, hi = s["rel_gain"]
            det = ", ".join(f"{e} {s['detect_rate'][e]:.2f} ({s['detect_round'][e] or '-'})" for e in s["detect_rate"])
            lines.append(f"| {name} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {s['noisy_share']:.3f} | {s['junk_share']:.3f} | "
                         f"{det} | {s['false_flags']:.3f} | {s['mbpp_passes']:.2f} | {s['max_weight']:.2f} |")
        lines.append("")
    lines += ["## Curator variants minus Curator (absolute noiseless score, same seed)", "",
              "| variant | " + " | ".join(SCENARIOS) + " |", "|---|" + "---|" * len(SCENARIOS)]
    for name in methods():
        if name.startswith("curator+"):
            cells = []
            for sid in SCENARIOS:
                m, lo, hi = vs_c0[f"{name}|{sid}"]
                cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    m, lo, hi = se_cmp["drift_minus_g095"]
    m1, lo1, hi1 = se_cmp["g1_minus_g095"]
    lines += ["", "## B2 check on S-E (abrupt regime change; base config, audit off)", "",
              f"- gamma 1 + drift minus gamma 0.95: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]",
              f"- gamma 1 (no drift) minus gamma 0.95: {m1:+.4f} [{lo1:+.4f}, {hi1:+.4f}]",
              f"- means: {json.dumps({k: round(v, 4) for k, v in se_cmp['means'].items()})}", "",
              f"Runtime {rep['seconds']} s."]
    md = "\n".join(lines) + "\n"
    (out / "component_study_gpu2.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
