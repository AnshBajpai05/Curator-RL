"""Pre-registered evaluation of the refit scenarios (D-88). Run ONCE.

Fresh evaluation seeds 400-449, frozen configurations (`configs/base.yaml`, no retuning),
calibration offered as in D-75 and charged per item to the methods that use it.

Primary (pass/fail):
  P1  Curator minus SEC-style and minus DUMP-style, paired 95% bootstrap CI excluding 0,
      in S-I-R1 AND S-I-R2 (Gate 2'' (i) on measured costs).
  P2  S-J-R: Curator minus Uniform, lower CI bound > -0.01 (Gate 2'' (ii)).
Secondary (reported only): Curator minus every other method; calibration on vs off; noisy-arm
S5 detection and false flags; allocation per arm after round 8; calibration cost share.

    python experiments/refit_gate.py --seeds 50 --seed0 400

Writes reports/gate2/refit_gate.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import CalibSimCfg, load_scenario  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.analysis.calibration_power import S5Recorder  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

REPORT_DIR = REPO_ROOT / "reports" / "gate2"
SCENARIOS = {"S-I-R1": "scenario_si_r1.yaml", "S-I-R2": "scenario_si_r2.yaml", "S-J-R": "scenario_sj_r.yaml"}
METHODS = ("uniform", "static", "lp", "ucb", "sec", "dump", "curator", "curator_nocalib")
NOISY = "noisy"
DELTA_GAP = gate2_check.DELTA_GAP
SKIP = 8
BOOT_SEED = 4041


def scenario_with_calibration(sid: str, interval: int):
    sc = load_scenario(REPO_ROOT / "configs" / "sim" / SCENARIOS[sid])
    cost_item = float(np.mean([e.cost_usd_per_prompt for e in sc.envs])) / sc.group_size
    calib = CalibSimCfg(enabled=True, interval_rounds=interval, paired=True, churn=0.02,
                        items_per_slice=100, cost_per_item_usd=cost_item)
    return sc.model_copy(update={"calib": calib})


def method_cfg(method: str):
    base = load_config(FROZEN_D68_D94)
    if method == "curator_nocalib":
        return base.model_copy(update={"calib": base.calib.model_copy(update={"enabled": False})})
    return base


def run_cell(task: tuple) -> dict:
    sid, method, seeds = task
    cfg = method_cfg(method)
    sc = scenario_with_calibration(sid, cfg.calib.interval_rounds)
    ids = [e.env_id for e in sc.envs]
    out = {"sid": sid, "method": method, "scores": [], "rounds": [], "calib_share": [],
           "noisy_detected": [], "false_flags": [], "weights": {}, "max_weight": []}
    w_acc = {e: [] for e in ids}
    for s in seeds:
        factory = "curator" if method.startswith("curator") else method
        sched = make_scheduler(factory, sc, s, cfg=cfg)
        rec = S5Recorder(sched) if getattr(sched, "uses_calibration", False) else sched
        r = run_episode(rec, sc, seed=s, method=method)
        out["scores"].append(r.final_score)
        out["rounds"].append(r.rounds)
        out["calib_share"].append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
        for e in ids:
            w_acc[e].append(float(np.mean([log["weights"][e] for log in r.round_logs[SKIP:]])))
        out["max_weight"].append(float(r.concentration(skip_rounds=SKIP)["mean_max_weight"]))
        if hasattr(rec, "ever_s5"):
            out["noisy_detected"].append(NOISY in rec.ever_s5)
            others = [e for e in ids if e != NOISY]
            out["false_flags"].append(sum(e in rec.ever_s5 for e in others) / len(others))
    out["weights"] = {e: float(np.mean(v)) for e, v in w_acc.items()}
    return out


def paired(a, b, rng) -> dict:
    mean, lo, hi = gate2_check.paired_bootstrap([x - y for x, y in zip(a, b)], rng)
    return {"mean_diff": mean, "ci95": [lo, hi]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered refit evaluation (D-88)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=400)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    t0 = time.perf_counter()
    tasks = [(sid, m, seeds) for sid in SCENARIOS for m in METHODS]
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        cells = list(pool.map(run_cell, tasks))
    res = {(c["sid"], c["method"]): c for c in cells}
    sc = {k: v["scores"] for k, v in res.items()}
    rng = np.random.default_rng(BOOT_SEED)

    comps, ok = {}, True
    for sid in ("S-I-R1", "S-I-R2"):
        for rival in ("sec", "dump"):
            r = paired(sc[(sid, "curator")], sc[(sid, rival)], rng)
            r["pass"] = bool(r["ci95"][0] > 0.0)
            ok &= r["pass"]
            comps[f"{sid}:curator-{rival}"] = r
    p2 = paired(sc[("S-J-R", "curator")], sc[("S-J-R", "uniform")], rng)
    p2["delta_gap"] = DELTA_GAP
    p2["pass"] = bool(p2["ci95"][0] > -DELTA_GAP)
    criteria = {"P1_beats_learnability_bandits": {"comparisons": comps, "pass": bool(ok)},
                "P2_homogeneous_no_underperformance": p2}

    secondary = {sid: {m: paired(sc[(sid, "curator")], sc[(sid, m)], rng) for m in METHODS if m != "curator"}
                 for sid in SCENARIOS}
    means = {sid: {m: float(np.mean(sc[(sid, m)])) for m in METHODS} for sid in SCENARIOS}
    detail = {sid: {m: {"weights": res[(sid, m)]["weights"],
                        "rounds": float(np.mean(res[(sid, m)]["rounds"])),
                        "mean_max_weight": float(np.nanmean(res[(sid, m)]["max_weight"])),
                        "calib_share": float(np.mean(res[(sid, m)]["calib_share"])),
                        "noisy_detected": (float(np.mean(res[(sid, m)]["noisy_detected"]))
                                           if res[(sid, m)]["noisy_detected"] else None),
                        "false_flags": (float(np.mean(res[(sid, m)]["false_flags"]))
                                        if res[(sid, m)]["false_flags"] else None)}
                    for m in METHODS} for sid in SCENARIOS}
    report = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": seeds,
              "criteria": criteria, "all_pass": bool(all(c["pass"] for c in criteria.values())),
              "means": means, "curator_minus": secondary, "detail": detail,
              "seconds": round(time.perf_counter() - t0, 1)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "refit_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "refit_gate.md").write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0 if report["all_pass"] else 1


def render_md(rep: dict) -> str:
    L = ["# Refit evaluation (D-88, pre-registered)", "",
         f"Generated {rep['generated_utc']}; fresh evaluation seeds {rep['seeds'][0]}..{rep['seeds'][-1]}; "
         "frozen configs (no retuning); scenarios refit from the Kaggle measurements "
         "(`reports/analysis/refit_scenarios.md`).", "",
         "## Mean final scores", "", "| scenario | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for sid, per in rep["means"].items():
        L.append(f"| {sid} | " + " | ".join(f"{per[m]:.4f}" for m in METHODS) + " |")
    L += ["", "## Primary criteria", ""]
    for name, c in rep["criteria"].items():
        L.append(f"- **{name}**: {'PASS' if c['pass'] else 'FAIL'}")
        for k, v in c.get("comparisons", {}).items():
            L.append(f"  - {k}: {v['mean_diff']:+.4f}, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                     f"-> {'pass' if v['pass'] else 'fail'}")
        if "delta_gap" in c:
            L.append(f"  - S-J-R curator-uniform: {c['mean_diff']:+.4f}, 95% CI [{c['ci95'][0]:+.4f}, "
                     f"{c['ci95'][1]:+.4f}], tolerance -{c['delta_gap']}")
    L += ["", f"**Overall: {'PASS' if rep['all_pass'] else 'FAIL'}**", "",
          "## Secondary: Curator minus each method (paired, 95% CI)", "",
          "| scenario | " + " | ".join(m for m in METHODS if m != "curator") + " |",
          "|---|" + "---|" * (len(METHODS) - 1)]
    for sid, per in rep["curator_minus"].items():
        L.append(f"| {sid} | " + " | ".join(
            f"{per[m]['mean_diff']:+.4f} [{per[m]['ci95'][0]:+.4f}, {per[m]['ci95'][1]:+.4f}]"
            for m in METHODS if m != "curator") + " |")
    L += ["", "## Secondary: mean allocation after round 8, concentration, calibration", ""]
    for sid, per in rep["detail"].items():
        envs = list(next(iter(per.values()))["weights"])
        L += [f"### {sid}", "", "| method | " + " | ".join(envs) + " | max-weight | rounds | calib share | noisy S5 | false flags |",
              "|---|" + "---|" * (len(envs) + 5)]
        for m, d in per.items():
            nd = "—" if d["noisy_detected"] is None else f"{d['noisy_detected']:.2f}"
            ff = "—" if d["false_flags"] is None else f"{d['false_flags']:.3f}"
            L.append(f"| {m} | " + " | ".join(f"{d['weights'][e]:.3f}" for e in envs)
                     + f" | {d['mean_max_weight']:.2f} | {d['rounds']:.1f} | {d['calib_share']:.3f} | {nd} | {ff} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
