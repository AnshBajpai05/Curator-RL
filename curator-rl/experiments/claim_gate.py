"""Pre-registered evaluation H-claim (D-94): calibration targeting `claim_stale`, 2 targets. Run ONCE.

Candidate = `configs/base.yaml` with calib.targeting = claim_stale, calib.max_targets = 2.
Fresh seeds 600-699, refit scenarios with calibration offered as in D-88.

Adopt iff all pass, in each scenario unless stated:
  C1  noisy-arm S5 detection: claim_stale/2 minus exposure/1, paired indicator, 95% CI lower bound > 0
  C2  mean false-flag rate <= 5% and calibration cost <= 10% of spend
  C3  score non-inferiority: Curator(claim_stale/2) minus Curator(exposure/1), CI lower bound > -0.01
  C4  Gate 2'' retained: minus SEC-style and DUMP-style > 0 (S-I-R1, S-I-R2); S-J-R minus Uniform > -0.01
Secondary (reported only): score difference, noisy budget share, minus Standard UCB / LP, and the
candidate with gamma = 1 (curator_g1_cs2, exploratory).

    python experiments/claim_gate.py --seeds 100 --seed0 600

Writes reports/gate2/claim_gate.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.analysis.calibration_power import S5Recorder  # noqa: E402
from experiments.refit_gate import SCENARIOS, scenario_with_calibration  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

REPORT_DIR = REPO_ROOT / "reports" / "gate2"
METHODS = ("uniform", "lp", "ucb", "sec", "dump", "curator", "curator_cs2", "curator_g1_cs2")
NOISY = "noisy"
DELTA = gate2_check.DELTA_GAP
BOOT_SEED = 8081


def cfg_for(method: str):
    base = load_config(FROZEN_D68_D94)
    if method in ("curator_cs2", "curator_g1_cs2"):
        base = base.model_copy(update={"calib": base.calib.model_copy(
            update={"targeting": "claim_stale", "max_targets": 2})})
    if method == "curator_g1_cs2":
        base = base.model_copy(update={"scheduler": base.scheduler.model_copy(update={"gamma": 1.0})})
    return base


def run_cell(task):
    sid, method, seeds = task
    cfg = cfg_for(method)
    sc = scenario_with_calibration(sid, cfg.calib.interval_rounds)
    ids = [e.env_id for e in sc.envs]
    out = {"scores": [], "calib": [], "noisy": [], "false": [], "noisy_w": []}
    for s in seeds:
        factory = "curator" if method.startswith("curator") else method
        sched = make_scheduler(factory, sc, s, cfg=cfg)
        rec = S5Recorder(sched) if getattr(sched, "uses_calibration", False) else sched
        r = run_episode(rec, sc, seed=s, method=method)
        out["scores"].append(r.final_score)
        out["calib"].append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
        out["noisy_w"].append(float(np.mean([lg["weights"][NOISY] for lg in r.round_logs[8:]])))
        if hasattr(rec, "ever_s5"):
            out["noisy"].append(float(NOISY in rec.ever_s5))
            others = [e for e in ids if e != NOISY]
            out["false"].append(sum(e in rec.ever_s5 for e in others) / len(others))
    return sid, method, out


def paired(a, b, rng):
    m, lo, hi = gate2_check.paired_bootstrap([x - y for x, y in zip(a, b)], rng)
    return {"mean_diff": m, "ci95": [lo, hi]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H-claim pre-registered evaluation (D-94)")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=600)
    args = ap.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(run_cell, [(s, m, seeds) for s in SCENARIOS for m in METHODS]))
    res = {(sid, m): o for sid, m, o in cells}
    rng = np.random.default_rng(BOOT_SEED)
    crit = {"C1_detection": {}, "C2_safety": {}, "C3_non_inferiority": {}, "C4_gate2_retained": {}}
    for sid in SCENARIOS:
        a, b = res[(sid, "curator_cs2")], res[(sid, "curator")]
        r = paired(a["noisy"], b["noisy"], rng)
        r["detect_cs2"], r["detect_exposure"] = float(np.mean(a["noisy"])), float(np.mean(b["noisy"]))
        r["pass"] = bool(r["ci95"][0] > 0.0)
        crit["C1_detection"][sid] = r
        crit["C2_safety"][sid] = {"false_flag": float(np.mean(a["false"])), "calib_cost": float(np.mean(a["calib"])),
                                  "pass": bool(np.mean(a["false"]) <= 0.05 and np.mean(a["calib"]) <= 0.10)}
        r = paired(a["scores"], b["scores"], rng)
        r["pass"] = bool(r["ci95"][0] > -DELTA)
        crit["C3_non_inferiority"][sid] = r
    for sid in ("S-I-R1", "S-I-R2"):
        for rival in ("sec", "dump"):
            r = paired(res[(sid, "curator_cs2")]["scores"], res[(sid, rival)]["scores"], rng)
            r["pass"] = bool(r["ci95"][0] > 0.0)
            crit["C4_gate2_retained"][f"{sid}:cs2-{rival}"] = r
    r = paired(res[("S-J-R", "curator_cs2")]["scores"], res[("S-J-R", "uniform")]["scores"], rng)
    r["pass"] = bool(r["ci95"][0] > -DELTA)
    crit["C4_gate2_retained"]["S-J-R:cs2-uniform"] = r
    overall = {k: bool(all(v["pass"] for v in d.values())) for k, d in crit.items()}
    adopt = bool(all(overall.values()))
    means = {s: {m: float(np.mean(res[(s, m)]["scores"])) for m in METHODS} for s in SCENARIOS}
    secondary = {s: {r_: paired(res[(s, "curator_cs2")]["scores"], res[(s, r_)]["scores"], rng)
                     for r_ in ("curator", "ucb", "lp")} for s in SCENARIOS}
    exploratory = {s: {"g1_cs2_minus_cs2": paired(res[(s, "curator_g1_cs2")]["scores"], res[(s, "curator_cs2")]["scores"], rng),
                       "g1_cs2_minus_ucb": paired(res[(s, "curator_g1_cs2")]["scores"], res[(s, "ucb")]["scores"], rng)}
                   for s in SCENARIOS}
    detail = {s: {m: {"noisy_weight": float(np.mean(res[(s, m)]["noisy_w"])),
                      "detect": float(np.mean(res[(s, m)]["noisy"])) if res[(s, m)]["noisy"] else None,
                      "false": float(np.mean(res[(s, m)]["false"])) if res[(s, m)]["false"] else None,
                      "calib": float(np.mean(res[(s, m)]["calib"]))}
                  for m in ("curator", "curator_cs2", "curator_g1_cs2")} for s in SCENARIOS}
    rep = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": [seeds[0], seeds[-1]],
           "criteria": crit, "overall": overall, "adopt": adopt, "means": means, "secondary": secondary,
           "exploratory": exploratory, "detail": detail}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "claim_gate.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = render(rep)
    (REPORT_DIR / "claim_gate.md").write_text(md, encoding="utf-8")
    print(md)
    return 0 if adopt else 1


def render(rep: dict) -> str:
    L = ["# H-claim: calibration targeting claim_stale/2 (D-94, pre-registered)", "",
         f"Generated {rep['generated_utc']}; fresh seeds {rep['seeds'][0]}..{rep['seeds'][1]}; refit scenarios; "
         "candidate = base config with calib.targeting claim_stale, max_targets 2.", "",
         "## Mean final scores", "", "| scenario | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for s, per in rep["means"].items():
        L.append(f"| {s} | " + " | ".join(f"{per[m]:.4f}" for m in METHODS) + " |")
    L += ["", "## Primary criteria", ""]
    for name, d in rep["criteria"].items():
        L.append(f"- **{name}**: {'PASS' if rep['overall'][name] else 'FAIL'}")
        for k, v in d.items():
            if "ci95" in v:
                extra = (f" (detection {v['detect_cs2']:.2f} vs {v['detect_exposure']:.2f})" if "detect_cs2" in v else "")
                L.append(f"  - {k}: {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]{extra} -> "
                         f"{'pass' if v['pass'] else 'fail'}")
            else:
                L.append(f"  - {k}: false-flag {v['false_flag']:.3f}, calibration cost {v['calib_cost']:.3f} -> "
                         f"{'pass' if v['pass'] else 'fail'}")
    L += ["", f"**Adopt claim_stale/2: {'YES' if rep['adopt'] else 'NO'}**", "",
          "## Secondary: score of the candidate minus Curator (exposure/1), Standard UCB, LP", "",
          "| scenario | curator | ucb | lp |", "|---|---|---|---|"]
    for s, per in rep["secondary"].items():
        L.append(f"| {s} | " + " | ".join(f"{per[r]['mean_diff']:+.4f} [{per[r]['ci95'][0]:+.4f}, {per[r]['ci95'][1]:+.4f}]"
                                          for r in ("curator", "ucb", "lp")) + " |")
    L += ["", "## Exploratory (no decision attached): candidate with gamma = 1", "",
          "| scenario | minus candidate (gamma 0.95) | minus Standard UCB |", "|---|---|---|"]
    for s, per in rep["exploratory"].items():
        L.append(f"| {s} | " + " | ".join(f"{per[k]['mean_diff']:+.4f} [{per[k]['ci95'][0]:+.4f}, {per[k]['ci95'][1]:+.4f}]"
                                          for k in ("g1_cs2_minus_cs2", "g1_cs2_minus_ucb")) + " |")
    L += ["", "## Detail", "", "| scenario | method | noisy S5 | false-flag | calibration cost | noisy budget share |",
          "|---|---|---|---|---|---|"]
    for s, per in rep["detail"].items():
        for m, d in per.items():
            L.append(f"| {s} | {m} | {d['detect']:.2f} | {d['false']:.3f} | {d['calib']:.3f} | {d['noisy_weight']:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
