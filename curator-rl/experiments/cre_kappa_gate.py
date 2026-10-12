"""Pre-registered evaluation of H-kappa (D-80/D-81): fresh seeds 300-349, run ONCE.

Primary: cre_tuned minus SEC-style and minus DUMP-style in S-I (paired 95% CI excluding 0).
Attribution: cre_tuned minus control_tuned (same kappa/tau, CRE off) in S-I and S-J.
No-regression: cre_tuned minus default-kappa targeted Curator, lower bound > -0.01, in
S-A, S-B, S-C, S-J. References: Uniform, Standard UCB, SEC, DUMP, default-kappa targeted
Curator and default-kappa CRE.

    python experiments/cre_kappa_gate.py --seeds 50 --seed0 300

Writes reports/gate2/cre_kappa_gate.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402

from curator_rl.simulator.harness import run_episode  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.analysis.calibration_power import S5Recorder  # noqa: E402
from experiments.analysis.cre_kappa_tune import with_scheduler  # noqa: E402
from experiments.cre_gate import method_cfg, scenario_variant  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "gate2"
SCENARIOS = ("S-A", "S-B", "S-C", "S-I", "S-J")
TUNED = {"exploration_coef": 0.2, "tau": 1.0}      # frozen in D-81 from tuning seeds
METHODS = ("uniform", "ucb", "sec", "dump", "curator_targeted", "cre", "control_tuned", "cre_tuned")
NOISY = "noisy"
BOOT_SEED = 4041


def cfg_for(method: str):
    if method == "control_tuned":
        return with_scheduler(method_cfg("curator_targeted"), TUNED)
    if method == "cre_tuned":
        return with_scheduler(method_cfg("cre"), TUNED)
    return method_cfg(method)


def factory(method: str) -> str:
    return "curator" if method in ("curator_targeted", "cre", "control_tuned", "cre_tuned") else method


def run_cell(task):
    sid, method, seeds = task
    cfg = cfg_for(method)
    sc = scenario_variant(sid, cfg.calib.interval_rounds)
    out = {"sid": sid, "method": method, "scores": [], "noisy_detected": [], "false_flags": [],
           "calib_share": [], "weights": {}}
    ids = [e.env_id for e in sc.envs]
    acc = {e: [] for e in ids}
    for s in seeds:
        sched = make_scheduler(factory(method), sc, s, cfg=cfg)
        rec = S5Recorder(sched) if getattr(sched, "uses_calibration", False) else sched
        r = run_episode(rec, sc, seed=s, method=method)
        out["scores"].append(r.final_score)
        out["calib_share"].append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
        for e in ids:
            acc[e].append(float(np.mean([log["weights"][e] for log in r.round_logs[8:]])))
        if hasattr(rec, "ever_s5") and NOISY in ids:
            out["noisy_detected"].append(NOISY in rec.ever_s5)
            others = [e for e in ids if e != NOISY]
            out["false_flags"].append(sum(e in rec.ever_s5 for e in others) / len(others))
    out["weights"] = {e: float(np.mean(v)) for e, v in acc.items()}
    return out


def f(v):
    return "-" if v is None else f"{v:.2f}"


def paired(a, b, rng):
    mean, lo, hi = gate2_check.paired_bootstrap([x - y for x, y in zip(a, b)], rng)
    return {"mean_diff": mean, "ci95": [lo, hi]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H-kappa evaluation (D-80/D-81)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=300)
    args = ap.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    tasks = [(sid, m, seeds) for sid in SCENARIOS for m in METHODS]
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(run_cell, tasks))
    res = {(c["sid"], c["method"]): c for c in cells}
    sc = {k: v["scores"] for k, v in res.items()}
    rng = np.random.default_rng(BOOT_SEED)

    prim = {f"S-I:cre_tuned-{r}": paired(sc[("S-I", "cre_tuned")], sc[("S-I", r)], rng) for r in ("sec", "dump")}
    for v in prim.values():
        v["pass"] = bool(v["ci95"][0] > 0.0)
    attr = {f"{sid}:cre_tuned-control_tuned": paired(sc[(sid, "cre_tuned")], sc[(sid, "control_tuned")], rng)
            for sid in ("S-I", "S-J")}
    regress = {}
    for sid in ("S-A", "S-B", "S-C", "S-J"):
        r = paired(sc[(sid, "cre_tuned")], sc[(sid, "curator_targeted")], rng)
        r["pass"] = bool(r["ci95"][0] > -gate2_check.DELTA_GAP)
        regress[f"{sid}:cre_tuned-curator_targeted"] = r
    info = {sid: {m: paired(sc[(sid, "cre_tuned")], sc[(sid, m)], rng) for m in METHODS if m != "cre_tuned"}
            for sid in SCENARIOS}
    means = {sid: {m: float(np.mean(sc[(sid, m)])) for m in METHODS} for sid in SCENARIOS}
    diag = {sid: {m: {"noisy_detected": float(np.mean(res[(sid, m)]["noisy_detected"])) if res[(sid, m)]["noisy_detected"] else None,
                      "false_flag": float(np.mean(res[(sid, m)]["false_flags"])) if res[(sid, m)]["false_flags"] else None,
                      "calib_cost_share": float(np.mean(res[(sid, m)]["calib_share"])),
                      "weights": res[(sid, m)]["weights"]}
                  for m in ("curator_targeted", "cre", "control_tuned", "cre_tuned")} for sid in ("S-I", "S-J")}
    primary_pass = all(v["pass"] for v in prim.values())
    cre_vs_control = attr["S-I:cre_tuned-control_tuned"]
    if primary_pass and cre_vs_control["ci95"][0] > 0:
        verdict = "PRIMARY PASS with a positive CRE-minus-control gap: supports 'calibrated reward adds value once kappa is matched'."
    elif primary_pass:
        verdict = "PRIMARY PASS but CRE is not distinguishable from the tuned control: the gain is from kappa/tau, not the CRE; CRE not adopted."
    else:
        verdict = "PRIMARY FAIL: the kappa explanation is closed for this scenario; CRE not adopted."
    rep = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": seeds, "tuned": TUNED,
           "primary": prim, "primary_pass": bool(primary_pass), "attribution": attr, "no_regression": regress,
           "verdict": verdict, "means": means, "cre_tuned_minus": info, "diagnostics": diag}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cre_kappa_gate.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = [f"# H-kappa evaluation (D-80 / D-81), seeds {seeds[0]}..{seeds[-1]}", "",
          f"Tuned point (D-81): {json.dumps(TUNED)} for both cre_tuned and control_tuned.", "",
          "## Mean final scores", "", "| scenario | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for sid, per in means.items():
        md.append(f"| {sid} | " + " | ".join(f"{per[m]:.4f}" for m in METHODS) + " |")
    md += ["", "## Primary (S-I)", ""]
    md += [f"- {k}: {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] -> {'pass' if v['pass'] else 'fail'}" for k, v in prim.items()]
    md += ["", "## Attribution", ""]
    md += [f"- {k}: {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in attr.items()]
    md += ["", "## No regression (lower bound > -0.01)", ""]
    md += [f"- {k}: {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] -> {'pass' if v['pass'] else 'fail'}" for k, v in regress.items()]
    md += ["", f"**Verdict:** {verdict}", "", "## cre_tuned minus each method (paired, 95% CI)", "",
           "| scenario | " + " | ".join(m for m in METHODS if m != "cre_tuned") + " |", "|---|" + "---|" * (len(METHODS) - 1)]
    for sid, per in info.items():
        md.append(f"| {sid} | " + " | ".join(f"{per[m]['mean_diff']:+.3f} [{per[m]['ci95'][0]:+.3f}, {per[m]['ci95'][1]:+.3f}]" for m in METHODS if m != "cre_tuned") + " |")
    md += ["", "## Diagnostics: S5 detection, false flags, calibration cost, allocation after round 8", ""]
    for sid, per in diag.items():
        md += [f"### {sid}", "", "| method | noisy detected | false flags | calib cost | gsm8k | math35 | mbpp | countdown | noisy | too_hard |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m, d in per.items():
            md.append(f"| {m} | {f(d['noisy_detected'])} | {f(d['false_flag'])} | {d['calib_cost_share']:.3f} | "
                      + " | ".join(f"{d['weights'][e]:.2f}" for e in ("gsm8k", "math35", "mbpp", "countdown", "noisy", "too_hard")) + " |")
        md.append("")
    (REPORT_DIR / "cre_kappa_gate.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0 if primary_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
