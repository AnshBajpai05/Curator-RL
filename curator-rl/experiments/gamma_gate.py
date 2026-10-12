"""Pre-registered evaluation of H-gamma1 (D-91): Curator without discounting. Run ONCE.

Candidate: Curator exactly as in `configs/base.yaml` except scheduler.gamma = 1.0 (chosen in the
D-90 diagnosis on tuning seeds 0-49; this run uses fresh seeds 500-549).

Primary (adopt gamma = 1 iff all pass):
  A1  Curator-g1 minus Curator (gamma 0.95), paired 95% CI excluding 0, in S-I-R1, S-I-R2 and S-J-R.
  A2  Curator-g1 minus SEC-style and minus DUMP-style, CI excluding 0, in S-I-R1 and S-I-R2.
  A3  S-J-R: Curator-g1 minus Uniform, lower CI bound > -0.01.
  A4  No regression on the original suite (S-A, S-B, S-C, S-D, S-E, S-F, S-H, S-I, S-J): Curator-g1
      minus Curator, lower CI bound > -0.01 in every scenario.
Secondary (reported only): Curator-g1 minus Standard UCB and minus LP per refit scenario; allocation,
concentration (mean max-weight), noisy-arm share and S5 detection.

    python experiments/gamma_gate.py --seeds 50 --seed0 500

Writes reports/gate2/gamma_gate.{json,md}.
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
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.analysis.calibration_power import S5Recorder  # noqa: E402
from experiments.refit_gate import scenario_with_calibration  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

REPORT_DIR = REPO_ROOT / "reports" / "gate2"
REFIT = ("S-I-R1", "S-I-R2", "S-J-R")
ORIGINAL = ("S-A", "S-B", "S-C", "S-D", "S-E", "S-F", "S-H", "S-I", "S-J")
REFIT_METHODS = ("uniform", "lp", "ucb", "sec", "dump", "curator", "curator_g1")
ORIGINAL_METHODS = ("curator", "curator_g1")
DELTA = gate2_check.DELTA_GAP
SKIP = 8
BOOT_SEED = 7071


def cfg_for(method: str):
    base = load_config(FROZEN_D68_D94)
    if method == "curator_g1":
        return base.model_copy(update={"scheduler": base.scheduler.model_copy(update={"gamma": 1.0})})
    return base


def scenario(sid: str, interval: int):
    if sid in REFIT:
        return scenario_with_calibration(sid, interval)
    return load_scenario(REPO_ROOT / "configs" / "sim" / f"scenario_{sid.replace('-', '').lower()}.yaml")


def run_cell(task: tuple) -> dict:
    sid, method, seeds = task
    cfg = cfg_for(method)
    sc = scenario(sid, cfg.calib.interval_rounds)
    ids = [e.env_id for e in sc.envs]
    out = {"sid": sid, "method": method, "scores": [], "weights": {e: [] for e in ids}, "max_w": [], "s5": []}
    for s in seeds:
        factory = "curator" if method.startswith("curator") else method
        sched = make_scheduler(factory, sc, s, cfg=cfg)
        rec = S5Recorder(sched) if (getattr(sched, "uses_calibration", False) and sc.calib.enabled) else sched
        r = run_episode(rec, sc, seed=s, method=method)
        out["scores"].append(r.final_score)
        for e in ids:
            out["weights"][e].append(float(np.mean([lg["weights"][e] for lg in r.round_logs[SKIP:]])))
        out["max_w"].append(float(r.concentration(skip_rounds=SKIP)["mean_max_weight"]))
        if hasattr(rec, "ever_s5"):
            out["s5"].append("noisy" in rec.ever_s5)
    return out


def paired(a, b, rng) -> dict:
    m, lo, hi = gate2_check.paired_bootstrap([x - y for x, y in zip(a, b)], rng)
    return {"mean_diff": m, "ci95": [lo, hi]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H-gamma1 pre-registered evaluation (D-91)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=500)
    args = ap.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    tasks = [(s, m, seeds) for s in REFIT for m in REFIT_METHODS]
    tasks += [(s, m, seeds) for s in ORIGINAL for m in ORIGINAL_METHODS]
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(run_cell, tasks))
    res = {(c["sid"], c["method"]): c for c in cells}
    sc = {k: v["scores"] for k, v in res.items()}
    rng = np.random.default_rng(BOOT_SEED)

    def block(pairs, lower):
        comps, ok = {}, True
        for key, a, b in pairs:
            r = paired(sc[a], sc[b], rng)
            r["pass"] = bool(r["ci95"][0] > lower)
            ok &= r["pass"]
            comps[key] = r
        return {"comparisons": comps, "pass": bool(ok)}

    crit = {
        "A1_improves_on_gamma_0.95": block(
            [(f"{s}:g1-curator", (s, "curator_g1"), (s, "curator")) for s in REFIT], 0.0),
        "A2_beats_learnability_bandits": block(
            [(f"{s}:g1-{r}", (s, "curator_g1"), (s, r)) for s in ("S-I-R1", "S-I-R2") for r in ("sec", "dump")], 0.0),
        "A3_homogeneous_vs_uniform": block([("S-J-R:g1-uniform", ("S-J-R", "curator_g1"), ("S-J-R", "uniform"))], -DELTA),
        "A4_no_regression_original_suite": block(
            [(f"{s}:g1-curator", (s, "curator_g1"), (s, "curator")) for s in ORIGINAL], -DELTA),
    }
    secondary = {s: {r: paired(sc[(s, "curator_g1")], sc[(s, r)], rng) for r in ("ucb", "lp", "uniform")}
                 for s in REFIT}
    means = {s: {m: float(np.mean(sc[(s, m)])) for m in (REFIT_METHODS if s in REFIT else ORIGINAL_METHODS)}
             for s in (*REFIT, *ORIGINAL)}
    detail = {s: {m: {"weights": {e: float(np.mean(w)) for e, w in res[(s, m)]["weights"].items()},
                      "max_w": float(np.mean(res[(s, m)]["max_w"])),
                      "noisy_s5": float(np.mean(res[(s, m)]["s5"])) if res[(s, m)]["s5"] else None}
                  for m in REFIT_METHODS} for s in REFIT}
    rep = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": seeds, "criteria": crit,
           "adopt": bool(all(c["pass"] for c in crit.values())), "means": means, "secondary": secondary,
           "detail": detail}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "gamma_gate.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = render(rep)
    (REPORT_DIR / "gamma_gate.md").write_text(md, encoding="utf-8")
    print(md)
    return 0 if rep["adopt"] else 1


def render(rep: dict) -> str:
    L = ["# H-gamma1: Curator without discounting (D-91, pre-registered)", "",
         f"Generated {rep['generated_utc']}; fresh seeds {rep['seeds'][0]}..{rep['seeds'][-1]}; candidate = "
         "`configs/base.yaml` with scheduler.gamma = 1.0, everything else frozen.", "",
         "## Mean final scores", ""]
    L += ["| scenario | " + " | ".join(REFIT_METHODS) + " |", "|---|" + "---|" * len(REFIT_METHODS)]
    for s in REFIT:
        L.append(f"| {s} | " + " | ".join(f"{rep['means'][s][m]:.4f}" for m in REFIT_METHODS) + " |")
    L += ["", "| original scenario | curator (gamma 0.95) | curator_g1 |", "|---|---|---|"]
    for s in ORIGINAL:
        L.append(f"| {s} | {rep['means'][s]['curator']:.4f} | {rep['means'][s]['curator_g1']:.4f} |")
    L += ["", "## Primary criteria", ""]
    for name, c in rep["criteria"].items():
        L.append(f"- **{name}**: {'PASS' if c['pass'] else 'FAIL'}")
        for k, v in c["comparisons"].items():
            L.append(f"  - {k}: {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] -> "
                     f"{'pass' if v['pass'] else 'fail'}")
    L += ["", f"**Adopt gamma = 1: {'YES' if rep['adopt'] else 'NO'}**", "",
          "## Secondary: Curator-g1 minus Standard UCB / LP / Uniform", "",
          "| scenario | ucb | lp | uniform |", "|---|---|---|---|"]
    for s, per in rep["secondary"].items():
        L.append(f"| {s} | " + " | ".join(
            f"{per[r]['mean_diff']:+.4f} [{per[r]['ci95'][0]:+.4f}, {per[r]['ci95'][1]:+.4f}]"
            for r in ("ucb", "lp", "uniform")) + " |")
    L += ["", "## Secondary: allocation after round 8", ""]
    for s, per in rep["detail"].items():
        envs = list(per["curator"]["weights"])
        L += [f"### {s}", "", "| method | " + " | ".join(envs) + " | max-weight | noisy S5 |",
              "|---|" + "---|" * (len(envs) + 2)]
        for m, d in per.items():
            s5 = "-" if d["noisy_s5"] is None else f"{d['noisy_s5']:.2f}"
            L.append(f"| {m} | " + " | ".join(f"{d['weights'][e]:.3f}" for e in envs) + f" | {d['max_w']:.2f} | {s5} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
