"""Pre-registered evaluation of the Calibrated Reward Engine (D-76 / D-77).

Fresh evaluation seeds (default 200-249). Run ONCE after the design is frozen; nothing
here may be changed after its outcome is seen (new ideas get a new pre-registration).

Primary (Gate 2'' as written): (i) CRE-Curator minus SEC-style and minus DUMP-style in
S-C and S-I, paired 95% bootstrap CI excluding 0; (ii) S-J: CRE minus Uniform lower
bound > -0.01; (iii) Gate 2 (a), (b), (c), (e) with the CRE variant ((d) status F1 is
not re-evaluated: classifier code unchanged).

Reported, never used to select: ablations (no_uncertainty, exact_cost [privileged],
discount 0.7), existing targeted Curator, Curator without calibration, references
(Uniform, Static, LP, Standard UCB, SEC, DUMP), allocation of MATH/MBPP/noisy, S5
detection and false-flag rates, calibration cost share, seed-level differences.

    python experiments/cre_gate.py --seeds 50 --seed0 200

Writes reports/gate2/cre_gate.{json,md}.
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
from curator_rl.simulator.scenarios import CalibSimCfg, ReportEvalCfg, load_scenario  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.analysis.calibration_power import S5Recorder  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

REPORT_DIR = REPO_ROOT / "reports" / "gate2"
SCENARIOS = ("S-A", "S-B", "S-C", "S-I", "S-J")
METHODS = ("uniform", "static", "lp", "ucb", "sec", "dump", "curator_nocalib", "curator_targeted",
           "cre", "cre_no_unc", "cre_exact", "cre_disc")
CRE_METHODS = ("cre", "cre_no_unc", "cre_exact", "cre_disc")
NOISY = "noisy"
DELTA_GAP = gate2_check.DELTA_GAP
SKIP = 8
BOOT_SEED = 3031


def _path(sid: str) -> Path:
    return REPO_ROOT / "configs" / "sim" / f"scenario_{sid.replace('-', '').lower()}.yaml"


def scenario_variant(sid: str, calib_interval: int):
    """Scenario offering the D-75 calibration (users pay per item); S-C also reports uncharged evals."""
    sc = load_scenario(str(_path(sid)))
    cost_item = float(np.mean([e.cost_usd_per_prompt for e in sc.envs])) / sc.group_size
    update = {"calib": CalibSimCfg(enabled=True, interval_rounds=calib_interval, paired=True, churn=0.02,
                                   items_per_slice=100, cost_per_item_usd=cost_item)}
    if sid == "S-C":
        update["report_eval"] = ReportEvalCfg(interval_rounds=2)
    return sc.model_copy(update=update)


def method_cfg(method: str):
    base = load_config(FROZEN_D68_D94)
    cre = base.cre
    calib = base.calib
    if method == "curator_nocalib":
        calib = calib.model_copy(update={"enabled": False})
    if method in CRE_METHODS:
        cre = cre.model_copy(update={"enabled": True})
    if method == "cre_no_unc":
        cre = cre.model_copy(update={"mode": "no_uncertainty"})
    if method == "cre_disc":
        cre = cre.model_copy(update={"discount": 0.7})
    return base.model_copy(update={"cre": cre, "calib": calib})


def _factory(method: str):
    return "curator" if method.startswith(("cre", "curator_")) else method


def run_cell(task: tuple) -> dict:
    sid, method, seeds = task
    cfg = method_cfg(method)
    sc = scenario_variant(sid, cfg.calib.interval_rounds)
    floor = cfg.scheduler.epsilon / len(sc.envs)
    out = {"sid": sid, "method": method, "scores": [], "rounds": [], "calib_share": [], "weights": {},
           "noisy_detected": [], "false_flags": [], "junk_share": [], "report": [], "first_cre_round": []}
    ids = [e.env_id for e in sc.envs]
    w_acc = {e: [] for e in ids}
    for s in seeds:
        sched = make_scheduler(_factory(method), sc, s, cfg=cfg)
        if method == "cre_exact":
            sched.cost_override = {e.env_id: e.cost_usd_per_prompt for e in sc.envs}
        rec = S5Recorder(sched) if getattr(sched, "uses_calibration", False) else sched
        r = run_episode(rec, sc, seed=s, method=method)
        out["scores"].append(r.final_score)
        out["rounds"].append(r.rounds)
        out["calib_share"].append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
        for e in ids:
            w_acc[e].append(float(np.mean([log["weights"][e] for log in r.round_logs[SKIP:]])))
        if hasattr(rec, "ever_s5") and NOISY in ids:
            out["noisy_detected"].append(NOISY in rec.ever_s5)
            others = [e for e in ids if e != NOISY]
            out["false_flags"].append(sum(e in rec.ever_s5 for e in others) / len(others))
        if sid == "S-B":
            late = [log for log in r.round_logs if log["round"] > int(r.rounds * 0.6)]
            out["junk_share"].append(sum(log["weights"]["too_hard"] for log in late) / max(len(late), 1)
                                     if late else float("nan"))
        if sid == "S-C":
            out["report"].append([(p["spend_usd"], p["score"]) for p in r.report_logs])
            out["budget"] = r.budget_usd
        reward_log = getattr(sched, "reward_log", None)
        if reward_log is not None and method in CRE_METHODS:
            first = next((row["round"] for row in reward_log
                          if any(isinstance(v, dict) and str(v.get("src", "")).startswith("cre:")
                                 for k, v in row.items() if k != "round")), None)
            out["first_cre_round"].append(first)
    out["weights"] = {e: float(np.mean(v)) for e, v in w_acc.items()}
    out["floor"] = floor
    return out


def paired(a, b, rng):
    mean, lo, hi = gate2_check.paired_bootstrap([x - y for x, y in zip(a, b)], rng)
    return {"mean_diff": mean, "ci95": [lo, hi]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CRE pre-registered evaluation (D-76)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    t0 = time.perf_counter()
    tasks = [(sid, m, seeds) for sid in SCENARIOS for m in METHODS]
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        cells = list(pool.map(run_cell, tasks))
    res = {(c["sid"], c["method"]): c for c in cells}
    rng = np.random.default_rng(BOOT_SEED)
    sc = {k: v["scores"] for k, v in res.items()}

    crit: dict[str, dict] = {}
    comps, ok = {}, True
    for sid in ("S-C", "S-I"):
        for rival in ("sec", "dump"):
            r = paired(sc[(sid, "cre")], sc[(sid, rival)], rng)
            r["pass"] = bool(r["ci95"][0] > 0.0)
            ok &= r["pass"]
            comps[f"{sid}:cre-{rival}"] = r
    crit["i_beats_learnability_bandits"] = {"comparisons": comps, "pass": bool(ok)}
    r = paired(sc[("S-J", "cre")], sc[("S-J", "uniform")], rng)
    r["delta_gap"] = DELTA_GAP
    r["pass"] = bool(r["ci95"][0] > -DELTA_GAP)
    crit["ii_homogeneous_no_underperformance"] = r
    ra = paired(sc[("S-A", "cre")], sc[("S-A", "uniform")], rng)
    ra["pass"] = bool(ra["ci95"][0] > 0.0)
    rb = paired(sc[("S-B", "cre")], sc[("S-B", "uniform")], rng)
    rb["pass"] = bool(rb["ci95"][0] > -DELTA_GAP)
    cell_b = res[("S-B", "cre")]
    ok_share = sum(1 for v in cell_b["junk_share"] if v <= gate2_check.JUNK_FACTOR * cell_b["floor"])
    rc = {"seeds_ok": ok_share, "n_seeds": len(cell_b["junk_share"]),
          "pass": bool(ok_share >= 0.9 * len(cell_b["junk_share"]))}
    target = float(np.median(sc[("S-C", "uniform")]))
    budget = res[("S-C", "cre")]["budget"]

    def tt(cell):
        return [gate2_check.time_to_target(p or [(budget, 0.0)], target, budget) for p in cell["report"]]

    impr = [(u - c) / u for c, u in zip(tt(res[("S-C", "cre")]), tt(res[("S-C", "uniform")]))]
    m_e, lo_e, hi_e = gate2_check.paired_bootstrap(impr, rng)
    re_ = {"target": target, "mean_improvement": m_e, "ci95": [lo_e, hi_e],
           "pass": bool(m_e >= gate2_check.COMPUTE_TO_TARGET_MIN and lo_e > 0.0)}
    crit["iii_gate2_with_cre"] = {"a": ra, "b": rb, "c": rc, "e": re_,
                                  "pass": bool(ra["pass"] and rb["pass"] and rc["pass"] and re_["pass"]),
                                  "d_status_f1": "not re-evaluated (classifier code unchanged)"}
    all_pass = all(c["pass"] for c in crit.values())

    info = {sid: {m: paired(sc[(sid, "cre")], sc[(sid, m)], rng) for m in METHODS if m != "cre"}
            for sid in SCENARIOS}
    means = {sid: {m: float(np.mean(sc[(sid, m)])) for m in METHODS} for sid in SCENARIOS}
    diag = {}
    for sid in ("S-I", "S-J"):
        diag[sid] = {m: {
            "noisy_detected": float(np.mean(res[(sid, m)]["noisy_detected"])) if res[(sid, m)]["noisy_detected"] else None,
            "false_flag": float(np.mean(res[(sid, m)]["false_flags"])) if res[(sid, m)]["false_flags"] else None,
            "calib_cost_share": float(np.mean(res[(sid, m)]["calib_share"])),
            "rounds": float(np.mean(res[(sid, m)]["rounds"])),
            "first_cre_round": (float(np.mean([v for v in res[(sid, m)]["first_cre_round"] if v is not None]))
                                if any(v is not None for v in res[(sid, m)]["first_cre_round"]) else None),
            "weights": res[(sid, m)]["weights"],
        } for m in ("uniform", "ucb", "sec", "curator_targeted", "cre", "cre_no_unc", "cre_exact", "cre_disc")}
    activations = {sid: [v for v in res[(sid, "cre")]["first_cre_round"]] for sid in SCENARIOS}
    inert = {sid: float(np.mean([v is None for v in activations[sid]])) for sid in SCENARIOS}

    report = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": seeds,
              "criteria": crit, "all_pass": bool(all_pass), "means": means, "cre_minus": info,
              "diagnostics": diag, "cre_inert_fraction": inert,
              "scores_by_seed": {f"{sid}:{m}": sc[(sid, m)] for sid in SCENARIOS for m in
                                 ("uniform", "ucb", "sec", "dump", "curator_targeted", "cre")},
              "seconds": round(time.perf_counter() - t0, 1)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cre_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "cre_gate.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0 if all_pass else 1


def render(rep: dict) -> str:
    L = ["# CRE pre-registered evaluation (D-76 / D-77)", "",
         f"Seeds {rep['seeds'][0]}..{rep['seeds'][-1]} (fresh); generated {rep['generated_utc']}; "
         f"S-A/S-B/S-C episodes are ~25 rounds, so with K=10 the CRE never activates there "
         f"(inert fraction: {json.dumps({k: round(v, 2) for k, v in rep['cre_inert_fraction'].items()})}).", "",
         "## Mean final scores", "", "| scenario | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for sid, per in rep["means"].items():
        L.append(f"| {sid} | " + " | ".join(f"{per[m]:.4f}" for m in METHODS) + " |")
    L += ["", "## Criteria (Gate 2'' as pre-registered)", ""]
    for name, c in rep["criteria"].items():
        L.append(f"- **{name}**: {'PASS' if c['pass'] else 'FAIL'}")
        for k, v in c.get("comparisons", {}).items():
            L.append(f"  - {k}: {v['mean_diff']:+.4f}, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] -> {'pass' if v['pass'] else 'fail'}")
        if "ci95" in c:
            L.append(f"  - S-J cre-uniform: {c['mean_diff']:+.4f}, CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], tolerance -{c['delta_gap']}")
        for k in ("a", "b", "c", "e"):
            if k in c:
                L.append(f"  - ({k}) {json.dumps({a: b for a, b in c[k].items()}, default=str)}")
    L += ["", f"**Overall: {'PASS' if rep['all_pass'] else 'FAIL'}**", "",
          "## CRE minus each method (paired, 95% CI)", "",
          "| scenario | " + " | ".join(m for m in METHODS if m != "cre") + " |", "|---|" + "---|" * (len(METHODS) - 1)]
    for sid, per in rep["cre_minus"].items():
        L.append(f"| {sid} | " + " | ".join(
            f"{per[m]['mean_diff']:+.3f} [{per[m]['ci95'][0]:+.3f}, {per[m]['ci95'][1]:+.3f}]" for m in METHODS if m != "cre") + " |")
    L += ["", "## Diagnostics (S-I, S-J): detection, false flags, calibration cost, allocation after round 8", ""]
    for sid, per in rep["diagnostics"].items():
        L += [f"### {sid}", "", "| method | noisy detected | false-flag rate | calib cost share | rounds | CRE first active round | "
              "gsm8k | math35 | mbpp | countdown | noisy | too_hard |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, d in per.items():
            w = d["weights"]
            f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
            L.append(f"| {m} | {f(d['noisy_detected'])} | {f(d['false_flag'])} | {d['calib_cost_share']:.3f} | "
                     f"{d['rounds']:.1f} | {f(d['first_cre_round'])} | " + " | ".join(
                         f"{w[e]:.2f}" for e in ("gsm8k", "math35", "mbpp", "countdown", "noisy", "too_hard")) + " |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
