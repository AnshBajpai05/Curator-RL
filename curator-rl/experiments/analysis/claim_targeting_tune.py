"""Tuning of calibration targeting by proxy claim (D-93). TUNING seeds 0-49 only.

Variants (targeting / max_targets): exposure/1 (current), claim/1, claim/2, claim_stale/1,
claim_stale/2, on the refit scenarios with calibration offered as in D-88 (paired items, churn
0.02, 100 items per slice, K = 10, charged per item). Everything else is `configs/base.yaml`.

Selection rule, fixed in D-93 before this was run: among variants with false-flag rate <= 5%,
calibration cost <= 10% of spend and mean score (over the three scenarios) not below exposure/1
by more than 0.005, pick the highest mean noisy-arm S5 detection; ties within 0.02 go to the
cheaper variant; if nothing beats exposure/1 on detection by >= 0.10 absolute, propose no change.

    python experiments/analysis/claim_targeting_tune.py

Writes reports/analysis/claim_targeting_tune.{json,md}.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from experiments.analysis.calibration_power import S5Recorder  # noqa: E402
from experiments.refit_gate import SCENARIOS, scenario_with_calibration  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

VARIANTS = (("exposure", 1), ("claim", 1), ("claim", 2), ("claim_stale", 1), ("claim_stale", 2))
SEEDS = list(range(50))
NOISY = "noisy"
MAX_FALSE, MAX_CALIB, MAX_LOSS, MIN_GAIN, TIE = 0.05, 0.10, 0.005, 0.10, 0.02


def name(v) -> str:
    return f"{v[0]}/{v[1]}"


def run_cell(task):
    sid, v = task
    base = load_config(FROZEN_D68_D94)
    cfg = base.model_copy(update={"calib": base.calib.model_copy(update={"targeting": v[0], "max_targets": v[1]})})
    sc = scenario_with_calibration(sid, cfg.calib.interval_rounds)
    ids = [e.env_id for e in sc.envs]
    out = {"scores": [], "calib_share": [], "noisy": [], "false": [], "noisy_checks": [], "noisy_weight": []}
    for s in SEEDS:
        sched = make_scheduler("curator", sc, s, cfg=cfg)
        rec = S5Recorder(sched)
        r = run_episode(rec, sc, seed=s, method="curator")
        out["scores"].append(r.final_score)
        out["calib_share"].append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
        out["noisy"].append(NOISY in rec.ever_s5)
        others = [e for e in ids if e != NOISY]
        out["false"].append(sum(e in rec.ever_s5 for e in others) / len(others))
        out["noisy_weight"].append(float(np.mean([lg["weights"][NOISY] for lg in r.round_logs[8:]])))
    return sid, v, out


def main() -> int:
    tasks = [(sid, v) for sid in SCENARIOS for v in VARIANTS]
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(run_cell, tasks))
    res = {(sid, name(v)): o for sid, v, o in cells}
    rows = {}
    for v in VARIANTS:
        n = name(v)
        per = {sid: res[(sid, n)] for sid in SCENARIOS}
        rows[n] = {
            "score": float(np.mean([np.mean(per[s]["scores"]) for s in SCENARIOS])),
            "score_by": {s: float(np.mean(per[s]["scores"])) for s in SCENARIOS},
            "detect": float(np.mean([np.mean(per[s]["noisy"]) for s in SCENARIOS])),
            "detect_by": {s: float(np.mean(per[s]["noisy"])) for s in SCENARIOS},
            "false": float(np.mean([np.mean(per[s]["false"]) for s in SCENARIOS])),
            "calib": float(np.mean([np.mean(per[s]["calib_share"]) for s in SCENARIOS])),
            "noisy_weight": float(np.mean([np.mean(per[s]["noisy_weight"]) for s in SCENARIOS])),
        }
    ref = rows["exposure/1"]
    for r in rows.values():
        r["ok"] = bool(r["false"] <= MAX_FALSE and r["calib"] <= MAX_CALIB and r["score"] >= ref["score"] - MAX_LOSS)
    eligible = {n: r for n, r in rows.items() if r["ok"] and n != "exposure/1"}
    best = None
    if eligible:
        top = max(r["detect"] for r in eligible.values())
        close = [n for n, r in eligible.items() if r["detect"] >= top - TIE]
        best = min(close, key=lambda n: rows[n]["calib"])
    chosen = best if (best and rows[best]["detect"] - ref["detect"] >= MIN_GAIN) else None
    out = {"seeds": [SEEDS[0], SEEDS[-1]], "rows": rows, "chosen": chosen}
    dest = REPO_ROOT / "reports" / "analysis"
    (dest / "claim_targeting_tune.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Calibration targeting by proxy claim, tuning (D-93)", "",
          f"Tuning seeds {SEEDS[0]}..{SEEDS[-1]}; refit scenarios S-I-R1, S-I-R2, S-J-R. Rule: false flags <= {MAX_FALSE}, "
          f"calibration cost <= {MAX_CALIB}, score loss vs exposure/1 <= {MAX_LOSS}; pick the highest detection, ties within "
          f"{TIE} to the cheaper; no change unless detection improves by >= {MIN_GAIN}.", "",
          "| variant | mean score | noisy detected | false-flag rate | calibration cost | noisy weight | eligible | per-scenario detection |",
          "|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        md.append(f"| {n} | {r['score']:.4f} | {r['detect']:.3f} | {r['false']:.3f} | {r['calib']:.3f} | "
                  f"{r['noisy_weight']:.3f} | {'yes' if r['ok'] else 'no'} | "
                  + ", ".join(f"{s} {d:.2f}" for s, d in r["detect_by"].items()) + " |")
    md += ["", f"**Chosen: {chosen or 'none (no change proposed)'}**"]
    (dest / "claim_targeting_tune.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
