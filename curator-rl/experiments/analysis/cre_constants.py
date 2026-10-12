"""Freeze the CRE constants from TUNING seeds only (D-76 / D-77 / D-78).

s    = cre.proxy_scale  : median over (seed, arm) with evidence of b_hat_j * c_j / xbar_j,
       the factor that turns the proxy signal into gain per dollar (c_j: measured unit
       cost, xbar_j: the arm's exposure-weighted mean proxy signal over the episode).
rho  = cre.prior_rel_sd : robust SD (1.4826 * MAD, clipped to [0.3, 3]) of the relative
       residual (b_hat_j - m0_j) / max(|m0_j|, floor) with m0_j = s * xbar_j / c_j.
R_max = cre.roi_scale   : 95th percentile of b_hat_j * B (benchmark gain if the whole
       budget were spent on j) over arms with positive evidence, floored at 0.02.

To measure the PROXY's error (not evaluation noise) calibration here is free and high
power: every slice, 400 paired items, cost 0. Tuning seeds 0..n-1, scenarios S-A, S-B,
S-C, S-I (never S-J, never evaluation seeds). The engine is PASSIVE (evidence only), so the
trajectory is the existing targeted Curator's and the constants do not depend on themselves.

    python experiments/analysis/cre_constants.py --seeds 50

Writes reports/analysis/cre_constants.{json,md}; it does NOT edit base.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import CalibSimCfg, load_scenario  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "analysis"
SCENARIOS = ("S-A", "S-B", "S-C", "S-I")
ITEMS = 400
RHO_BOUNDS = (0.3, 3.0)
ROI_FLOOR = 0.02


def _path(sid: str) -> Path:
    return REPO_ROOT / "configs" / "sim" / f"scenario_{sid.replace('-', '').lower()}.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze CRE constants (D-77)")
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    base = load_config(REPO_ROOT / "configs" / "base.yaml")
    cre = base.cre.model_copy(update={"enabled": True, "prior_rel_sd": 1.0, "roi_scale": 0.2})
    cfg = base.model_copy(update={
        "cre": cre,
        "calib": base.calib.model_copy(update={"enabled": True, "targeting": "all", "items_per_slice": ITEMS,
                                                "interval_rounds": 10, "k_min": 2}),
    })
    ratios, roi_b, per_scenario, samples = [], [], {}, []
    for sid in SCENARIOS:
        sc = load_scenario(str(_path(sid))).model_copy(update={"calib": CalibSimCfg(
            enabled=True, interval_rounds=10, paired=True, churn=0.02, items_per_slice=ITEMS, cost_usd=0.0)})
        r_s, b_s = [], []
        for s in seeds:
            sched = make_scheduler("curator", sc, s, cfg=cfg)
            engine = sched._cre  # noqa: SLF001
            # PASSIVE: the engine records evidence but never steers allocation, so the
            # constants come from the CRE-free trajectory (no circularity with the very
            # values being frozen).
            original_ingest = engine.ingest

            def passive(*a, _o=original_ingest, _e=engine, **k):
                _o(*a, **k)
                _e.active = False

            engine.ingest = passive
            run_episode(sched, sc, seed=s, method="constants")
            if not any(engine._n[e] > 0 for e in engine.env_ids):  # noqa: SLF001
                continue
            budget = sched._budget_total  # noqa: SLF001
            calibrator = sched._calibrator  # noqa: SLF001
            for e in sched.env_ids:
                b = engine.b_hat(e)
                if b is None or calibrator._x_w[e] <= 0:  # noqa: SLF001
                    continue
                xbar = calibrator._x_sum[e] / calibrator._x_w[e]  # noqa: SLF001
                c = float(sched._engine._unit_cost[e])  # noqa: SLF001
                if xbar > 1e-9 and c > 0:
                    samples.append((b, xbar, c, budget))
                    r_s.append(b * c / xbar)
                b_s.append(b * budget)
        per_scenario[sid] = {"n_arm_obs": len(r_s),
                             "median_ratio": float(np.median(r_s)) if r_s else None,
                             "p95_roi_b": float(np.percentile([v for v in b_s if v > 0], 95)) if any(v > 0 for v in b_s) else None}
        ratios += r_s
        roi_b += b_s
    roi_a = np.asarray([v for v in roi_b if v > 0])
    roi_scale = float(max(np.percentile(roi_a, 95), ROI_FLOOR))
    proxy_scale = float(np.median(ratios))
    rel = []
    for b, xbar, c, budget in samples:
        m0 = proxy_scale * xbar / c
        rel.append((b - m0) / max(abs(m0), 0.05 * roi_scale / budget))
    rel_a = np.asarray(rel)
    mad = float(np.median(np.abs(rel_a - np.median(rel_a))))
    rho = float(np.clip(1.4826 * mad, *RHO_BOUNDS))
    report = {"seeds": [seeds[0], seeds[-1]], "scenarios": list(SCENARIOS), "items_per_slice": ITEMS,
              "n_residuals": len(rel), "proxy_scale": float(f"{proxy_scale:.4g}"),
              "median_relative_residual": float(np.median(rel_a)), "raw_robust_sd": 1.4826 * mad,
              "prior_rel_sd": round(rho, 3), "roi_scale": round(roi_scale, 4), "per_scenario": per_scenario}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cre_constants.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# CRE constants frozen from tuning seeds (D-77)", "",
          f"Tuning seeds {seeds[0]}..{seeds[-1]}, scenarios {', '.join(SCENARIOS)}; free calibration, every slice, "
          f"{ITEMS} paired items. {len(rel)} (seed, arm) residuals.", "",
          f"- `cre.proxy_scale` (s) = **{report['proxy_scale']}** (median of b_hat*c/xbar over {len(ratios)} arm estimates)",
          f"- `cre.prior_rel_sd` (rho) = **{report['prior_rel_sd']}** (raw robust SD {report['raw_robust_sd']:.3f}, clipped to {RHO_BOUNDS}; median relative residual {report['median_relative_residual']:.3f})",
          f"- `cre.roi_scale` (R_max) = **{report['roi_scale']}** (95th percentile of b_hat * B over {len(roi_a)} positive arm estimates)", "",
          "| scenario | arm estimates | median b_hat*c/xbar | p95 b_hat * B |", "|---|---|---|---|"]
    for sid, v in per_scenario.items():
        md.append(f"| {sid} | {v['n_arm_obs']} | {v['median_ratio']} | {v['p95_roi_b']} |")
    (REPORT_DIR / "cre_constants.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
