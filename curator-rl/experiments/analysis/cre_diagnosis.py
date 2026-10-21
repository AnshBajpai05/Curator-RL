"""Why did the CRE barely change the allocation? (D-79; tuning seeds only)

After the pre-registered evaluation failed Gate 2'' (i) on S-I with the CRE statistically
indistinguishable from the existing targeted Curator, this study measures, on TUNING seeds,
how much leverage the reward channel has on the mixture once the engine is active:
the spread across arms of the discounted mean reward (mu_hat) versus the spread of the
count-based exploration bonus, the agreement between proxy-based and CRE-based mu_hat,
and the distance between the two methods' weights.

    python experiments/analysis/cre_diagnosis.py --seeds 20

Writes reports/analysis/cre_diagnosis.{json,md}.
"""

from __future__ import annotations

import argparse
import json

import numpy as np  # noqa: E402

from curator_rl.simulator.harness import run_episode  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.cre_gate import method_cfg, scenario_variant  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "analysis"
FIRST_ACTIVE_ROUND = 31      # K=10, k_min=2: the engine activates after the third calibration


def _trace(method, sc, seed):
    sch = make_scheduler("curator", sc, seed, cfg=method_cfg(method))
    dec = []
    original = sch.select_mixture

    def wrapped(obs, _o=original, _s=sch, _d=dec):
        w = _o(obs)
        _d.append(_s.last_decision)
        return w

    sch.select_mixture = wrapped
    return run_episode(sch, sc, seed=seed, method=method), dec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CRE leverage diagnosis (D-79)")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)
    sc = scenario_variant("S-I", 10)
    ids = [e.env_id for e in sc.envs]
    rows = []
    for seed in range(args.seeds):                       # tuning seeds only
        rt, dt = _trace("curator_targeted", sc, seed)
        rc, dc = _trace("cre", sc, seed)
        for i in range(FIRST_ACTIVE_ROUND, min(len(dt), len(dc))):
            if dt[i] is None or dc[i] is None:
                continue
            mu_c = np.array([dc[i].mu_hat[e] for e in ids])
            mu_t = np.array([dt[i].mu_hat[e] for e in ids])
            bonus = np.array([dc[i].bonus[e] for e in ids])
            wt = np.array([rt.round_logs[i]["weights"][e] for e in ids])
            wc = np.array([rc.round_logs[i]["weights"][e] for e in ids])
            corr = float(np.corrcoef(mu_t, mu_c)[0, 1]) if mu_t.std() > 0 and mu_c.std() > 0 else float("nan")
            rows.append((mu_c.std(), bonus.std(), float(np.abs(wt - wc).sum() / 2), corr, wt.max(), wc.max()))
    a = np.array(rows)
    rep = {"seeds": [0, args.seeds - 1], "scenario": "S-I", "rounds_analysed": len(a),
           "std_mu_hat_across_arms": float(np.nanmean(a[:, 0])),
           "std_bonus_across_arms": float(np.nanmean(a[:, 1])),
           "mean_total_variation_between_weights": float(np.nanmean(a[:, 2])),
           "mean_corr_mu_hat_proxy_vs_cre": float(np.nanmean(a[:, 3])),
           "mean_max_weight_targeted": float(np.nanmean(a[:, 4])),
           "mean_max_weight_cre": float(np.nanmean(a[:, 5]))}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cre_diagnosis.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = ["# CRE leverage diagnosis (D-79)", "",
          f"Tuning seeds 0..{args.seeds - 1}, S-I, rounds after the engine activates ({rep['rounds_analysed']} rounds).", "",
          "| quantity | value |", "|---|---|",
          f"| std across arms of mu_hat (CRE rewards) | {rep['std_mu_hat_across_arms']:.3f} |",
          f"| std across arms of the exploration bonus | {rep['std_bonus_across_arms']:.3f} |",
          f"| correlation of mu_hat across arms, proxy vs CRE | {rep['mean_corr_mu_hat_proxy_vs_cre']:.2f} |",
          f"| mean total-variation distance between weights | {rep['mean_total_variation_between_weights']:.3f} |",
          f"| mean max-weight, targeted / CRE | {rep['mean_max_weight_targeted']:.2f} / {rep['mean_max_weight_cre']:.2f} |", ""]
    (REPORT_DIR / "cre_diagnosis.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
