"""Status-threshold tuning study (Roadmap E.5 "How thresholds are chosen",
Phase 4 exit criterion).

Grid over (p_sat, p_hard, z_up, dwell_min, consecutive_rounds) with hysteresis
fixed; guessing is only on the replayed ground-truth labels (D-39: truth uses
the frozen base.yaml default thresholds), so the tuning maximises macro-F1
subject to flip_rate <= 5/100 rounds:

    python experiments/analysis/tune_status_thresholds.py \
        --scenarios S-A,S-B,S-E,S-G --tuning-seeds 10 --eval-seeds 30

The classifier is constructed directly from the overridden StatusCfg per grid
point; episodes are collected once and reused across grid points. Selected
candidate thresholds are reported as a base.yaml diff; validation runs on the
disjoint evaluation seeds (reported, never tuned).

Writes reports/analysis/tune_status_thresholds.{json,md}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import UTC, datetime

from curator_rl.core.config import StatusCfg  # noqa: E402
from experiments.analysis import common  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "analysis"
GRID = {
    "p_sat": [0.7, 0.85, 0.9],
    "p_hard": [0.05, 0.10, 0.15],
    "z_up": [1.5, 2.0, 3.0],
    "dwell_min": [3, 5, 8],
    "consecutive_rounds": [2, 3, 4],
}
FLIP_MAX = 5.0

# Frozen truth anchor (D-43): the pre-tuning roadmap/default thresholds. The
# grid varies the PREDICTION thresholds only; the truth labels always use
# these constants so a re-run after base.yaml changes stays comparable.
TRUTH_ANCHOR = {
    "n_min": 64, "r_min": 3, "p_sat": 0.85, "p_hard": 0.10, "sr_hard": 0.10,
    "z_up": 2.0, "z_neg": 2.0, "hysteresis_p": 0.05, "hysteresis_sr": 0.05,
    "dwell_min": 5, "consecutive_rounds": 3, "s1_entry_ratio": 0.5,
}


def candidate_cfgs(anchor: StatusCfg) -> list[tuple[dict, StatusCfg]]:
    """All grid points; unspecified keys come from the anchor cfg."""
    keys = list(GRID)
    out = []
    for values in itertools.product(*GRID.values()):
        over = dict(zip(keys, values))
        payload = anchor.model_dump() | over
        out.append((over, StatusCfg(**payload)))
    return out


def evaluate_point(inputs_by_capture: list, truth_by_capture: list, status_cfg: StatusCfg) -> dict:
    """Macro-F1 (truth-anchored) and mean flip rate for one grid point."""
    f1s: list[float] = []
    flips: list[float] = []
    for inputs, truth in zip(inputs_by_capture, truth_by_capture):
        traces = common.classify_with_status(status_cfg, inputs)
        for env_id in sorted(traces):
            f1s.append(common.macro_f1([s.value for s in truth[env_id]], traces[env_id]))
        flips.append(common.mean_flip_rate(traces, warmup=-1))
    return {
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "flip_rate": sum(flips) / max(len(flips), 1),
        "n_envs_scored": len(f1s),
    }


def select(all_scores: list[tuple[dict, dict]]) -> tuple[dict, dict] | None:
    eligible = [point for point in all_scores if point[1]["flip_rate"] <= FLIP_MAX]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[1]["macro_f1"], -item[1]["flip_rate"]))


def render_markdown(report: dict) -> str:
    lines = ["# Status threshold tuning study (Phase 4)", "",
             "Truth anchored to the frozen base.yaml default thresholds (D-39).",
             "Constraint: flip_rate <= 5 per 100 rounds. Objective: macro-F1.", ""]
    lines += render_table(report)
    return "\n".join(lines)


def render_table(report: dict) -> list[str]:
    lines: list[str] = []
    for phase in ("tuning", "evaluation"):
        top5 = report.get(phase, {}).get("top5", [])
        if not top5:
            continue
        lines += ["", f"### {phase}: top 5 grid points", "",
                  "| rank | overrides | macro-F1 | flip rate |", "|---|---|---|---|"]
        for i, row in enumerate(top5, 1):
            lines.append(
                f"| {i} | `{json.dumps(row['overrides'])}` |"
                f" {row['macro_f1']:.4f} | {row['flip_rate']:.2f} |"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Status threshold tuning (Roadmap E.5)")
    parser.add_argument("--scenarios", default="S-A,S-B,S-E,S-G")
    parser.add_argument("--tuning-seeds", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, default=30)
    parser.add_argument("--eval-seed0", type=int, default=100)
    args = parser.parse_args(argv)

    sim_dir = REPO_ROOT / "configs" / "sim"
    wanted = [s.strip().replace("-", "").lower() for s in args.scenarios.split(",")]
    paths = sorted(p for p in sim_dir.glob("scenario_*.yaml") if any(w in p.stem.lower() for w in wanted))
    (signals_cfg, proxy_cfg, calib_cfg, cost_exponent, group_size, _anchor_unused) = common.default_cfg()
    del proxy_cfg, calib_cfg, cost_exponent  # statuses do not need them
    anchor = StatusCfg(**TRUTH_ANCHOR)

    report: dict = {
        "generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "grid": GRID,
        "flip_max": FLIP_MAX,
        "scenarios": [p.stem for p in paths],
        "truth_anchor": anchor.model_dump(),
        "lp_method_in_engine": signals_cfg.lp_method,
    }

    for phase, seeds in (("tuning", list(range(args.tuning_seeds))),
                         ("evaluation",
                          list(range(args.eval_seed0, args.eval_seed0 + args.eval_seeds)))):
        t0 = time.perf_counter()
        captures = common.collect_episodes([str(p) for p in paths], seeds)
        inputs_by_capture = [common.precompute_status_inputs(c, signals_cfg, group_size) for c in captures]
        truth_by_capture = [common.truth_labels(c, anchor) for c in captures]
        all_points: list[tuple[dict, dict]] = []
        for i, (override, point_cfg) in enumerate(candidate_cfgs(anchor)):
            score = evaluate_point(inputs_by_capture, truth_by_capture, point_cfg)
            all_points.append((override, score))
            if i % 50 == 0:
                print(f"  [{phase}] grid {i}/{len(all_points)}...")
        eligible = [p for p in all_points if p[1]["flip_rate"] <= FLIP_MAX]
        picked = select(all_points)
        top5 = [p for p in sorted(eligible, key=lambda x: -x[1]["macro_f1"])[:5]]
        if picked is None:
            picked = max(all_points, key=lambda x: x[1]["macro_f1"])  # relaxed fallback
            relaxed = True
        else:
            relaxed = False
        report[phase] = {
            "seconds": round(time.perf_counter() - t0, 1),
            "n_captures": len(captures),
            "picked": picked[0],
            "score": picked[1],
            "relaxed": relaxed,
            "n_eligible": len(eligible),
            "top5": [{"overrides": o, **s} for (o, s) in top5],
        }
        print(f"[{phase}] picked {picked[0]} f1={picked[1]['macro_f1']:.4f} "
              f"flip={picked[1]['flip_rate']:.2f} (relaxed={relaxed})")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "tune_status_thresholds.json"
    md_path = REPORT_DIR / "tune_status_thresholds.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"report: {json_path} ({md_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
