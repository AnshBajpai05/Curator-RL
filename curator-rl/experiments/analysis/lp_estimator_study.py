"""LP estimator selection study (Roadmap E.3 protocol, Phase 4 exit criterion).

Compares the three candidate learning-progress estimators (LP-A fast-slow,
LP-B weighted slope, LP-C non-overlapping windows) on real simulator episode
streams:

    python experiments/analysis/lp_estimator_study.py \
        --scenarios S-A S-B S-E --tuning-seeds 20 --eval-seeds 50

selection rule (Roadmap E.3): best Spearman rank correlation with the true
skill velocity, subject to a <= 5% "significant LP" false-positive rate on
zero-velocity rounds and lag <= 5 rounds. Fallback: LP-A.

Writes reports/analysis/lp_estimator_study.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

from scipy.stats import spearmanr  # noqa: E402

from experiments.analysis import common  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402

ESTIMATORS = ("lp_a", "lp_b", "lp_c")
REPORT_DIR = REPO_ROOT / "reports" / "analysis"
WARMUP = 10          # rounds skipped in every metric (estimators need history)
VELOCITY_SIGN_EPS = 0.005   # |velocity| above this counts as "moving" for sign accuracy
LAG_MAX = 5
FPR_MAX = 0.05
LAG_MAX_ALLOWED = 5.0


def signals_for_method(base_signals, method: str):
    payload = base_signals.model_dump()
    payload["lp_method"] = method
    return type(base_signals).model_validate(payload)


def pairs_for_capture(capture, vectors_by_round):
    """Aligned (lp, true velocity) samples per env for rounds >= WARMUP."""
    velocities = capture.velocities()
    pairs: dict[str, list[tuple[float, float, float, float]]] = {}
    for t, per_round in enumerate(vectors_by_round):
        for env_id, sv in per_round.items():
            v = velocities[env_id][t]
            pairs.setdefault(env_id, []).append((sv.lp, sv.lp_z, v, t))
    return pairs


def score_estimator(pairs_by_capture: list[dict[str, list[tuple]]]) -> dict:
    """Pooled metrics over every capture, env and round (consistent estimator)."""
    lp_values: list[float] = []
    v_values: list[float] = []
    z_values: list[float] = []
    v_for_z: list[float] = []
    zero_v_flags: list[bool] = []

    for pairs in pairs_by_capture:
        for env, rows in pairs.items():
            for lp, z, v, _t in rows[WARMUP:]:
                lp_values.append(lp)
                v_values.append(v)
                z_values.append(z)
                v_for_z.append(v)
                zero_v_flags.append(env.endswith("noisy") or abs(v) < common.VELOCITY_S3_ANCHOR)
    # rank correlation with the true velocity
    rho, p_val = spearmanr(lp_values, v_values) if len(set(lp_values)) > 1 else (float("nan"), 1.0)

    # sign accuracy where the true velocity is meaningfully non-zero
    moving = [abs(v) > VELOCITY_SIGN_EPS for v in v_values]
    if any(moving):
        sign_ok = sum(
            1 for lp, v, m in zip(lp_values, v_values, moving) if m and lp * v > 0
        )
        moving_n = sum(moving)
        sign_accuracy = sign_ok / moving_n
    else:
        moving_n = 0
        sign_accuracy = float("nan")

    # false-positive rate on zero-velocity rounds
    zero_idx = [i for i, (v, noisy) in enumerate(zip(v_for_z, zero_v_flags)) if noisy]
    fpr = (
        sum(1 for i in zero_idx if abs(z_values[i]) >= common.Z_FPR_THRESHOLD) / len(zero_idx)
        if zero_idx else float("nan")
    )

    # lag: argmax cross-correlation of lp(t + lag) with v(t) for lag 0..LAG_MAX
    lag_scores = []
    for lag in range(LAG_MAX + 1):
        if len(lp_values) <= lag:
            continue
        r = spearmanr(lp_values[lag:], v_values[: len(v_values) - lag])
        lag_scores.append(getattr(r, "statistic", r[0]))
    best = max(range(len(lag_scores)), key=lambda i: lag_scores[i]) if lag_scores else 0
    return {
        "spearman_rho": _clean(rho),
        "spearman_p": _clean(p_val),
        "sign_accuracy": _clean(sign_accuracy),
        "sign_n_pairs": moving_n,
        "fpr": _clean(fpr),
        "fpr_n": len(zero_idx),
        "best_lag": best,
        "lag_correlations": [_clean(c) for c in lag_scores],
        "n_samples": len(lp_values),
    }


def _clean(x: float) -> float | None:
    return x if x == x else None  # NaN -> None


def pick_best(scores: dict[str, dict]) -> str | None:
    """Selection rule: best rho subject to FPR <= 5% and lag <= 5; fallback LP-A."""
    eligible = []
    for method, s in scores.items():
        if s["fpr"] is None or s["spearman_rho"] is None:
            continue
        if s["fpr"] <= FPR_MAX and s["best_lag"] <= LAG_MAX_ALLOWED:
            eligible.append((s["spearman_rho"], -s["fpr"], method))
    if not eligible:
        return None
    eligible.sort(reverse=True)
    return eligible[0][2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LP estimator selection study (Roadmap E.3)")
    parser.add_argument("--scenarios", default="S-A,S-B,S-E",
                        help="scenario ids matched against configs/sim/scenario_*")
    parser.add_argument("--tuning-seeds", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, default=30)
    parser.add_argument("--eval-seed0", type=int, default=100)
    args = parser.parse_args(argv)

    sim_dir = REPO_ROOT / "configs" / "sim"
    wanted = [s.strip().replace("-", "").lower() for s in args.scenarios.split(",")]
    paths = sorted(p for p in sim_dir.glob("scenario_*.yaml") if any(w in p.stem.lower() for w in wanted))
    if not paths:
        print("no matching scenarios")
        return 1

    signals_cfg, proxy_cfg, calib_cfg, cost_exponent, group_size, _status = common.default_cfg()

    report: dict = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                    "estimators": list(ESTIMATORS), "scenarios": [p.stem for p in paths],
                    "warmup": WARMUP}

    for phase, seeds in (("tuning", list(range(args.tuning_seeds))),
                         ("evaluation", list(range(args.eval_seed0, args.eval_seed0 + args.eval_seeds)))):
        t0 = time.perf_counter()
        captures = common.collect_episodes([str(p) for p in paths], seeds)
        scores: dict[str, dict] = {}
        for method in ESTIMATORS:
            signals = signals_for_method(signals_cfg, method)
            pair_sets = []
            for capture in captures:
                vectors_by_round = common.run_engine(
                    capture, signals, proxy_cfg, calib_cfg, cost_exponent, group_size
                )
                pair_sets.append(pairs_for_capture(capture, vectors_by_round))
            scores[method] = score_estimator(pair_sets)
        picked = pick_best(scores) or "lp_a"
        report[phase] = {"seconds": round(time.perf_counter() - t0, 1),
                         "n_captures": len(captures), "scores": scores, "picked": picked}
        print(f"[{phase}] picked: {picked}")
        for method in ESTIMATORS:
            s = scores[method]
            print(f"    {method:<5} rho={_fmt(s['spearman_rho'])}  sign={_fmt(s['sign_accuracy'])}"
                  f"  fpr={_fmt(s['fpr'])}  lag={s['best_lag']}  n={s['n_samples']}")

    if report["tuning"]["picked"] != report["evaluation"]["picked"]:
        print("WARNING: tuning and evaluation phases picked different estimators")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "lp_estimator_study.json"
    md_path = REPORT_DIR / "lp_estimator_study.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"report: {json_path} ({md_path})")
    return 0


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def render_markdown(report: dict) -> str:
    lines = ["# LP estimator selection study (Phase 4)", "",
             "Selection rule: best Spearman rho with true skill velocity subject to",
             "FPR <= 5% and lag <= 5 (fallback LP-A). NVS = none; P = none when constant.", ""]
    for phase in ("tuning", "evaluation"):
        picked = report.get(phase, {}).get("picked", "n/a")
        lines += [f"## {phase} phase (picked: **{picked}**)", "",
                  "| estimator | rho | sign acc | FPR | best lag | n |", "|---|---|---|---|---|---|"]
        for method, s in report.get(phase, {}).get("scores", {}).items():
            lines.append(
                f"| {method} | {_fmt(s['spearman_rho'])} | {_fmt(s['sign_accuracy'])} |"
                f" {_fmt(s['fpr'])} ({s['fpr_n']}) | {s['best_lag']} | {s['n_samples']} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
