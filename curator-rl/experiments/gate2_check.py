"""Gate 2 verification (Roadmap Part Q, v2 Gate 2 = v1 Phases 5-6).

Runs the frozen tuned Curator against the baselines on EVALUATION seeds
(100-149, disjoint from the tuning seeds) and checks the pre-registered
primary criteria:

(a) S-A: paired Curator - Uniform final-score difference, 95% bootstrap
    interval excluding 0 (robust improvement);
(b) S-B: the same paired interval's lower bound > -delta_gap (1 benchmark
    point = 0.01 score units; no underperformance);
(c) S-B: junk (too-hard arm) compute share <= 1.5 x (epsilon/N) after round 30
    in >= 90% of seeds (D-50: "junk" = the zero-signal arm; the noisy arm is
    the H3 reward-hacking probe that Phase 9 calibration targets);
(d) status macro-F1 >= 0.85 and pooled flip <= 5/100 on Curator's own engine
    statuses, truth anchored per Phase 4 (frozen constants, D-43);
(e) S-C: compute-to-target at least 15% better than Uniform (paired interval
    excluding 0) using the uncharged score-vs-cost curve (D-46);
dev target: oracle-gap closure >= 75% in S-A (60-75% needs documented analysis).

    python experiments/gate2_check.py --seeds 50

Writes reports/gate2/gate2.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np  # noqa: E402

from curator_rl.core.config import StatusCfg, load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.oracle import StaticOracle  # noqa: E402
from curator_rl.simulator.scenarios import ReportEvalCfg, build_world, load_scenario  # noqa: E402
from experiments.analysis import common  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.run_sim import FROZEN_D68_D94  # noqa: E402  (audit E-04: pre-registered config)

REPORT_DIR = REPO_ROOT / "reports" / "gate2"
DELTA_GAP = 0.01         # pre-registered margin: 1 benchmark point = 0.01 in score units (D-69)
FLIP_MAX = 5.0
F1_MIN = 0.85
COMPUTE_TO_TARGET_MIN = 0.15
JUNK_FACTOR = 1.5
EVAL_SEED0 = 100
BOOTSTRAP = 10000
TRUTH_ANCHOR = {  # frozen pre-tuning status defaults (D-43)
    "n_min": 64, "r_min": 3, "p_sat": 0.85, "p_hard": 0.10, "sr_hard": 0.10,
    "z_up": 2.0, "z_neg": 2.0, "hysteresis_p": 0.05, "hysteresis_sr": 0.05,
    "dwell_min": 5, "consecutive_rounds": 3, "s1_entry_ratio": 0.5,
}
METHODS = ("uniform", "static", "lp", "ucb", "curator")


def paired_bootstrap(diffs: list[float], rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval of paired differences."""
    arr = np.asarray(diffs, dtype=float)
    mean = float(arr.mean())
    boots = rng.choice(arr, size=(BOOTSTRAP, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def time_to_target(points: list[tuple[float, float]], target: float, budget: float) -> float:
    """Cost at which the score-vs-cost curve first reaches `target`, linearly
    interpolated between report points (B if never)."""
    if not points:
        return budget
    if points[0][1] >= target:
        return points[0][0]
    for (c0, s0), (c1, s1) in zip(points, points[1:]):
        if s1 >= target and s1 > s0:
            return c0 + (c1 - c0) * (target - s0) / (s1 - s0)
        if s1 >= target:
            return c1
    return budget


def junk_share(result, env_id: str, from_frac: float = 0.6) -> float:
    """Share over the post-warm-up tail (last `from_frac` of the episode);
    criterion (c) says 'after round 30', but short episodes (S-B ~ 24 rounds)
    would leave an empty window."""
    cut = int(result.rounds * from_frac)
    late = [log for log in result.round_logs if log["round"] > cut]
    if not late:
        return float("nan")
    return sum(log["weights"][env_id] for log in late) / len(late)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate 2 verification (Roadmap Part Q)")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed0", type=int, default=EVAL_SEED0)
    parser.add_argument("--skip-oracle", action="store_true", help="skip the static oracle dev target")
    args = parser.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    cfg = load_config(FROZEN_D68_D94)

    sim_dir = REPO_ROOT / "configs" / "sim"
    paths = {sid: sim_dir / f"scenario_{sid.replace('-', '').lower()}.yaml" for sid in ("S-A", "S-B", "S-C")}

    results: dict[str, dict[str, list]] = {}
    t0 = time.perf_counter()
    for sid, path in paths.items():
        scenario = load_scenario(str(path))
        if sid == "S-C":
            scenario = scenario.model_copy(
                update={"report_eval": ReportEvalCfg(interval_rounds=2)}
            )
        per_method = {}
        for method in METHODS:
            runs = []
            for seed in seeds:
                scheduler = _make(method, scenario, seed, cfg)
                runs.append(run_episode(scheduler, scenario, seed=seed, method=method))
            per_method[method] = runs
            print(f"  {sid} {method}: mean={sum(r.final_score for r in runs) / len(runs):.4f} "
                  f"({time.perf_counter() - t0:.0f}s)")
        results[sid] = per_method

    report: dict = {
        "generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "seeds": seeds,
        "delta_gap": DELTA_GAP,
        "config": {
            "gamma": cfg.scheduler.gamma, "exploration_coef": cfg.scheduler.exploration_coef,
            "tau": cfg.scheduler.tau, "epsilon": cfg.scheduler.epsilon,
        },
        "means": {
            sid: {m: sum(r.final_score for r in runs) / len(runs) for m, runs in per.items()}
            for sid, per in results.items()
        },
    }
    criteria: dict[str, dict] = {}
    rng = np.random.default_rng(2026)

    # (a) S-A robust improvement
    diffs = [c.final_score - u.final_score
             for c, u in zip(results["S-A"]["curator"], results["S-A"]["uniform"])]
    mean, lo, hi = paired_bootstrap(diffs, rng)
    criteria["a_sa_improvement"] = {
        "mean_diff": mean, "ci95": [lo, hi], "pass": bool(lo > 0.0),
    }

    # (b) S-B no underperformance
    diffs_b = [c.final_score - u.final_score
               for c, u in zip(results["S-B"]["curator"], results["S-B"]["uniform"])]
    mean_b, lo_b, hi_b = paired_bootstrap(diffs_b, rng)
    criteria["b_sb_no_underperformance"] = {
        "mean_diff": mean_b, "ci95": [lo_b, hi_b], "delta_gap": DELTA_GAP,
        "pass": bool(lo_b > -DELTA_GAP),
    }

    # (c) junk (too-hard) share in S-B
    scenario_b = load_scenario(str(paths["S-B"]))
    junk_env = next(e.env_id for e in scenario_b.envs if (e.difficulty or 0) >= 1.5)
    floor = cfg.scheduler.epsilon / len(scenario_b.envs)
    shares = [junk_share(r, junk_env) for r in results["S-B"]["curator"]]
    ok = sum(1 for s in shares if s <= JUNK_FACTOR * floor)
    criteria["c_junk_share"] = {
        "junk_env": junk_env, "floor": floor, "threshold": JUNK_FACTOR * floor,
        "mean_share": sum(shares) / len(shares), "seeds_ok": ok, "n_seeds": len(shares),
        "pass": bool(ok >= 0.9 * len(shares)),
    }

    # (d) status quality on the Curator engine stream (S-A)
    f1s: list[float] = []
    flips: list[float] = []
    anchor = StatusCfg(**TRUTH_ANCHOR)
    for seed in seeds[:20]:
        scenario_a = load_scenario(str(paths["S-A"]))
        capture = _capture(scenario_a, seed, cfg)
        inputs = common.precompute_status_inputs(capture, cfg.signals, cfg.group_size)
        traces = common.classify_with_status(cfg.signals.status, inputs)
        truth = common.truth_labels(capture, anchor)
        for env_id in capture.env_ids:
            f1s.append(common.macro_f1([s.value for s in truth[env_id]], traces[env_id]))
        flips.append(common.mean_flip_rate(traces, warmup=-1))
    criteria["d_status_quality"] = {
        "macro_f1": sum(f1s) / len(f1s), "pooled_flip": sum(flips) / len(flips),
        "pass": bool(sum(f1s) / len(f1s) >= F1_MIN and sum(flips) / len(flips) <= FLIP_MAX),
    }

    # (e) S-C compute-to-target
    target = float(np.median([r.final_score for r in results["S-C"]["uniform"]]))
    tt_cur = [time_to_target([(p["spend_usd"], p["score"]) for p in r.report_logs] or [(r.total_cost_usd, r.final_score)], target, r.budget_usd)
              for r in results["S-C"]["curator"]]
    tt_uni = [time_to_target([(p["spend_usd"], p["score"]) for p in r.report_logs] or [(r.total_cost_usd, r.final_score)], target, r.budget_usd)
              for r in results["S-C"]["uniform"]]
    impr = [(u - c) / u for c, u in zip(tt_cur, tt_uni)]
    mean_e, lo_e, hi_e = paired_bootstrap(impr, rng)
    criteria["e_compute_to_target"] = {
        "target": target, "mean_improvement": mean_e, "ci95": [lo_e, hi_e],
        "threshold": COMPUTE_TO_TARGET_MIN,
        "pass": bool(mean_e >= COMPUTE_TO_TARGET_MIN and lo_e > 0.0),
    }

    # dev target: oracle-gap closure in S-A (static oracle reference)
    if not args.skip_oracle:
        scenario_a = load_scenario(str(paths["S-A"]))
        oracle_runs = []
        for seed in seeds[:20]:
            world = build_world(scenario_a, seed)
            oracle = StaticOracle(world, simplex_step=scenario_a.oracle.simplex_step,
                                  random_search_draws=scenario_a.oracle.random_search_draws,
                                  seed=seed)
            oracle_runs.append(run_episode(oracle, scenario_a, seed=seed, method="static_oracle").final_score)
        # same seeds for all three means (audit E-22: a 20-seed oracle against 50-seed methods)
        n_or = len(oracle_runs)
        uni_mean = sum(r.final_score for r in results["S-A"]["uniform"][:n_or]) / n_or
        cur_mean = sum(r.final_score for r in results["S-A"]["curator"][:n_or]) / n_or
        oracle_mean = sum(oracle_runs) / n_or
        gap = (cur_mean - uni_mean) / (oracle_mean - uni_mean)
        criteria["dev_oracle_gap_closure"] = {
            "oracle": oracle_mean, "uniform": uni_mean, "curator": cur_mean,
            "gap_closure": gap, "dev_target": 0.75,
            "note": "dev target, not pass/fail (60-75% needs documented analysis)",
        }

    report["criteria"] = criteria
    report["all_pass"] = all(
        v.get("pass", True) for k, v in criteria.items() if k.startswith(("a_", "b_", "c_", "d_", "e_"))
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "gate2.json"
    md_path = REPORT_DIR / "gate2.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print("\nGATE 2:", "PASS" if report["all_pass"] else "FAIL")
    for name, c in criteria.items():
        print(f"  {name}: pass={c.get('pass', 'dev')} {json.dumps({k: v for k, v in c.items() if k != 'pass'}, default=str)[:160]}")
    print(f"report: {json_path}")
    return 0 if report["all_pass"] else 1


def _make(method: str, scenario, seed: int, cfg):
    from experiments.run_sim import make_scheduler

    return make_scheduler(method, scenario, seed, cfg=cfg)


def _capture(scenario, seed: int, cfg):
    """EpisodeCapture of the Curator episode (for the truth-label machinery)."""
    from experiments.analysis.common import RecordingScheduler

    inner = _make("curator", scenario, seed, cfg)
    rec = RecordingScheduler(inner, tag="curator", env_ids=tuple(e.env_id for e in scenario.envs))
    result = run_episode(rec, scenario, seed=seed, method="curator")
    from experiments.analysis.common import EpisodeCapture

    return EpisodeCapture(
        scenario_id=scenario.scenario_id, seed=seed, scheduler="curator",
        env_ids=rec.env_ids,
        observations=tuple(rec.observations),
        true_skills=tuple(log["true_skills"] for log in result.round_logs),
        true_pass_rates=tuple(log["true_pass_rates"] for log in result.round_logs),
    )


def render_md(report: dict) -> str:
    lines = ["# Gate 2 verification report", "",
             f"Generated {report['generated_utc']}; seeds {report['seeds'][0]}..{report['seeds'][-1]};",
             f"config: {json.dumps(report['config'])}", "",
             "## Final scores (mean over seeds)", "",
             "| scenario | " + " | ".join(METHODS) + " |",
             "|---|" + "---|" * len(METHODS)]
    for sid, per in report["means"].items():
        lines.append(f"| {sid} | " + " | ".join(f"{per[m]:.4f}" for m in METHODS) + " |")
    lines += ["", "## Criteria", ""]
    for name, c in report["criteria"].items():
        status = c.get("pass", "dev-target")
        lines.append(f"- **{name}**: pass={status} — `{json.dumps({k: v for k, v in c.items() if k != 'pass'}, default=str)}`")
    lines += ["", f"**Overall: {'PASS' if report['all_pass'] else 'FAIL'}**", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
