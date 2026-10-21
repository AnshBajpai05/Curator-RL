"""ROI vs leave-one-out in the simulator (H5 machinery check at the GPU shape; TUNING seeds only).

For each seed: Curator (Kaggle preset `gpu_tuned`) runs on the full portfolio; its calibration logs give
the in-run ROI leaderboard (roi/leaderboard.py). Then, per arm, Curator is retrained with that arm left
out at the same budget (evaluation/loo.py: the benchmark is unchanged). H5: Kendall tau between mean
ROI and mean LOO contribution >= 0.4, with the two junk arms (noisy: random reward; countdown: too
hard) in the bottom two of both. Also reports the noiseless truth the GPU cannot see: each arm's
LOO contribution IS the ground truth for "what the arm is worth here"; ROI is the cheap estimate.

    python experiments/analysis/roi_loo_sim.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from curator_rl.core.config import load_config, rescale_for_batch_shape  # noqa: E402
from curator_rl.evaluation.loo import WithoutArm, h5_verdict, loo_contributions  # noqa: E402
from curator_rl.roi.leaderboard import roi_table  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import domain_map, load_scenario  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

SCENARIOS = {"S-I-R1-GPU2": "scenario_si_r1_gpu2.yaml", "S-J-R-GPU2": "scenario_sj_r_gpu2.yaml"}
JUNK = ("noisy", "countdown")
PRESET = ["scheduler.gamma=1.0", "scheduler.exploration_coef=0.25", "scheduler.tau=1.0",
          "signals.reward_audit.enabled=true", "scheduler.s5_cap=floor", "calib.mismatch_windows=3"]


def _cfg(sc):
    return rescale_for_batch_shape(load_config(REPO_ROOT / "configs" / "base.yaml", PRESET), sc.prompts_per_round)


def run_seed(task) -> dict:
    sid, seed = task
    sc = load_scenario(REPO_ROOT / "configs" / "sim" / SCENARIOS[sid])
    cfg = _cfg(sc)
    ids = [e.env_id for e in sc.envs]
    full = run_episode(make_scheduler("curator", sc, seed, cfg=cfg), sc, seed=seed, method="curator")
    cost = {e.env_id: e.cost_usd_per_prompt for e in sc.envs}          # usd_per_gpu_hour = 1: GPU-hours
    weights = {e.env_id: e.bench_weight for e in sc.envs}
    roi = {r.env_id: r.roi for r in roi_table(full.calib_logs, ids, cost, weights, domain_of=domain_map(sc))}
    roi_full = {r.env_id: r.roi for r in roi_table(full.calib_logs, ids, cost, weights, structure="full")}
    uni = run_episode(make_scheduler("uniform", sc, seed), sc, seed=seed).final_score_expected
    without, without_u = {}, {}
    for arm in ids:
        reduced = sc.model_copy(update={"envs": [e for e in sc.envs if e.env_id != arm]})
        inner = make_scheduler("curator", reduced, seed, cfg=cfg)
        without[arm] = run_episode(WithoutArm(inner, arm, ids), sc, seed=seed).final_score_expected
        without_u[arm] = run_episode(WithoutArm(make_scheduler("uniform", reduced, seed), arm, ids), sc,
                                     seed=seed).final_score_expected
    return {"sid": sid, "seed": seed, "full": full.final_score_expected, "without": without, "roi": roi,
            "roi_full": roi_full, "uniform": uni, "without_uniform": without_u}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run_seed, [(sid, s) for sid in SCENARIOS for s in range(args.seeds)]))
    lines = ["# ROI vs leave-one-out in the simulator (GPU shape, realism v2; tuning seeds)", "",
             f"Curator at the Kaggle preset `gpu_tuned`; seeds 0..{args.seeds - 1}. ROI = in-run estimate from "
             "Curator's own calibration (benchmark points per GPU-hour); LOO = noiseless score lost when the arm "
             "is left out at the same budget (95% t-interval over seeds); the truth is LOO under Uniform (the arm's "
             "value in an equal mixture); LOO under Curator also reflects how well Curator uses the arm. H5: tau >= 0.4 and the junk arms "
             f"{list(JUNK)} in the bottom two of both rankings.", ""]
    report = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": [0, args.seeds - 1],
              "scenarios": {}}
    for sid in SCENARIOS:
        rs = [r for r in rows if r["sid"] == sid]
        arms = list(rs[0]["roi"])
        loo_u = loo_contributions([r["uniform"] for r in rs], {a: [r["without_uniform"][a] for r in rs] for a in arms})
        loo_c = loo_contributions([r["full"] for r in rs], {a: [r["without"][a] for r in rs] for a in arms})
        roi_m = {a: statistics.fmean(r["roi"][a] for r in rs) for a in arms}
        roi_f = {a: statistics.fmean(r["roi_full"][a] for r in rs) for a in arms}
        truth = {a: loo_u[a][0] for a in arms}          # ground truth: the arm's value in an equal mixture
        verdicts = {"members": h5_verdict(roi_m, truth, JUNK), "full": h5_verdict(roi_f, truth, JUNK)}
        single = {k: [h5_verdict(r[key], {a: r["uniform"] - r["without_uniform"][a] for a in arms}, JUNK)
                      for r in rs] for k, key in (("members", "roi"), ("full", "roi_full"))}
        report["scenarios"][sid] = {
            "roi_members": roi_m, "roi_full": roi_f, "loo_uniform": loo_u, "loo_curator": loo_c,
            "h5_pooled": verdicts,
            "single_run": {k: {"median_tau": statistics.median(v["tau"] for v in vs),
                               "pass_rate": sum(v["pass"] for v in vs) / len(vs)} for k, vs in single.items()}}
        lines += [f"## {sid}", "", "| arm | ROI members (points/GPU-h) | ROI full matrix | LOO under Uniform (truth) "
                  "| LOO under Curator |", "|---|---|---|---|---|"]
        for a in sorted(arms, key=lambda a: -truth[a]):
            mu, lou, hiu = loo_u[a]
            mc, loc, hic = loo_c[a]
            lines.append(f"| {a} | {100 * roi_m[a]:+.2f} | {100 * roi_f[a]:+.2f} | {mu:+.4f} [{lou:+.4f}, {hiu:+.4f}] | "
                         f"{mc:+.4f} [{loc:+.4f}, {hic:+.4f}] |")
        lines.append("")
        for k, v in verdicts.items():
            sr = report["scenarios"][sid]["single_run"][k]
            lines.append(f"- ROI `{k}` vs truth, pooled over seeds: tau {v['tau']:+.2f}, junk in the bottom two of both "
                         f"{v['junk_bottom']} -> H5 {'PASS' if v['pass'] else 'FAIL'}; one seed at a time: median tau "
                         f"{sr['median_tau']:+.2f}, H5 passes in {100 * sr['pass_rate']:.0f}% of seeds")
        lines.append("")
    out = REPO_ROOT / "reports" / "analysis"
    (out / "roi_loo_sim.json").write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    md = "\n".join(lines) + "\n"
    (out / "roi_loo_sim.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
