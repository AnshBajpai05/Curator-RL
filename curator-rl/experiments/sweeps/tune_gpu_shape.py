"""Tuning at the GPU shape (audit E-05): calibration design and fair re-tuning, TUNING seeds only.

Why: on the GPU-shape refits (R = 2 x P = 16, ~60 rounds, shared noisy slice) the post-audit Curator
stays near Uniform and never flags the noisy arm (reports/analysis/audit_impact.md). Its exploration
(kappa, tau) and calibration design (K, items) were tuned for the simulator's 80-prompt rounds and
~5,600-prompt runs. Before any GPU comparison every method is re-tuned at the shape it will run at,
with the same budget and rule as D-55/D-68 (8 configurations each; best within one SE, then the least
concentrated mixture), on S-I-R1-GPU and S-I-R2-GPU; S-J-R-GPU stays held out. Part A explores the
calibration design for detection (exploratory). Nothing here is adopted: the outputs feed a
pre-registration on fresh seeds (>= 700).

    python experiments/sweeps/tune_gpu_shape.py --seeds 40
    python experiments/sweeps/tune_gpu_shape.py --seeds 40 --suite gpu2   # realism v2 + reward audit

Suite `gpu2` re-tunes every method on the realism-v2 scenarios (configs/sim/*_gpu2.yaml), with Curator
carrying the components the component study kept (reports/analysis/component_study_gpu2.md): the
reward audit with S5 capped at the floor, and calibration flags after 3 consecutive mismatch windows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config, rescale_for_batch_shape  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import CalibSimCfg, load_scenario  # noqa: E402
from experiments.analysis.audit_impact import NOISY, Recorder  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

SUITES = {
    "gpu": {"tune": {"S-I-R1-GPU": "scenario_si_r1_gpu.yaml", "S-I-R2-GPU": "scenario_si_r2_gpu.yaml"},
            "curator_extra": {}, "out": "tune_gpu_shape", "calibration_designs": True},
    "gpu2": {"tune": {"S-I-R1-GPU2": "scenario_si_r1_gpu2.yaml", "S-I-R2-GPU2": "scenario_si_r2_gpu2.yaml"},
             "curator_extra": {"signals.reward_audit.enabled": True, "scheduler.s5_cap": "floor",
                               "calib.mismatch_windows": 3},
             "out": "tune_gpu2", "calibration_designs": False},
}
TUNE = SUITES["gpu"]["tune"]
SKIP = 6
GRIDS = {
    "curator": [{"gamma": g, "exploration_coef": k, "tau": t}
                for g, k, t in itertools.product((0.95, 1.0), (0.1, 0.25), (0.3, 1.0))],
    "ucb": [{"exploration_coef": k, "tau": t} for k, t in itertools.product((0.1, 0.25, 0.5, 1.0), (0.3, 1.0))],
    "lp": [{"tau": t} for t in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)],
    "sec": [{"alpha": a, "tau": t} for a, t in itertools.product((0.1, 0.3, 0.6, 1.0), (0.1, 0.4))],
    "dump": [{"exploration_coef": c, "tau": t, "window_rounds": w}
             for c, t, w in itertools.product((0.3, 2.0), (0.1, 0.4), (0, 10))],
}
CALIB_DESIGNS = [{"K": k, "items": n, "k_min": m}
                 for k, n, m in ((10, 100, 2), (5, 100, 2), (5, 200, 2), (3, 100, 2), (3, 200, 1), (2, 200, 1))]


def _base(extra: dict | None = None):
    # GPU shape (32 prompts per round): status evidence threshold rescaled (audit E-28)
    ov = [f"{k}={v}" for k, v in (extra or {}).items()]
    return rescale_for_batch_shape(load_config(REPO_ROOT / "configs" / "base.yaml", ov), 32)


def _variant(method: str, point: dict, suite: str = "gpu"):
    base = _base(SUITES[suite]["curator_extra"] if method == "curator" else None)
    if method == "curator":
        return base.model_copy(update={"scheduler": base.scheduler.model_copy(update=point)})
    sec = getattr(base.baselines, method).model_copy(update=point)
    return base.model_copy(update={"baselines": base.baselines.model_copy(update={method: sec})})


def _episode(method, cfg, sc, seed):
    rec = Recorder(make_scheduler(method, sc, seed, cfg=cfg))
    return rec, run_episode(rec, sc, seed=seed, method=method)


def tune_point(task) -> dict:
    method, idx, point, seeds, uniform, suite = task
    cfg = _variant(method, point, suite)
    per_seed, ent = [], []
    for i, s in enumerate(seeds):
        rel = []
        for sid, f in SUITES[suite]["tune"].items():
            sc = load_scenario(REPO_ROOT / "configs" / "sim" / f)
            _, r = _episode(method, cfg, sc, s)
            rel.append((r.final_score_expected - uniform[sid][i]) / uniform[sid][i])
            ent.append(r.concentration(skip_rounds=SKIP)["norm_entropy"])
        per_seed.append(float(np.mean(rel)))
    return {"method": method, "index": idx, "point": point, "objective": statistics.fmean(per_seed),
            "se": statistics.stdev(per_seed) / len(per_seed) ** 0.5, "entropy": float(np.nanmean(ent))}


def calib_point(task) -> dict:
    idx, design, seeds = task
    base = _base()
    cfg = base.model_copy(update={"calib": base.calib.model_copy(
        update={"interval_rounds": design["K"], "items_per_slice": design["items"], "k_min": design["k_min"]})})
    out = {"index": idx, "design": design, "detected": [], "false": [], "cost": [], "score": []}
    for _sid, f in TUNE.items():
        sc = load_scenario(REPO_ROOT / "configs" / "sim" / f)
        sc = sc.model_copy(update={"calib": CalibSimCfg(**{**sc.calib.model_dump(), "interval_rounds": design["K"],
                                                           "items_per_slice": design["items"]})})
        ids = [e.env_id for e in sc.envs]
        for s in seeds:
            rec, r = _episode("curator", cfg, sc, s)
            out["detected"].append(float(NOISY in rec.ever_s5))
            out["false"].append(sum(e in rec.ever_s5 for e in ids if e != NOISY) / (len(ids) - 1))
            out["cost"].append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
            out["score"].append(r.final_score_expected)
    return {k: (statistics.fmean(v) if isinstance(v, list) else v) for k, v in out.items()}


def select(points):
    best = max(points, key=lambda p: p["objective"])
    cands = [p for p in points if p["objective"] >= best["objective"] - best["se"]]
    return max(cands, key=lambda p: (p["entropy"], p["objective"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--suite", choices=sorted(SUITES), default="gpu")
    args = ap.parse_args(argv)
    suite = SUITES[args.suite]
    seeds = list(range(args.seeds))
    t0 = time.perf_counter()
    uniform = {sid: [run_episode(make_scheduler("uniform", load_scenario(REPO_ROOT / "configs" / "sim" / f), s),
                                 load_scenario(REPO_ROOT / "configs" / "sim" / f), seed=s).final_score_expected
                     for s in seeds] for sid, f in suite["tune"].items()}
    with ProcessPoolExecutor() as pool:
        calib = (list(pool.map(calib_point, [(i, d, seeds) for i, d in enumerate(CALIB_DESIGNS)]))
                 if suite["calibration_designs"] else [])
        tuned = list(pool.map(tune_point, [(m, i, p, seeds, uniform, args.suite) for m, grid in GRIDS.items()
                                           for i, p in enumerate(grid)]))
    chosen = {m: select([p for p in tuned if p["method"] == m]) for m in GRIDS}
    rep = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": [0, args.seeds - 1],
           "calibration_designs": calib, "tuning": tuned, "chosen": chosen,
           "seconds": round(time.perf_counter() - t0, 1)}
    rep["suite"], rep["curator_extra"] = args.suite, suite["curator_extra"]
    out = REPO_ROOT / "reports" / "sweeps"
    (out / f"{suite['out']}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    title = ("# Tuning at the GPU shape (tuning seeds; exploratory, nothing adopted)" if args.suite == "gpu" else
             "# Tuning at the GPU shape, realism v2 (tuning seeds; exploratory)")
    lines = [title, "",
             f"Seeds 0..{args.seeds - 1}; scenarios {list(suite['tune'])} (the S-J-R twin held out); objective = mean "
             "relative noiseless-score gain over Uniform; rule = best within one SE, then least concentrated (D-55)."
             + (f" Curator also carries {json.dumps(suite['curator_extra'])}." if suite["curator_extra"] else ""), "",
             "## A. Calibration design vs noisy-arm detection (post-audit Curator)", "",
             "| K | items/slice | k_min | detected (S5) | false flags | calib cost | score |", "|---|---|---|---|---|---|---|"]
    if not calib:
        lines[-4:] = []
    for c in calib:
        d = c["design"]
        lines.append(f"| {d['K']} | {d['items']} | {d['k_min']} | {c['detected']:.2f} | {c['false']:.3f} | "
                     f"{c['cost']:.3f} | {c['score']:.4f} |")
    lines += ["", "## B. Fair re-tuning (8 configurations per method)", "",
              "| method | chosen point | objective (rel. gain over Uniform) | SE | norm. entropy |", "|---|---|---|---|---|"]
    for m, p in chosen.items():
        lines.append(f"| {m} | `{json.dumps(p['point'])}` | {p['objective']:+.4f} | {p['se']:.4f} | {p['entropy']:.2f} |")
    lines += ["", "All points:", "", "| method | point | objective | SE | entropy |", "|---|---|---|---|---|"]
    for p in sorted(tuned, key=lambda p: (p["method"], -p["objective"])):
        lines.append(f"| {p['method']} | `{json.dumps(p['point'])}` | {p['objective']:+.4f} | {p['se']:.4f} | "
                     f"{p['entropy']:.2f} |")
    md = "\n".join(lines) + "\n"
    (out / f"{suite['out']}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
