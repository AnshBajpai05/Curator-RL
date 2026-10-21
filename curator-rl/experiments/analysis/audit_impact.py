"""Impact of the 2026-10-10 audit fixes, measured on TUNING seeds 0-49 (exploratory; no decision).

Scenarios: the GPU-shape refits (audit E-05): R = 2 x P = 16 per round, ~60 rounds, learning from
delivered prompts with deficit-carry rounding, the noisy arm scored on the GSM8K calibration slice
(as on real hardware, E-03), calibration charged at generation-only cost (E-15).

Methods:
- curator_pre_audit: the adopted D-95 configuration with every legacy behaviour (frozen D-68..D-94
  config + claim_stale/2): LP-B SE inflated (E-01), stale flags re-sent (E-02), own-slice credit
  (E-03), requested-weight counters (E-09);
- curator_fixed: configs/base.yaml after the audit;
- one-fix-reverted ablations of curator_fixed (isolates each fix), and two candidate improvements
  (B5 S5 cap at the floor; B2 gamma = 1) that would still need a pre-registration;
- the baselines with the post-audit base config.

Primary metric: the noiseless final score (EpisodeResult.final_score_expected, audit E-10), paired by
seed. Seeds 0-49 are tuning seeds: nothing here is a confirmatory result; it says which fixes
matter and what to pre-register (fresh seeds >= 700).

    python experiments/analysis/audit_impact.py --seeds 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config, rescale_for_batch_shape  # noqa: E402
from curator_rl.core.stats import paired_t_interval  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments.run_sim import FROZEN_D68_D94, make_scheduler  # noqa: E402

SCENARIOS = {"S-I-R1-GPU": "scenario_si_r1_gpu.yaml", "S-I-R2-GPU": "scenario_si_r2_gpu.yaml",
             "S-J-R-GPU": "scenario_sj_r_gpu.yaml"}
NOISY = "noisy"
SKIP = 6
REFERENCE = "curator_pre_audit"
GPU_PROMPTS_PER_ROUND = 32
GPU_TUNED = {                       # chosen points of reports/sweeps/tune_gpu_shape.md (tuning seeds 0-39)
    "curator": {"gamma": 1.0, "exploration_coef": 0.1, "tau": 0.3},
    "lp": {"tau": 0.5},
    "ucb": {"exploration_coef": 0.25, "tau": 0.3},
    "sec": {"alpha": 1.0, "tau": 0.4},
    "dump": {"exploration_coef": 0.3, "tau": 0.4, "window_rounds": 10},
}


def configs() -> dict:
    # GPU shape: 32 prompts per round; the status evidence threshold is rescaled (audit E-28) for every
    # post-audit config. The pre-audit reference keeps its unscaled threshold, as it would have run.
    base = rescale_for_batch_shape(load_config(REPO_ROOT / "configs" / "base.yaml"), GPU_PROMPTS_PER_ROUND)
    frozen = load_config(FROZEN_D68_D94)
    pre = frozen.model_copy(update={"calib": frozen.calib.model_copy(
        update={"targeting": "claim_stale", "max_targets": 2})})

    def tweak(cfg, section, **kw):
        return cfg.model_copy(update={section: getattr(cfg, section).model_copy(update=kw)})

    def baseline(cfg, name, point):
        sec = getattr(cfg.baselines, name).model_copy(update=point)
        return cfg.model_copy(update={"baselines": cfg.baselines.model_copy(update={name: sec})})

    def tweak_signals(cfg, **kw):
        return cfg.model_copy(update={"signals": cfg.signals.model_copy(update=kw)})

    return {
        "curator_pre_audit": ("curator", pre),
        "curator_fixed": ("curator", base),
        "fixed_but_legacy_lp_se": ("curator", tweak_signals(base, lp_se="residual_legacy")),
        "fixed_but_stale_flags": ("curator", tweak(base, "calib", flag_scope="all")),
        "fixed_but_own_slice_credit": ("curator", tweak(base, "calib", credit="own_slice")),
        "fixed_but_requested_counters": ("curator", tweak(base, "scheduler", pull_unit="round")),
        "fixed_but_unscaled_n_min": ("curator", load_config(REPO_ROOT / "configs" / "base.yaml")),
        "fixed_plus_s5_cap": ("curator", tweak(base, "scheduler", s5_cap="floor")),
        "fixed_plus_gamma1": ("curator", tweak(base, "scheduler", gamma=1.0)),
        "uniform": ("uniform", base), "lp": ("lp", base), "ucb": ("ucb", base),
        "sec": ("sec", base), "dump": ("dump", base),
        # every method at its own GPU-shape-tuned point (reports/sweeps/tune_gpu_shape.md, equal budget)
        "curator_gpu_tuned": ("curator", tweak(base, "scheduler", **GPU_TUNED["curator"])),
        "curator_gpu_tuned_K5": ("curator", tweak(base, "scheduler", **GPU_TUNED["curator"])),
        "lp_gpu_tuned": ("lp", baseline(base, "lp", GPU_TUNED["lp"])),
        "ucb_gpu_tuned": ("ucb", baseline(base, "ucb", GPU_TUNED["ucb"])),
        "sec_gpu_tuned": ("sec", baseline(base, "sec", GPU_TUNED["sec"])),
        "dump_gpu_tuned": ("dump", baseline(base, "dump", GPU_TUNED["dump"])),
    }


class Recorder:
    """Delegating wrapper: ever-S5 arms and the noisy arm's LP z-scores (Curator only)."""

    def __init__(self, inner) -> None:
        self.inner, self.env_ids = inner, inner.env_ids
        self.uses_calibration = getattr(inner, "uses_calibration", False)
        self.ever_s5: set[str] = set()
        self.noisy_z: list[float] = []

    def select_mixture(self, obs):
        w = self.inner.select_mixture(obs)
        vectors = getattr(self.inner, "_latest_vectors", None) or {}
        for e, sv in vectors.items():
            if sv.status.value.startswith("S5"):
                self.ever_s5.add(e)
        if NOISY in vectors and vectors[NOISY].n_rounds_seen >= SKIP:
            self.noisy_z.append(float(vectors[NOISY].lp_z))
        return w

    def update_observation(self, obs):
        self.inner.update_observation(obs)

    def update_calibration(self, obs):
        self.inner.update_calibration(obs)

    def calibration_request(self):
        return self.inner.calibration_request()


def run_cell(task) -> dict:
    sid, name, seeds = task
    factory, cfg = configs()[name]
    sc = load_scenario(REPO_ROOT / "configs" / "sim" / SCENARIOS[sid])
    if name.endswith("_K5"):            # calibration every 5 rounds (tune_gpu_shape.md part A)
        sc = sc.model_copy(update={"calib": sc.calib.model_copy(update={"interval_rounds": 5})})
    ids = [e.env_id for e in sc.envs]
    out = {"sid": sid, "name": name, "score": [], "noisy_share": [], "calib_cost": [], "detected": [],
           "false": [], "noisy_z": [], "max_w": []}
    for s in seeds:
        rec = Recorder(make_scheduler(factory, sc, s, cfg=cfg))
        r = run_episode(rec, sc, seed=s, method=name)
        out["score"].append(r.final_score_expected)
        shares = [lg["per_env"][NOISY]["n_prompts"] / max(sum(v["n_prompts"] for v in lg["per_env"].values()), 1)
                  for lg in r.round_logs[SKIP:]]
        out["noisy_share"].append(float(np.mean(shares)) if shares else float("nan"))
        out["calib_cost"].append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
        out["max_w"].append(r.concentration(skip_rounds=SKIP)["mean_max_weight"])
        if rec.uses_calibration:
            out["detected"].append(float(NOISY in rec.ever_s5))
            out["false"].append(sum(e in rec.ever_s5 for e in ids if e != NOISY) / (len(ids) - 1))
        out["noisy_z"].extend(rec.noisy_z)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit fix impact on tuning seeds")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    names = list(configs())
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        cells = list(pool.map(run_cell, [(sid, n, seeds) for sid in SCENARIOS for n in names]))
    res = {(c["sid"], c["name"]): c for c in cells}
    table = {}
    for sid in SCENARIOS:
        ref = res[(sid, REFERENCE)]["score"]
        for n in names:
            c = res[(sid, n)]
            diff = paired_t_interval([a - b for a, b in zip(c["score"], ref)]) if n != REFERENCE else (0.0, 0.0, 0.0)
            table[f"{sid}|{n}"] = {
                "score": statistics.fmean(c["score"]), "vs_pre_audit": diff,
                "noisy_share": statistics.fmean(c["noisy_share"]), "calib_cost": statistics.fmean(c["calib_cost"]),
                "max_weight": float(np.nanmean(c["max_w"])),
                "detected": statistics.fmean(c["detected"]) if c["detected"] else None,
                "false_flags": statistics.fmean(c["false"]) if c["false"] else None,
                "noisy_z_sd": statistics.stdev(c["noisy_z"]) if len(c["noisy_z"]) > 2 else None}
    report = {"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "seeds": [seeds[0], seeds[-1]],
              "note": "tuning seeds; exploratory; primary metric = noiseless final score", "table": table,
              "seconds": round(time.perf_counter() - t0, 1)}
    out = REPO_ROOT / "reports" / "analysis"
    (out / "audit_impact.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render(report, names)
    (out / "audit_impact.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def render(rep: dict, names: list[str]) -> str:
    def f(x, fmt):
        return "-" if x is None else format(x, fmt)

    lines = ["# Impact of the audit fixes (GPU-shape refits, tuning seeds; exploratory)", "",
             f"Generated {rep['generated_utc']}; seeds {rep['seeds'][0]}..{rep['seeds'][1]}. Score = noiseless final "
             "benchmark score (E-10). Paired differences against `curator_pre_audit` (the adopted D-95 Curator "
             "with every legacy behaviour), 95% t-interval. Noisy z SD = spread of the noisy arm's LP z-scores "
             "(true slope 0: should be about 1).", ""]
    for sid in SCENARIOS:
        lines += [f"## {sid}", "", "| method | score | vs pre-audit [95% CI] | noisy share | detected (S5) | "
                  "false flags | calib cost | max-weight | noisy z SD |", "|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            r = rep["table"][f"{sid}|{n}"]
            d = r["vs_pre_audit"]
            ci = "-" if n == REFERENCE else f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
            lines.append(f"| {n} | {r['score']:.4f} | {ci} | {r['noisy_share']:.3f} | {f(r['detected'], '.2f')} | "
                         f"{f(r['false_flags'], '.3f')} | {r['calib_cost']:.3f} | {r['max_weight']:.2f} | "
                         f"{f(r['noisy_z_sd'], '.2f')} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
