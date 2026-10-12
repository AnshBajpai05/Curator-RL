"""Calibration Power Study (Roadmap v3 4.3, D-57/D-74).

Question: with a REALISTIC evaluation cost, which calibration design (slice size,
interval K, evidence threshold k_min, S5 response) detects the spurious-reward
arm reliably, without flagging good arms, and does it pay for itself?

Cost model (assumption until the Phase B pilot measures c_eval): one calibration
item = one greedy generation; one training prompt = G generations plus an
update, so cost_per_item = mean training unit cost / G (slightly generous to
calibration, which skips the update). Eval cost = cost_per_item x items x slices.
Items are PAIRED (fixed calibration set) with churn 0.02 per evaluation.

Grid v1 (24 configs, D-74, every slice every time): items_per_slice {25, 50, 100}
x K {5, 10} x k_min {2, 3} x s5_shrink {0.5, 0.1}.
Grid v2 (24 configs, D-75, TARGETED: a one-time all-slice baseline, then only the
slices of the max_targets most-funded arms): items_per_slice {100, 200} x
max_targets {1, 2, 3} x K {5, 10} x s5_shrink {0.5, 0.1}, k_min 2.
Plus Curator without calibration as the reference. Tuning seeds 0..n-1 only;
scenarios S-I and S-J. The selection rule is applied across both grids.

Pre-registered selection rule: among configs with (a) false-flag rate <= 5%
(share of non-noisy arms ever placed in S5) and (b) calibration cost <= 10% of
total spend, pick the highest mean objective over S-I and S-J (objective =
relative final-score improvement over Uniform, paired by seed). The chosen
config is then compared with Curator WITHOUT calibration; if it is not better,
the report says so.

    python experiments/analysis/calibration_power.py --seeds 30

Writes reports/analysis/calibration_power.{json,md}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.scheduler.baselines.uniform import UniformScheduler  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import CalibSimCfg, load_scenario  # noqa: E402
from experiments.analysis.common import REPO_ROOT  # noqa: E402
from experiments.run_sim import make_scheduler  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "analysis"
SCENARIOS = ("S-I", "S-J")
NOISY = "noisy"
CHURN = 0.02
FALSE_FLAG_MAX = 0.05
COST_SHARE_MAX = 0.10
GRID = [
    {"targeting": "all", "max_targets": 1, "items_per_slice": n, "interval": k, "k_min": km, "s5_shrink": sh}
    for n, k, km, sh in itertools.product((25, 50, 100), (5, 10), (2, 3), (0.5, 0.1))
] + [
    {"targeting": "exposure", "max_targets": m, "items_per_slice": n, "interval": k, "k_min": 2, "s5_shrink": sh}
    for n, m, k, sh in itertools.product((100, 200), (1, 2, 3), (5, 10), (0.5, 0.1))
]


def _path(sid: str) -> Path:
    return REPO_ROOT / "configs" / "sim" / f"scenario_{sid.replace('-', '').lower()}.yaml"


class S5Recorder:
    """Delegating wrapper that records which arms were ever in S5."""

    def __init__(self, inner) -> None:
        self.inner, self.env_ids, self.ever_s5 = inner, inner.env_ids, set()
        self.uses_calibration = getattr(inner, "uses_calibration", False)

    def select_mixture(self, obs):
        w = self.inner.select_mixture(obs)
        for e in self.env_ids:
            if self.inner._engine.statuses(e).value.startswith("S5"):  # noqa: SLF001
                self.ever_s5.add(e)
        return w

    def update_observation(self, obs):
        return self.inner.update_observation(obs)

    def update_calibration(self, obs):
        return self.inner.update_calibration(obs)

    def calibration_request(self):
        return self.inner.calibration_request()


def evaluate(task: tuple) -> dict:
    index, point, seeds, uniform = task
    base = load_config(REPO_ROOT / "configs" / "base.yaml")
    out = {"index": index, "point": point, "per_scenario": {}}
    for sid in SCENARIOS:
        sc = load_scenario(str(_path(sid)))
        if point is None:                                   # Curator without calibration
            cfg = base.model_copy(update={"calib": base.calib.model_copy(update={"enabled": False})})
        else:
            cost_item = float(np.mean([e.cost_usd_per_prompt for e in sc.envs])) / sc.group_size
            sc = sc.model_copy(update={"calib": CalibSimCfg(
                enabled=True, interval_rounds=point["interval"], paired=True, churn=CHURN,
                items_per_slice=point["items_per_slice"], cost_per_item_usd=cost_item)})
            cfg = base.model_copy(update={
                "calib": base.calib.model_copy(update={
                    "enabled": True, "k_min": point["k_min"], "interval_rounds": point["interval"],
                    "targeting": point["targeting"], "max_targets": point["max_targets"],
                    "items_per_slice": point["items_per_slice"]}),
                "scheduler": base.scheduler.model_copy(update={"s5_shrink": point["s5_shrink"]}),
            })
        rel, cost_share, detected, false_flags, noisy_share = [], [], [], [], []
        for i, s in enumerate(seeds):
            rec = S5Recorder(make_scheduler("curator", sc, s, cfg=cfg))
            r = run_episode(rec, sc, seed=s, method="curator")
            rel.append((r.final_score - uniform[sid][i]) / uniform[sid][i])
            cost_share.append(sum(c["eval_cost_usd"] for c in r.calib_logs) / max(r.total_cost_usd, 1e-12))
            detected.append(NOISY in rec.ever_s5)
            others = [e for e in rec.env_ids if e != NOISY]
            false_flags.append(sum(e in rec.ever_s5 for e in others) / len(others))
            noisy_share.append(float(np.mean([log["weights"][NOISY] for log in r.round_logs[8:]])))
        rel_a = np.asarray(rel)
        out["per_scenario"][sid] = {
            "objective_mean": float(rel_a.mean()),
            "objective_se": float(rel_a.std(ddof=1) / np.sqrt(len(rel_a))),
            "rel_by_seed": [float(v) for v in rel_a],
            "calib_cost_share": float(np.mean(cost_share)),
            "noisy_detected_rate": float(np.mean(detected)),
            "false_flag_rate": float(np.mean(false_flags)),
            "noisy_share": float(np.mean(noisy_share)),
        }
    ps = out["per_scenario"]
    out["objective_mean"] = float(np.mean([ps[s]["objective_mean"] for s in SCENARIOS]))
    out["false_flag_rate"] = float(max(ps[s]["false_flag_rate"] for s in SCENARIOS))
    out["calib_cost_share"] = float(max(ps[s]["calib_cost_share"] for s in SCENARIOS))
    return out


def render(report: dict) -> str:
    sel, ref = report["selected"], report["reference_no_calibration"]
    lines = ["# Calibration Power Study (D-74)", "",
             f"Tuning seeds {report['seeds'][0]}..{report['seeds'][-1]}; S-I and S-J; paired items, churn {CHURN}; "
             "eval cost = (mean training unit cost / G) x items x slices. Objective: relative improvement over "
             f"Uniform. Constraints: false-flag rate <= {FALSE_FLAG_MAX:.0%}, calibration cost <= {COST_SHARE_MAX:.0%} of spend.", "",
             "| # | targeting | items/slice | K | k_min | S5 shrink | S-I obj | S-J obj | mean | noisy detected (I/J) | false flags | calib cost | noisy share (I) | |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in [ref] + report["configs"]:
        p = r["point"] or {}
        si, sj = r["per_scenario"]["S-I"], r["per_scenario"]["S-J"]
        mark = "**chosen**" if r is not ref and r["index"] == sel.get("index") else ("reference" if r is ref else
               ("ok" if r["feasible"] else ""))
        lines.append(
            f"| {'—' if r is ref else r['index']} | "
            f"{(p['targeting'] + ('' if p['targeting'] == 'all' else ' top-' + str(p['max_targets']))) if p else 'no calib'} | "
            f"{p.get('items_per_slice', '—')} | {p.get('interval', '—')} | "
            f"{p.get('k_min', '—')} | {p.get('s5_shrink', '—')} | {si['objective_mean']:+.4f} | {sj['objective_mean']:+.4f} | "
            f"{r['objective_mean']:+.4f} | {si['noisy_detected_rate']:.2f} / {sj['noisy_detected_rate']:.2f} | "
            f"{r['false_flag_rate']:.3f} | {r['calib_cost_share']:.3f} | {si['noisy_share']:.2f} | {mark} |")
    lines += ["", "## Chosen vs Curator without calibration (paired by seed, tuning seeds)", ""]
    for sid, c in report["chosen_vs_reference"].items():
        lines.append(f"- {sid}: {c['mean_diff']:+.4f} (se {c['se']:.4f})")
    lines += ["", f"**Verdict:** {report['verdict']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibration Power Study (D-74)")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    t0 = time.perf_counter()
    uniform = {}
    for sid in SCENARIOS:
        sc = load_scenario(str(_path(sid)))
        ids = [e.env_id for e in sc.envs]
        uniform[sid] = [run_episode(UniformScheduler(ids), sc, seed=s, method="u").final_score for s in seeds]
    tasks = [(None, None, seeds, uniform)] + [(i, p, seeds, uniform) for i, p in enumerate(GRID)]
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        results = list(pool.map(evaluate, tasks))
    ref, configs = results[0], results[1:]
    for r in configs:
        r["feasible"] = bool(r["false_flag_rate"] <= FALSE_FLAG_MAX and r["calib_cost_share"] <= COST_SHARE_MAX)
    feasible = [r for r in configs if r["feasible"]]
    chosen = max(feasible, key=lambda r: r["objective_mean"]) if feasible else None
    comp = {}
    if chosen is not None:
        for sid in SCENARIOS:
            d = np.asarray(chosen["per_scenario"][sid]["rel_by_seed"]) - np.asarray(ref["per_scenario"][sid]["rel_by_seed"])
            comp[sid] = {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        diffs = [comp[s]["mean_diff"] for s in SCENARIOS]
        ses = [comp[s]["se"] for s in SCENARIOS]
        if all(d > 2 * se for d, se in zip(diffs, ses)):
            outcome = "BEATS Curator without calibration in both scenarios (paired diff > 2 SE)"
        elif any(d < -2 * se for d, se in zip(diffs, ses)):
            outcome = "is WORSE than Curator without calibration in at least one scenario"
        else:
            outcome = "is statistically INDISTINGUISHABLE from Curator without calibration"
        det = min(chosen["per_scenario"][s]["noisy_detected_rate"] for s in SCENARIOS)
        verdict = (f"config #{chosen['index']} {json.dumps(chosen['point'])} is chosen by the rule; it {outcome}. "
                   f"It detects the noisy arm in {det:.0%} of seeds (min over scenarios)"
                   + (" - i.e. the rule selected calibration that effectively never fires." if det < 0.1 else "."))
    else:
        verdict = "no configuration meets both constraints."
    report = {"seeds": seeds, "grid": GRID, "reference_no_calibration": ref, "configs": configs,
              "selected": {"index": chosen["index"], "point": chosen["point"]} if chosen else {},
              "chosen_vs_reference": comp, "verdict": verdict,
              "seconds": round(time.perf_counter() - t0, 1)}
    for r in [ref] + configs:
        for sid in SCENARIOS:
            r["per_scenario"][sid].pop("rel_by_seed", None) if r is not ref and r is not chosen else None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "calibration_power.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "calibration_power.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
