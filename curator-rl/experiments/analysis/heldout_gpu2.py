"""Every method at its realism-v2 tuned point, on the held-out twin and the stress scenarios (TUNING seeds).

Reads the chosen points from reports/sweeps/tune_gpu2.json (equal tuning budget, D-55 rule) and runs
them on all six GPU2 scenarios: the two tuning scenarios, the held-out homogeneous-cost twin
S-J-R-GPU2, and three stress tests (a misaligned arm only calibration could catch; random-reward
training that hurts or helps GSM8K). Adds `pilot_uniform` (B10) and Curator without calibration.
These are the configurations the Kaggle preset `gpu_tuned` runs. Exploratory: tuning seeds 0-39.

    python experiments/analysis/heldout_gpu2.py --seeds 40
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import experiments.analysis.component_study_gpu2 as cs  # noqa: E402
from curator_rl.core.stats import paired_t_interval  # noqa: E402

PREFIX = {"curator": "scheduler", "ucb": "baselines.ucb", "lp": "baselines.lp", "sec": "baselines.sec",
          "dump": "baselines.dump"}


def tuned_methods() -> dict[str, tuple[str, dict]]:
    rep = json.loads((REPO_ROOT / "reports" / "sweeps" / "tune_gpu2.json").read_text(encoding="utf-8"))
    out: dict[str, tuple[str, dict]] = {"uniform": ("uniform", {}), "pilot_uniform": ("pilot_uniform", {})}
    for m, chosen in rep["chosen"].items():
        ov = {f"{PREFIX[m]}.{k}": v for k, v in chosen["point"].items()}
        if m == "curator":
            ov.update(rep["curator_extra"])
            out["curator_nocal"] = ("curator", {**ov, "calib.enabled": False})
        out[m] = (m, ov)
    return out


cs.methods = tuned_methods          # run_cell resolves methods through the module (also in worker processes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    names = list(tuned_methods())
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(cs.run_cell, [(m, sid, seeds) for m in names for sid in cs.SCENARIOS]))
    res = {(c["name"], c["sid"]): c["rows"] for c in cells}
    table: dict[str, dict] = {}
    lines = ["# Held-out check at the GPU shape, realism v2 (tuning seeds; exploratory)", "",
             f"Each method at its tuned point from `reports/sweeps/tune_gpu2.md` (seeds 0..{args.seeds - 1}). "
             "Relative gain of the noiseless final score over Uniform, same seed, 95% t-interval. "
             "S-I-R1-GPU2 and S-I-R2-GPU2 were used for tuning; the other four were not.", "",
             "| method | " + " | ".join(cs.SCENARIOS) + " |", "|---|" + "---|" * len(cs.SCENARIOS)]
    for name in names:
        cells_md = []
        for sid in cs.SCENARIOS:
            rows = res[(name, sid)]
            rel = [(a["score"] - u["score"]) / u["score"] for a, u in zip(rows, res[("uniform", sid)])]
            m, lo, hi = paired_t_interval(rel) if name != "uniform" else (0.0, 0.0, 0.0)
            table[f"{name}|{sid}"] = {"rel_gain": [m, lo, hi],
                                      "noisy_share": statistics.fmean(r["noisy_share"] for r in rows),
                                      "false_flags": statistics.fmean(r["false_flags"] for r in rows),
                                      "detected": {e: sum(r["detected"][e] is not None for r in rows) / len(rows)
                                                   for e in rows[0]["detected"]}}
            cells_md.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {name} | " + " | ".join(cells_md) + " |")
    lines += ["", "Budget share of the noisy arm / false flags (real arms ever in S5) / junk arms detected:", "",
              "| method | " + " | ".join(cs.SCENARIOS) + " |", "|---|" + "---|" * len(cs.SCENARIOS)]
    for name in names:
        cells_md = []
        for sid in cs.SCENARIOS:
            t = table[f"{name}|{sid}"]
            det = "/".join(f"{v:.2f}" for v in t["detected"].values())
            cells_md.append(f"{t['noisy_share']:.3f} / {t['false_flags']:.3f} / {det}")
        lines.append(f"| {name} | " + " | ".join(cells_md) + " |")
    out = REPO_ROOT / "reports" / "analysis"
    (out / "heldout_gpu2.json").write_text(json.dumps({"generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                                                       "seeds": [0, args.seeds - 1], "methods": tuned_methods(),
                                                       "table": table}, indent=2), encoding="utf-8")
    md = "\n".join(lines) + "\n"
    (out / "heldout_gpu2.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
