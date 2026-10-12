"""Gate 2'' verification (Roadmap v3 Phase C, D-68).

Runs the tuned Curator against every tuned baseline on EVALUATION seeds
(100-149, disjoint from the tuning seeds 0-49) and checks, as pre-registered in
`docs/GATES.md`:

(i)   Curator beats the SEC-style AND the DUMP-style bandit in S-C and in S-I:
      paired final-score difference with a 95% bootstrap interval excluding 0;
(ii)  Curator does not underperform Uniform in the homogeneous twin S-J: lower
      bound of the paired interval > -delta_gap (0.01 = 1 benchmark point);
(iii) the original Gate 2 criteria (a)-(e) still hold with the tuned configs
      (re-uses experiments/gate2_check.py);
(iv)  the junk-share criterion holds (it is criterion (c) of (iii)).

Also reported (informational, never pass/fail): paired Curator minus each of
Uniform, Static, LP, Standard UCB in every scenario, the allocation
concentration of every method (D-67), and the S-J result as the H6 regime check.
S-J was NOT used for tuning.

    python experiments/gate2b_check.py --seeds 50

Writes reports/gate2/gate2b.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from curator_rl.simulator.scenarios import load_scenario  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04)
    make_scheduler,  # noqa: E402
)

FROZEN_CFG = load_config(FROZEN_D68_D94)
REPORT_DIR = REPO_ROOT / "reports" / "gate2"
SCENARIOS = ("S-A", "S-B", "S-C", "S-I", "S-J")
METHODS = ("uniform", "static", "lp", "ucb", "sec", "dump", "curator")
SKIP_ROUNDS = 8
BOOTSTRAP_RNG_SEED = 2027


def _path(sid: str) -> Path:
    return REPO_ROOT / "configs" / "sim" / f"scenario_{sid.replace('-', '').lower()}.yaml"


def paired(results, a: str, b: str, rng) -> dict:
    diffs = [x.final_score - y.final_score for x, y in zip(results[a], results[b])]
    mean, lo, hi = gate2_check.paired_bootstrap(diffs, rng)
    return {"mean_diff": mean, "ci95": [lo, hi]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate 2'' verification (D-68)")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed0", type=int, default=gate2_check.EVAL_SEED0)
    args = parser.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)

    results: dict[str, dict[str, list]] = {}
    for sid in SCENARIOS:
        scenario = load_scenario(str(_path(sid)))
        results[sid] = {
            m: [run_episode(make_scheduler(m, scenario, s, cfg=FROZEN_CFG), scenario, seed=s, method=m)
                for s in seeds]
            for m in METHODS
        }
        print(f"  {sid}: " + " ".join(
            f"{m}={np.mean([r.final_score for r in results[sid][m]]):.4f}" for m in METHODS), flush=True)

    criteria: dict[str, dict] = {}

    # (i) Curator vs SEC-style and DUMP-style in S-C and S-I
    block, ok = {}, True
    for sid in ("S-C", "S-I"):
        for rival in ("sec", "dump"):
            r = paired(results[sid], "curator", rival, rng)
            r["pass"] = bool(r["ci95"][0] > 0.0)
            ok &= r["pass"]
            block[f"{sid}:curator-{rival}"] = r
    criteria["i_beats_learnability_bandits"] = {"comparisons": block, "pass": bool(ok)}

    # (ii) homogeneous twin: no underperformance vs Uniform (H6 regime)
    r = paired(results["S-J"], "curator", "uniform", rng)
    r["delta_gap"] = gate2_check.DELTA_GAP
    r["pass"] = bool(r["ci95"][0] > -gate2_check.DELTA_GAP)
    criteria["ii_homogeneous_no_underperformance"] = r

    # (iii)/(iv) original Gate 2 criteria with the tuned configuration
    code = gate2_check.main(["--seeds", str(args.seeds), "--seed0", str(args.seed0)])
    g2 = json.loads((REPORT_DIR / "gate2.json").read_text(encoding="utf-8"))
    criteria["iii_original_gate2_holds"] = {
        "sub": {k: v.get("pass") for k, v in g2["criteria"].items() if k[:2] in ("a_", "b_", "c_", "d_", "e_")},
        "pass": bool(g2["all_pass"]),
    }
    criteria["iv_junk_share"] = {"pass": bool(g2["criteria"]["c_junk_share"]["pass"])}

    # informational: paired Curator minus every method, per scenario; concentration
    info = {
        sid: {m: paired(results[sid], "curator", m, rng) for m in METHODS if m != "curator"}
        for sid in SCENARIOS
    }
    conc = {
        sid: {
            m: {k: float(np.nanmean([r.concentration(skip_rounds=SKIP_ROUNDS)[k] for r in results[sid][m]]))
                for k in ("mean_max_weight", "norm_entropy", "frac_rounds_over_half")}
            for m in METHODS
        }
        for sid in SCENARIOS
    }
    means = {sid: {m: float(np.mean([r.final_score for r in results[sid][m]])) for m in METHODS}
             for sid in SCENARIOS}

    report = {
        "generated_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "seeds": seeds,
        "criteria": criteria,
        "all_pass": bool(all(c["pass"] for c in criteria.values())) and code in (0, 1),
        "means": means,
        "curator_minus": info,
        "concentration": conc,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "gate2b.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "gate2b.md").write_text(render_md(report), encoding="utf-8")
    print("\nGATE 2'':", "PASS" if report["all_pass"] else "FAIL")
    for name, c in criteria.items():
        print(f"  {name}: pass={c['pass']}")
    for k, v in criteria["i_beats_learnability_bandits"]["comparisons"].items():
        print(f"     {k}: {v['mean_diff']:+.4f} {v['ci95'][0]:+.4f}..{v['ci95'][1]:+.4f} pass={v['pass']}")
    print(f"report: {REPORT_DIR / 'gate2b.md'}")
    return 0 if report["all_pass"] else 1


def render_md(report: dict) -> str:
    lines = ["# Gate 2'' verification report (D-68)", "",
             f"Generated {report['generated_utc']}; evaluation seeds {report['seeds'][0]}..{report['seeds'][-1]}; "
             "tuned configurations from `reports/sweeps/tune_all.md`; S-J was not used for tuning.", "",
             "## Mean final scores", "",
             "| scenario | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for sid, per in report["means"].items():
        lines.append(f"| {sid} | " + " | ".join(f"{per[m]:.4f}" for m in METHODS) + " |")
    lines += ["", "## Criteria", ""]
    for name, c in report["criteria"].items():
        lines.append(f"- **{name}**: {'PASS' if c['pass'] else 'FAIL'}")
        for k, v in c.get("comparisons", {}).items():
            lines.append(f"  - {k}: {v['mean_diff']:+.4f}, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                         f"-> {'pass' if v['pass'] else 'fail'}")
        if "ci95" in c:
            lines.append(f"  - S-J curator-uniform: {c['mean_diff']:+.4f}, 95% CI "
                         f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], tolerance -{c['delta_gap']}")
        if "sub" in c:
            lines.append(f"  - original criteria: {json.dumps(c['sub'])}")
    lines += ["", f"**Overall: {'PASS' if report['all_pass'] else 'FAIL'}**", "",
              "## Informational: Curator minus each method (paired, 95% CI)", "",
              "| scenario | " + " | ".join(m for m in METHODS if m != "curator") + " |",
              "|---|" + "---|" * (len(METHODS) - 1)]
    for sid, per in report["curator_minus"].items():
        lines.append(f"| {sid} | " + " | ".join(
            f"{per[m]['mean_diff']:+.3f} [{per[m]['ci95'][0]:+.3f}, {per[m]['ci95'][1]:+.3f}]"
            for m in METHODS if m != "curator") + " |")
    lines += ["", "## Allocation concentration after round 8 (mean max-weight / normalised entropy)", "",
              "| scenario | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for sid, per in report["concentration"].items():
        lines.append(f"| {sid} | " + " | ".join(
            f"{per[m]['mean_max_weight']:.2f} / {per[m]['norm_entropy']:.2f}" for m in METHODS) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
