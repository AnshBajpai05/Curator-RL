"""Improvement B3: does a longer learning-progress window help? (realism v2, tuning seeds 0-39)

The OLS slope's SE scales as W^(-3/2), so W = 20 cuts it 2.8x versus W = 10; the question is whether
the slower response costs more than the noise reduction gains. Curator at the Kaggle preset
`gpu_tuned`, W in {10, 15, 20, 30}. Writes reports/analysis/lp_window_gpu2.md.

    python experiments/analysis/lp_window_gpu2.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import experiments.analysis.component_study_gpu2 as st
from curator_rl.core.stats import paired_t_interval

BASE = {"scheduler.gamma": 1.0, "scheduler.exploration_coef": 0.25, "scheduler.tau": 1.0,
        "signals.reward_audit.enabled": True, "scheduler.s5_cap": "floor", "calib.mismatch_windows": 3}
WINDOWS = (10, 15, 20, 30)


def methods():
    out = {"uniform": ("uniform", {})}
    for w in WINDOWS:
        out[f"W{w}"] = ("curator", {**BASE, "signals.window_rounds": w})
    return out


st.methods = methods

if __name__ == "__main__":
    seeds = list(range(40))
    sids = ["S-I-R1-GPU2", "S-I-R2-GPU2", "S-J-R-GPU2"]
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(st.run_cell, [(m, sid, seeds) for m in methods() for sid in sids]))
    res = {(c["name"], c["sid"]): c["rows"] for c in cells}
    lines = ["# Learning-progress window length (improvement B3; realism v2, tuning seeds 0-39)", "",
             "Curator at the Kaggle preset `gpu_tuned`; relative gain of the noiseless final score over Uniform "
             "(95% t-interval); last column: absolute difference to W = 10 on S-I-R1-GPU2.", "",
             "| window | " + " | ".join(f"{s} gain over Uniform" for s in sids) + " | minus W10 (S-I-R1) |",
             "|---|" + "---|" * (len(sids) + 1)]
    for w in WINDOWS:
        name = f"W{w}"
        parts = []
        for sid in sids:
            rel = [(a["score"] - u["score"]) / u["score"] for a, u in zip(res[(name, sid)], res[("uniform", sid)])]
            m, lo, hi = paired_t_interval(rel)
            parts.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        d = [a["score"] - b["score"] for a, b in zip(res[(name, "S-I-R1-GPU2")], res[("W10", "S-I-R1-GPU2")])]
        m, lo, hi = paired_t_interval(d) if w != 10 else (0, 0, 0)
        lines.append(f"| {w} | " + " | ".join(parts) + f" | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "No window beats W = 10 consistently; W = 30 is worse on S-I-R1-GPU2. W stays 10."]
    md = "\n".join(lines) + "\n"
    (Path(__file__).resolve().parents[2] / "reports" / "analysis" / "lp_window_gpu2.md").write_text(md, encoding="utf-8")
    print(md)
