"""Compute efficiency: does a method reach accuracy with less GPU than Uniform? (proposed pre-registration D-108)

Why not "GPU-hours to reach Uniform's final level" (the runner's descriptive table): when two methods tie,
the other one crosses Uniform's final value before the end about half the time by chance alone, so a count
of early crossings is not evidence. This script compares whole learning curves at matched compute.

Per seed, with Uniform of the same seed as reference, on the periodic-report items (the same fixed random
subset for every method; the untrained model is the point at 0 GPU-hours):

- primary, `delta`: mean over a fixed GPU-hour grid of (method accuracy - Uniform accuracy), each curve
  linearly interpolated between its evaluations. Positive = the method is ahead at matched compute (area
  between the curves divided by the grid span), in percentage points;
- secondary, `saving`: both curves made monotone (isotonic, pool-adjacent-violators), then
  1 - (GPU-hours where the method's curve reaches Uniform's final smoothed value) / (Uniform's final
  GPU-hours); "not reached" counts as no saving and is reported as such.

Across seeds: mean, SE (sd / sqrt(n)), Student-t interval, sign-flip p, and the D-103 style rule (all seeds
the same sign and |mean| >= 2 SE).

    python experiments/analysis/compute_efficiency.py --runs ../curator_results_november/runs \
        --methods curator,curator_nocal,curator_q,lp,sec,dump --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from curator_rl.core.stats import paired_t_interval, sign_flip_pvalue  # noqa: E402
from curator_rl.evaluation.paired_items import learning_curve  # noqa: E402
from experiments.analysis.d103_confirm import (  # noqa: E402
    CONTEXT,
    GRID,
    HIGHLIGHT,
    INK,
    MUTED,
    SURFACE,
    _axes,
    load_runs,
)

DEFAULT_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def rows_of(s: dict) -> list[dict]:
    path = Path(s["_dir"]) / "reports.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def curve(s: dict, base_final: dict | None) -> list[tuple[float, float]]:
    n_subset = int((s.get("run_config") or {}).get("report_items", 100))
    return learning_curve(rows_of(s), n_subset=n_subset, base_final=base_final)


def gpu_hours_by_env(s: dict) -> dict[str, float]:
    """GPU-hours charged to each environment over the run (per-round `gpu_seconds`, cost/attribution.py)."""
    path = Path(s["_dir"]) / "rounds.jsonl"
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            for e, a in (json.loads(line).get("arms") or {}).items():
                out[e] = out.get(e, 0.0) + float(a.get("gpu_seconds") or 0.0) / 3600.0
    return out


def isotonic(ys: list[float]) -> list[float]:
    """Pool-adjacent-violators: the closest non-decreasing sequence (least squares)."""
    blocks = [[y, 1] for y in ys]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, n in blocks:
        out += [v] * n
    return out


def delta(cm, cu, grid) -> float | None:
    end = min(cm[-1][0], cu[-1][0])
    pts = [g for g in grid if g <= end + 1e-9]
    if not pts:
        return None
    hm, am = zip(*cm)
    hu, au = zip(*cu)
    return float(np.mean([np.interp(g, hm, am) - np.interp(g, hu, au) for g in pts]))


def saving(cm, cu) -> float | None:
    """1 - hours(method reaches Uniform's smoothed final) / Uniform's final hours; None = not reached."""
    hm, am = zip(*cm)
    hu, au = zip(*cu)
    sm, su = isotonic(list(am)), isotonic(list(au))
    target, h_ref = su[-1], hu[-1]
    for (h0, v0), (h1, v1) in zip(zip(hm, sm), list(zip(hm, sm))[1:]):
        if v0 >= target:
            return 1.0 - h0 / h_ref
        if v1 >= target:
            h = h0 + (h1 - h0) * (target - v0) / (v1 - v0) if v1 > v0 else h1
            return 1.0 - h / h_ref
    return None


def summarise(vals: list[float]) -> dict:
    out = {"n": len(vals), "mean": statistics.fmean(vals) if vals else None, "se": None, "lo": None, "hi": None, "p": None}
    if len(vals) >= 2:
        out["se"] = statistics.stdev(vals) / math.sqrt(len(vals))
        _, out["lo"], out["hi"] = paired_t_interval(vals)
        out["p"] = sign_flip_pvalue(vals)
    return out


def verdict(vals: list[float], need: int, min_positive: int | None = None) -> str:
    """Holds if mean >= 2 SE and at least `min_positive` seeds are positive (default: all of them)."""
    if len(vals) < need:
        return f"PENDING ({len(vals)} of {need} seeds)"
    s = summarise(vals)
    positive = sum(v > 0 for v in vals)
    enough = positive >= (len(vals) if min_positive is None else min_positive)
    big = s["se"] is not None and s["mean"] >= 2 * s["se"]
    return "HOLDS" if enough and big else "FAILS"


def fig_curves(curves: dict[str, list], ref: str, path: Path, grid) -> None:
    plt, fig, ax = _axes(8, 4.4)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", length=3)
    for name, cs in curves.items():
        color = CONTEXT if name == ref else HIGHLIGHT
        for c in cs:
            h, a = zip(*c)
            ax.plot(h, [100 * x for x in a], color=color, linewidth=0.8, alpha=0.35, zorder=2)
        g = [x for x in grid if all(c[-1][0] >= x - 1e-9 for c in cs)]
        mean = [100 * statistics.fmean(np.interp(x, *zip(*c)) for c in cs) for x in [0.0, *g]]
        ax.plot([0.0, *g], mean, color=color, linewidth=2.4, zorder=3, label=f"{name} (mean of {len(cs)} seeds)")
    ax.set_xlabel("charged GPU-hours")
    ax.set_ylabel("sampled accuracy, macro (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Accuracy at matched compute", loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "Thin lines: single seeds. Same 100 items per domain at every point; 0 GPU-h = untrained model.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", action="append", required=True, help="PATH or TAG=PATH of a runs folder")
    ap.add_argument("--base-runs", type=Path, default=None, help="folder holding base_s0 (default: the first --runs)")
    ap.add_argument("--methods", default="curator_nocal", help="comma-separated; the first is the primary")
    ap.add_argument("--reference", default="uniform")
    ap.add_argument("--grid", default=",".join(map(str, DEFAULT_GRID)))
    ap.add_argument("--seeds-needed", type=int, default=3)
    ap.add_argument("--min-positive", type=int, default=None, help="seeds that must be positive (default: all)")
    ap.add_argument("--expect-negative", default="", help="comma-separated methods hypothesised to be BEHIND Uniform")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=None,
                    help="price of one GPU-hour; when given, GPU-hours are also reported in dollars")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sources = []
    for item in args.runs:
        tag, _, path = item.rpartition("=")
        sources.append((tag, Path(path)))
    runs = load_runs(sources)
    base_dir = args.base_runs or sources[0][1]
    base_path = base_dir / "base_s0" / "summary.json"
    base_final = json.loads(base_path.read_text(encoding="utf-8"))["final"] if base_path.exists() else None
    runs.pop("base", None)
    grid = [float(x) for x in args.grid.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    negative = {m.strip() for m in args.expect_negative.split(",") if m.strip()}
    ref = args.reference
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    table, per_method = [], {}
    for m in methods:
        d_vals, s_vals, s_raw, h_vals = [], [], [], []
        for seed in sorted(set(runs.get(m, {})) & set(runs.get(ref, {}))):
            cm, cu = curve(runs[m][seed], base_final), curve(runs[ref][seed], base_final)
            if len(cm) < 2 or len(cu) < 2:
                continue
            d, sv = delta(cm, cu, grid), saving(cm, cu)
            table.append([m, seed, len(cm), len(cu), d, sv])
            if d is not None:
                d_vals.append(100 * d)
            s_raw.append(sv)
            s_vals.append(0.0 if sv is None else 100 * sv)
            h_vals.append(0.0 if sv is None else sv * cu[-1][0])
        per_method[m] = (d_vals, s_vals, s_raw, h_vals)

    with (out / "compute_efficiency.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "points_method", "points_uniform", "delta_matched_compute", "saving_isotonic"])
        for row in table:
            w.writerow(row)

    lines = ["# Compute efficiency against Uniform", "",
             f"Grid (GPU-hours): {grid}. `delta` = mean accuracy difference at matched compute (percentage points); "
             "`saving` = share of Uniform's compute saved to reach Uniform's final smoothed accuracy (not reached = 0).",
             f"Rule: mean >= 2 SE and at least {args.min_positive or args.seeds_needed} of {args.seeds_needed} seeds positive. "
             f"Primary method: `{methods[0]}`.", "",
             "| Method | Expected | Seeds | delta per seed | delta mean (SE) | 95% t-interval | p | Verdict (delta) | saving per seed | saving mean |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        d_vals, s_vals, s_raw, h_vals = per_method[m]
        if not d_vals:
            continue
        s = summarise(d_vals)
        ci = f"[{s['lo']:+.2f}, {s['hi']:+.2f}]" if s["lo"] is not None else "-"
        se = f"{s['se']:.2f}" if s["se"] is not None else "-"
        p = f"{s['p']:.3f}" if s["p"] is not None else "-"
        sav = ", ".join("not reached" if x is None else f"{100 * x:+.0f}%" for x in s_raw)
        sign = -1 if m in negative else 1
        lines.append(f"| {m} | {'behind' if sign < 0 else 'ahead'} | {s['n']} | {', '.join(f'{v:+.2f}' for v in d_vals)} | {s['mean']:+.2f} ({se}) | {ci} | {p} | "
                     f"{verdict([sign * v for v in d_vals], args.seeds_needed, args.min_positive)} | {sav} | {statistics.fmean(s_vals):+.0f}% |")
    pts = sorted({r[2] for r in table} | {r[3] for r in table})
    lines += ["", f"Points per learning curve (incl. the untrained model): {pts}."]

    # descriptive (D-108 record of 2 Nov): estimated GPU-hours saved, and where the GPU-hours went
    usd = args.usd_per_gpu_hour

    def money(h: float) -> str:
        return f" (${h * usd:.2f})" if usd is not None else ""

    lines += ["", "## Estimated GPU-hours saved (descriptive)", "",
              "Every run spends its full budget, so nothing is actually returned: this is the GPU Uniform needed "
              "minus the GPU the method needed to reach Uniform's final smoothed accuracy (not reached = 0).", "",
              "| Method | Saved per seed (GPU-h) | Mean saved |", "|---|---|---|"]
    for m in methods:
        _, _, s_raw, h_vals = per_method[m]
        if h_vals:
            lines.append(f"| {m} | {', '.join(f'{h:.2f}' for h in h_vals)} | {statistics.fmean(h_vals):.2f} GPU-h"
                         f"{money(statistics.fmean(h_vals))} |")
    lines += ["", "## Where the GPU went (descriptive)", "",
              "GPU-hours charged to each environment (generation + scoring + update, `cost/attribution.py`), "
              "mean over seeds; the share is of the run's environment GPU-hours.", "",
              "| Method | Seeds | Broken reward (noisy) | Countdown (never solved) | Useful environments | Total |",
              "|---|---|---|---|---|---|"]
    for m in [ref] + methods:
        per_seed = [gpu_hours_by_env(x) for x in runs.get(m, {}).values()]
        per_seed = [x for x in per_seed if x]
        if not per_seed:
            continue

        def mean_of(keys, rows=per_seed):
            return statistics.fmean(sum(r.get(k, 0.0) for k in keys) for r in rows)

        total = statistics.fmean(sum(r.values()) for r in per_seed)
        noisy, cd = mean_of(["noisy"]), mean_of(["countdown"])
        useful = total - noisy - cd
        lines.append(f"| {m} | {len(per_seed)} | {noisy:.2f} GPU-h, {100 * noisy / total:.0f}%{money(noisy)} | "
                     f"{cd:.2f} GPU-h, {100 * cd / total:.0f}%{money(cd)} | {useful:.2f} GPU-h{money(useful)} | "
                     f"{total:.2f} GPU-h{money(total)} |")
    if usd is None:
        lines += ["", "Dollar figures appear with `--usd-per-gpu-hour <price>` (the Kaggle runs themselves were free)."]
    (out / "compute_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    primary = methods[0]
    if runs.get(primary) and runs.get(ref):
        curves = {primary: [curve(s, base_final) for s in runs[primary].values()],
                  ref: [curve(s, base_final) for s in runs[ref].values()]}
        curves = {k: [c for c in v if len(c) >= 2] for k, v in curves.items()}
        if all(curves.values()):
            fig_curves(curves, ref, out / "learning_curves_matched_compute.png", grid)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
