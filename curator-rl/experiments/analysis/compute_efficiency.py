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
from curator_rl.evaluation.paired_items import learning_curve, paired_item_diff  # noqa: E402
from experiments.analysis.d103_confirm import (  # noqa: E402
    CONTEXT,
    GRID,
    HIGHLIGHT,
    INK,
    INK2,
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
        ax.plot([0.0, *g], mean, color=color, linewidth=2.4, zorder=3, label=f"{name} (mean of {len(cs)} runs)")
    ax.set_xlabel("charged GPU-hours")
    ax.set_ylabel("sampled accuracy, macro (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Accuracy at matched compute", loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "Thin lines: single runs. Same 100 items per domain at every point; 0 GPU-h = untrained model.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_forest(rows: list[tuple[str, list[float]]], title: str, xlabel: str, note: str, path: Path) -> None:
    """Mean and 95% t-interval per method (filled), one dot per block; Curator variants highlighted."""
    plt, fig, ax = _axes(8, 0.75 * len(rows) + 1.7)
    y = list(range(len(rows)))[::-1]
    lo_all, hi_all = 0.0, 0.0
    for yi, (label, vals) in zip(y, rows):
        color = HIGHLIGHT if label.startswith("curator") else CONTEXT
        st = summarise(vals)
        if st["lo"] is not None:
            ax.plot([st["lo"], st["hi"]], [yi, yi], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            lo_all, hi_all = min(lo_all, st["lo"]), max(hi_all, st["hi"])
        ax.scatter(vals, [yi + 0.22] * len(vals), s=22, color=SURFACE, edgecolor=color, linewidth=1.2, zorder=3)
        ax.scatter([st["mean"]], [yi], s=70, color=color, zorder=4)
        ax.text((st["hi"] if st["hi"] is not None else st["mean"]) + 0.12, yi, f"{st['mean']:+.2f}", va="center",
                ha="left", color=INK2, fontsize=10)
        lo_all, hi_all = min(lo_all, min(vals)), max(hi_all, max(vals))
    ax.set_yticks(y, [f"{lbl}  ({len(v)} blocks)" for lbl, v in rows])
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    ax.set_xlim(lo_all - 0.5, hi_all + 0.9)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.text(0.06, -0.62, "Uniform", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
    ap.add_argument("--no-verdict", action="store_true", help="descriptive run: print no rule verdict")
    ap.add_argument("--keep-blocks", default="", help="only these TAG:SEED blocks, e.g. a:3,b:4 (default: all)")
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
    tags = list(dict.fromkeys(t for t, _ in sources))          # a block = (account tag, seed)
    keep = None
    if args.keep_blocks:
        keep = {(t, int(sd)) for t, _, sd in (b.partition(":") for b in args.keep_blocks.split(","))}

    def lab(method: str, tag: str) -> str:
        return f"{method}[{tag}]" if tag else method

    def blocks_of(method: str):
        """(tag, seed, method run, Uniform run) paired inside one account, finished runs only."""
        for tag in tags:
            mr, ur = runs.get(lab(method, tag), {}), runs.get(lab(ref, tag), {})
            for seed in sorted(set(mr) & set(ur)):
                if keep is None or (tag, seed) in keep:
                    yield tag, seed, mr[seed], ur[seed]

    def name(tag: str, seed: int) -> str:
        return f"{tag}:s{seed}" if tag else f"s{seed}"

    table, per_method = [], {}
    for m in methods:
        rows = []
        for tag, seed, rm, ru in blocks_of(m):
            cm, cu = curve(rm, base_final), curve(ru, base_final)
            if len(cm) < 2 or len(cu) < 2:
                continue
            d, sv = delta(cm, cu, grid), saving(cm, cu)
            fin = paired_item_diff(rm["final"], ru["final"], "sampled")
            table.append([m, name(tag, seed), len(cm), len(cu), d, sv, None if fin is None else fin[0]])
            rows.append({"block": name(tag, seed), "delta": None if d is None else 100 * d, "saving": sv,
                         "hours": 0.0 if sv is None else sv * cu[-1][0],
                         "final": None if fin is None else 100 * fin[0]})
        per_method[m] = rows

    with (out / "compute_efficiency.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "block", "points_method", "points_uniform", "delta_matched_compute", "saving_isotonic",
                    "final_sampled_diff"])
        for row in table:
            w.writerow(row)

    need = args.seeds_needed
    lines = ["# Compute efficiency against Uniform", "",
             f"Grid (GPU-hours): {grid}. `delta` = mean accuracy difference at matched compute (percentage points); "
             "`saving` = share of Uniform's compute saved to reach Uniform's final smoothed accuracy (not reached = 0). "
             "A block is one seed on one account; every method is paired with Uniform of the same block.",
             f"Rule: mean >= 2 SE and at least {args.min_positive or need} of {need} blocks positive "
             f"(in the expected direction). Primary method: `{methods[0]}`."
             + (f" Blocks kept: {sorted(keep)}." if keep else ""), "",
             "| Method | Expected | Blocks | delta per block | delta mean (SE) | 95% t-interval | p | Verdict (delta) | "
             "saving per block | saving mean |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        rows = [r for r in per_method[m] if r["delta"] is not None]
        if not rows:
            lines.append(f"| {m} | - | 0 | - | - | - | - | PENDING (0 of {need}) | - | - |")
            continue
        d_vals = [r["delta"] for r in rows]
        st = summarise(d_vals)
        ci = f"[{st['lo']:+.2f}, {st['hi']:+.2f}]" if st["lo"] is not None else "-"
        se = f"{st['se']:.2f}" if st["se"] is not None else "-"
        p = f"{st['p']:.3f}" if st["p"] is not None else "-"
        sav = ", ".join("not reached" if r["saving"] is None else f"{100 * r['saving']:+.0f}%" for r in rows)
        sign = -1 if m in negative else 1
        per = ", ".join(f"{r['block']} {r['delta']:+.2f}" for r in rows)
        lines.append(f"| {m} | {'behind' if sign < 0 else 'ahead'} | {st['n']} | {per} | {st['mean']:+.2f} ({se}) | "
                     f"{ci} | {p} | {'descriptive' if args.no_verdict else verdict([sign * v for v in d_vals], need, args.min_positive)} | {sav} | "
                     f"{statistics.fmean(0.0 if r['saving'] is None else 100 * r['saving'] for r in rows):+.0f}% |")
    pts = sorted({r[2] for r in table} | {r[3] for r in table})
    lines += ["", f"Points per learning curve (incl. the untrained model): {pts}."]

    # descriptive: final accuracy on the same blocks (paired items, sampled)
    lines += ["", "## Final accuracy against Uniform (descriptive)", "",
              "Item-paired sampled macro at the end of each run, same block; not part of any rule.", "",
              "| Method | Blocks | per block | mean (SE) | 95% t-interval | blocks above Uniform |", "|---|---|---|---|---|---|"]
    for m in methods:
        rows = [r for r in per_method[m] if r["final"] is not None]
        if not rows:
            continue
        f_vals = [r["final"] for r in rows]
        st = summarise(f_vals)
        ci = f"[{st['lo']:+.2f}, {st['hi']:+.2f}]" if st["lo"] is not None else "-"
        se = f"{st['se']:.2f}" if st["se"] is not None else "-"
        per_final = ", ".join("{} {:+.2f}".format(r["block"], r["final"]) for r in rows)
        lines.append(f"| {m} | {len(rows)} | {per_final} | "
                     f"{st['mean']:+.2f} ({se}) | {ci} | {sum(v > 0 for v in f_vals)} of {len(f_vals)} |")

    # descriptive (D-108 record of 2 Nov): estimated GPU-hours saved, and where the GPU-hours went
    usd = args.usd_per_gpu_hour

    def money(h: float) -> str:
        return f" (${h * usd:.2f})" if usd is not None else ""

    lines += ["", "## Estimated GPU-hours saved (descriptive)", "",
              "Every run spends its full budget, so nothing is actually returned: this is the GPU Uniform needed "
              "minus the GPU the method needed to reach Uniform's final smoothed accuracy (not reached = 0).", "",
              "| Method | Saved per block (GPU-h) | Mean saved |", "|---|---|---|"]
    for m in methods:
        rows = per_method[m]
        if rows:
            per_hours = ", ".join("{} {:.2f}".format(r["block"], r["hours"]) for r in rows)
            mean_h = statistics.fmean(r["hours"] for r in rows)
            lines.append(f"| {m} | {per_hours} | "
                         f"{mean_h:.2f} GPU-h{money(mean_h)} |")
    lines += ["", "## Where the GPU went (descriptive)", "",
              "GPU-hours charged to each environment (generation + scoring + update, `cost/attribution.py`), "
              "mean over finished runs; the share is of the run's environment GPU-hours.", "",
              "| Method | Runs | Broken reward (noisy) | Countdown (never solved) | Useful environments | Total |",
              "|---|---|---|---|---|---|"]
    for m in [ref] + methods:
        per_run = [gpu_hours_by_env(x) for tag in tags for seed, x in runs.get(lab(m, tag), {}).items()
                   if keep is None or (tag, seed) in keep]
        per_run = [x for x in per_run if x]
        if not per_run:
            continue

        def mean_of(keys, rows=per_run):
            return statistics.fmean(sum(r.get(k, 0.0) for k in keys) for r in rows)

        total = statistics.fmean(sum(r.values()) for r in per_run)
        noisy, cd = mean_of(["noisy"]), mean_of(["countdown"])
        useful = total - noisy - cd
        lines.append(f"| {m} | {len(per_run)} | {noisy:.2f} GPU-h, {100 * noisy / total:.0f}%{money(noisy)} | "
                     f"{cd:.2f} GPU-h, {100 * cd / total:.0f}%{money(cd)} | {useful:.2f} GPU-h{money(useful)} | "
                     f"{total:.2f} GPU-h{money(total)} |")
    if usd is None:
        lines += ["", "Dollar figures appear with `--usd-per-gpu-hour <price>` (the Kaggle runs themselves were free)."]

    # descriptive: the reward checks in these runs
    lines += ["", "## Reward checks (descriptive)", "",
              "| Method | Run | Environments ever in S5 | First S5 round of the broken reward |", "|---|---|---|---|"]
    for m in [x for x in methods if x.startswith("curator")]:
        for tag in tags:
            for seed, x in sorted(runs.get(lab(m, tag), {}).items()):
                if keep is not None and (tag, seed) not in keep:
                    continue
                first = None
                path = Path(x["_dir"]) / "rounds.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
                    if line.strip():
                        row = json.loads(line)
                        if first is None and "S5" in str((row.get("statuses") or {}).get("noisy", "")):
                            first = row["round"]
                lines.append(f"| {m} | {name(tag, seed)} | {', '.join(x.get('ever_s5') or []) or 'none'} | {first} |")
    (out / "compute_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    primary = methods[0]
    curves = {primary: [], ref: []}
    for _tag, _seed, rm, ru in blocks_of(primary):
        curves[primary].append(curve(rm, base_final))
        curves[ref].append(curve(ru, base_final))
    curves = {k: [c for c in v if len(c) >= 2] for k, v in curves.items()}
    if all(curves.values()):
        fig_curves(curves, ref, out / "learning_curves_matched_compute.png", grid)
    note = ("Dots: blocks (one seed on one account). Bars: 95% t-interval across blocks"
            + (", which treats same-seed replicates as independent (optimistic)." if len(tags) > 1 and keep is None else "."))
    rows_d = [(m, [r["delta"] for r in per_method[m] if r["delta"] is not None]) for m in methods]
    rows_d = [(m, v) for m, v in rows_d if len(v) >= 2]
    if rows_d:
        fig_forest(rows_d, "Accuracy at matched compute against Uniform",
                   "mean accuracy difference over 0.25-1.5 GPU-hours (percentage points)", note,
                   out / "matched_compute_delta.png")
    rows_f = [(m, [r["final"] for r in per_method[m] if r["final"] is not None]) for m in methods]
    rows_f = [(m, v) for m, v in rows_f if len(v) >= 2]
    if rows_f:
        fig_forest(rows_f, "Final accuracy against Uniform", "sampled accuracy minus Uniform at the end, same "
                   "questions (percentage points)", note, out / "final_accuracy_vs_uniform.png")
    from experiments.analysis.report_figures import fig_gpu_split

    split = []
    for m in [ref] + methods:
        per_run = [gpu_hours_by_env(x) for tag in tags for seed, x in runs.get(lab(m, tag), {}).items()
                   if keep is None or (tag, seed) in keep]
        if any(per_run):
            split.append((m, [x for x in per_run if x]))
    if split:
        fig_gpu_split(split, out / "gpu_split_by_environment.png")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
