"""Every figure the final report needs, regenerated from the run files (no hand-typed numbers).

    python experiments/analysis/report_figures.py --runs ../curator_results_new/runs \
        --olmo-runs ../curator_results_olmo/runs --out "../Diagram_&_Documentation/figures"

After the D-103 runs, point --runs at the new download (it contains every run, old and new):
    --runs ../Curator-Nov-results/raw/main/curator_results/runs

Figures (Qwen2.5-0.5B unless named otherwise):
- accuracy_vs_uniform.png      every method against Uniform, item-paired sampled accuracy
- gain_over_untrained.png      what training bought each method (it works for all of them)
- noisy_arm_share.png          budget on the random-reward environment, seed dots
- noisy_share_by_round.png     the same over training rounds
- no_signal_groups.png         share of answer groups with no learning signal (real environments)
- cost_per_environment.png     generation-only cost (probe) against full GRPO-step cost (sysid fit)
- mixtures/*.png               the runner's per-run mixture plots (copied), Qwen and OLMo
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from curator_rl.evaluation.paired_items import paired_item_diff  # noqa: E402
from experiments.analysis.d103_confirm import (  # noqa: E402
    CONTEXT,
    GRID,
    HIGHLIGHT,
    INK,
    INK2,
    MUTED,
    SURFACE,
    _axes,
    across_seeds,
    fig_accuracy,
    fig_noisy_share,
    fig_share_by_round,
    load_runs,
    noisy_by_round,
    noisy_share,
    paired,
)

ENVS = ("gsm8k", "math35", "mbpp", "countdown", "noisy")
REAL = ("gsm8k", "math35", "mbpp", "countdown")
METHOD_ORDER = ["curator", "curator_nocal", "curator_q", "lp", "uniform", "uniform_q", "clean_uniform", "dump", "sec"]
MIXTURES = ["curator_s0", "curator_s1", "curator_s2", "sec_s0", "dump_s0", "uniform_s0", "lp_s0", "curator_q_s0"]


def ordered(labels) -> list[str]:
    labels = [x for x in labels if x not in ("dirichlet", "pilot_uniform", "base")]
    return [m for m in METHOD_ORDER if m in labels] + sorted(set(labels) - set(METHOD_ORDER))


def fig_bars(rows: list[tuple[str, list[float]]], title: str, xlabel: str, path: Path, note: str = "",
             unit: str = "%") -> None:
    plt, fig, ax = _axes(8, 0.6 * len(rows) + 1.5)
    y = list(range(len(rows)))[::-1]
    xmax = max(max(v) for _, v in rows) * 1.18
    gap = 0.02 * xmax                                    # label offset scales with the axis
    for yi, (label, vals) in zip(y, rows):
        m = statistics.fmean(vals)
        ax.barh(yi, m, height=0.42, color=HIGHLIGHT if label.startswith("curator") else CONTEXT, zorder=2)
        if len(vals) > 1:
            ax.scatter(vals, [yi] * len(vals), s=34, color=SURFACE, edgecolor=INK, linewidth=1.2, zorder=3)
        ax.text(max(vals) + gap, yi, f"{m:.1f}{unit}" if unit != "pts" else f"+{m:.1f}", va="center", ha="left",
                color=INK2, fontsize=10)
    ax.set_yticks(y, [f"{lbl}  ({len(v)} seed{'s' if len(v) != 1 else ''})" for lbl, v in rows])
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
    if note:
        fig.text(0.01, 0.01, note, color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def no_signal_share(s: dict) -> float | None:
    """Share of answer groups on the real environments whose answers all got the same reward."""
    path = Path(s["_dir"]) / "rounds.jsonl"
    if not path.exists():
        return None
    tot = mixed = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for e, a in (json.loads(line).get("arms") or {}).items():
            n, m = a.get("prompts") or 0, a.get("mixed")
            if e in REAL and n and m is not None:
                tot += n
                mixed += m * n
    return 1.0 - mixed / tot if tot else None


def probe_costs(md: Path) -> dict[str, float]:
    """Relative generation-only cost per environment from the first (0.5B) table of the cost probe report."""
    out: dict[str, float] = {}
    header = None
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("| env |"):
            if out:
                break                                   # the second table is the 1.5B model
            header = [c.strip() for c in line.strip("|").split("|")]
            continue
        if header and line.startswith("| ") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            col = next(i for i, h in enumerate(header) if h.startswith("rel. cost"))
            m = re.match(r"([\d.]+)x", cells[col])
            if m and cells[0] in ENVS:
                out[cells[0]] = float(m.group(1))
    return out


def fig_cost(probe: dict[str, float], fitted: list[dict[str, float]], path: Path) -> None:
    plt, fig, ax = _axes(8, 4.0)
    envs = [e for e in ENVS if e in probe]
    y = list(range(len(envs)))[::-1]
    h = 0.34
    for yi, e in zip(y, envs):
        ax.barh(yi + h / 2, probe[e], height=h, color=CONTEXT, zorder=2,
                label="generation only (cost probe)" if yi == y[0] else None)
        vals = [f[e] for f in fitted if e in f]
        if vals:
            ax.barh(yi - h / 2, statistics.fmean(vals), height=h, color=HIGHLIGHT, zorder=2,
                    label=f"full training step (sysid fit, {len(fitted)} runs)" if yi == y[0] else None)
            if len(vals) > 1:
                ax.scatter(vals, [yi - h / 2] * len(vals), s=28, color=SURFACE, edgecolor=INK, linewidth=1.1, zorder=3)
            ax.text(max(vals) + 0.05, yi - h / 2, f"{statistics.fmean(vals):.2f}x", va="center", color=INK2, fontsize=9.5)
        ax.text(probe[e] + 0.05, yi + h / 2, f"{probe[e]:.2f}x", va="center", color=INK2, fontsize=9.5)
    ax.set_yticks(y, envs)
    ax.axvline(1.0, color=INK2, linewidth=1, zorder=1)
    ax.set_xlim(0, max(probe.values()) * 1.2)
    ax.set_xlabel("cost per question relative to GSM8K")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Environments differ far less in real training cost", loc="left", color=INK, fontsize=13,
                 fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "A fixed 56-66 s per GRPO step dominates (fit R2 0.45-0.54). Qwen2.5-0.5B, Kaggle T4.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="the main runs folder (Qwen)")
    ap.add_argument("--olmo-runs", type=Path, default=None)
    ap.add_argument("--cost-probe", type=Path, default=REPO_ROOT / "reports" / "pilot" / "phase_e_cost_probe.md")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.out
    (out / "mixtures").mkdir(parents=True, exist_ok=True)

    runs = load_runs([("", args.runs)])
    base = next(iter(runs.pop("base", {}).values()), None)
    made = []

    # accuracy against Uniform
    rows = []
    for lbl in ordered(runs):
        if lbl == "uniform":
            continue
        ps = paired(runs, lbl, "uniform")
        if ps:
            r = across_seeds(ps)
            r["eval_se"] = statistics.fmean(se for _, se in ps.values())
            rows.append((lbl, r))
    if rows:
        fig_accuracy(rows, out / "accuracy_vs_uniform.png")
        made.append("accuracy_vs_uniform.png")

    # gain over the untrained model
    if base is not None:
        gains = []
        for lbl in ordered(runs):
            g = [100 * d[0] for s in runs[lbl].values()
                 if (d := paired_item_diff(s["final"], base["final"], "sampled")) is not None]
            if g:
                gains.append((lbl, g))
        if gains:
            fig_bars(gains, "Training works for every method", "sampled accuracy gained over the untrained model "
                     "(percentage points)", out / "gain_over_untrained.png",
                     "Item-paired against the untrained model on the same questions; dots = seeds.", unit="pts")
            made.append("gain_over_untrained.png")

    # budget on the random-reward arm
    share_rows = [(lbl, [100 * v for v in vals]) for lbl in ["sec", "dump", "uniform", "uniform_q", "lp", "curator",
                                                            "curator_nocal", "curator_q", "clean_uniform"]
                  if lbl in runs and (vals := [x for x in (noisy_share(s) for s in runs[lbl].values()) if x is not None])]
    if share_rows:
        fig_noisy_share([(lbl, [v / 100 for v in vals]) for lbl, vals in share_rows], out / "noisy_arm_share.png")
        made.append("noisy_arm_share.png")
    curves = {m: [c for c in (noisy_by_round(s) for s in runs[m].values()) if c]
              for m in ("uniform", "sec", "dump", "curator", "curator_q") if m in runs}
    if any(curves.values()):
        fig_share_by_round({m: c for m, c in curves.items() if c}, out / "noisy_share_by_round.png")
        made.append("noisy_share_by_round.png")

    # groups with no learning signal
    dead = [(lbl, vals) for lbl in ordered(runs) if lbl != "clean_uniform"
            and (vals := [100 * x for x in (no_signal_share(s) for s in runs[lbl].values()) if x is not None])]
    if dead:
        fig_bars(dead, "Groups that teach nothing (real environments)", "share of answer groups where every answer "
                 "got the same reward (%)", out / "no_signal_groups.png",
                 "GSM8K, MATH35, MBPP and Countdown only; dots = seeds.")
        made.append("no_signal_groups.png")

    # cost: generation-only probe against the full-step fit of the sysid runs
    fitted = []
    for s in runs.get("dirichlet", {}).values():
        rep = Path(s["_dir"]) / "sysid_report.json"
        if rep.exists():
            fitted.append(json.loads(rep.read_text(encoding="utf-8"))["cost_model"]["relative_cost"])
    if args.cost_probe.exists() and fitted:
        fig_cost(probe_costs(args.cost_probe), fitted, out / "cost_per_environment.png")
        made.append("cost_per_environment.png")

    # the runner's own mixture plots
    for folder, prefix in [(args.runs.parent / "plots", ""),
                           ((args.olmo_runs.parent / "plots") if args.olmo_runs else None, "olmo_")]:
        if folder is None or not folder.exists():
            continue
        for name in MIXTURES + ["score_vs_gpu_hours"]:
            src = folder / (name if name.endswith(".png") else f"{name}_mixture.png")
            if name == "score_vs_gpu_hours":
                src = folder / "score_vs_gpu_hours.png"
                if prefix != "olmo_":
                    continue                         # the Qwen one is unreadable (15 lines); OLMo's has 2
            if src.exists():
                shutil.copy2(src, out / "mixtures" / f"{prefix}{src.name}")
                made.append(f"mixtures/{prefix}{src.name}")
    print("figures:", ", ".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
