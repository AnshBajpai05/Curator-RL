"""Figures from run logs (improvement D5). Pure functions: data in, matplotlib Figure out.

matplotlib is an optional dependency (`pip install curator-rl[viz]`; Kaggle images ship it). Every
function takes plain Python data (the JSONL rows a GPU run writes, or the simulator's round logs), so
the same code draws simulator and hardware runs. `render_run_plots` writes PNGs for a directory of
GPU runs and returns the files it wrote (nothing, with a reason, when matplotlib is missing).

Colours: one categorical slot per arm, fixed across figures, so an arm keeps its colour everywhere.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

# categorical palette (colour-blind safe ordering); arms keep their slot in every figure
ARM_COLORS = {"gsm8k": "#4269d0", "math35": "#efb118", "mbpp": "#3ca951", "countdown": "#9498a0",
              "noisy": "#ff725c", "format": "#a463f2"}
_FALLBACK = ["#6cc5b0", "#ff8ab7", "#97bbf5", "#9c6b4e"]


def _color(arm: str, i: int) -> str:
    return ARM_COLORS.get(arm, _FALLBACK[i % len(_FALLBACK)])


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def score_vs_gpu_hours(curves: Mapping[str, Sequence[tuple[float, float | None]]], *, title: str = "",
                       baseline: float | None = None, ylabel: str = "macro dev accuracy"):
    """One line per method: (charged GPU-hours, macro dev score) points on the SAME items (protocol 2
    learning curves); `baseline` (the untrained model) as a dashed line."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (name, pts) in enumerate(sorted(curves.items())):
        clean = sorted((x, y) for x, y in pts if y is not None)
        if clean:
            ax.plot([p[0] for p in clean], [p[1] for p in clean], marker="o", lw=1.8, label=name,
                    color=_FALLBACK[i % len(_FALLBACK)] if name not in ARM_COLORS else ARM_COLORS[name])
    if baseline is not None:
        ax.axhline(baseline, ls="--", lw=1.2, color="#777777", label="untrained model")
    ax.set_xlabel("charged GPU-hours")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Score vs compute")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def weights_timeline(rounds: Sequence[Mapping], *, title: str = "", key: str = "delivered"):
    """Stacked shares per round (delivered by default) with a marker where an arm first enters S5."""
    plt = _plt()
    arms = sorted({e for r in rounds for e in (r.get(key) or {})})
    xs = [r["round"] for r in rounds]
    ys = [[float((r.get(key) or {}).get(e, 0.0)) for r in rounds] for e in arms]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if xs:
        ax.stackplot(xs, ys, labels=arms, colors=[_color(e, i) for i, e in enumerate(arms)], alpha=0.9)
    for i, e in enumerate(arms):
        first = next((r["round"] for r in rounds if str((r.get("statuses") or {}).get(e, "")).startswith("S5")), None)
        if first is not None:
            ax.axvline(first, color=_color(e, i), ls="--", lw=1.2)
            ax.annotate(f"{e}: S5", (first, 1.0), xytext=(2, -12), textcoords="offset points", fontsize=8,
                        color=_color(e, i))
    ax.set_ylim(0, 1)
    ax.set_xlabel("round")
    ax.set_ylabel(f"{key} share")
    ax.set_title(title or "Mixture over the run")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def roi_vs_loo(roi: Mapping[str, float], loo: Mapping[str, float], *, title: str = ""):
    """Scatter of the in-run ROI estimate against leave-one-out contribution, one labelled point per arm."""
    plt = _plt()
    arms = sorted(set(roi) & set(loo))
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for i, e in enumerate(arms):
        if math.isfinite(roi[e]) and math.isfinite(loo[e]):
            ax.scatter(100 * roi[e], loo[e], s=48, color=_color(e, i))
            ax.annotate(e, (100 * roi[e], loo[e]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("ROI (benchmark points per GPU-hour)")
    ax.set_ylabel("leave-one-out contribution")
    ax.set_title(title or "ROI vs leave-one-out")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def render_run_plots(root: Path) -> tuple[list[Path], str]:
    """PNG figures for every run under `root` (GPU layout); returns (files, note)."""
    try:
        _plt()
    except ImportError:
        return [], "matplotlib not installed: no figures (pip install matplotlib)"
    root = Path(root)
    out = root / "plots"
    out.mkdir(exist_ok=True)
    files: list[Path] = []
    from curator_rl.evaluation.paired_items import learning_curve

    base_dir = next(iter(sorted(root.glob("base_s*"))), None)          # the untrained model (one run)
    base_reports = _rows(base_dir / "reports.jsonl") if base_dir is not None else []
    base_final = next((r for r in reversed(base_reports) if r.get("tag") == "final" and r.get("items")), None)
    curves: dict[str, list[tuple[float, float | None]]] = {}      # protocol 2: sampled, on the same items
    finals: dict[str, list[tuple[float, float | None]]] = {}      # protocol 1: one greedy final point each
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p / "rounds.jsonl").exists()):
        rounds = _rows(run / "rounds.jsonl")
        if rounds:
            fig = weights_timeline(rounds, title=run.name)
            files.append(out / f"{run.name}_mixture.png")
            fig.savefig(files[-1], dpi=130)
            _plt().close(fig)
        if run == base_dir:
            continue
        reports = _rows(run / "reports.jsonl")
        cfg = _json(run / "run_config.json")
        curve = learning_curve(reports, n_subset=int(cfg.get("report_items", 100)), base_final=base_final)
        if len(curve) >= 2:
            curves[run.name] = list(curve)
        final = next((r for r in reversed(reports) if r.get("tag") == "final"), None)
        if final is not None:
            finals[run.name] = [(final["charged_gpu_hours"], final.get("macro"))]
    if curves or finals:
        if curves:
            first = next(iter(curves.values()))
            fig = score_vs_gpu_hours(curves, title="Sampled accuracy on the same items (learning curves)",
                                     baseline=first[0][1] if base_final is not None else None,
                                     ylabel="macro sampled accuracy")
        else:
            # older runs: final scores only (a periodic report was on other items, not comparable)
            fig = score_vs_gpu_hours(finals, title="Final greedy score (evaluation protocol 1)")
        files.append(out / "score_vs_gpu_hours.png")
        fig.savefig(files[-1], dpi=130)
        _plt().close(fig)
    return files, f"{len(files)} figures in {out}"


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
