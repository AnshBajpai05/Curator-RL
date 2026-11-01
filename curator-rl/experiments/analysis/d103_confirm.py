"""D-103 confirmation analysis: the pre-registered rules R1-R4, descriptive checks, tables and figures.

Written on 2026-11-01, before any `confirm`, `lite`, `clean` or `sec_tau1` run (DECISIONS.md D-103 and its
record of 1 Nov). It applies the rules exactly as D-103 states them:

- primary measure: sampled macro accuracy, item-paired against the comparison method, per seed; across
  seeds the mean of the per-seed differences, its SE (sd / sqrt(n)) and the 95% t-interval (df n - 1);
- R1 (primary): curator_q > uniform_q. Holds if all 3 seeds have the same sign and the mean is >= 2 SE;
- R2: SEC < Uniform and DUMP < Uniform, each by the same rule;
- R3 (mechanism, descriptive): SEC and DUMP spend more than Uniform's 20% of questions on the
  random-reward arm, the Curator variants under 13%;
- R4: Curator-lite (curator_nocal) vs Curator: the simplification holds if the difference is within
  +-1 point (the mean paired difference; the t-interval is reported next to it).
Greedy accuracy, pass@k and the random-reward share are reported for every run. clean_uniform and the
declared SEC check at the authors' default temperature (namespace `sec_tau1`) are descriptive only.

    python experiments/analysis/d103_confirm.py --raw ../Curator-Nov-results/raw --out ../Curator-Nov-results/analysis

`--raw` holds one folder per downloaded results zip: `main/` (the default namespace) and `<tag>/` for a
namespaced step (e.g. `sec_tau1/`); each contains `runs/<name>_s<seed>/summary.json`. Runs from a tagged
folder are labelled `<method>[<tag>]`. `--runs PATH` or `--runs TAG=PATH` adds a runs folder directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from curator_rl.core.stats import paired_t_interval, sign_flip_pvalue  # noqa: E402
from curator_rl.evaluation.paired_items import paired_item_diff, pass_at_k  # noqa: E402

NOISY = "noisy"
SEEDS = (0, 1, 2)
CURATOR_VARIANTS = ("curator", "curator_nocal", "curator_q")
R3_HIGH, R3_LOW, R4_MARGIN = 0.20, 0.13, 0.01

# chart tokens (dataviz reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
HIGHLIGHT, CONTEXT = "#2a78d6", "#898781"
SERIES = {"curator": "#2a78d6", "sec": "#eb6834", "dump": "#1baf7a", "curator_q": "#eda100"}


# ------------------------------------------------------------------------------------------------ loading

def load_runs(sources: list[tuple[str, Path]]) -> dict[str, dict[int, dict]]:
    """{label: {seed: summary}}; each summary also carries `_dir` (its run folder)."""
    runs: dict[str, dict[int, dict]] = {}
    for tag, folder in sources:
        for path in sorted(Path(folder).glob("*/summary.json")):
            s = json.loads(path.read_text(encoding="utf-8"))
            label = s.get("label") or s["method"]
            if tag:
                label = f"{label}[{tag}]"
            s["_dir"] = str(path.parent)
            runs.setdefault(label, {})[int(s["seed"])] = s
    return runs


def discover(raw: Path) -> list[tuple[str, Path]]:
    """One runs folder per subfolder of `raw`: `main/` untagged, any other name as the tag.

    The runner's zip unpacks to `curator_results/runs/...`, so the `runs` folder is found at any depth.
    """
    out = []
    for sub in sorted(p for p in raw.iterdir() if p.is_dir()):
        candidates = [sub] + sorted(p for p in sub.rglob("runs") if p.is_dir())
        runs_dir = next((c for c in candidates if any(c.glob("*/summary.json"))), None)
        if runs_dir is not None:
            out.append(("" if sub.name == "main" else sub.name, runs_dir))
    return out


def model_of(s: dict) -> str:
    return (s.get("run_config") or {}).get("model", "?")


def sampled_macro(s: dict) -> float | None:
    return ((s.get("final") or {}).get("sampled") or {}).get("macro")


def noisy_share(s: dict) -> float | None:
    return (s.get("mean_delivered_share") or {}).get(NOISY)


def noisy_by_round(s: dict) -> list[tuple[int, float]]:
    path = Path(s["_dir"]) / "rounds.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        delivered = row.get("delivered") or {}
        total = sum(delivered.values())
        if total > 0 and NOISY in delivered:
            out.append((int(row["round"]), delivered[NOISY] / total))
    return out


# -------------------------------------------------------------------------------------------- statistics

def paired(runs, a: str, b: str, kind: str = "sampled") -> dict[int, tuple[float, float]]:
    """{seed: (a - b, evaluation SE)} on the items both evaluated, same seed and same model only."""
    out = {}
    for seed in sorted(set(runs.get(a, {})) & set(runs.get(b, {}))):
        sa, sb = runs[a][seed], runs[b][seed]
        if model_of(sa) != model_of(sb):
            continue
        d = paired_item_diff(sa["final"], sb["final"], kind)
        if d is not None:
            out[seed] = (d[0], d[1])
    return out


def across_seeds(diffs: dict[int, tuple[float, float]]) -> dict:
    vals = [diffs[s][0] for s in sorted(diffs)]
    row = {"seeds": sorted(diffs), "n": len(vals), "diffs": vals,
           "mean": statistics.fmean(vals) if vals else None, "se": None, "lo": None, "hi": None, "p": None}
    if len(vals) >= 2:
        row["se"] = statistics.stdev(vals) / math.sqrt(len(vals))
        _, row["lo"], row["hi"] = paired_t_interval(vals)
        row["p"] = sign_flip_pvalue(vals)
    return row


def directional_rule(row: dict, sign: int) -> tuple[str, str]:
    """D-103 rule: all 3 seeds share the hypothesised sign and |mean| >= 2 SE. (verdict, reason)."""
    missing = [s for s in SEEDS if s not in row["seeds"]]
    if missing:
        return "PENDING", f"needs seeds {list(SEEDS)}; missing {missing}"
    same = all(sign * d > 0 for d in row["diffs"])
    big = row["se"] is not None and sign * row["mean"] >= 2 * row["se"]
    if same and big:
        return "HOLDS", f"all 3 seeds {'>' if sign > 0 else '<'} 0 and |mean| = {abs(row['mean']) / row['se']:.1f} SE"
    reasons = []
    if not same:
        reasons.append("seeds disagree in sign")
    if not big:
        reasons.append(f"|mean| = {abs(row['mean']) / row['se']:.1f} SE < 2" if row["se"] else "no SE")
    return "FAILS", "; ".join(reasons)


def pts(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{100 * x:+.{digits}f}"


# ------------------------------------------------------------------------------------------------ figures

def _axes(fig_w: float, fig_h: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return plt, fig, ax


def fig_accuracy(rows: list[tuple[str, dict]], path: Path) -> None:
    plt, fig, ax = _axes(8, 0.62 * len(rows) + 1.6)
    y = list(range(len(rows)))[::-1]
    for yi, (label, r) in zip(y, rows):
        color = HIGHLIGHT if label.startswith("curator") else CONTEXT
        mean = 100 * r["mean"]
        if r["n"] >= 2:
            lo, hi = 100 * r["lo"], 100 * r["hi"]
        else:
            se = 100 * r["eval_se"]
            lo, hi = mean - 1.96 * se, mean + 1.96 * se
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([mean], [yi], s=70, color=color if r["n"] >= 2 else SURFACE, edgecolor=color, linewidth=2, zorder=3)
        ax.text(hi + 0.12, yi, f"{mean:+.1f}", va="center", ha="left", color=INK2, fontsize=10)
    ax.set_yticks(y, [f"{lbl}  ({r['n']} seed{'s' if r['n'] != 1 else ''})" for lbl, r in rows])
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    ax.set_ylim(-0.7, len(rows) - 0.5)
    ax.text(0.08, -0.62, "Uniform", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel("sampled accuracy minus Uniform, same questions (percentage points)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Accuracy against Uniform", loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "Filled: 95% t-interval across training seeds.  Hollow: 1 seed, evaluation noise only "
             "(too narrow).", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_noisy_share(rows: list[tuple[str, list[float]]], path: Path) -> None:
    plt, fig, ax = _axes(8, 0.62 * len(rows) + 1.4)
    y = list(range(len(rows)))[::-1]
    for yi, (label, vals) in zip(y, rows):
        v = [100 * x for x in vals]
        m = statistics.fmean(v)
        ax.barh(yi, m, height=0.42, color=HIGHLIGHT if label.startswith("curator") else CONTEXT, zorder=2)
        if len(v) > 1:
            ax.scatter(v, [yi] * len(v), s=34, color=SURFACE, edgecolor=INK, linewidth=1.2, zorder=3)
        txt = f"{m:.0f}%" + (f"  (seeds {', '.join(f'{x:.0f}' for x in v)})" if len(v) > 1 else "")
        ax.text(max(v) + 1.0, yi, txt, va="center", ha="left", color=INK2, fontsize=10)
    ax.set_yticks(y, [f"{lbl}  ({len(v)} seed{'s' if len(v) != 1 else ''})" for lbl, v in rows])
    ax.set_xlim(0, max(55.0, max(100 * max(v) for _, v in rows) + 18))
    ax.set_xlabel("share of training questions on the random-reward environment (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Budget spent on a reward that ignores the question", loc="left", color=INK, fontsize=13,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_share_by_round(curves: dict[str, list[list[tuple[int, float]]]], path: Path) -> None:
    plt, fig, ax = _axes(8, 4.2)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", length=3)
    order = ["uniform", "sec", "dump", "curator", "curator_q"]
    ends = []
    for name in [n for n in order if n in curves]:
        per_round: dict[int, list[float]] = {}
        for curve in curves[name]:
            for r, v in curve:
                per_round.setdefault(r, []).append(100 * v)
        rounds = sorted(r for r, vs in per_round.items() if len(vs) == len(curves[name]))
        if not rounds:
            continue
        mean = [statistics.fmean(per_round[r]) for r in rounds]
        color = SERIES.get(name, CONTEXT)
        ax.plot(rounds, mean, color=color, linewidth=2, zorder=3 if name != "uniform" else 2,
                label=f"{name} ({len(curves[name])} seed{'s' if len(curves[name]) != 1 else ''})")
        ends.append([mean[-1], name, len(curves[name]), rounds[-1]])
    # end labels in one column right of the longest line, nudged apart so none overlap
    x_lab = max(e[3] for e in ends) + 1.2
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 3.2)
    for y_lab, name, _, _ in ends:
        ax.text(x_lab, y_lab, name, color=INK2, fontsize=9.5, va="center")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0.0, 1.0))
    ax.set_xlabel("training round")
    ax.set_ylabel("share on the random-reward arm (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    top = max(e[0] for e in ends) if ends else 50.0
    ax.set_ylim(0, max(top + 14.0, 30.0))          # headroom for the legend
    ax.set_xlim(0, x_lab + 6.0)
    ax.set_title("Who gets pulled into the broken reward, round by round", loc="left", color=INK, fontsize=13,
                 fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "Mean over seeds, rounds every seed reached; (n) = seeds. Rounds 1-6 are the shared "
             "equal warm-up.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 0.93, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_rule_seeds(rules: list[tuple[str, dict, int]], path: Path) -> None:
    plt, fig, ax = _axes(8, 0.75 * len(rules) + 1.4)
    y = list(range(len(rules)))[::-1]
    for yi, (_title, row, sign) in zip(y, rules):
        for d in row["diffs"]:
            ax.scatter([100 * d], [yi], s=60, color=HIGHLIGHT if sign * d > 0 else CONTEXT,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if row["se"] is not None:
            m, se = 100 * row["mean"], 100 * row["se"]
            ax.plot([m - 2 * se, m + 2 * se], [yi - 0.22, yi - 0.22], color=INK2, linewidth=1.5, zorder=2)
            ax.scatter([m], [yi - 0.22], marker="|", s=120, color=INK, zorder=3)
    ax.set_yticks(y, [t for t, _, _ in rules])
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    ax.set_xlabel("per-seed paired difference (percentage points); bar below = mean +- 2 SE")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("D-103 rules, seed by seed", loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "Blue dot: the seed points the way the rule predicts. A rule holds when all 3 seeds are "
             "blue and the bar clears 0.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------- report

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=None, help="folder with main/ and <tag>/ results folders")
    ap.add_argument("--runs", action="append", default=[], help="PATH or TAG=PATH of a runs folder")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sources = discover(args.raw) if args.raw else []
    for item in args.runs:
        tag, _, path = item.rpartition("=")
        sources.append((tag, Path(path)))
    if not sources:
        ap.error("no runs found: give --raw or --runs")
    runs = load_runs(sources)
    base = next(iter(runs.pop("base", {}).values()), None)
    qwen = {lbl: by for lbl, by in runs.items()
            if all(model_of(s).startswith("Qwen/Qwen2.5-0.5B") for s in by.values())}
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # per-seed paired differences against Uniform (every Qwen method)
    vs_uniform: dict[str, dict] = {}
    with (out / "tables" / "paired_vs_uniform.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "sampled_diff", "sampled_se", "greedy_diff", "greedy_se"])
        for label in sorted(qwen):
            if label == "uniform":
                continue
            ps, pg = paired(qwen, label, "uniform"), paired(qwen, label, "uniform", "greedy")
            for seed in sorted(ps):
                w.writerow([label, seed, f"{ps[seed][0]:.5f}", f"{ps[seed][1]:.5f}",
                            f"{pg[seed][0]:.5f}" if seed in pg else "", f"{pg[seed][1]:.5f}" if seed in pg else ""])
            if ps:
                row = across_seeds(ps)
                row["eval_se"] = statistics.fmean(se for _, se in ps.values())
                row["greedy"] = across_seeds(pg) if pg else None
                vs_uniform[label] = row

    # the four rules
    r1 = across_seeds(paired(qwen, "curator_q", "uniform_q"))
    r2s = across_seeds(paired(qwen, "sec", "uniform"))
    r2d = across_seeds(paired(qwen, "dump", "uniform"))
    r4 = across_seeds(paired(qwen, "curator_nocal", "curator"))
    v1, why1 = directional_rule(r1, +1)
    v2s, why2s = directional_rule(r2s, -1)
    v2d, why2d = directional_rule(r2d, -1)
    v2 = "HOLDS" if v2s == v2d == "HOLDS" else ("PENDING" if "PENDING" in (v2s, v2d) else "FAILS")

    shares = {lbl: {s: noisy_share(x) for s, x in by.items() if noisy_share(x) is not None} for lbl, by in qwen.items()}
    shares = {k: v for k, v in shares.items() if v}

    def mean_share(lbl: str) -> float | None:
        return statistics.fmean(shares[lbl].values()) if lbl in shares else None

    hi_ok = [mean_share(m) is not None and mean_share(m) > R3_HIGH for m in ("sec", "dump")]
    lo_ok = [mean_share(m) is not None and mean_share(m) < R3_LOW for m in CURATOR_VARIANTS if m in shares]
    v3 = "HOLDS" if all(hi_ok) and lo_ok and all(lo_ok) else "FAILS"
    if any(len(shares.get(m, {})) < 3 for m in ("sec", "dump", *CURATOR_VARIANTS)):
        v3 += " (provisional: some methods have fewer than 3 seeds)"
    missing4 = [s for s in SEEDS if s not in r4["seeds"]]
    if missing4:
        v4, why4 = "PENDING", f"needs seeds {list(SEEDS)}; missing {missing4}"
    else:
        inside = abs(r4["mean"]) <= R4_MARGIN
        v4 = "HOLDS" if inside else "FAILS"
        why4 = (f"mean {pts(r4['mean'])} points ({'within' if inside else 'outside'} +-1); t-interval "
                f"[{pts(r4['lo'])}, {pts(r4['hi'])}] {'inside' if r4['lo'] > -R4_MARGIN and r4['hi'] < R4_MARGIN else 'not inside'} +-1")

    with (out / "tables" / "rules.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "comparison", "seeds", "mean", "se", "t_lo", "t_hi", "sign_flip_p", "verdict", "reason"])
        for rule, comp, row, verdict, why in (("R1", "curator_q - uniform_q", r1, v1, why1),
                                              ("R2", "sec - uniform", r2s, v2s, why2s),
                                              ("R2", "dump - uniform", r2d, v2d, why2d),
                                              ("R4", "curator_nocal - curator", r4, v4, why4)):
            w.writerow([rule, comp, " ".join(map(str, row["seeds"])), row["mean"], row["se"], row["lo"], row["hi"],
                        row["p"], verdict, why])
    with (out / "tables" / "noisy_share.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "noisy_share"])
        for lbl in sorted(shares):
            for seed, v in sorted(shares[lbl].items()):
                w.writerow([lbl, seed, f"{v:.4f}"])

    # figures
    order = ["curator", "curator_nocal", "curator_q", "lp", "uniform_q", "dump", "sec"]
    acc_rows = [(lbl, vs_uniform[lbl]) for lbl in order + sorted(set(vs_uniform) - set(order)) if lbl in vs_uniform
                and lbl not in ("dirichlet", "pilot_uniform")]
    if acc_rows:
        fig_accuracy(acc_rows, out / "figures" / "accuracy_vs_uniform.png")
    share_order = ["sec", "sec[sec_tau1]", "dump", "uniform", "uniform_q", "lp", "curator", "curator_nocal", "curator_q"]
    share_rows = [(lbl, list(shares[lbl].values())) for lbl in share_order if lbl in shares]
    if share_rows:
        fig_noisy_share(share_rows, out / "figures" / "noisy_arm_share.png")
    curves = {m: [noisy_by_round(s) for s in qwen[m].values()] for m in ("uniform", "sec", "dump", "curator", "curator_q")
              if m in qwen}
    curves = {m: [c for c in cs if c] for m, cs in curves.items()}
    if any(curves.values()):
        fig_share_by_round({m: cs for m, cs in curves.items() if cs}, out / "figures" / "noisy_share_by_round.png")
    fig_rule_seeds([("R1  curator_q - uniform_q", r1, +1), ("R2  sec - uniform", r2s, -1),
                    ("R2  dump - uniform", r2d, -1)], out / "figures" / "rules_per_seed.png")

    # the report
    def rule_line(name, comp, row, verdict, why):
        ci = f"[{pts(row['lo'])}, {pts(row['hi'])}]" if row["lo"] is not None else "-"
        p = f"{row['p']:.3f}" if row["p"] is not None else "-"
        per = ", ".join(f"s{s} {pts(d)}" for s, d in zip(row["seeds"], row["diffs"])) or "-"
        se = pts(row["se"]).lstrip("+") if row["se"] is not None else "-"
        return f"| {name} | {comp} | {per} | {pts(row['mean'])} | {se} | {ci} | {p} | **{verdict}** | {why} |"

    lines = [
        "# D-103 confirmation: results",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `experiments/analysis/d103_confirm.py` "
        "(written before the runs). Model: Qwen2.5-0.5B-Instruct. Primary measure: sampled macro accuracy, "
        "item-paired, in percentage points. Sources: " + ", ".join(f"`{p}`" + (f" [{t}]" if t else "")
                                                                for t, p in sources) + ".",
        "",
        "## Pre-registered rules",
        "",
        "| Rule | Comparison | Per seed | Mean | SE | 95% t-interval | Sign-flip p | Verdict | Why |",
        "|---|---|---|---|---|---|---|---|---|",
        rule_line("R1 (primary)", "curator_q - uniform_q", r1, v1, why1),
        rule_line("R2", "sec - uniform", r2s, v2s, why2s),
        rule_line("R2", "dump - uniform", r2d, v2d, why2d),
        rule_line("R4", "curator_nocal - curator", r4, v4, why4),
        "",
        f"**R2 overall: {v2}** (needs both SEC and DUMP).",
        "",
        f"**R3 (mechanism, descriptive): {v3}.** Share of training questions on the random-reward arm "
        f"(SEC and DUMP above {100 * R3_HIGH:.0f}%, Curator variants under {100 * R3_LOW:.0f}%):",
        "",
        "| Method | Seeds | Per seed | Mean |",
        "|---|---|---|---|",
    ]
    for lbl in [m for m in share_order if m in shares] + sorted(set(shares) - set(share_order)):
        per = ", ".join(f"s{s} {100 * v:.1f}%" for s, v in sorted(shares[lbl].items()))
        lines.append(f"| {lbl} | {len(shares[lbl])} | {per} | {100 * mean_share(lbl):.1f}% |")
    lines += ["", "With 3 seeds, \"mean >= 2 SE\" is evidence at roughly one-sided p 0.1, not proof (D-103).", "",
              "## Every method against Uniform", "",
              "| Method | Seeds | Sampled diff (mean) | 95% t-interval | Greedy diff (mean) | Random-reward share | "
              "Gain over untrained (sampled) | pass@k |", "|---|---|---|---|---|---|---|---|"]
    for lbl, row in sorted(vs_uniform.items()):
        g = row.get("greedy")
        ci = f"[{pts(row['lo'])}, {pts(row['hi'])}]" if row["lo"] is not None else f"one seed (eval SE {100 * row['eval_se']:.2f})"
        gains = []
        if base is not None:
            for s in qwen[lbl].values():
                d = paired_item_diff(s["final"], base["final"], "sampled")
                if d is not None:
                    gains.append(d[0])
        pk = [x for x in (pass_at_k(s["final"]) for s in qwen[lbl].values()) if x is not None]
        lines.append(f"| {lbl} | {row['n']} | {pts(row['mean'])} | {ci} | {pts(g['mean']) if g else '-'} | "
                     f"{f'{100 * mean_share(lbl):.1f}%' if mean_share(lbl) is not None else '-'} | "
                     f"{pts(statistics.fmean(gains)) if gains else '-'} | {f'{statistics.fmean(pk):.3f}' if pk else '-'} |")
    if "sec[sec_tau1]" in qwen and "sec" in qwen:
        d = paired(qwen, "sec[sec_tau1]", "sec")
        lines += ["", "## Declared check: SEC at the authors' default temperature (descriptive)", "",
                  "SEC with Boltzmann temperature 1.0 (ServiceNow/sec default) instead of the tuned 0.4, seed 0 "
                  "(`reports/analysis/sec_dump_fidelity.md`). Not part of R1-R4.", ""]
        # overrides apply in order, so the last temperature override is the one the run used
        final_tau = {seed: next((o.split("=", 1)[1] for o in reversed((s.get("run_config") or {}).get("overrides", []))
                                 if o.startswith("baselines.sec.tau=")), None)
                     for seed, s in qwen["sec[sec_tau1]"].items()}
        ok = all(v is not None and float(v) == 1.0 for v in final_tau.values())
        lines.append(f"- temperature the run used (last override, by seed): {final_tau}"
                     + ("" if ok else "  **WARNING: not tau 1.0; do not report it as the declared check.**"))
        if 0 in d:
            lines.append(f"- tau 1.0 minus tau 0.4, same seed and items: {pts(d[0][0])} points (eval SE "
                         f"{100 * d[0][1]:.2f}).")
        if "sec[sec_tau1]" in vs_uniform:
            lines.append(f"- tau 1.0 minus Uniform: {pts(vs_uniform['sec[sec_tau1]']['mean'])} points.")
        if mean_share("sec[sec_tau1]") is not None:
            lines.append(f"- random-reward share: {100 * mean_share('sec[sec_tau1]'):.1f}% "
                         f"(tuned SEC {100 * (mean_share('sec') or 0):.1f}%, Uniform 20%).")
    if "clean_uniform" in vs_uniform:
        r = vs_uniform["clean_uniform"]
        lines += ["", "## Clean ceiling (descriptive)", "",
                  f"clean_uniform (equal mix without Countdown and the random-reward arm: what a perfect junk detector "
                  f"could reach) minus Uniform: {pts(r['mean'])} points over seeds {r['seeds']}."]
    lines += ["", "## Figures", "",
              "- `figures/accuracy_vs_uniform.png`: every method against Uniform (filled = 3-seed interval).",
              "- `figures/noisy_arm_share.png`: budget on the random-reward arm, seed dots.",
              "- `figures/noisy_share_by_round.png`: the same over training rounds (the trap, as it happens).",
              "- `figures/rules_per_seed.png`: R1 and R2 seed by seed.",
              "", "## Caveats", "",
              "- SEC and DUMP are re-implementations from the papers; SEC's tuned temperature (0.4) is sharper than "
              "the authors' default (1.0), which was outside the tuning grid (`reports/analysis/sec_dump_fidelity.md`).",
              "- The portfolio contains a random-reward arm on purpose: the comparison tests robustness, not "
              "accuracy on clean data.",
              "- Story decision (D-103): R1 and R2 hold -> the robustness thesis leads; otherwise the fallback "
              "(rigorous negative result with its mechanism, plus the pre-flight check and the in-run audit)."]
    (out / "d103_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"R1 {v1} | R2 {v2} (sec {v2s}, dump {v2d}) | R3 {v3} | R4 {v4}")
    print(f"written: {out / 'd103_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
