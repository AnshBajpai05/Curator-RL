"""System identification from ONE real run with randomized mixtures (the `sysid` design).

Three quantities the simulator had to assume or bracket, identified from data:

1. Per-arm training cost (audit A3, Gate 5). GEN + UPDATE seconds of every step are regressed on
   per-arm token counts plus a fixed per-step overhead (NNLS). Because the Dirichlet design varies
   the mixture every round, the coefficients are identified without timing each environment alone.
2. Learning and transfer (assumed eta and the 0.3 GSM8K <-> MATH35 transfer in every refit). Every
   calibration window evaluates every slice; slice d's paired gain over window k is regressed on
   every arm's prompt share in that window:  gain_{d,k} = sum_j W_{j,k} B_{dj} + noise(paired SE).
   B_{dd} is arm d's own learning rate, B_{dj} the transfer from j. B_{gsm8k, noisy} answers
   directly whether random-reward training moves real GSM8K accuracy on this model (the
   Spurious-Rewards question the roadmap planned to test on a second model family).
3. Group polarisation kappa per arm, from each round's pass rate and mixed-group share (the
   refit's kappa = 2.14 came from 64 untrained prompts; here it is tracked during training).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from curator_rl.calibration.credit import TRANSFER_PRIOR_SD
from curator_rl.cost.attribution import fit_step_cost_model


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()] \
        if Path(path).exists() else []


def mixed_share(p: float, kappa: float, g: int = 8) -> float:
    a, b = kappa * p, kappa * (1.0 - p)
    right = wrong = 1.0
    for j in range(g):
        right *= (a + j) / (a + b + j)
        wrong *= (b + j) / (a + b + j)
    return 1.0 - right - wrong


def fit_kappa(points: Sequence[tuple[float, float]], g: int = 8) -> float | None:
    """Least-squares kappa from (pass rate, mixed share) pairs (grid 0.1..50)."""
    pts = [(p, m) for p, m in points if p is not None and m is not None and 0.02 < p < 0.98]
    if len(pts) < 3:
        return None
    grid = np.arange(0.1, 50.0, 0.05)
    err = [sum((mixed_share(p, k, g) - m) ** 2 for p, m in pts) for k in grid]
    return float(grid[int(np.argmin(err))])


def fit_transfer(calib_rows: Sequence[Mapping], env_ids: Sequence[str]) -> dict:
    """Ridge WLS per slice of window gains on every arm's share (B matrix with posterior SDs)."""
    env_ids = list(env_ids)
    by_domain: dict[str, list[tuple[np.ndarray, float, float]]] = {}
    prev: dict[str, float] = {}
    for row in calib_rows:
        if "scores" not in row:          # a logged evaluation failure (skipped window), not a measurement
            continue
        exp = row["exposure"]
        total = sum(exp.values()) or 1.0
        w = np.array([exp.get(e, 0.0) / total for e in env_ids])
        for d, s in row["scores"].items():
            dse = (row.get("delta_se") or {}).get(d)
            if d in prev and dse:
                by_domain.setdefault(d, []).append((w, s - prev[d], max(float(dse), 1e-6)))
            prev[d] = s
    out = {}
    for d, obs in by_domain.items():
        x = np.array([o[0] for o in obs])
        y = np.array([o[1] for o in obs])
        inv = np.array([1.0 / o[2] ** 2 for o in obs])
        precision = (x * inv[:, None]).T @ x + np.eye(len(env_ids)) / TRANSFER_PRIOR_SD**2
        cov = np.linalg.inv(precision)
        mean = cov @ ((x * inv[:, None]).T @ y)
        out[d] = {e: {"b": float(mean[i]), "sd": float(math.sqrt(max(cov[i, i], 0.0)))}
                  for i, e in enumerate(env_ids)}
        out[d]["_windows"] = len(obs)
    return out


def analyse(run_dir: Path, env_ids: Sequence[str], group_size: int = 8) -> dict:
    run_dir = Path(run_dir)
    steps, rounds, calib = _rows(run_dir / "steps.jsonl"), _rows(run_dir / "rounds.jsonl"), _rows(run_dir / "calibration.jsonl")
    report: dict = {"n_steps": len(steps), "n_rounds": len(rounds),
                    "n_calibrations": sum(1 for c in calib if "scores" in c),
                    "n_failed_evaluations": sum(1 for c in calib if "errors" in c)}
    try:
        cost = fit_step_cost_model(steps)
        tokens_per_prompt = {}
        for e in env_ids:
            tok = sum(s["tokens"].get(e, 0.0) for s in steps)
            n = sum(s["prompts"].get(e, 0) for s in steps)
            tokens_per_prompt[e] = tok / n if n else None
        p_step = (sum(sum(s["prompts"].values()) for s in steps) / len(steps)) if steps else 1.0
        per_prompt = {e: (cost["seconds_per_1k_tokens"].get(e, 0.0) * tokens_per_prompt[e] / 1000.0
                          + cost["overhead_seconds_per_step"] / p_step) if tokens_per_prompt[e] else None
                      for e in env_ids}
        base = per_prompt.get("gsm8k") or next((v for v in per_prompt.values() if v), 1.0)
        cost["seconds_per_prompt"] = per_prompt
        cost["relative_cost"] = {e: (v / base if v else None) for e, v in per_prompt.items()}
        cost["tokens_per_prompt"] = tokens_per_prompt
        report["cost_model"] = cost
    except ValueError as exc:
        report["cost_model"] = {"error": str(exc)}
    report["transfer"] = fit_transfer(calib, env_ids)
    report["kappa"] = {e: fit_kappa([(r["arms"][e]["pass_rate"], r["arms"][e]["mixed"]) for r in rounds
                                     if e in r["arms"]], group_size) for e in env_ids}
    return report


def render(report: Mapping) -> str:
    lines = ["# System identification report (one randomized-mixture run)", ""]
    cm = report.get("cost_model", {})
    if "error" in cm:
        lines += [f"Cost model: {cm['error']}", ""]
    else:
        lines += ["## Per-arm training cost (NNLS on step time vs tokens)", "",
                  f"Fixed overhead {cm['overhead_seconds_per_step']:.2f} s/step, R^2 {cm['r2']:.3f}, {cm['n_steps']} steps.", "",
                  "| arm | s / 1k tokens | tokens / prompt | s / prompt | relative cost |", "|---|---|---|---|---|"]
        for e, a in cm["seconds_per_1k_tokens"].items():
            tpp, spp, rel = cm["tokens_per_prompt"].get(e), cm["seconds_per_prompt"].get(e), cm["relative_cost"].get(e)
            lines.append(f"| {e} | {a:.3f} | {tpp if tpp is None else round(tpp)} | "
                         f"{'-' if spp is None else f'{spp:.3f}'} | {'-' if rel is None else f'{rel:.2f}x'} |")
        lines.append("")
    lines += ["## Learning and transfer: slice gain per window of full compute share", "",
              "Row = calibration slice, column = arm trained; mean (posterior SD).", ""]
    tr = report.get("transfer", {})
    arms = sorted({e for d in tr.values() for e in d if not e.startswith("_")})
    if tr:
        lines += ["| slice | windows | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
        for d, row in sorted(tr.items()):
            lines.append(f"| {d} | {row['_windows']} | " + " | ".join(
                f"{row[e]['b']:+.3f} ({row[e]['sd']:.3f})" for e in arms) + " |")
        if "gsm8k" in tr and "noisy" in tr["gsm8k"]:
            b, sd = tr["gsm8k"]["noisy"]["b"], tr["gsm8k"]["noisy"]["sd"]
            lines += ["", f"**Random-reward (noisy) training -> GSM8K accuracy: {b:+.3f} (SD {sd:.3f}).** "
                      "Near zero means the spurious reward neither helps nor hurts this model; clearly "
                      "positive reproduces the Qwen Spurious-Rewards effect; negative means it harms."]
    lines += ["", "## Group polarisation kappa per arm (during training)", "",
              " ".join(f"{e}: {k:.2f};" if k is not None else f"{e}: -;" for e, k in report.get("kappa", {}).items()), ""]
    return "\n".join(lines) + "\n"
