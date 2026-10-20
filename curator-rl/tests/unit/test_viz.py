"""Figures (curator_rl.viz, improvement D5): every function draws from plain run-log data."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("matplotlib")

from curator_rl.viz.plots import render_run_plots, roi_vs_loo, score_vs_gpu_hours, weights_timeline  # noqa: E402

ROUNDS = [{"round": t, "delivered": {"gsm8k": 0.5, "noisy": 0.5 if t < 3 else 0.1, "mbpp": 0.0 if t < 3 else 0.4},
           "statuses": {"noisy": "S5_unreliable" if t >= 3 else "S2_learning"}} for t in range(1, 6)]


def test_figures_draw_from_log_rows():
    assert weights_timeline(ROUNDS).axes
    assert score_vs_gpu_hours({"curator": [(0.5, 0.2), (1.0, 0.3)], "uniform": [(0.5, 0.18)]}).axes
    assert roi_vs_loo({"gsm8k": 0.1, "noisy": -0.02}, {"gsm8k": 0.03, "noisy": -0.01}).axes


def test_render_run_plots_writes_pngs(tmp_path):
    run = tmp_path / "curator_s0"
    run.mkdir()
    (run / "rounds.jsonl").write_text("\n".join(json.dumps(r) for r in ROUNDS))
    (run / "reports.jsonl").write_text(json.dumps({"tag": "final", "charged_gpu_hours": 0.4, "macro": 0.21}))
    files, note = render_run_plots(tmp_path)
    assert {f.name for f in files} == {"curator_s0_mixture.png", "score_vs_gpu_hours.png"}
    assert all(f.stat().st_size > 1000 for f in files)
