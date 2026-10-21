"""Progress lines of the Kaggle supervisor: phases, time estimates, when lines are printed."""

from __future__ import annotations

import json
import os
import time

from curator_rl.trainer.progress_view import (
    FINAL_EVAL_SECONDS,
    QUIET_SECONDS,
    STARTUP_SECONDS,
    ProgressBoard,
    bar,
    duration,
    job_seconds,
    phase_of,
    step_seconds_left,
    training_left_seconds,
)


def test_bar_and_duration():
    assert bar(0.5, width=10) == "█████░░░░░"
    assert bar(1.7, width=4, fancy=False) == "####"
    assert duration(20) == "<1 min" and duration(7 * 60) == "7 min" and duration(3900) == "1 h 05 min"


def test_phases():
    assert phase_of(None, 0) == "starting"
    assert phase_of({"status": "running", "round": 0}, 0) == "starting"
    assert phase_of({"status": "running", "round": 13}, 12) == "training"
    assert phase_of({"status": "running", "round": 35, "stop_reason": "budget"}, 12) == "final evaluation"
    assert phase_of({"status": "finishing", "round": 60, "stop_reason": "max_rounds"}, 12) == "final evaluation"
    assert phase_of({"status": "preempting", "round": 20}, 12) == "stopping"
    assert phase_of({"status": "failed", "round": 20}, 12) == "crashed"
    assert phase_of({"status": "finished", "round": 35}, 12) == "finished"


def test_training_time_left_follows_the_binding_budget():
    # curator_s0 on Kaggle: 1.40 of 1.5 GPU-h after 30 rounds of ~188 s -> 3 more rounds
    p = {"round": 30, "max_rounds": 60, "charged_gpu_hours": 1.40145, "budget_gpu_hours": 1.5,
         "round_wall_seconds": 188.4}
    assert training_left_seconds(p, 1.5) == 3 * 188.4
    assert training_left_seconds(dict(p, max_rounds=31), 1.5) == 188.4          # max_rounds binds first
    assert training_left_seconds(None, 1.5) == 1.5 * 3600 * 1.1                  # not started: whole budget


def test_step_estimate_puts_waiting_runs_on_the_first_free_gpu():
    one = job_seconds(1.5)
    assert step_seconds_left([600.0, 100.0], 1, 2, 1.5) == 100.0 + one
    assert step_seconds_left([600.0], 2, 2, 1.5) == 600.0 + one               # idle GPU + the one freed at 600
    assert step_seconds_left([], 0, 2, 1.5) == 0.0


class _Clock:
    def __init__(self):
        self.t = time.time() - 1000          # progress files written now are newer than every launch

    def __call__(self):
        return self.t


def _write(out, name, **progress):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    return d / "progress.json"


def _board(tmp_path, lines, clock, n_jobs=2):
    return ProgressBoard(tmp_path, "compare", n_jobs, 2, 1.5, clock=clock, print_fn=lines.append, fancy=True)


def test_board_prints_changes_heartbeats_and_the_finish(tmp_path):
    lines, clock = [], _Clock()
    board = _board(tmp_path, lines, clock)
    board.launched("curator_s0", 0)
    board.tick(1)
    assert "curator_s0" in lines[0] and "starting: loading the model and vLLM" in lines[0]
    assert "compare: 0/2 runs finished, 1 running, 1 waiting · step done in ~" in lines[1]

    sync = {"target": "hf://o/r", "last_round": 1, "last_error": None}
    _write(tmp_path, "curator_s0", status="running", round=0, max_rounds=60, charged_gpu_hours=0.0,
           budget_gpu_hours=1.5)
    clock.t += 30
    board.tick(1)
    assert len(lines) == 2                   # the new process wrote round 0: still starting, nothing new

    _write(tmp_path, "curator_s0", status="running", round=2, max_rounds=60, charged_gpu_hours=0.1,
           budget_gpu_hours=1.5, round_wall_seconds=170.0, last_checkpoint_round=2, sync=sync)
    clock.t += 60
    board.tick(1)
    assert len(lines) == 3 and "round 2  training, ~" in lines[2] and "checkpoint 2 · backed up 1" in lines[2]
    board.tick(1)
    assert len(lines) == 3                   # no change: no line

    clock.t += QUIET_SECONDS
    board.tick(1)
    assert len(lines) == 4 and "round 2" in lines[3]       # heartbeat: shown again after a quiet spell

    _write(tmp_path, "curator_s0", status="running", round=35, max_rounds=60, charged_gpu_hours=1.5,
           budget_gpu_hours=1.5, stop_reason="budget", last_checkpoint_round=35, sync=dict(sync, last_error="x"))
    board.tick(1)
    assert "100%" in lines[4] and "final evaluation, ~" in lines[4] and "backup FAILING" in lines[4]

    (tmp_path / "curator_s0" / "summary.json").write_text(json.dumps(
        {"rounds": 35, "charged_gpu_hours": 1.5009, "final": {"macro": 0.41234}}), encoding="utf-8")
    board.ended("curator_s0", 0)
    assert lines[5].endswith("✓ curator_s0 finished: 35 rounds, 1.50 GPU-h, dev score 0.412")
    board.tick(0)
    assert "compare: 1/2 runs finished, 0 running, 0 waiting" in lines[6] and "step done" not in lines[6]


def test_a_previous_attempts_progress_file_does_not_show_a_crash(tmp_path):
    lines, clock = [], _Clock()
    old = _write(tmp_path, "lp_s0", status="failed", round=12, max_rounds=60, charged_gpu_hours=0.5,
                 budget_gpu_hours=1.5)
    os.utime(old, (clock.t - 60, clock.t - 60))           # written before this launch
    board = _board(tmp_path, lines, clock)
    board.launched("lp_s0", 1)
    board.tick(0)
    assert "starting: resuming from round 12" in lines[0] and "crashed" not in lines[0]
    board.launched("base_s0", 0, eval_only=True)
    board.tick(0)
    base = next(x for x in lines if "base_s0" in x)
    assert f"starting: loading the model for evaluation, ~{duration(FINAL_EVAL_SECONDS + STARTUP_SECONDS)}" in base
    board.ended("lp_s0", 1)
    board.ended("lp_s0", 75)
    board.tick(0)
    assert "1 stopped for the session limit, 1 crash(es)" in lines[-1]


def test_estimates_and_ascii_console(tmp_path):
    lines, clock = [], _Clock()
    board = ProgressBoard(tmp_path, "loo", 1, 1, 1.5, clock=clock, print_fn=lines.append, fancy=False)
    _write(tmp_path, "uniform_s0", status="running", round=35, stop_reason="budget", charged_gpu_hours=1.5,
           budget_gpu_hours=1.5, last_checkpoint_round=35, sync={"target": "d", "last_round": 35})
    board.launched("uniform_s0", 0)
    board.tick(0)
    full = duration(FINAL_EVAL_SECONDS)
    assert f"final evaluation, ~{full} to go" in lines[0] and f"step done in ~{full}" in lines[1]
    clock.t += max(QUIET_SECONDS, FINAL_EVAL_SECONDS)      # taking longer than expected: never below a minute
    board.tick(0)
    assert "#" * 20 in lines[2] and "final evaluation, ~1 min to go" in lines[2]
    assert "checkpoint 35, backed up 35" in lines[2] and "█" not in lines[2] and " - step done" in lines[1]


def test_display_errors_never_reach_the_supervisor(tmp_path):
    def broken(_):
        raise UnicodeEncodeError("cp1252", "█", 0, 1, "no")

    board = ProgressBoard(tmp_path, "compare", 1, 1, 1.5, print_fn=broken)
    board.launched("curator_s0", 0)
    board.tick(0)
    board.ended("curator_s0", 0)
