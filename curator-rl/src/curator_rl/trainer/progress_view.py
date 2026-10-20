"""Progress lines the Kaggle supervisor prints while runs train, so a notebook never looks frozen.

The training workers write to log files; without this the notebook shows nothing for hours.
`ProgressBoard` reads each running run's `progress.json` (rewritten every round) and prints:
- one line per run whenever it completes a round or changes phase (starting, training, final
  evaluation), with a bar of the budget used, the time to go, the checkpoint and the backed-up round;
- each run's line again after a few minutes without one (model loading and the final evaluation
  change nothing for minutes), and a summary of the step (finished / running / waiting, estimated end)
  when a run starts or ends and every quarter of an hour.

Only whole lines are printed, never carriage returns or cursor movement: a notebook cannot redraw, so
redrawn bars become thousands of lines (what the Hugging Face upload bars did). Standard library only
(run.py runs it with Kaggle's own python), and nothing here raises into the supervisor.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BAR_WIDTH = 20
STARTUP_SECONDS = 5 * 60          # model load and vLLM start-up before the first round
FINAL_EVAL_SECONDS = 10 * 60      # the uncharged final evaluation (greedy + 4 samples per item, D-101)
UNCHARGED_OVERHEAD = 1.1          # wall seconds per charged GPU-second (calibration, reports, checkpoints)
QUIET_SECONDS = 5 * 60            # a run's line again after this long without one
SUMMARY_SECONDS = 15 * 60         # the step summary again after this long

PHASE_TEXT = {"stopping": "stopping for the session limit (resumes next session)",
              "crashed": "crashed (restarted from its checkpoint)", "finished": "finished"}


def bar(fraction: float, width: int = BAR_WIDTH, fancy: bool = True) -> str:
    full, empty = ("█", "░") if fancy else ("#", "-")
    n = max(0, min(width, round(fraction * width)))
    return full * n + empty * (width - n)


def duration(seconds: float) -> str:
    """'<1 min', '7 min', '1 h 05 min'."""
    m = int(round(max(0.0, seconds) / 60))
    if m < 1:
        return "<1 min"
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def phase_of(p: dict | None, start_round: int) -> str:
    """starting | training | final evaluation | stopping | crashed | finished, from a progress.json."""
    if not p:
        return "starting"
    status, stop = p.get("status"), p.get("stop_reason") or ""
    if status == "finished":
        return "finished"
    if status == "failed":
        return "crashed"
    if status in ("preempting", "preempted"):
        return "stopping"
    if stop in ("budget", "max_rounds"):
        return "final evaluation"
    return "starting" if int(p.get("round") or 0) == start_round else "training"


def training_left_seconds(p: dict | None, budget_gpu_hours: float) -> float:
    """Wall seconds of training left: rounds until the budget (or max_rounds) binds, times the round time."""
    p = p or {}
    budget = float(p.get("budget_gpu_hours") or budget_gpu_hours)
    charged, rnd = float(p.get("charged_gpu_hours") or 0.0), int(p.get("round") or 0)
    if rnd <= 0 or charged <= 0:
        return max(0.0, budget - charged) * 3600 * UNCHARGED_OVERHEAD
    per_round = charged / rnd
    rounds_left = math.ceil(max(0.0, budget - charged) / per_round - 1e-9)
    if p.get("max_rounds"):
        rounds_left = min(rounds_left, max(0, int(p["max_rounds"]) - rnd))
    return rounds_left * float(p.get("round_wall_seconds") or per_round * 3600 * UNCHARGED_OVERHEAD)


def job_seconds(budget_gpu_hours: float) -> float:
    """Wall time of a whole run that has not started."""
    return STARTUP_SECONDS + budget_gpu_hours * 3600 * UNCHARGED_OVERHEAD + FINAL_EVAL_SECONDS


def step_seconds_left(running_left: list[float], n_waiting: int, n_gpus: int, budget_gpu_hours: float) -> float:
    """When the last run ends: waiting runs take the first GPU that frees up."""
    free = sorted(running_left + [0.0] * max(0, n_gpus - len(running_left)))
    for _ in range(n_waiting):
        free[0] += job_seconds(budget_gpu_hours)
        free.sort()
    return max(free) if free else 0.0


@dataclass
class _Run:
    name: str
    gpu: int | None
    started: float
    start_round: int
    phase: str = "starting"
    phase_since: float = 0.0
    shown: tuple | None = None        # (round, phase) last printed
    shown_at: float = 0.0
    eval_only: bool = False           # no training: the base model, or a finished run being re-evaluated


class ProgressBoard:
    """Tracks the runs of one supervisor step and prints their progress (see the module docstring)."""

    def __init__(self, out_dir: Path, title: str, n_jobs: int, n_gpus: int, budget_gpu_hours: float, *,
                 tz: str = "UTC", clock: Callable[[], float] = time.time,
                 print_fn: Callable[[str], None] | None = None, fancy: bool | None = None) -> None:
        self.out_dir, self.title = Path(out_dir), title
        self.n_jobs, self.n_gpus, self.budget = n_jobs, max(1, n_gpus), budget_gpu_hours
        self.clock = clock
        self.print = print_fn or (lambda s: print(s, flush=True))
        enc = (getattr(sys.stdout, "encoding", None) or "").lower()
        self.fancy = ("utf" in enc) if fancy is None else fancy     # block characters need a UTF-8 console
        self.tz = _zone(tz)
        self.runs: dict[str, _Run] = {}
        self.finished = self.failed = self.stopped = 0
        self.last_summary = clock()
        self.summary_due = True
        self.name_width = 12

    # ------------------------------------------------------------------ called by run_jobs

    def launched(self, name: str, gpu: int | None, *, eval_only: bool = False) -> None:
        p, _ = self._progress(name)
        now = self.clock()
        self.runs[name] = _Run(name, gpu, now, int((p or {}).get("round") or 0), phase_since=now,
                               eval_only=eval_only)
        self.name_width = max(self.name_width, len(name))
        self.summary_due = True

    def ended(self, name: str, code: int, *, preempted_code: int = 75) -> None:
        self.runs.pop(name, None)
        if code == 0:
            self.finished += 1
            self._say(self._finished_text(name))
        elif code == preempted_code:
            self.stopped += 1
        else:
            self.failed += 1            # per crash; the restart is launched again under the same name
        self.summary_due = True

    def tick(self, n_waiting: int) -> None:
        """Print what changed since the last call, and lines not shown for a while. Never raises."""
        try:
            now = self.clock()
            for run in self.runs.values():
                p, mtime = self._progress(run.name)
                # a progress.json older than this launch is the previous attempt's (it may say "failed"
                # or "finished training"): the run is starting until the new process rewrites it
                phase = phase_of(p if mtime >= run.started else None, run.start_round)
                if phase != run.phase:
                    run.phase, run.phase_since = phase, now
                key = (int((p or {}).get("round") or run.start_round), phase)
                if key != run.shown or now - run.shown_at >= QUIET_SECONDS:
                    self._say(self.run_line(run, p, now))
                    run.shown, run.shown_at = key, now
            if self.summary_due or now - self.last_summary >= SUMMARY_SECONDS:
                self._say(self.summary_line(n_waiting, now))
                self.summary_due, self.last_summary = False, now
        except Exception as exc:  # noqa: BLE001 - a display problem must never stop the runs
            self.summary_due = False
            self.print(f"[run.py] progress display skipped: {exc!r}")

    # ------------------------------------------------------------------ lines

    def run_line(self, run: _Run, p: dict | None, now: float) -> str:
        p = p or {}
        budget = float(p.get("budget_gpu_hours") or self.budget)
        frac = max(float(p.get("charged_gpu_hours") or 0.0) / budget if budget else 0.0,
                   int(p.get("round") or 0) / int(p["max_rounds"]) if p.get("max_rounds") else 0.0)
        if run.phase in ("final evaluation", "finished"):
            frac = 1.0
        gpu = f"GPU{run.gpu}" if run.gpu is not None else "CPU "
        parts = [f"{run.name:<{self.name_width}} {gpu} {bar(frac, fancy=self.fancy)} {min(frac, 1.0) * 100:3.0f}%",
                 f"round {int(p.get('round') or 0)}"]
        left = self.seconds_left(run, p, now)
        if run.phase == "starting":
            what = ("loading the model for evaluation" if run.eval_only else "loading the model and vLLM"
                    if run.start_round == 0 else f"resuming from round {run.start_round}")
            parts.append(f"starting: {what}, ~{duration(left)} to go")
        elif run.phase == "training":
            parts.append(f"training, ~{duration(left)} to go")
        elif run.phase == "final evaluation":
            parts.append(f"final evaluation, ~{duration(left)} to go")
        else:
            parts.append(PHASE_TEXT[run.phase])
        keep = [f"checkpoint {p['last_checkpoint_round']}"] if p.get("last_checkpoint_round") is not None else []
        sync = p.get("sync") or {}
        if sync.get("target"):
            if sync.get("last_error"):
                keep.append("backup FAILING (see the log)")
            elif sync.get("last_round") is not None:
                keep.append(f"backed up {sync['last_round']}")
            else:
                keep.append("backup pending")
        if keep:
            parts.append(" · ".join(keep) if self.fancy else ", ".join(keep))
        return "  ".join(parts)

    def summary_line(self, n_waiting: int, now: float) -> str:
        running = list(self.runs.values())
        left = step_seconds_left([self.seconds_left(r, self._progress(r.name)[0], now) for r in running],
                                 n_waiting, self.n_gpus, self.budget)
        text = (f"{self.title}: {self.finished}/{self.n_jobs} runs finished, {len(running)} running, "
                f"{n_waiting} waiting")
        if self.stopped:
            text += f", {self.stopped} stopped for the session limit"
        if self.failed:
            text += f", {self.failed} crash(es)"
        if running or n_waiting:
            text += f"{' · ' if self.fancy else ' - '}step done in ~{duration(left)} (about {self._clock_text(now + left)})"
        return text

    def seconds_left(self, run: _Run, p: dict | None, now: float) -> float:
        if run.phase in ("finished", "stopping", "crashed"):
            return 0.0
        if run.phase == "final evaluation":
            return max(60.0, FINAL_EVAL_SECONDS - (now - run.phase_since))
        left = (0.0 if run.eval_only else training_left_seconds(p, self.budget)) + FINAL_EVAL_SECONDS
        if run.phase == "starting":
            left += max(0.0, STARTUP_SECONDS - (now - run.started))
        return left

    # ------------------------------------------------------------------ internals

    def _finished_text(self, name: str) -> str:
        mark = "✓" if self.fancy else "OK"
        try:
            s = json.loads((self.out_dir / name / "summary.json").read_text(encoding="utf-8"))
            score = (s.get("final") or {}).get("macro")
            return (f"{mark} {name} finished: {s.get('rounds')} rounds, {float(s.get('charged_gpu_hours') or 0):.2f} "
                    f"GPU-h" + (f", dev score {score:.3f}" if isinstance(score, (int, float)) else ""))
        except (OSError, ValueError, TypeError):
            return f"{mark} {name} finished"

    def _progress(self, name: str) -> tuple[dict | None, float]:
        """(progress.json of the run or None, its modification time)."""
        path = self.out_dir / name / "progress.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")), path.stat().st_mtime
        except (OSError, ValueError):
            return None, 0.0

    def _clock_text(self, t: float) -> str:
        return dt.datetime.fromtimestamp(t, self.tz).strftime("%H:%M %Z")

    def _say(self, line: str) -> None:
        try:
            self.print(f"[{self._clock_text(self.clock())}] {line}")
        except Exception:  # noqa: BLE001 - e.g. a console that cannot encode a character: skip the line
            pass


def _zone(name: str) -> dt.tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - no time zone database: UTC
        return dt.UTC
