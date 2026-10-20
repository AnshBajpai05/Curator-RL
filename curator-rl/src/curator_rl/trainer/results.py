"""The `results` folder: everything worth keeping from a set of runs, plus one downloadable zip.

`export_results(runs_root, results_dir)` rebuilds, from the run directories under `runs_root`:

    results/
      README.txt              what each file is
      comparison.md           methods item by item and by seed, gain over base, ROI, leave-one-out
      status.md               where every run is (status, round, GPU-hours, checkpoint, sync)
      plots/*.png             mixture over rounds per run, score vs GPU-hours
      runs/<run>/             summary, progress, config, round/step/calibration/report logs, sysid report
      logs/*.log              the training processes' console logs
      curator_results.zip     all of the above in one file (replaced atomically)

Checkpoints are never copied (they live, at most `keep` per run, under each run directory), so the
folder stays a few MB. Safe to call repeatedly and while runs are training: files are copied, the zip
is written to a temporary name and renamed into place, so a download never sees a half-written zip.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

ZIP_NAME = "curator_results.zip"
RUN_FILES = ("summary.json", "progress.json", "run_config.json", "rounds.jsonl", "steps.jsonl",
             "calibration.jsonl", "reports.jsonl", "sysid_report.md", "sysid_report.json")

README = """CURATOR results ({stamp} UTC)

comparison.md   sampled + greedy scores, item-by-item differences, gain over the untrained model,
                GPU-hours to reach Uniform's level, ROI leaderboards, leave-one-out
status.md       every run: status, round, GPU-hours used, last checkpoint, last sync
plots/          <run>_mixture.png (share of each arm per round, S5 marked), score_vs_gpu_hours.png
                (learning curves on the same items)
runs/<run>/     summary.json (final scores, shares, S5 arms, errors), progress.json (heartbeat),
                run_config.json, rounds.jsonl / steps.jsonl / calibration.jsonl / reports.jsonl (logs),
                sysid_report.md (system identification runs only)
logs/           console log of every training process
{zip_name}  everything above in one file

Checkpoints are not included (they stay in curator_runs/<run>/checkpoints, newest 3 per run).
"""


def status_table(runs_root: Path) -> str:
    """Markdown table of every run's progress.json heartbeat."""
    rows = []
    for p in sorted(Path(runs_root).glob("*/progress.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sync = s.get("sync") or {}
        rows.append(f"| {p.parent.name} | {s.get('status')} | {s.get('round')}/{s.get('max_rounds')} | "
                    f"{s.get('charged_gpu_hours')}/{s.get('budget_gpu_hours')} | {s.get('last_checkpoint_round')} | "
                    f"{sync.get('last_round', '-')} ({sync.get('last_ok_utc') or 'never'}) | "
                    f"{s.get('stop_reason') or ''} | {s.get('updated_utc')} |")
    head = ["| run | status | round | GPU-h used/budget | checkpoint | synced round (when) | stop reason | updated |",
            "|---|---|---|---|---|---|---|---|"]
    return "\n".join(head + (rows or ["| (no runs yet) |  |  |  |  |  |  |  |"])) + "\n"


def _replace_dir(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)


def export_results(runs_root: Path, results_dir: Path) -> Path:
    """Rebuild `results_dir` from the runs under `runs_root`; returns the path of the zip."""
    runs_root, results_dir = Path(runs_root), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    (results_dir / "README.txt").write_text(README.format(stamp=stamp, zip_name=ZIP_NAME), encoding="utf-8")
    (results_dir / "status.md").write_text(status_table(runs_root), encoding="utf-8")
    comparison = runs_root / "comparison.md"
    if comparison.exists():
        shutil.copy2(comparison, results_dir / "comparison.md")
    for sub, pattern in (("plots", "*.png"), ("logs", "*.log")):
        _replace_dir(results_dir / sub)
        for f in sorted((runs_root / sub).glob(pattern)) if (runs_root / sub).exists() else []:
            shutil.copy2(f, results_dir / sub / f.name)
    _replace_dir(results_dir / "runs")
    for run in sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "run_config.json").exists()):
        dst = results_dir / "runs" / run.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            if (run / name).exists():
                shutil.copy2(run / name, dst / name)
    tmp = results_dir / (ZIP_NAME + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(results_dir.rglob("*")):
            if f.is_file() and f.name not in (ZIP_NAME, tmp.name):
                z.write(f, Path("curator_results") / f.relative_to(results_dir))
    os.replace(tmp, results_dir / ZIP_NAME)
    return results_dir / ZIP_NAME
