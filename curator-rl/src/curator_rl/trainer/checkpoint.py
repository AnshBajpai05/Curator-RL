"""Crash-safe checkpoints for runs that outlive a Kaggle session (Phase F, audit A4).

Layout:  <out>/checkpoints/round-0005/{controller.json, backend files...}
         <out>/checkpoints/latest.json -> {"dir": "round-0005", "round": 5}
A checkpoint is written into `round-0005.tmp`, then renamed; `latest.json` is replaced atomically
afterwards. A session killed at any moment leaves either the previous checkpoint or the new one,
never a half-written one. Only the newest `keep` checkpoints (default 3) are kept: older ones and
half-written `.tmp` leftovers are deleted automatically, the one `latest.json` names never is. The controller state is versioned
and refuses to load into a run with a different method, seed or batch shape.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from curator_rl.core.atomic import atomic_write_json

LATEST = "latest.json"


DEFAULT_KEEP = 3


def save_checkpoint(out_dir: Path, controller, backend_save: Callable[[Path], None], *,
                    keep: int = DEFAULT_KEEP) -> Path:
    """Write a checkpoint atomically, then delete all but the newest `keep` (never the one just written)."""
    keep = max(keep, 1)
    root = Path(out_dir) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    name = f"round-{controller.round:04d}"
    tmp, final = root / f"{name}.tmp", root / name
    for stale in root.glob("*.tmp"):           # half-written leftovers of a crash or a killed session
        shutil.rmtree(stale, ignore_errors=True)
    tmp.mkdir()
    atomic_write_json(tmp / "controller.json", controller.get_state())
    backend_save(tmp)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    atomic_write_json(root / LATEST, {"dir": name, "round": controller.round,
                                      "global_step": controller.global_step})
    done = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("round-")
                  and not p.name.endswith(".tmp"))
    for old in done[:-keep]:
        if old != final:                        # the checkpoint latest.json points to is never removed
            shutil.rmtree(old, ignore_errors=True)
    return final


def latest_checkpoint(out_dir: Path) -> Path | None:
    pointer = Path(out_dir) / "checkpoints" / LATEST
    if not pointer.exists():
        return None
    path = pointer.parent / json.loads(pointer.read_text(encoding="utf-8"))["dir"]
    return path if (path / "controller.json").exists() else None


def load_controller_state(ckpt_dir: Path) -> dict:
    return json.loads((Path(ckpt_dir) / "controller.json").read_text(encoding="utf-8"))
