"""Mirror a run's artifacts to persistent storage while it trains, and fetch them back to resume.

A Kaggle session can end at any moment (12-hour limit, preemption, a crashed kernel), and
`/kaggle/working` disappears with it. `ArtifactSync` copies the run directory somewhere that
survives: a private Hugging Face dataset repo (`hf://owner/repo[/prefix]`, token from `HF_TOKEN`)
or any directory (a mounted drive, an attached disk). `fetch_runs` brings the newest state back,
and the run resumes from its last checkpoint.

How a sync stays consistent without slowing training:
- The training thread STAGES a snapshot right after a checkpoint is written (under a second): logs
  and small files are copied, and the files of the checkpoint `latest.json` points to are
  hard-linked (checkpoint files are never modified in place, only replaced by new directories).
- A background thread uploads the staged snapshot. Training never waits for the network, and an
  upload error is logged and retried at the next sync. Nothing in this module raises into training.
- Each upload is an exact mirror of the snapshot: remote files that are not in it (older
  checkpoints) are deleted in the same commit, and unchanged files are not uploaded again.
- Uploads are rate-limited (`min_interval_s`); `flush()` forces a final blocking sync.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

HF_PREFIX = "hf://"
SKIP_DIRS = {"trl"}                       # TRL's own output dir (save_strategy="no": scratch only)


def _quiet_hub() -> None:
    """No Hugging Face progress bars: in a notebook they flood the output (and the page) every upload."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils.tqdm import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001 - older hub versions: the environment variable suffices
        pass


def parse_target(target: str | None) -> tuple[str, str, str]:
    """('none' | 'hf' | 'dir', location, prefix) for a sync target string."""
    if not target:
        return "none", "", ""
    if target.startswith(HF_PREFIX):
        parts = target[len(HF_PREFIX):].strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"hf target must be hf://owner/repo[/prefix], got {target!r}")
        return "hf", "/".join(parts[:2]), "/".join(parts[2:])
    return "dir", str(Path(target)), ""


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_snapshot(run_dir: Path, stage: Path) -> Path:
    """Consistent copy of `run_dir` in `stage`: small files copied, the latest checkpoint linked."""
    run_dir, stage = Path(run_dir), Path(stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for item in sorted(run_dir.iterdir()):
        if item.name in SKIP_DIRS or item.name.endswith(".tmp"):
            continue
        if item.name == "checkpoints" and item.is_dir():
            pointer = item / "latest.json"
            if not pointer.exists():
                continue
            name = json.loads(pointer.read_text(encoding="utf-8"))["dir"]
            ckpt = item / name
            if not (ckpt / "controller.json").exists():
                continue
            for f in sorted(ckpt.rglob("*")):
                if f.is_file():
                    _link_or_copy(f, stage / "checkpoints" / name / f.relative_to(ckpt))
            # the pointer goes last: a snapshot never names a checkpoint it does not contain
            shutil.copy2(pointer, stage / "checkpoints" / "latest.json")
        elif item.is_file():
            shutil.copy2(item, stage / item.name)
        elif item.is_dir():
            shutil.copytree(item, stage / item.name)
    return stage


class ArtifactSync:
    """Background mirror of one run directory (no-op when `target` is empty)."""

    def __init__(self, run_dir: Path, target: str | None, *, min_interval_s: float = 300.0,
                 log: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic,
                 uploader: Callable[[Path, str], None] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.kind, self.location, self.prefix = parse_target(target)
        self.min_interval_s = float(min_interval_s)
        self.log, self.clock = log, clock
        self.stage = self.run_dir.parent / ".sync_stage" / self.run_dir.name
        self._upload = uploader or self._default_uploader()
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_start = -float("inf")
        self.status = {"target": target or "", "ok": 0, "failed": 0, "last_ok_utc": None, "last_error": None,
                       "last_round": None}

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    # ------------------------------------------------------------------ public

    def request(self, *, force: bool = False, label: str = "") -> bool:
        """Stage now and upload in the background, if due. Never raises; returns True if started."""
        if not self.enabled or self._busy.is_set():
            return False
        if not force and self.clock() - self._last_start < self.min_interval_s:
            return False
        try:
            stage_snapshot(self.run_dir, self.stage)
        except Exception as exc:  # noqa: BLE001 - a sync problem must never stop training
            self._record_failure(f"staging failed: {exc!r}")
            return False
        self._last_start = self.clock()
        self._busy.set()
        self._thread = threading.Thread(target=self._run_upload, args=(label,), daemon=True)
        self._thread.start()
        return True

    def flush(self, timeout: float = 900.0, label: str = "final") -> bool:
        """Wait for any upload in flight, then sync the current state and wait for it."""
        if not self.enabled:
            return True
        self.wait(timeout)
        ok_before = self.status["ok"]
        if not self.request(force=True, label=label):
            return False
        self.wait(timeout)
        return self.status["ok"] > ok_before

    def wait(self, timeout: float = 900.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)

    # ----------------------------------------------------------------- internals

    def _run_upload(self, label: str) -> None:
        try:
            self._upload(self.stage, label)
            with self._lock:
                self.status["ok"] += 1
                self.status["last_ok_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self.status["last_error"] = None
                pointer = self.stage / "checkpoints" / "latest.json"
                if pointer.exists():
                    self.status["last_round"] = json.loads(pointer.read_text(encoding="utf-8")).get("round")
        except Exception as exc:  # noqa: BLE001
            self._record_failure(f"upload failed: {exc!r}\n{traceback.format_exc(limit=3)}")
        finally:
            self._busy.clear()

    def _record_failure(self, message: str) -> None:
        with self._lock:
            self.status["failed"] += 1
            self.status["last_error"] = message.splitlines()[0][:300]
        self.log(f"[sync] {message.strip()}")

    def _default_uploader(self) -> Callable[[Path, str], None]:
        if self.kind == "hf":
            return self._upload_hf
        if self.kind == "dir":
            return self._upload_dir
        return lambda stage, label: None

    def _upload_dir(self, stage: Path, label: str) -> None:
        dest = Path(self.location) / self.run_dir.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp, old = dest.with_name(dest.name + ".incoming"), dest.with_name(dest.name + ".old")
        for p in (tmp, old):
            if p.exists():
                shutil.rmtree(p)
        shutil.copytree(stage, tmp)
        if dest.exists():
            os.replace(dest, old)
        os.replace(tmp, dest)
        shutil.rmtree(old, ignore_errors=True)

    def _upload_hf(self, stage: Path, label: str) -> None:
        _quiet_hub()
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        if not getattr(self, "_hf_repo_ready", False):
            api.create_repo(self.location, repo_type="dataset", private=True, exist_ok=True)
            self._hf_repo_ready = True
        path = "/".join(p for p in (self.prefix, self.run_dir.name) if p)
        api.upload_folder(repo_id=self.location, repo_type="dataset", folder_path=str(stage), path_in_repo=path,
                          delete_patterns=["*"], commit_message=f"{self.run_dir.name}: {label or 'sync'}")


# ------------------------------------------------------------------------------ fetching back

def _ckpt_round(run_dir: Path) -> int:
    pointer = Path(run_dir) / "checkpoints" / "latest.json"
    try:
        return int(json.loads(pointer.read_text(encoding="utf-8"))["round"])
    except (OSError, ValueError, KeyError):
        return -1


def _finished(run_dir: Path) -> bool:
    return (Path(run_dir) / "summary.json").exists()


def _restore_dropped_lines(src: Path, dst: Path) -> list[str]:
    """Log files a local copy of a finished run lost lines of, brought back from the backup.

    Only where the local file is a strict prefix of the backed-up one (lines missing at the end, never
    different content): an older version rewound a finished run's logs to its last checkpoint when it
    was relaunched, dropping the final-evaluation rows written after that checkpoint.
    """
    fixed = []
    for f in sorted(Path(src).glob("*.jsonl")):
        d = Path(dst) / f.name
        try:
            local, backup = (d.read_bytes() if d.exists() else b""), f.read_bytes()
        except OSError:
            continue
        if len(local) < len(backup) and backup.startswith(local):
            shutil.copy2(f, d)
            fixed.append(f.name)
    return fixed


def merge_runs(source_root: Path, out_root: Path, *, log: Callable[[str], None] = print) -> list[str]:
    """Copy every run under `source_root` whose state is newer than the local one into `out_root`."""
    taken = []
    source_root, out_root = Path(source_root), Path(out_root)
    if not source_root.exists():
        return taken
    for src in sorted(p for p in source_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not ((src / "run_config.json").exists() or (src / "checkpoints").exists()):
            continue
        dst = out_root / src.name
        newer = (_finished(src) and not _finished(dst)) or _ckpt_round(src) > _ckpt_round(dst)
        if dst.exists() and not newer:
            if _finished(src) and _finished(dst) and (fixed := _restore_dropped_lines(src, dst)):
                log(f"[sync] {src.name}: restored log lines missing locally ({', '.join(fixed)})")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        taken.append(src.name)
        log(f"[sync] restored {src.name} (checkpoint round {_ckpt_round(dst)}, "
            f"{'finished' if _finished(dst) else 'resumable'})")
    return taken


def fetch_runs(source: str, out_root: Path, *, log: Callable[[str], None] = print) -> list[str]:
    """Bring runs back from a sync target or a plain directory (e.g. an attached Kaggle input)."""
    kind, location, prefix = parse_target(source)
    if kind == "none":
        return []
    if kind == "dir":
        return merge_runs(Path(location), out_root, log=log)
    _quiet_hub()
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    if not api.repo_exists(location, repo_type="dataset"):
        log(f"[sync] {source}: no such dataset repo yet (first session?)")
        return []
    tmp = Path(out_root) / ".sync_fetch"
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(repo_id=location, repo_type="dataset", local_dir=str(tmp),
                      allow_patterns=[f"{prefix}/*"] if prefix else None,
                      token=os.environ.get("HF_TOKEN") or None)
    taken = merge_runs(tmp / prefix if prefix else tmp, out_root, log=log)
    shutil.rmtree(tmp, ignore_errors=True)
    return taken


def squash_remote_history(target: str, *, log: Callable[[str], None] = print) -> None:
    """Collapse the sync repo's history so superseded checkpoints stop using storage (best effort)."""
    kind, location, _ = parse_target(target)
    if kind != "hf":
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN") or None).super_squash_history(location, repo_type="dataset")
        log(f"[sync] squashed the history of {location}")
    except Exception as exc:  # noqa: BLE001
        log(f"[sync] history squash skipped: {exc!r}")
