"""Sealed-test guard (Roadmap D.3).

The sealed test is read ONLY through `SealedTest.open()`, which:
1. requires the environment variable CURATOR_FINAL_EVAL=1,
2. requires a frozen-config hash file whose hash matches the experiment
   config (Gate 10 freeze rule),
3. appends an entry to `reports/test_access_log.jsonl`,
4. refuses a second open for the same (checkpoint_hash, split) pair.

`evaluation.final_test` (Phase 15) is the only production caller; the
separate loader lives in this package so the architecture test can enforce
that no L1/L2 module ever imports it (Roadmap D.3 leakage test 4).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class SealedTestError(RuntimeError):
    """Raised when sealed-test access is refused or repeated."""


class SealedTest:
    """Context manager owning one sealed-test access session."""

    def __init__(self, data_root: Path, repo_root: Path, split: str, checkpoint_hash: str) -> None:
        self.data_root = Path(data_root)
        self.repo_root = Path(repo_root)
        self.split = split
        self.checkpoint_hash = checkpoint_hash
        self._log_path = self.repo_root / "reports" / "test_access_log.jsonl"
        self._open = True

    @classmethod
    def open(
        cls,
        split: str,
        checkpoint_hash: str,
        cfg,
        repo_root: Path,
        frozen_cfg_hash_path: Path | None = None,
    ) -> SealedTest:
        """Gate-checked construction; raises SealedTestError on refusal."""
        if os.environ.get("CURATOR_FINAL_EVAL") != "1":
            raise SealedTestError(
                "sealed test requires CURATOR_FINAL_EVAL=1 in the environment"
            )
        if split != "test":
            raise SealedTestError(f"SealedTest only serves split 'test', got '{split}'")

        # The frozen-config hash must exist and match the current config hash.
        from curator_rl.core.config import config_hash

        actual_hash = config_hash(cfg)
        path = frozen_cfg_hash_path if frozen_cfg_hash_path is not None else (
            Path(repo_root) / "reports" / "frozen_config.hash"
        )
        if not path.exists():
            raise SealedTestError(
                f"frozen config hash file not found: {path} "
                "(Gate 10 freeze rule: configs must be frozen before final runs)"
            )
        recorded = path.read_text(encoding="utf-8").strip()
        if recorded != actual_hash:
            raise SealedTestError(
                "config hash does not match the frozen hash: "
                f"frozen={recorded} current={actual_hash}"
            )

        guard = cls(repo_root / "data", repo_root, split, checkpoint_hash)
        guard._register_access()
        return guard

    def _register_access(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        seen = self._previous_accesses()
        key = (self.checkpoint_hash, self.split)
        if key in seen:
            raise SealedTestError(
                f"sealed test already opened for (checkpoint={self.checkpoint_hash}, "
                f"split={self.split}); a second open is refused"
            )
        import time as _time

        entry = {
            "event": "sealed_test_open",
            "split": self.split,
            "checkpoint_hash": self.checkpoint_hash,
            "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }
        # append-only (audit E-25): a read-modify-write rewrite could drop an entry written by a
        # second process (two Kaggle GPUs) between the read and the rename
        fd = os.open(self._log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(entry) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def _previous_accesses(self) -> set[tuple[str, str]]:
        if not self._log_path.exists():
            return set()
        keys: set[tuple[str, str]] = set()
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # fail CLOSED (audit E-25): a corrupted entry must not permit a second opening
                raise SealedTestError(f"sealed-test access log is corrupted ({self._log_path}); "
                                      "refusing access until it is repaired by hand") from exc
            keys.add((record.get("checkpoint_hash", ""), record.get("split", "")))
        return keys

    def load_items(self, env_id: str) -> list[dict[str, Any]]:
        """Load the sealed items of one environment (single open required)."""
        if not self._open:
            raise SealedTestError("sealed-test session is closed")
        path = self.data_root / "test_sealed" / f"{env_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"sealed data for '{env_id}' not found: {path}")
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def close(self) -> None:
        self._open = False

    def __enter__(self) -> SealedTest:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
