"""Split machinery (Roadmap D.3 / SPEC section 3).

Four splits: train / calib / dev / test. For finite datasets, official test
items are set aside first; the remaining pool is ordered by
``u = sha256(salt + dataset + problem_id)`` and the first `calib_size`
positions become calib, the next `dev_size` become dev, the rest train.
This keeps split sizes exact and assignment deterministic and
salt-sensitive. Procedural environments (Countdown) use disjoint seed
ranges instead and have no manifests.

Manifests make splits reviewable: one ids file per (env, split), and
`MANIFEST.sha256` records the SHA-256 of each file so a changed split is
visible in Git.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from curator_rl.core.types import SPLITS

SPLIT_NAMES = SPLITS  # train, calib, dev, test


def split_hash_u(dataset: str, problem_id: str, salt: str) -> float:
    """u = int(sha256(salt + dataset + problem_id)[:8], 16) / 2**32 (SPEC s3)."""
    digest = hashlib.sha256((salt + dataset + problem_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 2**32


def assign_splits_finite(
    dataset: str, ids: list[str], salt: str, calib_size: int, dev_size: int
) -> dict[str, str]:
    """Hash-rank assignment with exact split sizes for a finite dataset.

    `ids` must already exclude any official test items (those are sealed
    separately). Ties (astronomically unlikely) are broken by id so the
    result is fully deterministic.
    """
    if calib_size + dev_size > len(ids):
        raise ValueError(
            f"calib({calib_size}) + dev({dev_size}) exceed pool of {len(ids)} items"
        )
    ranked = sorted(ids, key=lambda pid: (split_hash_u(dataset, pid, salt), pid))
    assignment: dict[str, str] = {}
    for pos, pid in enumerate(ranked):
        if pos < calib_size:
            assignment[pid] = "calib"
        elif pos < calib_size + dev_size:
            assignment[pid] = "dev"
        else:
            assignment[pid] = "train"
    return assignment


def manifest_dir(data_root: Path) -> Path:
    return Path(data_root) / "manifests"


def write_manifests(data_root: Path, env_id: str, split_ids: dict[str, list[str]]) -> dict[str, str]:
    """Write `<env>_<split>.txt` (sorted ids) and return the sha of each file."""
    base = manifest_dir(data_root)
    base.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for split, ids in split_ids.items():
        path = base / f"{env_id}_{split}.txt"
        # LF on every OS: in text mode Windows wrote CRLF, so the checksums verified only there
        path.write_text("\n".join(sorted(ids)) + ("\n" if ids else ""), encoding="utf-8", newline="\n")
        hashes[f"{env_id}_{split}.txt"] = _file_sha256(path)
    return hashes


def rewrite_manifest_index(data_root: Path, records: dict[str, str]) -> None:
    """Write MANIFEST.sha256 as lines '<sha256>  <name>'."""
    base = manifest_dir(data_root)
    lines = [f"{sha}  {name}" for name, sha in sorted(records.items())]
    (base / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def rebuild_manifest_index(data_root: Path) -> dict[str, str]:
    """Rewrite MANIFEST.sha256 from EVERY `*.txt` manifest present (several builders share it)."""
    base = manifest_dir(data_root)
    records = {p.name: _file_sha256(p) for p in sorted(base.glob("*.txt"))}
    rewrite_manifest_index(data_root, records)
    return records


def verify_manifests(data_root: Path) -> list[str]:
    """Return the list of mismatches between MANIFEST.sha256 and the files."""
    index_path = manifest_dir(data_root) / "MANIFEST.sha256"
    if not index_path.exists():
        return ["MANIFEST.sha256 missing"]
    problems: list[str] = []
    expected: dict[str, str] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, name = line.split(None, 1)
        expected[name.strip()] = sha
    for name, sha in expected.items():
        path = manifest_dir(data_root) / name
        if not path.exists():
            problems.append(f"manifest file missing: {name}")
            continue
        if _file_sha256(path) != sha:
            problems.append(f"manifest hash mismatch: {name}")
    return problems


def read_manifest_ids(data_root: Path, env_id: str, split: str) -> list[str]:
    path = manifest_dir(data_root) / f"{env_id}_{split}.txt"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_disjoint(split_ids: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split, ids in split_ids.items():
        for pid in ids:
            if pid in seen:
                raise ValueError(f"item '{pid}' assigned to both {seen[pid]} and {split}")
            seen[pid] = split
