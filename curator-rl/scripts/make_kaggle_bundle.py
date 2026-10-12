"""Build the zip you upload to Kaggle as a Dataset (Roadmap v3 4.9).

Contains only what GPU notebooks need: `src/`, `configs/`, `scripts/`, `experiments/`,
`data/manifests/`, `data/processed/` and `pyproject.toml`. It NEVER contains the sealed
test set (`data/test_sealed/`), raw downloads, run outputs, checkpoints, git history or
caches; the script refuses to write the archive if a forbidden path slips in.

    python scripts/make_kaggle_bundle.py            # -> dist/curator-rl-bundle.zip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = ("src", "configs", "scripts", "experiments", "data/manifests", "data/processed")
INCLUDE_FILES = ("pyproject.toml",)
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", "curator_rl.egg-info"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN = ("test_sealed", "data/raw", "checkpoints", "runs/", ".git/")


def collect() -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE_DIRS:
        base = REPO_ROOT / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not (set(path.parts) & SKIP_PARTS) and path.suffix not in SKIP_SUFFIXES:
                files.append(path)
    files += [REPO_ROOT / f for f in INCLUDE_FILES if (REPO_ROOT / f).exists()]
    return files


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Kaggle bundle")
    p.add_argument("--out", default=str(REPO_ROOT / "dist" / "curator-rl-bundle.zip"))
    args = p.parse_args(argv)

    files = collect()
    names = [f.relative_to(REPO_ROOT).as_posix() for f in files]
    bad = [n for n in names if any(tok in n for tok in FORBIDDEN)]
    if bad:
        raise SystemExit(f"refusing to bundle forbidden paths: {bad[:5]}")

    from curator_rl.core.config import config_hash, load_config

    info = {
        "built_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": git_sha(),
        "config_hash": config_hash(load_config(REPO_ROOT / "configs" / "base.yaml")),
        "n_files": len(files),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f, n in zip(files, names):
            zf.write(f, n)
        zf.writestr("BUNDLE_INFO.json", json.dumps(info, indent=2))
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(files)} files, {size_mb:.1f} MB)")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
