"""Build the Kaggle GPU package: run.py + requirements + notebook + code + processed data, one zip.

Layout inside the zip:
    run.py, setup via `python run.py setup`, requirements*.txt, README_KAGGLE.md, curator_gpu.ipynb
    curator-rl/{src, configs, data/processed, data/manifests, pyproject.toml}
    BUNDLE_INFO.json (git commit, dirty flag, config hash, build time)

Like scripts/make_kaggle_bundle.py it NEVER packages the sealed test set, raw downloads, run outputs,
checkpoints, virtualenvs or git history, and refuses to write the archive if one slips in.

    python scripts/make_kaggle_gpu_bundle.py          # -> dist/curator-gpu-kaggle.zip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TOP_LEVEL = ("run.py", "requirements.txt", "requirements-hf.txt", "requirements-vllm.txt", "README_KAGGLE.md",
             "curator_gpu.ipynb")
CODE_DIRS = ("src", "configs", "data/processed", "data/manifests")
CODE_FILES = ("pyproject.toml",)
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", "curator_rl.egg-info"}
FORBIDDEN = ("test_sealed", "data/raw", "checkpoints", "runs/", ".git/", "venv", "curator_runs")


def collect() -> list[tuple[Path, str]]:
    out = [(REPO_ROOT / "kaggle" / f, f) for f in TOP_LEVEL]
    for rel in CODE_DIRS:
        base = REPO_ROOT / rel
        if not base.exists():
            raise SystemExit(f"missing {rel}: run scripts/download_data*.py and scripts/build_splits*.py first")
        for p in sorted(base.rglob("*")):
            if p.is_file() and not (set(p.parts) & SKIP_PARTS) and p.suffix not in (".pyc", ".pyo"):
                out.append((p, "curator-rl/" + p.relative_to(REPO_ROOT).as_posix()))
    out += [(REPO_ROOT / f, "curator-rl/" + f) for f in CODE_FILES]
    return out


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Kaggle GPU package")
    ap.add_argument("--out", default=str(REPO_ROOT / "dist" / "curator-gpu-kaggle.zip"))
    args = ap.parse_args(argv)
    files = collect()
    bad = [name for _, name in files if any(tok in name for tok in FORBIDDEN)]
    if bad:
        raise SystemExit(f"refusing to bundle forbidden paths: {bad[:5]}")
    for need in ("data/processed/gsm8k/train.jsonl", "data/processed/math35/calib.jsonl", "data/processed/mbpp/dev.jsonl"):
        if not (REPO_ROOT / need).exists():
            raise SystemExit(f"missing {need}: build the processed data first")
    from curator_rl.core.config import config_hash, load_config

    info = {"built_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--", ".")),     # the packaged tree only
            "config_hash": config_hash(load_config(REPO_ROOT / "configs" / "base.yaml")), "n_files": len(files)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, name in files:
            zf.write(path, name)
        zf.writestr("BUNDLE_INFO.json", json.dumps(info, indent=2))
    print(f"wrote {out} ({len(files)} files, {out.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
