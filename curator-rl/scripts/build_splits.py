"""Build processed splits + manifests for GSM8K and noisy (Roadmap Phase 2, D.3).

- Official GSM8K test items -> data/test_sealed/gsm8k.jsonl (never committed).
- Official train pool -> hash-split into calib (calib_size), dev (dev_size),
  train (the rest) via `assign_splits_finite`.
- Processed JSONL written to data/processed/{env}/{split}.jsonl.
- Committed manifests to data/manifests/{env}_{split}.txt + MANIFEST.sha256.
- The noisy environment reuses the GSM8K train rows verbatim.

Usage: python scripts/build_splits.py
Idempotent: running twice yields byte-identical manifests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.envs.gsm8k import normalize_gsm8k_gold  # noqa: E402
from curator_rl.envs.splits import (  # noqa: E402
    assert_disjoint,
    assign_splits_finite,
    read_manifest_ids,
    rebuild_manifest_index,
    write_manifests,
)


def _rows_to_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _load_official(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    train = pd.read_parquet(raw_dir / "train.parquet")
    test = pd.read_parquet(raw_dir / "test.parquet")
    train_rows = [
        {
            "problem_id": f"gsm8k-train-{idx}",
            "question": str(row["question"]),
            "answer": normalize_gsm8k_gold(str(row["answer"])),
            "answer_raw": str(row["answer"]),
        }
        for idx, row in train.iterrows()
    ]
    test_rows = [
        {
            "problem_id": f"gsm8k-test-{idx}",
            "question": str(row["question"]),
            "answer": normalize_gsm8k_gold(str(row["answer"])),
            "answer_raw": str(row["answer"]),
        }
        for idx, row in test.iterrows()
    ]
    return train_rows, test_rows


def main() -> int:
    data_root = REPO_ROOT / "data"
    raw_dir = data_root / "raw" / "gsm8k"
    if not (raw_dir / "train.parquet").exists():
        print("raw GSM8K missing; run `python scripts/download_data.py` first", file=sys.stderr)
        return 1

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    salt = cfg.data.split_salt
    calib_size, dev_size = cfg.data.calib_size, cfg.data.dev_size

    train_rows, test_rows = _load_official(raw_dir)
    bad_gold = [r["problem_id"] for r in train_rows + test_rows if not r["answer"]]
    if bad_gold:
        print(f"warning: {len(bad_gold)} rows normalised to empty answers", file=sys.stderr)

    # 1. official test -> sealed dir (ids also go to a committed manifest)
    _rows_to_jsonl(data_root / "test_sealed" / "gsm8k.jsonl", test_rows)
    print(f"sealed test rows: {len(test_rows)}")

    # 2. hash-split the official train pool
    pool_ids = [r["problem_id"] for r in train_rows]
    assignment = assign_splits_finite("gsm8k", pool_ids, salt, calib_size, dev_size)
    assert_disjoint({s: [i for i in pool_ids if assignment[i] == s] for s in ("train", "calib", "dev")})
    assert_disjoint(
        {
            "train": [i for i in pool_ids if assignment[i] == "train"],
            "calib": [i for i in pool_ids if assignment[i] == "calib"],
            "dev": [i for i in pool_ids if assignment[i] == "dev"],
            "test": [r["problem_id"] for r in test_rows],
        }
    )
    by_split: dict[str, list[dict]] = {"train": [], "calib": [], "dev": []}
    for row in train_rows:
        row = {**row, "split": assignment[row["problem_id"]]}
        by_split[assignment[row["problem_id"]]].append(row)
    for split, rows in by_split.items():
        _rows_to_jsonl(data_root / "processed" / "gsm8k" / f"{split}.jsonl", rows)
    print(f"gsm8k splits: { {s: len(rows) for s, rows in by_split.items()} }")

    # 3. noisy env: verbatim copy of the gsm8k train rows
    _rows_to_jsonl(data_root / "processed" / "noisy" / "train.jsonl", by_split["train"])

    # 4. manifests: gsm8k (4 splits incl. sealed ids), noisy (train only)
    hashes: dict[str, str] = {}
    hashes.update(
        write_manifests(
            data_root,
            "gsm8k",
            {
                "train": [r["problem_id"] for r in by_split["train"]],
                "calib": [r["problem_id"] for r in by_split["calib"]],
                "dev": [r["problem_id"] for r in by_split["dev"]],
                "test": [r["problem_id"] for r in test_rows],
            },
        )
    )
    hashes.update(
        write_manifests(
            data_root,
            "noisy",
            {
                "train": [r["problem_id"] for r in by_split["train"]],
                "calib": [],
                "dev": [],
                "test": [],
            },
        )
    )
    # countdown is procedural: no id manifests, only seed-range disjointness
    rebuild_manifest_index(data_root)  # hashes every manifest incl. other environments'

    # 5. verify idempotency signal: reload via manifest readers
    sealed_ids = read_manifest_ids(data_root, "gsm8k", "test")
    print(f"manifests written: {len(hashes)}; sealed ids in manifest: {len(sealed_ids)}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
