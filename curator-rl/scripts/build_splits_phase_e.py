"""Build processed splits and manifests for MATH35 and MBPP (Roadmap v3 Phase E, D.3).

math35: official MATH train problems of level 3-5 (data.envs.math35.level_min/max) are hash-split
        into calib (data.calib_size) / dev (data.dev_size) / train via `assign_splits_finite`;
        the sealed test is MATH-500 (data/test_sealed/math35.jsonl, never committed).
mbpp:   official train (374) -> dev (data.envs.mbpp.dev_size, hash-chosen) / train (rest);
        official validation (90) -> calib; official test (500) -> sealed;
        the 10 `prompt` rows are few-shot material and are not used.

Writes data/processed/{math35,mbpp}/{split}.jsonl, data/test_sealed/{env}.jsonl and committed
manifests data/manifests/{env}_{split}.txt, then rebuilds MANIFEST.sha256 over all manifests.
Idempotent (byte-identical outputs). Usage: python scripts/build_splits_phase_e.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.envs.math35 import extract_boxed, parse_level  # noqa: E402
from curator_rl.envs.splits import (  # noqa: E402
    assert_disjoint,
    assign_splits_finite,
    rebuild_manifest_index,
    write_manifests,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def build_math(cfg, data_root: Path) -> None:
    raw = data_root / "raw" / "math"
    train = pd.read_parquet(raw / "train.parquet")
    lo, hi = cfg.data.envs.math35.level_min, cfg.data.envs.math35.level_max
    pool, skipped = [], 0
    for idx, row in train.iterrows():
        level = parse_level(str(row["level"]))
        gold = extract_boxed(str(row["solution"]))
        if level is None or not (lo <= level <= hi) or not gold:
            skipped += 1
            continue
        pool.append({"problem_id": f"math-train-{idx}", "problem": str(row["problem"]), "answer": gold,
                     "level": level, "type": str(row["type"])})
    ids = [r["problem_id"] for r in pool]
    assignment = assign_splits_finite("math35", ids, cfg.data.split_salt, cfg.data.calib_size, cfg.data.dev_size)
    by_split: dict[str, list[dict]] = {"train": [], "calib": [], "dev": []}
    for r in pool:
        by_split[assignment[r["problem_id"]]].append({**r, "split": assignment[r["problem_id"]]})
    test_rows = []
    for line in (raw / "math500_test.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        test_rows.append({"problem_id": f"math500-{r['unique_id']}", "problem": r["problem"],
                          "answer": str(r["answer"]), "level": int(r["level"]), "type": r["subject"]})
    assert_disjoint({**{s: [r["problem_id"] for r in v] for s, v in by_split.items()},
                     "test": [r["problem_id"] for r in test_rows]})
    for split, rows in by_split.items():
        write_jsonl(data_root / "processed" / "math35" / f"{split}.jsonl", rows)
    write_jsonl(data_root / "test_sealed" / "math35.jsonl", test_rows)
    write_manifests(data_root, "math35", {**{s: [r["problem_id"] for r in v] for s, v in by_split.items()},
                                          "test": [r["problem_id"] for r in test_rows]})
    print(f"math35: pool {len(pool)} (skipped {skipped}); splits { {s: len(v) for s, v in by_split.items()} }; sealed {len(test_rows)}")


def build_mbpp(cfg, data_root: Path) -> None:
    raw = data_root / "raw" / "mbpp"

    def rows(name: str) -> list[dict]:
        df = pd.read_parquet(raw / f"{name}.parquet")
        return [{"problem_id": f"mbpp-{int(r['task_id'])}", "text": str(r["text"]), "code": str(r["code"]),
                 "test_list": [str(t) for t in r["test_list"]], "test_setup_code": str(r["test_setup_code"] or ""),
                 "challenge_test_list": [str(t) for t in r["challenge_test_list"]]} for _, r in df.iterrows()]

    official_train, validation, test = rows("train"), rows("validation"), rows("test")
    dev_size = cfg.data.envs.mbpp.dev_size
    ids = [r["problem_id"] for r in official_train]
    # calib_size=0: the whole official train pool is split into dev (hash-ranked) and train
    assignment = assign_splits_finite("mbpp", ids, cfg.data.split_salt, 0, dev_size)
    by_split = {"train": [], "dev": [], "calib": [{**r, "split": "calib"} for r in validation]}
    for r in official_train:
        by_split[assignment[r["problem_id"]]].append({**r, "split": assignment[r["problem_id"]]})
    assert_disjoint({**{s: [r["problem_id"] for r in v] for s, v in by_split.items()},
                     "test": [r["problem_id"] for r in test]})
    for split, rs in by_split.items():
        write_jsonl(data_root / "processed" / "mbpp" / f"{split}.jsonl", rs)
    write_jsonl(data_root / "test_sealed" / "mbpp.jsonl", test)
    write_manifests(data_root, "mbpp", {**{s: [r["problem_id"] for r in v] for s, v in by_split.items()},
                                        "test": [r["problem_id"] for r in test]})
    print(f"mbpp: splits { {s: len(v) for s, v in by_split.items()} }; sealed {len(test)}")


def main() -> int:
    data_root = REPO_ROOT / "data"
    for need in (data_root / "raw" / "math" / "train.parquet", data_root / "raw" / "mbpp" / "train.parquet"):
        if not need.exists():
            print(f"missing {need}; run `python scripts/download_data_phase_e.py` first", file=sys.stderr)
            return 1
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    build_math(cfg, data_root)
    build_mbpp(cfg, data_root)
    records = rebuild_manifest_index(data_root)
    print(f"manifests: {len(records)} files indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
