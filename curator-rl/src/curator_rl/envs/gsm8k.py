"""GSM8K environment (Roadmap D.2).

Data flow: `scripts/download_data.py` stores the official parquet files in
`data/raw/gsm8k/`; `scripts/build_splits.py` normalises them, hash-splits
the official train pool into train/calib/dev and writes processed JSONL plus
committed manifests. The official test split is written to the sealed
directory and is served only through `evaluation.guard`.

Verdict semantics: the model must give its final numeric answer inside
\\boxed{...} (fixed prompt template). `evaluate_response` verifies only;
reward = verdict.score in [0, 1].
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from curator_rl.core.types import CostPrior, Prompt, Verdict
from curator_rl.envs.base import Environment, run_with_timeout
from curator_rl.envs.splits import manifest_dir, read_manifest_ids

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

SYSTEM_PROMPT = "You are a careful math tutor. Solve the problem and be concise."
ANSWER_INSTRUCTION = (
    "Solve the following grade-school math problem step by step and give "
    "the final numeric answer inside \\boxed{}."
)


def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST balanced \\boxed{...} in the text, or None.

    Audit E-08: the old single-level regex `\\boxed\\{([^{}]*)\\}` found nothing in
    `\\boxed{18 \\text{ dollars}}` or `\\boxed{1{,}000}` (braces inside), so correct answers in
    these common formats scored 0. Brace-balanced extraction (shared with MATH35) fixes it.
    """
    from curator_rl.envs.math35 import extract_boxed as _balanced

    boxed = _balanced(text)
    return boxed.strip() if boxed is not None else None


_LATEX_NOISE = ("\\$", "{,}", "\\,", "\\!", "\\;", "\\:", "\\ ", "\\%")
_UNIT_TEXT_RE = re.compile(r"\\(?:text|mbox|mathrm)\{[^{}]*\}")


def normalize_number_text(raw: str) -> str | None:
    """Normalise a numeric strip: LaTeX spacing/units, commas, dollar and percent signs, spaces.

    Returns None when the value is not a plain number (parse failure).
    Exact rational semantics via Fraction; '2.50' == '2.5'. `\\$18`, `18 \\text{ dollars}` and
    `1{,}000` normalise like `18` and `1000` (audit E-08); words or two numbers stay a failure.
    """
    cleaned = raw
    for tok in _LATEX_NOISE:
        cleaned = cleaned.replace(tok, "," if tok == "{,}" else "")
    cleaned = _UNIT_TEXT_RE.sub("", cleaned)
    cleaned = cleaned.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if not cleaned:
        return None
    try:
        dec = Decimal(cleaned)
        frac = Fraction(dec)          # NaN / Infinity raise here: a parse failure, not a crash
    except (InvalidOperation, ValueError, OverflowError):
        return None
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def normalize_gsm8k_gold(raw_answer: str) -> str:
    """Official GSM8K answers look like '#### 1,034' (possibly with units)."""
    final = raw_answer.rsplit("####", 1)[-1]
    m = re.search(r"(-?[\d,]*\.?\d+)", final)
    if not m:
        return ""
    normalized = normalize_number_text(m.group(1))
    return normalized or ""


def _verify_boxed(response: str, reference: str) -> tuple[str | None, bool]:
    """Extract the boxed answer and compare exactly with the reference.

    Returns (boxed_content_or_None, is_correct). A missing or non-numeric
    boxed value is a parse failure, never a correct answer.
    """
    boxed = extract_boxed(response)
    if boxed is None:
        return None, False
    parsed = normalize_number_text(boxed)
    if parsed is None:
        return boxed, False
    return boxed, parsed == reference


class Gsm8kEnv(Environment):
    """Finite dataset over the official GSM8K train pool; official test sealed.

    `generate_prompt` is deterministic given (split, rng state): the position
    drawn depends only on the rng. `generate_batch` enforces distinct ids
    (sampling without replacement within a batch); pass-level reshuffling is
    the trainer sampler's job (Roadmap D.4).
    """

    def __init__(self, repo_root: Path, cfg, data_root: Path | None = None) -> None:
        super().__init__("gsm8k")
        self.cfg = cfg
        self.repo_root = Path(repo_root)
        self.data_root = data_root if data_root is not None else self.repo_root / "data"
        self._processed = self.data_root / "processed" / "gsm8k"
        self._items: dict[str, list[dict[str, Any]]] = {}
        self._load_processed()

    def _load_processed(self) -> None:
        for path in sorted(self._processed.glob("*.jsonl")):
            split = path.stem
            self._items[split] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def split_ids(self, split: str) -> list[str]:
        ids = [item["problem_id"] for item in self._items.get(split, [])]
        if split == "test" and not ids:
            ids = read_manifest_ids(self.data_root, self.env_id, "test")
        return ids

    def generate_prompt(self, split: str, rng) -> Prompt:
        item = self._draw_item(split, rng)
        return self._to_prompt(item)

    def split_size(self, split: str) -> int | None:
        self._require_split(split)
        return len(self._items[split])

    def prompt_at(self, split: str, index: int) -> Prompt:
        self._require_split(split)
        return self._to_prompt(self._items[split][index])

    def generate_batch(self, split: str, n: int, rng, exclude: set[str] | None = None) -> list[Prompt]:
        self._require_split(split)
        out: list[Prompt] = []
        exclude = set(exclude or ())
        tries = 0
        while len(out) < n and tries < 10 * n:
            tries += 1
            prompt = self.generate_prompt(split, rng)
            if prompt.prompt_id in exclude or prompt.prompt_id in {p.prompt_id for p in out}:
                continue
            out.append(prompt)
        if len(out) < n:
            raise ValueError(f"could not draw {n} distinct prompts from split '{split}'")
        return out

    def evaluate_response(self, prompt: Prompt, response: str) -> Verdict:
        start = time.perf_counter()
        try:
            boxed, correct = run_with_timeout(
                lambda: _verify_boxed(response, prompt.reference),
                self.cfg.verifier_timeout_s,
            )
            parse_ok = boxed is not None
            success = parse_ok and correct
            info: dict[str, Any] = {"boxed": boxed}
        except Exception:
            parse_ok = False
            success = False
            info = {"timeout": True}
        return Verdict(
            success=success,
            score=1.0 if success else 0.0,
            parse_ok=parse_ok,
            verifier_seconds=time.perf_counter() - start,
            info=info,
        )

    def compute_reward(self, verdict: Verdict) -> float:
        reward = verdict.score
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward {reward} outside [0, 1]")
        return reward

    def estimate_cost(self) -> CostPrior:
        return CostPrior(
            env_id=self.env_id,
            per_prompt_usd=self.cfg.prior_usd_per_prompt,
            note="config prior for cold start only (Roadmap H.5)",
        )

    def metadata(self) -> dict[str, Any]:
        manifest_hashes: dict[str, str] = {}
        mdir = manifest_dir(self.data_root)
        if mdir.exists():
            for path in sorted(mdir.glob(f"{self.env_id}_*.txt")):
                manifest_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "env_id": self.env_id,
            "dataset": "openai/gsm8k (config main)",
            "dataset_version": "main-branch parquet; SHA-256 of files in data/raw/gsm8k/metadata.json",
            "split_sizes": {split: len(items) for split, items in sorted(self._items.items())},
            "manifest_hashes": manifest_hashes,
            "max_completion_tokens": self.cfg.max_completion_tokens,
            "max_prompt_tokens": self.cfg.max_prompt_tokens,
            "verifier_timeout_s": self.cfg.verifier_timeout_s,
        }

    # --- internals -------------------------------------------------------
    def _require_split(self, split: str) -> None:
        if split == "test":
            raise ValueError("the sealed test split is served only through evaluation.guard")
        if not self._items.get(split):
            raise ValueError(f"split '{split}' has no processed data")

    def _draw_item(self, split: str, rng) -> dict[str, Any]:
        self._require_split(split)
        items = self._items[split]
        index = int(rng.integers(len(items)))
        return items[index]

    def _to_prompt(self, item: dict[str, Any]) -> Prompt:
        return Prompt(
            prompt_id=item["problem_id"],
            env_id=self.env_id,
            split=item.get("split", ""),
            messages=(
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{item['question']}\n\n{ANSWER_INSTRUCTION}"},
            ),
            reference=item["answer"],
            meta={"question": item["question"]},
        )
