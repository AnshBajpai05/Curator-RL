"""MATH levels 3-5 environment (Roadmap v3 Phase E, D.2; env id `math35`).

Data: `DigitalLearningGmbH/MATH-lighteval` train problems of level 3, 4 and 5 (5,582 problems).
The training pool is hash-split into train / calib / dev; the sealed test set is MATH-500
(all levels, an out-of-distribution slice by level). The gold answer is the last balanced
`\\boxed{...}` of the reference solution.

Reward: 1 if the model's last `\\boxed{...}` equals the gold answer after a CONSERVATIVE
normalisation, else 0. The checker is deliberately self-contained (no optional library) so a
reward is identical on every machine; it may reject unusual but correct spellings (false
negatives) and never accepts two different values as equal (no false positives), which the
tests check. A missing boxed answer is a parse failure.
"""

from __future__ import annotations

import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from curator_rl.core.types import Prompt, Verdict
from curator_rl.envs.base import run_with_timeout
from curator_rl.envs.finite import FiniteJsonlEnv

SYSTEM_PROMPT = "You are a careful mathematics tutor. Reason step by step, then state the final answer."
ANSWER_INSTRUCTION = "Put the final answer inside \\boxed{}."

_LEVEL_RE = re.compile(r"Level\s+(\d)")


def parse_level(raw: str) -> int | None:
    m = _LEVEL_RE.search(raw or "")
    return int(m.group(1)) if m else None


def extract_boxed(text: str) -> str | None:
    """Content of the LAST balanced `\\boxed{...}` (handles nested braces such as \\frac{1}{2})."""
    start = text.rfind("\\boxed")
    while start != -1:
        i = start + len("\\boxed")
        while i < len(text) and text[i] == " ":
            i += 1
        if i < len(text) and text[i] == "{":
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1 : j].strip()
                j += 1
        start = text.rfind("\\boxed", 0, start)
    return None


_STRIP_TOKENS = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\displaystyle", "\\$", "$")
_UNIT_RE = re.compile(r"\\(?:text|mbox|mathrm)\{\s*[a-zA-Z .]*\}")


def _sub_frac(s: str) -> str:
    """\\frac{a}{b} -> (a)/(b) for single-level braces, repeatedly; also \\frac12 -> 1/2."""
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"(\1)/(\2)", s)
    pattern = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
    while True:
        new = pattern.sub(r"(\1)/(\2)", s)
        if new == s:
            return s
        s = new


def normalize_answer(raw: str) -> str:
    """Conservative canonical form of a MATH final answer."""
    s = raw.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for tok in _STRIP_TOKENS:
        s = s.replace(tok, "")
    whole = re.fullmatch(r"\\(?:text|mbox|mathrm)\{([^{}]*)\}", s)
    if whole:                                    # an answer that IS text: \text{yes} -> yes
        s = whole.group(1)
    else:
        s = _UNIT_RE.sub("", s)                  # trailing units such as 5\text{ cm}
        s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = s.rstrip(".").strip()
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)       # x = 5 -> 5
    s = _sub_frac(s)
    s = s.replace(" ", "")
    return s


_NUM = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
_PLAIN_RE = re.compile(rf"^{_NUM}$")
_SLASH_RE = re.compile(rf"^({_NUM})/({_NUM})$")
_FRAC_RE = re.compile(rf"^(-?)\(({_NUM})\)/\(({_NUM})\)$")          # what _sub_frac produces
_MIXED_RE = re.compile(r"^(-?)(\d+)\((\d+)\)/\((\d+)\)$")            # 2\frac{1}{2} -> 2(1)/(2)
_PAREN_RE = re.compile(rf"^\(({_NUM})\)$")


def _numeric_value(s: str) -> Fraction | None:
    """Exact rational value of a plain number / simple fraction / mixed number, else None.

    Audit E-06: parentheses used to be deleted wholesale, so the mixed number 2\\frac{1}{2}
    (= 5/2) became 21/2 and the product 2(3) became 23: wrong answers were accepted and
    correct ones rejected. Parentheses are now accepted only in the exact shapes `_sub_frac`
    produces; anything else is not a plain number and falls back to string comparison
    (which can only cause false negatives, never false positives).
    """
    t = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s) else s
    try:
        if m := _MIXED_RE.match(t):
            if int(m.group(4)) == 0:
                return None
            value = int(m.group(2)) + Fraction(int(m.group(3)), int(m.group(4)))
            return -value if m.group(1) else value
        if m := _FRAC_RE.match(t):
            value = Fraction(m.group(2)) / Fraction(m.group(3))
            return -value if m.group(1) else value
        if m := _PAREN_RE.match(t):
            return Fraction(m.group(1))
        if m := _SLASH_RE.match(t):
            return Fraction(m.group(1)) / Fraction(m.group(2))
        if _PLAIN_RE.match(t):
            return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def answers_equal(pred: str, gold: str) -> bool:
    a, b = normalize_answer(pred), normalize_answer(gold)
    if a == b:
        return True
    va, vb = _numeric_value(a), _numeric_value(b)
    return va is not None and vb is not None and va == vb


class Math35Env(FiniteJsonlEnv):
    """Finite dataset over MATH train levels 3-5; MATH-500 sealed."""

    dataset_label = "DigitalLearningGmbH/MATH-lighteval train, levels 3-5; sealed test = HuggingFaceH4/MATH-500"

    def __init__(self, repo_root: Path, cfg, data_root: Path | None = None) -> None:
        super().__init__("math35", repo_root, cfg, data_root)

    def _to_prompt(self, item: dict[str, Any], split: str) -> Prompt:
        return Prompt(
            prompt_id=item["problem_id"],
            env_id=self.env_id,
            split=item.get("split", split),
            messages=(
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{item['problem']}\n\n{ANSWER_INSTRUCTION}"},
            ),
            reference=item["answer"],
            meta={"level": item.get("level"), "type": item.get("type")},
        )

    def evaluate_response(self, prompt: Prompt, response: str) -> Verdict:
        start = time.perf_counter()
        try:
            boxed = extract_boxed(response)
            pred: str = boxed or ""
            success = boxed is not None and run_with_timeout(
                lambda: answers_equal(pred, prompt.reference), self.cfg.verifier_timeout_s)
            info: dict[str, Any] = {"boxed": boxed}
        except Exception:
            boxed, success, info = None, False, {"timeout": True}
        return Verdict(success=bool(success), score=1.0 if success else 0.0, parse_ok=boxed is not None,
                       verifier_seconds=time.perf_counter() - start, info=info)
