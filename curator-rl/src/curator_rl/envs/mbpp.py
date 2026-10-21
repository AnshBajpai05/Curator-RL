"""MBPP environment (Roadmap v3 Phase E, D.2; env id `mbpp`).

Data: `google-research-datasets/mbpp` (full). Official train (374) is hash-split into train and
dev; official validation (90) is the calibration slice (small, so its SE is large: pool it into the
aggregate); official test (500) is the sealed test; the 10 `prompt` rows are few-shot material and
are not used. The prompt shows the task text and ONE example assertion; the verifier runs ALL
assertions of the task (the shown one, the hidden ones and any challenge tests), so special-casing
the shown example fails the hidden ones (Roadmap D.2 hacking mitigation).

Reward: 1 iff the extracted code passes every assertion in the sandbox (`envs.sandbox`), else 0.
Parse failure: no fenced code block and no definition in the response.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from curator_rl.core.types import Prompt, Verdict
from curator_rl.envs.finite import FiniteJsonlEnv
from curator_rl.envs.sandbox import run_python_tests

SYSTEM_PROMPT = "You are an expert Python programmer."
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_HINT_RE = re.compile(r"^\s*(def |class |import |from )", re.MULTILINE)


def extract_code(response: str) -> str | None:
    """Last fenced code block; else the whole response if it contains code; else None."""
    blocks = _FENCE_RE.findall(response)
    if blocks:
        return blocks[-1].strip("\n")
    if "```" in response:                       # unterminated fence: take what follows the last one
        tail = response.rsplit("```", 1)[-1]
        tail = re.sub(r"^(?:python|py)?\s*\n", "", tail, flags=re.IGNORECASE)
        return tail.strip("\n") if _CODE_HINT_RE.search(tail) else None
    return response.strip("\n") if _CODE_HINT_RE.search(response) else None


class MbppEnv(FiniteJsonlEnv):
    """Finite dataset over MBPP; sandboxed unit-test verifier."""

    dataset_label = "google-research-datasets/mbpp (full); calib = official validation; sealed test = official test"

    def __init__(self, repo_root: Path, cfg, data_root: Path | None = None) -> None:
        super().__init__("mbpp", repo_root, cfg, data_root)

    def _to_prompt(self, item: dict[str, Any], split: str) -> Prompt:
        shown = item["test_list"][0]
        user = (f"{item['text']}\n\nYour code should pass this test:\n{shown}\n\n"
                "Return the complete Python code in a single ```python code block.")
        return Prompt(
            prompt_id=item["problem_id"],
            env_id=self.env_id,
            split=item.get("split", split),
            messages=({"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}),
            reference=json.dumps({"tests": list(item["test_list"]) + list(item.get("challenge_test_list") or []),
                                  "setup": item.get("test_setup_code") or ""}),
            meta={"shown_test": shown, "n_tests": len(item["test_list"]) + len(item.get("challenge_test_list") or [])},
        )

    def evaluate_response(self, prompt: Prompt, response: str) -> Verdict:
        start = time.perf_counter()
        code = extract_code(response)
        if code is None:
            return Verdict(success=False, score=0.0, parse_ok=False,
                           verifier_seconds=time.perf_counter() - start, info={"no_code": True})
        spec = json.loads(prompt.reference)
        res = run_python_tests(code, spec["tests"], spec["setup"], timeout_s=self.cfg.verifier_timeout_s,
                               memory_mb=self.cfg.memory_limit_mb)
        return Verdict(success=res.passed, score=1.0 if res.passed else 0.0, parse_ok=True,
                       verifier_seconds=time.perf_counter() - start,
                       info={"timed_out": res.timed_out, "returncode": res.returncode,
                             "stderr_tail": res.stderr_tail, "rejected": res.rejected})
