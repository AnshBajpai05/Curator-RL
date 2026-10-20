"""Pre-flight verifier check (D-102): does an arm's reward pay for answers that cannot be right?

Verifier bugs become reward hacks under optimisation pressure: a checker that accepts the reference
number anywhere in the response, an empty box, or a program that exits before its tests run hands out
reward for junk, and RL learns to produce it ("Before the Model Learns the Bug: Fuzzing RLVR
Verifiers", 2026; false positives drive reward hacking, Cai et al. 2025). Before any training, each
arm's full reward path (verification and reward shaping) scores a fixed set of junk responses on a
few of its own training prompts. Junk that earns reward is a finding, reported per probe.

This complements the in-run reward audit (signals/audit.py), which needs rounds of data and only sees
rewards that ignore the prompt; a buggy checker that still pays for correct answers passes the audit
but fails here. With `signals.reward_audit.preflight_gate`, a flagged arm starts in S5 (capped at the
exploration floor) instead of waiting for evidence. Costs seconds and no GPU.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FUZZ_SEED = 20261022
N_PROMPTS = 6
FLAG_RATE = 0.05          # junk reward rate above this (or one probe paid on half the prompts) flags the arm


def _wrong_number(reference: str) -> str:
    try:
        value = float(str(reference).replace(",", ""))
    except ValueError:
        return "-999983"
    return str(int(value) + 1) if value.is_integer() else "-999983"


def probes(env_id: str, prompt) -> list[tuple[str, str]]:
    """(probe name, junk response) pairs for one prompt: none of them is a correct answer."""
    user = next((m.get("content", "") for m in prompt.messages if m.get("role") == "user"), "")
    common = [("empty", ""), ("echo_prompt", user)]
    if env_id == "mbpp":
        return common + [("empty_code", "```python\n\n```"), ("pass_only", "```python\npass\n```"),
                         ("exit_early", "```python\nimport sys\nsys.exit(0)\n```"),
                         ("print_globals", "```python\nprint(globals())\n```")]
    if env_id == "countdown":
        out = common + [("empty_box", "\\boxed{}")]
        target, numbers = prompt.meta.get("target"), prompt.meta.get("numbers") or []
        if target is not None and target not in numbers:      # the bare target uses none of the numbers
            out.append(("target_literal", f"\\boxed{{{target}}}"))
        return out
    ref, wrong = str(prompt.reference), _wrong_number(prompt.reference)
    return common + [("empty_box", "\\boxed{}"), ("wrong_box", f"\\boxed{{{wrong}}}"),
                     ("number_spray", f"It is one of 0, 1, 2, {ref}, 10 or 100. \\boxed{{{wrong}}}")]


def fuzz_arm(env, env_id: str, prompts: Sequence) -> dict:
    per_probe: dict[str, list[float]] = {}
    for prompt in prompts:
        for name, text in probes(env_id, prompt):
            reward = float(env.compute_reward(env.evaluate_response(prompt, text)))
            per_probe.setdefault(name, []).append(reward)
    rates = {name: sum(v) / len(v) for name, v in per_probe.items()}
    n = sum(len(v) for v in per_probe.values())
    junk = sum(sum(v) for v in per_probe.values()) / n if n else 0.0
    return {"junk_reward_rate": round(junk, 4), "probes": {k: round(v, 4) for k, v in rates.items()},
            "flagged": bool(junk > FLAG_RATE or any(v >= 0.5 for v in rates.values()))}


_CACHE: dict[tuple, dict] = {}       # deterministic per arm and configuration: probe once per process


def fuzz_arms(envs: Mapping[str, Any], env_ids: Sequence[str], *, n_prompts: int = N_PROMPTS,
              seed: int = FUZZ_SEED) -> dict:
    """Probe every arm; an arm that cannot be probed is reported with its error, never flagged."""
    report: dict = {"n_prompts": n_prompts, "flag_rate": FLAG_RATE, "arms": {}}
    for e in env_ids:
        key = int(hashlib.sha256(e.encode()).hexdigest()[:8], 16)
        try:
            cache_key = (e, type(envs[e]).__qualname__, json.dumps(envs[e].metadata(), sort_keys=True, default=str),
                         n_prompts, seed)
            if cache_key not in _CACHE:
                prompts = envs[e].generate_batch("train", n_prompts, np.random.default_rng([seed, key]))
                _CACHE[cache_key] = fuzz_arm(envs[e], e, prompts)
            report["arms"][e] = dict(_CACHE[cache_key])
        except Exception as exc:  # noqa: BLE001 - a probe problem must never stop a run
            report["arms"][e] = {"error": repr(exc)[:300], "flagged": False}
    report["flagged"] = sorted(e for e, r in report["arms"].items() if r.get("flagged"))
    return report
