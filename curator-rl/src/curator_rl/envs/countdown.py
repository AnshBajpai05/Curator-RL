"""Countdown environment (procedural, no downloads).

Instances are fresh (Roadmap D.2): each call draws an instance seed from
the split's disjoint seed range (train < 1e7 <= calib < dev < 2e7 <= test),
so seed-range disjointness plus a content-hash check is the leakage
protection (Roadmap D.3 item 3 for procedural envs).

Verification is an AST-based safe evaluator - NEVER `eval` - with a strict
node whitelist, a bounded expression size, and exact rational (Fraction)
arithmetic.
"""

from __future__ import annotations

import ast
import operator
import re
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np

from curator_rl.core.types import CostPrior, Prompt, Verdict
from curator_rl.envs.base import Environment, run_with_timeout

# Disjoint instance-seed ranges per split (Roadmap D.2 Countdown): these are
# split-protocol constants, fixed alongside the salt in DECISIONS.md.
SPLIT_SEED_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 10_000_000),
    "calib": (10_000_000, 15_000_000),
    "dev": (15_000_000, 20_000_000),
    "test": (20_000_000, 30_000_000),
}
MAX_GENERATOR_ATTEMPTS = 100  # resamples numbers when no target is reachable in range
MAX_EXPR_LEN = 256
MAX_NESTING = 24

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

SYSTEM_PROMPT = "You are a precise arithmetic assistant."


def _answer_instruction(numbers: list[int], target: int, usage_rule: str) -> str:
    return (
        f"Using the numbers {numbers}, create an arithmetic expression using "
        f"+, -, * and / that equals exactly {target}. Use each number "
        f"{usage_rule}. Division is only allowed when it divides exactly. "
        "Return only the expression inside \\boxed{}."
    )


class CountdownError(ValueError):
    """The expression is malformed, uses banned nodes, or misuses numbers."""


_BIN_OPS = (
    ("+", operator.add),
    ("-", operator.sub),
    ("*", operator.mul),
    ("/", operator.truediv),
)

_ALLOWED_NODES = (
    ast.Expression,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Constant,
)


def safe_arith_eval(expression: str, allowed_numbers: list[int], require_all: bool,
                    exact_division: bool = False) -> Fraction:
    """Evaluate an arithmetic expression safely, exactly, and within the rules.

    Guarantees (contract test suite):
    - only + - * / with unary +/- and integer literals: a strict ast node
      whitelist, so no code execution of any kind;
    - bounded expression length and nesting depth;
    - only the allowed numbers are used, each at most once; when
      `require_all`, every allowed number is used exactly once;
    - exact rational arithmetic (no float blowups); division by zero is a
      CountdownError, not an exception escape;
    - with `exact_division`, every division must give an integer, as the prompt states
      ("Division is only allowed when it divides exactly"; audit E-19).
    """
    if len(expression) > MAX_EXPR_LEN:
        raise CountdownError("expression too long")
    if expression.count("(") > MAX_NESTING:
        raise CountdownError("expression nesting too deep")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CountdownError(f"invalid expression: {exc}") from exc
    for node in ast.walk(tree):
        # bool is an int subclass: `True` must not stand in for the number 1
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, int)):
            raise CountdownError("only integer literals are allowed")
        if not isinstance(node, _ALLOWED_NODES):
            raise CountdownError(f"banned node: {ast.dump(node)[:80]}")
    used = [cast(int, node.value) for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    available = list(allowed_numbers)
    for value in used:
        if value in available:
            available.remove(value)
        else:
            raise CountdownError(f"number {value} is not available (each at most once)")
    if require_all and available:
        raise CountdownError(f"unused numbers: {available}")

    def _eval(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant):
            return Fraction(cast(int, node.value))      # validated above: an integer literal
        if isinstance(node, ast.UnaryOp):
            inner = _eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -inner
            if isinstance(node.op, ast.UAdd):
                return inner
            raise CountdownError("banned unary operator")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise CountdownError("division by zero")
                if exact_division and (left / right).denominator != 1:
                    raise CountdownError("inexact division")
                return left / right
            raise CountdownError("banned binary operator")
        raise CountdownError(f"unexpected node: {ast.dump(node)[:80]}")

    return _eval(tree.body)


@lru_cache(maxsize=4096)
def _reachable_targets(numbers: tuple[int, ...], exact_division: bool = False) -> dict[Fraction, str]:
    """All exactly-reachable target values with one example expression each.

    Enumerates unordered partitions of the number multiset recursively over
    index bitmasks; '-' and '/' are non-commutative, so both orders (a, b)
    and (b, a) appear because every proper submask is enumerated. One example
    expression per reachable value.
    """
    memo: dict[int, dict[Fraction, str]] = {}

    def rec(mask: int) -> dict[Fraction, str]:
        if mask in memo:
            return memo[mask]
        if mask & (mask - 1) == 0:
            idx = mask.bit_length() - 1
            node = {Fraction(numbers[idx]): str(numbers[idx])}
            memo[mask] = node
            return node
        out: dict[Fraction, str] = {}
        sub = (mask - 1) & mask
        while sub:
            a, b = sub, mask ^ sub
            for lv, le in rec(a).items():
                for rv, rel in rec(b).items():
                    for sym, op in _BIN_OPS:
                        if sym == "/" and (rv == 0 or (exact_division and (lv / rv).denominator != 1)):
                            continue
                        value = op(lv, rv)
                        if value not in out:
                            out[value] = f"({le} {sym} {rel})"
            sub = (sub - 1) & mask
        memo[mask] = out
        return out

    return rec((1 << len(numbers)) - 1)


def _draw_instance(cfg, split: str, rng) -> tuple[int, list[int], int, str]:
    """Draw (seed, numbers, target, solution_expr) for a split. Deterministic given rng."""
    lo, hi = SPLIT_SEED_RANGES[split]
    for _ in range(MAX_GENERATOR_ATTEMPTS):
        seed = int(rng.integers(lo, hi))
        local = np.random.default_rng(seed)
        numbers = [int(v) for v in local.integers(cfg.numbers_min, cfg.numbers_max + 1, size=cfg.n_numbers)]
        reachable = _reachable_targets(tuple(numbers), bool(cfg.exact_division))
        candidates = [
            value
            for value in reachable
            if value.denominator == 1 and cfg.target_min <= int(value) <= cfg.target_max
        ]
        if not candidates:
            continue
        pos = int(local.integers(len(candidates)))
        target = candidates[pos]
        return seed, numbers, int(target), reachable[target]
    raise ValueError(
        f"no reachable Countdown target in [{cfg.target_min}, {cfg.target_max}] "
        f"after {MAX_GENERATOR_ATTEMPTS} attempts (n_numbers={cfg.n_numbers})"
    )


def _verify_boxed(boxed: str, numbers: list[int], target: int, require_all: bool,
                  exact_division: bool = False) -> bool:
    try:
        value = safe_arith_eval(boxed, numbers, require_all, exact_division)
    except CountdownError:
        return False
    return value == Fraction(target)


class CountdownEnv(Environment):
    """Procedural Countdown: fresh instances per prompt, seed-range splits."""

    def __init__(self, repo_root: Path | None = None, cfg=None) -> None:
        super().__init__("countdown")
        self.cfg: Any = cfg
        self.repo_root = Path(repo_root) if repo_root is not None else Path(".")

    def split_ids(self, split: str) -> list[str]:
        if split not in SPLIT_SEED_RANGES:
            raise ValueError(f"unknown split '{split}'")
        return []  # procedural: ids are drawn, membership is by seed range

    def generate_prompt(self, split: str, rng) -> Prompt:
        if split not in SPLIT_SEED_RANGES:
            raise ValueError(f"unknown split '{split}'")
        seed, numbers, target, solution = _draw_instance(self.cfg, split, rng)
        usage_rule = "exactly once" if self.cfg.use_all_numbers else "at most once"
        return Prompt(
            prompt_id=f"cd-{split}-{seed}",
            env_id=self.env_id,
            split=split,
            messages=(
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _answer_instruction(numbers, target, usage_rule)},
            ),
            reference=str(target),
            meta={"numbers": numbers, "target": target, "solution": solution, "seed": seed},
        )

    def generate_batch(self, split: str, n: int, rng, exclude: set[str] | None = None) -> list[Prompt]:
        if split not in SPLIT_SEED_RANGES:
            raise ValueError(f"unknown split '{split}'")
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
        matches = _BOXED_RE.findall(response)
        boxed = matches[-1].strip() if matches else None
        parse_ok = boxed is not None
        success = False
        if boxed is not None:
            try:
                success = run_with_timeout(
                    lambda: _verify_boxed(
                        boxed,
                        list(prompt.meta["numbers"]),
                        int(prompt.meta["target"]),
                        self.cfg.use_all_numbers,
                        bool(self.cfg.exact_division),
                    ),
                    self.cfg.verifier_timeout_s,
                )
            except Exception:
                success = False
        info: dict[str, Any] = {"boxed": boxed}
        return Verdict(
            success=success,
            score=1.0 if success else 0.0,
            parse_ok=parse_ok,
            verifier_seconds=time.perf_counter() - start,
            info=info,
        )

    def compute_reward(self, verdict: Verdict) -> float:
        return verdict.score

    def estimate_cost(self) -> CostPrior:
        return CostPrior(
            env_id=self.env_id,
            per_prompt_usd=self.cfg.prior_usd_per_prompt,
            note="config prior for cold start only (Roadmap H.5)",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "dataset": "procedural countdown (generator parameters below)",
            "n_numbers": self.cfg.n_numbers,
            "numbers_range": [self.cfg.numbers_min, self.cfg.numbers_max],
            "target_range": [self.cfg.target_min, self.cfg.target_max],
            "split_seed_ranges": SPLIT_SEED_RANGES,
            "max_completion_tokens": self.cfg.max_completion_tokens,
            "max_prompt_tokens": self.cfg.max_prompt_tokens,
            "verifier_timeout_s": self.cfg.verifier_timeout_s,
        }
