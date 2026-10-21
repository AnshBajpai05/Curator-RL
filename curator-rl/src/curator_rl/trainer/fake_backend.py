"""CPU stand-in for the GPU trainer (tests, dry runs, `run.py --backend fake`).

A toy learner with one latent skill per arm answers prompts with REAL gold answers (or wrong ones)
so the real verifiers, signal engine, scheduler, calibration, ledger and checkpoints all run end
to end without a GPU. Its pieces mirror the simulator: pass probability sigmoid(2 (s - d)),
per-prompt difficulty Beta(kappa p, kappa (1 - p)) with the measured kappa = 2.14, learning
proportional to the delivered share times 4 p (1 - p), GSM8K <-> MATH35 transfer 0.3, a noisy arm
that teaches nothing. Step time follows a known per-arm token cost, so the cost-model fit of the
system-identification report can be checked against the truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from curator_rl.core.types import Prompt
from curator_rl.trainer.group_filter import has_signal, select_groups

DIFFICULTY = {"gsm8k": 0.31, "math35": 0.80, "mbpp": 0.71, "countdown": 4.0, "noisy": 0.31}
ETA = {"gsm8k": 0.015, "math35": 0.012, "mbpp": 0.012, "countdown": 0.01, "noisy": 0.0}
TRANSFER = {("math35", "gsm8k"): 0.3, ("gsm8k", "math35"): 0.3}   # (target, source)
TOKENS = {"gsm8k": 340, "math35": 650, "mbpp": 150, "countdown": 290, "noisy": 310}
SECONDS_PER_1K_TOKENS = {"gsm8k": 2.2, "math35": 2.6, "mbpp": 3.0, "countdown": 2.2, "noisy": 2.2}
STEP_OVERHEAD_SECONDS = 4.0
KAPPA = 2.14


def _unit(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


class FakePolicy:
    def __init__(self, envs: Mapping[str, object], seed: int) -> None:
        self.envs = dict(envs)
        self.skill = {e: 0.0 for e in self.envs}
        self.rng = np.random.default_rng([seed, 4242])
        self._mbpp_code: dict[str, str] | None = None

    def p(self, env_id: str) -> float:
        return 1.0 / (1.0 + math.exp(-2.0 * (self.skill[env_id] - DIFFICULTY.get(env_id, 0.5))))

    def _gold(self, prompt: Prompt) -> str | None:
        e = prompt.env_id
        if e in ("gsm8k", "noisy"):
            return f"\\boxed{{{prompt.reference}}}" if "/" not in prompt.reference else None
        if e == "math35":
            return f"\\boxed{{{prompt.reference}}}"
        if e == "countdown":
            return f"\\boxed{{{prompt.meta['solution']}}}"
        if e == "mbpp":
            if self._mbpp_code is None:
                env = self.envs["mbpp"]
                self._mbpp_code = {it["problem_id"]: it["code"] for split in ("train", "calib", "dev")
                                   for it in getattr(env, "_items", {}).get(split, [])}
            code = self._mbpp_code.get(prompt.prompt_id)
            return f"```python\n{code}\n```" if code else None
        return None

    @staticmethod
    def _wrong(env_id: str) -> str:
        return "```python\ndef unused():\n    return None\n```" if env_id == "mbpp" else "\\boxed{-999983}"

    def sample_group(self, prompt: Prompt, g: int) -> list[str]:
        p = min(max(self.p(prompt.env_id), 1e-6), 1 - 1e-6)
        q = self.rng.beta(KAPPA * p, KAPPA * (1 - p))
        gold = self._gold(prompt)
        return [gold if (gold is not None and self.rng.random() < q) else self._wrong(prompt.env_id)
                for _ in range(g)]

    def greedy(self, prompts: Sequence[Prompt], cap: int) -> list[str]:
        """Deterministic per item: correct iff its fixed difficulty draw is below the pass rate."""
        out = []
        for pr in prompts:
            gold = self._gold(pr)
            ok = gold is not None and _unit("greedy", pr.prompt_id) < self.p(pr.env_id)
            out.append(gold if ok else self._wrong(pr.env_id))
        return out

    def sampled(self, prompts: Sequence[Prompt], cap: int, k: int, seeds: Sequence[int]) -> list[list[str]]:
        """k samples per item, each correct with the current pass rate; drawn from the item's own seed,
        never from the training generator (an evaluation must not change the run it measures)."""
        out = []
        for pr, seed in zip(prompts, seeds):
            rng = np.random.default_rng([int(seed), 7])
            gold = self._gold(pr)
            out.append([gold if (gold is not None and rng.random() < self.p(pr.env_id)) else self._wrong(pr.env_id)
                        for _ in range(int(k))])
        return out

    def learn(self, prompts_by_env: Mapping[str, int]) -> None:
        total = sum(prompts_by_env.values()) or 1
        g = {e: 4 * self.p(e) * (1 - self.p(e)) for e in self.skill}
        share = {e: prompts_by_env.get(e, 0) / total for e in self.skill}
        for target in self.skill:
            inflow = share[target] * g[target]
            for (t, src), coef in TRANSFER.items():
                if t == target and src in share:
                    inflow += coef * share[src] * g[src]
            self.skill[target] += ETA.get(target, 0.0) * inflow * 2.0

    def get_state(self) -> dict:
        return {"skill": dict(self.skill), "rng": self.rng.bit_generator.state}

    def load_state(self, state: Mapping) -> None:
        self.skill = {e: float(v) for e, v in state["skill"].items()}
        self.rng.bit_generator.state = state["rng"]


class FakeTrainer:
    """Drives a RoundController exactly as the TRL callback does, one optimizer step at a time."""

    def __init__(self, controller, seed: int) -> None:
        self.ctl = controller
        self.policy = FakePolicy(controller.envs, seed)
        self.ctl.generate = self.policy.greedy
        self.ctl.generate_sampled = self.policy.sampled
        self.len_rng = np.random.default_rng([seed, 99])

    def save(self, ckpt_dir: Path) -> None:
        state = {"policy": self.policy.get_state(), "len_rng": self.len_rng.bit_generator.state}
        (Path(ckpt_dir) / "fake_policy.json").write_text(json.dumps(state), encoding="utf-8")

    def load(self, ckpt_dir: Path) -> FakeTrainer:
        state = json.loads((Path(ckpt_dir) / "fake_policy.json").read_text(encoding="utf-8"))
        self.policy.load_state(state["policy"])
        self.len_rng.bit_generator.state = state["len_rng"]
        return self

    def train(self, on_checkpoint=None, *, stop_after_steps: int | None = None) -> None:
        cfg = self.ctl.cfg
        g, n_rows = cfg.group_size, cfg.prompts_per_step * cfg.group_size
        steps = 0
        while not self.ctl.stop:
            n_keep = cfg.prompts_per_step
            n_draw = self.ctl.candidates_per_step(n_keep)          # > n_keep with question filtering (D-102)
            rows = self.ctl.fill_generation_batch([{}] * n_rows, g, n_draw)
            completions, ids, tokens = [], [], {}
            for i in range(0, len(rows), g):
                row = rows[i]
                prompt = Prompt(prompt_id=row["prompt_id"], env_id=row["env_id"], split="train", messages=(),
                                reference=row["reference"], meta=json.loads(row["meta_json"]))
                # true cost follows prompt + completion tokens, exactly what the step log records
                tokens[row["env_id"]] = tokens.get(row["env_id"], 0) + int(
                    self.ctl.router.count_prompt_tokens(row["prompt"]))
                for text in self.policy.sample_group(prompt, g):
                    n_tok = max(int(self.len_rng.normal(TOKENS.get(row["env_id"], 300), 40)), 8)
                    completions.append([{"role": "assistant", "content": text}])
                    ids.append([0] * n_tok)
                    tokens[row["env_id"]] = tokens.get(row["env_id"], 0) + n_tok
            cols = {k: [r[k] for r in rows] for k in ("env_id", "prompt_id", "reference", "meta_json")}
            rewards = self.ctl.router(prompts=[r["prompt"] for r in rows], completions=completions,
                                      completion_ids=ids, **cols)
            kept = range(n_draw)
            if n_draw > n_keep:          # train on the groups with a reward signal; all of them were generated
                kept = select_groups([has_signal(rewards[i * g:(i + 1) * g]) for i in range(n_draw)], n_keep)
            counts: dict[str, int] = {}
            for k in kept:
                counts[rows[k * g]["env_id"]] = counts.get(rows[k * g]["env_id"], 0) + 1
            self.policy.learn(counts)
            # a real step's wall time includes scoring: add the measured verification wall time
            seconds = STEP_OVERHEAD_SECONDS + sum(SECONDS_PER_1K_TOKENS.get(e, 2.0) * t / 1000.0
                                                  for e, t in tokens.items())
            seconds += self.ctl.router._record.score_wall_seconds  # noqa: SLF001 - same package
            self.ctl.on_step_end(seconds)
            if self.ctl.checkpoint_due and on_checkpoint is not None:
                on_checkpoint()
                self.ctl.checkpoint_due = False
            steps += 1
            if stop_after_steps is not None and steps >= stop_after_steps:
                return
