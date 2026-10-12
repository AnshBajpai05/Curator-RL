"""Reward function for a mixed-environment GRPO batch (Phase F; D-87 parallel verification).

TRL calls the reward function once per generation batch with every completion and the dataset
columns of its prompt (`env_id`, `prompt_id`, `reference`, `meta_json`). The router sends each
completion to its environment's verifier through a thread pool (MBPP's sandbox costs about 1.3 s
per prompt serially, 0.33 s with 4 threads), returns the shaped reward, and accumulates exactly
the per-arm statistics a `RoundObservation` needs: prompts, rollouts, successes, mixed groups,
score moments, mean |advantage|, tokens and verifier seconds. Pure Python: the TRL adapter, the
fake backend and the tests all drive the same object.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from curator_rl.core.advantage import group_mean_abs_advantage
from curator_rl.core.types import Prompt, Verdict
from curator_rl.envs.base import Environment


@dataclass
class ArmStats:
    n_prompts: int = 0
    n_rollouts: int = 0
    k_success: int = 0
    n_groups_mixed: int = 0
    sum_score: float = 0.0
    sum_score_sq: float = 0.0
    sum_abs_adv: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verifier_seconds: float = 0.0
    true_success: int = 0          # honest verdicts (differs from k_success only on the noisy arm)
    rejected: int = 0              # sandbox screening refusals (reward-hacking attempts, E-07)
    truncated: int = 0
    verifier_errors: int = 0       # verifier raised: scored 0 and counted, never allowed to stop training

    def merge(self, other: ArmStats) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


@dataclass
class StepRecord:
    arms: dict[str, ArmStats] = field(default_factory=dict)
    score_wall_seconds: float = 0.0

    def arm(self, env_id: str) -> ArmStats:
        return self.arms.setdefault(env_id, ArmStats())


def _text(completion) -> str:
    if isinstance(completion, list):          # conversational: [{"role": "assistant", "content": ...}]
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


class RewardRouter:
    """Routes completions to verifiers; callable with TRL's reward-function signature."""

    __name__ = "curator_reward"   # TRL names reward functions in its logs

    def __init__(self, envs: Mapping[str, Environment], *, group_size: int, workers: int = 4,
                 count_prompt_tokens: Callable[[Sequence[dict]], int] | None = None,
                 max_completion_tokens: int | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.envs = dict(envs)
        self.group_size = int(group_size)
        self.pool = ThreadPoolExecutor(max_workers=max(int(workers), 1))
        self.count_prompt_tokens = count_prompt_tokens or (lambda msgs: sum(len(m.get("content", "")) for m in msgs) // 4)
        self.max_completion_tokens = max_completion_tokens
        self.clock = clock
        self._record = StepRecord()
        self._prompt_tokens_cache: dict[str, int] = {}
        self.errors_seen = 0

    def pop_step(self) -> StepRecord:
        rec, self._record = self._record, StepRecord()
        return rec

    def _verify(self, job):
        env_id, prompt, text = job
        env = self.envs[env_id]
        try:
            verdict = env.evaluate_response(prompt, text)
            reward = float(env.compute_reward(verdict))
        except Exception as exc:  # noqa: BLE001 - one bad completion must not kill a GPU run
            self.errors_seen += 1
            if self.errors_seen <= 20:
                print(f"[reward] verifier error on {env_id}/{prompt.prompt_id}: {exc!r}", flush=True)
            verdict = Verdict(success=False, score=0.0, parse_ok=False, verifier_seconds=0.0,
                              info={"verifier_error": repr(exc)[:200]})
            reward = 0.0
        return verdict, reward

    def __call__(self, prompts, completions, completion_ids=None, *, env_id, prompt_id, reference,
                 meta_json, **_kw) -> list[float]:
        start = self.clock()
        jobs = []
        position = 0
        for i, comp in enumerate(completions):
            # index of this rollout inside its prompt's group (groups are contiguous, at most G long):
            # lets a reward that should be independent per rollout key on it (audit E-24)
            position = position + 1 if i > 0 and prompt_id[i] == prompt_id[i - 1] and position + 1 < self.group_size else 0
            meta = json.loads(meta_json[i] or "{}")
            meta["rollout_index"] = position
            p = Prompt(prompt_id=prompt_id[i], env_id=env_id[i], split="train", messages=(),
                       reference=reference[i], meta=meta)
            jobs.append((env_id[i], p, _text(comp)))
        results = list(self.pool.map(self._verify, jobs))
        rewards = [r for _, r in results]
        self._record.score_wall_seconds += self.clock() - start

        # group consecutive completions of the same prompt (TRL keeps a prompt's G completions
        # contiguous; TRL_SPIKE G.2 #1) and fold them into the arm statistics
        i = 0
        while i < len(jobs):
            j = i
            while j < len(jobs) and prompt_id[j] == prompt_id[i] and j - i < self.group_size:
                j += 1
            env = env_id[i]
            arm = self._record.arm(env)
            k = sum(1 for r in rewards[i:j] if r >= 0.5)
            g = j - i
            arm.n_prompts += 1
            arm.n_rollouts += g
            arm.k_success += k
            arm.n_groups_mixed += int(0 < k < g)
            rate = k / g
            arm.sum_score += rate
            arm.sum_score_sq += rate * rate
            arm.sum_abs_adv += group_mean_abs_advantage(k, g) if g >= 2 else 0.0
            pid = prompt_id[i]
            if pid not in self._prompt_tokens_cache:
                msgs = prompts[i] if isinstance(prompts[i], list) else [{"content": str(prompts[i])}]
                self._prompt_tokens_cache[pid] = int(self.count_prompt_tokens(msgs))
            arm.prompt_tokens += self._prompt_tokens_cache[pid]
            for t in range(i, j):
                verdict = results[t][0]
                arm.verifier_seconds += float(verdict.verifier_seconds)
                arm.true_success += int(bool(verdict.info.get("diagnostic_true_success", verdict.success)))
                arm.rejected += int(bool(verdict.info.get("rejected")))
                arm.verifier_errors += int("verifier_error" in verdict.info)
                n_tok = len(completion_ids[t]) if completion_ids is not None else len(jobs[t][2]) // 4
                arm.completion_tokens += n_tok
                if self.max_completion_tokens and n_tok >= self.max_completion_tokens:
                    arm.truncated += 1
            i = j
        return rewards

    def close(self) -> None:
        self.pool.shutdown(wait=False)
