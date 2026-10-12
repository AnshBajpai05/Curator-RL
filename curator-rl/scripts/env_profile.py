"""Per-environment throughput and verifier-time profile (Roadmap D.5 item 6).

Writes reports/env_profiles/phase2_profile.json with, per environment:
- prompt generation rate (prompts/s)
- verifier rate and mean verifier_seconds on gold and garbage responses
Usage: python scripts/env_profile.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.envs.countdown import CountdownEnv  # noqa: E402
from curator_rl.envs.gsm8k import Gsm8kEnv  # noqa: E402
from curator_rl.envs.noisy import NoisyRewardEnv  # noqa: E402

N_PROMPTS = 200
N_VERDICTS = 200


def profile_env(name, env):
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    prompts = env.generate_batch("train", N_PROMPTS, rng)
    gen_seconds = time.perf_counter() - t0

    gold_times = []
    gold_success = 0
    for p in prompts[:N_VERDICTS]:
        gold_answer = p.meta.get("solution", p.reference)
        v = env.evaluate_response(p, f"... \\boxed{{{gold_answer}}}")
        gold_times.append(v.verifier_seconds)
        gold_success += v.success
    junk_times = []
    for i, p in enumerate(prompts[:N_VERDICTS]):
        v = env.evaluate_response(p, f"filler {i} \\boxed{{banana}}")
        junk_times.append(v.verifier_seconds)
    return {
        "env_id": name,
        "prompts_sampled": len(prompts),
        "prompt_generation_per_s": round(len(prompts) / gen_seconds, 1),
        "verifier_mean_seconds_gold": round(sum(gold_times) / len(gold_times), 6),
        "verifier_mean_seconds_garbage": round(sum(junk_times) / len(junk_times), 6),
        "gold_success_rate": round(gold_success / len(gold_times), 4),
    }


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    built = (REPO_ROOT / "data" / "processed" / "gsm8k" / "train.jsonl").exists()
    candidates = [
        ("countdown", CountdownEnv(REPO_ROOT, cfg.data.envs.countdown)),
        ("gsm8k", Gsm8kEnv(REPO_ROOT, cfg.data.envs.gsm8k)) if built else None,
        ("noisy", NoisyRewardEnv(REPO_ROOT, cfg.data.envs.noisy)) if built else None,
    ]
    out = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "envs": []}
    for entry in candidates:
        if entry is None:
            continue
        name, env = entry
        out["envs"].append(profile_env(name, env))
    dest = REPO_ROOT / "reports" / "env_profiles" / "phase2_profile.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
