"""Shape and budget of one real GRPO run under a scheduler (Phase F).

Scientific hyperparameters (scheduler, signals, calibration) still come from `configs/base.yaml`;
this dataclass holds the run's SHAPE (model, batch, horizon, budget, evaluation sizes), which the
Kaggle runner chooses per hardware. Defaults are the measured Level-1 plan (D-84/D-87): Qwen2.5-0.5B,
R = 2 optimizer steps per round, P = 16 prompts x G = 8 completions per step, about 60 rounds.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

LEVEL1_ENVS = ("gsm8k", "math35", "mbpp", "countdown", "noisy")
# arm -> calibration slice it is scored on (the real noisy arm has no slice of its own, audit E-03)
DOMAIN_OF = {"gsm8k": "gsm8k", "math35": "math35", "mbpp": "mbpp", "countdown": "countdown",
             "noisy": "gsm8k"}
# benchmark = macro-average over domains with a test set (Roadmap v3 4.8); noisy has none
BENCH_DOMAINS = ("gsm8k", "math35", "mbpp", "countdown")


@dataclass
class RunConfig:
    method: str = "curator"              # curator | uniform | static | lp | ucb | sec | dump | dirichlet
    seed: int = 0
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    backend: str = "auto"                # auto | vllm | hf | fake
    envs: tuple[str, ...] = LEVEL1_ENVS
    steps_per_round: int = 2             # R (D-84)
    prompts_per_step: int = 16           # P
    group_size: int = 8                  # G
    micro_batch: int = 8                 # completions per forward/backward pass
    max_completion_tokens: int = 1024    # one cap per trainer; MATH35 needs 1024 (pilot: 61% cut at 512)
    learning_rate: float = 1e-5
    lora_r: int = 16
    max_rounds: int = 60
    budget_gpu_hours: float = 1.5        # charged GPU-hours (training + scheduler-owned calibration)
    calib_interval: int = 10             # K
    calib_items: int = 100               # paired items per slice
    report_every: int = 20               # uncharged dev evaluation every N rounds (0 = final only)
    report_items: int = 100              # dev items per domain for the periodic reports
    final_items: int = 300               # dev items per domain for the final score
    checkpoint_every: int = 5            # rounds
    verify_workers: int = 4              # MBPP verification is CPU-bound (D-87)
    vllm_gpu_memory_utilization: float = 0.3
    out_dir: str = "runs"
    overrides: list[str] = field(default_factory=list)   # base.yaml overrides, e.g. scheduler.gamma=1.0

    @property
    def prompts_per_round(self) -> int:
        return self.steps_per_round * self.prompts_per_step

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict) -> RunConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in raw.items() if k in known}
        if "envs" in data:
            data["envs"] = tuple(data["envs"])
        return cls(**data)
