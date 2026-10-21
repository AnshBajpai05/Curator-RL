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
             "noisy": "gsm8k", "hackable": "gsm8k"}
# stress-test portfolio (D-102): the random-reward arm replaced by one whose checker has a false-positive bug
STRESS_ENVS = ("gsm8k", "math35", "mbpp", "countdown", "hackable")
# benchmark = macro-average over domains with a test set (Roadmap v3 4.8); noisy has none
BENCH_DOMAINS = ("gsm8k", "math35", "mbpp", "countdown")


@dataclass
class RunConfig:
    method: str = "curator"              # curator[_variant] | uniform | static | pilot_uniform | lp | ucb | sec | dump | dirichlet
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
    eval_samples: int = 4                # reports: samples per item for the sampled accuracy (D-101; 0 = greedy only)
    eval_temperature: float = 1.0        # ... drawn at the training temperature
    group_filter: float = 0.0            # questions drawn per question trained ("_q" methods, D-102); 0 = off
    checkpoint_every: int = 1            # rounds; every round => a crash loses at most one round
    keep_checkpoints: int = 3            # newest checkpoints kept per run; older ones are deleted
    sync_to: str = ""                    # mirror artifacts while training: hf://owner/repo[/prefix] or a directory
    sync_every_min: float = 5.0          # minimum minutes between background uploads (a final one always runs)
    deadline_unix: float = 0.0           # wall-clock limit (e.g. Kaggle's 12 h): stop cleanly before it; 0 = none
    verify_workers: int = 4              # MBPP verification is CPU-bound (D-87)
    vllm_gpu_memory_utilization: float = 0.3
    out_dir: str = "runs"
    label: str = ""                      # run name (default: method); leave-one-out runs are "<method>-wo-<arm>"
    overrides: list[str] = field(default_factory=list)   # base.yaml overrides, e.g. scheduler.gamma=1.0

    @property
    def name(self) -> str:
        return self.label or self.method

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
