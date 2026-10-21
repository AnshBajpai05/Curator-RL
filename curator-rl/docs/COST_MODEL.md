# COST MODEL

Source: Roadmap v2.0 Part H. **Status:** sections 1-5 are the definitions; sections 6-7 hold the Kaggle
measurements (Phase B pilot, Phase E cost probe). Implemented: `cost/attribution.py` (measured,
CUDA-synchronised step time attributed to arms: verification exactly, the rest by tokens),
`cost/ledger.py` (matched-budget ledger), and the per-arm cost model fitted by NNLS from one
randomized-mixture run (`trainer/sysid.py`, Gate 5 on hardware: still to be run). Any `TBD` left below
is replaced by a measurement and logged in `DECISIONS.md`.

## 1. Definition

```
gpu_seconds = wall_seconds × n_gpus            reserved-GPU model
usd         = gpu_seconds × usd_per_gpu_hour / 3600
```

If no real price applies (Kaggle is free), set `usd_per_gpu_hour = 1.0` and report **GPU-units**. Verifier time counts as GPU time unless `cost.count_cpu_verifier_as_gpu_time` is turned off because verification is fully overlapped with generation; the choice is stated in every report.

## 2. Spans (H.2)

`GEN`, `SCORE`, `UPDATE`, `EVAL` (calibration vs reporting kept separate), `OVERHEAD`. Clocks: `time.perf_counter_ns()` with a device synchronize before each boundary read. Every wall-clock second belongs to one span or to `unattributed`, which must stay below 3%.

## 3. Attribution to environments (H.4)

- `SCORE`: exact, split by per-item verifier seconds.
- `GEN`: split by `a_g·completion_tokens + b_g·n_seqs + c_g·prompt_tokens` (coefficients TBD from micro-benchmarks).
- `UPDATE`: split by `a_u·tokens + b_u·n_seqs` (TBD).
- Shares are rescaled each step so the attributed total equals the measured total.
- Validation (Gate 5): attributed per-environment cost within 10% of exact single-environment timing.

## 4. Budget ledger (H.6)

| Category | Charged to B |
|---|---|
| `train_gen`, `train_score`, `train_update` | yes |
| `calib_eval` | yes |
| `curator_overhead` | yes |
| `report_eval` | no (identical for every method) |
| `setup` (e.g. too-hard filter run) | no, but logged |

## 5. Platform: Kaggle (v3 §4.9)

- 2×T4 per session; two seeds run in parallel, one per GPU. fp16 only (no bf16).
- All runs in a comparison use the same GPU type; costs in GPU-seconds are not comparable across hardware.
- Kaggle machines are shared, so wall-clock cost is noisier than on a dedicated GPU. Plan: record GPU type, driver and CPU count per run; run the drift monitor (identical micro-batch every N steps; > 10% drift raises a flag). If unit-cost CV stays above 10%, switch to a token-based cost model (attributed tokens × a fitted cost per token), validated against wall-clock, and state this here.

## 6. Measured in the Phase B pilot (Kaggle T4, fp16, HF generate, G = 8, 512 max tokens)

| Quantity | Value |
|---|---|
| Unit cost per prompt, Qwen2.5-0.5B (gen + verify) | gsm8k 3.44 s, countdown 3.23 s, noisy 3.55 s (rel. 1.00 / 0.94 / 1.03) |
| Unit cost per prompt, Qwen2.5-1.5B | gsm8k 6.35 s, countdown 6.09 s, noisy 6.67 s (1.85× the 0.5B) |
| Unit-cost CV across batches | 0.00–0.07 (target < 0.10) |
| Verifier seconds per prompt | negligible against generation for these three environments |
| Choice: wall-clock vs token-based cost model | **wall-clock** (CV under target) |
| Step time at the real batch shape (P = 16, G = 8, 512 tokens), vLLM colocate | **42.2 s** (37–41 s per step; HF backend TBD) |
| Calibration-eval time per item | TBD |
| Unattributed fraction | TBD (needs the cost meter inside the trainer) |
| Cost ordering sanity (code / long reasoning above GSM8K) | MATH35 above GSM8K (2.67x), MBPP close to GSM8K (1.23x, verifier-bound); see section 7 |

## 7. Measured in the Phase E cost probe (Kaggle T4, vLLM 0.31, G = 8, each env at its own cap; D-87)

| Quantity | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---|---|
| Generation s/prompt: gsm8k / math35 / mbpp / countdown / noisy | 0.524 / 1.403 / 0.317 / 0.535 / 0.526 | 1.100 / 3.213 / 0.845 / 0.706 / 1.072 |
| Verifier s/prompt, MBPP (serial / 4-thread pool) | 1.298 / 0.328 | 1.238 / 0.314 |
| Relative unit cost, gen + pooled verify (gsm8k = 1) | math35 2.67, mbpp 1.23, countdown 1.02, noisy 1.00 | math35 2.92, mbpp 1.05, countdown 0.64, noisy 0.97 |
| Relative tokens per prompt (gsm8k = 1) | math35 1.88, mbpp 0.43, countdown 0.84, noisy 0.90 | math35 2.28, mbpp 0.52, countdown 0.30, noisy 0.92 |
| GRPO step, HF backend (P = 16, 512 tokens) | 83.5 s (generation 52.0 s) | — |
| GRPO step, vLLM colocate | 42.2 s (generation ≈ 8.4 s, update + sync ≈ 34 s) | — |
| Training generation backend | **vLLM colocate** (D-87) | |
| Training-cost bracket (update fully token-proportional .. half fixed) | math35 1.7-2.0x, mbpp 0.6-0.8x of gsm8k | |
| Per-run estimate (60 rounds, R = 2) | 1.4-2.8 GPU-h | |

The update dominates a vLLM step, so per-environment training cost is not the generation cost alone. Until the cost meter measures attribution (Gate 5), simulator refits use both ends of the bracket.
