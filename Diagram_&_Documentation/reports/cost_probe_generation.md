# Phase E cost probe (Kaggle, 2026-10-09): unit cost under vLLM, TRL smoke test, HF step time

Measured with `notebooks/kaggle_cost_probe.ipynb` on Tesla T4 GPUs.
- **Cost probe** (`scripts/kaggle_cost_probe.py`): vLLM 0.31.0, torch 2.13.0+cu130 in a separate virtualenv. G = 8,
  16 prompts per chunk (one GRPO step's generation), 3 timed chunks after one warm-up chunk, each environment at its
  own token cap (MATH35 1024, the rest 512). Verifier time is reported serial and through a 4-thread pool.
  Prompts come from the same RNG streams as the pilot (`pilot_<env>`), dev split (noisy: train).
- **TRL smoke test and HF step probe**: kernel Python, torch 2.11, TRL 1.15.0, PEFT 0.20.0, torchao removed (D-86).

## Cost probe, Qwen2.5-0.5B-Instruct

| env | cap | pass@1 | pass@8 | mixed | trunc | tokens/prompt | gen s/prompt | verify s/prompt (serial / pool) | rel. cost (pool) | rel. tokens | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 512 | 0.350 | 0.797 | 0.766 | 0.100 | 2755 | 0.524 | 0.008 / 0.002 | 1.00x | 1.00x | 0.05 |
| math35 | 1024 | 0.168 | 0.500 | 0.500 | 0.125 | 5192 | 1.403 | 0.008 / 0.002 | 2.67x | 1.88x | 0.05 |
| mbpp | 512 | 0.194 | 0.483 | 0.467 | 0.000 | 1187 | 0.317 | 1.298 / 0.328 | 1.23x | 0.43x | 0.10 |
| countdown | 512 | 0.000 | 0.000 | 0.000 | 0.352 | 2311 | 0.535 | 0.002 / 0.001 | 1.02x | 0.84x | 0.00 |
| noisy | 512 | 0.373 | 1.000 | 1.000 | 0.064 | 2479 | 0.526 | 0.009 / 0.002 | 1.00x | 0.90x | 0.06 |

Cost spread among learnable environments (max/min, generation + pooled verification): **2.67x**.

## Cost probe, Qwen2.5-1.5B-Instruct

| env | cap | pass@1 | pass@8 | mixed | trunc | tokens/prompt | gen s/prompt | verify s/prompt (serial / pool) | rel. cost (pool) | rel. tokens | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 512 | 0.537 | 0.906 | 0.844 | 0.029 | 2117 | 1.100 | 0.008 / 0.002 | 1.00x | 1.00x | 0.08 |
| math35 | 1024 | 0.346 | 0.641 | 0.594 | 0.090 | 4831 | 3.213 | 0.009 / 0.002 | 2.92x | 2.28x | 0.06 |
| mbpp | 512 | 0.415 | 0.750 | 0.650 | 0.004 | 1104 | 0.845 | 1.238 / 0.314 | 1.05x | 0.52x | 0.13 |
| countdown | 512 | 0.000 | 0.000 | 0.000 | 0.057 | 642 | 0.706 | 0.002 / 0.001 | 0.64x | 0.30x | 0.05 |
| noisy | 512 | 0.332 | 0.969 | 0.969 | 0.006 | 1940 | 1.072 | 0.008 / 0.002 | 0.97x | 0.92x | 0.06 |

Cost spread among learnable environments: **2.99x**.

## GRPO step time, HF backend (P = 16 prompts x G = 8, 512 tokens, LoRA, fp16, GSM8K train)

| Quantity | HF backend | vLLM colocate (D-83) |
|---|---|---|
| Step time after warm-up | **83.5 s** (81.9, 80.0, 83.4, 79.0 logged) | **42.2 s** |
| Generation alone (same shape) | 52.0 s (774 tok/s) | ≈ 8.4 s (0.524 s/prompt x 16, standalone engine) |
| Update + overhead (difference) | ≈ 31.5 s (38% of the step) | ≈ 34 s (≈ 80% of the step) |
| Peak GPU memory | 2.85 GB | 6.11 GB (vLLM reserves 30%) |

Clipped completions 0-9% at 512 tokens; zero-std groups 0-31% per step; reward 0.29-0.51.

## TRL GRPO smoke test (HF backend): passes

3 LoRA steps (256 tokens, one prompt), step time 25.8 / 20.1 / 19.9 s, peak 1.55 GB. The reward function receives
`completion_ids`, `completions`, `prompts`, every dataset column (`env_id`, `prompt_id`, `reference`, `meta_json`),
`trainer_state`, `log_extra`, `log_metric`; the G completions of a prompt arrive contiguously. The torchao removal
(D-86) was the only fix needed.

## Portfolio acceptance rule (Roadmap v3 §4.1), Qwen2.5-0.5B: PASS

| Criterion | Result | Verdict |
|---|---|---|
| >= 2 arms with base pass@8 in [0.15, 0.85] | gsm8k 0.797, math35 0.500, mbpp 0.483 | pass |
| too-hard arm, pass@16 <= 0.02 | countdown: 0 successes in 1,024 samples over two runs | pass (pass@16 itself not measured) |
| cost spread >= 2x | 2.67x (generation + pooled verification) | pass, see the caveat below |

## Reading

- **vLLM is the training backend** (D-87): the same GRPO step takes 42.2 s instead of 83.5 s.
- **The cost spread is real once costs follow tokens.** MATH35 is the expensive arm (2.67x at 0.5B, 2.92x at 1.5B):
  long answers at a 1024-token cap. The roadmap expected MBPP to be the expensive arm; it is not. Its answers are
  short (1187 tokens per prompt incl. 8 completions), so it is the cheapest to generate, and its verifier is what
  costs: 1.3 s per prompt serially (0.16 s per completion on Kaggle's CPU), 0.33 s through 4 threads. **The
  Phase F reward wrapper must verify in parallel**; serially, MBPP verification would add 21 s to a 16-prompt step.
- **Caveat: the update is not in these numbers.** Under vLLM, generation is only about 20% of a GSM8K step; the
  policy update and weight sync take the rest, and that part scales with tokens only partly. Bracketing it
  (update fully token-proportional vs half fixed): MATH35 costs 1.7-2.0x GSM8K per prompt in training, MBPP
  0.6-0.8x, so the training-cost spread among learnable arms is about 2.1-3.5x. The cost meter (Phase F, Gate 5)
  measures the real attribution; the S-I/S-J refit uses the probe's numbers with this range as a sensitivity check.
- **Pass rates at the right caps.** MATH35 at 1024 tokens: pass@8 0.50 (0.34 at 512; truncation 12.5% vs 61%).
  GSM8K pass@8 0.797 vs 0.688 in the HF pilot on the same 64 prompts, with pass@1 nearly equal (0.350 vs 0.344):
  within about two standard errors, so not treated as a backend difference.
- **Run cost.** A mixed-portfolio step will cost between ~42 s (GSM8K-like) and ~70-85 s (MATH35-heavy), so
  one 60-round run at R = 2 is about 1.4-2.8 GPU-hours. The Tier-1 budget estimate is updated in COST_MODEL.md.
- **CV across chunks 0.00-0.13**: chunks hold different prompts, so this mixes content with noise; repeatability is
  still to be judged on identical batches (D-86 c).
- The `Error in sitecustomize ... wrapt` lines come from Kaggle's sitecustomize running inside the venv; harmless.
  The notebook now installs `wrapt` into the venv to silence them.
