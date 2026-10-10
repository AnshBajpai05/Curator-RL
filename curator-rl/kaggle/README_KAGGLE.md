# CURATOR on Kaggle GPUs: first real GRPO runs under the scheduler

This package runs CURATOR's scheduler around a real TRL 1.15 GRPO trainer (Qwen2.5-0.5B-Instruct, LoRA)
on Kaggle's free GPUs. It answers two questions no simulator can:

1. **Does CURATOR beat Uniform and the learning-progress curriculum on a real model,** at a matched
   GPU-hour budget, on the real cost-heterogeneous portfolio (GSM8K, MATH levels 3-5, MBPP, Countdown,
   and a random-reward "noisy" arm)?
2. **What are the real learning speeds, transfers and per-arm costs** that the simulator still assumes
   (the system-identification run)?

## Steps (about 10 minutes of setup, then the runs)

1. Upload `curator-gpu-kaggle.zip` as a **Kaggle Dataset** (Datasets -> New Dataset; keep it Private).
2. New Notebook -> **File -> Import notebook** -> `curator_gpu.ipynb` (inside the zip), or paste the cells below.
3. Settings: **Accelerator GPU T4 x2** (or GPU P100, see below), **Internet On**, add your dataset,
   Persistence "Files only".
4. Run the cells in order:

```bash
# 1. unpack
mkdir -p /kaggle/working/curator && cd /kaggle/working/curator && unzip -q -o /kaggle/input/*/curator-gpu-kaggle.zip
# 2. install the stack for this GPU (T4: vLLM in its own virtualenv; P100: HF backend)
python run.py setup
# 3. ~10 min end-to-end check on the GPU: 3 rounds, calibration each round, checkpoint, resume to round 4
python run.py smoke
# 4. the comparison: Curator vs Uniform vs LP, seed 0, 1.5 charged GPU-hours each (two run in parallel on 2xT4)
python run.py compare --methods curator,uniform,lp --seeds 0
# 5. the tables
cat /kaggle/working/curator_runs/comparison.md
```

**Before closing the session: Save Version** (Save & Run All or Quick Save with outputs) so
`/kaggle/working/curator_runs` is kept. To continue in a new session, add that notebook version's output
as an input and run `python run.py compare --resume-from /kaggle/input/<that-output>/curator_runs`:
runs resume from their last checkpoint (every 5 rounds) and completed runs are skipped.

## Which GPU for what

| | GPU T4 x2 | GPU P100 |
|---|---|---|
| Backend | vLLM colocate (~42 s per GRPO step, measured) | HF generate (~85 s per step), PyTorch log-prob path |
| Parallel runs | 2 (one per GPU) | 1 |
| Best use | `compare` (cost-sensitive, same hardware for all methods) | `sysid` (learning/transfer are hardware-independent) |
| Suggested | `python run.py compare --seeds 0` (~2 h wall) | `python run.py sysid --calib-items 50 --final-items 100` |

Why the P100 needs special handling: TRL 1.15 computes GRPO log-probabilities with a Triton kernel, and
Triton needs compute capability >= 7.0; the P100 is 6.0. `curator_rl/trainer/logprob_fallback.py`
replaces that one function with an exact PyTorch equivalent (tested against the naive computation,
values and gradients) when Triton cannot run. Never put T4 and P100 runs in the same comparison.

## What the runs do

- Every round (2 optimizer steps x 16 prompts x 8 completions) the method chooses a mixture; the prompts of
  each step are planned at generation time (no dataloader lag) with deficit-carry rounding, so delivered
  shares match requested ones.
- Rewards come from the real verifiers in a thread pool (MBPP runs in the hardened sandbox; screening
  refusals are counted as `sandbox_rejections`).
- Step time is measured (CUDA-synchronised) and attributed to arms (verification exactly, the rest by tokens).
  Training and Curator's calibration are charged; reporting evaluations are not. Each run stops at the
  same budget (default 1.5 GPU-hours).
- Curator calibrates every 10 rounds on fixed paired items (greedy); the noisy arm is scored on the GSM8K
  slice with transfer-aware credit.
- The final score is greedy accuracy on the **dev** split (300 items per domain), macro-averaged over GSM8K,
  MATH35, MBPP and Countdown. The sealed test sets are not in the zip and are never touched.

Each run writes `/kaggle/working/curator_runs/<method>_s<seed>/`: `rounds.jsonl` (mixture, delivered
shares, pass rates, statuses, costs), `steps.jsonl`, `calibration.jsonl`, `reports.jsonl`,
`summary.json`, `checkpoints/`. `comparison.md` pairs the methods by seed (t-interval; with one seed it
is descriptive only).

## Presets

`--preset gpu_tuned` (default): every method at its own configuration tuned at this GPU shape on tuning
seeds (reports/sweeps/tune_gpu_shape.md; equal tuning budget). `--preset base`: configs/base.yaml as is.
Any base.yaml key can be overridden with `--set key=value`.

## What to send back

`comparison.md`, every `summary.json`, the `sysid_report.md` (if run), and `logs/*.log` of any failed run.

## If something breaks

- `setup` fails installing vLLM: run `python run.py setup --hf` (HF backend on T4, slower but works).
- CUDA out of memory: `--micro-batch 4` (same effective batch, smaller passes) or `--max-tokens 768`.
- A run died mid-way: rerun the same command; it resumes from the last checkpoint.
