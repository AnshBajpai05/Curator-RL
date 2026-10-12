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
2. **Recommended: a place to mirror the runs.** Create a Hugging Face *write* token
   (huggingface.co -> Settings -> Access Tokens) and add it to the notebook as a Kaggle secret named
   `HF_TOKEN` (Add-ons -> Secrets). `run.py` creates the private dataset repo you name on first use.
3. New Notebook -> **File -> Import notebook** -> `curator_gpu.ipynb` (inside the zip), or paste the cells below.
4. Settings: **Accelerator GPU T4 x2** (or GPU P100, see below), **Internet On**, add your dataset.
5. Run the cells in order:

```bash
# 1. unpack
mkdir -p /kaggle/working/curator && cd /kaggle/working/curator && unzip -q -o /kaggle/input/*/curator-gpu-kaggle.zip
# 2. install the stack for this GPU (T4: vLLM in its own virtualenv; P100: HF backend)
python run.py setup
# 3. ~10 min end-to-end check on the GPU: 3 rounds, calibration each round, checkpoint, a restore of the
#    run from its synced copy, resume to round 4 (add --sync-to hf://<you>/curator-runs to test the upload too)
python run.py smoke
# 4. the comparison: Curator vs Uniform vs LP, seed 0, 1.5 charged GPU-hours each (two run in parallel on 2xT4)
python run.py compare --methods curator,uniform,lp --seeds 0 --sync-to hf://<you>/curator-runs
# 5. where things are, and the tables
python run.py status
cat /kaggle/working/curator_runs/comparison.md
```

## Nothing is lost if the session dies

- **Every round is checkpointed** (LoRA adapter, optimizer, scheduler, prompt-stream positions, RNG
  streams, ledger; written atomically, so a kill mid-write leaves the previous checkpoint intact).
- **With `--sync-to`, each run is mirrored while it trains**: the newest checkpoint plus every log, at
  most every `--sync-every-min` minutes (default 5) and always at the end. Uploads run in the background:
  training never waits for the network, and a failed upload is retried at the next one.
  `--sync-to` also accepts a plain directory.
- **Resuming is the same command.** In a new session, `python run.py compare ... --sync-to hf://<you>/curator-runs`
  first fetches every run that is newer remotely than locally, then continues each one from its last
  checkpoint; finished runs are not re-run. `--resume-from <dir or hf://...>` restores from anywhere else
  (e.g. a previous notebook version's output attached as an input).
- **Before Kaggle's 12-hour limit** (counted from container start; `--session-hours` changes it) every
  run stops at a round boundary, checkpoints, uploads and exits as *preempted*; `run.py` exits with
  code 75 and tells you to rerun.
- `python run.py status` shows, for every run: status, round, GPU-hours used, last checkpoint, last
  synced round and when, and why it stopped (from each run's `progress.json`, rewritten every round).
- When everything is done, `--squash` collapses the sync repo's history so superseded checkpoints stop
  taking storage.

## A run does not die of one bad component

- A verifier that raises (sandbox, parser) scores that completion 0, is counted (`verifier_errors` per
  arm in `rounds.jsonl` and `summary.json`), and training continues.
- A calibration that fails is retried, then skipped and logged in `calibration.jsonl`; the training
  exposure of the skipped window carries over into the next one, so no evidence is lost. A failed report
  evaluation is logged; a final score with a missing domain is left empty, never computed on a subset.
- A checkpoint that cannot be written is retried; after three failures in a row the run stops cleanly.
- If the worker process still dies (CUDA error, out of memory, a killed kernel), `run.py` restarts it from
  its last checkpoint, up to `--max-restarts` times (default 3). After an out-of-memory crash the restart
  halves the micro-batch (same effective batch). A run that crashes twice at the same round with no
  progress is not restarted again (that is a bug: see its log).
- Before any GPU time is spent each run checks free disk, the data manifests and the batch shape.

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
- Curator audits every arm's reward each round (does it depend on the prompt? `signals/audit.py`; zero
  extra compute) and sends an arm whose reward ignores the prompt to S5, capped at the exploration floor.
  On the pilot data the random-reward arm has dispersion D = 1.0-1.09 and every real verifier 3.0-4.4.
- Curator calibrates every 10 rounds on fixed paired items (greedy); the noisy arm is scored on the GSM8K
  slice with transfer-aware credit. Calibration flags an arm after 3 consecutive mismatches.
- The final score is greedy accuracy on the **dev** split (300 items per domain), macro-averaged over GSM8K,
  MATH35, MBPP and Countdown. The sealed test sets are not in the zip and are never touched.

Each run writes `/kaggle/working/curator_runs/<method>_s<seed>/`: `rounds.jsonl` (mixture, delivered
shares, pass rates, statuses, costs), `steps.jsonl`, `calibration.jsonl`, `reports.jsonl`,
`summary.json`, `progress.json`, `checkpoints/`. `comparison.md` pairs the methods by seed (t-interval; with one
seed it is descriptive only), adds the ROI leaderboard of every calibrating run and, after `loo`, the
leave-one-out table with the H5 verdict; `plots/` holds each run's mixture over rounds and the
score-vs-GPU-hours curves. Every `rounds.jsonl` row also records each arm's reward dispersion (`dispersion`).

## Presets and methods

`--preset gpu_tuned` (default): every method at its own configuration, tuned with an equal budget on the
realism-v2 GPU-shape scenarios (`reports/sweeps/tune_gpu2.md`, held-out check
`reports/analysis/heldout_gpu2.md`). Curator carries the reward audit and stricter calibration flags
(DECISIONS.md D-98). `--preset gpu_tuned_v1`: the previous tuned points (before the audit).
`--preset base`: configs/base.yaml as is. Any base.yaml key can be overridden with `--set key=value`.

Methods: `curator`, `curator_nocal` (Curator without calibration: the ablation), `uniform`,
`pilot_uniform` (uniform over the arms the pilot could solve: what a practitioner would run), `lp`,
`ucb`, `sec`, `dump`, `static` (size-proportional), and `dirichlet` (the sysid design, via `sysid`).

## What to run, in order (Kaggle's weekly GPU quota is about 30 hours)

| step | command | GPU | about |
|---|---|---|---|
| 1 | `python run.py smoke --sync-to hf://<you>/curator-runs` | any | 10 min |
| 2 | `python run.py compare --methods curator,uniform,lp --seeds 0 --sync-to ...` | T4 x2 | 4.5 GPU-h (2.3 h wall) |
| 3 | `python run.py sysid --calib-items 50 --final-items 100 --sync-to ...` | P100 | 1.5 GPU-h |
| 4 | `python run.py compare --methods curator_nocal,pilot_uniform --seeds 0 --sync-to ...` | T4 x2 | 3 GPU-h |
| 5 | `python run.py compare --methods curator,uniform,lp --seeds 1,2 --sync-to ...` | T4 x2 | 9 GPU-h |
| 6 | `python run.py loo --methods uniform --seeds 0 --sync-to ...` | T4 x2 | 9 GPU-h (full + 5 arms) |

Step 2 answers the main question on one seed (descriptive); step 5 gives the first intervals (n = 3);
step 3 measures the costs, learning speeds and transfers the simulator still assumes; step 6 is the
ground truth the ROI leaderboard is checked against (H5). All steps resume if a session ends.

## What to send back

`comparison.md`, `plots/`, every `summary.json` and `progress.json`, the `sysid_report.md` (if run), and
`logs/*.log` of any failed run. With `--sync-to`, all of it is already in your Hugging Face dataset.

## If something breaks

- `setup` fails installing vLLM: run `python run.py setup --hf` (HF backend on T4, slower but works).
- Repeated CUDA out of memory: `--micro-batch 4` (same effective batch, smaller passes) or `--max-tokens 768`.
- The session died: start a new one and rerun the same command (with the same `--sync-to`); it resumes.
- Without a sync target: Save Version before closing, then attach that output as an input and pass
  `--resume-from /kaggle/input/<that-output>/curator_runs`.
