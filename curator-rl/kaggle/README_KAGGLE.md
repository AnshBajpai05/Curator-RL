# CURATOR on Kaggle GPUs: first real GRPO runs under the scheduler

This package runs CURATOR's scheduler around a real TRL 1.15 GRPO trainer (Qwen2.5-0.5B-Instruct, LoRA)
on Kaggle's free GPUs. It answers two questions no simulator can:

1. **Does CURATOR beat Uniform and the learning-progress curriculum on a real model,** at a matched
   GPU-hour budget, on the real cost-heterogeneous portfolio (GSM8K, MATH levels 3-5, MBPP, Countdown,
   and a random-reward "noisy" arm)?
2. **What are the real learning speeds, transfers and per-arm costs** that the simulator still assumes
   (the system-identification run)?

## Steps

**Once**
1. **Hugging Face write token:** huggingface.co -> Settings -> Access Tokens -> Create new token -> type
   **Write** -> copy it (never paste it anywhere except the Kaggle secret below).
2. **Upload the package:** kaggle.com -> Datasets -> New Dataset (or, later, your dataset -> **New
   Version**) -> drop `curator-gpu-kaggle.zip` -> Private -> Create. Kaggle usually unzips it; fine.
3. **Notebook:** Code -> New Notebook -> File -> **Import Notebook** -> `curator_gpu.ipynb` (in the
   package; on your PC `curator-rl/kaggle/curator_gpu.ipynb`).
4. **Right panel:** Session options -> Accelerator **GPU T4 x2** · Internet **On** · Input -> Add Input ->
   your dataset (latest version) · Add-ons -> Secrets -> add `HF_TOKEN` = your token, box **ticked**.

**Every session**
5. In **cell 1** set `STEPS` (see the plan below; `HF_USER` is already `AnshBajpai05`).
6. **Save Version -> Save & Run All (Commit)**: runs in the background, the browser can be closed (up to
   12 h). Or **Run All** in the open notebook (keep the tab open).

Everything else is automatic: cell 2 stops leftovers of an interrupted run and checks the package and
the GPU; cell 3 checks the Hugging Face token before any GPU time is spent; cell 4 installs only when
needed; cell 5 runs the smoke test only until it has passed once; cell 6 runs the steps; cell 7 prints
the results, the error of any run that crashed, and where the zip is; cell 8 removes the multi-GB
install in background runs so only results are saved. Every cell stops with a plain instruction when
something needs fixing (dataset not attached, old package, no GPU, no or read-only token).

**While cell 6 runs** it prints a line per run after every round (about every 3 minutes), and again
after 5 quiet minutes (loading the model, the final evaluation), plus a step summary when a run starts
or ends and every 15 minutes (clock in IST; set the environment variable `CURATOR_TZ` for another zone):

```
[00:05 IST] curator_s0  GPU0 ██████████████████░░  93%  round 30  training, ~14 min to go  checkpoint 30 · backed up 28
[00:05 IST] uniform_s0  GPU1 ████████████████████ 100%  round 35  final evaluation, ~4 min to go  checkpoint 35 · backed up 35
[00:05 IST] compare: 0/3 runs finished, 2 running, 1 waiting · step done in ~2 h 05 min (about 02:10 IST)
[00:09 IST] ✓ uniform_s0 finished: 35 rounds, 1.50 GPU-h, dev score 0.412
```

The bar is the share of the run's GPU-hour budget used. "backed up" is the newest round in the Hugging
Face backup; `backup FAILING` means the backup stopped working (the run itself continues). The workers'
full output is in `/kaggle/working/curator_runs/logs/<run>.log`.

## Nothing is lost if the session dies

- **Every round is checkpointed** (LoRA adapter, optimizer, scheduler, prompt-stream positions, RNG
  streams, ledger; written atomically, so a kill mid-write leaves the previous checkpoint intact).
  Only the **newest 3 checkpoints per run** are kept (about 0.1 GB each); older ones and half-written
  leftovers are deleted automatically (`--keep-checkpoints N` to change it).
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

Use **GPU T4 x2** for everything: vLLM colocate, about 42 s per GRPO step, two runs in parallel (one per
GPU). Kaggle no longer offers the P100, and TPU v5e-8 cannot run this stack (PyTorch CUDA, vLLM, TRL).
Running sysid on the T4s too is an advantage: its cost measurements come from the same hardware as the
comparison runs. Never put runs from different GPU types (e.g. a local RTX 4090) in one comparison: costs
are GPU-seconds, so keep each GPU type in its own sync repo.

If a P100 (compute capability 6.0) is ever available again: `setup` installs the HF backend there, and
`curator_rl/trainer/logprob_fallback.py` replaces TRL's Triton log-prob kernel (Triton needs 7.0+) with
an exact PyTorch equivalent, tested against the naive computation (values and gradients).

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

## What to run, in order (three sessions)

| session | `STEPS` in cell 1 | wall time | runs |
|---|---|---|---|
| 1 | `["compare", "ablation"]` | ~5.5 h | curator, uniform, lp, curator_nocal, pilot_uniform (seed 0) |
| 2 | `["sysid", "seeds12"]` | ~7.5 h | sysid seeds 0, 1; curator, uniform, lp seeds 1, 2 |
| 3 | `["loo"]` | ~5.5 h | uniform with each arm left out (seed 0) |

About 19 hours of Kaggle's ~30-hour weekly GPU quota. Each run is capped at 1.5 charged GPU-hours
(~1.7 h of wall time); two run in parallel on T4 x2. If a session ends early, start a new one with the
**same** `STEPS`: runs resume from their last checkpoint and finished runs are skipped.
`STEPS = ["status"]` only prints where everything is.

## What to send back

`results/curator_results.zip` (it contains `comparison.md`, `plots/`, every `summary.json` and
`progress.json`, the sysid report and the logs). With `--sync-to`, everything is also in your Hugging
Face dataset; download it on your PC with

```
hf download <you>/curator-runs --repo-type dataset --local-dir kaggle_results
python kaggle/run.py report --out kaggle_results        # rebuilds comparison.md and the figures locally
```

Before **Save Version** without a sync target, run the notebook's last cell: it deletes the vLLM
virtualenv (several GB) so only the runs are saved as output.

## If something breaks

- `setup` fails installing vLLM: run `python run.py setup --hf` (HF backend on T4, slower but works).
- Repeated CUDA out of memory: `--micro-batch 4` (same effective batch, smaller passes) or `--max-tokens 768`.
- The session died: start a new one and rerun the same command (with the same `--sync-to`); it resumes.
- Without a sync target: Save Version before closing, then attach that output as an input and pass
  `--resume-from /kaggle/input/<that-output>/curator_runs`.
