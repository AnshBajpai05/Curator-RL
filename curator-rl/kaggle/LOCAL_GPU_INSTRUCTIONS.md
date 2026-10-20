# Instructions: CURATOR runs on a local GPU (for you and your Claude Code)

Give this file to Claude Code together with `curator-gpu-kaggle.zip` (Ansh sends the zip; it contains
the task data, which is not in this repository) and ask it to follow it. `BUNDLE_INFO.json` in the zip
must say `"git_sha": "a3d5d41"` (or later) and `"git_dirty": false`: the code the Kaggle runs use,
with evaluation protocol 2 (sampled accuracy, item-by-item comparison, untrained-model reference).

## What this is

CURATOR picks which training data (math, code, puzzles, ...) a small model learns from, under a fixed
compute budget. Ansh is running the main experiment on Kaggle T4 GPUs. **The same experiment on your
GPU** answers two questions:

1. Does Curator beat Uniform (equal mixing) on this GPU? Both methods use the same seed on the same machine.
2. What does a GPU-hour buy on this GPU? The budget is 1.5 hours of GPU wall time, so a different GPU
   completes a different number of training steps, and each data source costs a different amount.

Each Kaggle run uses one T4, so the comparison is GPU against GPU. The results only count if the
experiment is **identical** to the Kaggle one. Every rule below protects that.

## Rules for Claude Code

**R1. Do not change the experiment.** Use `run.py` with its defaults. Never pass or change any of:
`--model`, `--preset`, `--budget-gpu-hours`, `--rounds`, `--max-tokens`, `--calib-items`,
`--final-items`, `--report-items`, `--report-every`, `--eval-samples`, `--eval-temperature`, `--no-base`,
`--group-filter`, `--envs`, `--namespace`,
`--set`, or `prompts_per_step`, `group_size`,
`steps_per_round` (in `--shape`). Do not edit anything under `curator-rl/src/`, `curator-rl/configs/`
or `curator-rl/data/`, and do not edit `run.py`'s presets.

**R2. Only these changes are allowed** (they change speed or memory, not what is trained):

| knob | how | when |
|---|---|---|
| micro-batch | `--micro-batch 4`, then 2, then 1 | `CUDA out of memory` during training (same effective batch through gradient accumulation; `run.py` also halves it by itself after an out-of-memory crash) |
| vLLM memory share | `--shape '{"vllm_gpu_memory_utilization": 0.4}'` (default 0.3; try 0.4, then 0.5) | vLLM fails at start-up with too little memory for its KV cache (e.g. "No available memory for the cache blocks") |
| backend | `python run.py setup --hf`, then add `--backend hf` to every command | only if vLLM cannot be installed or started at all; much slower, so far fewer steps per GPU-hour |

Use the same knobs for **every** run on a machine (Curator and Uniform must be paired exactly).

**R3. If it does not fit with R2, stop and report.** Never reduce `--max-tokens`, the prompts or the
group size to make it fit; that would make it a different experiment. A 4 GB GPU will very likely not
fit; that is a valid result to report.

**R4. Fix the platform, not the method.** Installation problems, drivers, paths and WSL issues may be
fixed freely. If a crash can only be fixed by editing code, the edit must not change any number the
training uses. Make it as small as possible and record it (R5).

**R5. Every change must be visible.** Right after unzipping:
```
cd ~/curator && git init -q && git add -A && git commit -qm "package as received"
```
At the end, run `git diff > local_changes.diff` (empty if nothing was edited), and write
`CHANGES_LOCAL.md`: every knob from R2 that was used and why (paste the error message).

**R6. Do not touch Ansh's backup.** Never use `--sync-to hf://AnshBajpai05/curator-runs`. The run
names (`curator_s0`, ...) are the same as on Kaggle and would overwrite those runs. The runs stay on
the local disk. For a backup, use a folder: `--sync-to /mnt/d/curator_backup`.

## Machine requirements

- **Linux, or Windows with WSL2 (Ubuntu).** Not native Windows: vLLM runs only on Linux, and the code
  sandbox (which runs code written by the model, for the MBPP tasks) enforces its CPU, memory and file
  limits only on Linux. On native Windows, model-written code would run without those limits.
- `nvidia-smi` works inside Linux/WSL. vLLM 0.31 needs a recent NVIDIA driver (CUDA 12).
- Python 3.11, about 15 GB free disk. In WSL, keep the folder in the Linux file system (`~/curator`),
  not under `/mnt/c/...` (much slower).
- Laptop plugged in, in performance power mode, nothing else using the GPU while a run trains. The
  budget is wall time, so anything else steals steps from the run.

## Steps

```
mkdir -p ~/curator && cd ~/curator && unzip /path/to/curator-gpu-kaggle.zip
git init -q && git add -A && git commit -qm "package as received"          # R5
nvidia-smi > nvidia_smi.txt                                                 # GPU model, memory, power limit
python run.py setup                                                         # vLLM environment in ./venv_vllm (~15 min)
python run.py smoke                                                         # ~10 min, must end with "SMOKE PASSED"
```

The smoke test uses tiny settings, so it does **not** prove that the full-size run fits in memory. The
first `compare` does. Start a GPU log in a second terminal (it runs until stopped with Ctrl+C):

```
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv -l 30 > gpu_log.csv
```

Then the runs. They go one after the other on one GPU; each takes about 2 h. The first `compare` also
evaluates the untrained model first (`base_s0`, about 10 minutes): the reference every method is
measured against.

```
python run.py compare --methods curator,uniform --seeds 0      # ~4 h (required)
python run.py compare --methods curator,uniform --seeds 1      # ~4 h (if there is time)
```

Add the R2 knobs to these lines if needed, the same on both. The terminal shows a progress line per
run every round. A finished run prints its dev score. `python run.py status` shows where everything is.

**Interrupted** (crash, sleep, closed terminal)? Run the same command again. It resumes from the last
checkpoint and skips finished runs. A crashed run is restarted automatically up to 3 times. Its full
output is in `curator_runs/logs/<run>.log`; look there first when something fails.

## What to send back to Ansh

1. `results/curator_results.zip`, renamed after the GPU, e.g. `results_rtx4050.zip`
2. `nvidia_smi.txt` and `gpu_log.csv`
3. `CHANGES_LOCAL.md` and `local_changes.diff` (R5)
4. `venv_vllm/bin/pip freeze > pip_freeze.txt` (the exact library versions)
5. the exact command lines that were run
