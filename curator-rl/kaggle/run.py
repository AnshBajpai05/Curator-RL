#!/usr/bin/env python
"""CURATOR on Kaggle GPUs: one entry point for setup, smoke test, paired comparison and sysid.

    python run.py setup                      # install the right stack for this GPU (T4: vLLM venv; P100: HF)
    python run.py smoke                      # ~10 min end-to-end check incl. checkpoint + resume (do this first)
    python run.py compare                    # Curator vs Uniform vs LP, same seed, matched GPU-hour budget
                                             # (--methods curator,curator_nocal,pilot_uniform,ucb,sec,dump,static too)
    python run.py sysid                      # randomized-mixture run -> costs, learning/transfer, kappa
    python run.py loo --methods uniform      # leave-one-out: full portfolio + one run without each arm (H5)
    python run.py status                     # where every run is (progress.json heartbeat)
    python run.py report                     # rebuild comparison.md / sysid_report.md from the outputs

Results: after every finished run, every 20 minutes during long runs and at the end, the folder
`results/` next to the runs (on Kaggle /kaggle/working/results) is rebuilt: comparison.md, status.md,
figures, every run's summary and logs, and one downloadable `curator_results.zip`. Each run keeps only
its newest 3 checkpoints (--keep-checkpoints); older ones are deleted automatically.

Never lose a run (recommended on Kaggle):
    python run.py compare --sync-to hf://<you>/curator-runs
Every round is checkpointed locally, and the newest checkpoint plus all logs are mirrored to a private
Hugging Face dataset at most every --sync-every-min minutes (and always at the end). Training never
waits for an upload. In the next session the same command fetches the newest state and resumes each run
from its last checkpoint; `--resume-from <dir or hf://...>` does the same from any other location
(e.g. a previous notebook output attached as an input). The token comes from $HF_TOKEN or the Kaggle
secret named HF_TOKEN (Add-ons > Secrets).

Never break mid-run: a crashed run is restarted from its last checkpoint (up to --max-restarts times;
after a CUDA out-of-memory the micro-batch is halved, same effective batch). Before the session limit
(--session-hours, counted from container start) every run stops at a round boundary, saves, syncs and
exits as "preempted"; rerun the same command in a new session to continue.

Hardware: on "GPU T4 x2" two runs go in parallel, one per GPU (vLLM colocate, ~42 s per GRPO step).
On "GPU P100" (compute capability 6.0: no vLLM, no Triton) runs are sequential with the HF backend and
the PyTorch log-prob path; use it for `sysid` (learning and transfer do not depend on the GPU type).
Never mix T4 and P100 runs in one comparison: costs are GPU-seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# Hugging Face progress bars (backup uploads, restores) printed thousands of lines into the notebook,
# which made the Kaggle page sluggish enough to look crashed. The training workers inherit this.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

HERE = Path(__file__).resolve().parent
CODE = HERE / "curator-rl" if (HERE / "curator-rl" / "src").exists() else HERE.parent   # bundle or repo layout
PY_FILE = HERE / ".curator_python"
ON_KAGGLE = Path("/kaggle/working").exists()
DEFAULT_OUT = Path("/kaggle/working/curator_runs") if ON_KAGGLE else HERE / "curator_runs"
EXIT_PREEMPTED = 75            # a run stopped cleanly before the session limit (resumable)
KAGGLE_SESSION_HOURS = 11.75   # Kaggle GPU sessions end at 12 h; leave a margin for the last upload
CLOCK_TZ = os.environ.get("CURATOR_TZ", "Asia/Kolkata")   # clock of the progress lines (the team is in India)
LEVEL1_ENVS = ("gsm8k", "math35", "mbpp", "countdown", "noisy")   # = curator_rl.trainer.run_config.LEVEL1_ENVS

# Each method at its own tuned point on the realism-v2 GPU-shape scenarios (reports/sweeps/tune_gpu2.md:
# equal tuning budget, tuning seeds, one-SE + least-concentrated rule; held-out check in
# reports/analysis/heldout_gpu2.md). Curator carries the reward audit (S5 for a reward that ignores the
# prompt, capped at the exploration floor) and needs 3 consecutive calibration mismatches to flag
# (reports/analysis/component_study_gpu2.md). `curator_nocal` is the same without calibration (ablation).
# Exploratory until a pre-registered confirmation (DECISIONS.md D-98, proposed). `gpu_tuned_v1` is the
# previous preset (pre-audit components, reports/sweeps/tune_gpu_shape.md); `base` uses configs/base.yaml.
_CURATOR_V2 = ["scheduler.gamma=1.0", "scheduler.exploration_coef=0.25", "scheduler.tau=1.0",
               "signals.reward_audit.enabled=true", "scheduler.s5_cap=floor", "calib.mismatch_windows=3"]
PRESETS = {
    "gpu_tuned": {
        "curator": _CURATOR_V2,
        "curator_nocal": _CURATOR_V2 + ["calib.enabled=false"],
        "lp": ["baselines.lp.tau=0.75"],
        "ucb": ["baselines.ucb.exploration_coef=1.0", "baselines.ucb.tau=0.3"],
        "sec": ["baselines.sec.alpha=0.1", "baselines.sec.tau=0.4"],
        "dump": ["baselines.dump.exploration_coef=2.0", "baselines.dump.tau=0.4", "baselines.dump.window_rounds=10"],
    },
    "gpu_tuned_v1": {
        "curator": ["scheduler.gamma=1.0", "scheduler.exploration_coef=0.1", "scheduler.tau=0.3"],
        "lp": ["baselines.lp.tau=0.5"],
        "ucb": ["baselines.ucb.exploration_coef=0.25", "baselines.ucb.tau=0.3"],
        "sec": ["baselines.sec.alpha=1.0", "baselines.sec.tau=0.4"],
        "dump": ["baselines.dump.exploration_coef=0.3", "baselines.dump.tau=0.4", "baselines.dump.window_rounds=10"],
    },
    "base": {},
}


def sh(cmd: list[str], **kw) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kw)


def gpus() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        idx, name, cap, mem = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "name": name, "cc": float(cap), "memory": mem})
    return rows


def worker_python() -> str:
    return PY_FILE.read_text().strip() if PY_FILE.exists() else sys.executable


def load_hf_token() -> None:
    """$HF_TOKEN, else the Kaggle secret HF_TOKEN; exported so the training workers inherit it."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("[run.py] HF_TOKEN loaded from Kaggle secrets")
    except Exception as exc:  # noqa: BLE001 - not on Kaggle, or the secret is not attached
        print(f"[run.py] no HF_TOKEN ({exc.__class__.__name__}); hf:// sync needs one", flush=True)


def session_age_seconds() -> float:
    """Seconds since this container started (Linux /proc; 0 elsewhere)."""
    try:
        start_ticks = float(Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return max(uptime - start_ticks / os.sysconf("SC_CLK_TCK"), 0.0)
    except (OSError, ValueError, IndexError, AttributeError):
        return 0.0


def deadline_unix(args) -> float:
    hours = args.session_hours if args.session_hours is not None else (KAGGLE_SESSION_HOURS if ON_KAGGLE else 0.0)
    if not hours:
        return 0.0
    return time.time() - session_age_seconds() + hours * 3600.0


# ----------------------------------------------------------------------------------------------- setup

def venv_dir() -> Path:
    """Where the vLLM virtualenv lives.

    On Kaggle: /tmp, which session persistence never saves. A virtualenv restored from /kaggle/working by
    "Persistence: variables and files" comes back without execute permissions (its python cannot run),
    and saving it costs several GB per session. /kaggle/working only if /tmp is short of space.
    """
    if not ON_KAGGLE:
        return HERE / "venv_vllm"
    tmp = Path("/tmp/venv_vllm")
    try:
        if tmp.exists() or shutil.disk_usage("/tmp").free > 15e9:
            return tmp
    except OSError:
        pass
    return Path("/kaggle/working/venv_vllm")


def runs(python: str) -> bool:
    """True if this interpreter starts (a restored virtualenv may exist but not be executable)."""
    try:
        return subprocess.run([python, "-c", "import sys"], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def cmd_setup(args) -> int:
    devices = gpus()
    print("GPUs:", devices or "none found")
    if devices and min(d["cc"] for d in devices) >= 7.0 and not args.hf:
        venv = venv_dir()
        stale = Path("/kaggle/working/venv_vllm")
        if ON_KAGGLE and stale.exists() and stale != venv:
            shutil.rmtree(stale, ignore_errors=True)       # a broken copy restored by session persistence
        if (venv / "bin" / "python").exists() and not runs(str(venv / "bin" / "python")):
            print(f"[setup] {venv} is broken (its python cannot run): rebuilding it", flush=True)
            shutil.rmtree(venv, ignore_errors=True)
        if not (venv / "bin" / "python").exists():
            print(f"[setup] creating the vLLM virtualenv in {venv}", flush=True)
            sh([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])
            sh([sys.executable, "-m", "virtualenv", "-q", str(venv)])     # `python -m venv` fails on Kaggle 3.13
        py = str(venv / "bin" / "python")
        if sh([py, "-m", "pip", "install", "-q", "-r", str(HERE / "requirements-vllm.txt")]) != 0:
            print("pinned vLLM stack failed; retrying with the newest vllm (record the versions!)")
            sh([py, "-m", "pip", "install", "-q", "vllm", "-r", str(HERE / "requirements-hf.txt"),
                "-r", str(HERE / "requirements.txt")])
        PY_FILE.write_text(py)
    else:
        sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])   # PEFT 0.20 rejects Kaggle's torchao (D-86)
        sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(HERE / "requirements-hf.txt"),
            "-r", str(HERE / "requirements.txt")])
        PY_FILE.write_text(sys.executable)
    py = worker_python()
    src = str(CODE / "src")
    check = (f"import sys; sys.path.insert(0, {src!r}); import torch, trl, transformers, peft, huggingface_hub; "
             "from curator_rl.trainer.runner import resolve_backend; "
             "print('torch', torch.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, "
             "'| backend', resolve_backend('auto'))")
    return sh([py, "-c", check])


# ------------------------------------------------------------------------------------------ train worker

def run_config(args, method: str, seed: int, **shape):
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.run_config import RunConfig

    if "envs" in shape:
        shape["envs"] = tuple(shape["envs"])
    from curator_rl.trainer.group_filter import split_method

    # "_q" (D-102): the base method's preset plus question filtering; Curator-Q also starts arms whose
    # reward paid for junk in the pre-flight verifier check in S5
    base, filtered = split_method(method)
    overrides = list(PRESETS[args.preset].get(base, []))
    if filtered and base.startswith("curator"):
        overrides.append("signals.reward_audit.preflight_gate=true")
    overrides += list(args.set or [])
    cfg = RunConfig(method=method, seed=seed, model=args.model, backend=args.backend, out_dir=str(args.out),
                    group_filter=args.group_filter if filtered else 0.0,
                    budget_gpu_hours=args.budget_gpu_hours, max_rounds=args.rounds, overrides=overrides,
                    micro_batch=args.micro_batch, max_completion_tokens=args.max_tokens,
                    calib_items=args.calib_items, final_items=args.final_items, report_items=args.report_items,
                    eval_samples=args.eval_samples, eval_temperature=args.eval_temperature,
                    report_every=args.report_every, sync_to=args.sync_to or "", sync_every_min=args.sync_every_min,
                    deadline_unix=args.deadline_unix, label=getattr(args, "label", None) or "",
                    keep_checkpoints=args.keep_checkpoints)
    for k, v in shape.items():
        setattr(cfg, k, v)
    return cfg


def cmd_train(args) -> int:
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.runner import run_single

    shape = json.loads(args.shape) if args.shape else {}
    cfg = run_config(args, args.method, args.seed, **shape)
    summary = run_single(cfg, CODE, resume=not args.fresh, gradient_checkpointing=not args.no_grad_ckpt)
    return EXIT_PREEMPTED if summary.get("status") == "preempted" else 0


def spawn(args, job: dict, gpu: int | None) -> subprocess.Popen:
    method, seed, name = job["method"], job["seed"], job["name"]
    cmd = [worker_python(), str(Path(__file__).resolve()), "train", "--method", method, "--seed", str(seed),
           "--out", str(args.out), "--preset", args.preset, "--model", args.model, "--backend", args.backend,
           "--budget-gpu-hours", str(args.budget_gpu_hours), "--rounds", str(args.rounds),
           "--micro-batch", str(job["micro_batch"]), "--max-tokens", str(args.max_tokens),
           "--calib-items", str(args.calib_items), "--final-items", str(args.final_items),
           "--report-items", str(args.report_items), "--report-every", str(args.report_every),
           "--eval-samples", str(args.eval_samples), "--eval-temperature", str(args.eval_temperature),
           "--group-filter", str(args.group_filter),
           "--sync-every-min", str(args.sync_every_min), "--deadline-unix", str(args.deadline_unix),
           "--keep-checkpoints", str(args.keep_checkpoints),
           "--shape", json.dumps(job["shape"]), *job["extra"]]
    if job["label"]:
        cmd += ["--label", job["label"]]
    if args.sync_to:
        cmd += ["--sync-to", args.sync_to]
    if args.no_grad_ckpt:
        cmd += ["--no-grad-ckpt"]
    for s in args.set or []:
        cmd += ["--set", s]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # vLLM colocate initialises torch.distributed on a local port. TRL picks the first free one from a
    # fixed list, so two runs started together could both pick 29500 and the second crashed; each GPU
    # now gets its own port range, chosen here
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(free_port(29500 + 100 * (gpu or 0)))
    log = log_path(args, name, seed)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run.py] {name} seed {seed} on GPU {gpu} (attempt {job['attempt'] + 1}, micro-batch "
          f"{job['micro_batch']}) -> {log}", flush=True)
    return subprocess.Popen(cmd, env=env, stdout=log.open("a"), stderr=subprocess.STDOUT)


def log_path(args, method: str, seed: int) -> Path:
    return Path(args.out) / "logs" / f"{method}_s{seed}.log"


def _progress_round(args, job: dict) -> int | None:
    try:
        return json.loads((Path(args.out) / f"{job['name']}_s{job['seed']}" / "progress.json").read_text())["round"]
    except (OSError, ValueError, KeyError):
        return None


def _out_of_memory(args, job: dict) -> bool:
    try:
        tail = log_path(args, job["name"], job["seed"]).read_text(errors="replace")[-20000:]
    except OSError:
        return False
    return "out of memory" in tail.lower() or "OutOfMemoryError" in tail


def _retry(args, job: dict) -> dict | None:
    """The job to run after a crash (resumes from its checkpoint), or None to give up."""
    round_now = _progress_round(args, job)
    if job["attempt"] >= args.max_restarts:
        print(f"[run.py] {job['name']} seed {job['seed']}: giving up after {job['attempt'] + 1} attempts", flush=True)
        return None
    if job["failed_at_round"] is not None and round_now == job["failed_at_round"]:
        print(f"[run.py] {job['name']} seed {job['seed']}: crashed twice at round {round_now} with no progress; "
              "giving up (see the log)", flush=True)
        return None
    nxt = dict(job, attempt=job["attempt"] + 1, failed_at_round=round_now,
               extra=[x for x in job["extra"] if x != "--fresh"])     # a restart always resumes
    p_g = int(job["shape"].get("prompts_per_step", 16)) * int(job["shape"].get("group_size", 8))
    if _out_of_memory(args, job) and job["micro_batch"] > 1 and p_g % (job["micro_batch"] // 2) == 0:
        nxt["micro_batch"] = job["micro_batch"] // 2
        print(f"[run.py] out of memory: retrying with micro-batch {nxt['micro_batch']} (same effective batch)",
              flush=True)
    if args.deadline_unix and time.time() > args.deadline_unix - 900:
        print("[run.py] too close to the session limit to restart; resume next session", flush=True)
        return None
    return nxt


def run_jobs(args, jobs: list[tuple], shape: dict | None = None, extra: list[str] | None = None) -> dict:
    """Run jobs one per GPU; restart crashed ones from their checkpoint. Returns outcome lists.

    A job is (method, seed) or (method, seed, arm_left_out); the latter trains without that arm (LOO).
    A dict job sets {"method", "seed", "label", "extra"} itself (e.g. BASE_JOB).
    """
    devices = [d["index"] for d in gpus()] or [None]
    pending = []
    for job in jobs:
        if isinstance(job, dict):
            m, s, arm, label, own_extra = job["method"], job["seed"], None, job.get("label", ""), job.get("extra", [])
        else:
            (m, s, arm), label, own_extra = (tuple(job) + (None,))[:3], "", []
        sh_ = dict(shape or {})
        if arm:
            label = f"{m}-wo-{arm}"
            sh_["envs"] = [e for e in sh_.get("envs", LEVEL1_ENVS) if e != arm]
        pending.append({"method": m, "seed": s, "label": label, "name": label or m, "attempt": 0,
                        "failed_at_round": None, "micro_batch": args.micro_batch, "shape": sh_,
                        "extra": list(extra or []) + list(own_extra)})
    running: list[tuple[subprocess.Popen, int | None, dict]] = []
    outcome = {"finished": [], "preempted": [], "failed": []}
    board = progress_board(args, len(pending), len(devices))
    last_export = time.time()
    last_launch = 0.0
    while pending or running:
        while pending and len(running) < len(devices):
            free = [d for d in devices if d not in {g for _, g, _ in running}]
            job = pending.pop(0)
            wait = LAUNCH_STAGGER_SECONDS - (time.time() - last_launch)
            if args.backend != "fake" and running and wait > 0:
                time.sleep(wait)            # never two model downloads / vLLM start-ups at the same moment
            running.append((spawn(args, job, free[0]), free[0], job))
            name = f"{job['name']}_s{job['seed']}"
            # evaluation only: the untrained model, or a finished run (skipped, or re-evaluated, D-101)
            board.launched(name, free[0], eval_only=job["label"] == "base"
                           or (Path(args.out) / name / "summary.json").exists())
            last_launch = time.time()
        board.tick(len(pending))
        time.sleep(5 if args.backend == "fake" else 15)
        if time.time() - last_export > RESULTS_EVERY_SECONDS:
            refresh_results(args)                # progress snapshot of long runs
            last_export = time.time()
        for proc, g, job in list(running):
            code = proc.poll()
            if code is None:
                continue
            running.remove((proc, g, job))
            name = f"{job['name']}_s{job['seed']}"
            board.ended(name, code, preempted_code=EXIT_PREEMPTED)
            refresh_results(args)                # a run ended (finished, stopped or crashed)
            last_export = time.time()
            if code == 0:
                outcome["finished"].append(name)
            elif code == EXIT_PREEMPTED:
                outcome["preempted"].append(name)
                print(f"[run.py] {name} stopped cleanly before the session limit; rerun to resume", flush=True)
            else:
                print(f"[run.py] {name} crashed on GPU {g} with code {code}", flush=True)
                nxt = _retry(args, job)
                if nxt is None:
                    outcome["failed"].append(name)
                else:
                    pending.insert(0, nxt)
    board.tick(0)                                # the closing summary of the step
    return outcome


RESULTS_EVERY_SECONDS = 20 * 60
LAUNCH_STAGGER_SECONDS = 45


def progress_board(args, n_jobs: int, n_gpus: int):
    """Progress lines in the notebook while the runs train (their own output goes to the log files)."""
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.progress_view import ProgressBoard

    return ProgressBoard(Path(args.out), args.command, n_jobs, n_gpus, args.budget_gpu_hours, tz=CLOCK_TZ)


def free_port(start: int, tries: int = 100) -> int:
    """First TCP port >= start that is free now on localhost."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0                         # 0: let TRL choose


def results_dir(args) -> Path:
    return Path(args.results_dir) if args.results_dir else Path(args.out).parent / "results"


def refresh_results(args, *, quiet: bool = True) -> None:
    """Rebuild comparison, figures, the results folder and its zip (uploaded when syncing). Never raises."""
    if getattr(args, "no_results", False):
        return
    try:
        cmd_report(args, quiet=quiet)
    except Exception as exc:  # noqa: BLE001 - results are a convenience; never stop the runs over them
        print(f"[run.py] results not refreshed: {exc!r}", flush=True)


def restore(args) -> None:
    """Bring back earlier state: from --resume-from, and from the sync target itself."""
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.artifacts import fetch_runs

    for source in [s for s in (args.resume_from, args.sync_to) if s]:
        try:
            taken = fetch_runs(source, Path(args.out))
            print(f"[run.py] {source}: {len(taken)} run(s) newer than the local copy restored", flush=True)
        except Exception as exc:  # noqa: BLE001 - a missing backup must not block a fresh start
            print(f"[run.py] could not fetch from {source}: {exc!r}", flush=True)


def finish(args, outcome: dict) -> int:
    print(f"[run.py] finished: {outcome['finished'] or '-'} | stopped for the session limit: "
          f"{outcome['preempted'] or '-'} | failed: {outcome['failed'] or '-'}", flush=True)
    if args.sync_to and args.squash and not outcome["preempted"]:
        sys.path.insert(0, str(CODE / "src"))
        from curator_rl.trainer.artifacts import squash_remote_history

        squash_remote_history(args.sync_to)
    if outcome["preempted"]:
        print("[run.py] rerun the same command in a new session to continue (it resumes from the checkpoints"
              + (f" synced to {args.sync_to})" if args.sync_to else ")"), flush=True)
    return 1 if outcome["failed"] else (EXIT_PREEMPTED if outcome["preempted"] else 0)


def _shape(args) -> dict:
    shape = json.loads(args.shape) if args.shape else {}
    if getattr(args, "envs", None):
        shape["envs"] = list(args.envs)          # e.g. the stress portfolio (hackable checker instead of noisy)
    return shape


# the untrained model, evaluated exactly as the trained runs are: the reference every method starts from
BASE_JOB = {"method": "uniform", "seed": 0, "label": "base", "extra": ["--rounds", "0"]}


def with_base(args, jobs: list) -> list:
    """`jobs` led by the base-model evaluation (a few minutes), unless it exists or --no-base."""
    if args.no_base or (Path(args.out) / "base_s0" / "summary.json").exists():
        return jobs
    return [BASE_JOB, *jobs]


def cmd_compare(args) -> int:
    restore(args)
    args.methods = args.methods or ["curator", "uniform", "lp"]
    # finished runs evaluated with an older protocol are re-evaluated from their final checkpoint (D-101)
    outcome = run_jobs(args, with_base(args, [(m, s) for s in args.seeds for m in args.methods]), _shape(args))
    refresh_results(args, quiet=False)
    return finish(args, outcome)


def cmd_loo(args) -> int:
    """Leave-one-out (Gate 7/9, H5): per method and seed, the full portfolio plus one run without each arm.

    Same budget per run; the benchmark is unchanged (an arm left out is still scored). Default method:
    Uniform, whose contributions are the arm's value in an equal mixture (the ground truth the in-run ROI
    leaderboard is checked against). 1 + 5 runs per method and seed.
    """
    restore(args)
    args.methods = args.methods or ["uniform"]
    arms = args.arms or list(LEVEL1_ENVS)
    jobs = [j for s in args.seeds for m in args.methods for j in [(m, s)] + [(m, s, a) for a in arms]]
    outcome = run_jobs(args, jobs, _shape(args))
    refresh_results(args, quiet=False)
    return finish(args, outcome)


def cmd_sysid(args) -> int:
    restore(args)
    outcome = run_jobs(args, [("dirichlet", s) for s in args.seeds], _shape(args))
    refresh_results(args, quiet=False)
    return finish(args, outcome)


def cmd_smoke(args) -> int:
    """Tiny run on the first GPU: 3 rounds with calibration every round, then a resume to round 4."""
    args.out = Path(args.out) / "smoke"
    args.no_results = True                      # a check, not a result
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    if not args.sync_to:                       # exercise the sync + restore path even without a token
        args.sync_to = str(Path(args.out) / "_sync")
    else:
        # its own folder in the sync target: the smoke run is also called curator_s0, and must never be
        # restored as the real curator_s0 by a later compare
        args.sync_to = args.sync_to.rstrip("/") + "/smoke"
    shape = {"prompts_per_step": 4, "group_size": 4, "micro_batch": 8, "max_completion_tokens": 128,
             "steps_per_round": 1, "calib_interval": 1, "calib_items": 4, "report_every": 0, "final_items": 4,
             "checkpoint_every": 1}
    args.rounds, args.budget_gpu_hours, args.max_tokens, args.micro_batch = 3, 1.0, 128, 8
    args.sync_every_min = 0.0
    t0 = time.time()
    if run_jobs(args, [("curator", 0)], shape, ["--fresh"])["failed"]:
        print("SMOKE FAILED: see", log_path(args, "curator", 0))
        return 1
    # resume test: wipe the local run and bring it back from the synced copy, then extend to round 4
    shutil.rmtree(Path(args.out) / "curator_s0")
    restore(args)
    args.rounds = 4
    if run_jobs(args, [("curator", 0)], shape)["failed"]:
        print("SMOKE FAILED on resume: see the log")
        return 1
    summary = json.loads((Path(args.out) / "curator_s0" / "summary.json").read_text())
    # the untrained model through the evaluation-only engine (protocol 2: greedy and sampled)
    if run_jobs(args, [BASE_JOB], shape)["failed"]:
        print("SMOKE FAILED in the base-model evaluation: see", log_path(args, "base", 0))
        return 1
    base = json.loads((Path(args.out) / "base_s0" / "summary.json").read_text())
    sampled_ok = all(((x.get("final") or {}).get("sampled") or {}).get("macro") is not None for x in (summary, base))
    ok = summary["rounds"] == 4 and summary["calibrations"] >= 3 and sampled_ok
    if not sampled_ok:
        print("SMOKE: the sampled evaluation produced no score (see 'errors' in reports.jsonl)")
    print(json.dumps({k: summary.get(k) for k in ("rounds", "steps", "calibrations", "charged_gpu_hours",
                                                  "backend", "logprob_path", "verifier_errors", "eval_errors")},
                     indent=1))
    print(f"SMOKE {'PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f}s "
          "(including a restore of the run from its synced copy)")
    return 0 if ok else 1


def cmd_status(args) -> int:
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.results import ZIP_NAME, status_table

    print(status_table(Path(args.out)), end="")
    zip_path = results_dir(args) / ZIP_NAME
    print(f"\nresults zip: {zip_path if zip_path.exists() else 'not written yet'}")
    if args.deadline_unix:
        print(f"\nsession time left: {(args.deadline_unix - time.time()) / 3600:.2f} h")
    return 0


def cmd_report(args, *, quiet: bool = False) -> int:
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.runner import compare_report
    from curator_rl.trainer.sysid import analyse, render

    out = Path(args.out)
    if any(out.glob("*/summary.json")):
        md = compare_report(out)
        (out / "comparison.md").write_text(md, encoding="utf-8")
        if not quiet:
            print(md)
    try:
        from curator_rl.viz.plots import render_run_plots

        _, note = render_run_plots(out)
        if not quiet:
            print(f"[run.py] figures: {note}")
    except Exception as exc:  # noqa: BLE001 - figures are a convenience; never fail a report over them
        print(f"[run.py] figures skipped: {exc!r}")
    for run in sorted(out.glob("dirichlet_s*")):
        if not (run / "run_config.json").exists():
            continue
        cfg = json.loads((run / "run_config.json").read_text())
        rep = analyse(run, cfg["envs"], cfg["group_size"])
        (run / "sysid_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        (run / "sysid_report.md").write_text(render(rep), encoding="utf-8")
        if not quiet:
            print(render(rep))
    _export(args, quiet)
    return 0


def _export(args, quiet: bool) -> None:
    """Rebuild the results folder and its zip from what is on disk, and upload it when syncing."""
    if getattr(args, "no_results", False):
        return
    try:
        from curator_rl.trainer.results import export_results

        zip_path = export_results(Path(args.out), results_dir(args))
        if args.sync_to:
            from curator_rl.trainer.artifacts import ArtifactSync

            ArtifactSync(results_dir(args), args.sync_to, min_interval_s=0.0).flush(timeout=600.0, label="results")
        if not quiet:
            print(f"[run.py] results: {zip_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[run.py] results not written: {exc!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["setup", "smoke", "compare", "sysid", "loo", "train", "report", "status"])
    ap.add_argument("--methods", type=lambda s: s.split(","), default=None,
                    help="compare: default curator,uniform,lp; loo: default uniform")
    ap.add_argument("--arms", type=lambda s: s.split(","), default=None, help="loo: arms to leave out (default all)")
    ap.add_argument("--label", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0])
    ap.add_argument("--method", default="curator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="gpu_tuned")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf", "fake"])
    ap.add_argument("--budget-gpu-hours", type=float, default=1.5)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--calib-items", type=int, default=100, help="paired calibration items per slice")
    ap.add_argument("--final-items", type=int, default=300, help="dev items per domain for the final score")
    ap.add_argument("--report-items", type=int, default=100, help="dev items per domain, periodic reports")
    ap.add_argument("--report-every", type=int, default=10, help="uncharged dev report every N rounds (0: final only)")
    ap.add_argument("--eval-samples", type=int, default=4, help="samples per item for the sampled accuracy (D-101)")
    ap.add_argument("--eval-temperature", type=float, default=1.0, help="... drawn at this temperature (training's)")
    ap.add_argument("--no-base", action="store_true", help="compare: skip the untrained-model evaluation")
    ap.add_argument("--group-filter", type=float, default=1.75,
                    help="'_q' methods: questions drawn per question trained (D-102)")
    ap.add_argument("--envs", type=lambda s: s.split(","), default=None,
                    help="training arms, e.g. gsm8k,math35,mbpp,countdown,hackable (stress portfolio)")
    ap.add_argument("--namespace", default=None,
                    help="keep these runs apart (another model or portfolio): <out>_<ns>, results_<ns>, <sync>/<ns>")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="where results + curator_results.zip go (default: 'results' next to --out)")
    ap.add_argument("--keep-checkpoints", type=int, default=3, help="newest checkpoints kept per run")
    ap.add_argument("--quiet", action="store_true", help="report: write the files, print only their location")
    ap.add_argument("--resume-from", default=None, help="directory or hf://owner/repo[/prefix] with earlier runs")
    ap.add_argument("--sync-to", default=None, help="mirror runs while training: hf://owner/repo[/prefix] or a dir")
    ap.add_argument("--sync-every-min", type=float, default=5.0, help="minimum minutes between uploads")
    ap.add_argument("--squash", action="store_true", help="squash the hf sync repo history when all runs finish")
    ap.add_argument("--session-hours", type=float, default=None,
                    help=f"session limit counted from container start (Kaggle default {KAGGLE_SESSION_HOURS}; 0 = none)")
    ap.add_argument("--max-restarts", type=int, default=3, help="restarts of a crashed run from its checkpoint")
    ap.add_argument("--deadline-unix", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--set", action="append", help="extra base.yaml override, e.g. scheduler.gamma=0.95")
    ap.add_argument("--shape", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster steps if memory allows (C1)")
    ap.add_argument("--hf", action="store_true", help="setup: force the HF backend even on a T4")
    args = ap.parse_args(argv)
    if args.namespace:
        # runs of another model or portfolio share run names (curator_s0, ...) with the main ones: their
        # own output folder, results folder and backup prefix keep them from ever overwriting each other.
        # Workers receive the resolved --out and --sync-to, never --namespace.
        ns = args.namespace.strip("/")
        args.out = args.out.parent / f"{args.out.name}_{ns}"
        if args.results_dir is None:
            args.results_dir = args.out.parent / f"results_{ns}"
        if args.sync_to:
            args.sync_to = args.sync_to.rstrip("/") + f"/{ns}"
    args.out.mkdir(parents=True, exist_ok=True)
    if args.deadline_unix is None:            # workers receive the supervisor's deadline explicitly
        args.deadline_unix = deadline_unix(args)
    if args.command != "train":
        load_hf_token()
    return {"setup": cmd_setup, "smoke": cmd_smoke, "compare": cmd_compare, "sysid": cmd_sysid, "loo": cmd_loo,
            "train": cmd_train, "report": lambda a: cmd_report(a, quiet=a.quiet), "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
