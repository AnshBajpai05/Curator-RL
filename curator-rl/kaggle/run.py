#!/usr/bin/env python
"""CURATOR on Kaggle GPUs: one entry point for setup, smoke test, paired comparison and sysid.

    python run.py setup                      # install the right stack for this GPU (T4: vLLM venv; P100: HF)
    python run.py smoke                      # ~10 min end-to-end check incl. checkpoint + resume (do this first)
    python run.py compare                    # Curator vs Uniform vs LP, same seed, matched GPU-hour budget
    python run.py sysid                      # randomized-mixture run -> costs, learning/transfer, kappa
    python run.py report                     # rebuild comparison.md / sysid_report.md from the outputs
    python run.py compare --resume-from /kaggle/input/<previous-output>/curator_runs   # next session

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
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "curator-rl" if (HERE / "curator-rl" / "src").exists() else HERE.parent   # bundle or repo layout
PY_FILE = HERE / ".curator_python"
DEFAULT_OUT = Path("/kaggle/working/curator_runs") if Path("/kaggle/working").exists() else HERE / "curator_runs"

# Each method at its own GPU-shape-tuned point (reports/sweeps/tune_gpu_shape.md: equal tuning budget,
# tuning seeds, one-SE + least-concentrated rule). Exploratory until a pre-registered confirmation
# (DECISIONS.md D-96, proposed); `--preset base` uses configs/base.yaml unchanged.
PRESETS = {
    "gpu_tuned": {
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


# ----------------------------------------------------------------------------------------------- setup

def cmd_setup(args) -> int:
    devices = gpus()
    print("GPUs:", devices or "none found")
    if devices and min(d["cc"] for d in devices) >= 7.0 and not args.hf:
        venv = Path("/kaggle/working/venv_vllm") if Path("/kaggle/working").exists() else HERE / "venv_vllm"
        if not (venv / "bin" / "python").exists():
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
    check = (f"import sys; sys.path.insert(0, {src!r}); import torch, trl, transformers, peft; "
             "from curator_rl.trainer.runner import resolve_backend; "
             "print('torch', torch.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, "
             "'| backend', resolve_backend('auto'))")
    return sh([py, "-c", check])


# ------------------------------------------------------------------------------------------ train worker

def run_config(args, method: str, seed: int, **shape):
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.run_config import RunConfig

    overrides = list(PRESETS[args.preset].get(method, [])) + list(args.set or [])
    cfg = RunConfig(method=method, seed=seed, model=args.model, backend=args.backend, out_dir=str(args.out),
                    budget_gpu_hours=args.budget_gpu_hours, max_rounds=args.rounds, overrides=overrides,
                    micro_batch=args.micro_batch, max_completion_tokens=args.max_tokens,
                    calib_items=args.calib_items, final_items=args.final_items, report_items=args.report_items,
                    report_every=args.report_every)
    for k, v in shape.items():
        setattr(cfg, k, v)
    return cfg


def cmd_train(args) -> int:
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.runner import run_single

    shape = json.loads(args.shape) if args.shape else {}
    cfg = run_config(args, args.method, args.seed, **shape)
    run_single(cfg, CODE, resume=not args.fresh, gradient_checkpointing=not args.no_grad_ckpt)
    return 0


def spawn(args, method: str, seed: int, gpu: int | None, shape: dict, extra: list[str]) -> subprocess.Popen:
    cmd = [worker_python(), str(Path(__file__).resolve()), "train", "--method", method, "--seed", str(seed),
           "--out", str(args.out), "--preset", args.preset, "--model", args.model, "--backend", args.backend,
           "--budget-gpu-hours", str(args.budget_gpu_hours), "--rounds", str(args.rounds),
           "--micro-batch", str(args.micro_batch), "--max-tokens", str(args.max_tokens),
           "--calib-items", str(args.calib_items), "--final-items", str(args.final_items),
           "--report-items", str(args.report_items), "--report-every", str(args.report_every),
           "--shape", json.dumps(shape), *extra]
    for s in args.set or []:
        cmd += ["--set", s]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = Path(args.out) / "logs" / f"{method}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run.py] {method} seed {seed} on GPU {gpu} -> {log}", flush=True)
    return subprocess.Popen(cmd, env=env, stdout=log.open("a"), stderr=subprocess.STDOUT)


def run_jobs(args, jobs: list[tuple[str, int]], shape: dict | None = None, extra: list[str] | None = None) -> int:
    devices = [d["index"] for d in gpus()] or [None]
    pending, running, failed = list(jobs), [], 0
    while pending or running:
        while pending and len(running) < len(devices):
            free = [d for d in devices if d not in {g for _, g in running}]
            m, s = pending.pop(0)
            running.append((spawn(args, m, s, free[0], shape or {}, extra or []), free[0]))
        time.sleep(15)
        for proc, g in list(running):
            if proc.poll() is not None:
                running.remove((proc, g))
                failed += proc.returncode != 0
                print(f"[run.py] job on GPU {g} finished with code {proc.returncode}", flush=True)
    return failed


def restore(args) -> None:
    if args.resume_from:
        src = Path(args.resume_from)
        for d in src.iterdir():
            dst = Path(args.out) / d.name
            if not dst.exists():
                shutil.copytree(d, dst) if d.is_dir() else shutil.copy2(d, dst)
        print(f"[run.py] restored previous outputs from {src}")


def cmd_compare(args) -> int:
    restore(args)
    jobs = [(m, s) for s in args.seeds for m in args.methods]
    failed = run_jobs(args, jobs)
    cmd_report(args)
    return 1 if failed else 0


def cmd_sysid(args) -> int:
    restore(args)
    failed = run_jobs(args, [("dirichlet", s) for s in args.seeds])
    cmd_report(args)
    return 1 if failed else 0


def cmd_smoke(args) -> int:
    """Tiny run on the first GPU: 3 rounds with calibration every round, then a resume to round 4."""
    args.out = Path(args.out) / "smoke"
    shutil.rmtree(args.out, ignore_errors=True)
    shape = {"prompts_per_step": 4, "group_size": 4, "micro_batch": 8, "max_completion_tokens": 128,
             "steps_per_round": 1, "calib_interval": 1, "calib_items": 4, "report_every": 0, "final_items": 4,
             "checkpoint_every": 1}
    args.rounds, args.budget_gpu_hours, args.max_tokens, args.micro_batch = 3, 1.0, 128, 8
    t0 = time.time()
    if run_jobs(args, [("curator", 0)], shape, ["--fresh"]):
        print("SMOKE FAILED: see", Path(args.out) / "logs" / "curator_s0.log")
        return 1
    args.rounds = 4
    if run_jobs(args, [("curator", 0)], shape):
        print("SMOKE FAILED on resume: see the log")
        return 1
    summary = json.loads((Path(args.out) / "curator_s0" / "summary.json").read_text())
    ok = summary["rounds"] == 4 and summary["calibrations"] >= 3
    print(json.dumps({k: summary[k] for k in ("rounds", "steps", "calibrations", "charged_gpu_hours",
                                              "backend", "logprob_path")}, indent=1))
    print(f"SMOKE {'PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f}s")
    return 0 if ok else 1


def cmd_report(args) -> int:
    sys.path.insert(0, str(CODE / "src"))
    from curator_rl.trainer.runner import compare_report
    from curator_rl.trainer.sysid import analyse, render

    out = Path(args.out)
    if any(out.glob("*/summary.json")):
        md = compare_report(out)
        (out / "comparison.md").write_text(md, encoding="utf-8")
        print(md)
    for run in sorted(out.glob("dirichlet_s*")):
        cfg = json.loads((run / "run_config.json").read_text())
        rep = analyse(run, cfg["envs"], cfg["group_size"])
        (run / "sysid_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        (run / "sysid_report.md").write_text(render(rep), encoding="utf-8")
        print(render(rep))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["setup", "smoke", "compare", "sysid", "train", "report"])
    ap.add_argument("--methods", type=lambda s: s.split(","), default=["curator", "uniform", "lp"])
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
    ap.add_argument("--report-every", type=int, default=20, help="uncharged dev report every N rounds (0: final only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--set", action="append", help="extra base.yaml override, e.g. scheduler.gamma=0.95")
    ap.add_argument("--shape", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="faster steps if memory allows (C1)")
    ap.add_argument("--hf", action="store_true", help="setup: force the HF backend even on a T4")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return {"setup": cmd_setup, "smoke": cmd_smoke, "compare": cmd_compare, "sysid": cmd_sysid,
            "train": cmd_train, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
