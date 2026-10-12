"""Single-run driver and comparison report for real (or fake) GRPO runs under a scheduler.

`run_single` builds the environments and config, resumes from the newest checkpoint when asked,
trains until the budget or the round limit, runs the final uncharged dev evaluation and writes
`summary.json`. `compare_report` pairs the summaries of several methods by seed. With fewer than
20 seeds it uses the t-interval and the exact sign-flip test (core.stats, audit E-11), never a
percentile bootstrap.
"""

from __future__ import annotations

import json
import shutil
import signal
import threading
import time
from pathlib import Path

from curator_rl.core.atomic import atomic_write_json
from curator_rl.core.config import load_config, rescale_for_batch_shape
from curator_rl.envs.registry import EnvRegistry
from curator_rl.trainer.artifacts import ArtifactSync
from curator_rl.trainer.checkpoint import latest_checkpoint, load_controller_state
from curator_rl.trainer.round_controller import SYNC_RESERVE_SECONDS, RoundController
from curator_rl.trainer.run_config import RunConfig


def resolve_backend(requested: str) -> str:
    """auto -> vllm on compute capability >= 7.0 with vLLM importable, else hf; fake stays fake."""
    if requested in ("fake", "hf", "vllm"):
        return requested
    try:
        import torch
    except ImportError:
        return "fake"
    if not torch.cuda.is_available():
        return "hf"
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 7:
        try:
            import vllm  # noqa: F401
            return "vllm"
        except ImportError:
            return "hf"
    return "hf"                       # P100 (6.0): vLLM needs 7.0+


def run_dir(cfg: RunConfig) -> Path:
    return Path(cfg.out_dir) / f"{cfg.method}_s{cfg.seed}"


MIN_FREE_DISK_GB = 2.0      # a checkpoint is ~0.1 GB (LoRA + Adam); keep room for several plus logs


def preflight(cfg: RunConfig, repo_root: Path, out: Path) -> list[str]:
    """Problems that would only surface hours into a run, checked before any GPU time is spent."""
    problems = []
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < MIN_FREE_DISK_GB:
        problems.append(f"only {free_gb:.1f} GB free in {out} (need {MIN_FREE_DISK_GB} GB for checkpoints)")
    from curator_rl.envs.splits import verify_manifests

    bad = verify_manifests(Path(repo_root) / "data")
    if bad:
        problems.append(f"data manifests do not verify: {bad[:3]}")
    g, p = cfg.group_size, cfg.prompts_per_step
    if (g * p) % min(cfg.micro_batch, g * p):
        problems.append(f"micro_batch {cfg.micro_batch} must divide P x G = {g * p}")
    if cfg.deadline_unix and cfg.deadline_unix - time.time() < SYNC_RESERVE_SECONDS + 600:
        problems.append("less than 15 minutes left before the session deadline")
    return problems


def run_single(cfg: RunConfig, repo_root: Path, *, resume: bool = True,
               gradient_checkpointing: bool = True) -> dict:
    """Train one run to its budget, a clean stop, or a crash (re-raised after saving what exists).

    Returns the summary; `summary["status"]` is "finished" or "preempted" (stopped cleanly before
    a limit, resumable: `run.py` exits with EXIT_PREEMPTED).
    """
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    problems = preflight(cfg, repo_root, out)
    if problems:
        raise RuntimeError("preflight failed: " + "; ".join(problems))
    root_cfg = load_config(Path(repo_root) / "configs" / "base.yaml", cfg.overrides)
    if not any(o.startswith("signals.status.n_min=") for o in cfg.overrides):
        root_cfg = rescale_for_batch_shape(root_cfg, cfg.prompts_per_round)      # audit E-28
    registry = EnvRegistry(Path(repo_root), root_cfg.data)
    envs = {e: registry.get(e) for e in cfg.envs}
    ctl = RoundController(cfg, root_cfg, envs, out_dir=out)
    ctl.sync = ArtifactSync(out, cfg.sync_to, min_interval_s=60.0 * cfg.sync_every_min)
    backend = resolve_backend(cfg.backend)
    ckpt = latest_checkpoint(out) if resume else None
    (out / "run_config.json").write_text(cfg.to_json(), encoding="utf-8")
    print(f"[curator] {cfg.method} seed {cfg.seed}: backend={backend}, "
          f"{'resuming from ' + ckpt.name if ckpt else 'fresh start'}"
          f"{', syncing to ' + cfg.sync_to if cfg.sync_to else ''}", flush=True)

    if ckpt is not None:
        ctl.load_state(load_controller_state(ckpt))
        ctl.last_checkpoint_round = ctl.round
    else:
        ctl.begin_round()
    previous = _install_stop_handlers(ctl)
    ctl.write_progress("running")
    try:
        if not ctl.stop:
            if backend == "fake":
                from curator_rl.trainer.fake_backend import FakeTrainer

                trainer = FakeTrainer(ctl, cfg.seed)
                if ckpt is not None:
                    trainer.load(ckpt)
                ctl.save_backend = trainer.save
                trainer.train(lambda: ctl.save(trainer.save))
            else:
                from curator_rl.trainer.trl_adapter import run_trl

                run_trl(ctl, cfg, out, backend=backend, resume_dir=ckpt,
                        gradient_checkpointing=gradient_checkpointing)
            if ctl.save_backend is not None and ctl.last_checkpoint_round != ctl.round:
                ctl.save(ctl.save_backend)        # final state, for later analysis
        if ctl.preempted:
            ctl.write_progress("preempted")
            ok = ctl.sync.flush(label=f"preempted at round {ctl.round}")
            print(f"[curator] {cfg.method} seed {cfg.seed}: stopped cleanly at round {ctl.round} "
                  f"({ctl.stop_reason}); checkpoint saved{', synced' if ok else ''}. Resume to continue.",
                  flush=True)
            ctl.close()
            return {"status": "preempted", "method": cfg.method, "seed": cfg.seed, "rounds": ctl.round,
                    "stop_reason": ctl.stop_reason}
        final = ctl.report(cfg.final_items, tag="final")
        summary = ctl.summary(final)
        summary["backend"] = backend
        summary["status"] = "finished"
        atomic_write_json(out / "summary.json", summary)
        ctl.write_progress("finished")
        ctl.sync.flush(label="finished")
    except BaseException as exc:
        # save what exists (logs, the last checkpoint) before the process dies; the supervisor
        # in run.py restarts the run from that checkpoint
        ctl.write_progress("failed", error=repr(exc)[:500])
        ctl.sync.flush(timeout=300.0, label=f"failed at round {ctl.round}")
        ctl.close()
        raise
    finally:
        _restore_stop_handlers(previous)
    ctl.close()
    print(f"[curator] {cfg.method} seed {cfg.seed}: final macro {final.get('macro')} "
          f"after {summary['rounds']} rounds, {summary['charged_gpu_hours']:.3f} charged GPU-h", flush=True)
    return summary


def _install_stop_handlers(ctl) -> dict:
    """SIGTERM / SIGINT -> stop at the next round boundary with a checkpoint (main thread only)."""
    previous = {}
    if threading.current_thread() is not threading.main_thread():
        return previous

    def handler(signum, _frame):
        ctl.request_stop(f"signal {signal.Signals(signum).name}")
        if ctl.sync is not None:
            ctl.sync.request(force=True, label="signal")     # upload the last checkpoint right away
        if signum == signal.SIGINT and previous.get(signum) not in (None, signal.SIG_IGN):
            print("[curator] interrupt: stopping at the next round boundary (press again to abort)", flush=True)
            signal.signal(signal.SIGINT, previous[signum])

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_stop_handlers(previous: dict) -> None:
    for sig, h in previous.items():
        try:
            signal.signal(sig, h)
        except (ValueError, OSError, TypeError):
            pass


def compare_report(root: Path, *, reference: str = "uniform") -> str:
    """Markdown comparison of every method against `reference`, paired by seed."""
    from curator_rl.core.stats import paired_t_interval, sign_flip_pvalue

    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(Path(root).glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(s["method"], {})[int(s["seed"])] = s
    lines = ["# CURATOR GPU comparison", "",
             "Final score = macro-average of greedy dev accuracy over gsm8k, math35, mbpp, countdown "
             "(uncharged evaluation, identical for every method; the sealed test is not used).", "",
             "| method | seeds | final macro (mean) | charged GPU-h | rounds | ever S5 | sandbox rejections |",
             "|---|---|---|---|---|---|---|"]
    for m, by_seed in sorted(runs.items()):
        vals = [s["final"].get("macro") for s in by_seed.values() if s["final"].get("macro") is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        hours = sum(s["charged_gpu_hours"] for s in by_seed.values()) / len(by_seed)
        rnds = sum(s["rounds"] for s in by_seed.values()) / len(by_seed)
        s5 = sorted({e for s in by_seed.values() for e in s["ever_s5"]})
        rej = sum(sum(s["sandbox_rejections"].values()) for s in by_seed.values())
        lines.append(f"| {m} | {len(by_seed)} | {mean:.4f} | {hours:.3f} | {rnds:.1f} | {', '.join(s5) or '-'} | {rej} |")
    if reference in runs:
        lines += ["", f"## Paired differences against {reference} (same seed)", "",
                  "| method | n | mean diff | 95% t-interval | sign-flip p |", "|---|---|---|---|---|"]
        for m, by_seed in sorted(runs.items()):
            if m == reference:
                continue
            seeds = sorted(set(by_seed) & set(runs[reference]))
            diffs = [by_seed[s]["final"]["macro"] - runs[reference][s]["final"]["macro"] for s in seeds
                     if by_seed[s]["final"].get("macro") is not None and runs[reference][s]["final"].get("macro") is not None]
            if len(diffs) >= 2:
                mean, lo, hi = paired_t_interval(diffs)
                lines.append(f"| {m} | {len(diffs)} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {sign_flip_pvalue(diffs):.3f} |")
            elif diffs:
                lines.append(f"| {m} | 1 | {diffs[0]:+.4f} | one seed: no interval | - |")
        lines += ["", "With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers "
                  "badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only."]
    return "\n".join(lines) + "\n"
