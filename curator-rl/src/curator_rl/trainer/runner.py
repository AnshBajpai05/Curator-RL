"""Single-run driver and comparison report for real (or fake) GRPO runs under a scheduler.

`run_single` builds the environments and config, resumes from the newest checkpoint when asked,
trains until the budget or the round limit, runs the final uncharged dev evaluation and writes
`summary.json`. `compare_report` pairs the summaries of several methods by seed. With fewer than
20 seeds it uses the t-interval and the exact sign-flip test (core.stats, audit E-11), never a
percentile bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path

from curator_rl.core.atomic import atomic_write_json
from curator_rl.core.config import load_config, rescale_for_batch_shape
from curator_rl.envs.registry import EnvRegistry
from curator_rl.trainer.checkpoint import latest_checkpoint, load_controller_state, save_checkpoint
from curator_rl.trainer.round_controller import RoundController
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


def run_single(cfg: RunConfig, repo_root: Path, *, resume: bool = True,
               gradient_checkpointing: bool = True) -> dict:
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    root_cfg = load_config(Path(repo_root) / "configs" / "base.yaml", cfg.overrides)
    if not any(o.startswith("signals.status.n_min=") for o in cfg.overrides):
        root_cfg = rescale_for_batch_shape(root_cfg, cfg.prompts_per_round)      # audit E-28
    registry = EnvRegistry(Path(repo_root), root_cfg.data)
    envs = {e: registry.get(e) for e in cfg.envs}
    ctl = RoundController(cfg, root_cfg, envs, out_dir=out)
    backend = resolve_backend(cfg.backend)
    ckpt = latest_checkpoint(out) if resume else None
    (out / "run_config.json").write_text(cfg.to_json(), encoding="utf-8")
    print(f"[curator] {cfg.method} seed {cfg.seed}: backend={backend}, "
          f"{'resuming from ' + ckpt.name if ckpt else 'fresh start'}", flush=True)

    if ckpt is not None:
        ctl.load_state(load_controller_state(ckpt))
    else:
        ctl.begin_round()

    if not ctl.stop:
        if backend == "fake":
            from curator_rl.trainer.fake_backend import FakeTrainer

            trainer = FakeTrainer(ctl, cfg.seed)
            if ckpt is not None:
                trainer.load(ckpt)
            trainer.train(lambda: save_checkpoint(out, ctl, trainer.save))
            ctl.save_backend = trainer.save
        else:
            from curator_rl.trainer.trl_adapter import run_trl

            run_trl(ctl, cfg, out, backend=backend, resume_dir=ckpt, gradient_checkpointing=gradient_checkpointing)
        if getattr(ctl, "save_backend", None) is not None:
            save_checkpoint(out, ctl, ctl.save_backend)        # final state, for later analysis
    final = ctl.report(cfg.final_items, tag="final")
    summary = ctl.summary(final)
    summary["backend"] = backend
    atomic_write_json(out / "summary.json", summary)
    ctl.close()
    print(f"[curator] {cfg.method} seed {cfg.seed}: final macro {final.get('macro')} "
          f"after {summary['rounds']} rounds, {summary['charged_gpu_hours']:.3f} charged GPU-h", flush=True)
    return summary


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
