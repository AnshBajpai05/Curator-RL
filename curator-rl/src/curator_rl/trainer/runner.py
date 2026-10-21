"""Single-run driver and comparison report for real (or fake) GRPO runs under a scheduler.

`run_single` builds the environments and config, resumes from the newest checkpoint when asked,
trains until the budget or the round limit, runs the final uncharged dev evaluation and writes
`summary.json`. `compare_report` pairs the summaries of several methods by seed. With fewer than
20 seeds it uses the t-interval and the exact sign-flip test (core.stats, audit E-11), never a
percentile bootstrap.
"""

from __future__ import annotations

import json
import math
import os
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
from curator_rl.trainer.round_controller import EVAL_PROTOCOL, SYNC_RESERVE_SECONDS, RoundController
from curator_rl.trainer.run_config import BENCH_DOMAINS, RunConfig


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
    return Path(cfg.out_dir) / f"{cfg.name}_s{cfg.seed}"


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
    # training arms plus every benchmark domain: a leave-one-out run trains without an arm but is still
    # scored on the full benchmark
    envs = {e: registry.get(e) for e in sorted(set(cfg.envs) | set(BENCH_DOMAINS))}
    ctl = RoundController(cfg, root_cfg, envs, out_dir=out)
    sync = ArtifactSync(out, cfg.sync_to, min_interval_s=60.0 * cfg.sync_every_min)
    ctl.sync = sync
    backend = resolve_backend(cfg.backend)
    ckpt = latest_checkpoint(out) if resume else None
    summary_path = out / "summary.json"
    if ckpt is not None and summary_path.exists() and _protocol(summary_path) < EVAL_PROTOCOL:
        # finished under an older evaluation protocol: keep that summary and evaluate the saved final
        # policy again with the current one (no training; D-101)
        os.replace(summary_path, out / "summary_v1.json")
        print(f"[curator] {cfg.name} seed {cfg.seed}: evaluated with protocol {_protocol(out / 'summary_v1.json')}; "
              f"re-evaluating the final checkpoint with protocol {EVAL_PROTOCOL} (old summary kept as "
              "summary_v1.json)", flush=True)
    (out / "run_config.json").write_text(cfg.to_json(), encoding="utf-8")
    print(f"[curator] {cfg.method} seed {cfg.seed}: backend={backend}, "
          f"{'resuming from ' + ckpt.name if ckpt else 'fresh start'}"
          f"{', syncing to ' + cfg.sync_to if cfg.sync_to else ''}", flush=True)
    if ckpt is None and cfg.max_rounds > 0:
        _preflight_verifiers(ctl, cfg, root_cfg, out)     # a resumed run has its flags in the checkpoint

    if ckpt is not None:
        ctl.load_state(load_controller_state(ckpt))
        ctl.last_checkpoint_round = ctl.round
    elif cfg.max_rounds == 0:
        # the untrained model (run "base"): no training, only the final evaluation. The reference every
        # method starts from, evaluated exactly as they are
        ctl.stop, ctl.stop_reason = True, "max_rounds"
    else:
        ctl.begin_round()
    if ctl.stop and summary_path.exists():
        # training is complete and the final evaluation already ran: nothing to redo (re-launching a
        # finished run must never overwrite its result)
        ctl.close()
        print(f"[curator] {cfg.name} seed {cfg.seed}: already finished; keeping {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
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
        elif ctl.generate is None:
            # training finished in an earlier session but the final evaluation did not complete (or
            # used an older protocol), or this is the untrained model: evaluation only
            _attach_eval_backend(ctl, cfg, backend, ckpt)
        if ctl.preempted:
            ctl.write_progress("preempted")
            ok = sync.flush(label=f"preempted at round {ctl.round}")
            ctl.write_progress("preempted")               # again, with the final sync recorded
            print(f"[curator] {cfg.name} seed {cfg.seed}: stopped cleanly at round {ctl.round} "
                  f"({ctl.stop_reason}); checkpoint saved{', synced' if ok else ''}. Resume to continue.",
                  flush=True)
            ctl.close()
            return {"status": "preempted", "method": cfg.method, "seed": cfg.seed, "rounds": ctl.round,
                    "stop_reason": ctl.stop_reason}
        final = ctl.report(cfg.final_items, tag="final")
        summary = ctl.summary(final)
        summary["backend"] = backend
        if (out / "fuzz.json").exists():
            fz = json.loads((out / "fuzz.json").read_text(encoding="utf-8"))
            summary["preflight"] = {"flagged": fz.get("flagged", []),
                                    "junk_reward_rate": {e: r.get("junk_reward_rate") for e, r in fz["arms"].items()}}
        summary["status"] = "finished"
        atomic_write_json(out / "summary.json", summary)
        ctl.write_progress("finished")
        sync.flush(label="finished")
        ctl.write_progress("finished")                    # again, with the final sync recorded
    except BaseException as exc:
        # save what exists (logs, the last checkpoint) before the process dies; the supervisor
        # in run.py restarts the run from that checkpoint
        ctl.write_progress("failed", error=repr(exc)[:500])
        sync.flush(timeout=300.0, label=f"failed at round {ctl.round}")
        ctl.close()
        raise
    finally:
        _restore_stop_handlers(previous)
    ctl.close()
    print(f"[curator] {cfg.method} seed {cfg.seed}: final macro {final.get('macro')} "
          f"after {summary['rounds']} rounds, {summary['charged_gpu_hours']:.3f} charged GPU-h", flush=True)
    return summary


def _preflight_verifiers(ctl, cfg: RunConfig, root_cfg, out: Path) -> None:
    """Pre-flight verifier check (D-102): probe each arm's reward with junk before training.

    Writes fuzz.json. With `signals.reward_audit.preflight_gate`, a flagged arm starts in S5 (schedulers
    with a reward audit). Seconds, no GPU, never stops the run.
    """
    from curator_rl.signals.verifier_fuzz import fuzz_arms

    try:
        report = fuzz_arms(ctl.envs, list(cfg.envs))
        atomic_write_json(out / "fuzz.json", report)
    except Exception as exc:  # noqa: BLE001
        print(f"[curator] pre-flight verifier check skipped: {exc!r}", flush=True)
        return
    if report["flagged"]:
        print(f"[curator] pre-flight verifier check: the reward pays for junk on {', '.join(report['flagged'])}",
              flush=True)
    preflag = getattr(ctl.scheduler, "preflag_unreliable", None)
    if report["flagged"] and preflag is not None and root_cfg.signals.reward_audit.preflight_gate:
        preflag(report["flagged"])
        print(f"[curator] {', '.join(report['flagged'])}: S5 from the start (pre-flight gate)", flush=True)


def _protocol(summary_path: Path) -> int:
    try:
        return int((json.loads(summary_path.read_text(encoding="utf-8")).get("final") or {}).get("protocol", 1))
    except (OSError, ValueError):
        return EVAL_PROTOCOL           # unreadable: never move it aside on a guess


def _attach_eval_backend(ctl, cfg: RunConfig, backend: str, ckpt) -> None:
    if ckpt is None and cfg.max_rounds > 0:
        raise RuntimeError("nothing to evaluate: the run is complete but has no checkpoint")
    if backend == "fake":
        from curator_rl.trainer.fake_backend import FakeTrainer

        trainer = FakeTrainer(ctl, cfg.seed)      # sets ctl.generate (the untrained policy)
        if ckpt is not None:
            trainer.load(ckpt)
    else:
        from curator_rl.trainer.trl_adapter import attach_eval_generate

        attach_eval_generate(ctl, cfg, ckpt, backend=backend)
    what = "the untrained model" if ckpt is None else "the saved policy (training already complete)"
    print(f"[curator] {cfg.name} seed {cfg.seed}: evaluating {what}", flush=True)


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
    """Markdown comparison of every method against `reference`: item-paired (protocol 2) and by seed."""
    from curator_rl.core.stats import paired_t_interval, sign_flip_pvalue
    from curator_rl.evaluation.paired_items import (
        hours_to_reach,
        learning_curve,
        paired_item_diff,
        pass_at_k,
    )

    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(Path(root).glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(s.get("label") or s["method"], {})[int(s["seed"])] = s
    base = next(iter(runs.pop("base", {}).values()), None)          # the untrained model (one run)

    def sampled(s: dict) -> float | None:
        return ((s.get("final") or {}).get("sampled") or {}).get("macro")

    def mean(xs: list) -> float | None:
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def fmt(x: float | None, spec: str = ".4f") -> str:
        return "-" if x is None else format(x, spec)

    k = next((((s.get("final") or {}).get("sampled") or {}).get("k") for by in runs.values() for s in by.values()
              if sampled(s) is not None), None)
    lines = ["# CURATOR GPU comparison", "",
             f"Sampled accuracy = mean over {k or 'k'} samples per item at the training temperature (evaluation "
             "protocol 2, D-101: the primary measure, since RL raises sampled accuracy before greedy). Greedy = "
             "the single most likely answer. Both: macro-average over gsm8k, math35, mbpp, countdown on the "
             "same dev items for every method (uncharged; the sealed test is not used).", "",
             "| method | seeds | sampled macro (mean) | pass@k macro | greedy macro (mean) | charged GPU-h | rounds | "
             "ever S5 | sandbox rejections |", "|---|---|---|---|---|---|---|---|---|"]
    if base is not None:
        lines.append(f"| base (untrained) | - | {fmt(sampled(base))} | {fmt(pass_at_k(base['final']))} | "
                     f"{fmt(base['final'].get('macro'))} | 0 | 0 | - | - |")
    for m, by_seed in sorted(runs.items()):
        hours = sum(s["charged_gpu_hours"] for s in by_seed.values()) / len(by_seed)
        rnds = sum(s["rounds"] for s in by_seed.values()) / len(by_seed)
        s5 = sorted({e for s in by_seed.values() for e in s["ever_s5"]})
        rej = sum(sum(s["sandbox_rejections"].values()) for s in by_seed.values())
        lines.append(f"| {m} | {len(by_seed)} | {fmt(mean([sampled(s) for s in by_seed.values()]))} | "
                     f"{fmt(mean([pass_at_k(s['final']) for s in by_seed.values()]))} | "
                     f"{fmt(mean([s['final'].get('macro') for s in by_seed.values()]))} | {hours:.3f} | {rnds:.1f} | "
                     f"{', '.join(s5) or '-'} | {rej} |")
    domains = list(BENCH_DOMAINS)
    lines += ["", "## Per domain (sampled accuracy; greedy in brackets; mean over seeds)", "",
              "| method | " + " | ".join(domains) + " |", "|---|" + "---|" * len(domains)]
    for m, group in ([("base (untrained)", [base])] if base is not None else []) + \
            [(m, list(by.values())) for m, by in sorted(runs.items())]:
        cells = []
        for d in domains:
            sv = mean([((s["final"].get("sampled") or {}).get("scores") or {}).get(d) for s in group])
            gv = mean([(s["final"].get("scores") or {}).get(d) for s in group])
            cells.append(f"{fmt(sv, '.3f')} ({fmt(gv, '.3f')})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    # pre-flight verifier check (D-102): the same arms give the same answer in every run, so merge them
    junk: dict[str, float] = {}
    flagged: set[str] = set()
    for by in runs.values():
        for s in by.values():
            pre = s.get("preflight") or {}
            junk.update({e: r for e, r in (pre.get("junk_reward_rate") or {}).items() if r is not None})
            flagged |= set(pre.get("flagged") or [])
    if junk:
        lines += ["", "## Pre-flight verifier check (D-102)", "",
                  "Reward each arm paid for junk before any training (empty or echoed answers, wrong or sprayed "
                  "numbers, code that exits early or prints its globals). Above 5% the checker can be gamed.", "",
                  "| arm | junk reward rate | flagged |", "|---|---|---|"]
        lines += [f"| {e} | {junk[e]:.2f} | {'yes' if e in flagged else '-'} |" for e in sorted(junk)]

    # item by item: against the reference, and against the untrained model
    if reference in runs:
        lines += ["", f"## Item by item against {reference} (same seed, same items)", "",
                  "Paired over items: the SE is the evaluation's, for one seed. Whether a difference holds across "
                  "training seeds is the across-seed interval below it.", "",
                  "| method | seed | sampled macro diff (SE) | greedy macro diff (SE) | sampled diff per domain |",
                  "|---|---|---|---|---|"]
        per_seed: dict[str, list[float]] = {}
        for m, by_seed in sorted(runs.items()):
            if m == reference:
                continue
            for seed in sorted(set(by_seed) & set(runs[reference])):
                a, b = by_seed[seed]["final"], runs[reference][seed]["final"]
                ps, pg = paired_item_diff(a, b, "sampled"), paired_item_diff(a, b, "greedy")
                if ps is None and pg is None:
                    continue
                if ps is not None:
                    per_seed.setdefault(m, []).append(ps[0])
                dom = ", ".join(f"{d} {v:+.3f}" for d, v in ps[2].items()) if ps else "-"
                lines.append(f"| {m} | {seed} | {f'{ps[0]:+.4f} ({ps[1]:.4f})' if ps else '-'} | "
                             f"{f'{pg[0]:+.4f} ({pg[1]:.4f})' if pg else '-'} | {dom} |")
        for m, diffs in sorted(per_seed.items()):
            if len(diffs) >= 2:
                mu, lo, hi = paired_t_interval(diffs)
                lines.append(f"\n{m} vs {reference}, sampled macro over {len(diffs)} seeds: {mu:+.4f}, 95% t-interval "
                             f"[{lo:+.4f}, {hi:+.4f}], sign-flip p {sign_flip_pvalue(diffs):.3f}")
    if base is not None:
        lines += ["", "## Gain over the untrained model (item by item, sampled)", "",
                  "| method | seed | sampled macro gain (SE) | per domain |", "|---|---|---|---|"]
        for m, by_seed in sorted(runs.items()):
            for seed, s in sorted(by_seed.items()):
                ps = paired_item_diff(s["final"], base["final"], "sampled")
                if ps is not None:
                    dom = ", ".join(f"{d} {v:+.3f}" for d, v in ps[2].items())
                    lines.append(f"| {m} | {seed} | {ps[0]:+.4f} ({ps[1]:.4f}) | {dom} |")

    # learning curves on the same items: compute to reach the reference's final level
    n_subset = next((int(s["run_config"].get("report_items", 100)) for by in runs.values() for s in by.values()), 100)
    rows_out = []
    for seed in sorted(runs.get(reference, {})):
        ref_curve = learning_curve(_rows(Path(root) / f"{reference}_s{seed}" / "reports.jsonl"), n_subset=n_subset,
                                   base_final=base["final"] if base else None)
        if len(ref_curve) < 2:
            continue
        target, ref_hours = ref_curve[-1][1], ref_curve[-1][0]
        for m, by_seed in sorted(runs.items()):
            if m == reference or seed not in by_seed or "-wo-" in m:
                continue
            curve = learning_curve(_rows(Path(root) / f"{m}_s{seed}" / "reports.jsonl"), n_subset=n_subset,
                                   base_final=base["final"] if base else None)
            if len(curve) < 2:
                continue
            h = hours_to_reach(curve, target)
            saved = f"{100 * (1 - h / ref_hours):+.0f}%" if h is not None and ref_hours else "-"
            rows_out.append(f"| {m} | {seed} | {target:.4f} at {ref_hours:.2f} | "
                            f"{'not reached (best ' + format(max(v for _, v in curve), '.4f') + ')' if h is None else f'{h:.2f}'} "
                            f"| {saved} |")
    if rows_out:
        lines += ["", f"## GPU-hours to reach {reference}'s final level (sampled, the same {n_subset} items per domain)",
                  "", "Learning curves start at the untrained model (0 GPU-h). Fewer GPU-hours = the same accuracy "
                  "for less compute.", "",
                  f"| method | seed | {reference} final (at GPU-h) | GPU-h to reach it | compute saved |",
                  "|---|---|---|---|---|"] + rows_out

    # protocol 1 (greedy, by seed): kept for runs evaluated before D-101
    if reference in runs:
        lines += ["", f"## Greedy, paired by seed against {reference} (protocol 1)", "",
                  "| method | n | mean diff | 95% t-interval | sign-flip p |", "|---|---|---|---|---|"]
        for m, by_seed in sorted(runs.items()):
            if m == reference:
                continue
            seeds = sorted(set(by_seed) & set(runs[reference]))
            diffs = [by_seed[s]["final"]["macro"] - runs[reference][s]["final"]["macro"] for s in seeds
                     if by_seed[s]["final"].get("macro") is not None and runs[reference][s]["final"].get("macro") is not None]
            if len(diffs) >= 2:
                mu, lo, hi = paired_t_interval(diffs)
                lines.append(f"| {m} | {len(diffs)} | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {sign_flip_pvalue(diffs):.3f} |")
            elif diffs:
                lines.append(f"| {m} | 1 | {diffs[0]:+.4f} | one seed: no interval | - |")
        lines += ["", "With n < 20 seeds the interval is a Student-t interval (a percentile bootstrap undercovers "
                  "badly at this size: 75% at n = 3, 83% at n = 5; audit E-11). One seed is descriptive only."]
    lines += roi_sections(Path(root), runs)
    return "\n".join(lines) + "\n"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def gpu_hours_per_prompt(run: Path) -> dict[str, float]:
    """Measured training GPU-hours per prompt per arm, from the run's round log."""
    secs: dict[str, float] = {}
    prompts: dict[str, int] = {}
    for r in _rows(run / "rounds.jsonl"):
        for e, a in r["arms"].items():
            secs[e] = secs.get(e, 0.0) + float(a.get("gpu_seconds") or 0.0)
            prompts[e] = prompts.get(e, 0) + int(a.get("prompts") or 0)
    return {e: secs[e] / prompts[e] / 3600.0 for e in secs if prompts.get(e)}


def roi_sections(root: Path, runs: dict[str, dict[int, dict]]) -> list[str]:
    """ROI leaderboard of every calibrating run, and the leave-one-out table with the H5 verdict."""
    from curator_rl.evaluation.loo import h5_verdict
    from curator_rl.roi.leaderboard import render, roi_table
    from curator_rl.trainer.run_config import BENCH_DOMAINS, DOMAIN_OF

    weights = {d: 1.0 / len(BENCH_DOMAINS) for d in BENCH_DOMAINS}
    lines: list[str] = []
    roi_by_seed: dict[int, dict[str, float]] = {}
    for name, by_seed in sorted(runs.items()):
        if "-wo-" in name:
            continue
        for seed, summ in sorted(by_seed.items()):
            run = root / f"{name}_s{seed}"
            calib = [r for r in _rows(run / "calibration.jsonl") if "scores" in r]
            if len(calib) < 2:
                continue
            envs = summ.get("envs") or list(summ.get("mean_delivered_share", {}))
            table = roi_table(calib, envs, gpu_hours_per_prompt(run), weights, domain_of=DOMAIN_OF)
            if name.startswith("curator") and seed not in roi_by_seed:
                roi_by_seed[seed] = {r.env_id: r.roi for r in table}
            s5 = sorted(set(summ.get("ever_s5") or []))
            informative = any(r.roi_lo > 0 for r in table if r.env_id not in s5 and math.isfinite(r.roi_lo))
            lines += ["", f"## ROI leaderboard: {name}, seed {seed} ({len(calib)} calibration evaluations)", "",
                      "Benchmark gain per GPU-hour spent on each arm, credited on the slice the arm is scored on "
                      "(roi/leaderboard.py, `members`). From one run this is descriptive: in the simulator one run "
                      "ranks arms like leave-one-out with median tau 0.3-0.5 (reports/analysis/roi_loo_sim.md). "
                      "Gains are fitted non-negative: 0.0000 means no positive evidence, not a measured zero.", ""]
            if s5:
                lines += [f"Flagged S5 (rewards that do not depend on the prompt): {', '.join(s5)}. Their ROI is "
                          "not meaningful, whatever its rank.", ""]
            if not informative:
                lines += [f"**Not a ranking:** with {len(calib)} calibration evaluations no (unflagged) arm's interval "
                          "excludes zero, so the order below is noise.", ""]
            lines.append(render(table))
    loo_bases = sorted({n.split("-wo-")[0] for n in runs if "-wo-" in n})
    for base in loo_bases:
        for seed, full in sorted(runs.get(base, {}).items()):
            arms = {n.split("-wo-")[1]: by[seed] for n, by in runs.items()
                    if n.startswith(base + "-wo-") and seed in by}
            if not arms or full["final"].get("macro") is None:
                continue
            contrib = {a: full["final"]["macro"] - s["final"]["macro"] for a, s in arms.items()
                       if s["final"].get("macro") is not None}
            lines += ["", f"## Leave-one-out under {base}, seed {seed}", "",
                      "| arm left out | final macro without it | contribution (full - without) |", "|---|---|---|"]
            for a in sorted(contrib, key=lambda a: -contrib[a]):
                lines.append(f"| {a} | {arms[a]['final']['macro']:.4f} | {contrib[a]:+.4f} |")
            roi = roi_by_seed.get(seed)
            if roi and set(contrib) <= set(roi):
                v = h5_verdict({a: roi[a] for a in contrib}, contrib, [a for a in ("noisy", "countdown") if a in contrib])
                lines += ["", f"H5 (ROI of the Curator run vs these contributions): Kendall tau {v['tau']:+.2f} "
                          f"(needs >= 0.4); junk arms in the bottom two of both: {v['junk_bottom']} -> "
                          f"**{'PASS' if v['pass'] else 'FAIL'}** (one seed: descriptive)."]
    return lines
