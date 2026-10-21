"""Phase F runner on the CPU fake backend: real verifiers, scheduler, calibration, ledger, checkpoints.

Needs the processed datasets (scripts/download_data*.py + build_splits*.py); skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from curator_rl.core.config import load_config
from curator_rl.envs.registry import EnvRegistry
from curator_rl.trainer.checkpoint import latest_checkpoint, load_controller_state, save_checkpoint
from curator_rl.trainer.fake_backend import (
    SECONDS_PER_1K_TOKENS,
    STEP_OVERHEAD_SECONDS,
    FakeTrainer,
)
from curator_rl.trainer.round_controller import RoundController
from curator_rl.trainer.run_config import RunConfig

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (REPO / "data" / "processed" / "math35" / "train.jsonl").exists(),
    reason="run scripts/download_data*.py && scripts/build_splits*.py")


def _cfg(tmp_path, **kw) -> RunConfig:
    base = dict(method="curator", seed=3, backend="fake", steps_per_round=2, prompts_per_step=8, group_size=4,
                max_rounds=8, budget_gpu_hours=10.0, calib_interval=2, calib_items=12, report_every=4,
                report_items=8, final_items=8, checkpoint_every=2, verify_workers=4, out_dir=str(tmp_path))
    base.update(kw)
    return RunConfig(**base)


def _controller(cfg: RunConfig, out: Path) -> RoundController:
    root = load_config(REPO / "configs" / "base.yaml",
                       ["scheduler.warmup_rounds=2", "signals.status.n_min=8", "signals.status.r_min=2"])
    envs = {e: EnvRegistry(REPO, root.data).get(e) for e in cfg.envs}
    return RoundController(cfg, root, envs, out_dir=out, clock=lambda: 0.0)   # deterministic costs


def _run(cfg, out, *, stop_after_steps=None, resume=False):
    ctl = _controller(cfg, out)
    trainer = FakeTrainer(ctl, cfg.seed)
    ckpt = latest_checkpoint(out) if resume else None
    if ckpt is not None:
        ctl.load_state(load_controller_state(ckpt))
        trainer.load(ckpt)
    else:
        ctl.begin_round()
    trainer.train(lambda: save_checkpoint(out, ctl, trainer.save), stop_after_steps=stop_after_steps)
    return ctl, trainer


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_fake_curator_run_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    ctl, _ = _run(cfg, tmp_path / "run")
    final = ctl.report(cfg.final_items, tag="final")
    summary = ctl.summary(final)
    ctl.close()
    rounds = _lines(tmp_path / "run" / "rounds.jsonl")
    assert summary["rounds"] == 8 and len(rounds) == 8
    assert summary["calibrations"] >= 2                      # K = 2, Curator calibrates
    assert summary["ledger_seconds"]["calib"] > 0 or summary["calibrations"] > 0
    for r in rounds:                                          # every round delivered exactly R x P prompts
        assert sum(a["prompts"] for a in r["arms"].values()) == 2 * 8
        assert abs(sum(r["delivered"].values()) - 1.0) < 1e-9
    assert 0.0 <= final["macro"] <= 1.0
    assert (tmp_path / "run" / "checkpoints" / "latest.json").exists()


def test_uniform_never_calibrates_matched_budget(tmp_path):
    cfg = _cfg(tmp_path, method="uniform")
    ctl, _ = _run(cfg, tmp_path / "u")
    assert ctl.calib_k == 0 and ctl.ledger.seconds["calib"] == 0.0      # calibration charged only to its user
    ctl.close()


def test_kill_and_resume_reproduces_the_uninterrupted_run(tmp_path):
    cfg_a = _cfg(tmp_path / "a")
    ctl_a, _ = _run(cfg_a, tmp_path / "a" / "run")
    full = _lines(tmp_path / "a" / "run" / "rounds.jsonl")
    ctl_a.close()

    cfg_b = _cfg(tmp_path / "b")
    out_b = tmp_path / "b" / "run"
    ctl_b, _ = _run(cfg_b, out_b, stop_after_steps=11)     # "crash" mid round 6 (checkpoint at round 4)
    ctl_b.close()
    ckpt = latest_checkpoint(out_b)
    assert ckpt is not None and load_controller_state(ckpt)["round"] == 4
    ctl_c, _ = _run(cfg_b, out_b, resume=True)
    ctl_c.close()
    resumed = _lines(out_b / "rounds.jsonl")
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("_seq",)} for r in rows]  # noqa: E731
    assert strip(resumed) == strip(full)                   # identical decisions, deliveries and costs


def test_sysid_recovers_the_cost_model_and_separates_the_noisy_arm(tmp_path):
    from curator_rl.trainer.sysid import analyse

    cfg = _cfg(tmp_path, method="dirichlet", max_rounds=24, calib_interval=4, calib_items=40, report_every=0)
    ctl, _ = _run(cfg, tmp_path / "s")
    ctl.close()
    rep = analyse(tmp_path / "s", cfg.envs, cfg.group_size)
    cm = rep["cost_model"]
    assert cm["r2"] > 0.999
    assert cm["overhead_seconds_per_step"] == pytest.approx(STEP_OVERHEAD_SECONDS, abs=0.05)
    for e, a in SECONDS_PER_1K_TOKENS.items():
        assert cm["seconds_per_1k_tokens"][e] == pytest.approx(a, rel=0.05)
    assert rep["transfer"]["gsm8k"]["_windows"] >= 4


def test_clean_stop_then_resume_reproduces_the_uninterrupted_run(tmp_path):
    """A deadline / SIGTERM stop at a round boundary, then a resume, changes nothing."""
    cfg = _cfg(tmp_path, checkpoint_every=1)
    ctl_a, _ = _run(cfg, tmp_path / "full")
    full = _lines(tmp_path / "full" / "rounds.jsonl")
    final_a = ctl_a.report(cfg.final_items, tag="final")
    ctl_a.close()

    out = tmp_path / "cut"
    ctl = _controller(cfg, out)
    trainer = FakeTrainer(ctl, cfg.seed)
    ctl.save_backend = trainer.save
    ctl.begin_round()
    end_round = ctl._end_round

    def end_round_with_stop_request():
        if ctl.round == 2:
            ctl.request_stop("test: session deadline")
        end_round()

    ctl._end_round = end_round_with_stop_request
    trainer.train(lambda: ctl.save(trainer.save))
    assert ctl.preempted and ctl.round == 3 and ctl.last_checkpoint_round == 3
    assert json.loads((out / "progress.json").read_text())["status"] == "preempting"
    ctl.close()

    ctl_b, _ = _run(cfg, out, resume=True)
    final_b = ctl_b.report(cfg.final_items, tag="final")
    ctl_b.close()
    strip = lambda rows: [{k: v for k, v in r.items() if k != "_seq"} for r in rows]  # noqa: E731
    assert strip(_lines(out / "rounds.jsonl")) == strip(full)
    assert final_b["scores"] == final_a["scores"]


def test_a_failed_calibration_is_skipped_and_its_exposure_carries_over(tmp_path):
    cfg = _cfg(tmp_path)
    ctl = _controller(cfg, tmp_path / "r")
    trainer = FakeTrainer(ctl, cfg.seed)
    real = ctl.generate

    def flaky(items, cap):
        if ctl.round <= 2:                        # the first calibration (round 2) fails on every slice
            raise RuntimeError("generation backend hiccup")
        return real(items, cap)

    ctl.generate = flaky
    ctl.begin_round()
    trainer.train(lambda: ctl.save(trainer.save))
    rows = _lines(tmp_path / "r" / "calibration.jsonl")
    assert rows[0].get("skipped") is True and ctl.eval_errors >= 1
    ok = [r for r in rows if "scores" in r]
    assert ok and sum(ok[0]["exposure"].values()) == 4 * 2 * 8   # rounds 1-4: two windows of R x P prompts
    assert ctl.round == cfg.max_rounds                            # the run finished regardless
    ctl.close()


def test_supervisor_restarts_a_crashed_run_and_mirrors_everything(tmp_path):
    """run.py: a worker dies mid-run, is restarted from its checkpoint, finishes; the backup holds it."""
    import os
    import subprocess
    import sys

    out, backup = tmp_path / "out", tmp_path / "backup"
    shape = {"prompts_per_step": 4, "group_size": 4, "steps_per_round": 1, "calib_interval": 2,
             "calib_items": 6, "report_every": 0, "final_items": 6}
    env = dict(os.environ, CURATOR_FAULT_AT_STEP="3")
    cmd = [sys.executable, str(REPO / "kaggle" / "run.py"), "compare", "--backend", "fake", "--methods", "curator",
           "--seeds", "0", "--rounds", "5", "--budget-gpu-hours", "10", "--out", str(out), "--sync-to", str(backup),
           "--sync-every-min", "0", "--max-restarts", "2", "--shape", json.dumps(shape),
           "--set", "signals.status.n_min=8"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "crashed" in res.stdout and "attempt 2" in res.stdout
    # the notebook sees progress lines (the workers' own output goes to their logs)
    assert "curator_s0 finished: 5 rounds" in res.stdout and "1 crash(es)" in res.stdout
    # compare evaluates the untrained model too (D-101), once
    assert "compare: 2/2 runs finished, 0 running, 0 waiting" in res.stdout and "base_s0 finished: 0 rounds" in res.stdout
    log = (out / "logs" / "curator_s0.log").read_text(errors="replace")
    assert "injected fault at step 3" in log and "resuming from round-0002" in log
    summary = json.loads((out / "curator_s0" / "summary.json").read_text())
    assert summary["rounds"] == 5 and summary["status"] == "finished"
    assert json.loads((out / "curator_s0" / "progress.json").read_text())["status"] == "finished"
    mirrored = backup / "curator_s0"
    assert (mirrored / "summary.json").exists() and (mirrored / "checkpoints" / "latest.json").exists()
    assert len(_lines(mirrored / "rounds.jsonl")) == 5


def test_leave_one_out_runs_keep_the_full_benchmark_and_report_h5(tmp_path):
    """run.py loo: full portfolio + one run per arm left out; the benchmark always covers every domain."""
    import subprocess
    import sys

    out = tmp_path / "out"
    shape = {"prompts_per_step": 4, "group_size": 4, "steps_per_round": 1, "calib_interval": 1,
             "calib_items": 6, "report_every": 0, "final_items": 6}
    base = [sys.executable, str(REPO / "kaggle" / "run.py"), "--backend", "fake", "--seeds", "0", "--rounds", "3",
            "--budget-gpu-hours", "10", "--out", str(out), "--shape", json.dumps(shape), "--set", "signals.status.n_min=8"]
    res = subprocess.run(base[:2] + ["compare", "--methods", "curator"] + base[2:], capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    res = subprocess.run(base[:2] + ["loo", "--methods", "curator", "--arms", "noisy,countdown"] + base[2:],
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    wo = json.loads((out / "curator-wo-noisy_s0" / "summary.json").read_text())
    assert wo["label"] == "curator-wo-noisy" and "noisy" not in wo["envs"]
    assert set(wo["final"]["scores"]) == {"gsm8k", "math35", "mbpp", "countdown"}     # benchmark unchanged
    assert wo["mean_delivered_share"].get("noisy") is None
    report = (out / "comparison.md").read_text()
    assert "## Leave-one-out under curator, seed 0" in report and "ROI leaderboard: curator, seed 0" in report
    assert "H5 (ROI of the Curator run" in report


def test_a_finished_run_is_never_overwritten_and_a_lost_final_is_recomputed(tmp_path):
    from curator_rl.trainer.runner import run_single

    cfg = _cfg(tmp_path, max_rounds=3, report_every=0, overrides=["signals.status.n_min=8"])
    first = run_single(cfg, REPO)
    path = tmp_path / "curator_s3" / "summary.json"
    assert first["status"] == "finished" and first["final"]["macro"] is not None
    stamp = path.stat().st_mtime_ns
    reports = tmp_path / "curator_s3" / "reports.jsonl"
    logged = reports.read_text(encoding="utf-8")
    again = run_single(cfg, REPO)                       # re-launching a finished run changes nothing
    assert again["final"] == first["final"] and path.stat().st_mtime_ns == stamp
    assert reports.read_text(encoding="utf-8") == logged      # its final-evaluation row stays (it was dropped)
    path.unlink()                                       # the session died during the final evaluation
    redo = run_single(cfg, REPO)
    assert redo["final"]["scores"] == first["final"]["scores"] and redo["rounds"] == 3


def test_smoke_then_compare_with_the_same_sync_target(tmp_path):
    """The first Kaggle session: smoke and compare share --sync-to; the smoke run (also curator_s0,
    tiny shape) must never be restored as the real curator_s0."""
    import subprocess
    import sys

    out, backup = tmp_path / "out", tmp_path / "backup"
    base = [sys.executable, str(REPO / "kaggle" / "run.py")]
    common = ["--backend", "fake", "--out", str(out), "--sync-to", str(backup)]
    res = subprocess.run(base + ["smoke"] + common, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "SMOKE PASSED" in res.stdout, res.stdout[-2000:] + res.stderr[-2000:]
    assert (backup / "smoke" / "curator_s0" / "summary.json").exists() and not (backup / "curator_s0").exists()
    shape = {"prompts_per_step": 8, "group_size": 4, "steps_per_round": 1, "calib_interval": 2,
             "calib_items": 6, "report_every": 0, "final_items": 6}
    res = subprocess.run(base + ["compare", "--methods", "curator", "--seeds", "0", "--rounds", "3",
                                 "--budget-gpu-hours", "10", "--shape", json.dumps(shape),
                                 "--set", "signals.status.n_min=8"] + common,
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    summary = json.loads((out / "curator_s0" / "summary.json").read_text())
    assert summary["rounds"] == 3 and summary["run_config"]["prompts_per_step"] == 8
    assert (backup / "curator_s0" / "summary.json").exists()
    # at most 3 checkpoints per run; the results folder and its zip next to the runs, and uploaded
    assert len([p for p in (out / "curator_s0" / "checkpoints").iterdir() if p.is_dir()]) <= 3
    results = tmp_path / "results"
    assert (results / "curator_results.zip").exists() and (results / "runs" / "curator_s0" / "summary.json").exists()
    assert (backup / "results" / "curator_results.zip").exists()
    assert not (out / "smoke" / "results").exists() and not (tmp_path / "out" / "results").exists()


# ------------------------------------------------------------------ evaluation protocol 2 (D-101)

def test_the_untrained_model_is_evaluated_like_every_run(tmp_path):
    from curator_rl.trainer.runner import run_single

    s = run_single(_cfg(tmp_path, method="uniform", label="base", max_rounds=0,
                        overrides=["signals.status.n_min=8"]), REPO)
    assert s["rounds"] == 0 and s["charged_gpu_hours"] == 0 and s["final"]["protocol"] == 2
    assert s["final"]["sampled"]["macro"] is not None and s["final"]["sampled"]["k"] == 4
    assert set(s["final"]["items"]) == {"gsm8k", "math35", "mbpp", "countdown"}
    assert (tmp_path / "base_s3" / "summary.json").exists()


def test_a_run_finished_under_protocol_1_is_reevaluated_without_retraining(tmp_path):
    from curator_rl.trainer.runner import run_single

    cfg = _cfg(tmp_path, max_rounds=3, report_every=0, overrides=["signals.status.n_min=8"])
    first = run_single(cfg, REPO)
    run = tmp_path / "curator_s3"
    old = dict(first, final={k: v for k, v in first["final"].items() if k not in ("protocol", "sampled", "items")})
    (run / "summary.json").write_text(json.dumps(old), encoding="utf-8")
    steps = (run / "steps.jsonl").read_text(encoding="utf-8")
    redo = run_single(cfg, REPO)
    assert redo["final"]["protocol"] == 2 and redo["final"]["sampled"]["macro"] is not None
    assert "protocol" not in json.loads((run / "summary_v1.json").read_text(encoding="utf-8"))["final"]
    assert (run / "steps.jsonl").read_text(encoding="utf-8") == steps and redo["rounds"] == first["rounds"]
    assert [r["tag"] for r in _lines(run / "reports.jsonl")].count("final") == 1
    assert redo["final"]["scores"] == first["final"]["scores"]        # the same policy, the same verdicts


def test_sampled_evaluations_never_change_the_run_they_measure(tmp_path):
    rows = []
    for k in (4, 0):
        out = tmp_path / f"k{k}"
        _run(_cfg(tmp_path, max_rounds=6, report_every=2, eval_samples=k), out)
        rows.append([{key: v for key, v in r.items() if key != "_seq"} for r in _lines(out / "rounds.jsonl")])
        reports = _lines(out / "reports.jsonl")
        assert reports and all(("sampled" in r) == (k > 0) for r in reports)
    assert rows[0] == rows[1]


def test_compare_report_pairs_items_and_measures_the_gain_over_the_untrained_model(tmp_path):
    from curator_rl.trainer.runner import compare_report, run_single

    kw = dict(max_rounds=6, report_every=2, overrides=["signals.status.n_min=8"])
    for method in ("curator", "uniform"):
        run_single(_cfg(tmp_path, method=method, **kw), REPO)
    run_single(_cfg(tmp_path, method="uniform", label="base", max_rounds=0, overrides=kw["overrides"]), REPO)
    report = compare_report(tmp_path)
    assert "| base (untrained) |" in report and "## Item by item against uniform" in report
    assert "## Gain over the untrained model" in report and "## GPU-hours to reach uniform's final level" in report
    assert "Gains are fitted non-negative" in report


# ------------------------------------------------------------------ D-102: filtering, pre-flight check, namespaces

def test_preflight_verifier_check_flags_exactly_the_broken_rewards():
    from curator_rl.signals.verifier_fuzz import fuzz_arms

    root = load_config(REPO / "configs" / "base.yaml", [])
    reg = EnvRegistry(REPO, root.data)
    arms = ("gsm8k", "math35", "mbpp", "countdown", "noisy", "hackable")
    report = fuzz_arms({e: reg.get(e) for e in arms}, arms)
    assert report["flagged"] == ["hackable", "noisy"]
    for real in ("gsm8k", "math35", "mbpp", "countdown"):                # every junk probe earns nothing
        assert report["arms"][real]["junk_reward_rate"] == 0.0
    assert report["arms"]["hackable"]["probes"]["number_spray"] == 1.0  # the right number anywhere is paid
    assert report["arms"]["mbpp"]["probes"]["exit_early"] == 0.0         # the sandbox token cannot be skipped


def test_question_filtering_draws_more_questions_and_trains_on_informative_groups(tmp_path):
    from curator_rl.trainer.runner import run_single

    s = run_single(_cfg(tmp_path, method="uniform_q", group_filter=1.75, max_rounds=4,
                        overrides=["signals.status.n_min=8"]), REPO)
    steps = _lines(tmp_path / "uniform_q_s3" / "steps.jsonl")
    assert s["status"] == "finished" and steps
    assert all(sum(st["prompts"].values()) == 14 for st in steps)      # ceil(8 x 1.75) drawn per step
    assert s["preflight"]["flagged"] == ["noisy"]                       # reported for every method


def test_curator_q_starts_an_arm_whose_reward_pays_for_junk_in_s5(tmp_path):
    from curator_rl.trainer.runner import run_single

    over = ["signals.status.n_min=8", "signals.reward_audit.enabled=true", "signals.reward_audit.preflight_gate=true"]
    run_single(_cfg(tmp_path, method="curator_q", group_filter=1.75, max_rounds=3, overrides=over), REPO)
    rounds = _lines(tmp_path / "curator_q_s3" / "rounds.jsonl")
    assert str(rounds[0]["statuses"]["noisy"]).startswith("S5")          # from the first round boundary
    assert not any(str(v).startswith("S5") for e, v in rounds[0]["statuses"].items() if e != "noisy")


def test_run_py_namespace_keeps_runs_results_and_backup_apart(tmp_path):
    import subprocess
    import sys

    out, backup = tmp_path / "curator_runs", tmp_path / "backup"
    res = subprocess.run([sys.executable, str(REPO / "kaggle" / "run.py"), "compare", "--backend", "fake",
                          "--methods", "uniform_q", "--seeds", "0", "--rounds", "2", "--out", str(out),
                          "--sync-to", str(backup), "--sync-every-min", "0", "--namespace", "stress", "--no-base",
                          "--envs", "gsm8k,math35,mbpp,countdown,hackable"],
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    run = tmp_path / "curator_runs_stress" / "uniform_q_s0"
    assert (run / "summary.json").exists() and not out.exists()
    assert json.loads((run / "run_config.json").read_text())["envs"] == ["gsm8k", "math35", "mbpp", "countdown",
                                                                         "hackable"]
    assert (tmp_path / "results_stress" / "comparison.md").exists()
    assert (backup / "stress" / "uniform_q_s0" / "summary.json").exists()
    report = (tmp_path / "results_stress" / "comparison.md").read_text(encoding="utf-8")
    assert "pass@k macro" in report and "| hackable |" in report
