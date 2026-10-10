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
