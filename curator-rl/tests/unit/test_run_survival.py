"""A real run must survive losing its machine and must not die of one bad component.

Covers: artifact staging and mirroring (`trainer.artifacts`), fetching the newest state back,
verifier and evaluation failures isolated (`RewardRouter`, `EvalSuite`), and independent
per-rollout noisy rewards (audit E-24). No datasets or GPU needed.
"""

from __future__ import annotations

import json
import types

import pytest

from curator_rl.core.types import Prompt, Verdict
from curator_rl.envs.noisy import NoisyRewardEnv
from curator_rl.trainer.artifacts import (
    ArtifactSync,
    fetch_runs,
    merge_runs,
    parse_target,
    stage_snapshot,
)
from curator_rl.trainer.calibration_eval import EvalSuite
from curator_rl.trainer.reward_router import RewardRouter


def _fake_run(root, name="curator_s0", rounds=(4, 5), finished=False):
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    for r in rounds:
        ck = run / "checkpoints" / f"round-{r:04d}"
        ck.mkdir()
        (ck / "controller.json").write_text(json.dumps({"round": r}))
        (ck / "adapter").mkdir()
        (ck / "adapter" / "weights.bin").write_bytes(bytes([r]) * 64)
    last = max(rounds)
    (run / "checkpoints" / "latest.json").write_text(json.dumps({"dir": f"round-{last:04d}", "round": last}))
    (run / "rounds.jsonl").write_text('{"round": 1}\n{"round": 2}\n')
    (run / "run_config.json").write_text("{}")
    (run / "trl").mkdir()
    (run / "trl" / "scratch.bin").write_bytes(b"x" * 10)
    if finished:
        (run / "summary.json").write_text("{}")
    return run


# ------------------------------------------------------------------------------------ artifacts

def test_parse_target():
    assert parse_target("") == ("none", "", "")
    assert parse_target("hf://me/runs") == ("hf", "me/runs", "")
    assert parse_target("hf://me/runs/exp1/a") == ("hf", "me/runs", "exp1/a")
    with pytest.raises(ValueError):
        parse_target("hf://onlyowner")
    assert parse_target("/mnt/backup")[0] == "dir"


def test_snapshot_holds_only_the_latest_checkpoint_and_the_logs(tmp_path):
    run = _fake_run(tmp_path)
    stage = stage_snapshot(run, tmp_path / "stage")
    assert sorted(p.name for p in (stage / "checkpoints").iterdir()) == ["latest.json", "round-0005"]
    assert (stage / "checkpoints" / "round-0005" / "adapter" / "weights.bin").read_bytes() == bytes([5]) * 64
    assert (stage / "rounds.jsonl").exists() and not (stage / "trl").exists()


def test_snapshot_is_unaffected_by_later_appends(tmp_path):
    run = _fake_run(tmp_path)
    stage = stage_snapshot(run, tmp_path / "stage")
    with (run / "rounds.jsonl").open("a") as fh:          # training keeps logging while uploading
        fh.write('{"round": 3}\n')
    assert (stage / "rounds.jsonl").read_text().count("\n") == 2


def test_dir_sync_mirrors_the_run_and_drops_superseded_checkpoints(tmp_path):
    run = _fake_run(tmp_path / "out")
    sync = ArtifactSync(run, str(tmp_path / "backup"), min_interval_s=0.0, log=lambda m: None)
    assert sync.flush()
    ck = tmp_path / "backup" / "curator_s0" / "checkpoints"
    assert sorted(p.name for p in ck.iterdir()) == ["latest.json", "round-0005"]
    # a newer checkpoint replaces the old one in the mirror
    new = run / "checkpoints" / "round-0006"
    new.mkdir()
    (new / "controller.json").write_text("{}")
    (run / "checkpoints" / "latest.json").write_text(json.dumps({"dir": "round-0006", "round": 6}))
    assert sync.flush()
    assert sorted(p.name for p in ck.iterdir()) == ["latest.json", "round-0006"]
    assert sync.status["ok"] == 2 and sync.status["last_round"] == 6


def test_upload_failures_are_recorded_never_raised(tmp_path):
    run = _fake_run(tmp_path / "out")

    def broken(stage, label):
        raise ConnectionError("network down")

    messages = []
    sync = ArtifactSync(run, "hf://me/runs", min_interval_s=0.0, uploader=broken, log=messages.append)
    assert sync.request(force=True)
    sync.wait()
    assert sync.status["failed"] == 1 and "network down" in sync.status["last_error"]
    assert not sync.flush()                   # still failing: reported as False, no exception
    assert messages


def test_sync_is_rate_limited_but_flush_always_runs(tmp_path):
    run = _fake_run(tmp_path / "out")
    calls = []
    now = [0.0]
    sync = ArtifactSync(run, "hf://me/runs", min_interval_s=300.0, clock=lambda: now[0],
                        uploader=lambda stage, label: calls.append(label))
    assert sync.request(label="a")
    sync.wait()
    now[0] = 100.0
    assert not sync.request(label="b")        # too soon
    assert sync.flush(label="final")          # forced
    assert calls == ["a", "final"]


def test_no_target_is_a_no_op(tmp_path):
    run = _fake_run(tmp_path / "out")
    sync = ArtifactSync(run, "", log=lambda m: None)
    assert not sync.enabled and not sync.request(force=True) and sync.flush()


def test_merge_takes_only_newer_state(tmp_path):
    backup, out = tmp_path / "backup", tmp_path / "out"
    _fake_run(backup, "curator_s0", rounds=(7, 8))
    _fake_run(backup, "uniform_s0", rounds=(2, 3))
    _fake_run(out, "uniform_s0", rounds=(4, 5))           # local is ahead: keep it
    taken = merge_runs(backup, out, log=lambda m: None)
    assert taken == ["curator_s0"]
    assert json.loads((out / "curator_s0" / "checkpoints" / "latest.json").read_text())["round"] == 8
    assert json.loads((out / "uniform_s0" / "checkpoints" / "latest.json").read_text())["round"] == 5


def test_merge_prefers_a_finished_run(tmp_path):
    backup, out = tmp_path / "backup", tmp_path / "out"
    _fake_run(backup, "lp_s1", rounds=(4, 5), finished=True)
    _fake_run(out, "lp_s1", rounds=(4, 5))
    assert merge_runs(backup, out, log=lambda m: None) == ["lp_s1"]
    assert (out / "lp_s1" / "summary.json").exists()


def test_merge_restores_lines_a_finished_runs_local_copy_lost(tmp_path):
    """An older version rewound a finished run's logs on relaunch, dropping its final-evaluation row."""
    backup, out = tmp_path / "backup", tmp_path / "out"
    src = _fake_run(backup, "uniform_s0", finished=True)
    dst = _fake_run(out, "uniform_s0", finished=True)
    (src / "reports.jsonl").write_text('{"tag": "periodic"}\n{"tag": "final"}\n')
    (dst / "reports.jsonl").write_text('{"tag": "periodic"}\n')
    (dst / "rounds.jsonl").write_text('{"round": 9}\n')               # different content: never replaced
    log: list[str] = []
    assert merge_runs(backup, out, log=log.append) == []                  # not newer: the run is not replaced
    assert (dst / "reports.jsonl").read_text().count('"final"') == 1
    assert (dst / "rounds.jsonl").read_text() == '{"round": 9}\n'
    assert any("restored log lines missing locally (reports.jsonl)" in m for m in log)


def test_fetch_from_a_directory_round_trips_a_sync(tmp_path):
    run = _fake_run(tmp_path / "session1")
    ArtifactSync(run, str(tmp_path / "backup"), min_interval_s=0.0, log=lambda m: None).flush()
    taken = fetch_runs(str(tmp_path / "backup"), tmp_path / "session2", log=lambda m: None)
    assert taken == ["curator_s0"]
    restored = tmp_path / "session2" / "curator_s0" / "checkpoints" / "round-0005" / "adapter" / "weights.bin"
    assert restored.read_bytes() == bytes([5]) * 64


# ------------------------------------------------------------------------- isolation of failures

class _Env:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.seen = []

    def evaluate_response(self, prompt, text):
        self.seen.append(prompt.meta.get("rollout_index"))
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError("verifier crashed")
        ok = text == "right"
        return Verdict(success=ok, score=float(ok), parse_ok=True, verifier_seconds=0.01)

    def compute_reward(self, verdict):
        return 1.0 if verdict.success else 0.0


def _batch(texts, g, pid="p1", env="gsm8k"):
    n = len(texts)
    return dict(prompts=[[{"role": "user", "content": "q"}]] * n, completions=texts, env_id=[env] * n,
                prompt_id=[pid if i < g else pid + "b" for i in range(n)], reference=["r"] * n,
                meta_json=["{}"] * n)


def test_router_scores_a_crashing_verifier_as_zero_and_counts_it():
    env = _Env(fail_on="boom")
    router = RewardRouter({"gsm8k": env}, group_size=4, workers=2)
    rewards = router(**_batch(["right", "boom", "wrong", "right"], 4))
    assert rewards == [1.0, 0.0, 0.0, 1.0]
    arm = router.pop_step().arms["gsm8k"]
    assert arm.verifier_errors == 1 and arm.k_success == 2 and arm.n_prompts == 1
    router.close()


def test_router_tags_each_rollout_with_its_index_in_the_group():
    env = _Env()
    router = RewardRouter({"gsm8k": env}, group_size=3, workers=1)
    router(**_batch(["a", "b", "c", "d", "e", "f"], 3))
    assert env.seen == [0, 1, 2, 0, 1, 2]
    router.close()


def _noisy(q=0.5):
    env = NoisyRewardEnv.__new__(NoisyRewardEnv)
    env.mode = "random"
    env.cfg = types.SimpleNamespace(q=q, flip_p=0.5)
    env._base = types.SimpleNamespace(evaluate_response=lambda p, r: Verdict(True, 1.0, True, 0.0))
    return env


def test_e24_identical_completions_get_independent_rewards_per_rollout():
    env = _noisy()
    rewards = []
    for prompt_idx in range(200):
        group = [env.compute_reward(env.evaluate_response(
            Prompt(f"p{prompt_idx}", "noisy", "train", (), "1", {"rollout_index": i}), "same answer"))
            for i in range(8)]
        rewards.append(group)
    mixed = sum(1 for g in rewards if 0 < sum(g) < 8)
    assert mixed > 150                       # identical text, yet rewards vary inside the group
    again = env.compute_reward(env.evaluate_response(
        Prompt("p0", "noisy", "train", (), "1", {"rollout_index": 3}), "same answer"))
    assert again == rewards[0][3]            # reproducible


def test_e24_callers_without_an_index_keep_the_old_key():
    env = _noisy()
    v = env.evaluate_response(Prompt("p0", "noisy", "train", (), "1", {}), "x")
    assert v.info["noise_key"] == "p0|x"


class _SuiteEnv:
    def __init__(self, n, tag):
        self.items = [Prompt(f"{tag}{k}", tag, "calib", (), str(k)) for k in range(n)]

    def split_size(self, split):
        return len(self.items)

    def prompt_at(self, split, i):
        return self.items[i]

    def evaluate_response(self, prompt, text):
        return Verdict(success=text == prompt.reference, score=0.0, parse_ok=True, verifier_seconds=0.0)


def test_eval_suite_skips_a_failing_domain_and_keeps_its_paired_state():
    envs = {"a": _SuiteEnv(6, "a"), "b": _SuiteEnv(6, "b")}
    suite = EvalSuite(envs, ["a", "b"], split="calib", items_per_domain=6, caps={})
    good = lambda items, cap: [p.reference for p in items]          # noqa: E731
    first = suite.evaluate({"a": 6, "b": 6}, good)
    assert first["score_by_domain"] == {"a": 1.0, "b": 1.0} and not first["errors"]

    def flaky(items, cap):
        if items[0].env_id == "b":
            raise RuntimeError("generation backend died")
        return [p.reference for p in items]

    second = suite.evaluate({"a": 6, "b": 6}, flaky, retries=1)
    assert set(second["score_by_domain"]) == {"a"} and "b" in second["errors"]
    assert suite.get_state()["last"]["b"] == [True] * 6      # untouched by the failed attempt
    third = suite.evaluate({"b": 6}, good)
    assert third["delta_se_by_domain"] is not None and "b" in third["delta_se_by_domain"]


# ------------------------------------------------------------------- checkpoint retention and results

class _Ctl:
    def __init__(self):
        self.round, self.global_step = 0, 0

    def get_state(self):
        return {"round": self.round}


def test_only_the_newest_three_checkpoints_are_kept(tmp_path):
    from curator_rl.trainer.checkpoint import latest_checkpoint, save_checkpoint

    ctl = _Ctl()
    (tmp_path / "checkpoints" / "round-0000.tmp").mkdir(parents=True)        # leftover of a killed session
    for r in range(1, 8):
        ctl.round = ctl.global_step = r
        save_checkpoint(tmp_path, ctl, lambda d: (d / "adapter.bin").write_bytes(b"x" * 100))
    kept = sorted(p.name for p in (tmp_path / "checkpoints").iterdir() if p.is_dir())
    assert kept == ["round-0005", "round-0006", "round-0007"]
    assert latest_checkpoint(tmp_path).name == "round-0007"
    ctl.round = 8
    save_checkpoint(tmp_path, ctl, lambda d: None, keep=1)
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()) == ["round-0008"]


def test_results_folder_and_zip(tmp_path):
    import zipfile

    from curator_rl.trainer.results import ZIP_NAME, export_results

    runs = tmp_path / "curator_runs"
    run = _fake_run(runs, "curator_s0", finished=True)
    (run / "progress.json").write_text(json.dumps({"status": "finished", "round": 5, "max_rounds": 5}))
    (runs / "comparison.md").write_text("# CURATOR GPU comparison\n")
    (runs / "plots").mkdir()
    (runs / "plots" / "curator_s0_mixture.png").write_bytes(b"\x89PNG")
    (runs / "logs").mkdir()
    (runs / "logs" / "curator_s0.log").write_text("training log")
    zip_path = export_results(runs, tmp_path / "results")
    res = tmp_path / "results"
    assert zip_path == res / ZIP_NAME and (res / "comparison.md").exists() and (res / "status.md").exists()
    assert (res / "runs" / "curator_s0" / "summary.json").exists()
    assert not list(res.rglob("*checkpoint*")) and not list(res.rglob("weights.bin"))
    names = zipfile.ZipFile(zip_path).namelist()
    assert "curator_results/runs/curator_s0/rounds.jsonl" in names and "curator_results/plots/curator_s0_mixture.png" in names
    assert not any(ZIP_NAME in n for n in names)                       # the zip does not contain itself
    assert "finished" in (res / "status.md").read_text()
    export_results(runs, res)                                          # repeatable; replaces the zip
    assert len(list(res.glob("*.zip"))) == 1 and not list(res.glob("*.tmp"))
