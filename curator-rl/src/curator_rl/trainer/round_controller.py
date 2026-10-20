"""The round loop of a real run: scheduler <-> GRPO trainer (Phase F, Roadmap G).

Backend-free by design: the TRL adapter and the fake CPU backend call the same four hooks, so
every line of scheduling, accounting and calibration logic is exercised by CPU tests.

    begin_round()             scheduler.select_mixture(previous observation) -> prompt stream
    fill_generation_batch()   called by the trainer right before generation: this step's prompts
                              are planned NOW, with the current mixture (no dataloader lag)
    reward router             verifies completions and accumulates per-arm statistics
    on_step_end(seconds)      cost attribution + ledger; every R steps: RoundObservation ->
                              scheduler, calibration every K rounds (methods that use it),
                              uncharged dev reports, checkpoint flag, budget stop

The order inside a round boundary is the simulator harness's (simulator/harness.py), so a
method's GPU behaviour is the behaviour its simulator gates measured.

Surviving a long run on borrowed hardware:
- `save()` checkpoints (retried once; three consecutive failures stop the run cleanly, since
  progress could no longer be protected) and hands the new checkpoint to `ArtifactSync`;
- `progress.json` is rewritten atomically every round: where the run is, what was last saved and
  synced, and why it stopped;
- a calibration or report evaluation that fails is logged and skipped; the training exposure of a
  skipped calibration carries over into the next window, so no evidence is lost;
- a wall-clock deadline (Kaggle's session limit) or an external stop request (SIGTERM) ends the
  run at the next round boundary with a fresh checkpoint: `preempted`, resumable as is.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from curator_rl.core.atomic import atomic_write_json
from curator_rl.core.jsonl import JsonlWriter
from curator_rl.core.types import CalibrationObservation, EnvRoundObs, RoundObservation
from curator_rl.cost.attribution import attribute_step
from curator_rl.cost.ledger import Ledger
from curator_rl.envs.base import Environment
from curator_rl.trainer.artifacts import ArtifactSync
from curator_rl.trainer.calibration_eval import EvalSuite, GenerateFn, SampledFn
from curator_rl.trainer.checkpoint import save_checkpoint
from curator_rl.trainer.prompt_stream import PromptStream
from curator_rl.trainer.reward_router import ArmStats, RewardRouter
from curator_rl.trainer.run_config import BENCH_DOMAINS, DOMAIN_OF, RunConfig
from curator_rl.trainer.schedulers import make_run_scheduler

LOG_FILES = ("steps", "rounds", "calibration", "reports")
EXIT_PREEMPTED = 75            # EX_TEMPFAIL: stopped cleanly before a limit; resume to continue
FAULT_ENV = "CURATOR_FAULT_AT_STEP"   # tests: raise once at this global step (a simulated crash)
SYNC_RESERVE_SECONDS = 300.0   # time kept free before the deadline for the last checkpoint upload
EVAL_PROTOCOL = 2              # reports: greedy + sampled accuracy with per-item results (D-101)


class RoundController:
    def __init__(self, run_cfg: RunConfig, root_cfg, envs: Mapping[str, Environment], *, out_dir: Path,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.cfg = run_cfg
        self.root = root_cfg
        self.envs = dict(envs)
        self.env_ids = tuple(sorted(run_cfg.envs))
        self.out_dir = Path(out_dir)
        self.clock = clock
        self.domain_of = {e: DOMAIN_OF.get(e, e) for e in self.env_ids}
        nominal = {e: float(self.envs[e].split_size("train") or 5000) for e in self.env_ids}
        self.scheduler = make_run_scheduler(
            run_cfg.method, root_cfg, self.env_ids, seed=run_cfg.seed,
            prompts_per_round=run_cfg.prompts_per_round, domain_of=self.domain_of, nominal_sizes=nominal)
        self.stream = PromptStream({e: self.envs[e] for e in self.env_ids}, seed=run_cfg.seed)
        self.router = RewardRouter(self.envs, group_size=run_cfg.group_size, workers=run_cfg.verify_workers,
                                   max_completion_tokens=run_cfg.max_completion_tokens, clock=clock)
        self.domains = tuple(sorted({d for d in self.domain_of.values() if d}))
        caps = {d: min(int(self.envs[d].cfg.max_completion_tokens), run_cfg.max_completion_tokens)
                for d in self.domains}
        self.calib_suite = EvalSuite(self.envs, self.domains, split="calib",
                                     items_per_domain=run_cfg.calib_items, caps=caps, workers=run_cfg.verify_workers)
        # the benchmark is fixed (every domain with a test set that has an environment), whatever is trained
        self.bench = tuple(d for d in BENCH_DOMAINS if d in self.envs)
        caps.update({d: min(int(self.envs[d].cfg.max_completion_tokens), run_cfg.max_completion_tokens)
                     for d in self.bench})
        self.report_suite = EvalSuite(self.envs, self.bench, split="dev",
                                      items_per_domain=max(run_cfg.report_items, run_cfg.final_items), caps=caps,
                                      workers=run_cfg.verify_workers)
        self.ledger = Ledger(run_cfg.budget_gpu_hours)
        self.generate: GenerateFn | None = None          # set by the backend (greedy decoding)
        self.generate_sampled: SampledFn | None = None   # set by the backend (k samples per item, one seed each)
        self.logprob_path = "n/a"                         # set by the TRL backend (triton | pytorch-fallback)
        self.round = 0
        self.global_step = 0
        self.step_in_round = 0
        self.calib_k = 0
        self.weights: dict[str, float] = {}
        self.prev_obs: RoundObservation | None = None
        self.stop = False
        self.checkpoint_due = False
        self._next_ready = False
        self._exposure = {e: 0.0 for e in self.env_ids}
        self._window_cost = 0.0
        self._reset_round()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logs = {name: JsonlWriter(self.out_dir / f"{name}.jsonl", schema_version=1) for name in LOG_FILES}
        self._log_lines = {name: 0 for name in LOG_FILES}
        # survival state (not checkpointed: it describes this process, not the experiment)
        self.sync: ArtifactSync | None = None             # set by the runner
        self.save_backend: Callable[[Path], None] | None = None
        self.preempted = False
        self.stop_reason = ""
        self._stop_requested = False
        self._checkpoint_failures = 0
        self.last_checkpoint_round: int | None = None
        self.eval_errors = 0
        self._round_wall_start = time.time()
        self._round_wall_seconds: float | None = None
        fault = os.environ.get(FAULT_ENV, "")
        self._fault_step = int(fault) if fault.strip().isdigit() else 0

    @property
    def uses_calibration(self) -> bool:
        return bool(getattr(self.scheduler, "uses_calibration", False)) or self.cfg.method == "dirichlet"

    # ---------------------------------------------------------------- the four hooks

    def begin_round(self) -> None:
        self.weights = self.scheduler.select_mixture(self.prev_obs)
        self.stream.set_weights(self.weights)
        self._next_ready = True

    def candidates_per_step(self, n_keep: int) -> int:
        """Questions to draw for a step that trains on n_keep (more with question filtering, D-102)."""
        from curator_rl.trainer.group_filter import candidates

        return candidates(n_keep, float(self.cfg.group_filter))

    def fill_generation_batch(self, rows: list[dict], group_size: int, n_prompts: int | None = None) -> list[dict]:
        """This step's prompts as rows, G consecutive rows per prompt (n_prompts, default one per G rows)."""
        if len(rows) % group_size:
            raise ValueError(f"generation batch of {len(rows)} rows is not a multiple of G={group_size}")
        prompts = self.stream.next_step(n_prompts or len(rows) // group_size)
        out = []
        for p in prompts:
            row = {"prompt": [dict(m) for m in p.messages], "env_id": p.env_id, "prompt_id": p.prompt_id,
                   "reference": p.reference, "meta_json": json.dumps(p.meta, default=str)}
            out.extend(dict(row) for _ in range(group_size))
        return out

    def on_step_end(self, step_seconds: float) -> None:
        rec = self.router.pop_step()
        tokens = {e: float(a.prompt_tokens + a.completion_tokens) for e, a in rec.arms.items()}
        verif = {e: a.verifier_seconds for e, a in rec.arms.items()}
        per_env = attribute_step(step_seconds, rec.score_wall_seconds, verif, tokens)
        self.ledger.charge("train", step_seconds)
        for e, arm in rec.arms.items():
            self._round_arms.setdefault(e, ArmStats()).merge(arm)
            self._round_seconds[e] = self._round_seconds.get(e, 0.0) + per_env.get(e, 0.0)
        self.global_step += 1
        self.step_in_round += 1
        marker = self.out_dir / ".fault_fired"
        if self._fault_step and self.global_step == self._fault_step and not marker.exists():
            marker.write_text(str(self.global_step), encoding="utf-8")
            raise RuntimeError(f"injected fault at step {self.global_step} ({FAULT_ENV})")
        self._write("steps", {"step": self.global_step, "round": self.round + 1, "seconds": step_seconds,
                              "score_wall_seconds": rec.score_wall_seconds,
                              "gen_update_seconds": max(step_seconds - rec.score_wall_seconds, 0.0),
                              "tokens": tokens, "prompts": {e: a.n_prompts for e, a in rec.arms.items()},
                              "charged_gpu_hours": self.ledger.charged_gpu_hours})
        if self.step_in_round >= self.cfg.steps_per_round:
            self._end_round()

    # ---------------------------------------------------------------- round boundary

    def _reset_round(self) -> None:
        self._round_arms: dict[str, ArmStats] = {}
        self._round_seconds: dict[str, float] = {}
        self.step_in_round = 0

    def _end_round(self) -> None:
        self._next_ready = False
        per_env, delivered = {}, {}
        total_prompts = sum(a.n_prompts for a in self._round_arms.values()) or 1
        for e in self.env_ids:
            a = self._round_arms.get(e, ArmStats())
            secs = self._round_seconds.get(e, 0.0)
            per_env[e] = EnvRoundObs(
                env_id=e, n_prompts=a.n_prompts, n_rollouts=a.n_rollouts, k_success=a.k_success,
                n_groups_mixed=a.n_groups_mixed, sum_score=a.sum_score, sum_score_sq=a.sum_score_sq,
                prompt_tokens=a.prompt_tokens, completion_tokens=a.completion_tokens,
                verifier_seconds=a.verifier_seconds, gpu_seconds=secs, cost_usd=secs / 3600.0,
                sum_abs_adv=a.sum_abs_adv)
            delivered[e] = a.n_prompts / total_prompts
            self._exposure[e] += a.n_prompts
        round_cost = sum(o.cost_usd for o in per_env.values())
        obs = RoundObservation(round=self.round + 1, steps=self.global_step, per_env=per_env,
                               weights_used=dict(self.weights), round_cost_usd=round_cost, overhead_usd=0.0,
                               budget_remaining_usd=self.ledger.remaining_gpu_hours)
        self.scheduler.update_observation(obs)
        self.prev_obs = obs
        self.round += 1
        self._window_cost += round_cost
        self._write("rounds", self._round_record(obs, delivered))
        self._reset_round()

        if self.uses_calibration and self.round % self._calib_interval() == 0:
            self._calibrate()
        if self.cfg.report_every and self.round % self.cfg.report_every == 0:
            self.report(self.cfg.report_items, tag="periodic")
        now = time.time()
        self._round_wall_seconds = now - self._round_wall_start
        self._round_wall_start = now
        if self.ledger.exhausted() or self.round >= self.cfg.max_rounds:
            self.stop = True
            self.stop_reason = "budget" if self.ledger.exhausted() else "max_rounds"
            self.write_progress("finishing")
            return
        self.begin_round()
        # checkpoint AFTER the next mixture is chosen: a resumed run continues with exactly it
        if self.cfg.checkpoint_every and self.round % self.cfg.checkpoint_every == 0:
            self.checkpoint_due = True
        reason = self._preempt_reason(now)
        if reason:
            # stop at this boundary with a checkpoint of exactly this state; the next session resumes it
            self.preempted, self.stop, self.stop_reason, self.checkpoint_due = True, True, reason, True
            print(f"[curator] stopping cleanly at round {self.round}: {reason}", flush=True)
        self.write_progress("preempting" if self.preempted else "running")

    # ---------------------------------------------------------------- survival

    def request_stop(self, reason: str = "stop requested (signal)") -> None:
        """Ask the run to stop at the next round boundary (signal handlers call this)."""
        self._stop_requested = reason or "stop requested"

    def _preempt_reason(self, now: float) -> str:
        if self._stop_requested:
            return str(self._stop_requested)
        if self.cfg.deadline_unix:
            per_round = self._round_wall_seconds or 0.0
            reserve = 1.5 * per_round + SYNC_RESERVE_SECONDS
            if now + reserve >= self.cfg.deadline_unix:
                left = (self.cfg.deadline_unix - now) / 60
                return f"session deadline in {left:.0f} min (one round takes {per_round / 60:.1f} min)"
        return ""

    def save(self, backend_save: Callable[[Path], None] | None = None) -> bool:
        """Checkpoint now (retried once), then start a background artifact sync. Never raises."""
        backend_save = backend_save or self.save_backend or (lambda d: None)
        for attempt in range(2):
            try:
                save_checkpoint(self.out_dir, self, backend_save, keep=self.cfg.keep_checkpoints)
                self._checkpoint_failures = 0
                self.last_checkpoint_round = self.round
                self.write_progress("preempting" if self.preempted else "running")
                if self.sync is not None:
                    self.sync.request(force=self.preempted, label=f"round {self.round}")
                return True
            except Exception as exc:  # noqa: BLE001 - e.g. disk full; retried, then counted
                print(f"[curator] checkpoint at round {self.round} failed (attempt {attempt + 1}): {exc!r}",
                      flush=True)
        self._checkpoint_failures += 1
        if self._checkpoint_failures >= 3:
            self.preempted, self.stop = True, True
            self.stop_reason = "three consecutive checkpoint failures: progress can no longer be protected"
            print(f"[curator] {self.stop_reason}; stopping", flush=True)
        return False

    def write_progress(self, status: str, **extra) -> None:
        """Atomically rewrite progress.json (a heartbeat; never raises)."""
        try:
            atomic_write_json(self.out_dir / "progress.json", {
                "status": status, "method": self.cfg.method, "label": self.cfg.name, "seed": self.cfg.seed,
                "round": self.round,
                "max_rounds": self.cfg.max_rounds, "global_step": self.global_step,
                "charged_gpu_hours": round(self.ledger.charged_gpu_hours, 5),
                "budget_gpu_hours": self.cfg.budget_gpu_hours,
                "last_checkpoint_round": self.last_checkpoint_round,
                "round_wall_seconds": self._round_wall_seconds, "stop_reason": self.stop_reason,
                "eval_errors": self.eval_errors, "verifier_errors": self.router.errors_seen,
                "sync": dict(self.sync.status) if self.sync is not None else None,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})
        except Exception as exc:  # noqa: BLE001
            print(f"[curator] progress.json not written: {exc!r}", flush=True)

    def _calib_interval(self) -> int:
        return self.cfg.calib_interval if self.cfg.method != "dirichlet" else max(self.cfg.calib_interval // 2, 1)

    def _round_record(self, obs: RoundObservation, delivered: Mapping[str, float]) -> dict:
        arms = {}
        for e, o in obs.per_env.items():
            a = self._round_arms.get(e, ArmStats())
            arms[e] = {"prompts": o.n_prompts, "pass_rate": (o.k_success / o.n_rollouts) if o.n_rollouts else None,
                       "true_pass_rate": (a.true_success / o.n_rollouts) if o.n_rollouts else None,
                       "mixed": (o.n_groups_mixed / o.n_prompts) if o.n_prompts else None,
                       "gpu_seconds": o.gpu_seconds, "tokens": o.prompt_tokens + o.completion_tokens,
                       "rejected": a.rejected, "truncated": a.truncated, "verifier_errors": a.verifier_errors}
        # statuses after this round's update, read from the signal engine's output (part of the
        # checkpointed state), not from the last MixtureDecision (which is not), so a resumed run
        # logs exactly what an uninterrupted one does
        vectors = getattr(self.scheduler, "_latest_vectors", None) or {}
        statuses = {e: sv.status.value for e, sv in vectors.items()} or None
        for e, sv in vectors.items():          # reward-audit dispersion (1 = reward ignores the prompt)
            if e in arms and sv.reward_dispersion == sv.reward_dispersion:
                arms[e]["dispersion"] = round(float(sv.reward_dispersion), 4)
                arms[e]["audit_flag"] = bool(sv.audit_flag)
        return {"round": obs.round, "step": obs.steps, "requested": dict(self.weights), "delivered": dict(delivered),
                "arms": arms, "statuses": statuses, "charged_gpu_hours": self.ledger.charged_gpu_hours}

    def _calibrate(self) -> None:
        if self.generate is None:
            raise RuntimeError("calibration needs a generate function (set by the backend)")
        request = ({d: self.cfg.calib_items for d in self.domains} if self.cfg.method == "dirichlet"
                   else cast(Any, self.scheduler).calibration_request())   # only calibrating methods get here
        if not request:
            return
        t0 = self.clock()
        try:
            res = self.calib_suite.evaluate(request, self.generate)
        except Exception as exc:  # noqa: BLE001 - e.g. generation backend hiccup; the run goes on
            res = {"score_by_domain": {}, "errors": {"*": repr(exc)[:300]}}
        secs = self.clock() - t0
        self.ledger.charge("calib", secs)           # spent compute is charged even when it failed
        if res.get("errors"):
            self.eval_errors += len(res["errors"])
            self._write("calibration", {"round": self.round, "request": dict(request), "seconds": secs,
                                        "errors": res["errors"], "skipped": not res["score_by_domain"]})
        if not res["score_by_domain"]:
            # nothing scored: skip this window; its exposure carries over into the next one
            return
        self.calib_k += 1
        scores = res["score_by_domain"]
        obs = CalibrationObservation(
            window_k=self.calib_k, round=self.round, score_total=sum(scores.values()) / len(scores),
            score_by_domain=scores, se_total=0.0, se_by_domain=res["se_by_domain"], n_items=res["n_items"],
            eval_cost_usd=secs / 3600.0, exposure_by_env=dict(self._exposure), window_cost_usd=self._window_cost,
            delta_se_by_domain=res["delta_se_by_domain"])
        if getattr(self.scheduler, "uses_calibration", False):
            self.scheduler.update_calibration(obs)
        report = getattr(self.scheduler, "last_calibration", None)
        self._write("calibration", {
            "window_k": self.calib_k, "round": self.round, "request": dict(request), "seconds": secs,
            "scores": scores, "se": res["se_by_domain"], "delta_se": res["delta_se_by_domain"],
            "exposure": dict(self._exposure), "window_cost_gpu_hours": self._window_cost,
            "z": getattr(report, "z", None), "mismatch": getattr(report, "mismatch", None),
            "gain_hat": getattr(report, "gain_hat", None), "ready": getattr(report, "ready", None),
            "proxy_weights": getattr(self.scheduler, "proxy_weights", None),
            "proxy_fit": (fit.as_log() if (fit := getattr(self.scheduler, "last_proxy_fit", None)) else None)})
        self._exposure = {e: 0.0 for e in self.env_ids}
        self._window_cost = 0.0

    def report(self, n_items: int, *, tag: str) -> dict:
        """Uncharged dev evaluation, identical for every method (score-vs-GPU-hours curve).

        Protocol 2 (D-101): greedy accuracy as before, plus sampled accuracy (k samples per item at
        the training temperature, one seed per item) and per-item results. Periodic reports use a
        fixed random subset of the final items, so curve and final score share their items.
        """
        if self.generate is None:
            return {}
        t0 = self.clock()
        k = int(self.cfg.eval_samples) if self.generate_sampled is not None else 0
        try:
            res = self.report_suite.evaluate_items({d: n_items for d in self.bench}, self.generate,
                                                   sampled=self.generate_sampled, k=k, spread=tag != "final",
                                                   retries=2 if tag == "final" else 1)
        except Exception as exc:  # noqa: BLE001
            res = {"score_by_domain": {}, "se_by_domain": {}, "errors": {"*": repr(exc)[:300]}}
        secs = self.clock() - t0
        self.ledger.charge("report", secs)
        scores = res["score_by_domain"]
        missing = [d for d in self.bench if d not in scores]
        self.eval_errors += len(res.get("errors") or {})
        # a macro over a subset of domains is not the benchmark: leave it empty and say why
        row = {"tag": tag, "round": self.round, "charged_gpu_hours": self.ledger.charged_gpu_hours,
               "scores": scores, "se": res.get("se_by_domain", {}), "n_items": n_items,
               "macro": (sum(scores.values()) / len(scores)) if scores and not missing else None,
               "missing": missing, "errors": res.get("errors") or {}, "seconds": secs,
               "protocol": EVAL_PROTOCOL, "subset": "first" if tag == "final" else "spread"}
        s_scores = res.get("sampled_by_domain") or {}
        if k:
            row["sampled"] = {"k": k, "temperature": self.cfg.eval_temperature, "scores": s_scores,
                              "se": res.get("sampled_se_by_domain") or {},
                              "macro": (sum(s_scores.values()) / len(s_scores))
                              if s_scores and all(d in s_scores for d in self.bench) else None}
        row["items"] = res.get("items") or {}
        self._write("reports", row)
        return row

    # ---------------------------------------------------------------- logs and state

    def _write(self, name: str, record: dict) -> None:
        self.logs[name].write(record)
        self._log_lines[name] += 1

    def get_state(self) -> dict:
        sched_state = self.scheduler.get_state()
        return {
            "schema": 1, "run_config": json.loads(self.cfg.to_json()),
            "round": self.round, "global_step": self.global_step, "calib_k": self.calib_k,
            "weights": dict(self.weights), "exposure": dict(self._exposure), "window_cost": self._window_cost,
            "scheduler": sched_state, "stream": self.stream.get_state(), "ledger": self.ledger.get_state(),
            "calib_suite": self.calib_suite.get_state(), "report_suite": self.report_suite.get_state(),
            "log_lines": dict(self._log_lines), "next_ready": self._next_ready,
        }

    def load_state(self, state: Mapping) -> None:
        if state.get("schema") != 1:
            raise ValueError(f"unknown controller checkpoint schema {state.get('schema')!r}")
        saved = RunConfig.from_dict(state["run_config"])
        for key in ("method", "seed", "steps_per_round", "prompts_per_step", "group_size", "envs"):
            if getattr(saved, key) != getattr(self.cfg, key):
                raise ValueError(f"checkpoint was made with {key}={getattr(saved, key)!r}, "
                                 f"this run has {getattr(self.cfg, key)!r}")
        self.round, self.global_step = int(state["round"]), int(state["global_step"])
        self.calib_k = int(state["calib_k"])
        self.weights = {e: float(v) for e, v in state["weights"].items()}
        self._exposure = {e: float(v) for e, v in state["exposure"].items()}
        self._window_cost = float(state["window_cost"])
        if state["scheduler"] is not None:
            self.scheduler.load_checkpoint(state["scheduler"])
        self.stream.load_state(state["stream"])
        self.ledger.load_state(state["ledger"])
        self.calib_suite.load_state(state["calib_suite"])
        self.report_suite.load_state(state["report_suite"])
        # stop is recomputed from THIS run's limits, so a finished run can be extended
        self.stop = self.ledger.exhausted() or self.round >= self.cfg.max_rounds
        self._next_ready = bool(state.get("next_ready", True))
        # drop log lines written after the checkpoint (they will be re-run). Not for a run that is
        # complete (summary written) and not being extended: it is not re-run, and those lines hold its
        # final evaluation
        if self.stop and (self.out_dir / "summary.json").exists():
            self._reset_round()
            return
        for name, n in state["log_lines"].items():
            path = self.out_dir / f"{name}.jsonl"
            self.logs[name].__exit__()
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)[: int(n)]
                path.write_text("".join(lines), encoding="utf-8")
            self.logs[name] = JsonlWriter(path, schema_version=1)
            self._log_lines[name] = int(n)
        self._reset_round()
        if not self.stop and not self._next_ready:
            self.begin_round()          # the run ended at a boundary before choosing the next mixture

    def summary(self, final: Mapping | None) -> dict:
        rounds = [json.loads(line) for line in (self.out_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        mean_share = {e: (sum(r["delivered"].get(e, 0.0) for r in rounds) / len(rounds) if rounds else None)
                      for e in self.env_ids}
        s5 = sorted({e for r in rounds for e, s in (r.get("statuses") or {}).items() if str(s).startswith("S5")})
        rejected = {e: sum(r["arms"][e]["rejected"] for r in rounds) for e in self.env_ids}
        verr = {e: sum(r["arms"][e].get("verifier_errors", 0) for r in rounds) for e in self.env_ids}
        return {"method": self.cfg.method, "label": self.cfg.name, "seed": self.cfg.seed, "envs": list(self.env_ids),
                "rounds": self.round, "steps": self.global_step,
                "charged_gpu_hours": self.ledger.charged_gpu_hours, "ledger_seconds": dict(self.ledger.seconds),
                "final": dict(final or {}), "mean_delivered_share": mean_share, "ever_s5": s5,
                "sandbox_rejections": rejected, "verifier_errors": verr, "eval_errors": self.eval_errors,
                "calibrations": self.calib_k, "logprob_path": self.logprob_path, "stop_reason": self.stop_reason,
                "run_config": json.loads(self.cfg.to_json())}

    def close(self) -> None:
        for w in self.logs.values():
            w.__exit__()
        self.router.close()


def observation_to_dict(obs: RoundObservation) -> dict:
    return asdict(obs)
