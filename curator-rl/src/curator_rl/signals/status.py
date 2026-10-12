"""S1-S5 status classification (Roadmap E.5, resolves S-1 / S-3).

One `StatusClassifier` per environment. Priority when classifying from
scratch: S1 > S5 > S3 > S4 > S2. Entry into S3/S4 requires the condition to
hold for `consecutive_rounds` (h) rounds; S1 exits immediately when both
minimums are met (an established S3/S4 arm is never demoted to S1 by evidence
decay, D-51); S5 enters immediately on the mismatch streak reaching q
windows and leaves after q' clean windows. Dwell (D_min completed rounds in
a status) gates the exit from S3 and S4; leaving S1 and entering S5 are
exempt (roadmap E.5); interpreted choices are logged as D-36/D-37.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from curator_rl.core.config import StatusCfg
from curator_rl.core.types import EnvStatus


class StatusState(TypedDict):
    status: str
    rounds_in_status: int
    s3_run: int
    s4_run: int
    hot: int
    clear: int
    audit: NotRequired[bool]            # absent in checkpoints made before the reward audit
    s5_from_audit: NotRequired[bool]


def flip_rate(trace: list[EnvStatus], *, warmup: int = 0) -> float:
    """Transitions per 100 rounds over `trace`, ignoring the first `warmup` entries.

    The denominator is the number of observed intervals (len - 1), matching
    'transitions per round, scaled to 100'.
    """
    tail = trace[warmup:]
    if len(tail) <= 1:
        return 0.0
    # count transitions inside the tail only (audit E-21: the whole trace was counted)
    transitions = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    return transitions * 100.0 / (len(tail) - 1)


class StatusClassifier:
    """Per-environment status state machine (Roadmap E.5)."""

    def __init__(self, cfg: StatusCfg, *, mismatch_windows: int, clear_windows: int) -> None:
        self.status = EnvStatus.S1
        self.status_note = "S1 exploring"
        self._cfg = cfg
        self._dwell = max(cfg.dwell_min, 1)
        self._rounds_in_status = 0
        self._s3_run = 0
        self._s4_run = 0
        self._hot = 0
        self._clear = 0
        self._audit = False            # reward audit: the reward does not depend on the prompt
        self._s5_from_audit = False    # the current S5 was entered by the audit, not by calibration
        self._q_mis = mismatch_windows
        self._q_clear = clear_windows
        self._n_min = float(cfg.n_min)
        # anti-flap: re-entering S1 needs evidence to drop BELOW the ratio;
        # leaving needs it back at n_min (set to 1.0 ratio => symmetric band)
        self._s1_floor = self._n_min * max(min(cfg.s1_entry_ratio, 1.0), 0.0)

    # -------------------------------------------------------------- calibration

    def set_mismatch(self, flag: bool) -> None:
        """Engine-forwarded calibration flag (one call per calibration window)."""
        if flag:
            self._hot += 1
            self._clear = 0
        else:
            self._hot = 0
            self._clear += 1

    def set_audit(self, flag: bool) -> None:
        """Engine-forwarded reward-audit verdict (signals/audit.py), once per round."""
        self._audit = bool(flag)

    # -------------------------------------------------------------------- step

    def step(
        self,
        n_groups_eff: float,
        rounds_seen: int,
        p_hat: float,
        p_lo: float,
        p_hi: float,
        z_lp: float,
        sr: float,
    ) -> tuple[EnvStatus, str]:
        """Classify this round against the current state and return (status, note)."""
        cfg = self._cfg
        if self.status == EnvStatus.S1:
            s1_needed = (n_groups_eff < self._n_min) or (rounds_seen < cfg.r_min)
        else:
            s1_needed = (n_groups_eff < self._s1_floor) or (rounds_seen < cfg.r_min)

        s3_cond = p_lo >= cfg.p_sat and z_lp < cfg.z_up
        self._s3_run = self._s3_run + 1 if s3_cond else 0
        s4_cond = p_hi <= cfg.p_hard and sr <= cfg.sr_hard and z_lp < cfg.z_up
        self._s4_run = self._s4_run + 1 if s4_cond else 0

        current = self.status
        target = self._target(current, s1_needed, p_hat, p_lo, p_hi, z_lp, sr)
        note = self._note(target, z_lp, sr)

        if target != current:
            self.status = target
            self._rounds_in_status = 0
        self.status_note = note
        self._rounds_in_status += 1
        return target, note

    # ---------------------------------------------------------------- internal

    def _target(
        self,
        current: EnvStatus,
        s1_needed: bool,
        p_hat: float,
        p_lo: float,
        p_hi: float,
        z_lp: float,
        sr: float,
    ) -> EnvStatus:
        cfg = self._cfg

        # 0. a reward shown not to depend on the prompt is unreliable whatever else holds. It takes
        # priority over S1: the audit has its own evidence requirement (min_df), and exploring an arm
        # whose reward is noise wastes exactly the budget S1 would spend.
        if self._audit:
            self._s5_from_audit = True
            return EnvStatus.S5
        if current == EnvStatus.S5 and self._s5_from_audit and self._hot < self._q_mis:
            current = EnvStatus.S2          # audit cleared: classify afresh (no calibration streak to wait for)
        self._s5_from_audit = False

        # 1. S1 keeps its E.5 priority over S5 for fresh arms, but evidence
        # decay alone does NOT demote an established S3/S4 arm to S1 (D-51): a
        # starved too-hard arm must not cycle S1 -> quota -> S4.
        established = current in (EnvStatus.S3, EnvStatus.S4)
        if s1_needed and not established:
            return EnvStatus.S1

        # 2. S5 mismatch gates (enter exempt from anywhere; leaves after q' clean windows)
        if self._hot >= self._q_mis:
            return EnvStatus.S5
        if current == EnvStatus.S5 and self._clear < self._q_clear:
            return EnvStatus.S5

        # 3. incumbency of S3 / S4: leave only via the hysteresis rule + dwell.
        # Once the leave rule fires (with dwell satisfied), classification
        # falls through to the fresh rules below -- a forgetting spike leaves
        # S3 to S2 even while p_lo is still high.
        if current == EnvStatus.S3:
            leave = p_hat < cfg.p_sat - cfg.hysteresis_p or z_lp <= -cfg.z_neg
            if not leave or self._rounds_in_status < self._dwell:
                return EnvStatus.S3
            return self._from_s3_exit(s4_run_ok=self._s4_run >= cfg.consecutive_rounds)
        if current == EnvStatus.S4:
            leave = (
                p_hat >= cfg.p_hard + cfg.hysteresis_p
                or z_lp >= cfg.z_up
                or sr > cfg.sr_hard + cfg.hysteresis_sr
            )
            if not leave or self._rounds_in_status < self._dwell:
                return EnvStatus.S4
            return self._from_s4_exit()

        # 4. fresh classification (entries need h consecutive rounds)
        if self._s3_run >= cfg.consecutive_rounds:
            return EnvStatus.S3
        if self._s4_run >= cfg.consecutive_rounds:
            return EnvStatus.S4
        return EnvStatus.S2

    @staticmethod
    def _from_s3_exit(*, s4_run_ok: bool) -> EnvStatus:
        # exit into S2 by default; S4 only if the too-hard entry rule is mid-run
        return EnvStatus.S4 if s4_run_ok else EnvStatus.S2

    @staticmethod
    def _from_s4_exit() -> EnvStatus:
        return EnvStatus.S2

    @staticmethod
    def _note(status: EnvStatus, z_lp: float, sr: float) -> str:
        if status == EnvStatus.S2:
            return "S2a_progressing" if z_lp > 0 else "S2b_plateau"
        if status == EnvStatus.S5:
            return "S5 proxy-benchmark mismatch"
        if status == EnvStatus.S3:
            return "S3 saturated"
        if status == EnvStatus.S4:
            return f"S4 too hard (sr={sr:.2f})"
        if status == EnvStatus.S1:
            return "S1 exploring"
        return status.value

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> StatusState:
        return {
            "status": self.status.value,
            "rounds_in_status": self._rounds_in_status,
            "s3_run": self._s3_run,
            "s4_run": self._s4_run,
            "hot": self._hot,
            "clear": self._clear,
            "audit": self._audit,
            "s5_from_audit": self._s5_from_audit,
        }

    def load_checkpoint(self, state: StatusState) -> None:
        self.status = EnvStatus(str(state["status"]))
        self._rounds_in_status = int(state["rounds_in_status"])
        self._s3_run = int(state["s3_run"])
        self._s4_run = int(state["s4_run"])
        self._hot = int(state["hot"])
        self._clear = int(state["clear"])
        self._audit = bool(state.get("audit", False))  # absent in pre-audit checkpoints
        self._s5_from_audit = bool(state.get("s5_from_audit", False))
