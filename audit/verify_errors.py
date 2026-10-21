"""Reproduce the verified errors listed in errors_found.md (tuning seeds only; writes nothing).

Run from the repository root, with the project venv:

    curator-rl/.venv/Scripts/python audit/verify_errors.py curator-rl            # all checks
    curator-rl/.venv/Scripts/python audit/verify_errors.py curator-rl lp stale   # selected checks

Checks: lp, stale, math, sandbox, jsonl, flip, countdown, gsm8k.
"""
import statistics
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402

CFG = load_config(REPO / "configs/base.yaml")
which = set(sys.argv[2:]) or {"lp", "stale", "math", "sandbox", "jsonl", "flip", "countdown", "gsm8k"}


# ---------------------------------------------------------------- 1. LP-B z calibration
if "lp" in which:
    from curator_rl.signals.engine import SignalEngine
    from curator_rl.simulator.scenarios import build_world, load_scenario

    sc = load_scenario(REPO / "configs/sim/scenario_si_r1.yaml")
    ids = [e.env_id for e in sc.envs]
    zs, slopes, ses = [], [], []
    for seed in range(0, 200):
        world = build_world(sc, seed)
        eng = SignalEngine(CFG.signals, CFG.proxy, CFG.calib, 1.0, sc.group_size, ids)
        for t in range(30):
            obs = world.step_round({e: 1 / len(ids) for e in ids})
            v = eng.update(obs)
        sv = v["noisy"]                       # true null: reward Bernoulli(0.35), never learns
        zs.append(sv.lp_z)
        slopes.append(sv.lp)
        ses.append(sv.lp_se)
    print("1) LP-B on a true-null arm (noisy, uniform weights, round 30, 200 seeds)")
    print(f"   SD of z (should be ~1 if SE is right): {statistics.stdev(zs):.3f}")
    print(f"   empirical SD of slope {statistics.stdev(slopes):.5f} vs mean reported SE {statistics.fmean(ses):.5f}"
          f"  -> SE overstated x{statistics.fmean(ses) / statistics.stdev(slopes):.2f}")
    print(f"   share |z| > 1.5: {np.mean(np.abs(zs) > 1.5):.3f} (nominal two-sided 0.134)")

# ---------------------------------------------------------------- 2. stale calibration flags
if "stale" in which:
    from experiments.refit_gate import scenario_with_calibration
    from experiments.run_sim import make_scheduler

    from curator_rl.simulator.harness import run_episode

    class Spy:
        def __init__(self, inner):
            self.inner, self.env_ids = inner, inner.env_ids
            self.uses_calibration = inner.uses_calibration
            self.windows = []          # (evaluated set, ready, mismatch dict)
            self.s5_round = {}

        def select_mixture(self, obs):
            w = self.inner.select_mixture(obs)
            for e in self.env_ids:
                if self.inner._engine.statuses(e).value.startswith("S5") and e not in self.s5_round:
                    self.s5_round[e] = len(self.windows)
            return w

        def update_observation(self, obs):
            self.inner.update_observation(obs)

        def calibration_request(self):
            return self.inner.calibration_request()

        def update_calibration(self, obs):
            self.inner.update_calibration(obs)
            r = self.inner.last_calibration
            self.windows.append((set(r.evaluated), r.ready, dict(r.mismatch)))

    sc = scenario_with_calibration("S-I-R1", CFG.calib.interval_rounds)
    entries = single_evidence = 0
    flags_without_new_eval = flags_total = 0
    for seed in range(50):
        spy = Spy(make_scheduler("curator", sc, seed, cfg=CFG))
        run_episode(spy, sc, seed=seed, method="curator")
        for env in spy.env_ids:
            for evald, ready, mis in spy.windows:
                if ready and mis.get(env):
                    flags_total += 1
                    flags_without_new_eval += env not in evald
        if "noisy" in spy.s5_round:
            entries += 1
            k = spy.s5_round["noisy"]           # windows seen before S5 was first observed
            # the two flags that caused entry are the last two ready windows before k
            ready_idx = [i for i in range(k) if spy.windows[i][1]][-2:]
            if len(ready_idx) == 2 and "noisy" not in spy.windows[ready_idx[1]][0]:
                single_evidence += 1
    print("2) Calibration flags re-issued without new evidence (S-I-R1, adopted claim_stale/2, seeds 0-49)")
    print(f"   mismatch flags on arms NOT evaluated in that window: {flags_without_new_eval}/{flags_total}")
    print(f"   noisy S5 entries: {entries}/50; entries whose 2nd flag had no new evaluation: {single_evidence}")

# ---------------------------------------------------------------- 3. MATH35 checker
if "math" in which:
    from curator_rl.envs.math35 import answers_equal
    cases = [
        ("50.5", "10\\frac{1}{2}", False, "wrong pred matches mixed-number gold (10 1/2 read as 101/2)"),
        ("\\frac{21}{2}", "10\\frac{1}{2}", True, "correct improper fraction vs mixed-number gold"),
        ("23", "2(3)", False, "implicit product 2(3)=6 read as 23"),
        ("\\frac{5}{2}", "2\\frac{1}{2}", True, "correct 5/2 vs gold 2 1/2"),
    ]
    print("3) MATH35 answers_equal")
    for pred, gold, should, note in cases:
        got = answers_equal(pred, gold)
        flag = "OK " if got == should else "BUG"
        print(f"   {flag} pred={pred!r:16} gold={gold!r:18} -> {got} (correct answer: {should}) [{note}]")

# ---------------------------------------------------------------- 4. sandbox bypasses
if "sandbox" in which:
    from curator_rl.envs.sandbox import run_python_tests
    tests = ["assert add(2, 3) == 5", "assert add(10, 20) == 30"]
    forge = (
        "import re as _r\n"
        "print(_r.search(r\"SANDBOX_OK_[0-9a-f]+\", open(__file__).read()).group(0), flush=True)\n"
        "import os as _o; _o._exit(0)\n"
        "def add(a, b):\n    return 0\n"
    )
    always_eq = (
        "class _Any:\n    def __eq__(self, other):\n        return True\n"
        "def add(a, b):\n    return _Any()\n"
    )
    honest_wrong = "def add(a, b):\n    return a - b\n"
    print("4) MBPP sandbox (should all be False)")
    for name, code in (("wrong solution", honest_wrong), ("reads sentinel from its own source", forge),
                       ("always-equal object", always_eq)):
        res = run_python_tests(code, tests, timeout_s=10)
        print(f"   {'BUG' if res.passed and name != 'wrong solution' else 'OK '} {name}: passed={res.passed}")

# ---------------------------------------------------------------- 5. JSONL crash recovery
if "jsonl" in which:
    from curator_rl.core.jsonl import JsonlWriter, read_jsonl
    d = Path(tempfile.mkdtemp())
    p = d / "log.jsonl"
    with JsonlWriter(p, 1) as w:
        w.write({"round": 1})
        w.write({"round": 2})
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"_schema": 1, "_seq": 3, "rou')        # crash mid-write: truncated, no newline
    w = JsonlWriter(p, 1)                                    # resume after crash
    for r in (4, 5, 6):
        w.write({"round": r})
    w.__exit__()
    skipped = []
    rows = list(read_jsonl(p, on_skip=skipped.append))
    print("5) JSONL append after a crash-truncated line")
    print(f"   records readable: {[r['round'] for r in rows]} (written after resume: [4, 5, 6]); skipped lines: {skipped}")

# ---------------------------------------------------------------- 6. flip_rate
if "flip" in which:
    from curator_rl.core.types import EnvStatus as S
    from curator_rl.signals.status import flip_rate
    trace = [S.S1, S.S2, S.S2, S.S3]
    got = flip_rate(trace, warmup=2)
    print("6) status.flip_rate([S1,S2,S2,S3], warmup=2)")
    print(f"   returned {got:.0f} per 100 rounds; tail [S2,S3] has 1 transition in 1 interval -> expected 100")

# ---------------------------------------------------------------- 7. Countdown division rule
if "countdown" in which:
    from curator_rl.envs.countdown import _BIN_OPS, _draw_instance

    def int_reachable(numbers):
        memo = {}
        def rec(mask):
            if mask in memo:
                return memo[mask]
            if mask & (mask - 1) == 0:
                memo[mask] = {numbers[mask.bit_length() - 1]}
                return memo[mask]
            out = set()
            sub = (mask - 1) & mask
            while sub:
                for a in rec(sub):
                    for b in rec(mask ^ sub):
                        out.update((a + b, a - b, a * b))
                        if b != 0 and a % b == 0:
                            out.add(a // b)
                sub = (sub - 1) & mask
            memo[mask] = out
            return out
        return rec((1 << len(numbers)) - 1)

    rng = np.random.default_rng(0)
    cfg = CFG.data.envs.countdown
    n, bad = 300, 0
    for _ in range(n):
        _, numbers, target, _ = _draw_instance(cfg, "train", rng)
        if target not in int_reachable(tuple(numbers)):
            bad += 1
    print("7) Countdown: targets reachable ONLY through non-exact division (prompt forbids it)")
    print(f"   {bad}/{n} instances ({bad / n:.1%})")

# ---------------------------------------------------------------- 8. GSM8K formats
if "gsm8k" in which:
    from curator_rl.envs.gsm8k import _verify_boxed
    print("8) GSM8K _verify_boxed on correct answers in common formats (gold 18 / 1000)")
    for resp, gold in (("\\boxed{18}", "18"), ("\\boxed{\\$18}", "18"), ("\\boxed{18 \\text{ dollars}}", "18"),
                       ("\\boxed{1{,}000}", "1000"), ("\\boxed{1,000}", "1000")):
        boxed, ok = _verify_boxed(resp, gold)
        print(f"   {'OK ' if ok else 'MISS'} {resp!r:30} -> extracted {boxed!r}, correct={ok}")
