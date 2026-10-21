"""Full-data profile of the MATH35 and MBPP environments (Roadmap D.5, Gate 1B' evidence).

Writes reports/env_profiles/phase_e_profile.json:
- split sizes and sealed counts;
- gold verification rate over ALL rows of every split (MBPP: the official reference code; MATH35:
  the gold answer boxed) and the failing ids;
- garbage success rate (must be ~0);
- mean verifier seconds on gold (the MBPP sandbox start-up cost is the cost driver);
- exact-text overlap between MATH35 training splits and the sealed MATH-500 items.

Usage: python scripts/env_profile_phase_e.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.envs.math35 import Math35Env  # noqa: E402
from curator_rl.envs.mbpp import MbppEnv  # noqa: E402

GARBAGE = ["", "I do not know.", "\\boxed{}", "\\boxed{banana}", "```python\npass\n```", "```python\nprint('hi')\n```"]


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    math_env = Math35Env(REPO_ROOT, cfg.data.envs.math35)
    mbpp = MbppEnv(REPO_ROOT, cfg.data.envs.mbpp)
    out = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    rows = {s: math_env._items[s] for s in ("train", "calib", "dev")}  # noqa: SLF001
    ok = tot = 0
    times, fails, junk_ok, junk_tot = [], [], 0, 0
    for s, rs in rows.items():
        for r in rs:
            p = math_env._to_prompt(r, s)  # noqa: SLF001
            v = math_env.evaluate_response(p, f"So the answer is \\boxed{{{r['answer']}}}.")
            ok += v.success
            tot += 1
            times.append(v.verifier_seconds)
            if not v.success:
                fails.append(r["problem_id"])
    for r in rows["dev"][:100]:
        p = math_env._to_prompt(r, "dev")  # noqa: SLF001
        for g in GARBAGE:
            junk_ok += math_env.evaluate_response(p, g).success
            junk_tot += 1
    sealed = [json.loads(x) for x in (REPO_ROOT / "data" / "test_sealed" / "math35.jsonl").read_text(encoding="utf-8").splitlines()]
    train_text = {r["problem"].strip() for rs in rows.values() for r in rs}
    out["math35"] = {
        "split_sizes": {s: len(v) for s, v in rows.items()}, "sealed": len(sealed),
        "levels": {str(lv): sum(r["level"] == lv for rs in rows.values() for r in rs) for lv in (3, 4, 5)},
        "gold_rate": round(ok / tot, 4), "gold_failures": fails[:20], "n_gold_failures": len(fails),
        "garbage_success_rate": round(junk_ok / junk_tot, 4), "verifier_mean_seconds_gold": round(sum(times) / len(times), 6),
        "exact_text_overlap_with_sealed": [r["problem_id"] for r in sealed if r["problem"].strip() in train_text],
    }

    ok = tot = 0
    times, fails, junk_ok, junk_tot = [], [], 0, 0
    for s in ("train", "dev", "calib"):
        for r in mbpp._items[s]:  # noqa: SLF001
            p = mbpp._to_prompt(r, s)  # noqa: SLF001
            v = mbpp.evaluate_response(p, f"```python\n{r['code']}\n```")
            ok += v.success
            tot += 1
            times.append(v.verifier_seconds)
            if not v.success:
                fails.append(r["problem_id"])
    p0 = mbpp._to_prompt(mbpp._items["dev"][0], "dev")  # noqa: SLF001
    for g in GARBAGE:
        junk_ok += mbpp.evaluate_response(p0, g).success
        junk_tot += 1
    out["mbpp"] = {
        "split_sizes": {s: len(mbpp._items[s]) for s in ("train", "dev", "calib")}, "sealed": 500,  # noqa: SLF001
        "gold_rate": round(ok / tot, 4), "gold_failures": fails, "n_gold_failures": len(fails),
        "garbage_success_rate": round(junk_ok / junk_tot, 4),
        "verifier_mean_seconds_gold": round(sum(times) / len(times), 4),
        "verifier_p95_seconds_gold": round(sorted(times)[int(0.95 * len(times))], 4),
    }
    dest = REPO_ROOT / "reports" / "env_profiles" / "phase_e_profile.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
