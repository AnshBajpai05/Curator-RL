"""Diagnosis of D-89 on TUNING seeds only (0-49): why does full Curator trail Standard UCB and LP
on the refit scenarios, and why does S5 rarely catch the noisy arm?

Standard UCB is Curator with four components switched off (gamma = 1, cost_exponent = 0,
status_control off, calibration off), so a full 2^4 factorial over those components, crossed with
the proxy's richness weight (beta 0.5 vs 0: alpha = 1, beta = 0 is "LP only"), attributes the gap.
Mechanism logs for every Curator-family run:
- per-arm mean of the signals after warm-up: LP z, richness SR, proxy x, mapped reward r_bar;
- calibration: how often each arm's slice is evaluated (targeting), the noisy arm's z-score at
  every ready window, the robust proxy scale, mismatch flags.

This is a diagnosis, not a selection. Nothing here changes `configs/base.yaml`; a candidate fix
must be pre-registered and evaluated on fresh seeds (>= 500), as D-89 requires.

    python experiments/analysis/refit_diagnosis.py --seeds 50

Writes reports/analysis/refit_diagnosis.{json,md}.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.simulator.harness import run_episode  # noqa: E402
from experiments import gate2_check  # noqa: E402
from experiments.refit_gate import SCENARIOS, scenario_with_calibration  # noqa: E402
from experiments.run_sim import (  # noqa: E402
    FROZEN_D68_D94,  # noqa: E402  (audit E-04: pre-registered config)
    make_scheduler,  # noqa: E402
)

SKIP = 8
NOISY = "noisy"
BOOT_SEED = 5051
FACTORS = ("gamma", "cost", "status", "calib", "richness")


def variant_name(v: dict) -> str:
    return (f"g{'.95' if v['gamma'] else '1'}-c{int(v['cost'])}-s{'H' if v['status'] else '0'}"
            f"-k{int(v['calib'])}-b{'.5' if v['richness'] else '0'}")


def variants() -> list[dict]:
    out = []
    for g, c, s, k, b in itertools.product((True, False), repeat=5):
        out.append({"gamma": g, "cost": c, "status": s, "calib": k, "richness": b})
    return out


def variant_cfg(v: dict):
    base = load_config(FROZEN_D68_D94)
    sched = base.scheduler.model_copy(update={
        "gamma": base.scheduler.gamma if v["gamma"] else 1.0,
        "cost_exponent": base.scheduler.cost_exponent if v["cost"] else 0.0,
        "status_control": base.scheduler.status_control if v["status"] else "off",
    })
    calib = base.calib.model_copy(update={"enabled": bool(v["calib"])})
    proxy = base.proxy if v["richness"] else base.proxy.model_copy(update={"alpha": 1.0, "beta": 0.0})
    return base.model_copy(update={"scheduler": sched, "calib": calib, "proxy": proxy})


class Recorder:
    """Delegates to a DiscountedUCB-family scheduler and logs signals and calibration internals."""

    def __init__(self, inner) -> None:
        self.inner, self.env_ids = inner, inner.env_ids
        self.uses_calibration = getattr(inner, "uses_calibration", False)
        self.signals: list[dict] = []
        self.requests: dict[str, int] = {e: 0 for e in self.env_ids}
        self.n_calib = 0
        self.noisy_z: list[float] = []
        self.scales: list[float] = []
        self.ever_s5: set[str] = set()

    def select_mixture(self, obs):
        w = self.inner.select_mixture(obs)
        for e in self.env_ids:
            if self.inner._engine.statuses(e).value.startswith("S5"):  # noqa: SLF001
                self.ever_s5.add(e)
        return w

    def update_observation(self, obs):
        self.inner.update_observation(obs)
        vec = self.inner._latest_vectors  # noqa: SLF001
        self.signals.append({e: (sv.lp_z, sv.richness, sv.proxy_raw, sv.proxy_reward) for e, sv in vec.items()})

    def calibration_request(self):
        req = self.inner.calibration_request()
        self.n_calib += 1
        for e in (req or {}):
            self.requests[e] += 1
        return req

    def update_calibration(self, obs):
        self.inner.update_calibration(obs)
        rep = getattr(self.inner, "last_calibration", None)
        if rep is not None and rep.ready:
            self.noisy_z.append(float(rep.z.get(NOISY, float("nan"))))
            self.scales.append(float(rep.proxy_scale))


def run_cell(task: tuple) -> dict:
    sid, name, v, seeds = task
    cfg = variant_cfg(v) if v is not None else load_config(FROZEN_D68_D94)
    sc = scenario_with_calibration(sid, cfg.calib.interval_rounds)
    ids = [e.env_id for e in sc.envs]
    method = "curator" if v is not None else name
    out = {"sid": sid, "name": name, "scores": [], "weights": {e: [] for e in ids}, "max_w": [],
           "signals": {e: [] for e in ids}, "requests": {e: [] for e in ids}, "n_calib": [],
           "noisy_z": [], "scales": [], "noisy_s5": []}
    for s in seeds:
        sched = make_scheduler(method, sc, s, cfg=cfg)
        rec = Recorder(sched) if v is not None else sched
        r = run_episode(rec, sc, seed=s, method=method)
        out["scores"].append(r.final_score)
        for e in ids:
            out["weights"][e].append(float(np.mean([lg["weights"][e] for lg in r.round_logs[SKIP:]])))
        out["max_w"].append(float(r.concentration(skip_rounds=SKIP)["mean_max_weight"]))
        if v is not None:
            for e in ids:
                rows = [row[e] for row in rec.signals[SKIP:] if e in row]
                out["signals"][e].append([float(np.mean([x[i] for x in rows])) for i in range(4)] if rows else None)
                out["requests"][e].append(rec.requests[e])
            out["n_calib"].append(rec.n_calib)
            out["noisy_z"].append(rec.noisy_z)
            out["scales"].append(rec.scales)
            out["noisy_s5"].append(NOISY in rec.ever_s5)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D-89 diagnosis on tuning seeds")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args(argv)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    assert args.seed0 + args.seeds <= 50, "diagnosis uses tuning seeds 0-49 only"
    vs = variants()
    tasks = [(sid, variant_name(v), v, seeds) for sid in SCENARIOS for v in vs]
    tasks += [(sid, m, None, seeds) for sid in SCENARIOS for m in ("uniform", "lp", "ucb", "sec", "dump")]
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(run_cell, tasks))
    res = {(c["sid"], c["name"]): c for c in cells}
    rng = np.random.default_rng(BOOT_SEED)
    full = variant_name(vs[0])                     # all components on = Curator as configured
    replica = variant_name({"gamma": False, "cost": False, "status": False, "calib": False, "richness": True})

    summary: dict = {"seeds": seeds, "full": full, "ucb_replica": replica, "scenarios": {}}
    for sid in SCENARIOS:
        base = res[(sid, full)]["scores"]
        rows = {}
        for (s2, name), c in res.items():
            if s2 != sid:
                continue
            m, lo, hi = gate2_check.paired_bootstrap([x - y for x, y in zip(c["scores"], base)], rng)
            rows[name] = {"mean": float(np.mean(c["scores"])), "minus_full": [m, lo, hi],
                          "weights": {e: float(np.mean(w)) for e, w in c["weights"].items()},
                          "max_w": float(np.mean(c["max_w"]))}
        # main effects over the 2^5 factorial (mean with factor on minus off)
        effects = {}
        for f in FACTORS:
            on = [np.mean(res[(sid, variant_name(v))]["scores"]) for v in vs if v[f]]
            off = [np.mean(res[(sid, variant_name(v))]["scores"]) for v in vs if not v[f]]
            effects[f] = float(np.mean(on) - np.mean(off))
        cf = res[(sid, full)]
        sig = {e: [float(np.nanmean([x[i] for x in cf["signals"][e] if x is not None])) for i in range(4)]
               for e in cf["signals"]}
        z_all = [z for zs in cf["noisy_z"] for z in zs if not np.isnan(z)]
        diag = {
            "signals_full": sig,
            "requests_per_episode": {e: float(np.mean(r)) for e, r in cf["requests"].items()},
            "calibrations_per_episode": float(np.mean(cf["n_calib"])),
            "noisy_z_median": float(np.median(z_all)) if z_all else None,
            "noisy_z_frac_below_minus2": float(np.mean([z < -2 for z in z_all])) if z_all else None,
            "noisy_ready_windows_per_episode": float(np.mean([len(zs) for zs in cf["noisy_z"]])),
            "proxy_scale_median": float(np.median([s for ss in cf["scales"] for s in ss])) if any(cf["scales"]) else None,
            "noisy_s5_rate": float(np.mean(cf["noisy_s5"])),
        }
        summary["scenarios"][sid] = {"rows": rows, "main_effects": effects, "mechanism": diag,
                                     "replica_vs_ucb": [float(np.mean(res[(sid, replica)]["scores"])),
                                                        float(np.mean(res[(sid, "ucb")]["scores"]))]}
    out = REPO_ROOT / "reports" / "analysis"
    (out / "refit_diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = render(summary)
    (out / "refit_diagnosis.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def render(s: dict) -> str:
    L = ["# Diagnosis of D-89 on tuning seeds (refit scenarios)", "",
         f"Tuning seeds {s['seeds'][0]}..{s['seeds'][-1]}. Variant code: g = gamma (.95 / 1), c = cost "
         "normalisation (1 on / 0 off), s = status layer (H hard / 0 off), k = calibration (1 / 0), "
         f"b = richness weight beta (.5 / 0, i.e. LP only). Full Curator = `{s['full']}`; "
         f"Standard UCB replica = `{s['ucb_replica']}`.", ""]
    for sid, d in s["scenarios"].items():
        rows = d["rows"]
        envs = list(next(iter(rows.values()))["weights"])
        L += [f"## {sid}", "",
              f"Replica check: `{s['ucb_replica']}` {d['replica_vs_ucb'][0]:.4f} vs Standard UCB "
              f"{d['replica_vs_ucb'][1]:.4f}.", "",
              "Main effects over the 2^5 factorial (mean score with the component on minus off): "
              + ", ".join(f"{k} {v:+.4f}" for k, v in d["main_effects"].items()) + ".", "",
              "| variant | score | minus full Curator [95% CI] | " + " | ".join(envs) + " | max-w |",
              "|---|---|---|" + "---|" * len(envs) + "---|"]
        for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean"])[:14]:
            m, lo, hi = r["minus_full"]
            L.append(f"| {name} | {r['mean']:.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     + " | ".join(f"{r['weights'][e]:.2f}" for e in envs) + f" | {r['max_w']:.2f} |")
        L.append("| ... | | | " + " | ".join("" for _ in envs) + " | |")
        for name in (s["full"],):
            r = rows[name]
            m, lo, hi = r["minus_full"]
            L.append(f"| **{name} (full)** | {r['mean']:.4f} | — | "
                     + " | ".join(f"{r['weights'][e]:.2f}" for e in envs) + f" | {r['max_w']:.2f} |")
        mech = d["mechanism"]
        L += ["", "Mechanism (full Curator): mean after warm-up of LP z / richness SR / proxy x / mapped reward r_bar:", "",
              "| arm | LP z | SR | x | r_bar | calibration evaluations per episode |", "|---|---|---|---|---|---|"]
        for e, (z, sr, x, rb) in mech["signals_full"].items():
            L.append(f"| {e} | {z:+.2f} | {sr:.3f} | {x:.3f} | {rb:.3f} | {mech['requests_per_episode'][e]:.2f} |")
        L += ["", f"- calibrations per episode {mech['calibrations_per_episode']:.1f}; ready windows "
              f"{mech['noisy_ready_windows_per_episode']:.1f}; noisy z median {mech['noisy_z_median']}; share of ready "
              f"windows with noisy z < -2: {mech['noisy_z_frac_below_minus2']}; proxy scale median "
              f"{mech['proxy_scale_median']}; noisy reached S5 in {mech['noisy_s5_rate']:.2f} of episodes.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
