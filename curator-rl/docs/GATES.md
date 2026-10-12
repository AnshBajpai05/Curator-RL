# GATES — evidence log

A gate passes only if every criterion is met (Roadmap v2.0 Part Q, v3 §6). Evidence is a committed file or a command that regenerates it. Criteria for a gate are committed **before** the gate is run (v3 §6.1).

Legend: ✅ passed · 🟡 passed with a documented shortfall · ⏳ pending · ⬜ not started

| Gate | Scope | State | Evidence |
|---|---|---|---|
| **0** Spec + scope frozen | D-1…D-19 | ⏳ owner sign-off pending | `docs/SPEC.md`, `docs/DECISIONS.md`. The tag `spec-v1` is **not** created until the owner confirms D-1…D-15 (single-owner confirmation: one dated entry in `DECISIONS.md`) |
| **1** Interface, 3 environments | Phase 2 | ✅ | `tests/contract/test_env_contract.py`; `data/manifests/MANIFEST.sha256`; `reports/env_profiles/phase2_profile.json` |
| **1B'** New environments (MATH, MBPP, too-hard) | v3 Phase E | ✅ portfolio rule passes on 0.5B (D-87): pass@8 gsm8k 0.80, math35 0.50, mbpp 0.48; too-hard = Countdown (0/1,024); cost spread 2.67x under vLLM (gen + pooled verify; update share pending Gate 5); `reports/pilot/phase_e_cost_probe.md` | `reports/env_profiles/phase_e_profile.json`, `tests/contract/test_env_contract_phase_e.py`, `tests/unit/test_{math35,sandbox}.py`, D-85. Gold 100% / garbage 0% on both; sandbox timeout and reward-hack tests pass; manifests reproduce byte-identically |
| **2** Simulator recovery (Curator v0) | v2 Phases 5–6 | 🟡 | `python experiments/gate2_check.py --seeds 50` → `reports/gate2/gate2.{json,md}`. Criteria a, b, c, d, e **all pass** on evaluation seeds 100–149 (re-run after the Phase C re-tune, D-68). Dev target (oracle-gap closure ≥ 75% in S-A) is **not** met (43%); analysed in D-65 (ceiling 75.9% with a 6-round warm-up in a 25-round episode) |
| **2''** Beats SEC/DUMP-style bandits; homogeneous twin | v3 Phase C | ❌ **fails (i) on S-I** before calibration; (ii), (iii), (iv) pass | `python experiments/gate2b_check.py --seeds 50` → `reports/gate2/gate2b.{json,md}`; diagnosis in `reports/analysis/curator_diagnosis.md`, D-69. Acceptance test for Phase D: re-run with the calibrated Curator (no privileged information) and pass (i) on S-I |
| **2''-CRE** Calibrated Reward Engine, pre-registered (D-76) | v3 Phase D | ❌ **fails (i) on S-I**; (ii), (iii) pass; CRE ≈ targeted Curator | `python experiments/cre_gate.py --seeds 50 --seed0 200` → `reports/gate2/cre_gate.{json,md}`; diagnosis `reports/analysis/cre_diagnosis.md`, D-78, D-79. Engine inert in S-A/B/C by construction |
| **2''-κ** H-κ: CRE with tuned κ/τ (D-80) | v3 Phase D | ❌ primary fails (S-I −0.010 vs SEC, −0.012 vs DUMP); CRE − control +0.001; regresses S-A and S-C | `python experiments/cre_kappa_gate.py --seeds 50 --seed0 300` → `reports/gate2/cre_kappa_gate.md`, D-80…D-82 |
| **2''-R** Gate 2'' on scenarios refit from measured data (D-88) | v3 Phase E | ✅ P1 passes in both cost brackets (Curator − SEC/DUMP-style +0.027…+0.029, CIs > 0); P2 passes (S-J-R +0.033). Caveats (D-89): LP and Standard UCB beat Curator by 0.011–0.014; calibration on vs off not significant; gain is not cost-driven | `python experiments/refit_gate.py --seeds 50 --seed0 400` → `reports/gate2/refit_gate.{json,md}`; refit `experiments/analysis/refit_scenarios.py` |
| **H-γ1** Curator without discounting (D-91) | v3 Phase E | ❌ not adopted: A1–A3 pass (+0.017…+0.034 over Curator; +0.048…+0.063 over SEC/DUMP-style), A4 no-regression fails narrowly on S-A and S-D (underpowered at 50 seeds) (D-92) | `python experiments/gamma_gate.py --seeds 50 --seed0 500` → `reports/gate2/gamma_gate.{json,md}`; diagnosis `reports/analysis/refit_diagnosis.md`, `gamma_robustness.md` (D-90) |
| **H-claim** Calibration targeting by proxy claim + staleness, 2 slices (D-94) | v3 Phase E | ✅ adopted: noisy-arm S5 detection 0.78–0.86 vs 0.25–0.32, false flags ≤ 0.025, calibration cost 3.4%, score +0.004…+0.006, Gate 2'' retained (D-95) | `python experiments/claim_gate.py --seeds 100 --seed0 600` → `reports/gate2/claim_gate.{json,md}`; tuning `reports/analysis/claim_targeting_tune.md` (D-93) |
| **3** Static GRPO works | v1 Phase 7 | ⬜ | — |
| **4** Curator does not alter GRPO | v1 Phase 8 | ⬜ | equivalence margins pre-registered in `EXPERIMENT_PROTOCOL.md` §3 |
| **5** Cost verified | v1 Phase 8 | ⬜ | — |
| **6** Calibration works | v1 Phase 9 | ❌ 6-sim not met | calibrator + targeted calibration built (D-73, D-75); `reports/analysis/calibration_power.md`: targeting detects the noisy arm in up to 87% of seeds under 10% cost but with 7.3% false flags; the rule's choice detects 33% / 10% |
| **7** ROI leaderboard | v1 Phase 10 | ⬜ | simulator part (7-sim) in v3 Phase D |
| **8** Baselines reproducible | v1 Phase 11 | ⬜ | — |
| **9** LOO works | v1 Phase 13 | ⬜ | simulator part (9-sim) in v3 Phase D |
| **10** Experiment Freeze (per level) | before every GPU matrix | ⬜ | `docs/ANALYSIS_PLAN.md`, config hashes |
| **M-1 / M-2** MVP milestones | v3 L1 | ⬜ | — |

## Phase-level exit log

| Phase | Tag | Date | Tests | Notes |
|---|---|---|---|---|
| 1 | — | 2026-10-02 | green | commit `35a8b6a` |
| 2 | — | 2026-10-02 | green | commit `35a8b6a` |
| 3 | — | 2026-10-03 | 119 | commit `35a8b6a` |
| 4 | — | 2026-10-05 | 183 | commit `2d3b40a` |
| 5 / Phase A | — (proposed `phase-5`) | 2026-10-09 | 217 | Gate 2 at 50 seeds; D-44…D-51 and D-63…D-67 logged; commit `9cec931` |
| C | — | 2026-10-09 | 248 | SEC/DUMP-style baselines, fair tuning, S-I/S-J, Gate 2'' run (fails (i) on S-I); D-68…D-71 logged; commit `9cec931` |
| D/E | — (proposed `phase-e3`) | 2026-10-10 | 349 | calibration, CRE, refit scenarios, claim targeting (D-72…D-95); commits `30052e8`, `bd9099e` |
| Audit + F | — | 2026-10-10 | 404 | audit fixes E-01…E-30, Phase F runner (CPU-verified), Kaggle package (D-96); commits `ff74171`…`54508c6` |
| F (survival) + research | — | 2026-10-12 | see README | run survival (sync, resume, supervised restarts), reward audit, alpha/beta fit, drift discounting, realism-v2 scenarios, ROI + LOO, figures, lock, typing (D-98, D-99); commits `b259f9c` onward |

Tags: none exist yet. Creating `spec-v1` needs the owner's sign-off of D-1…D-15 and D-52…D-62 first
(improvement E12); `phase-e3` would point at `bd9099e`.

## Gate 2 detail (evaluation seeds 100–149, tuned config of D-68, `status_control: hard`)

| Criterion | Result | Threshold |
|---|---|---|
| (a) S-A, Curator − Uniform, 95% paired bootstrap | +0.0618 [0.0506, 0.0730] | interval excludes 0 |
| (b) S-B, Curator − Uniform | +0.0110 [0.0069, 0.0149] | lower bound > −δ_gap = −0.01 (1 point; unit bug fixed in D-69) |
| (c) junk (too-hard) share, last 40% of rounds | mean 0.0125; 50/50 seeds ≤ 0.01875 | ≥ 90% of seeds ≤ 1.5 × floor |
| (d) status macro-F1 / pooled flip rate | 0.932 / 0.0 per 100 rounds | ≥ 0.85 / ≤ 5 |
| (e) S-C compute-to-target vs Uniform | +26.2% [22.2%, 30.0%] | ≥ 15% and interval excludes 0 |
| dev: oracle-gap closure, S-A | 43% | target ≥ 75% (not a pass/fail threshold) |

## Gate 2'' detail (evaluation seeds 100–149; S-J never tuned on)

| Criterion | Result | State |
|---|---|---|
| (i) Curator − SEC-style, S-C | +0.0611 [+0.0500, +0.0717] | pass |
| (i) Curator − DUMP-style, S-C | +0.0590 [+0.0478, +0.0700] | pass |
| (i) Curator − SEC-style, S-I | −0.0083 [−0.0130, −0.0036] | **fail** |
| (i) Curator − DUMP-style, S-I | −0.0077 [−0.0126, −0.0026] | **fail** |
| (ii) S-J, Curator − Uniform (H6 regime) | +0.0099 [+0.0048, +0.0149] vs tolerance −0.01 | pass |
| (iii) original Gate 2 (a)–(e) | all pass | pass |
| (iv) junk share | 50/50 seeds | pass |

Mean final scores (50 seeds):

| | Uniform | Static | LP | Std UCB | SEC-style | DUMP-style | Curator |
|---|---|---|---|---|---|---|---|
| S-A | 0.5786 | 0.6081 | 0.6096 | 0.6238 | 0.5782 | 0.5715 | **0.6404** |
| S-B | 0.4052 | 0.4178 | 0.4174 | 0.4210 | 0.4130 | 0.4129 | 0.4162 |
| S-C | 0.3416 | 0.3436 | 0.3464 | 0.3591 | 0.3420 | 0.3442 | **0.4031** |
| S-I | 0.3561 | 0.3516 | 0.3691 | **0.3720** | 0.3646 | 0.3640 | 0.3562 |
| S-J | 0.3552 | 0.3459 | 0.3716 | **0.3838** | 0.3673 | 0.3669 | 0.3651 |

Read with care:
- Curator's advantage is large where cost varies independently of value (S-C) and absent where the valuable arms are the expensive ones (S-I). The noisy arm absorbs 20–25% of Curator's compute before calibration; a privileged (perfect) S5 flag lifts Curator in S-I to +0.030–0.036 over Uniform on tuning seeds, i.e. to the level of the SEC/DUMP-style baselines (+0.020–0.031) but still below Standard UCB (+0.053) (`reports/analysis/curator_diagnosis.md`). That is an upper bound, not a result: the S-I criterion stays failed until Phase D passes it without privileged information.
- Standard UCB (cost-blind, no statuses) is the strongest baseline in S-I and S-J, which is why the paper's cost-aware claim must be carried by the real cost-heterogeneous portfolio and by calibration.
- Allocation is near one-hot for UCB and Curator in S-A and S-C (mean max-weight 0.84–0.89); SEC/DUMP-style stay close to uniform (0.17–0.47). See D-67.
