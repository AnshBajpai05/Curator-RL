# D-100 — Evidence by identifiability: audit for shared-slice arms, gain test for private ones

**Status: PROPOSED, NOT FROZEN.** Written 2026-10-11, after D-98's Stage T failed (D-99) and after a
tuning-seed diagnosis. Seeds 700–799 were never used, because D-98 Stage E was not run. This
proposal nevertheless uses seeds **800–899**, as D-98 §6 requires for any new idea. Nothing is
implemented or run until the owner freezes it with a dated entry in `docs/DECISIONS.md`.

## 1. Motivation (tuning seeds only)

`reports/analysis/audit_diagnosis.md`: with the noisy arm on the shared GSM8K slice, every GSM8K
false flag came from the gain-mismatch test, none from the audit. On a shared slice the gain test
cannot separate the two arms (D-97). Once the audit demotes the noisy arm, the gain test misfires on
GSM8K more often (4% → 10%). With gain flags off, the audit alone caught the noisy arm in 97% of
runs with no false flags. That last arm was an exploratory diagnostic on tuning seeds, not a
pre-registered candidate, so it motivates this hypothesis and proves nothing.

## 2. Rule (one candidate, fixed now; no Stage T)

Use each kind of evidence only where it is identifiable:
- **Arms on a private slice**: the gain-mismatch test, unchanged (D-73/D-95).
- **Arms on a shared slice**: the gain-mismatch test raises no S5 flag. Gains are still estimated
  with joint credit and logged. The reward-validity audit (D-98, frozen rule φ₀ = 0.3, z = 1.645,
  n_min = 100) decides S5 for these arms.

Configuration:
- new `calib.gain_flags_shared: true` (the default keeps current behaviour); the candidate sets it
  to false;
- `credit: joint`;
- `audit.enabled: true`, `action: s5`;
- `shared_slices: {noisy: gsm8k}`.

Everything else is the adopted configuration.

## 3. Experiment (fresh seeds 800–899, run once)

The same simulator, scenarios, calibration settings, arms and references as D-98 §5, with `S_cand`
being this rule. The private-slice arms are run too. In the private condition no slice is shared,
so the rule there equals gain + audit.

## 4. Criteria (identical to D-98 C1–C5; no threshold changes)

All must hold, in each of the three scenarios, shared-slice condition:
- **C1:** noisy-arm detection, Wilson 95% lower bound ≥ 0.70.
- **C2:** GSM8K flagged ≤ 5% and false-flag rate ≤ 5%.
- **C3:** detection of the candidate minus gain-only with the same credit (`S_gain_joint`); paired
  bootstrap lower bound > 0.
- **C4:** score of the candidate minus `S_gain_own`; lower bound > −0.01.
- **C5:** calibration + audit cost ≤ 10% of spend.

Intersection-union; 10,000 bootstrap draws; rng seed 10101.

Secondary (no decision attached):
- In the private-slice condition, the candidate's detection and false flags against `P_gain_own`.
  This checks that the rule leaves private-slice detection alone.
- Reasons for every S5 entry.
- Score against the references.

## 5. Interpretation (written now)

**Pass.** Spurious-reward detection holds in the realistic portfolio, by a declared division of
labour:
- gain evidence for arms whose slice identifies them;
- reward-validity evidence for arms that share a slice.

The paper states that the audit checks reward validity, not training benefit, and repeats the
simulator caveat of D-98 §1. The GPU use is still subject to the real-data check of D-98 §6.

**Fail.** D-97's narrower claim stands; no further variant is tried on this question in Level 1.

## 6. Implementation after the freeze

- `Calibrator`: suppress gain flags for arms in a slice with more than one arm when
  `gain_flags_shared` is false.
- Tests: the shared-slice arm is not gain-flagged; a private arm still is; defaults are unchanged.
- `experiments/audit_gate.py` for Stage E.
