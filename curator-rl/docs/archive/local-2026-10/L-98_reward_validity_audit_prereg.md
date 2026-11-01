# D-98 — Reward-validity audit: design and pre-registration

**Status: FROZEN 2026-10-11** on the owner's instruction ("suggest the best approach … and proceed
accordingly"), after a pre-freeze review that added three amendments: the repeated-looks operating
characteristics (§4), the intersection-union statement (§5), and the Phase G real-data check (§6).
The freeze is dated in `docs/DECISIONS.md`. Nothing in sections 3–6 may change from here on. The
document was written after D-97 and before any audit code or audit result; the review used only
the statistics of the rule (`experiments/analysis/audit_operating_characteristics.py`), no data.

## 1. Problem and scope

On real hardware the noisy arm trains on GSM8K prompts, so it can only be calibrated on the GSM8K
slice. With that shared slice, gain-based S5 detection collapses: 0.78 → 0.07, with GSM8K wrongly
flagged in 11% of runs; joint credit reaches only 0.15 (D-97, tuning seeds). The gain test asks
"did training on this arm move a held-out benchmark?", and it cannot separate two arms whose budget
shares move together.

The audit asks a different question, and only that question: **is this arm's training reward
informative about held-out correctness?** It does not measure, and is never reported as measuring,
whether training on the arm improves the model. Reward-validity evidence and benchmark-gain evidence
are kept in separate fields, logged with separate reasons, and analysed separately.

**What it needs, and therefore where it applies.**
- (i) A calibration slice with a gold verifier. The GSM8K calibration split has gold answers.
- (ii) Answers already generated on that slice by the calibration evaluation.
- (iii) The arm's own reward function, runnable on the slice's prompts and answers.

For the Level 1 portfolio all three hold. The noisy arm's reward function
(`NoisyRewardEnv.compute_reward ∘ evaluate_response`) accepts GSM8K prompts and is a deterministic
function of (prompt id, answer). No hidden simulator quantity is used: the audit reads only the
arm's reward output and the slice verifier's output, which the deployed system also has. **The audit
is therefore not simulator-only for this portfolio.** Arms whose domain has no gold verifier cannot
be audited, and the paper must say so.

**Simulator caveat.** In the simulator, a valid arm's reward on its own slice equals the slice
verifier by construction (φ = 1), and the noisy reward is independent of correctness by
construction (φ ≈ 0). The simulator comparison therefore tests the mechanism's plumbing, timing,
cost, false-flag behaviour and interaction with allocation. Its discrimination power comes from the
design of the noise model, and the paper must say that too. Section 4 gives the rule's operating
characteristics for partially valid rewards (label flips), computed separately with no training.

## 2. Mechanism (opt-in; defaults unchanged)

**Data.**
- At each calibration window, for every evaluated slice d and every arm j that maps to d (its own
  slice, or a shared one through `calib.shared_slices`), the calibration runner emits audit pairs:
  (key, r_j, v_d) for each slice item.
  - r_j: the arm's binary reward on the item's answer.
  - v_d: the slice verifier's binary verdict on the same answer.
  - key: a hash of (item id, answer). Repeated identical greedy answers produce the same key and
    are counted once.
- No answer is generated for the audit. The audit only reads answers the calibration evaluation
  already produced.
- Simulator: an answer's identity is (item, correctness, latent version). It changes only when the
  item's correctness changes or its latent is redrawn by churn. This is conservative: real greedy
  text changes more often than real correctness.

**Statistic.**
- Per arm, a 2×2 table pooled over unique keys since the start of the run.
- φ = Pearson correlation of r and v over the pooled pairs.
- Chance level: φ = 0, the value under independence of reward and correctness.
- One-sided 95% upper bound: U = tanh(atanh(φ) + 1.645 / √(n − 3)), the Fisher z interval.

**Decision** (per arm, recomputed at every calibration window):
- **inconclusive** if n < 100 pooled unique pairs, or any margin has fewer than 5 items in a class
  (reward or verdict almost constant). This also covers missing or unavailable verifier outputs.
- **invalid** if U < φ₀ = 0.3.
- **valid** otherwise.

**Effect** (`calib.audit.action`):
- `report`: logged only.
- `s5`: an *invalid* arm enters S5 with the note `S5 reward invalid (audit)`, and leaves S5 when it
  is no longer invalid.
- The gain-mismatch path (D-73/D-95) is unchanged and keeps its own note and counters.
- S5's response (halving the arm's weight) is unchanged.

**Cost.**
- Running reward functions on existing answers is charged to the method that uses the audit, under
  a new ledger category `calib_audit` (charged).
- Real runs charge measured CPU time.
- In the simulator, the cost per audited pair = 0.01 × the calibration cost per item. The measured
  verifier time per GSM8K answer is about 0.4% of the generation time per answer (Phase E cost
  probe), so this is about 2.5× conservative.

**Configuration** (new; all defaults leave current behaviour bit-for-bit unchanged):

```yaml
calib:
  audit:
    enabled: false        # opt-in
    action: s5            # s5 | report
    phi0: 0.3
    z: 1.645
    min_pairs: 100
    min_class: 5
  gain_flags: true        # false = gain-mismatch flags off (only for the audit-only analysis arm)
```

## 3. Hypotheses

- **H-A (primary).** With the noisy arm sharing the GSM8K slice, Curator with the audit enabled
  detects the noisy arm (S5) in most runs, without raising false flags on the arms that learn.
- **H-B (incremental value).** The audit adds detection beyond gain-based calibration alone, with
  the same credit rule.
- **H-C (no harm).** Enabling the audit does not lower the final score by more than 0.01.

## 4. Threshold choice (made before any experiment, from the rule's statistics alone)

`experiments/analysis/audit_operating_characteristics.py` gives P(flag "invalid") with a base
correctness of 0.4 and 4,000 Monte Carlo draws. The rows are pooled pairs n; the columns are reward
quality (random means independent of correctness; flip ε means the verifier's verdict is flipped
with probability ε).

| φ₀, n | random | flip 0.0 | flip 0.1 | flip 0.2 | flip 0.3 | flip 0.4 |
|---|---|---|---|---|---|---|
| 0.2, 100 | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| **0.3, 100** | **0.92** | 0.00 | 0.00 | 0.00 | 0.00 | 0.29 |
| 0.3, 200 | 0.995 | 0.00 | 0.00 | 0.00 | 0.00 | 0.47 |

**Rule used to fix φ₀ and n_min.** Take the smallest φ₀ that reaches ≥ 0.90 power against an
independent reward at n = 100 while flagging ≤ 1% of rewards with up to 30% label flips. That gives
φ₀ = 0.3 and n_min = 100. A reward with 40% flips (φ ≈ 0.2) is close to uninformative and may be
flagged; this is a stated property of the rule, not a false positive.

**Amendment (pre-freeze review): repeated looks.** The decision is recomputed at every calibration
window on growing pooled data, so the relevant false-positive rate is the probability of being
flagged at *any* look, not at one. Monte Carlo with 6 looks, +20 pairs per look and 3,000 draws gives
P(flag) at the first look / at any look:

| z (one-sided) | random | flip 0.0–0.2 | flip 0.3 | flip 0.4 |
|---|---|---|---|---|
| **1.645** | **0.913 / 0.996** | 0.000 / 0.000 | **0.003 / 0.006** | 0.289 / 0.570 |
| 2.326 | 0.757 / 0.983 | 0.000 / 0.000 | 0.001 / 0.002 | 0.115 / 0.296 |
| 2.576 | 0.675 / 0.972 | 0.000 / 0.000 | 0.000 / 0.000 | 0.065 / 0.222 |

The rule above, now applied to the repeated-look rate, still holds at z = 1.645: a reward with up to
30% flips is ever flagged in 0.6% of runs. z, φ₀ and n_min are therefore unchanged. A reward with
40% flips is flagged at some look in 57% of runs; the paper states this as the rule's resolution
limit.

## 5. Experiment

**Simulator.**
- Refit scenarios S-I-R1, S-I-R2, S-J-R with calibration as in D-88/D-95 (paired items, churn 0.02,
  100 items per slice, K = 10, claim_stale targeting, 2 slices, charged per item).
- The adopted Curator configuration everywhere else; γ = 0.95.

**Stage T: candidate selection (tuning seeds 0–49 only).**
- Candidates, both with the noisy arm sharing the GSM8K slice, audit on, `action: s5`:
  - `cand_own`: own-slice credit;
  - `cand_joint`: joint credit.
- **Eligible** if, averaged over the three scenarios: GSM8K flagged ≤ 0.05 AND false-flag rate
  ≤ 0.05 AND calibration + audit cost ≤ 10% of spend.
- **Choose** the eligible candidate with the highest noisy-arm detection. Within 0.02, choose
  `cand_joint` (fewer wrong flags expected, D-97).
- **If neither is eligible**, stop. No Stage E is run, and D-97's narrower claim stands.
- φ₀, n_min, z, the cost model and S5's response are **not** tuned.

**Stage E: confirmation (fresh seeds 700–799, 100 seeds, run once).**

| Arm | Slice | Credit | Gain flags | Audit |
|---|---|---|---|---|
| `S_gain_own` (today's real system) | shared | own | on | off |
| `S_gain_joint` | shared | joint | on | off |
| `S_audit_only` | shared | candidate's | **off** | on |
| `S_cand` (**the candidate**) | shared | candidate's | on | on |
| `P_gain_own` (D-95 adopted, idealised) | private | own | on | off |
| `P_cand` | private | candidate's | on | on |

References (Uniform, Standard UCB, LP, SEC-style, DUMP-style) are run for context only.

**Primary criteria.** All must hold, in each of the three scenarios, for `S_cand`:
- **C1 detection:** the noisy arm reaches S5 (any reason) in a share of runs whose Wilson 95%
  lower bound is ≥ 0.70.
- **C2 false flags:** GSM8K reaches S5 in ≤ 5% of runs, AND the mean over non-noisy arms of the
  share of runs in S5 is ≤ 5% (point estimates; Wilson upper bounds reported).
- **C3 incremental value:** detection of `S_cand` minus detection of the gain-only arm with the
  same credit (`S_gain_own` or `S_gain_joint`), paired over seeds; 95% bootstrap lower bound > 0
  (10,000 draws, rng seed 9091).
- **C4 no harm:** final score of `S_cand` minus `S_gain_own`; 95% bootstrap lower bound > −0.01.
- **C5 cost:** calibration + audit cost ≤ 10% of spend.

**Multiplicity (amendment).** The candidate passes only if every criterion holds in every scenario,
which is an intersection-union test. That is conservative, so no further correction is applied. The
secondary analyses carry no decision and are not corrected.

**Secondary, reported only (no decision attached).**
- Detection attributed to each reason (gain vs audit).
- The audit-only arm, showing how much the audit does alone.
- Private-slice arms, i.e. whether the audit costs anything where gain detection already works.
- Rounds to first detection.
- Noisy-arm budget share.
- Pooled pairs per arm.
- The share of windows where the audit is inconclusive.
- Score against the references.

## 6. Decision and interpretation (written before results)

**If C1–C5 all pass.**
- Spurious-reward detection is restored for an arm sharing its slice, through **reward-validity
  evidence**.
- The claim in the paper becomes: "an online audit of each arm's reward against held-out verified
  answers, using only answers already generated for calibration, flags rewards uninformative about
  correctness". It is not "calibration measures that the arm does not help".
- The audit stays **disabled by default**. A GPU calibration run (Phase G) enables it explicitly in
  its own pre-registration, together with `shared_slices: {noisy: gsm8k}`.

**If any criterion fails.**
- D-97's narrower claim stands: detection holds only for spurious arms with a private slice. The
  audit result is reported as a negative result with all numbers.
- Nothing is retuned on seeds 700–799. A new idea needs a new pre-registration and seeds ≥ 800.

**Either way**, the simulator caveat in section 1 is stated next to the result.

**Real-data check before any GPU use (amendment).** The simulator cannot test discrimination on real
answers. Before a Phase G run relies on the audit, compute φ for every arm on the base model's
calibration answers, which the calibration baseline generates anyway, before any training. Expected:
the noisy arm is invalid or φ ≈ 0, and the arms whose reward is the slice verifier have φ = 1. Any
other outcome stops the GPU use of the audit and is reported. This check uses no extra generation.

## Outcome (appended after the run; sections 1-6 unchanged)

Stage T, tuning seeds 0-49: no eligible candidate (cand_own: GSM8K flagged 0.167, false flags 0.133;
cand_joint: 0.100, 0.055; detection 0.98-0.99, cost 3.2%). Stage E was not run. D-97's narrower
claim stands. Details and diagnosis: D-99. Follow-up proposal: D-100.

## 7. Implementation plan (after the freeze; no comparison is run before it)

| File | Change |
|---|---|
| `calibration/audit.py` (new) | `RewardAudit`: pooled unique-pair tables, the decision rule above, state for checkpoints |
| `core/types.py` | `CalibrationObservation.audit_pairs: dict[arm, list[(key, reward, verdict)]] | None = None` (optional; absent = no audit) |
| `core/config.py`, `configs/base.yaml` | `calib.audit` block and `calib.gain_flags`, defaults as in section 2 |
| `calibration/calibrator.py` | consumes `audit_pairs` when enabled; `CalibrationReport.reward_audit` kept separate from gain fields; `gain_flags=false` suppresses only gain mismatch flags |
| `signals/status.py` | `set_reward_invalid(flag)`: S5 entry/exit by audit, separate note and counter; gain path untouched |
| `scheduler/ducb.py` | forwards audit verdicts; checkpoints the audit state |
| `simulator/world.py`, `scenarios.py` | audit pairs for evaluated slices (noisy reward = deterministic hash keyed by answer identity, Bernoulli(q); other arms = slice verdict); `audit_cost_per_pair_usd` |
| `simulator/harness.py` | charges audit cost only to methods with the audit enabled |
| `cost/ledger.py` | charged category `calib_audit` |
| `trainer/audit_pairs.py` (new) | real-environment helper: applies each arm's reward function and the slice verifier to existing calibration answers, with timing for the ledger |
| `experiments/analysis/audit_tune.py`, `experiments/audit_gate.py` | Stage T and Stage E exactly as in section 5 |

**Tests**, preserving all existing ones:
- a shared slice audited under its slice name;
- a noisy reward flagged as invalid;
- a valid reward never flagged;
- missing or degenerate verifier outputs → inconclusive;
- duplicate answers counted once;
- checkpoint/resume of the audit state (scheduler replay test extended);
- audit cost charged only when enabled, under `calib_audit`;
- defaults reproduce the D-93/D-95 numbers bit for bit;
- the Phase F runner and its ledger are unchanged with the audit disabled.
