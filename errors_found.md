# CURATOR — errors found (audit of completed phases and of the plan for the rest)

Audit date: 2026-10-10, against commit `bd9099e`. Every source module in `curator-rl/src`, the gate and analysis scripts, the Kaggle scripts and the planning documents were read.

**How the errors were verified.** Each error marked **verified** was reproduced by running the project's own code, on tuning seeds only, so no evaluation seed range was spent. The scripts are in `audit/`:

```
curator-rl/.venv/Scripts/python audit/verify_errors.py curator-rl                # E-01, E-02, E-06, E-07, E-08, E-13, E-19, E-21
curator-rl/.venv/Scripts/python audit/measure_noise_and_quotas.py curator-rl     # E-09, E-10
```

The remaining errors are verified by reading the code at the cited line, or by arithmetic on documented numbers.

## Fix status (branch `audit-fixes-and-gpu-run`, see `CHANGES_AND_IMPACT.md`)

| Fixed, with a regression test | Fixed, design-level | Open |
|---|---|---|
| E-01 (config `signals.lp_se`), E-02 (`calib.flag_scope`), E-04 (`configs/frozen/`), E-06, E-07, E-08, E-09 (`CarryQuotas`, `pull_unit: delivered`, world `learning_from`), E-10 (`final_score_expected`), E-11 (`core/stats.py`), E-13, E-16 (`dump.window_rounds`), E-17, E-18, E-19, E-21, E-24 (rollout index in the noisy key), E-25, E-28 (`rescale_for_batch_shape`), E-31, E-32 | E-03 (`calib.credit: transfer` + shared slice), E-05 (GPU-shape scenarios), E-12 (README_KAGGLE budget), E-14 (`train_items` + `repeat_decay` in realism-v2 scenarios; MBPP passes reported per episode), E-15 (GPU-shape calibration cost), E-20 (stats module; design note), E-22, E-23, E-26 | — |

Legacy behaviour stays available behind each new switch, so earlier decisions remain reproducible (proved by re-running the D-94 gate exactly).

**E-28 (found after this audit, while fixing E-05): status evidence thresholds do not transfer across batch shapes.** `signals.status.n_min = 64` prompt groups was tuned at 80 prompts per round. At the GPU plan's 32, an arm's discounted evidence settles at about 60, so 77% of arm-rounds stayed in S1 and its exploration quota pinned Curator near Uniform. Fixed by `core.config.rescale_for_batch_shape` (used by the GPU runner and the GPU-shape studies). Impact: +0.0060 [+0.0043, +0.0078] over the pre-audit Curator; after fair re-tuning, +6.0% over Uniform instead of +0.8%.

**E-29 (found when opening the PR): CI had never run.** The workflow sat in `curator-rl/.github/workflows/`; GitHub only executes `.github/workflows/` at the repository root. Moved there with `working-directory: curator-rl`.

**E-30 (found by that first CI run): split manifests verified only on Windows.** `write_manifests` wrote in text mode, so on Windows every line ended in CRLF and `MANIFEST.sha256` hashed CRLF bytes; on Linux (CI, Kaggle) the check failed. Manifests and index are now written with LF, pinned by `.gitattributes`, and the index is regenerated (split contents unchanged). CI now passes on Python 3.11 and 3.12.

**E-31 (found while building leave-one-out): re-launching a run whose training had finished overwrote its result.** `run_single` skipped training when the checkpoint said the run was complete, so no backend set a generate function; the final evaluation returned nothing and an empty "finished" summary replaced the real one. This is exactly what a lost session during the final evaluation would trigger. Fixed: a complete run with a summary is left untouched; a complete run without one loads the saved policy for evaluation only. Regression test in `tests/integration/test_trainer_fake.py`.

**E-32: Countdown accepted `True` as the number 1.** `isinstance(True, int)` holds in Python, so the literal check let a boolean consume the number 1. Harmless in effect (same value), wrong in principle; booleans are now rejected (found by the pyright pass, improvement E8).

**Correction to E-05 below:** η must NOT be rescaled. The world's update is R·η per round over R·P prompts, i.e. η/P per prompt, independent of R. The GPU-shape scenarios keep η.

**Severity:**
- **Critical** — changes an adopted decision or a headline claim.
- **High** — corrupts rewards or measurements in the GPU phases, or makes a planned test invalid.
- **Medium** — wrong, but limited impact or in code not yet used for results.
- **Low** — inconsistencies, dead configuration, documentation.

Related: `improvement_required.md` (improvements, not bugs). Where an error has a matching improvement item, the ID is given.

---

## Summary

| ID | Severity | Phase | Error | Verified |
|---|---|---|---|---|
| E-01 | Critical | 4 | Learning-progress (LP-B) standard error is ~3.7× too large; z-scores are deflated | run: null-arm z SD = 0.281 (should be 1) |
| E-02 | Critical | D/E | Calibration re-issues mismatch flags for arms not re-evaluated; "2 consecutive windows" is not 2 pieces of evidence | run: 100/157 flags on unevaluated arms; 26/40 noisy S5 entries |
| E-03 | Critical | plan G | The real noisy arm has no calibration slice, so S5 can never fire on it on GPU | code + README |
| E-04 | Critical | C–E | Pre-registered gate scripts read the mutable `base.yaml`; re-running them no longer tests what was registered | run: claim gate control == candidate |
| E-05 | Critical | E / plan F | Refit simulator uses 80 prompts per round and about 5,600 per run; the GPU plan is 32 per round and 1,920 per run | config + docs |
| E-06 | High | E | MATH checker accepts wrong answers and rejects correct ones (mixed numbers, implicit products) | run: 4/4 cases wrong |
| E-07 | High | E | MBPP sandbox can be passed without solving the task (2 bypasses) | run: both pass |
| E-08 | High | 2 | GSM8K checker rejects correct answers in common formats | run: 3/5 formats missed |
| E-09 | High | 3/5 | Requested weights ≠ delivered prompts; bookkeeping and simulated learning use requested, cost uses delivered | run: 0.031 delivered vs 0.020 requested |
| E-10 | High | 3 | Simulator's final score has independent evaluation noise per method: 68% of paired variance | run |
| E-11 | High | plan H | Planned statistics (percentile bootstrap over 3–5 seeds) gives intervals that are far too narrow | run: 82.8% coverage at n = 5, 74.9% at n = 3 |
| E-12 | High | plan | Compute plan still assumes 1 GPU-hour per run; measured 1.4–2.8 | arithmetic |
| E-13 | Medium | 1 | JSONL log: after a crash-truncated line, every later record is unreadable | run |
| E-14 | Medium | E | MBPP (314 train items) treated as inexhaustible; simulated allocations imply 4.7–8.2 passes over it | arithmetic |
| E-15 | Medium | D | Simulated calibration charged at full training cost per item; likely ~4–5× too high | arithmetic |
| E-16 | Medium | C | DUMP-style baseline uses an all-time mean, not a recent window; handicapped under drift | code |
| E-17 | Medium | 3 | Trace replayer emits zero mixed groups, wrong score sums, prompt counts that do not sum to M | code |
| E-18 | Medium | 2 / plan F | "Without replacement within a pass" (SPEC §8) not implemented; dead state in `Gsm8kEnv` | code |
| E-19 | Medium | 2 | Countdown prompt forbids inexact division; generator and verifier allow it | run: 1.3% of instances |
| E-20 | Medium | plan F | Gate 4 equivalence (±1 point, 3 seeds) is unattainable unless run-to-run SD < 0.40 points | arithmetic |
| E-21 | Low | 4 | `status.flip_rate` counts transitions over the whole trace but divides by the tail | run: 200 vs 100 |
| E-22 | Low | 5 | Gate 2 oracle-gap closure divides a 20-seed oracle mean by 50-seed method means | code |
| E-23 | Low | 1/4 | Dead or mis-described config keys: `calib.target`, `calib.trust_region`, `warmup.total_rounds`; comments for `s3/s4/s5`, `pull_unit` | code |
| E-24 | Low | 2 | Noisy-arm rewards of identical completions are identical, not independent | code |
| E-25 | Low | 2 | Sealed-test guard fails open on a corrupt log line; read-modify-write is unsafe across processes | code |
| E-26 | Low | B/E | Cost probe counts the noisy arm as "learnable" (harmless here) | code |
| E-27 | Low | all | Stale documents (README status, COST_MODEL header, GATES log, docstrings) | see `improvement_required.md` E11 |

---

## Critical

### E-01. LP-B standard error is ~3.7× too large, so every learning-progress z-score is deflated

**Where:** `curator-rl/src/curator_rl/signals/progress.py:118-130` (`_slope_lp`) and `:54` (`observe`).

**What is wrong**
- `observe` stores the **raw** per-round pass rate (`k_success / n_rollouts` of that round alone, passed in by `signals/engine.py:100-101`).
- `_slope_lp` stage 1 fits with weights `w_i = n_prompts_i` and computes
  ```
  resid_var = Σ w_i · r_i² / (k − 2)
  ```
  With weights equal to prompt counts, this is the variance of **one prompt group**, σ². The variance of a round's raw rate is σ²/n_i.
- Stage 2 then uses `eff_var_i = se_i² + resid_var`, i.e. the per-*prompt* variance, as the variance of each per-*round* point. The slope SE is therefore inflated by about √n_i: √16 = 4 for an arm with 16 prompts per round.

**History.** D-37 designed the inflation for a *pooled, discounted posterior*, whose rounds are strongly autocorrelated. The code now fits raw, independent per-round rates, but the SE was not revisited. The class docstring still says "slope of the per-round posterior".

**Evidence** (`audit/verify_errors.py lp`; noisy arm, whose true learning rate is exactly zero, uniform weights, round 30, 200 seeds):
```
SD of z (should be ~1 if SE is right): 0.281
empirical SD of slope 0.00506 vs mean reported SE 0.01883  -> SE overstated x3.72
share |z| > 1.5: 0.000 (nominal two-sided 0.134)
```

**What it affects**
- **Status machine** (`signals/status.py`). Every rule that tests `z_LP` against `z_up = 1.5` or `−z_neg = −2` almost never fires:
  - S3 and S4 entry guards (`z < 1.5`) are almost always true;
  - S3 exit by forgetting (`z ≤ −2`), S4 exit by progress (`z ≥ 1.5`) and the S2a/S2b label effectively never trigger.
- **D-40 (choice of LP estimator).**
  - LP-B was selected because its false-positive rate at |z| ≥ 2 (`experiments/analysis/common.py:38`) was 0.000. That zero is produced by this bug: the effective threshold was |z| ≥ 7.4.
  - LP-A, which had the best rank correlation (ρ 0.23–0.27 vs 0.19–0.20), was excluded on FPR. The selection must be re-run.
- **D-42 (status thresholds).** `z_up = 1.5` was tuned against deflated z-scores.
- **D-90 finding 3** ("learning progress is barely detectable, mean z 0.06–0.15"). Mostly an artefact of this bug, not only of noise.
- **Not affected directly:** the proxy reward. It uses the slope `lp / σ_LP`, not `z`. The slope estimate stays unbiased; stage 2 merely weights rounds equally instead of by prompt count.

**Fix**
- In `_slope_lp`, treat `resid_var` as the per-prompt variance σ² and use per-round variance `v_i = σ² / n_i`. Drop the pooled-posterior `se_group` term, which describes a different quantity.
- Equivalently: WLS with weights `n_i`, `SE(slope) = sqrt(σ² / Σ n_i (x_i − x̄_w)²)`.
- Correct the docstring and D-37.

**Retest**
1. New unit test `tests/unit/test_progress.py::test_lp_b_z_is_calibrated_on_null`: on 2,000 simulated flat sequences with known per-prompt variance and varying `n_i`, SD(z) ∈ [0.9, 1.1].
2. Re-run `experiments/analysis/lp_estimator_study.py` (D-40) and `tune_status_thresholds.py` (D-42) on tuning seeds.
3. Then one pre-registered evaluation on fresh seeds (≥ 700), together with E-02 and E-05.

### E-02. Calibration counts the same evidence twice

**Where:** `curator-rl/src/curator_rl/calibration/calibrator.py:174-180` and `curator-rl/src/curator_rl/scheduler/ducb.py:214-216`.

**What is wrong**
- `Calibrator.observe` computes `z` and `mismatch` for **every** arm with past evidence, whether or not its slice was evaluated in this window.
- `DiscountedUCB.update_calibration` forwards a flag for every arm on every ready window.
- With targeted calibration (2 slices per window since D-95), an arm that is not re-evaluated keeps its old gain estimate, so its flag simply repeats. The S5 rule "2 consecutive mismatch windows" (`calib.mismatch_windows: 2`) was meant to require replication. In practice one bad evaluation can produce both flags.
- The same mechanism keeps an arm locked in S5 until it is re-evaluated.

**Evidence** (`audit/verify_errors.py stale`; S-I-R1, adopted `claim_stale/2`, seeds 0–49):
```
mismatch flags on arms NOT evaluated in that window: 100/157
noisy S5 entries: 40/50; entries whose 2nd flag had no new evaluation: 26
```

**What it affects**
- The adopted D-95 numbers (noisy-arm detection 78–86%, false flags 2.0–2.5%) were measured with this behaviour.
- In 65% of the detections (26 of 40), the "confirmation" was a re-count of the same evaluation.

**Fix**
- In `DiscountedUCB.update_calibration`, call `set_calibration_mismatch` only for arms in `report.evaluated`. Other arms' streak counters stay unchanged.
- Alternatively, require `mismatch_windows` consecutive *evaluations* of the arm, by tracking an evaluation counter per arm in the calibrator.
- Document the chosen semantics in a new decision entry.

**Retest:** re-run the D-93 tuning and the D-94 pre-registration on fresh seeds. Expect lower detection; B6 in `improvement_required.md` (more frequent checks) becomes the way to recover it.

### E-03. The real noisy arm cannot be flagged

**Where:**
- `README.md` §6: noisy has GSM8K train prompts only, with no calib or test split.
- `curator-rl/src/curator_rl/calibration/calibrator.py` docstring: "Environments without their own slice (`domain_of[j] is None`) are never evaluated or flagged."

**What is wrong**
- In the simulator, the noisy arm has its own independent calibration slice whose score never moves. Detecting it is therefore easy.
- On GPU it has no slice at all, so the whole S5 pathway, the evidence for H4, does not apply to it.
- D-76 already notes that a shared slice needs transfer-matrix credit (C2d), which is not built.

**What it affects:** H4 ("calibration detects the noisy arm") cannot be tested on GPU as designed. All simulator detection rates are optimistic.

**Fix:** `improvement_required.md` A1 — map noisy to the GSM8K slice, add `TransferCredit` (C2d), check identifiability, and add shared-slice simulator variants.

### E-04. Pre-registered evaluations are not reproducible

**Where:**
- `curator-rl/experiments/claim_gate.py:48-55`. Also `refit_gate.py::method_cfg`, `gate2_check.py`, `gate2b_check.py`, `cre_gate.py`, `gamma_gate.py`, `sweeps/tune_all.py`.
- All of them build configurations from the live `configs/base.yaml`.

**What is wrong**
- After D-95, `base.yaml` itself holds `targeting: claim_stale, max_targets: 2`. The H-claim script's control ("curator", meant to be `exposure/1`) is now the same configuration as its candidate.
- Every gate script now runs a different Curator than the one its committed report describes.

**Evidence:**
```
$ python -c "from experiments.claim_gate import cfg_for; ..."
curator      targeting=claim_stale max_targets=2
curator_cs2  targeting=claim_stale max_targets=2
```

**What it affects**
- No committed report is wrong, but none of them can be regenerated, which the project's own "every number comes from a script" rule requires.
- A re-run of `claim_gate.py` would report detection difference 0 and "Adopt: NO".

**Fix**
- At pre-registration, freeze the exact configuration: `configs/frozen/D-94.yaml` plus its `config_hash` recorded in `DECISIONS.md`. The gate script loads that file, never `base.yaml`.
- Add a test asserting each gate script's control and candidate hashes equal the recorded ones.
- Stamp reports with commit and config hash (`improvement_required.md` E6).

### E-05. The "refit from measured data" simulator does not match the GPU run shape

**Where:**
- `curator-rl/experiments/analysis/refit_scenarios.py:37-38` (`ROUNDS_UNDER_UNIFORM = 70`, `PROMPTS_PER_ROUND = 5 * 16`);
- `curator-rl/configs/base.yaml:164` (`steps_per_round: 5`);
- the plan in README §10 and D-84: R = 2, P = 16, 60 rounds.

**What is wrong**

| | Simulator refit | GPU plan |
|---|---|---|
| Prompts per round | 80 | 32 |
| Prompts per run | ≈ 5,600 | ≈ 1,920 |
| LP window (W = 10 rounds) covers | 800 prompts | 320 prompts |
| Calibration window (K = 10) covers | 800 prompts | 320 prompts |
| Prompts per round for an arm at 20% share | 16 | ≈ 6 |

- Per-round signal variance on GPU is therefore about 2.5× the simulated variance.
- Each run sees about 2.9× fewer prompts, so learning effects are smaller.
- Quota rounding is much coarser (E-09).
- `base.yaml`, which the GPU runs are meant to use, still says `steps_per_round: 5`.

**What it affects:** every refit result (D-89 … D-95) is internally valid as a simulator result. But it was produced at a signal-to-noise ratio and run length the GPU runs will not have, so it overstates how well the signals, statuses and calibration will work on GPU.

**Fix**
- Regenerate the refit scenarios with `steps_per_round: 2`, `prompts_per_step: 16`, a budget of about 60 rounds under Uniform.
- ~~Rescale η~~ (corrected: η/P per prompt does not depend on R, so η stays).
- Add a GPU-shape config (`configs/experiment/gpu_l1.yaml`) with `steps_per_round: 2`, rather than overloading `base.yaml`.
- Re-tune on tuning seeds before any new pre-registration.

---

## High

### E-06. MATH checker accepts wrong answers and rejects correct ones

**Where:** `curator-rl/src/curator_rl/envs/math35.py:97` (`_numeric_value` strips every parenthesis) and `:63-72` (`_sub_frac`).

**What is wrong**
- `2\frac{1}{2}` (a mixed number, 5/2) is normalised to `2(1)/(2)`.
- The parentheses are then deleted, giving `21/2`.
- Any implicit product such as `2(3)` becomes `23`.
- This contradicts the module's guarantee that it "never accepts two different values as equal".

**Evidence** (`audit/verify_errors.py math`):
```
BUG pred='50.5'          gold='10\frac{1}{2}' -> True  (correct answer: False)
BUG pred='\frac{21}{2}'  gold='10\frac{1}{2}' -> False (correct answer: True)
BUG pred='23'            gold='2(3)'          -> True  (correct answer: False)
BUG pred='\frac{5}{2}'   gold='2\frac{1}{2}'  -> False (correct answer: True)
```

**What it affects**
- MATH35 rewards during GRPO: a wrong answer can be rewarded.
- The measured pass rates (pass@1 0.168) that set the refit difficulty.
- The claimed 100% gold / 0% garbage check did not include these forms.

**Fix**
- Recognise mixed numbers explicitly: `^(-?\d+)\s*\(?(\d+)\)?/\(?(\d+)\)?$` after `_sub_frac` → `n + a/b`, with the sign applied to the whole value.
- In `_numeric_value`, accept parentheses only in the exact `(a)/(b)` shape that `_sub_frac` produces. Any other parenthesis means "not a plain number" (string comparison only).
- Add the four cases above as tests.
- After downloading the data, count gold answers containing a mixed number or implicit product (`grep -c "\d\\\\frac" data/processed/math35/*.jsonl`) and re-run the gold/garbage check.

### E-07. MBPP sandbox can be passed without solving the task

**Where:** `curator-rl/src/curator_rl/envs/sandbox.py:87` (the sentinel is written into the program file as a literal) and the plain `assert expr == expected` tests.

**What is wrong:** the module claims "a program cannot print a sentinel it cannot guess", and that early exit fails. Two solutions that do not solve the task both pass:
1. **Read the sentinel from its own source and exit early.**
   ```
   print(re.search(r"SANDBOX_OK_[0-9a-f]+", open(__file__).read()).group(0), flush=True)
   os._exit(0)
   ```
2. **Always-equal object.** Return an object whose `__eq__` always returns `True`; every `assert f(x) == y` passes.

**Evidence** (`audit/verify_errors.py sandbox`):
```
OK  wrong solution: passed=False
BUG reads sentinel from its own source: passed=True
BUG always-equal object: passed=True
```

**What it affects:** GRPO optimises reward. The always-equal bypass is a known reward hack in code-RL training, and a policy that finds it will be rewarded on MBPP without learning to code.

**Fix** (defence in depth; in-process verification can never be made fully tamper-proof):
1. **Keep the sentinel out of the source.** Read it from an environment variable at harness start-up and delete it from `os.environ` before solution code runs; keep it only in a local variable of the harness.
2. **AST screening of the solution before execution.** Reject solutions that:
   - reference `__file__`, `open`, `os._exit`, `sys.exit`, `exit`, `quit`;
   - define `__eq__`, `__ne__` or `__hash__`;
   - use `globals`, `builtins`, `gc`, `inspect`, `sys._getframe`, `ctypes`, `importlib`, `exec`, `eval`.
   Log every rejection so hacking attempts are visible during training.
3. **Type-strict assertions.** Rewrite each `assert A == B` into `assert _eq(A, B)`, where `_eq` is captured before user code runs and checks `type(a) is type(b)` recursively before `==`.

**Retest:** both bypasses as unit tests in `tests/unit/test_sandbox.py`; gold MBPP solutions must still pass at 100%.

### E-08. GSM8K checker rejects correct answers in common formats

**Where:** `curator-rl/src/curator_rl/envs/gsm8k.py:29` (`\boxed\{([^{}]*)\}` cannot contain braces) and `:50` (removes `$` but leaves the backslash of `\$`).

**Evidence** (`audit/verify_errors.py gsm8k`):
```
OK   '\boxed{18}'                -> correct=True
MISS '\boxed{\$18}'              -> correct=False
MISS '\boxed{18 \text{ dollars}}' -> correct=False
MISS '\boxed{1{,}000}'           -> correct=False
OK   '\boxed{1,000}'             -> correct=True
```

**What it affects**
- GSM8K and noisy-arm verification.
- Measured pass rates (the 0.350 pass@1 that sets the refit difficulty).
- The training reward, which penalises a correct answer written as `\$18`.
- How often Qwen2.5-0.5B uses these forms is unknown until the saved pilot samples are audited.

**Fix**
- Use the brace-balanced `extract_boxed` from `math35.py`.
- Strip `\$`, `\text{…}` / `\mbox{…}` units, `{,}`, `\,`, `\!` before numeric parsing.
- Add the five cases as tests.
- Re-score `samples_<model>_gsm8k.jsonl` from the Kaggle pilot with the fixed checker and report the change in pass@1. If it moves by more than 0.02, refit the difficulty (E-05 regeneration covers this).

### E-09. Requested weights are not what is delivered

**Where:**
- `curator-rl/src/curator_rl/core/quotas.py`;
- `scheduler/ducb.py::update_observation` (`u = weights_used`);
- `simulator/harness.py` (`exposure += w × M`);
- `simulator/world.py:210` (skill update uses requested `weights`; rollouts and costs use integer quotas).

**Evidence** (`audit/measure_noise_and_quotas.py`): an arm requested at 0.020 receives 0.025 of prompts at M = 80 and **0.031 at M = 32** (the GPU shape), every round.

**What it affects**
- Bandit counters, calibration credit and the simulated learning all use 0.020 while 0.031 is trained and paid for.
- The simulated world is internally inconsistent: learning from requested weights, cost from delivered.

**Fix:** `improvement_required.md` A2 (deficit-carry rounding; delivered shares everywhere; world flag `learning_from: delivered`).

### E-10. Simulator final scores carry independent evaluation noise per method

**Where:**
- `curator-rl/src/curator_rl/simulator/harness.py:212`: the final score is `world.evaluate_benchmark(observed=True)`.
- `simulator/world.py:326`: Binomial noise drawn from the *training* RNG stream.

**Evidence** (S-I-R1, seeds 0–39, Curator − Uniform):

| Final score | CI half-width | SD of paired difference |
|---|---|---|
| observed | 0.0062 | 0.0202 |
| noiseless | 0.0035 | 0.0115 |

So 68% of the variance of paired differences is evaluation noise. On GPU, all methods are scored on the same test items, so this noise would be largely shared across methods.

**What it affects:** every simulator CI is about 1.8× wider than necessary. D-92's γ = 1 rejection came from an underpowered non-inferiority test this would have resolved. Reporting evaluations also shift training random draws.

**Fix:** `improvement_required.md` B1 — record `expected_benchmark_score()` as the primary simulator metric (or paired test items), and give evaluations their own RNG stream.

### E-11. The planned statistics for the GPU matrix give intervals that are too narrow

**Where:** `curator-rl/docs/EXPERIMENT_PROTOCOL.md` §4 ("95% percentile bootstrap, 10,000 draws; seeds are the resampling unit"), applied to "5 paired seeds for the main matrix, 3 for ablations, regime H6 and LOO".

**Evidence** (simulated normal paired differences, 4,000 replications each):
```
n= 3: percentile-bootstrap coverage 0.749   t-interval coverage 0.947
n= 5: percentile-bootstrap coverage 0.828   t-interval coverage 0.945
n=50: percentile-bootstrap coverage 0.939   t-interval coverage 0.944
```

**What it affects:** with 5 seeds a "95%" interval is really about 83%; with 3 seeds about 75%. H1/H2 and every ablation would be claimed on intervals that are too narrow. The simulator gates (50–100 seeds) are essentially unaffected.

**Fix**
- For n < 20 paired seeds, use the t-interval on paired differences (`scipy.stats.t.ppf(0.975, n−1)·s/√n`), or an exact sign-flip permutation test (2⁵ = 32 sign patterns at n = 5).
- For equivalence (H6, Gate 4), use TOST with t.
- Keep the Holm correction on H1/H2.
- Update EXPERIMENT_PROTOCOL §4 before Gate 10 (the freeze).

### E-12. The compute plan was not updated after measurement

**Where:** `curator-rl/docs/ROADMAP_v3.md` §7 ("one Tier-1 run = B ≈ 1.0 GPU-h"; L1 total "≈ 105 (+20% ≈ 125)"), repeated in README §12.

**What is wrong:** measured run cost is 1.4–2.8 GPU-hours (README §5, COST_MODEL §7). The Level-1 table has 78 training runs:

| Block | Runs |
|---|---|
| Integration | 6 |
| Calibration | 3 |
| Main matrix | 35 |
| Ablations | 9 |
| Homogeneous regime | 9 |
| Second model family | 4 |
| Leave-one-out | 12 |

78 runs × 1.4–2.8 GPU-hours = **110–220 GPU-hours** before pilots, against the stated 125. The second model family (Llama-3.2-1B) is unmeasured; a larger model costs more per prompt (1.5B measured at about 2× the 0.5B), so its 4 runs are underestimated too.

**Fix**
- Recompute §7 from measured costs and state Kaggle's weekly quota (about 30 GPU-hours, to be confirmed on the account) as calendar time.
- Decide in advance which Plan-B cuts apply if the budget exceeds what is available.

---

## Medium

### E-13. JSONL logs lose every record written after a crash

**Where:** `curator-rl/src/curator_rl/core/jsonl.py:48` (reopens with `"a"`, no check that the file ends with a newline) and `:75` (the reader stops at the first corrupt line).

**Evidence:** two records written; then a crash leaves `{"_schema": 1, "_seq": 3, "rou` with no newline; then three records written after resuming:
```
records readable: [1, 2] (written after resume: [4, 5, 6]); skipped lines: [1]
```
The first resumed record is glued to the partial line, and the reader stops there.

**What it affects:** Phase F trainer logs on Kaggle, where sessions are killed. Logs written after the first crash are silently lost.

**Fix**
- In `JsonlWriter.__enter__`, if the file does not end with `\n`, truncate it back to the last newline (record the truncated bytes in a sidecar file), then append.
- Make `read_jsonl` skip each corrupt line individually and report the count, rather than stopping.
- Add this scenario as a test.

### E-14. MBPP is treated as an inexhaustible dataset

- **Fact:** MBPP has 314 training problems (`data/manifests/mbpp_train.txt`).
- **Simulator:** a refit episode is ≈ 70 rounds × 80 = 5,600 prompts. Curator gives MBPP 0.265 (`refit_gate.md`), i.e. 1,484 prompts = 4.7 passes over the dataset. The best D-90 cell gives it 0.46: 2,576 prompts = 8.2 passes.
- **GPU plan:** 1,920 prompts per run, so 0.46 means 2.8 passes.
- The simulator's MBPP learning speed never decays with repetition. D-90's "move budget to MBPP, the cheapest learnable arm" therefore rests on an arm that will be memorised.
- **Fix:** add per-arm dataset size to `SimEnvParams`. Multiply learning by `max(0, 1 − passes/passes_max)`, or apply a memorisation penalty so that once passes > 1 the benchmark gain per unit of training-reward gain falls. Report MBPP passes per run in every gate.

### E-15. Calibration cost in the simulator is probably about 5× too high

- **How it is charged:** each calibration item costs `mean unit cost / G` (`experiments/refit_gate.py::scenario_with_calibration`). Unit cost includes the policy update, about 80% of a vLLM step. For GSM8K in R1 that is 2.64 s / 8 = 0.33 s per item.
- **What it really costs:** a calibration item is one greedy generation plus verification, with no update: about 0.524 / 8 ≈ 0.066 s on the measured probe.
- **Effect:** averaged over arms (mean unit cost 2.86 s / 8 = 0.36 s charged vs about 0.08–0.09 s real), calibration looks about 4–5× more expensive than it will be, which limited every calibration design in D-74 … D-95 (items per slice, K, number of targets).
- **Fix:** charge calibration at generation-only cost (`gen s/prompt / G` + verify), and confirm against the `EVAL` span once the cost meter exists.

### E-16. The DUMP-style baseline is weaker than DUMP

- **Where:** `curator-rl/src/curator_rl/scheduler/baselines/dump.py:54`. It keeps cumulative prompt counts and an all-time mean of |advantage|. The docstring admits "the exact windowing … of the original [is] not reproduced".
- **Why it matters:** DUMP estimates learnability from recent samples. An all-time mean reacts slowly as arms saturate, which weakens this baseline in exactly the drifting settings where H2 is tested.
- **Fix:** add a sliding window of the last W rounds (W ∈ {5, 10, 20}) to DUMP-style. Keep the equal tuning budget of 8 configurations by trading grid points (e.g. 2 c × 2 τ × 2 W).

### E-17. Trace replayer emits inconsistent observations

- **Where:** `curator-rl/src/curator_rl/simulator/replay.py:82-95`.
- **What is wrong:**
  - `n_groups_mixed=0` for every arm, so richness is always 0.
  - `sum_score` is set to one rate instead of a sum over prompts.
  - Quotas use `int(round(w·M))`, which does not sum to M.
- **Coverage:** 0% (`pytest --cov`).
- **Fix:** carry `n_groups_mixed`, `sum_score`, `sum_score_sq` and `sum_abs_adv` in the trace format and scale them like `k_success`; use `largest_remainder_quotas`; add a test that replaying a simulator trace reproduces the simulator's observations.

### E-18. "Without replacement within a pass" is not implemented

- `docs/SPEC.md` §8 requires `generate_batch` to sample "without replacement within a pass".
- `Gsm8kEnv` and `FiniteJsonlEnv` sample with replacement across batches; only within-batch duplicates are avoided.
- `Gsm8kEnv` initialises `_remaining` and `_passes` (`envs/gsm8k.py:114-115`) and never uses them. The docstring delegates the job to "the trainer sampler", which does not exist yet.
- **Fix:** build pass-based sampling (a shuffled permutation per pass, persisted in the checkpoint) in Phase F `trainer/prompt_stream.py`; delete the dead attributes; correct SPEC §8 to name the owner.

### E-19. Countdown instructions contradict the generator and verifier

- **Where:** the prompt says "Division is only allowed when it divides exactly" (`envs/countdown.py:50`). The generator (`_reachable_targets`) and verifier (`:138`, `left / right` on Fractions) allow inexact intermediate division.
- **Evidence:** 4 of 300 sampled training instances (1.3%) are reachable *only* through a division the prompt forbids.
- **Fix:** either drop the sentence from the prompt, or enforce integer intermediates in both generator and verifier. Prefer enforcing, since it matches the stated rules. Add a test.

### E-20. Gate 4's equivalence test is likely unattainable with 3 seeds

- **The test:** Gate 4 requires the passthrough Curator to be equivalent to plain GRPO within δ_S = 1 benchmark point (`EXPERIMENT_PROTOCOL.md` §3), and the plan uses 3 seeds.
- **Arithmetic:** with a t-interval (E-11), the half-width is `t₀.₉₇₅,₂ · SD/√3 = 4.30 · SD / 1.73 = 2.48·SD`. Fitting inside ±1 point needs a run-to-run SD of paired differences **below 0.40 points**, which is very small for GRPO on a 0.5B model.
- **Fix:** use Gate 3's three uniform seeds to measure run-to-run SD, then size Gate 4 (seeds or margin) by a written power calculation before running it. A same-seed passthrough comparison (identical data order and sampling seeds, so differences come only from GPU non-determinism) is a stronger and cheaper test of "does not alter GRPO".

---

## Low

### E-21. `status.flip_rate` counts warm-up transitions
- **Where:** `curator-rl/src/curator_rl/signals/status.py:28` sums transitions over the whole trace but divides by the tail length. `flip_rate([S1,S2,S2,S3], warmup=2)` returns 200; the correct value is 100.
- **Impact:** no reported number is affected; the gates use the pooled estimator (`common.flip_rate_pooled`). The unit test (`tests/unit/test_status.py:183`) passes only because the prepended state equals the trace's first state.
- **Fix:** count over `tail`; add a test with a prepended transition.

### E-22. Gate 2 oracle-gap closure mixes seed sets
- **Where:** `curator-rl/experiments/gate2_check.py`: the oracle mean is over the first 20 seeds, while Uniform and Curator means are over 50.
- **Fix:** compute all three on the same seeds.

### E-23. Dead or mis-described configuration
- `calib.target`, `calib.trust_region` and `warmup.total_rounds` are validated and hashed but never read (only `scheduler.warmup_rounds` controls warm-up). Changing them changes the config hash but not behaviour.
- `base.yaml` comments:
  - `s3_multiplier` / `s4_multiplier` say "UCB score × m" and `s5_shrink` says "shrink proxy mass", but `mixture.py::status_multiplier` multiplies **weights**;
  - `pull_unit` says "scale of the UCB log argument", but it selects the counter increment.
- `signals/progress.py` docstring says LP-B uses the per-round posterior; it uses raw rates (the root of E-01).
- **Fix:** remove the dead keys (or wire them: `trust_region` belongs to the α/β fitter) and correct the comments. SPEC §7's key list must change with them.

### E-24. Noisy-arm rewards are not independent per rollout
- `envs/noisy.py` draws the reward from `sha256(salt | prompt_id | response)`. Identical completions in a group get identical rewards, which matters as a trained model's outputs collapse.
- **Fix:** add the rollout index within the group to the key, so the reward is reproducible but independent per sample.

### E-25. Sealed-test guard can fail open
- `evaluation/guard.py::_previous_accesses` skips corrupt log lines, so a corrupted entry permits a second opening.
- `_register_access` rewrites the whole log (read-modify-write), so two processes opening at once (two Kaggle GPUs) can lose an entry.
- **Fix:** fail closed on any unparseable line; append with `O_APPEND` and `fsync`; take a lock file during registration.

### E-26. Cost probe's "learnable" filter includes the noisy arm
- `scripts/kaggle_cost_probe.py:152` uses `pass_at_G > 0`, which includes noisy (pass@8 = 1.0). The reported 2.67× spread is unchanged, but the definition differs from the portfolio rule.
- **Fix:** take the learnable set from the portfolio definition.

### E-27. Stale documents
Listed with exact fixes in `improvement_required.md` E11: README status line and test counts, COST_MODEL header, GATES phase log, `scheduler/curator.py` docstring, duplicate `Curator/` folder, MATH problem count.

---

## Which decisions and claims need to be re-run

| Decision / claim | Affected by | Status after this audit |
|---|---|---|
| D-40 LP estimator choice | E-01 | re-run the study after the fix; LP-A may win |
| D-42 status thresholds | E-01 | re-tune after the fix |
| Gate 2 (d) status F1 0.932 | E-01 | valid as measured behaviour; thresholds were tuned on deflated z |
| D-90 diagnosis, item 3 (LP undetectable) | E-01, E-05 | largely an artefact; re-diagnose |
| D-95 adoption of `claim_stale/2` (detection 78–86%) | E-02, E-05, E-15 | re-run on fresh seeds after fixes |
| D-89 / D-92 / D-95 score comparisons | E-05, E-09, E-10, E-14 | valid as simulator results; weak transfer to GPU |
| H4 (spurious-arm detection) | E-02, E-03 | not testable on GPU as designed |
| Gate 1B' portfolio, refit difficulties | E-06, E-08 | re-score pilot samples with fixed checkers |
| Any reported CI from 3–5 GPU seeds | E-11 | change the method before Gate 10 |
| Level-1 compute budget | E-12 | recompute |
| D-89 and later reproducibility | E-04 | freeze configurations per pre-registration |

---

## Plan to fix everything

Each step lists its dependencies first. One branch and pull request per step, with a teammate reviewing.

1. **Verifiers and infrastructure** (no simulator result changes; 2–3 days):
   - E-06 MATH checker, E-07 sandbox, E-08 GSM8K checker, E-19 Countdown, E-24 noisy key — each with the failing cases above as tests;
   - E-13 JSONL, E-25 guard, E-17 replay, E-21 flip rate, E-23 config, E-26, E-27.
   - Then re-score the saved Kaggle pilot samples with the fixed checkers and record the new pass rates in a decision entry.
2. **Statistics and reproducibility (1 day):**
   - E-04: freeze the configurations of D-88, D-91, D-94 into `configs/frozen/` (`exposure/1` for D-88/D-91/D-94 controls);
   - E-11: protocol change to t-intervals for n < 20;
   - `evaluation/stats.py` shared by all gates (`improvement_required.md` E4).
3. **Signal and calibration logic** (tuning seeds only; 2–3 days):
   - E-01 LP-B SE fix, then re-run the D-40 study and D-42 tuning;
   - E-02 flag-only-evaluated-arms;
   - E-10 noiseless final score with a separate RNG stream;
   - E-09 delivered shares;
   - E-15 generation-only calibration cost.
4. **Simulator fidelity (2–3 days):**
   - E-05: regenerate the refit at the GPU shape (R = 2, 60 rounds), using pilot numbers from step 1;
   - E-14: dataset exhaustion for MBPP;
   - E-03 / A1: shared-slice noisy arm and C2d credit;
   - E-16: windowed DUMP-style.
   - Re-run fair tuning (`tune_all.py`) on tuning seeds for every method.
5. **One consolidated pre-registration on fresh seeds 700–799**, written before running:
   - Full Curator after all fixes vs all baselines (plus `pilot_uniform`, `improvement_required.md` B10) on the regenerated refit scenarios;
   - detection, false flags and calibration cost on the shared-slice variant.
   - This replaces D-89/D-95 as the project's current simulator result.
6. **GPU plan corrections before Phase F/G:**
   - E-12 compute budget;
   - E-20 Gate 4 design (same-seed passthrough plus a power-sized equivalence test);
   - E-18 pass-based prompt stream;
   - the measurement list for the cost meter (update share, calibration item cost).
