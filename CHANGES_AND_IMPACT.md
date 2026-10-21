# CURATOR — what this branch changes, and why it matters

## Round 2 (2026-10-12): the code is complete for the GPU runs

All numbers below are on simulator **tuning seeds 0-39** (exploratory; the confirmatory run is pre-registered as D-99 on fresh seeds 700-799 and waits for the owner).

### The decisive new component: a reward audit that costs nothing

**Problem.** After the audit fixes, tuned Curator still gave the random-reward ("noisy") arm about 35% of the budget. Its reward produces the most mixed groups of any arm, so the proxy loves it, and benchmark calibration cannot catch it at the Level-1 budget: 1,920 prompts barely move a held-out slice (detection 10-28%).

**Idea.** A verifier scores a completion against *its* prompt, and prompts differ in difficulty, so the pass rates of a round's prompt groups spread out more than coin flips would. A reward that ignores the prompt does not. The dispersion ratio D of group pass rates is 1 for a prompt-independent reward and (G + kappa)/(kappa + 1) otherwise. It comes from statistics every round already logs, at no extra compute (`signals/audit.py`).

**Grounded in real data.** On the Kaggle pilot every real verifier arm has D = 3.0-4.4 and the random-reward arm 1.0-1.09, on both Qwen2.5-0.5B and 1.5B. The threshold (d_max = 1.5) was set by Monte Carlo: a random reward is flagged in 99.8-100% of 60-round runs, by round ~11. Arms as heterogeneous as the measured ones are never flagged; arms with kappa = 10 are flagged in 0.2-3% of runs.

| at the GPU shape, realism v2 | before (tuned Curator) | with the audit |
|---|---|---|
| noisy arm's budget share | 0.27-0.35 | **0.06** |
| noisy arm detected (S5) | 7-30% of runs | **90-97%** (100% after re-tuning), median round 14-21 |
| gain over Uniform, S-I-R1-GPU2 | +4.5% | **+11.3%** |
| real arms ever flagged by the audit | — | **0** |

After fair re-tuning of every method on the realism-v2 scenarios (same 8-configuration budget), with each method at its tuned point:

| method | tuning scenarios | held-out twin S-J-R-GPU2 | stress: misaligned arm / noisy hurts / noisy helps |
|---|---|---|---|
| **Curator + audit** | **+11.2 to +11.9%** | **+11.8%** | **+5.4% / +12.3% / +8.8%** |
| LP curriculum | +5.7 to +6.0% | +6.5% | +3.4% / +5.9% / +5.8% |
| pilot-filtered uniform (B10) | +3.6 to +3.9% | +4.5% | +2.7% / +3.1% / +4.2% |
| Standard UCB | +4.0% | +4.2% | +2.0% / +2.5% / +4.0% |

**What it cannot do (stated in the code and D-98).** It catches rewards that ignore the prompt (random, constant, prompt-unrelated format checks). A reward that depends on the prompt but rewards the wrong thing is benchmark calibration's job. The misaligned-arm stress scenario shows calibration catches such an arm only about 10% of the time at this budget.

### What did not help (kept, tested, switched off)

- **alpha/beta fit to calibrated gains (B4):** recovers planted weights (R² 0.86, 65% lower error than unweighted least squares) but changes Curator's score by -0.001 to +0.001.
- **Drift-triggered discounting (B2):** catches an abrupt shift within 5 rounds with fewer than 1 false alarm per 100 rounds, but scores the same as gamma 0.95 on S-E.
- **Weight cap / smoothing (B7):** lower concentration, no score gain. **Longer LP window (B3):** W = 30 is worse; W = 10 stays.
- **Benchmark calibration at the Level-1 budget:** Curator without it scores +0.5 to +0.9 points higher. It stays on in the preset because it is the measurement the ROI leaderboard needs; `curator_nocal` is the ablation.

### ROI and leave-one-out (D3): honest limits

The in-run ROI, credited on each arm's own slice, ranks arms like leave-one-out once pooled over seeds (Kendall tau +0.6 and +1.0). From one run, which is all one GPU seed gives, the median tau is only 0.4-0.6. A one-seed GPU ROI result is descriptive. A full transfer-matrix ROI ranked arms almost backwards at this budget, which is why the per-slice form is the default.

### Runs that survive Kaggle

- **Every round is checkpointed** and mirrored in the background to a private Hugging Face dataset (`--sync-to hf://you/curator-runs`). Training never waits for an upload.
- **A lost session resumes** by rerunning the same command.
- **A crashed worker is restarted** from its checkpoint; after an out-of-memory crash the micro-batch is halved.
- **Before the 12-hour limit** runs stop cleanly and save.
- **One bad verifier or evaluation** is logged and skipped instead of killing the run.
- **Re-launching a finished run** no longer overwrites its result (a bug found and fixed here, E-31).

The bundle's smoke test restores a deleted run from its synced copy and resumes it.

### Engineering

- **Lock file and pins:** `requirements.lock` for the CPU stack; exact GPU-stack pins in the `train` extra.
- **Import paths:** no `sys.path` hacks; `experiments` is a package.
- **Typing:** TypedDict checkpoint states; pyright at 0 errors in CI.
- **CI data job:** CI rebuilds the datasets from their public sources, proves the committed split manifests are reproduced byte for byte, and enforces 85% coverage (89% locally).
- **Tests:** 481 pass.
- **Simulator reproducibility:** every new simulator and scheduler switch defaults to the old behaviour. 210 of 210 fingerprinted episodes are bit-identical to before.

---

## Round 1 (2026-10-10)

Branch `audit-fixes-and-gpu-run` (2026-10-10). Three parts:
1. fixes for the verified errors in `errors_found.md`;
2. the Phase F runner, which lets CURATOR drive a real TRL GRPO trainer;
3. a Kaggle GPU package (`dist/curator-gpu-kaggle.zip`, built by `scripts/make_kaggle_gpu_bundle.py`).

**How each claim below was checked:**
- **Tests:** 404 pass (1 Linux-only skip), up from 349, and lint is clean.
- **Simulator numbers:** tuning seeds only (0–49), so no evaluation seed range was spent. They are exploratory; the confirmatory run is pre-registered as D-96 on fresh seeds 700–799.

---

## 1. Decisions with measured impact

### 1.1 Check the GPU shape on a CPU before spending GPU hours (new error E-28)

**What we did.** The "refit from measured data" simulator ran 80 prompts per round and about 5,600 per run. The GPU plan is 32 and 1,920. We regenerated the refit scenarios at the GPU shape (`configs/sim/scenario_*_gpu.yaml`) and re-ran every method there.

**What it found.** In that setting, Curator, both before and after the other fixes, was **pinned near Uniform**:
- 77% of arm-rounds were locked in the S1 "unexplored" status;
- the S1 exit threshold (64 effective prompt groups) was tuned for 80-prompt rounds;
- at 32 prompts per round the evidence settles at 59.9, so an arm never exits S1, and S1's exploration quota held the mixture flat.

**Fix.** `rescale_for_batch_shape()` scales the threshold with the batch shape.

**Impact** (`reports/analysis/audit_impact.md`, `reports/sweeps/tune_gpu_shape.md`):

| | Before | After |
|---|---|---|
| Curator vs pre-audit Curator | +0.0008 | **+0.0060 [+0.0043, +0.0078]** |
| Curator after fair re-tuning, gain over Uniform | **+0.8%** (worst of the five adaptive methods) | **+6.0%** (level with the LP curriculum's +6.2%) |

**Result at the GPU shape** (every method at its own tuned point, equal tuning budget):

| Scenario | Ranking |
|---|---|
| S-I-R1-GPU (3.46× cost spread) | **Curator** first, ahead of LP, UCB, SEC, DUMP |
| S-I-R2-GPU (2.06× spread) | LP first, Curator second |
| S-J-R-GPU (equal costs) | LP first, Curator second |

Cost-awareness pays when the cost spread is large (H3, narrowly), and only once the scheduler is allowed to act at the GPU's batch shape. Without this check, the first Kaggle runs would have compared a scheduler stuck at Uniform.

### 1.2 Make spurious-reward detection honest, then measure it where it will run

Three fixes:
- **E-02:** flags only from new evaluations, so one evaluation is no longer counted twice;
- **E-03:** the noisy arm is scored on the GSM8K slice with joint (transfer-aware) credit, as on real hardware;
- **E-01:** learning-progress z-scores are calibrated.

**Impact.** The adopted 78–86% detection (D-95) came from the long simulated horizon, an independent noisy slice and double-counted flags.
- **At the GPU shape the pre-audit Curator detects 0%.**
- After the fixes it detects 10–18%.
- Tuned and calibrating every 5 rounds, it detects 22–28% (false flags 5.5–8.5%).
- Checking every 2 rounds with 200 items (18% of the budget) does not help.

**Why it matters.** H4 ("calibration catches the spurious arm") is **not achievable by benchmark calibration at the Level-1 GPU budget**. In 1,920 prompts the real arms barely move on held-out slices, so there is no gain to compare against. Writing this before the GPU matrix saves the project from a negative result discovered late. The next design, a cheap reward-vs-verifier audit on the arm's own completions, is proposed in D-96.

### 1.3 A learning-progress signal that means what it says (E-01)

**What was wrong.** The LP-B standard error treated per-prompt variance as per-round variance: it was about √n too large. On a flat arm the z-scores had SD 0.28 instead of 1, so every z-threshold in the status machine was inert, and the D-40 estimator choice rested on a "0.000 false positives" that this bug produced.

**Fix.** Each round's variance is now measured from the spread of pass rates *between prompts inside that round* (moments the observations already carry), floored at the binomial minimum. It is design-based and needs no extra data.

**Measured:** z SD 0.97 on the noisy arm in the GPU-shape simulator (was 0.41). Unit tests check calibration on a null arm and power on a real slope (mean z 2.8 where theory says 2.8).

### 1.4 Answer-checker and sandbox holes closed before GRPO could learn to exploit them (E-06, E-07, E-08, E-19)

- **MATH:** a wrong answer was accepted against a mixed-number gold answer (`50.5` vs `10\frac{1}{2}`), and correct answers were rejected. 6 MATH gold answers are mixed numbers.
- **MBPP sandbox:** two bypasses passed without solving the task (an always-equal object; reading the success token from the program's own file and exiting). Now the token arrives on stdin and lives in a closure, an AST screen logs hacking attempts as `sandbox_rejections`, and assertions are type-strict. All 464 gold solutions still pass.
- **GSM8K:** `\$18`, `18 \text{ dollars}` and `1{,}000` were rejected.
- **Countdown:** the generator and verifier now enforce the "exact division" rule the prompt states.
- **Gold/garbage contract tests still pass** on every dataset (now run locally with real data: 34 contract tests).

### 1.5 Reproducibility restored and proven (E-04)

Every pre-registered gate script read the live `base.yaml`. After D-95, the H-claim gate's control silently became its own candidate. Gate and study scripts now load `configs/frozen/d68_d94.yaml`: the exact config of commit 30052e8, plus the legacy value of every new switch.

**Proof:** re-running the D-94 gate on its registered seeds 600–699 reproduces the committed report exactly — every score, every pass/fail criterion, every detection rate and the adoption verdict. The only differences are floating-point noise around 10⁻¹⁷ and the timestamp.

### 1.6 Three times the statistical power for free (E-10), and honest intervals on GPU (E-11)

- **Simulator.** Every episode now also records the noiseless final score. 68% of the variance of paired differences was evaluation noise, so the new metric gives about 1.8× narrower intervals (worth about 3× the seeds).
- **GPU matrix.** It will use t-intervals and exact sign-flip tests (`core/stats.py`). The planned 95% percentile bootstrap covers only 83% at 5 seeds and 75% at 3, and the new code refuses to use it below 20.

---

## 2. Phase F: CURATOR drives an unmodified TRL 1.15 GRPO trainer

`src/curator_rl/trainer/` (previously an empty package):

| Module | What it does |
|---|---|
| `round_controller.py` | The round loop. Same order as the simulator harness: observation → scheduler → calibration every K rounds → uncharged reports → checkpoint → budget stop |
| `trl_adapter.py` | The only TRL import. `CuratorGRPOTrainer` fills each generation batch at generation time, so there is **no dataloader lag** (TRL prefetches batches; G.2 #4) |
| `prompt_stream.py` | Deficit-carry quotas (delivered share = requested within 1 prompt over the run; E-09) and sampling without replacement within each pass (E-18) |
| `reward_router.py` | Verifies completions in a thread pool (MBPP is CPU-bound) and accumulates exactly the `RoundObservation` statistics |
| `calibration_eval.py` | Fixed paired items, greedy decoding, McNemar SE; dev-split reports. The sealed test is never touched |
| `cost/attribution.py`, `cost/ledger.py` | Measured, CUDA-synchronised step time attributed to arms; matched-budget ledger |
| `checkpoint.py` | Atomic checkpoints (rename-on-complete), versioned and refusing mismatched runs; resumes across Kaggle sessions |
| `logprob_fallback.py` | Exact PyTorch replacement for TRL's Triton log-prob kernel. **TRL 1.15 GRPO cannot otherwise run on Kaggle's P100** (compute capability 6.0, Triton needs ≥ 7.0) |
| `sysid.py` | System identification from one randomized-mixture run: per-arm cost (NNLS on step time vs tokens), the learning/transfer matrix (including random-reward training's effect on GSM8K accuracy), polarisation κ |
| `fake_backend.py` | A CPU stand-in with real verifiers and a known cost model, used by the tests |

**Verified on CPU:**
- **Real TRL path:** the real TRL 1.15 `GRPOTrainer` ran under CURATOR with a tiny Qwen2 model — rewards routed, calibration each round, checkpoint, resume extended to a further round, uniform never calibrating, comparison report.
- **Fake-backend runs:** kill-and-resume reproduces the uninterrupted run exactly. System identification recovers the true per-arm cost coefficients (R² > 0.999) and the per-step overhead.

## 3. The Kaggle GPU package

`dist/curator-gpu-kaggle.zip` (3.5 MB, no sealed or raw data) contains `run.py`, `requirements*.txt`, `README_KAGGLE.md`, `curator_gpu.ipynb`, the code and the processed data.

| Command | Does |
|---|---|
| `python run.py setup` | T4: vLLM in its own virtualenv (as proven on Kaggle). P100: HF backend plus the PyTorch log-prob path |
| `python run.py smoke` | About 10 minutes: 3 rounds, calibration each round, checkpoint, resume to round 4 |
| `python run.py compare` | Curator vs Uniform vs LP, same seed, 1.5 charged GPU-hours each; two runs in parallel on 2×T4 |
| `python run.py sysid` | The randomized-mixture run; best on the P100, since learning and transfer don't depend on GPU type |

Runs resume across sessions with `--resume-from`. The zip was tested by extracting it to a clean folder and running `smoke`, `compare`, `sysid` and `report` with the fake backend.

**What is not yet done (honest list):**
- no run on a real GPU yet (the first command to run on Kaggle is `smoke`);
- the α/β proxy fit (B4), which should stop the noisy arm attracting budget (tuned Curator gives it about 35%);
- MBPP dataset exhaustion (E-14) and forgetting (B7) in the simulator;
- the confirmatory D-96 run on fresh seeds.
