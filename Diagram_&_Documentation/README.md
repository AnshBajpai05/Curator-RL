# Diagram & Documentation: everything the final report needs

State of 3 November 2026: every GPU experiment so far is in and analysed: the D-103 confirmation (sessions
A and B) and the D-108 compute test (fresh seeds 3-4, run on three machines; seed 5 incomplete). Every number
here comes from the run files or a committed report; nothing is typed by hand.

## Headline 1: D-103 (accuracy and mechanism, rules written before the runs)

| Rule | Verdict | Numbers (sampled accuracy, item-paired, percentage points) |
|---|---|---|
| R1: Curator + question filter beats Uniform + filter | **fails** | +1.8, -0.2, +0.3 by seed; mean +0.65 (1.1 SE) |
| R2: SEC and DUMP lose to Uniform | **fails** | SEC -2.3, -0.3, -0.2 (mean -1.0, 1.4 SE); DUMP -1.5, -0.6, -0.2 (mean -0.8, 1.9 SE). Every seed points the predicted way, none clears the 2-SE bar |
| R3: SEC and DUMP over-fund the broken reward, Curator does not | **holds** | SEC 33-34% and DUMP 31% in every seed (Uniform 20%); Curator variants 6-10% on average (single seeds up to 16%) |
| R4: Curator without calibration is as good as Curator | **holds** | +0.15 points (within +-1) |

By D-103's own rule (R1 and R2 did not both hold) the story is the fallback: a rigorous negative result
on accuracy, the mechanism (learning-signal curricula are drawn into a broken reward), and the pre-flight
check and in-run reward audit as the contribution. Full report: `analysis/d103_results/`.

**Session B** (descriptive, plus the pre-registered D-102 H-C): SEC at the authors' default temperature still
over-funds the broken reward (25%; the simulator predicted 26%; tuned SEC 34%, Uniform 20%) and scores -1.1
against Uniform (1 seed); perfect junk removal (`clean_uniform`) ties Uniform (-0.00, 2 seeds); with a realistic
buggy checker the pre-flight check flags it before training and Curator-Q caps it at 5.7% from round 1 against
Uniform's 20%, but accuracy is +0.65 points (0.86 SE): D-102 H-C fails.

## Headline 2: D-108 (compute efficiency, pre-registered 2 Nov)

**What ran.** Each of three team accounts ran seeds 3, 4 and 5 (not one seed each as planned); all finished
seeds 3 and 4 and were stopped by Kaggle's 12-hour limit early in seed 5. Same-seed runs on different machines
diverge, so there are six complete blocks (3 machines x 2 seeds), each pairing all five methods.

**Primary test (one block per seed, account k -> seed k): PENDING.** Seed 5 is missing. The two blocks in hand:
Curator-Q -0.28 and +0.71 at matched compute. Finishing seed 5 on account `compute_s5` settles it.

**Descriptive, all six blocks** (`analysis/d108_compute/replicates/`; same-seed replicates are not independent,
so the intervals are optimistic):

| Measure | Curator-Q | Curator-lite | SEC | DUMP |
|---|---|---|---|---|
| Accuracy at matched compute vs Uniform (mean over 0.25-1.5 GPU-h) | +0.21 | +0.44 (5 of 6 positive) | -0.14 | -0.42 (5 of 6 negative) |
| **Final accuracy vs Uniform** | **+1.36, 6 of 6 blocks above** | +0.86, 5 of 6 | +0.15, 4 of 6 | -0.57, 1 of 6 |
| Estimated GPU-hours saved to reach Uniform's final level | 0.39 (26%) | 0.45 (29%) | 0.16 | 0.09 |
| Questions on the broken reward | 6.3% | 10.6% | 33.2% | 30.6% |
| GPU-hours on useful environments (of about 1.5) | 1.29 | 1.22 | 0.85 | 0.87 |

- **The shape of the curve** (`figures/compute/learning_curves_matched_compute.png`): Curator-Q starts *behind*
  Uniform (its question filter spends extra generation up front), overtakes it at about 0.8 GPU-hours and ends
  about 1.2 points higher. That is why the matched-compute average is small while the final gap is large.
- **Where Curator-Q's final gain comes from:** MBPP +3.4, GSM8K +1.2, MATH35 +0.8 points.
- **Why this is not yet a claim:** on seeds 0-2 (D-103) Curator-Q *tied* Uniform (-0.25, +0.5, -0.69); the six
  new blocks cover only two seeds; final accuracy was not the pre-registered measure. A pre-registered test on
  fresh seeds would turn it into a result.
- **Reward checks:** the pre-flight gate (Curator-Q) put the broken reward in S5 at round 1 in 6 of 6 runs; the
  in-run audit (Curator-lite) flagged it in 5 of 6 (rounds 10-17), and in one run its dispersion (1.11) stayed
  below 1.5 without a tight enough bound to flag; that run still kept it to 11%. No real environment was ever
  flagged.

## Diagrams (open in a browser; the sources are in `source/`)

| File | What it shows | Use in the report |
|---|---|---|
| `01-project-overview.html` | budget and environments in, the scheduler sets the mix, GRPO trains, reward checks run alongside | introduction |
| `02-system-design.html` | every component: prompt stream, TRL trainer, reward router, sandbox, cost meter, signal engine, scheduler, calibrator, pre-flight check, checkpoints, simulator | system design |
| `03-one-round-game-changers.html` | one scheduling round, standard steps marked apart from ours (status rules, reward audit) | method |
| `04-status-lifecycle.html` | environment statuses S1-S5 and what moves an environment between them | method |
| `05-reward-health-checks.html` | pre-flight check, reward audit and calibration, with the GPU counts (21 of 22 runs flagged, never a real arm) | method, results |
| `06-real-run-walkthrough.html` | Curator seed 0 round by round from the real log: warm-up, Countdown cut at round 7, broken reward caught at round 8 | results |

## Figures

### `figures/` (D-103 runs, seeds 0-2)

| File | Finding |
|---|---|
| `gain_over_untrained.png` | training works for every method: +4.6 to +5.7 points of sampled accuracy over the untrained model |
| `accuracy_vs_uniform.png` | no method beats Uniform: Curator -0.1 [-1.2, +1.0], LP +0.2, Curator-lite 0.0, DUMP -0.8, SEC -1.0 (95% intervals all cross 0) |
| `noisy_arm_share.png` | budget on the broken reward: SEC 34%, DUMP 31%, Uniform 20%, LP 13%, Curator 10%, Curator-lite 9%, Curator-Q 6%, clean Uniform 0% |
| `noisy_share_by_round.png` | SEC climbs to about 42% by round 32, DUMP holds near 31-34%, Curator falls once the audit flags the arm |
| `no_signal_groups.png` | answer groups that teach nothing on the real environments: Curator 44-48%, LP 49%, SEC 53%, DUMP 54%, Uniform 59% |
| `gpu_split_by_environment.png` | useful GPU-hours: Curator variants 1.27-1.31, LP 1.17, Uniform, SEC, DUMP 0.85-0.89 (of about 1.5) |
| `cost_per_environment.png` | MATH35 costs 2.67x GSM8K to generate but only 1.22-1.58x per full training step |
| `mixtures/*.png` | per-run mixture over rounds; `sec_s0`-`sec_s2` and `dump_s0`-`dump_s2` show the over-funding repeating in every seed, `curator_s0` the cut at round 8; `olmo_*` the OLMo-2-1B pair |

### `figures/compute/` (D-108, six blocks on fresh seeds 3-4; descriptive)

| File | Finding |
|---|---|
| `learning_curves_matched_compute.png` | Curator-Q behind early, overtakes Uniform at about 0.8 GPU-h, ends about 1.2 points higher |
| `final_accuracy_vs_uniform.png` | Curator-Q +1.36 (6 of 6 above Uniform), Curator-lite +0.86, SEC +0.15, DUMP -0.57 |
| `matched_compute_delta.png` | at matched compute: Curator-lite +0.44, Curator-Q +0.21, SEC -0.14, DUMP -0.42 |
| `gpu_split_by_environment.png` | useful GPU-hours: Curator-Q 1.29, Curator-lite 1.22, Uniform 0.88, SEC 0.85, DUMP 0.87 |

## Analyses

- `analysis/d103_results/`: every D-103 rule with per-seed numbers, all methods against Uniform, `clean`,
  `sec_tau1` (`curator-rl/experiments/analysis/d103_confirm.py`).
- `analysis/d108_compute/primary/`: the pre-registered D-108 test (PENDING); `replicates/`: all six blocks,
  descriptive (`curator-rl/experiments/analysis/compute_efficiency.py`).

## Reports (`reports/`)

| File | Source | Supports |
|---|---|---|
| `gpu_comparison_qwen.md` | `curator_results_november/comparison.md` | every D-103 GPU number for Qwen2.5-0.5B, all seeds |
| `gpu_compute_test_d108.md` | `Curator-Nov-results/compute_analysis/replicates/` | the D-108 runs: matched compute, final accuracy, GPU split, reward checks |
| `gpu_stress_hackable_checker.md` | `Curator-Nov-results/raw/stress/curator_results/comparison.md` | the realistic buggy checker (D-102 H-C) |
| `gpu_comparison_olmo.md` | `curator_results_olmo/comparison.md` | the OLMo-2-1B pair |
| `sysid_seed0.md`, `sysid_seed1.md` | `curator_results_new/runs/dirichlet_s*/sysid_report.md` | measured training cost per environment, group polarisation, transfer |
| `cost_probe_generation.md` | `curator-rl/reports/pilot/phase_e_cost_probe.md` | generation-only cost spread (2.67x) |
| `fair_tuning_gpu2.md` | `curator-rl/reports/sweeps/tune_gpu2.md` | every method got 8 configurations under the same rule |
| `sec_dump_fidelity.md` | `curator-rl/reports/analysis/sec_dump_fidelity.md` | our SEC and DUMP against the authors' code |
| `sim_component_study_gpu2.md` | `curator-rl/reports/analysis/component_study_gpu2.md` | the reward audit is the component that matters (simulator) |
| `sim_heldout_gpu2.md` | `curator-rl/reports/analysis/heldout_gpu2.md` | simulator gains (+11-12% over Uniform) that did not transfer to GPU |
| `sim_audit_impact.md` | `curator-rl/reports/analysis/audit_impact.md` | why calibration cannot detect the arm at the GPU shape |
| `self_critic.md` | `../self_critic.md` | why the mix barely moves accuracy at this scale |

## What the evidence supports

- **Claim (pre-registered or replicated in every run):** training works (+4.6 to +5.7 points); no pre-registered
  test shows a method beating Uniform; SEC and DUMP over-fund a broken reward in all 9 runs each (33% and 31%
  against 20%) while the Curator variants keep it to 6-10% on average; the reward checks flagged it in 21 of 22 Curator
  runs and never flagged a real environment; calibration can be dropped (R4); the Curator variants put
  1.22-1.31 of about 1.5 GPU-hours into environments that can teach, against 0.85-0.89 for Uniform, SEC and DUMP.
- **Claim (session B):** SEC still over-funds the broken reward at the authors' own setting (25%, simulator
  predicted 26%); perfect junk removal ties Uniform; the pre-flight check catches a realistic buggy checker
  before training and Curator keeps it at 5.7% of questions against 20%.
- **Promising, not confirmed (D-108 replicates):** on fresh seeds 3-4, Curator-Q ended above Uniform in 6 of 6
  runs (+1.36 points), overtaking it after about 0.8 GPU-hours. It tied Uniform on seeds 0-2, so present it as
  a signal that needs a pre-registered confirmation, never as "Curator beats Uniform".
- **Claim as a trend, not a result:** SEC and DUMP scored below Uniform in every D-103 seed (about -1 point),
  below the pre-registered bar; DUMP was also behind at matched compute in 5 of 6 D-108 runs.
- **Claim as simulation only:** Curator +11.6% over Uniform against +1.2% for SEC and DUMP (tuning seeds);
  the budget behaviour transferred to GPU almost exactly, the accuracy gain did not.
- **Do not claim:** Curator beats Uniform; measured compute savings (the 0.39-0.45 GPU-hours are estimates
  from learning curves); gains from cost-awareness on GPU; detection by calibration; leave-one-out or
  sealed-test results (not run).

## Papers we compared against (`../papers/`)

DataFlex-RL (2026, the "nothing beats Uniform" evaluation), SEC (COLM 2026), DUMP (2025), Graves et al.
(ICML 2017) and TSCL (the LP baseline). Full citations, arXiv links and how each is used:
`../papers/README.md`. The PDFs are kept locally, not in git (authors' licences); fetch them from the links.
Quote to use for DataFlex-RL: "None of the eight selection or reweighting methods has a paired 95%
confidence interval excluding zero relative to uniform sampling."

## Pending

- **D-108 primary:** seed 5, run fresh in namespace `compute_s5_final` (the first attempt was never backed up, so it cannot resume; D-108 record of 3 Nov).
- **Optional:** a pre-registered confirmation of Curator-Q's final-accuracy signal on fresh seeds (6+).

## Living documents (linked, not copied)

`../curator-rl/README.md` (project status), `../curator-rl/docs/DECISIONS.md` (decision log),
`../viva.md` (viva notes), `../report.md` (full project report), `../errors_found.md` (audit).

## Regenerate

From `curator-rl/`:

```
.venv/Scripts/python experiments/analysis/report_figures.py --runs ../curator_results_november/runs --olmo-runs ../curator_results_olmo/runs --out "../Diagram_&_Documentation/figures"
.venv/Scripts/python experiments/analysis/d103_confirm.py --runs ../curator_results_november/runs --runs sec_tau1=../Curator-Nov-results/raw/sec_tau1/curator_results/runs --out ../Curator-Nov-results/analysis
.venv/Scripts/python experiments/analysis/compute_efficiency.py --runs acct3=../Curator-Nov-results/raw/compute_s3/curator_results/runs --runs acct4=../Curator-Nov-results/raw/compute_s4/curator_results/runs --runs acct5=../Curator-Nov-results/raw/compute_s5/curator_results/runs --base-runs ../curator_results_november/runs --methods curator_q,curator_nocal,sec,dump --expect-negative sec,dump --seeds-needed 3 --keep-blocks acct3:3,acct4:4,acct5:5 --out ../Curator-Nov-results/compute_analysis/primary
```

For the replicates, run the last command with `--no-verdict` instead of `--seeds-needed 3 --keep-blocks ...` and
`--out ../Curator-Nov-results/compute_analysis/replicates`. Then copy the outputs into `analysis/` and
`figures/compute/` here.
