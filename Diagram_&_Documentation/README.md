# Diagram & Documentation: everything the final report needs

State of 2 November 2026: D-103 sessions A and B are in (every main comparison has 3 seeds; `clean`,
`sec_tau1` and `stress` are analysed). The D-108 compute test is running (see "In progress"). Every number here comes from the run files or a committed
report; nothing is typed by hand.

## Headline (D-103, decided by rules written before the runs)

| Rule | Verdict | Numbers (sampled accuracy, item-paired, percentage points) |
|---|---|---|
| R1: Curator + question filter beats Uniform + filter | **fails** | +1.8, -0.2, +0.3 by seed; mean +0.65 (1.1 SE) |
| R2: SEC and DUMP lose to Uniform | **fails** | SEC -2.3, -0.3, -0.2 (mean -1.0, 1.4 SE); DUMP -1.5, -0.6, -0.2 (mean -0.8, 1.9 SE). Every seed points the predicted way, none clears the 2-SE bar |
| R3: SEC and DUMP over-fund the broken reward, Curator does not | **holds** | SEC 33-34% and DUMP 31% in every seed (Uniform 20%); Curator variants 6-10% on average (single seeds up to 16%) |
| R4: Curator without calibration is as good as Curator | **holds** | +0.15 points (within +-1) |

By D-103's own rule (R1 and R2 did not both hold) the story is the fallback: a rigorous negative result
on accuracy, the mechanism (learning-signal curricula are drawn into a broken reward), and the
pre-flight check and in-run reward audit as the contribution. Full report: `analysis/d103_results/`.

**Session B** (descriptive, plus the pre-registered D-102 H-C): SEC at the authors' default temperature still over-funds the broken reward (25%, simulator predicted 26%; tuned SEC 34%, Uniform 20%) and scores -1.1 against Uniform (1 seed); perfect junk removal (`clean_uniform`) ties Uniform (-0.00, 2 seeds); with a realistic buggy checker the pre-flight check flags it before training (junk reward 0.20) and Curator-Q caps it at 5.7% from round 1 against Uniform's 20%, but accuracy is +0.65 points (0.86 SE): D-102 H-C fails.

## Diagrams (open in a browser; the sources are in `source/`)

| File | What it shows | Use in the report |
|---|---|---|
| `01-project-overview.html` | budget and environments in, the scheduler sets the mix, GRPO trains, reward checks run alongside | introduction |
| `02-system-design.html` | every component: prompt stream, TRL trainer, reward router, sandbox, cost meter, signal engine, scheduler, calibrator, pre-flight check, checkpoints, simulator | system design |
| `03-one-round-game-changers.html` | one scheduling round, standard steps marked apart from ours (status rules, reward audit) | method |
| `04-status-lifecycle.html` | environment statuses S1-S5 and what moves an environment between them | method |
| `05-reward-health-checks.html` | pre-flight check, reward audit and calibration, with the 3-seed GPU results | method, results |
| `06-real-run-walkthrough.html` | Curator seed 0 round by round from the real log: warm-up, Countdown cut at round 7, broken reward caught at round 8 | results |

## Figures (`figures/`, 3 seeds per main method)

| File | Finding | Status |
|---|---|---|
| `gain_over_untrained.png` | training works for every method: +4.6 to +5.7 points of sampled accuracy over the untrained model | final |
| `accuracy_vs_uniform.png` | no method beats Uniform: Curator -0.1 [-1.2, +1.0], LP +0.2, Curator-lite 0.0, DUMP -0.8, SEC -1.0 (95% intervals all cross 0) | final |
| `noisy_arm_share.png` | budget on the broken reward: SEC 34%, DUMP 31%, Uniform 20%, LP 13%, Curator 10%, Curator-lite 9%, Curator-Q 6%, clean Uniform 0% | final |
| `noisy_share_by_round.png` | SEC climbs to about 42% by round 32, DUMP holds near 31-34%, Curator falls once the audit flags the arm | final |
| `no_signal_groups.png` | answer groups that teach nothing on the real environments: Curator 44-48%, LP 49%, SEC 53%, DUMP 54%, Uniform 59% | final |
| `cost_per_environment.png` | MATH35 costs 2.67x GSM8K to generate but only 1.22-1.58x per full training step | final |
| `gpu_split_by_environment.png` | where the 1.5 GPU-hours went: useful environments get 1.27-1.31 under the Curator variants, 1.17 under LP, 0.85-0.89 under Uniform, SEC and DUMP | final (fresh seeds add a check after D-108) |
| `mixtures/*.png` | per-run mixture over rounds; `sec_s0`-`sec_s2` and `dump_s0`-`dump_s2` show the over-funding repeating in every seed, `curator_s0` the cut at round 8 | final |
| `mixtures/olmo_*.png` | OLMo-2-1B pair (1 seed, descriptive) | final |

## D-103 results (`analysis/d103_results/`)

`d103_report.md` (every rule with per-seed numbers, all methods against Uniform), `figures/` (including
`rules_per_seed.png`, the R1 and R2 seeds), `tables/*.csv`. Produced by
`curator-rl/experiments/analysis/d103_confirm.py`, written and recorded in the decision log before the runs.

## Reports (`reports/`)

| File | Source | Supports |
|---|---|---|
| `gpu_comparison_qwen.md` | `curator_results_november/comparison.md` (2 Nov) | every GPU number for Qwen2.5-0.5B, all seeds |
| `gpu_comparison_olmo.md` | `curator_results_olmo/comparison.md` | the OLMo-2-1B pair |
| `gpu_stress_hackable_checker.md` | `Curator-Nov-results/raw/stress/curator_results/comparison.md` | the realistic buggy checker (D-102 H-C) |
| `sysid_seed0.md`, `sysid_seed1.md` | `curator_results_new/runs/dirichlet_s*/sysid_report.md` | measured training cost per environment, group polarisation, transfer |
| `cost_probe_generation.md` | `curator-rl/reports/pilot/phase_e_cost_probe.md` | generation-only cost spread (2.67x) |
| `fair_tuning_gpu2.md` | `curator-rl/reports/sweeps/tune_gpu2.md` | every method got 8 configurations under the same rule |
| `sec_dump_fidelity.md` | `curator-rl/reports/analysis/sec_dump_fidelity.md` | our SEC and DUMP against the authors' code |
| `sim_component_study_gpu2.md` | `curator-rl/reports/analysis/component_study_gpu2.md` | the reward audit is the component that matters (simulator) |
| `sim_heldout_gpu2.md` | `curator-rl/reports/analysis/heldout_gpu2.md` | simulator gains (+11-12% over Uniform) that did not transfer to GPU |
| `sim_audit_impact.md` | `curator-rl/reports/analysis/audit_impact.md` | why calibration cannot detect the arm at the GPU shape |
| `self_critic.md` | `../self_critic.md` | why no method beats Uniform at this scale |

## What the evidence supports

- **Claim (3 seeds):** training works for every method (+4.6 to +5.7 points); Curator matches Uniform;
  SEC and DUMP over-fund a broken reward in every seed (34% and 31% against 20%) while Curator keeps it to
  6-10%; the label-free audit flagged that reward in all 10 Curator runs and never a real environment;
  calibration can be dropped (R4); of about 1.5 GPU-hours, the Curator variants spend 1.27-1.31 on
  environments that can teach, LP 1.17, Uniform, SEC and DUMP 0.85-0.89 (descriptive,
  `figures/gpu_split_by_environment.png`, seeds 0-2).
- **Claim (session B):** SEC still over-funds the broken reward at the authors' own setting (25%,
  simulator predicted 26%); perfect junk removal ties Uniform; the pre-flight check catches a realistic
  buggy checker before training and Curator keeps it at 5.7% of questions against 20%.
- **Claim as a trend, not a result:** SEC and DUMP scored below Uniform in every seed (about -1 point),
  below the pre-registered 2-SE bar; Curator is about 0.6-0.9 points ahead of both, untested by any rule.
- **Claim as simulation only:** Curator +11.6% over Uniform against +1.2% for SEC and DUMP (tuning seeds);
  the budget behaviour transferred to GPU, the accuracy gain did not.
- **Hypothesis, not a claim:** the budget gap matters once a broken reward actually harms training or runs
  get longer. The `hackable` checker did not harm accuracy here either (D-102 H-C fails); OLMo and longer
  runs remain untested.
- **Do not claim:** Curator beats Uniform, SEC or DUMP on accuracy; compute savings; gains from
  cost-awareness on GPU; detection by calibration; leave-one-out or sealed-test results (not run).

## Papers we compared against (`../papers/`)

DataFlex-RL (2026, the "nothing beats Uniform" evaluation), SEC (COLM 2026), DUMP (2025), Graves et al.
(ICML 2017) and TSCL (the LP baseline). Full citations, arXiv links and how each is used:
`../papers/README.md`. The PDFs are kept locally, not in git (authors' licences); fetch them from the links.
Quote to use for DataFlex-RL: "None of the eight selection or reweighting methods has a paired 95%
confidence interval excluding zero relative to uniform sampling."

## In progress (do not put these numbers in the report yet)

- **D-108 compute test** (pre-registered 2 Nov): Uniform, Curator-Q, Curator-lite, SEC and DUMP on fresh
  seeds 3-5, evaluated every 3 rounds, one team account per seed. Question: does a method reach accuracy
  with less GPU than Uniform? Analysis: `curator-rl/experiments/analysis/compute_efficiency.py`.
When it is in, the figures, `analysis/` and this index are regenerated (commands below).

## Living documents (linked, not copied)

`../curator-rl/README.md` (project status), `../curator-rl/docs/DECISIONS.md` (decision log),
`../viva.md` (viva notes), `../report.md` (full project report), `../errors_found.md` (audit).

## Regenerate (after session B)

Unzip the new download as described in `../Curator-Nov-results/README.md` (main zip over
`../curator_results_november/`, the `sec_tau1` zip into `../Curator-Nov-results/raw/sec_tau1/`), then from
`curator-rl/`:

```
.venv/Scripts/python experiments/analysis/report_figures.py --runs ../curator_results_november/runs --olmo-runs ../curator_results_olmo/runs --out "../Diagram_&_Documentation/figures"
.venv/Scripts/python experiments/analysis/d103_confirm.py --runs ../curator_results_november/runs --runs sec_tau1=../Curator-Nov-results/raw/sec_tau1/curator_results/runs --out ../Curator-Nov-results/analysis
```

Copy `../Curator-Nov-results/analysis/` (report, figures, tables) over `analysis/d103_results/` and the new
`comparison.md` over `reports/gpu_comparison_qwen.md`.
