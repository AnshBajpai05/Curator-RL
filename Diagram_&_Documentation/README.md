# Diagram & Documentation: everything the final report needs

Snapshot of 1 November 2026, before the D-103 confirmation runs. Every number here comes from the run
files or a committed report; nothing is typed by hand. Items marked **update after D-103** change once
the confirmation runs are in (see "Regenerate" at the end).

## Diagrams (open in a browser; the sources are in `source/`)

| File | What it shows | Use in the report |
|---|---|---|
| `01-project-overview.html` | budget and environments in, the scheduler sets the mix, GRPO trains, reward checks run alongside | introduction |
| `02-system-design.html` | every component: prompt stream, TRL trainer, reward router, sandbox, cost meter, signal engine, scheduler, calibrator, pre-flight check, checkpoints, simulator | system design |
| `03-one-round-game-changers.html` | one scheduling round, standard steps marked apart from ours (status rules, reward audit) | method |
| `04-status-lifecycle.html` | environment statuses S1-S5 and what moves an environment between them | method |
| `05-reward-health-checks.html` | pre-flight check, reward audit and calibration, with GPU results | method, results |
| `06-real-run-walkthrough.html` | Curator seed 0 round by round from the real log: warm-up, Countdown cut at round 7, broken reward caught at round 8 | results |

## Figures (`figures/`)

| File | Finding | Status |
|---|---|---|
| `gain_over_untrained.png` | training works for every method: +4.2 to +6.2 points of sampled accuracy over the untrained model | update after D-103 |
| `accuracy_vs_uniform.png` | Curator ties Uniform (-0.13 points, 95% interval [-1.22, +0.96], 3 seeds); SEC -2.3, DUMP -1.5 on seed 0 | **update after D-103** |
| `noisy_arm_share.png` | budget on the random-reward arm: SEC 34%, DUMP 31%, Uniform 20%, LP 13%, Curator 10% (seeds 6, 16, 7) | **update after D-103** |
| `noisy_share_by_round.png` | the same over rounds: SEC climbs to 44%, DUMP stays near 31%, Curator falls once the audit flags the arm | **update after D-103** |
| `no_signal_groups.png` | answer groups that teach nothing on the real environments: Curator 46%, Uniform 59% (3 seeds each) | update after D-103 |
| `cost_per_environment.png` | MATH35 costs 2.67x GSM8K to generate but only 1.22-1.58x per full training step | final |
| `mixtures/*.png` | per-run mixture over rounds (the runner's own plots); `curator_s0` next to `sec_s0` is the clearest pair | add new seeds after D-103 |
| `mixtures/olmo_*.png` | OLMo-2-1B pair (1 seed, descriptive) | final |

## Reports (`reports/`, copied from their sources on 1 Nov)

| File | Source | Supports |
|---|---|---|
| `gpu_comparison_qwen.md` | `curator_results_new/comparison.md` | every GPU number for Qwen2.5-0.5B |
| `gpu_comparison_olmo.md` | `curator_results_olmo/comparison.md` | the OLMo-2-1B pair |
| `sysid_seed0.md`, `sysid_seed1.md` | `curator_results_new/runs/dirichlet_s*/sysid_report.md` | measured training cost per environment, group polarisation, transfer |
| `cost_probe_generation.md` | `curator-rl/reports/pilot/phase_e_cost_probe.md` | generation-only cost spread (2.67x) |
| `fair_tuning_gpu2.md` | `curator-rl/reports/sweeps/tune_gpu2.md` | every method got 8 configurations under the same rule |
| `sec_dump_fidelity.md` | `curator-rl/reports/analysis/sec_dump_fidelity.md` | our SEC and DUMP against the authors' code |
| `sim_component_study_gpu2.md` | `curator-rl/reports/analysis/component_study_gpu2.md` | the reward audit is the component that matters (simulator) |
| `sim_heldout_gpu2.md` | `curator-rl/reports/analysis/heldout_gpu2.md` | simulator gains on held-out scenarios (did not transfer to GPU) |
| `sim_audit_impact.md` | `curator-rl/reports/analysis/audit_impact.md` | why calibration cannot detect the arm at the GPU shape |
| `self_critic.md` | `../self_critic.md` | why no method beats Uniform at this scale |

## D-103 pre-run snapshot (`analysis/d103_prerun_snapshot/`)

The confirmation analysis run on today's data: R1, R2 and R4 are PENDING (seed 0 only), R3 holds
provisionally. After the runs, the real verdicts are written to `../Curator-Nov-results/analysis/`; replace
this folder with that output.

## What the evidence supports today

- **Claim:** training works (+4 to +6 points for every method); Curator matches Uniform; the label-free
  reward audit flags a reward that ignores the question in every Curator run on GPU (rounds 7-9 in five
  runs, 23 in one) and never a real environment; Curator moves budget off useless environments
  (Countdown 5% vs 20%).
- **Claim with "seed 0":** SEC and DUMP lose 2.3 and 1.5 points and spend 31-34% on the broken reward.
- **Do not claim:** Curator beats Uniform, saves compute, or gains from cost-awareness on GPU;
  calibration detects anything on GPU; leave-one-out or sealed-test results (not run).

## Living documents (linked, not copied)

`../curator-rl/README.md` (project status), `../curator-rl/docs/DECISIONS.md` (decision log),
`../viva.md` (viva notes), `../report.md` (full project report), `../errors_found.md` (audit).

## Regenerate

From `curator-rl/`, after unzipping the D-103 download as described in `../Curator-Nov-results/README.md`:

```
.venv/Scripts/python experiments/analysis/report_figures.py --runs ../Curator-Nov-results/raw/main/curator_results/runs --olmo-runs ../curator_results_olmo/runs --out "../Diagram_&_Documentation/figures"
.venv/Scripts/python experiments/analysis/d103_confirm.py --raw ../Curator-Nov-results/raw --out ../Curator-Nov-results/analysis
```

Then copy `../Curator-Nov-results/analysis/` over `analysis/d103_prerun_snapshot/` (renamed), and refresh
`reports/gpu_comparison_qwen.md` from the new `comparison.md`.
