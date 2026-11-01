# Curator-Nov-results: the D-103 confirmation runs

This folder holds everything from the November confirmation runs: the downloaded raw results and the
analysis built from them. The rules being tested were fixed on 21 October 2026, before any of these runs
existed (`curator-rl/docs/DECISIONS.md`, D-103 and its record of 1 November).

## What is being tested

| Rule | Question | Holds if |
|---|---|---|
| R1 (primary) | Does Curator + question filter beat Uniform + question filter? | all 3 seeds positive and mean >= 2 SE |
| R2 | Do SEC (COLM 2026) and DUMP lose to Uniform when a broken reward is present? | for each: all 3 seeds negative and mean >= 2 SE |
| R3 | Do SEC and DUMP over-fund the broken reward, and Curator not? | SEC, DUMP above 20% of questions; Curator variants under 13% |
| R4 | Is Curator without calibration as good as full Curator? | mean difference within +-1 point |

Also reported, not part of the rules: `clean_uniform` (Uniform without the two junk environments, the best
a perfect junk detector could reach) and the declared check `sec_tau1` (SEC at the authors' default
temperature 1.0; see `curator-rl/reports/analysis/sec_dump_fidelity.md`).

Story decision (D-103): if R1 and R2 hold, the robustness result leads; if not, the project reports a
rigorous negative result with its mechanism. No change of story after the analysis day.

## Steps

1. **Build the Kaggle package** (on your PC, from `curator-rl/`):
   `.venv/Scripts/python scripts/make_kaggle_gpu_bundle.py`. Upload `dist/curator-gpu-kaggle.zip` as a new
   version of your Kaggle dataset. In the Kaggle notebook, use the updated `curator_gpu.ipynb` from the zip
   (File -> Import notebook), or add this line to the `COMMANDS` of cell 6 by hand:
   `"sec_tau1":  "compare --methods sec --seeds 0 --set baselines.sec.tau=1.0 --namespace sec_tau1 --no-base",`
2. **Session A** (about 10 h): accelerator **GPU T4 x2**, Internet on, `HF_TOKEN` secret ticked.
   In cell 1 set `STEPS = ["confirm", "lite"]`, then Save Version -> Save & Run All.
   If it stops early (exit code 75), run the same `STEPS` again: every run resumes from its checkpoint.
3. **Session B** (about 3.5 h): `STEPS = ["clean", "sec_tau1"]`.
4. **Download** from the notebook's Output panel (or the Hugging Face backup):
   - `results/curator_results.zip` -> unzip into `Curator-Nov-results/raw/main/`
   - `results_sec_tau1/curator_results.zip` -> unzip into `Curator-Nov-results/raw/sec_tau1/`

   Each should end up with a `runs/` folder inside (for example `raw/main/runs/sec_s1/summary.json`).
   The main zip contains every run, old and new, because each session restores the earlier runs from the
   backup first.
5. **Analyse** (from `curator-rl/`):

   ```
   .venv/Scripts/python experiments/analysis/d103_confirm.py --raw ../Curator-Nov-results/raw --out ../Curator-Nov-results/analysis
   ```

## What you get

| Path | Content |
|---|---|
| `analysis/d103_report.md` | verdict for R1-R4 with every number, all methods against Uniform, the declared checks |
| `analysis/figures/accuracy_vs_uniform.png` | every method against Uniform (filled = 3-seed interval) |
| `analysis/figures/noisy_arm_share.png` | budget on the broken reward, with seed dots |
| `analysis/figures/noisy_share_by_round.png` | the same over training rounds: who gets pulled in, and when |
| `analysis/figures/rules_per_seed.png` | R1 and R2 seed by seed |
| `analysis/tables/*.csv` | per-seed paired differences, rule verdicts, budget shares |
| `raw/main/comparison.md`, `raw/main/plots/` | the runner's own comparison and per-run mixture plots |

## Budget

About 13.5 GPU-hours of session time in total (Kaggle's weekly quota is 30 h; whether a T4 x2 session
counts double is unverified, so check the quota page before session B).
