# Curator-Nov-results: the D-103 confirmation runs

The rules tested here were fixed on 21 October 2026, before any of these runs existed
(`curator-rl/docs/DECISIONS.md`, D-103 and its records of 1 and 2 November).

## Status (2 November 2026)

| Session | Steps | State | Data |
|---|---|---|---|
| A | `confirm`, `lite` | **done, analysed** | `../curator_results_november/` (every run, old and new) |
| B | `clean`, `sec_tau1`, `stress` | **done, analysed**: clean ties Uniform; SEC at the authors' default 25% on the broken reward; D-102 H-C fails | `../curator_results_november/`, `raw/sec_tau1/`, `raw/stress/` |
| C (D-108) | each account ran seeds 3-5 (not one seed each); seeds 3-4 finished on all three, seed 5 cut by the 12-hour limit | **analysed**: primary PENDING (seed 5); Curator-Q +1.36 final accuracy in 6 of 6 replicate runs (descriptive) | `raw/compute_s3/`, `raw/compute_s4/`, `raw/compute_s5/`; analysis `compute_analysis/primary/`, `compute_analysis/replicates/` |

**Outcome of session A** (`analysis/d103_report.md`):

| Rule | Question | Verdict |
|---|---|---|
| R1 (primary) | Curator + question filter beats Uniform + filter? | **fails** (+0.65 points, 1.1 SE) |
| R2 | SEC and DUMP lose to Uniform? | **fails** (SEC -1.0, DUMP -0.8; every seed negative, below 2 SE) |
| R3 | SEC and DUMP over-fund the broken reward, Curator not? | **holds** (SEC 34%, DUMP 31%, Uniform 20%, Curator 6-10%) |
| R4 | Curator without calibration as good as Curator? | **holds** (+0.15 points) |

By D-103's own rule the story is the fallback: a rigorous negative result on accuracy with its
mechanism, plus the pre-flight check and the in-run reward audit as the contribution. Session B is
descriptive and cannot change this.

## Session B steps

1. Kaggle notebook, accelerator **GPU T4 x2**, Internet on, `HF_TOKEN` ticked, latest
   `curator-gpu-kaggle` dataset. Cell 1: `STEPS = ["clean", "sec_tau1", "stress"]`, then Save Version -> Save & Run All.
2. Download from the Output panel:
   - `results/curator_results.zip` -> unzip its `curator_results/` contents over `../curator_results_november/`
     (it contains every run, so it replaces the session A copy);
   - `results_sec_tau1/curator_results.zip` -> unzip into `raw/sec_tau1/`;
   - `results_stress/curator_results.zip` -> unzip into `raw/stress/` (its `comparison.md` answers D-102 H-C).
3. From `curator-rl/`:

   ```
   .venv/Scripts/python experiments/analysis/d103_confirm.py --runs ../curator_results_november/runs --runs sec_tau1=../Curator-Nov-results/raw/sec_tau1/curator_results/runs --out ../Curator-Nov-results/analysis
   .venv/Scripts/python experiments/analysis/report_figures.py --runs ../curator_results_november/runs --olmo-runs ../curator_results_olmo/runs --out "../Diagram_&_Documentation/figures"
   ```

   Then copy `analysis/` over `../Diagram_&_Documentation/analysis/d103_results/`.

## Session C (D-108, pre-registered compute test)

Question: does a method reach accuracy with less GPU than Uniform? Primary: Curator-Q ahead of Uniform at
matched GPU-hours (all 3 seeds positive and mean >= 2 SE); secondary: Curator-lite ahead, SEC and DUMP behind.

Split by seed, one team account each (about 7.5 h, all in parallel):

| Account | Cell 1 `STEPS` | Download -> unzip into |
|---|---|---|
| teammate 1 | `["compute_s3"]` | `results_compute/curator_results.zip` -> `raw/compute_s3/` |
| teammate 2 | `["compute_s4"]` | `results_compute/curator_results.zip` -> `raw/compute_s4/` |
| teammate 3 | `["compute_s5"]` | `results_compute/curator_results.zip` -> `raw/compute_s5/` |

Each account: import the latest `curator_gpu.ipynb` from the `curator-gpu-kaggle` dataset (shared with them),
GPU T4 x2 only, Internet on, their own `HF_USER` and `HF_TOKEN` (or a fine-grained token limited to
`datasets/AnshBajpai05/curator-runs`). Never mix `compute` with `compute_s3/4/5`: seeds would run twice.

Then from `curator-rl/` (the analysis fixed in D-108, given the three folders):

```
.venv/Scripts/python experiments/analysis/compute_efficiency.py --runs ../Curator-Nov-results/raw/compute_s3/curator_results/runs --runs ../Curator-Nov-results/raw/compute_s4/curator_results/runs --runs ../Curator-Nov-results/raw/compute_s5/curator_results/runs --base-runs ../curator_results_november/runs --methods curator_q,curator_nocal,sec,dump --expect-negative sec,dump --seeds-needed 3 --out ../Curator-Nov-results/compute_analysis
```

The final accuracy of the five methods on each seed is in the `comparison.md` inside each zip.

## What is here

| Path | Content |
|---|---|
| `analysis/d103_report.md` | verdict for R1-R4 with every number, all methods against Uniform |
| `analysis/figures/` | accuracy against Uniform, budget on the broken reward, the same by round, R1/R2 seed by seed |
| `analysis/tables/*.csv` | per-seed paired differences, rule verdicts, budget shares |
| `raw/` | where the `sec_tau1`, `stress` and `compute` downloads go |
