# Curator-Nov-results: the D-103 confirmation runs

The rules tested here were fixed on 21 October 2026, before any of these runs existed
(`curator-rl/docs/DECISIONS.md`, D-103 and its records of 1 and 2 November).

## Status (2 November 2026)

| Session | Steps | State | Data |
|---|---|---|---|
| A | `confirm`, `lite` | **done, analysed** | `../curator_results_november/` (every run, old and new) |
| B | `clean`, `sec_tau1` | to run (about 3.5 GPU-hours) | - |

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
   `curator-gpu-kaggle` dataset. Cell 1: `STEPS = ["clean", "sec_tau1"]`, then Save Version -> Save & Run All.
2. Download from the Output panel:
   - `results/curator_results.zip` -> unzip its `curator_results/` contents over `../curator_results_november/`
     (it contains every run, so it replaces the session A copy);
   - `results_sec_tau1/curator_results.zip` -> unzip into `raw/sec_tau1/`.
3. From `curator-rl/`:

   ```
   .venv/Scripts/python experiments/analysis/d103_confirm.py --runs ../curator_results_november/runs --runs sec_tau1=../Curator-Nov-results/raw/sec_tau1/curator_results/runs --out ../Curator-Nov-results/analysis
   .venv/Scripts/python experiments/analysis/report_figures.py --runs ../curator_results_november/runs --olmo-runs ../curator_results_olmo/runs --out "../Diagram_&_Documentation/figures"
   ```

   Then copy `analysis/` over `../Diagram_&_Documentation/analysis/d103_results/`.

## What is here

| Path | Content |
|---|---|
| `analysis/d103_report.md` | verdict for R1-R4 with every number, all methods against Uniform |
| `analysis/figures/` | accuracy against Uniform, budget on the broken reward, the same by round, R1/R2 seed by seed |
| `analysis/tables/*.csv` | per-seed paired differences, rule verdicts, budget shares |
| `raw/` | where the session B `sec_tau1` download goes |
