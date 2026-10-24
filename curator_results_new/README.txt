CURATOR results (2026-10-21 10:00 UTC)

comparison.md   sampled + greedy scores, item-by-item differences, gain over the untrained model,
                GPU-hours to reach Uniform's level, ROI leaderboards, leave-one-out
status.md       every run: status, round, GPU-hours used, last checkpoint, last sync
plots/          <run>_mixture.png (share of each arm per round, S5 marked), score_vs_gpu_hours.png
                (learning curves on the same items)
runs/<run>/     summary.json (final scores, shares, S5 arms, errors), progress.json (heartbeat),
                run_config.json, rounds.jsonl / steps.jsonl / calibration.jsonl / reports.jsonl (logs),
                sysid_report.md (system identification runs only)
logs/           console log of every training process
curator_results.zip  everything above in one file

Checkpoints are not included (they stay in curator_runs/<run>/checkpoints, newest 3 per run).
