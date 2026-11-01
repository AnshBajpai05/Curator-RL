# D-98 Stage T: audit candidate selection (tuning seeds 0-49)

Noisy arm shares the GSM8K slice; audit on (action s5); frozen rule in `docs/prereg/D98_reward_validity_audit.md` section 5.

| candidate | noisy detected | by gain | by audit | GSM8K flagged | false-flag rate | calib + audit cost | score | noisy share | eligible |
|---|---|---|---|---|---|---|---|---|---|
| cand_own | 0.987 | 0.020 | 0.980 | 0.167 | 0.133 | 0.032 | 0.3436 | 0.085 | False |
| cand_joint | 0.980 | 0.120 | 0.980 | 0.100 | 0.055 | 0.032 | 0.3419 | 0.086 | False |

**Chosen: none: no candidate is eligible, Stage E is not run (D-97 claim stands)**
