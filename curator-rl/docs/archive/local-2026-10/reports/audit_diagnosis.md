# Diagnosis after D-98 Stage T (tuning seeds 0-49, exploratory)

Noisy arm shares the GSM8K slice. Reasons are taken from the S5 note of each entry.

| arm | noisy detected | GSM8K flagged by gain | GSM8K flagged by audit | false-flag rate | score | noisy share | GSM8K share |
|---|---|---|---|---|---|---|---|
| gain_joint (no audit) | 0.153 | 0.040 | 0.000 | 0.035 | 0.3194 | 0.229 | 0.283 |
| cand_joint (gain + audit) | 0.980 | 0.100 | 0.000 | 0.055 | 0.3419 | 0.086 | 0.332 |
| audit_only (gain flags off) | 0.973 | 0.000 | 0.000 | 0.000 | 0.3425 | 0.086 | 0.336 |
