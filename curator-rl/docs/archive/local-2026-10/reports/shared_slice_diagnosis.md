# Shared calibration slice: diagnosis on tuning seeds (D-97)

Tuning seeds 0..49; refit scenarios S-I-R1, S-I-R2, S-J-R; adopted Curator (claim_stale, 2 slices).

| variant | noisy detected | GSM8K flagged | false-flag rate (all other arms) | calibration cost | score | noisy budget share |
|---|---|---|---|---|---|---|
| own_private | 0.780 | 0.000 | 0.017 | 0.034 | 0.3258 | 0.188 |
| own_shared | 0.073 | 0.113 | 0.112 | 0.032 | 0.3179 | 0.242 |
| joint_shared | 0.153 | 0.040 | 0.035 | 0.032 | 0.3194 | 0.229 |

Per scenario (noisy detected / GSM8K flagged):

- own_private: S-I-R1 0.80 / 0.00, S-I-R2 0.76 / 0.00, S-J-R 0.78 / 0.00
- own_shared: S-I-R1 0.06 / 0.08, S-I-R2 0.08 / 0.16, S-J-R 0.08 / 0.10
- joint_shared: S-I-R1 0.16 / 0.02, S-I-R2 0.20 / 0.08, S-J-R 0.10 / 0.02
