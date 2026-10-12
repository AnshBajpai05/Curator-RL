# Learning-progress window length (improvement B3; realism v2, tuning seeds 0-39)

Curator at the Kaggle preset `gpu_tuned`; relative gain of the noiseless final score over Uniform (95% t-interval); last column: absolute difference to W = 10 on S-I-R1-GPU2.

| window | S-I-R1-GPU2 gain over Uniform | S-I-R2-GPU2 gain over Uniform | S-J-R-GPU2 gain over Uniform | minus W10 (S-I-R1) |
|---|---|---|---|---|
| 10 | +0.1193 [+0.1040, +0.1346] | +0.1120 [+0.0939, +0.1302] | +0.1185 [+0.1016, +0.1354] | +0.0000 [+0.0000, +0.0000] |
| 15 | +0.1158 [+0.0977, +0.1340] | +0.1143 [+0.0948, +0.1339] | +0.1237 [+0.1075, +0.1399] | -0.0007 [-0.0032, +0.0018] |
| 20 | +0.1114 [+0.0915, +0.1314] | +0.1143 [+0.0934, +0.1352] | +0.1302 [+0.1108, +0.1495] | -0.0016 [-0.0049, +0.0017] |
| 30 | +0.1006 [+0.0790, +0.1221] | +0.1134 [+0.0901, +0.1366] | +0.1218 [+0.1033, +0.1402] | -0.0039 [-0.0074, -0.0003] |

No window beats W = 10 consistently; W = 30 is worse on S-I-R1-GPU2. W stays 10.
