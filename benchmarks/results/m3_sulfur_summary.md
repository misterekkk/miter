# M3 Sulfur Multi-Seed Check (OpenML 44145)

- Runs: 5 folds x 5 seeds = 25 paired runs (elapsed: 187.9s)
- `miter` R_uni (C++ Q16r int): mean nRMSE = 0.3711, median = 0.3613, std = 0.0475
- `histree` R_uni (float): mean nRMSE = 0.3678, median = 0.3703, std = 0.0382
- `histree` R_uni (Q16r float): mean nRMSE = 0.3553, median = 0.3677, std = 0.0466
- `LightGBM` (axis): mean nRMSE = 0.4454, median = 0.4444, std = 0.0185

| comparison | mean rel diff (%) | median rel diff (%) | wins (x/25) | Wilcoxon p |
|---|---|---|---|---|
| miter int vs histree float | +1.10% | +1.66% | 11/25 | 0.615 |
| miter int vs histree Q16r float | +5.52% | +2.01% | 8/25 | 0.08514 |
| miter int vs LightGBM | -16.52% | -14.65% | 25/25 | 5.96e-08 |

(Negative relative diff = lower error than reference.)
