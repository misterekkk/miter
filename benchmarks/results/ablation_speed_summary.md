# Speed: seconds for 100 trees (median of 3)

| data | threads | BASE | SP_NONE | SP_E2_ONLY | SP_EFB_ONLY |
|---|---:|---:|---:|---:|---:|
| covertype 581k x 54 | 1 | 10.41 | 55.97 (5.38x) | 22.51 (2.16x) | 44.93 (4.32x) |
| covertype 581k x 54 | 10 | 2.75 | 9.20 (3.35x) | 4.22 (1.54x) | 7.57 (2.76x) |
| synthetic 500k x 50 | 1 | 8.18 | 19.15 (2.34x) | 8.42 (1.03x) | 19.48 (2.38x) |
| synthetic 500k x 50 | 10 | 2.10 | 4.44 (2.12x) | 2.39 (1.14x) | 4.58 (2.18x) |
