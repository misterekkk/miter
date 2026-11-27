# Training speed: seconds for 100 trees

- Machine: Apple M4, 10 threads; Darwin 25.6.0
- miter 0.1.0.dev0 with the code of commit `1a3683f` (the v1.0 default; run before that commit was made); 100 trees, no early stopping; median of 3 run(s).
- Data: synthetic regression (Gaussian features, oblique signal) and covertype (binary).
- Time of `fit` for 100 trees, including dataset construction. Runs above 200k rows: one run.

## 1 thread

| data | LightGBM (31 leaves) | CatBoost (depth 6) | miter default | miter fast | miter leaf-wise | miter + auto quantizer + CTR |
|---|---:|---:|---:|---:|---:|---:|
| 20k x 16 | 0.13 | 0.31 | 0.36 (2.7× / 1.17×) | 0.22 (1.7× / 0.73×) | 0.28 (2.1× / 0.91×) | 0.36 (2.7× / 1.17×) |
| 20k x 100 | 0.84 | 1.44 | 1.26 (1.5× / 0.87×) | 0.91 (1.1× / 0.63×) | 1.21 (1.4× / 0.84×) | 1.25 (1.5× / 0.87×) |
| 100k x 50 | 1.33 | 1.70 | 3.41 (2.6× / 2.01×) | 2.11 (1.6× / 1.24×) | 2.74 (2.1× / 1.61×) | 3.49 (2.6× / 2.06×) |
| 500k x 50 | 6.29 | 6.13 | 19.79 (3.1× / 3.23×) | 10.04 (1.6× / 1.64×) | 15.59 (2.5× / 2.55×) | 19.63 (3.1× / 3.21×) |
| covertype 581k x 54 | 1.63 | 8.20 | 57.03 (35.0× / 6.96×) | 29.70 (18.2× / 3.62×) | 49.34 (30.3× / 6.02×) | 55.58 (34.1× / 6.78×) |

miter cells: seconds (× LightGBM / × CatBoost).

## 10 threads

| data | LightGBM (31 leaves) | CatBoost (depth 6) | miter default | miter fast | miter leaf-wise | miter + auto quantizer + CTR |
|---|---:|---:|---:|---:|---:|---:|
| 20k x 16 | 0.37 | 0.13 | 0.17 (0.5× / 1.40×) | 0.12 (0.3× / 0.95×) | 0.28 (0.8× / 2.28×) | 0.17 (0.5× / 1.38×) |
| 20k x 100 | 0.61 | 0.33 | 0.34 (0.6× / 1.04×) | 0.25 (0.4× / 0.77×) | 0.84 (1.4× / 2.55×) | 0.35 (0.6× / 1.06×) |
| 100k x 50 | 0.73 | 0.43 | 0.83 (1.1× / 1.91×) | 0.51 (0.7× / 1.17×) | 1.34 (1.8× / 3.09×) | 0.83 (1.1× / 1.91×) |
| 500k x 50 | 1.91 | 1.45 | 4.16 (2.2× / 2.87×) | 2.32 (1.2× / 1.61×) | 5.72 (3.0× / 3.95×) | 4.26 (2.2× / 2.94×) |
| covertype 581k x 54 | 0.84 | 2.02 | 8.81 (10.5× / 4.36×) | 4.59 (5.5× / 2.28×) | 10.34 (12.3× / 5.12×) | 8.64 (10.3× / 4.28×) |

miter cells: seconds (× LightGBM / × CatBoost).

