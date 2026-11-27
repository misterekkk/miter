<p align="center"><img src="docs/assets/logo.svg" width="112" alt="miter logo"></p>

<h1 align="center">miter</h1>

<p align="center">
  Gradient-boosted decision trees with <b>random oblique splits</b>,<br>
  deterministic to the bit and scikit-learn compatible.
</p>

<p align="center">
  <a href="https://github.com/misterekkk/miter/actions/workflows/tests.yml"><img src="https://github.com/misterekkk/miter/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/misterekkk/miter/actions/workflows/wheels.yml"><img src="https://github.com/misterekkk/miter/actions/workflows/wheels.yml/badge.svg" alt="Wheels"></a>
  <a href="https://misterekkk.github.io/miter/"><img src="https://github.com/misterekkk/miter/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a>
  <br>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License: Apache 2.0"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platforms: Linux, macOS, Windows"></a>
  <a href="https://misterekkk.github.io/miter/benchmarks/"><img src="https://img.shields.io/badge/TabArena-%231%20GBDT-d9480f" alt="TabArena: #1 GBDT"></a>
</p>

```bash
pip install miter
```

```python
from miter import MiterClassifier

clf = MiterClassifier().fit(X_train, y_train, eval_set=[(X_val, y_val)])
proba = clf.predict_proba(X_test)
```

Wheels for Linux (x86_64, aarch64), macOS (arm64) and Windows (x64); Python 3.10+.

## Why miter

- **Oblique splits.** Every node tries axis thresholds and, alongside them, a few random 2–3-feature
  combinations `w₁·x₁ + w₂·x₂ (+ w₃·x₃) ≥ t`. A diagonal boundary takes one split instead of a
  staircase. Directions are random, not optimised, so the extra cost stays small.
- **Accurate.** On TabArena (51 tasks), miter is ahead of LightGBM and XGBoost at every tuning
  level and level with CatBoost. On held-out OpenML data, its default also beats CatBoost's
  ([benchmarks](#benchmarks)).
- **Deterministic to the bit.** Splits are integer arithmetic on 16-bit feature codes, and the
  randomness is counter-based. The same seed gives the same model on any machine and thread count,
  and prediction follows exactly the path training took.
- **Two tree shapes.** Symmetric (oblivious) trees like CatBoost, which are the default, or
  leaf-wise trees like LightGBM (`grow_policy="leafwise"`), both with oblique splits.
- **Native categories and missing values.** LightGBM-style set splits and CatBoost-style ordered
  CTRs with feature combinations; NaN and unseen categories need no preprocessing.
- **Fits in.** scikit-learn estimators (they pass `check_estimator`), exact pickling and JSON
  models, and an AutoGluon / TabArena model. Predictions run on
  [lignum](https://github.com/misterekkk/lignum): 16 ns per row and tree on one Apple M4 core,
  compared with LightGBM's 19.

## Benchmarks

[TabArena](https://tabarena.ai) `lite`, all 51 tasks (regression, binary, multiclass), one split per
task:
- **Training:** only miter was trained. The other results are TabArena's published runs, with up to
  200 configurations each against miter's 25.
- **Paired comparison:** wins over the 51 tasks, median relative error (negative = miter better),
  and Wilcoxon p.

| miter vs | CatBoost | LightGBM | XGBoost |
|---|---|---|---|
| **default** | 24/51, +0.31%, p = 0.40 | 41/51, −3.23%, p < 0.001 | 37/51, −1.50%, p = 0.01 |
| **tuned** | 31/51, −0.44%, p = 0.44 | 39/51, −0.99%, p = 0.003 | 37/51, −1.11%, p = 0.003 |
| **tuned + ensemble** | 35/51, −0.78%, p = 0.14 | 37/51, −0.62%, p = 0.01 | 38/51, −1.27%, p < 0.001 |

- **Elo in the GBDT pool:**
  - tuned + ensemble: miter **1218**, LightGBM 1111, CatBoost 1081;
  - default: miter **998**, CatBoost 996, XGBoost 788, LightGBM 722.
- **LightGBM and XGBoost:** miter is ahead at every level.
- **CatBoost:** miter is level at every level (no difference is statistically significant).
- **Held-out data.** On 32 OpenML datasets with no overlap with TabArena (5-fold CV), the default
  beat CatBoost's default on 23 (median −1.6%, p = 0.005) and LightGBM's on 28 (median −7.0%).
  So the results generalise beyond the benchmark the configurations were developed on.

**Training time.** With early stopping on those 32 datasets (one thread), the default trained in
1.7 s per dataset (median):
- about 2.4× faster than CatBoost in total (3.3 s median);
- about 8× slower than LightGBM in total (0.2 s median).

At a fixed 100 trees (Apple M4, seconds):

| data | threads | LightGBM | CatBoost | miter | miter `fast` |
|---|---:|---:|---:|---:|---:|
| 20k × 100 | 1 | 0.84 | 1.44 | 1.26 | 0.91 |
| 20k × 100 | 10 | 0.61 | 0.33 | 0.34 | 0.25 |
| 500k × 50 | 1 | 6.3 | 6.1 | 19.8 | 10.0 |
| 500k × 50 | 10 | 1.9 | 1.5 | 4.2 | 2.3 |

About 70% of miter's training time goes into the oblique candidates; with axis splits alone it is as
fast as LightGBM. Data with many binary (one-hot) columns is the slow case: covertype (44 of 54
columns binary) trains 4–7× slower than CatBoost. Details and protocols:
**[misterekkk.github.io/miter/benchmarks](https://misterekkk.github.io/miter/benchmarks/)** and
[benchmarks/results/](benchmarks/results/).

## Documentation

**[misterekkk.github.io/miter](https://misterekkk.github.io/miter/)**:
- getting started and all parameters;
- categorical features, scikit-learn, saving models and TabArena;
- how oblique splits and determinism work;
- the Python reference.

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Contributing

Bug reports, datasets where miter does clearly worse than LightGBM or CatBoost, and benchmark
results are all welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains the setup and the rules every
change keeps:
- determinism;
- pre-registered accuracy claims;
- measured performance;
- tests on three platforms.

[ARCHITECTURE.md](ARCHITECTURE.md) describes the engine. Security problems: see
[SECURITY.md](SECURITY.md).

```bash
make venv && make py-build && make test   # C++ (GoogleTest) + Python tests
```

## Acknowledgments

miter predicts with [lignum](https://github.com/misterekkk/lignum). Its categorical handling follows
ideas from [LightGBM](https://github.com/microsoft/LightGBM) (set splits) and
[CatBoost](https://github.com/catboost/catboost) (ordered target statistics, symmetric trees).

## License

[Apache License 2.0](LICENSE). If you use miter in research, [CITATION.cff](CITATION.cff) has the
citation.
