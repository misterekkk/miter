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
  level. It is level with CatBoost after tuning, and the same holds on held-out OpenML data
  ([benchmarks](#benchmarks)).
- **Deterministic to the bit.** Splits are integer arithmetic on 16-bit feature codes, and the
  randomness is counter-based. The same seed gives the same model on any machine and thread count,
  and prediction follows exactly the path training took.
- **Two tree shapes.** Leaf-wise trees like LightGBM, or symmetric (oblivious) trees like CatBoost
  (`grow_policy="symmetric"`), both with oblique splits.
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
- **Elo:** computed in the GBDT pool.
- **Paired comparison:** wins over the 51 tasks, median relative error (negative = miter better),
  and Wilcoxon p.

| | Elo: miter | CatBoost | LightGBM | XGBoost | miter vs CatBoost | miter vs LightGBM |
|---|---:|---:|---:|---:|---|---|
| **tuned + ensemble** | **1218** | 1081 | 1111 | 1048 | 35/51, −0.78%, p = 0.14 | 37/51, −0.62%, p = 0.01 |
| **tuned** | **1176** | 1076 | 1007 | 995 | 31/51, −0.44%, p = 0.44 | 39/51, −0.99%, p = 0.003 |
| **default** | 897 | **967** | 685 | 740 | 22/51, +0.88%, p = 0.06 | 42/51, −1.72%, p < 0.001 |

- **LightGBM and XGBoost:** miter is ahead at every level.
- **CatBoost:** miter is level after tuning (higher Elo, but the per-task differences are not
  significant) and slightly behind with default parameters.
- **Single configurations:** symmetric trees with the auto quantizer and CTRs
  (`grow_policy="symmetric", quantizer="auto", max_ctr_complexity="auto"`) are the strongest single
  configuration, with Elo 1065 without any tuning, above CatBoost's default (994).
- **Generalisation:** on 16 OpenML datasets with no overlap with TabArena, this configuration is
  level with CatBoost's default (9/16 wins, median −0.39%). So the TabArena result is
  generalisation, not tuning to the benchmark.

Training is slower than LightGBM because of the oblique candidates, about 1.5–9× per tree on one
thread; use `preset="fast"` on large data. Details, the held-out experiments and the
pre-registered protocols:
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
