# Contributing to miter

Thanks for your interest. Bug reports, datasets where miter does clearly worse than LightGBM or
CatBoost, and benchmark results from your hardware are all very welcome as
[issues](https://github.com/misterekkk/miter/issues). Security problems go through the private
channel described in [SECURITY.md](SECURITY.md).

## Setting up

```bash
make venv       # .venv with Python 3.12 via uv and the test dependencies
make py-build   # editable install (run again after every C++ change)
make test       # C++ (GoogleTest) + Python tests
```

macOS needs `brew install libomp`. CMake fetches [lignum](https://github.com/misterekkk/lignum)
automatically; to build against a local checkout, pass `-DMITER_LIGNUM_DIR=/path/to/lignum`.
[ARCHITECTURE.md](ARCHITECTURE.md) describes the engine.

## The rules every change keeps

1. **Deterministic and exact.** The same `random_state` gives a bit-identical model for any number
   of threads, and `predict` equals miter's own evaluator bit for bit. Everything a split depends
   on stays in integer arithmetic on the 16-bit codes; floating-point sums are reduced in a fixed
   order. Tests compare with `np.testing.assert_array_equal`, never `allclose`. No `-ffast-math`.
2. **Accuracy claims are pre-registered.** A change that is meant to improve accuracy, or a change
   of a default, comes with a pre-registration in `benchmarks/results/` (datasets, seeds, metric,
   decision rule) committed with its SHA-256 *before* the run, and a summary after it. Results that
   fail the rule are kept and reported, not dropped.
3. **No performance regressions.** A change in a hot path comes with a before / after timing
   against the previous commit, and the numbers go into the commit message.
4. **Tests with every feature.** GoogleTest in `tests/cpp` for the engine, pytest in
   `tests/python` for the Python API, including NaN, unseen categories and the determinism checks.
5. **Portable.** Linux, macOS and Windows (MSVC) are all tested in CI. C++20, no new dependencies
   beyond OpenMP and lignum.
6. **Clean static checks.** `ruff check .` and `mypy` for Python; `clang-tidy` (`.clang-tidy`) and
   `-DMITER_WARNINGS_AS_ERRORS=ON` for C++. CI runs all of them.

Code, comments and commit messages are in English. Commit messages say what changed and why, with
measurements for accuracy and performance work.

## Pull requests

Open an issue first for larger changes, so we can agree on the approach. Keep pull requests
focused, make sure `make test` passes, and describe how you tested the change.

## License

miter is licensed under the [Apache License 2.0](LICENSE). By submitting a contribution you agree
that it is licensed under the same terms (section 5 of the license).
