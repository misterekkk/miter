# miter

A C++20 gradient-boosted decision tree (GBDT) library with random sparse oblique splits and bit-exact integer inference via `lignum`.

## Setup (`extern/lignum`)

`miter` builds against a local, gitignored clone of `lignum` in `extern/lignum/` (see `docs/LIGNUM_INTEGRATION.md`):

```bash
git clone --branch v0.4.3 https://github.com/misterekkk/lignum extern/lignum
git -C extern/lignum checkout -b miter-integration
```

If `lignum_patches/` contains patches, apply them inside `extern/lignum/`:

```bash
git -C extern/lignum am "$PWD"/lignum_patches/*.patch
```

## Build & Test (C++)

```bash
cmake -S . -B build -DMITER_BUILD_TESTS=ON -DMITER_WARNINGS_AS_ERRORS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Build & Test (Python)

```bash
pip install ".[test]"
pytest tests/python -v
```
