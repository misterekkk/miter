# miter benchmarks

Every accuracy claim about miter comes from a script in this folder. Its summary is in
[`results/`](results/).

- **`results/*_prereg.md`** are pre-registrations: the datasets, seeds, metric and decision rule of
  a confirmatory experiment. Each was committed, with its SHA-256 in the commit message, *before*
  the experiment ran.
- **`results/*_summary.md`** are the results, including the experiments that failed their rule.
- **`tabarena/`** runs miter in [TabArena](https://tabarena.ai) through `miter.tabarena.MiterModel`.
  - `run_tabarena_v12_51.py` is the latest 51-task run.
  - `analyze_v12_default_candidates.py` scores its single configurations.
- **`bench_exp_*.py`** are the pre-registered experiments. Each one is the script named in its
  pre-registration.
  - `bench_exp_external_holdout.py` and `bench_exp_scaling_defaults.py` use OpenML datasets with no
    overlap with TabArena. They check that results generalise beyond the benchmark the
    configurations were developed on.
- **`bench_m*.py`, `check_*.py`** are milestone and parity checks from development.

Raw outputs (`results/raw/`, datasets, caches) are not committed. The commit ids that summaries
written before 2026-11-27 cite are mapped to the current history in [COMMIT_MAP.md](COMMIT_MAP.md).
