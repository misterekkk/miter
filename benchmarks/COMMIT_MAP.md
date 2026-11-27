# Commit map after the 2026-11-27 history rewrite

The internal design notes (`GEMINI.md`, `docs/`) and the old lignum patch files (`lignum_patches/`, now
part of lignum v0.5.0) were removed from the whole history with `git filter-repo`. Every commit got a
new id. Pre-registrations and summaries written before the rewrite cite the old ids; this table maps
them. The SHA-256 hashes of pre-registration files are unchanged (the files are unchanged).
Commits that only touched the removed files no longer exist (marked `removed`).

| old | new | date | subject |
|---|---|---|---|
| `1b2d234` | `f522a9a` | 2026-11-24 | feat(data): add 64-byte aligned buffer, Q16r quantizer, axis binner, and categorical encoder |
| `1ef30a4` | `b953f17` | 2026-11-24 | feat(python): wire oblique splits and presets into MiterRegressor and MiterClassifier |
| `2941752` | `499f026` | 2026-11-24 | chore: add .gitignore, formatting/lint configs, and directory layout |
| `3466ac2` | `9961c13` | 2026-11-24 | docs: add README and record M0 engineering decisions |
| `4b8d527` | `9f157c4` | 2026-11-24 | feat(core): add objectives, compact axis histograms with subtraction, and split finder |
| `5492bcc` | `4312970` | 2026-11-24 | feat(core): add K1 categorical bitset split finder and NaN-vs-rest handling |
| `5c2dd0a` | `00803f4` | 2026-11-24 | bench: add M4 categorical K1 and NaN benchmarks vs histree, LightGBM, and CatBoost |
| `60103d5` | `7a7c928` | 2026-11-24 | perf(core): parallelize histograms, stable row partition, objective, and evaluator deterministically |
| `63a4e69` | `8fdd56e` | 2026-11-24 | feat(core): add leaf-wise tree learner, evaluator, JSON serializer, and AX booster |
| `69913e0` | `01f33da` | 2026-11-24 | feat(core): wire categorical K1 splits into tree learner, booster, evaluator, and JSON I/O |
| `6bbe399` | `406475d` | 2026-11-24 | docs: label M5 Part C as own TabArena-style protocol, record sulfur weakness and FastDiv64 MSVC path |
| `768ae26` | `fcd96ab` | 2026-11-24 | bench: add M3 sulfur multi-seed check and oblique speed/parity benchmarks |
| `85ff910` | `9d8b403` | 2026-11-24 | fix(core): add MSVC (__umulh) and int128-free paths to FastDiv64 with tests |
| `92862d1` | `7be7006` | 2026-11-24 | test: add bitwise multi-thread determinism (1/2/4/8 threads) and FastDiv64 tests |
| `964080c` | `69a3737` | 2026-11-24 | feat: add nanobind Python module, pyproject.toml, and pytest smoke tests |
| `9980be2` | removed | 2026-11-24 | docs: add initial implementation plan, lignum integration plan, and agent rules |
| `9f17c96` | `3fcf349` | 2026-11-24 | ci: add GitHub Actions workflow for 3 OSes, sanitizers, lint, and wheel smoke test |
| `9fd0e61` | `e0b2149` | 2026-11-24 | ci: extend lint workflow to cover all src and include files |
| `a8739df` | `c8b28d8` | 2026-11-24 | build: add CMake build, C++20 scaffold, extern/lignum integration, and GoogleTest |
| `af413f9` | `6f32b14` | 2026-11-24 | feat(core): integrate oblique splits into tree learner, booster, evaluator, and JSON I/O |
| `bd1b656` | `c9a6874` | 2026-11-24 | feat(data): add Dataset with row-major Q16r codes and column-major bins |
| `c4d6507` | `6c99aec` | 2026-11-24 | style: sort imports in test_smoke.py |
| `ccf4ac7` | `0216557` | 2026-11-24 | perf(core): add FastDiv64, compact oblique metadata, and deterministic wave-chunked OpenMP |
| `d1b4d12` | `26bd865` | 2026-11-24 | bench: add M5 speed, 500k+ scaling, and 12-dataset TabArena leaderboard benchmark |
| `d249ac2` | `00998ab` | 2026-11-24 | feat(core): add counter-based SplitMix64 RNG and integer oblique R_uni split finder |
| `d36dd3e` | `acfa6a9` | 2026-11-24 | feat(python): add mixed categorical+numeric prediction and K1 params to estimators |
| `ea416f1` | `02ee33d` | 2026-11-24 | feat(python): add MiterRegressor, MiterClassifier, JSON I/O, and booster tests |
| `f675f48` | `cad146c` | 2026-11-24 | feat(python): bind Dataset and data preprocessing for NumPy and pandas |
| `faaa091` | `82febf2` | 2026-11-24 | bench: add M2 AX benchmark harness and summary results vs LightGBM and histree |
| `044c8ba` | `6705099` | 2026-11-25 | feat(model): miter models in lignum binary format 0.10 (M6 step 4) |
| `11afc68` | `317a6c0` | 2026-11-25 | perf(symmetric): pairwise histogram subtraction, 4x unroll, wave determinism fix, TabArena config |
| `165f373` | `81c53ad` | 2026-11-25 | bench: Brazilian_houses residual causal test (no cause found among f32 g/h, oblique range, Q16r scale) |
| `17577d9` | `0048331` | 2026-11-25 | bench(prereg): freeze exp-cat-oblique-ts pre-registration (sha256: 6594bf6736280041468cb59b958e1fc587e7bec8d8c7e4582fe071cacf2a9be0) |
| `1ac76b5` | `5d99313` | 2026-11-25 | style(bench): sort imports in residual check |
| `1f1aefd` | `7eabb49` | 2026-11-25 | feat(python): miter.tabarena AutoGluon/TabArena model wrapper (M7) |
| `25ee660` | `e671122` | 2026-11-25 | style(bench): make M3 parity scripts pass ruff |
| `322982e` | `de964b2` | 2026-11-25 | feat(python): predict through lignum (M6 step 5) |
| `33b4316` | `0a090dd` | 2026-11-25 | build: release wheels (cibuildwheel) and a self-contained sdist (M7) |
| `3aca572` | `b7b344e` | 2026-11-25 | feat(python): scikit-learn compatible estimators and pickling (M7) |
| `3c5758b` | `239a76e` | 2026-11-25 | feat(core): feature_fraction, bagging_fraction, extra_trees options (M7) |
| `3de1c6a` | `5f557e8` | 2026-11-25 | bench: pre-registered n-adaptive complexity check (M7), not adopted |
| `3ec855a` | `870b0fe` | 2026-11-25 | docs: user-facing README for v1.0 (M7) |
| `4c82448` | removed | 2026-11-25 | docs: accept M6 inference speed for v1.0 |
| `6640218` | `475e548` | 2026-11-25 | bench(tabarena): add MITER_BS6 and MITER_T12 lite ensemble evaluation across 20 tasks |
| `671c6ab` | `84a89c8` | 2026-11-25 | feat(core): add GrowPolicy::Symmetric (oblivious + oblique trees) |
| `6920d8f` | `6475ff6` | 2026-11-25 | bench: score miter C++ on the TabArena protocol (13 regression datasets, full method pool) |
| `783aa47` | `1e86c7b` | 2026-11-25 | bench: TabArena regression with the accurate preset (m = 4·m0) |
| `7f4ee48` | `1c6e56e` | 2026-11-25 | fix(ci): apply lignum patches with a path the shell can expand |
| `8061992` | `1a469cf` | 2026-11-25 | bench: run pre-registered symmetric oblique experiment (exp-symmetric-oblique) |
| `8656340` | `cbd21e7` | 2026-11-25 | perf(data): switch codes to col-major, add L1 LUT binner and OpenMP Dataset |
| `876f133` | `1603ba2` | 2026-11-25 | fix(data): build axis bins with the research quantile rule |
| `896d1fc` | `7bcd038` | 2026-11-25 | docs: document sampling options in README |
| `923ef48` | `83961a8` | 2026-11-25 | ci(wheels): macos-latest runner, deployment target from the runner |
| `9a8e058` | removed | 2026-11-25 | feat(model): lignum compiled mode for miter models (M6 step 3b) |
| `9cfd19f` | `09f9369` | 2026-11-25 | style: clang-format the lignum bridge test |
| `9fb7971` | `f2a235d` | 2026-11-25 | chore: ignore .venv_tabarena |
| `a1ca3dc` | `b5c87a4` | 2026-11-25 | feat(cat): opt-in OOF target statistics for oblique splits + benchmark summary (FAIL confirmatory, stays opt-in) |
| `c03d299` | `ba828bd` | 2026-11-25 | bench: add M3 classification parity vs prototype (13 datasets x 5 folds, pre-registered) |
| `c321033` | `3e6820c` | 2026-11-25 | feat(model): lignum binned kernel (ONode) for miter models (M6 step 3a) |
| `d0ac58f` | `9d74b22` | 2026-11-25 | feat(model): export miter trees in lignum's split semantics (M6 step 1) |
| `d1d48d9` | `644adc2` | 2026-11-25 | fix(build): make M5 code pass GCC -Werror, Linux clang, and Python CI |
| `d54edc6` | `68c8a17` | 2026-11-25 | bench(prereg): freeze symmetric oblique experiment pre-registration (4604e243) |
| `eb8a030` | `25bd828` | 2026-11-25 | perf(oblique,tree): L1 micro-block scan, preordered work buffers, race-free partition |
| `ebb7d3b` | `389e71f` | 2026-11-25 | chore(tabarena): add cat_oblique_ts and conditional symmetric depth to search space, record DECISIONS.md |
| `ed7333f` | `49117d7` | 2026-11-25 | feat(model): build lignum models from miter models (M6 step 2) |
| `f42164f` | `499a2c3` | 2026-11-25 | docs,ci: record 500k col-major optimization benchmark and patch lignum in sdist |
| `f71de4d` | `23e660b` | 2026-11-25 | bench: add M3 Brazilian_houses/pol 5x5 parity vs live histree T7 and axis-binning root cause |
| `ff9d80f` | `45a62dd` | 2026-11-25 | bench: record M3 parity after the axis-binning fix |
| `00c2784` | `a2c65af` | 2026-11-26 | feat(ctr): native C++20 online ordered CTR splits and dynamic category combinations |
| `062cc6b` | `6fe446e` | 2026-11-26 | feat(ctr): add gain margin, depth, onehot, and table gating for num-cat CTR |
| `0d2b8db` | `87a4b0c` | 2026-11-26 | bench(prereg): freeze TabArena T12 full-43 pre-registration (sha256: a4536283f7f60ae121db0d965bd02d45e9e204fddc8937ff0f68d74f2b927036) |
| `0f66675` | `8bb8109` | 2026-11-26 | bench(prereg): freeze exp-cat-ctr (sha256: 766aed4a45a3e699d9254985c97398ec8fa10537b951f6fecf57c2f5db1df95d) and exp-softmax-tabarena51 (sha256: 3d1ba53df612e637539f98cf1e503faeb77297f2fe269a1a5b0813bbb15d0c0e) |
| `0f7ae9b` | `1dd1cbd` | 2026-11-26 | bench: freeze pre-registrations for exp-quantizer-auto and exp-ctr-auto-confirm |
| `1718e6d` | `17243f0` | 2026-11-26 | bench(tabarena): T12 on all 43 lite reg+binary tasks (pre-registered): Elo #1 in gbdt pool, level with CAT tuned+ens (p=0.91) |
| `1abf1e0` | `c4caca7` | 2026-11-26 | feat(ctr): add adaptive complexity, support gate, and num-cat combinations |
| `1eba0cb` | `b1af7d3` | 2026-11-26 | feat(quantizer): add per-feature adaptive quantile Q16 mode (Option A) |
| `289c870` | `b94d8f5` | 2026-11-26 | feat(multiclass): native C++20 softmax multiclass objective and MiterClassifier support |
| `48defd8` | `926d83b` | 2026-11-26 | bench: record exp-cat-ctr, exp-softmax-multiclass, and tabarena v11 51-task summaries |
| `62f5047` | `fc4dac1` | 2026-11-26 | bench: record exp-quantizer-auto and exp-ctr-auto-confirm confirmatory summaries |
| `6f2eff5` | `1f1fc57` | 2026-11-26 | fix(multiclass): drop noexcept on allocating multiclass helpers |
| `76d4ca9` | removed | 2026-11-26 | docs(ideas): regularised TAO and stochastic top-2 root split |
| `81d4430` | `01dc022` | 2026-11-26 | feat(python): bind native C++20 online ordered CTR parameters and tests |
| `82fc887` | `f8b187f` | 2026-11-26 | bench: freeze pre-registration and harness for exp-native-ctr-v2 |
| `868a022` | `eed29f9` | 2026-11-26 | fix(bench): separate pandas-3 OpenML cache from pandas-2 TabArena cache |
| `8b9a126` | removed | 2026-11-26 | docs: record two-stage residual boosting idea in IDEAS.md |
| `8fcecdc` | `a13e6f3` | 2026-11-26 | feat(tabarena,bench): native softmax in MiterModel and frozen evening benchmark suite |
| `b8eaa67` | `067f8aa` | 2026-11-26 | feat(cat): CatBoost-style ordered CTRs, pair combinations, counter, and multiclass support |
| `bba378a` | `34e0d1f` | 2026-11-26 | feat(tabarena): add v12 search space with quantizer and native CTR, plus small-sample CTR guard |
| `d127097` | `30abb3a` | 2026-11-26 | bench: record exp-native-ctr-v2 confirmatory summary |
| `e04eed9` | `e0f5946` | 2026-11-26 | fix(cat,bench): use constant global prior in ordered CTR and fix benchmark venv/string dtypes |
| `ef4c18b` | `2db7ad7` | 2026-11-26 | bench(tabarena): T12 on all 51 lite tasks (multiclass via one-vs-rest, secondary) |
| `f17b958` | `a0ec17d` | 2026-11-26 | feat(tabarena): one-vs-rest multiclass in MiterModel; 51-task addendum (sha256: af23a3316832d7f2dbb65ff153ce22ae3c47b380cde71bb3cbe6d85d59420c1a) |
| `fabcf52` | `6e9121f` | 2026-11-26 | perf(ctr): cache quantized CTRs, parallelize scoring, and add CatBoost priors/reg |
| `24ee897` | `378491a` | 2026-11-27 | bench: add and summarize out-of-TabArena scaling laws and default settings study |
| `751c26b` | `6790dde` | 2026-11-27 | feat: switch lignum integration to v0.5.0 FetchContent and enable multiclass/quantile/CTR/symmetric |
| `75a04df` | `417b792` | 2026-11-27 | bench: record exploratory TabArena lite 51-task summary for MITER_V12_T25 |
| `82678c3` | `f997169` | 2026-11-27 | chore: slim superseded Python cat_ctr options, update docs, and enable manual CI |
| `94acdf3` | `55fd687` | 2026-11-27 | chore: keep internal docs and agent instructions local (docs/ gitignored) |
| `e7b8911` | `b1a410c` | 2026-11-27 | bench: run out-of-TabArena external OpenML holdout validation (32 zero-overlap datasets) |
| `fa75eac` | `452d53c` | 2026-11-27 | refactor(ctr): remove unused num-cat CTR combinations and gating parameters |
