# E6-A / E6-F: presets `accurate` and `fast`, verdict

Pre-registration: `exp_e6_options_prereg.md` (E6-A, E6-F rules). Inputs:
`exp_e6_presets_heldout_summary.md` (32 held-out datasets x 5 folds, vs `BASE`),
`tabarena_speedup_e6_{acc,fast}_51_summary.md` (51 TabArena tasks, vs `DEF2`),
`exp_e6_presets_speed_summary.md` (100 trees, one thread, 5 interleaved runs, Apple M4).

| | held-out median | held-out wins | held-out p (worse) | TabArena median | TabArena p (worse) | covertype time | 500k x 50 time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `accurate` | -0.000% | 16/32 | 0.87 | -0.004% | 0.84 | 1.24x | 1.19x |
| `fast` | +0.620% | 7/32 | 0.02 | +0.065% | 0.42 | 0.74x | 0.83x |

## Verdict
- **E6-A, `accurate`: FAIL, removed.** The rule needs a held-out improvement (median < 0 and
  one-sided Wilcoxon p < 0.05); the median is 0.000% with 16 wins of 32, at about 1.2x the
  training time. TabArena agrees (median -0.004%).
- **E6-F, `fast`: FAIL, removed.** The rule needs time <= 0.67x the default on both large
  datasets; it is 0.74x on covertype and 0.83x on 500k x 50 (the oblique row sampling already
  removes most of the oblique cost on large data). The quality condition held (held-out median
  +0.62% <= +1.0%), but it is not fast enough to be worth an option.
- Both presets are removed, so the `preset` parameter goes. `m_mult` stays as a tuning parameter.
