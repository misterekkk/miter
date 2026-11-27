// Benchmark numbers shown on the site. Sources (in the repository):
//   benchmarks/results/tabarena_v12_t25_51_summary.md               (TabArena lite, 51 tasks)
//   benchmarks/results/tabarena_v12_default_candidates_summary.md   (single configurations)
//   benchmarks/results/exp_external_holdout_summary.md              (OpenML, no TabArena overlap)
//   benchmarks/results/exp_scaling_defaults_summary.md              (learning curves)

export interface Group {
  label: string;
  values: number[];
}

// gbdt_only pool, 51 tasks: Elo per method. Tuned rows: MITER_V12_T25 run; default row: the
// default-only comparison (tabarena_default_v1_51_summary.md), a separate pool.
export const eloByTuning: Group[] = [
  { label: 'tuned + ensemble', values: [1218, 1081, 1111, 1048] },
  { label: 'tuned', values: [1176, 1076, 1007, 994.9] },
  { label: 'default', values: [998.3, 996, 722.1, 787.5] },
];

// Single configurations (no tuning), one pool (tabarena_default_v1_51_summary.md).
export const singleConfigs: Group[] = [
  { label: 'symmetric + quantile + CTR, m_mult 4', values: [1090] },
  { label: 'symmetric + auto quantizer + CTR', values: [1064] },
  { label: 'symmetric + auto quantizer', values: [1026] },
  { label: 'default (symmetric, depth 6, m_mult 4)', values: [998.3] },
  { label: 'symmetric, m_mult 2', values: [991.4] },
  { label: 'leaf-wise (earlier default)', values: [930.4] },
];
export const singleConfigRefs = { catDefault: 996, catTuned: 1091 };

// Seconds for 100 trees, Apple M4 (benchmarks/results/train_speed_summary.md).
export const speed1: Group[] = [
  { label: '20k × 16', values: [0.13, 0.31, 0.36, 0.22] },
  { label: '20k × 100', values: [0.84, 1.44, 1.26, 0.91] },
  { label: '100k × 50', values: [1.33, 1.7, 3.41, 2.11] },
  { label: '500k × 50', values: [6.29, 6.13, 19.79, 10.04] },
];
export const speed10: Group[] = [
  { label: '20k × 16', values: [0.37, 0.13, 0.17, 0.12] },
  { label: '20k × 100', values: [0.61, 0.33, 0.34, 0.25] },
  { label: '100k × 50', values: [0.73, 0.43, 0.83, 0.51] },
  { label: '500k × 50', values: [1.91, 1.45, 4.16, 2.32] },
];
