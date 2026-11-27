// Benchmark numbers shown on the site. Sources (in the repository):
//   benchmarks/results/tabarena_v12_t25_51_summary.md               (TabArena lite, 51 tasks)
//   benchmarks/results/tabarena_v12_default_candidates_summary.md   (single configurations)
//   benchmarks/results/exp_external_holdout_summary.md              (OpenML, no TabArena overlap)
//   benchmarks/results/exp_scaling_defaults_summary.md              (learning curves)

export interface Group {
  label: string;
  values: number[];
}

// gbdt_only pool, 51 tasks: Elo per method (MITER_V12_T25 run).
export const eloByTuning: Group[] = [
  { label: 'tuned + ensemble', values: [1218, 1081, 1111, 1048] },
  { label: 'tuned', values: [1176, 1076, 1007, 994.9] },
  { label: 'default', values: [897.1, 966.7, 684.6, 739.8] },
];

// Single configurations (no tuning), same pool and tasks (candidates scored from the V12 cache).
export const singleConfigs: Group[] = [
  { label: 'miter S6 + auto quantizer + CTR, m_mult 4', values: [1092] },
  { label: 'miter S6 + auto quantizer + CTR', values: [1065] },
  { label: 'miter S6 + auto quantizer', values: [1026] },
  { label: 'miter S6 (symmetric, depth 6)', values: [993.4] },
  { label: 'miter default (leaf-wise)', values: [928.8] },
];
export const singleConfigRefs = { catDefault: 994.4, catTuned: 1090 };
