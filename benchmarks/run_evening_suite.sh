#!/usr/bin/env bash
# Ready-to-run evening benchmark suite for `feat-v11-softmax-cat-ctr`.
# Follows frozen pre-registrations:
#   - benchmarks/results/exp_cat_ctr_prereg.md
#   - benchmarks/results/exp_softmax_tabarena51_prereg.md
#
# Usage:
#   ./benchmarks/run_evening_suite.sh                 # run all stages sequentially (3 shards each)
#   ./benchmarks/run_evening_suite.sh --stage cat     # run exp-cat-ctr only
#   ./benchmarks/run_evening_suite.sh --stage softmax # run exp-softmax-multiclass only
#   ./benchmarks/run_evening_suite.sh --stage tabarena # run TabArena 51 MITER_V11_T12 only
#   ./benchmarks/run_evening_suite.sh --shards 4      # override shard count (1 thread/shard)

set -euo pipefail

STAGE="all"
SHARDS=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="${2:?missing --stage value}"
      shift 2
      ;;
    --shards)
      SHARDS="${2:?missing --shards value}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${ROOT_DIR}/.venv_tabarena/bin/python"
LOG_DIR="${ROOT_DIR}/benchmarks/results/raw/logs"
mkdir -p "${LOG_DIR}"

run_sharded() {
  local script_path="$1"
  local tag="$2"
  local pids=()
  echo "=== Starting ${tag} (${SHARDS} shards, nice -n 10) ==="
  for ((s = 0; s < SHARDS; s++)); do
    nice -n 10 "${PY}" "${script_path}" run --shard "${s}" --nshards "${SHARDS}" \
      > "${LOG_DIR}/${tag}_shard${s}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  echo "=== Summarizing ${tag} ==="
  "${PY}" "${script_path}" summarize
}

case "${STAGE}" in
  cat|all)
    run_sharded "${ROOT_DIR}/benchmarks/bench_exp_cat_ctr.py" "exp_cat_ctr"
    ;;
esac

case "${STAGE}" in
  softmax|all)
    run_sharded "${ROOT_DIR}/benchmarks/bench_exp_softmax_multiclass.py" "exp_softmax_multiclass"
    ;;
esac

case "${STAGE}" in
  tabarena|all)
    run_sharded "${ROOT_DIR}/benchmarks/tabarena/run_tabarena_v11_51.py" "tabarena_v11_51"
    ;;
esac

echo "=== Done stage=${STAGE} ==="
