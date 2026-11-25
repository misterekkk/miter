#!/usr/bin/env bash
# Overnight chain (pre-registered in tabarena_t12_full43_prereg.md + tabarena_t12_full51_prereg.md):
# wait for the 43-task shards -> summarize 43 -> install miter from main into .venv_tabarena ->
# 8 multiclass tasks (one-vs-rest) in 3 shards -> summarize 51. Logs go to $1 (a directory).
set -u
LOG="${1:?log dir}"
cd "$(dirname "$0")/../.."
PY=.venv_tabarena/bin/python
RUN=benchmarks/tabarena/run_tabarena_t12_full43.py
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(stamp) waiting for 43-task shards" >>"$LOG/chain.log"
while pgrep -f "run_tabarena_t12_full43.py run --shard" >/dev/null; do sleep 60; done
echo "$(stamp) 43-task shards finished" >>"$LOG/chain.log"

nice -n 10 "$PY" "$RUN" summarize >"$LOG/summarize43.log" 2>&1
echo "$(stamp) summarize43 exit=$?" >>"$LOG/chain.log"

nice -n 10 "$PY" -m pip install -q --no-deps --force-reinstall . >"$LOG/install.log" 2>&1
echo "$(stamp) install exit=$?" >>"$LOG/chain.log"

for s in 0 1 2; do
  nice -n 10 "$PY" "$RUN" run-multiclass --shard "$s" --nshards 3 >"$LOG/mc_shard$s.log" 2>&1 &
done
wait
echo "$(stamp) multiclass shards finished" >>"$LOG/chain.log"

nice -n 10 "$PY" "$RUN" summarize51 >"$LOG/summarize51.log" 2>&1
echo "$(stamp) summarize51 exit=$? CHAIN DONE" >>"$LOG/chain.log"
