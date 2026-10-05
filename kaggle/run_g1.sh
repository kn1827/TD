#!/usr/bin/env bash
# G1 pilot on Kaggle T4 x2: two independent lanes, one per GPU, each with its own vLLM server and
# its own Python driver. Start both in the background from the notebook (see tdmad_g1.ipynb):
#
#   bash kaggle/run_g1.sh lane0     # GPU0: pool model A -> S1 screening -> S2 MAD
#   bash kaggle/run_g1.sh lane1     # GPU1: pool model B -> S1 -> S2 -> generators -> S3 messages -> S4 sheets
#   bash kaggle/run_g1.sh bench     # both pool models, throughput bench (~10 min), before the big run
#   bash kaggle/run_g1.sh smoke     # 3 items per task through every stage, both lanes (~20 min)
#   bash kaggle/run_g1.sh analyze   # CPU: gate report (results/<run>/analysis/g1_report.md)
#
# Every stage resumes from what is on disk, so after a 12-hour cut-off rerun the same lanes in a
# new session (results/ restored first). Env: CFG (default configs/g1.yaml), LIMIT (items per task).
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/g1.yaml}
get() { python -m scripts.cfg_get "$CFG" "$@"; }
RUN_DIR="results/$(get run_name)"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
mkdir -p "$RUN_DIR" logs

lane() {   # lane name
  local L=$1
  local gpu port model hf pools gens
  gpu=$(get lanes.$L.gpu); port=$(get lanes.$L.port)
  model=$(get lanes.$L.model); hf=$(get lanes.$L.model --hf); rev=$(get --rev-of "$model")
  pools=$(get lanes.$L.pools)
  gens=$(get lanes.$L.generators 2>/dev/null || true)

  echo "=== [$L] $model on GPU $gpu:$port  $(date)"
  bash kaggle/start_vllm.sh one "$gpu" "$port" "$hf" "$rev"
  python -m scripts.run_screen --config "$CFG" --model "$model" $LIMIT_ARG
  touch "$RUN_DIR/.screen_done_$model"
  for pool in $pools; do
    python -m scripts.run_mad --config "$CFG" --pool "$pool" $LIMIT_ARG
  done
  bash kaggle/start_vllm.sh stop_port "$port"

  if [[ -n "$gens" ]]; then
    # message targets need S1 of every pool model
    for m in $(get lanes.lane0.model) $(get lanes.lane1.model); do
      until [[ -f "$RUN_DIR/.screen_done_$m" ]]; do echo "[$L] waiting for S1 of $m"; sleep 60; done
    done
    for g in $gens; do
      echo "=== [$L] generator $g  $(date)"
      bash kaggle/start_vllm.sh one "$gpu" "$port" "$(get --hf-of "$g")" "$(get --rev-of "$g")"
      python -m scripts.run_messages --config "$CFG" --generator "$g"
      bash kaggle/start_vllm.sh stop_port "$port"
    done
    python -m scripts.make_annotation --config "$CFG"
  fi
  echo "=== [$L] done  $(date)"
}

case "${1:-}" in
  lane0|lane1)
    lane "$1" ;;
  bench)
    m0=$(get lanes.lane0.model); m1=$(get lanes.lane1.model)
    bash kaggle/start_vllm.sh pair "$(get --hf-of "$m0")" "$(get --hf-of "$m1")"       "$(get --rev-of "$m0")" "$(get --rev-of "$m1")"
    python -m scripts.bench_throughput --config "$CFG" --model "$(get lanes.lane0.model)" &
    python -m scripts.bench_throughput --config "$CFG" --model "$(get lanes.lane1.model)" &
    wait
    bash kaggle/start_vllm.sh stop
    python -m scripts.estimate_budget --config "$CFG" --calibrate "results/bench/*.json" ;;
  smoke)
    export LIMIT=${LIMIT:-3}
    CFG_SMOKE="configs/g1_smoke.yaml"
    CFG=$CFG_SMOKE bash kaggle/run_g1.sh lane0 > logs/smoke_lane0.log 2>&1 &
    CFG=$CFG_SMOKE bash kaggle/run_g1.sh lane1 > logs/smoke_lane1.log 2>&1 &
    wait
    tail -n 5 logs/smoke_lane0.log logs/smoke_lane1.log
    CFG=$CFG_SMOKE python -m analysis.g1_gate --config "$CFG_SMOKE" | head -n 20 ;;
  analyze)
    python -m analysis.g1_gate --config "$CFG" ;;
  *)
    echo "usage: bash kaggle/run_g1.sh {lane0|lane1|bench|smoke|analyze}"; exit 1 ;;
esac
