#!/usr/bin/env bash
# Kaggle-only wrapper (GPU T4 x2). Everything else in the repository runs without this folder.
#
#   bash kaggle/kaggle_run.sh smoke     # configs/smoke.yaml: every model answers a few questions (~1 h)
#   bash kaggle/kaggle_run.sh main      # configs/exp.yaml: the full run; rerun in the next session to resume
#
# What it adds to the plain command `python -m hmad.run --config <cfg>`:
#   * results go to /kaggle/working/results (kept when the notebook version is saved);
#   * model weights are cached in /tmp/models (NOT under /kaggle/working, whose 20 GB are for outputs),
#     with a disk budget = free space on /tmp minus 10 GB;
#   * local copies of models attached as notebook inputs (/kaggle/input/...) are used before downloading;
#   * TD_DEADLINE_TS (set by the notebook) makes the run stop before the 12-hour limit; exit code 3 =
#     stopped early, rerun in a new session with this session's output attached as input.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-smoke}
case "$MODE" in
  smoke) BASE=configs/smoke.yaml ;;
  main)  BASE=configs/exp.yaml ;;
  *) echo "usage: bash kaggle/kaggle_run.sh {smoke|main}"; exit 1 ;;
esac

mkdir -p /tmp/models /kaggle/working/results
FREE_GB=$(df -BG --output=avail /tmp | tail -n 1 | tr -dc '0-9')
BUDGET=$(( FREE_GB > 30 ? FREE_GB - 10 : 20 ))
CFG=/kaggle/working/kaggle_${MODE}.yaml
cat > "$CFG" <<EOF
extends: $BASE
results_dir: /kaggle/working/results
engine:
  model_cache_dir: /tmp/models
  disk_budget_gb: $BUDGET
  model_search_paths: [/kaggle/input]
EOF
echo "[kaggle] $MODE: /tmp has ${FREE_GB} GB free -> model cache budget ${BUDGET} GB; config $CFG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

set +e
python -m hmad.run --config "$CFG"
RC=$?
set -e
if [[ $RC -eq 3 ]]; then
  echo "[kaggle] stopped before the session limit: save this version, then rerun with its output as input"
fi
exit $RC
