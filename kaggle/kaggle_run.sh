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

mkdir -p /kaggle/working/results
# Model cache: the first candidate where 256 MB can really be written (df on Kaggle's overlay shows
# the whole host's free space, not the session's quota, and /tmp was once mounted read-only).
CACHE=""
for d in ${MODEL_CACHE:-} /tmp/models /root/hmad_models /kaggle/tmp/models; do
  [[ -z "$d" ]] && continue
  if ERR=$( { mkdir -p "$d" && dd if=/dev/zero of="$d/.write_test" bs=1M count=256 status=none; } 2>&1 ); then
    rm -f "$d/.write_test"; CACHE=$d; break
  fi
  rm -f "$d/.write_test" 2>/dev/null || true
  echo "[kaggle] $d is not writable ($ERR), trying the next folder"
done
[[ -z "$CACHE" ]] && { echo "[kaggle] no writable folder for model weights"; exit 1; }
# Budget capped (default 40 GB = 2 models of 7-9B: the running one + the prefetched next one). A session
# has a hidden disk quota (~120 GB incl. the ~30 GB of packages): past it the disk turns read-only.
# Above the budget, least-recently-used models are
# deleted and downloaded again later (~1 min each). Override with CACHE_GB=... if the quota is known.
FREE_GB=$(df -BG --output=avail "$CACHE" | tail -n 1 | tr -dc '0-9')
CAP=${CACHE_GB:-40}
BUDGET=$(( FREE_GB - 10 < CAP ? FREE_GB - 10 : CAP ))
CFG=/kaggle/working/kaggle_${MODE}.yaml
cat > "$CFG" <<EOF
extends: $BASE
results_dir: /kaggle/working/results
engine:
  model_cache_dir: $CACHE
  disk_budget_gb: $BUDGET
  model_search_paths: [/kaggle/input]
EOF
echo "[kaggle] $MODE: model cache $CACHE (df shows ${FREE_GB} GB free) -> budget ${BUDGET} GB; config $CFG"
df -h "$CACHE" /kaggle/working | cat
mount | grep -E " on (/|/tmp|/kaggle/working) " | cat || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv

set +e
python -m hmad.run --config "$CFG"
RC=$?
set -e
if [[ $RC -eq 3 ]]; then
  echo "[kaggle] stopped before the session limit: save this version, then rerun with its output as input"
fi
exit $RC
