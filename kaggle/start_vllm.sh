#!/usr/bin/env bash
# vLLM servers on a Kaggle "GPU T4 x2" machine (adapted from DUS kaggle/start_vllm.sh).
#
#   bash kaggle/start_vllm.sh one  <gpu> <port> <hf_id> [revision]   # one model, one GPU (lanes)
#   bash kaggle/start_vllm.sh pair <hf_id_gpu0> <hf_id_gpu1> [rev0] [rev1]   # GPU0:8001, GPU1:8002
#   bash kaggle/start_vllm.sh dp   <hf_id> [revision]   # same model on both GPUs (8001, 8002):
#                                                       #   list both URLs under servers: in the config
#   bash kaggle/start_vllm.sh tp   <hf_id> [revision]   # one server over both GPUs (tensor parallel), 8001
#   bash kaggle/start_vllm.sh stop_port <port>
#   bash kaggle/start_vllm.sh stop
#   bash kaggle/start_vllm.sh status
#
# T4 = 16 GB, compute capability 7.5: fp16 only (no bf16, no fp8). One server per GPU here (DUS
# put two 7-8B servers on GPU 0), so each server gets the whole card and CUDA graphs stay on.
# MAXLEN 10240 covers the longest Du-debate request with memory last_round (round-0 prompt + own
# answer + 5 neighbour answers of up to 1024 tokens each + 1024 to generate); use 32768 with
# memory full. revision pins the HF commit (configs/models.yaml) of model and tokenizer.
# Env: MAXLEN (10240) MAXSEQS (64) GPU_UTIL (0.90) EAGER=1 (no CUDA graphs) PREFIX=0 (no prefix
# cache) VLLM_EXTRA (more flags). Logs, pids, arguments: logs/vllm_<port>.{log,pid,args}.
set -euo pipefail

MAXLEN=${MAXLEN:-10240}
MAXSEQS=${MAXSEQS:-64}
GPU_UTIL=${GPU_UTIL:-0.90}
mkdir -p logs

flags() {
  local f="--dtype half --gpu-memory-utilization $GPU_UTIL --max-model-len $MAXLEN"
  f="$f --max-num-seqs $MAXSEQS --max-logprobs 20 --seed 0"
  [[ "${EAGER:-0}" == "1" ]] && f="$f --enforce-eager"
  [[ "${PREFIX:-1}" == "1" ]] && f="$f --enable-prefix-caching"
  echo "$f ${VLLM_EXTRA:-}"
}

start() {   # gpus port hf_id revision [extra vLLM flags...]
  local gpus=$1 port=$2 model=$3 rev=${4:-}
  shift $(( $# < 4 ? $# : 4 ))
  local revflags=""
  if [[ -n "$rev" ]]; then revflags="--revision $rev --tokenizer-revision $rev"; fi
  if curl -s "localhost:$port/v1/models" | grep -q "\"$model\""; then
    echo "port $port already serves $model"; return 0
  fi
  echo "starting $model@${rev:-main} on GPU $gpus, port $port"
  echo "--model $model $revflags $(flags) $*" > "logs/vllm_$port.args"
  CUDA_VISIBLE_DEVICES=$gpus nohup python -m vllm.entrypoints.openai.api_server \
    --model "$model" --port "$port" $revflags $(flags) "$@" > "logs/vllm_$port.log" 2>&1 &
  echo $! > "logs/vllm_$port.pid"
}

wait_up() { # port
  for _ in $(seq 1 240); do      # up to 20 min (first download of the weights included)
    if curl -s "localhost:$1/v1/models" > /dev/null; then echo "port $1 ready"; return 0; fi
    if [[ -f "logs/vllm_$1.pid" ]] && ! kill -0 "$(cat "logs/vllm_$1.pid")" 2>/dev/null; then
      echo "server on port $1 died:"; tail -n 40 "logs/vllm_$1.log"; return 1
    fi
    sleep 5
  done
  echo "port $1 did not come up — see logs/vllm_$1.log"; tail -n 40 "logs/vllm_$1.log"; return 1
}

stop_port() {
  if [[ -f "logs/vllm_$1.pid" ]]; then
    kill "$(cat "logs/vllm_$1.pid")" 2>/dev/null || true
    rm -f "logs/vllm_$1.pid"
  fi
  pkill -f "api_server.*--port $1( |$)" 2>/dev/null || true
  for _ in $(seq 1 30); do
    curl -s "localhost:$1/v1/models" > /dev/null || { echo "port $1 stopped"; sleep 3; return 0; }
    sleep 2
  done
}

case "${1:-}" in
  one)       start "$2" "$3" "$4" "${5:-}"; wait_up "$3" ;;
  pair)      start 0 8001 "$2" "${4:-}"; start 1 8002 "$3" "${5:-}"; wait_up 8001; wait_up 8002 ;;
  dp)        start 0 8001 "$2" "${3:-}"; start 1 8002 "$2" "${3:-}"; wait_up 8001; wait_up 8002 ;;
  tp)        start 0,1 8001 "$2" "${3:-}" --tensor-parallel-size 2; wait_up 8001 ;;
  stop_port) stop_port "$2" ;;
  stop)      pkill -f vllm.entrypoints.openai.api_server || true; rm -f logs/vllm_*.pid; sleep 5; echo stopped ;;
  status)    for p in 8001 8002 8003 8004; do
               printf "%s: " "$p"; curl -s "localhost:$p/v1/models" | head -c 200 || true; echo
             done; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv ;;
  *) echo "usage: bash kaggle/start_vllm.sh {one|pair|dp|tp|stop_port|stop|status} ..."; exit 1 ;;
esac
