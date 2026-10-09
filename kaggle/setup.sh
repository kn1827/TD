#!/usr/bin/env bash
# Kaggle environment setup (called by run_kaggle.ipynb; lives in the repo so fixes arrive with `git pull`).
#
#   bash kaggle/setup.sh [VLLM_VERSION]
#
# 1. installs vLLM (pinned if a version is given). vLLM brings its own torch build;
# 2. removes torchaudio: Kaggle's preinstalled torchaudio is built for another CUDA version than the
#    torch vLLM installs, and transformers imports it when present -> "PyTorch and TorchAudio were
#    compiled with different CUDA versions". Nothing here uses audio;
# 3. installs requirements.txt;
# 4. prints versions and checks that torch really computes on the GPU, so a CUDA / driver mismatch
#    stops here and not in the middle of the run.
set -euo pipefail
cd "$(dirname "$0")/.."

VER=${1:-}
if [[ -n "$VER" ]]; then
  pip install -q "vllm==$VER" 2>&1 | tail -n 3
else
  pip install -q vllm 2>&1 | tail -n 3
fi
pip uninstall -y -q torchaudio 2>/dev/null || true
pip install -q -r requirements.txt 2>&1 | tail -n 3

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
python - <<'EOF'
import sys
import torch, transformers, vllm
print(f"python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"vllm {vllm.__version__} | transformers {transformers.__version__}")
if not torch.cuda.is_available():
    sys.exit("torch cannot use the GPU: the NVIDIA driver is probably too old for this torch CUDA "
             "build. Set VLLM_VERSION to an older vLLM release (one built on CUDA 12.x) and rerun.")
x = torch.arange(4.0, device="cuda")
assert float((x * 2).sum()) == 12.0
print("GPU check ok:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF
df -h /tmp /kaggle/working | cat
