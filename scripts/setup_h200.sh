#!/bin/bash
# One-shot setup for SHUTEN on RunPod 1x H200
set -euo pipefail

REPO="${REPO:-/workspace/SHUTEN-D-JI}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/qwen3.5-122b}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-122B-A10B}"

echo "=== SHUTEN H200 Setup ==="
echo "Repo:   $REPO"
echo "Model:  $MODEL_DIR"
echo "========================="

# uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p /workspace/models "$(dirname "$REPO")"

if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/MagistrTheOne/SHUTEN-D-JI.git "$REPO"
else
  git -C "$REPO" pull --ff-only || true
fi

cd "$REPO"

if [[ ! -d .venv ]]; then
  uv venv .venv
fi

export PATH="$REPO/.venv/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"

# PyTorch + project deps (use system CUDA on RunPod)
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio
uv pip install -e .
uv pip install "llamafactory>=0.9.5" bitsandbytes accelerate datasets peft trl transformers

mkdir -p /workspace/models /workspace/outputs
mkdir -p "$REPO/data/trajectories"

# Minimal dataset_info for LLaMA Factory (data files may come from old pod or generation)
if [[ ! -f "$REPO/data/trajectories/dataset_info.json" ]]; then
  cat > "$REPO/data/trajectories/dataset_info.json" <<'EOF'
{
  "shuten_sft_train": {
    "file_name": "sft_train.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" }
  },
  "shuten_sft_eval": {
    "file_name": "sft_eval.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" }
  }
}
EOF
fi

if [[ ! -f "$REPO/data/trajectories/sft_train.json" ]]; then
  echo "[]" > "$REPO/data/trajectories/sft_train.json"
  echo "[]" > "$REPO/data/trajectories/sft_eval.json"
  echo "[WARN] Empty sft_train.json — copy from old pod or run generate_with_llm.py"
fi

chmod +x scripts/*.sh 2>/dev/null || true

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "=== Python ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import llamafactory; print('llamafactory OK')"
python -c "import bitsandbytes; print('bitsandbytes OK')"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo ""
  echo "Starting model download (~233GB) in background..."
  nohup bash "$REPO/scripts/download_qwen35_122b.sh" > /workspace/download_model.log 2>&1 &
  echo "Log: /workspace/download_model.log"
else
  echo ""
  echo "Model already at $MODEL_DIR"
fi

echo ""
echo "=== Done ==="
echo "Monitor download: tail -f /workspace/download_model.log"
echo "Start vLLM:       bash scripts/serve_model_h200.sh"
echo "Start SFT:        bash scripts/train_sft_h200.sh"
