#!/bin/bash
# One-shot setup: SHUTEN on RunPod 2x A100 + Qwen3.6-27B
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash scripts/setup_a100_27b.sh
#
set -euo pipefail

REPO="${REPO:-/workspace/SHUTEN-D-JI}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/qwen3.6-27b}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.6-27B}"
REPO_URL="${REPO_URL:-https://github.com/MagistrTheOne/SHUTEN-D-JI.git}"

echo "=== SHUTEN A100 2x Setup (Qwen3.6-27B) ==="
echo "Repo:   $REPO"
echo "Model:  $MODEL_DIR"
echo "HF:     $HF_MODEL"
echo "========================================"

# uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p /workspace/models "$(dirname "$REPO")"

if [[ ! -d "$REPO/.git" ]]; then
  git clone "$REPO_URL" "$REPO"
else
  git -C "$REPO" pull --ff-only || true
fi

cd "$REPO"
chmod +x scripts/*.sh 2>/dev/null || true

if [[ ! -d .venv ]]; then
  uv venv .venv
fi

export PATH="$REPO/.venv/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"

uv pip install --upgrade pip
uv pip install torch torchvision torchaudio
uv pip install -e .
uv pip install "llamafactory>=0.9.5" bitsandbytes accelerate datasets peft trl transformers
uv pip install "huggingface_hub[cli]"
# Qwen3.6 needs vLLM >= 0.19 (see HF model card)
uv pip install "vllm>=0.19.0"

mkdir -p /workspace/models /workspace/outputs
mkdir -p "$REPO/data/trajectories"

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

# Bootstrap template data if empty
if [[ ! -s "$REPO/data/trajectories/sft_train.json" ]] || [[ "$(cat "$REPO/data/trajectories/sft_train.json" 2>/dev/null)" == "[]" ]]; then
  echo "[setup] Generating template trajectories..."
  python scripts/generate_data.py --num 320 --export || true
  python scripts/bootstrap_sft_json.py 2>/dev/null || true
fi

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
echo "=== Python ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
python -c "import llamafactory; print('llamafactory OK')"
python -c "import vllm; print('vllm', vllm.__version__)"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo ""
  echo "Starting model download (~55 GB) in background..."
  export HF_TOKEN="${HF_TOKEN:-}"
  nohup env HF_TOKEN="$HF_TOKEN" MODEL_DIR="$MODEL_DIR" HF_MODEL="$HF_MODEL" \
    bash "$REPO/scripts/download_qwen36_27b.sh" > /workspace/download_model.log 2>&1 &
  echo "Log: tail -f /workspace/download_model.log"
else
  echo ""
  echo "Model already at $MODEL_DIR"
fi

cat <<EOF

=== Setup complete ===

Monitor download:
  tail -f /workspace/download_model.log

After download finishes:
  bash scripts/serve_model_a100.sh          # vLLM TP=2, port 8000
  bash scripts/train_sft_a100.sh            # QLoRA SFT rank 64

Generate LLM trajectories (vLLM must be running):
  python scripts/generate_with_llm.py --num 200

Monitor training:
  tail -f /workspace/train_sft.log

EOF
