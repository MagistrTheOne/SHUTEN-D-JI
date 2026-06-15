#!/bin/bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/qwen3.6-27b}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.6-27B}"

source "${HOME}/.local/bin/env" 2>/dev/null || true
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  export HF_TOKEN
fi

mkdir -p "$MODEL_DIR"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  uv pip install "huggingface_hub[cli]"
  HF_CLI=hf
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  $HF_CLI login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

echo "Downloading $HF_MODEL -> $MODEL_DIR (~55 GB BF16)"
$HF_CLI download "$HF_MODEL" --local-dir "$MODEL_DIR"
echo "Download complete: $MODEL_DIR"
