#!/bin/bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/qwen3.5-122b}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-122B-A10B}"

source "${HOME}/.local/bin/env" 2>/dev/null || true
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

mkdir -p "$MODEL_DIR"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  uv pip install "huggingface_hub[cli]"
  HF_CLI=hf
fi

echo "Downloading $HF_MODEL -> $MODEL_DIR"
$HF_CLI download "$HF_MODEL" --local-dir "$MODEL_DIR"
echo "Download complete: $MODEL_DIR"
