#!/bin/bash
# SHUTEN SFT — Qwen3.5-122B QLoRA on 1x H200 via LLaMA Factory
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI

CONFIG="${1:-configs/training/shuten_sft_h200.yaml}"
LOG="${LOG:-/workspace/train_sft.log}"

echo "=== SHUTEN SFT (H200 QLoRA) ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"
echo "Log: $LOG"
echo "==============================="

exec llamafactory-cli train "$CONFIG" 2>&1 | tee -a "$LOG"
