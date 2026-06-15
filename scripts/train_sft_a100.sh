#!/bin/bash
# QLoRA SFT — Qwen3.6-27B on 1x A100 (4-bit via LLaMA Factory)
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI
CONFIG="${1:-configs/training/shuten_sft_a100_27b.yaml}"
LOG="${LOG:-/workspace/train_sft.log}"

echo "=== SHUTEN SFT A100 (Qwen3.6-27B) ===" | tee "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "GPU:    $CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

exec llamafactory-cli train "$CONFIG" 2>&1 | tee -a "$LOG"
