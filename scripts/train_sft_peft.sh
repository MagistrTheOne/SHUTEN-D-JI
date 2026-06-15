#!/bin/bash
# SHUTEN SFT — native PEFT+TRL, no LLaMA Factory (GPTQ 235B on 3xA100)

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI

echo "=== SHUTEN SFT (PEFT fallback) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Log: /workspace/train_sft_peft.log"
echo "=================================="

exec /workspace/SHUTEN-D-JI/.venv/bin/python -u scripts/train_sft_peft.py
