#!/bin/bash
# PEFT QLoRA on H200 — explicit 4-bit (avoids LF bf16 OOM on 122B)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI
LOG="${LOG:-/workspace/train_sft.log}"

echo "=== SHUTEN SFT H200 PEFT ===" | tee "$LOG"
exec python -u scripts/train_sft_h200_peft.py configs/training/shuten_sft_h200.yaml 2>&1 | tee -a "$LOG"
