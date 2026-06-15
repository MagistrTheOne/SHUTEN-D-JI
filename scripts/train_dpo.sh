#!/bin/bash
# SHUTEN DPO — 235B GPTQ on 3xA100 (after SFT adapter exists)

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export FORCE_TORCHRUN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/SHUTEN-D-JI

echo "=== SHUTEN DPO Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Mode: FORCE_TORCHRUN=0 (device_map auto)"
echo "Config: configs/training/shuten_dpo.yaml"
echo "==========================="

/workspace/SHUTEN-D-JI/.venv/bin/llamafactory-cli train configs/training/shuten_dpo.yaml
