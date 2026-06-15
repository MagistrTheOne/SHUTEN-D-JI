#!/bin/bash
# SHUTEN DPO — 235B GPTQ on 3xA100 (after SFT adapter exists)

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NPROC_PER_NODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI

CONFIG="${1:-configs/training/shuten_dpo.yaml}"
TORCHRUN="/workspace/SHUTEN-D-JI/.venv/bin/torchrun"

echo "=== SHUTEN DPO Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Mode: torchrun nproc=1 + device_map=auto"
echo "Config: $CONFIG"
echo "==========================="

exec "$TORCHRUN" --nnodes 1 --node_rank 0 --nproc_per_node 1 \
  scripts/lf_train_entry.py "$CONFIG"
