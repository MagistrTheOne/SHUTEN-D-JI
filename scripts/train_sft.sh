#!/bin/bash
# SHUTEN SFT — 235B GPTQ on 3xA100
# GPTQ is incompatible with DeepSpeed ZeRO-3/FSDP + torchrun DDP.
# Single process + device_map auto shards the model across all visible GPUs.

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export FORCE_TORCHRUN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/SHUTEN-D-JI

echo "=== SHUTEN SFT Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Mode: FORCE_TORCHRUN=0 (device_map auto)"
echo "Config: configs/training/shuten_sft.yaml"
echo "==========================="

/workspace/SHUTEN-D-JI/.venv/bin/llamafactory-cli train configs/training/shuten_sft.yaml
