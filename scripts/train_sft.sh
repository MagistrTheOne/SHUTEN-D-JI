#!/bin/bash
# SHUTEN SFT — 235B GPTQ on 3xA100
# PTQ GPTQ requires single-process + device_map auto (see scripts/lf_gptq_mp.py).

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/SHUTEN-D-JI

VENV_PY="/workspace/SHUTEN-D-JI/.venv/bin/python"
CONFIG="${1:-configs/training/shuten_sft.yaml}"

echo "=== SHUTEN SFT Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Mode: GPTQ model-parallel (single process, device_map=auto)"
echo "Python: $VENV_PY"
echo "Config: $CONFIG"
echo "==========================="

exec "$VENV_PY" scripts/train_sft_gptq.py "$CONFIG"
