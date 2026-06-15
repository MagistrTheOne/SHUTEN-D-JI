#!/bin/bash
# SHUTEN DPO — 235B GPTQ on 3xA100 (after SFT adapter exists)

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/SHUTEN-D-JI

VENV_PY="/workspace/SHUTEN-D-JI/.venv/bin/python"
LAUNCHER="$("$VENV_PY" -c "import llamafactory, os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'launcher.py'))")"
CONFIG="${1:-configs/training/shuten_dpo.yaml}"

echo "=== SHUTEN DPO Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Mode: single-process (launcher.py direct, no torchrun)"
echo "Python: $VENV_PY"
echo "Config: $CONFIG"
echo "==========================="

exec "$VENV_PY" "$LAUNCHER" "$CONFIG"
