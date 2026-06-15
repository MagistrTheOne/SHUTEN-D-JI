#!/bin/bash
# SHUTEN SFT — 235B GPTQ on 3xA100
# LLaMA Factory 0.9.5 auto-launches torchrun when GPU count > 1, even with
# FORCE_TORCHRUN=0. GPTQ needs a single process + device_map auto.
# Bypass llamafactory-cli and invoke launcher.py __main__ directly.

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/SHUTEN-D-JI

VENV_PY="/workspace/SHUTEN-D-JI/.venv/bin/python"
LAUNCHER="$("$VENV_PY" -c "import llamafactory, os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'launcher.py'))")"
CONFIG="${1:-configs/training/shuten_sft.yaml}"

echo "=== SHUTEN SFT Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Mode: single-process (launcher.py direct, no torchrun)"
echo "Python: $VENV_PY"
echo "Config: $CONFIG"
echo "==========================="

exec "$VENV_PY" "$LAUNCHER" "$CONFIG"
