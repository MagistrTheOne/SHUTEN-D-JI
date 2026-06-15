#!/bin/bash
# vLLM — Qwen3.5-122B-A10B on 1x H200 for trajectory generation
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen3.5-122b}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.92}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI

echo "=== SHUTEN vLLM (H200) ==="
echo "Model: $MODEL_PATH"
echo "Port:  $PORT"
echo "Max len: $MAX_MODEL_LEN"
echo "=========================="

# Text-only: skip vision encoder memory (multimodal model)
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --disable-log-requests
