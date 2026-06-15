#!/bin/bash
# vLLM — Qwen3.5-122B-A10B on 1x H200 for trajectory generation
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen3.5-122b}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.95}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI

echo "=== SHUTEN vLLM (H200) ==="
echo "Model: $MODEL_PATH"
echo "Port:  $PORT"
echo "Max len: $MAX_MODEL_LEN"
echo "=========================="

# BF16 weights ~233GB — use FP8 on H200 (141GB). Text-only skips vision RAM.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --quantization fp8 \
  --trust-remote-code \
  --max-num-seqs 16 \
  --disable-log-requests
