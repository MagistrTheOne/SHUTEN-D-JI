#!/bin/bash
# vLLM — Qwen3.6-27B on 2x A100 (TP=2), text-only for SHUTEN trajectories
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen3.6-27b}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP_SIZE="${TP_SIZE:-2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PATH="/workspace/SHUTEN-D-JI/.venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI

echo "=== SHUTEN vLLM (2x A100) ==="
echo "Model:   $MODEL_PATH"
echo "Port:    $PORT"
echo "Max len: $MAX_MODEL_LEN"
echo "TP:      $TP_SIZE"
echo "============================="

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --language-model-only \
  --max-num-seqs 32 \
  --disable-log-requests
