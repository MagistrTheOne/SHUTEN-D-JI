#!/bin/bash
# vLLM — Qwen3.6-27B + SHUTEN LoRA on 2x A100
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen3.6-27b}"
LORA_PATH="${LORA_PATH:-/workspace/outputs/shuten-sft-27b-v2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP_SIZE="${TP_SIZE:-2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PATH="/root/shuten-venv/bin:${PATH}"

cd /workspace/SHUTEN-D-JI

echo "=== SHUTEN vLLM + LoRA ==="
echo "Base:  $MODEL_PATH"
echo "LoRA:  $LORA_PATH"
echo "Port:  $PORT"
echo "=========================="

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --enable-lora \
  --lora-modules "shuten=${LORA_PATH}" \
  --max-lora-rank 64 \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --trust-remote-code \
  --language-model-only \
  --max-num-seqs 16
