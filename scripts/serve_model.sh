#!/bin/bash
# Launch vLLM serving Qwen3-235B-A22B-GPTQ-Int4 on 3xA100
# This serves as the data generation backbone for SHUTEN training

set -e

MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen3-235b-gptq}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"

echo "=== SHUTEN Data Generation Server ==="
echo "Model: $MODEL_PATH"
echo "Tensor Parallel: $TP_SIZE"
echo "Max Context: $MAX_MODEL_LEN"
echo "GPU Utilization: $GPU_UTIL"
echo "Port: $PORT"
echo "======================================"

source /root/.local/bin/env
cd /workspace/SHUTEN-D-JI

uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --dtype auto \
    --trust-remote-code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --max-num-seqs 64 \
    --disable-log-requests
