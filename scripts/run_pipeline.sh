#!/bin/bash
# SHUTEN Full Training Pipeline
# Run on RunPod 3xA100 SXM
#
# Stages:
#   1. Serve Qwen3-235B-GPTQ for synthetic data generation
#   2. Generate SHUTEN training corpus
#   3. Stop inference, run SFT via LLaMA Factory
#   4. Run DPO alignment
#   5. Export final SHUTEN model
#
# Usage: bash scripts/run_pipeline.sh [--skip-gen] [--num 50000]

set -e

NUM_TRAJECTORIES="${NUM:-10000}"
SKIP_GEN=false
PORT=8000

for arg in "$@"; do
    case $arg in
        --skip-gen) SKIP_GEN=true ;;
        --num=*) NUM_TRAJECTORIES="${arg#*=}" ;;
    esac
done

echo "============================================"
echo "  SHUTEN Training Pipeline"
echo "  NULLXES DAI"
echo "============================================"
echo "  Trajectories: $NUM_TRAJECTORIES"
echo "  Skip generation: $SKIP_GEN"
echo "============================================"
echo ""

source /root/.local/bin/env
cd /workspace/SHUTEN-D-JI

# ==========================================
# STAGE 1: DATA GENERATION
# ==========================================
if [ "$SKIP_GEN" = false ]; then
    echo "[STAGE 1] Starting vLLM server for data generation..."
    bash scripts/serve_model.sh &
    VLLM_PID=$!

    echo "Waiting for vLLM to load model (this takes 3-5 minutes)..."
    sleep 30
    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "vLLM server ready!"
            break
        fi
        echo "  Waiting... ($i/60)"
        sleep 10
    done

    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "ERROR: vLLM server failed to start!"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi

    echo ""
    echo "[STAGE 1] Generating synthetic training data..."
    uv run python scripts/generate_with_llm.py \
        --num "$NUM_TRAJECTORIES" \
        --output data/trajectories \
        --port $PORT \
        --batch-size 16 \
        --concurrency 32

    echo ""
    echo "[STAGE 1] Data generation complete. Stopping vLLM..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null || true
    sleep 10

    echo "Clearing GPU memory..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

# ==========================================
# STAGE 2: SFT TRAINING
# ==========================================
echo ""
echo "[STAGE 2] Starting SFT training via LLaMA Factory..."
echo "  Config: configs/training/shuten_sft.yaml"
echo ""

uv run llamafactory-cli train configs/training/shuten_sft.yaml

echo ""
echo "[STAGE 2] SFT complete!"

# ==========================================
# STAGE 3: DPO ALIGNMENT
# ==========================================
echo ""
echo "[STAGE 3] Starting DPO alignment..."
echo "  Config: configs/training/shuten_dpo.yaml"
echo ""

uv run llamafactory-cli train configs/training/shuten_dpo.yaml

echo ""
echo "[STAGE 3] DPO complete!"

# ==========================================
# STAGE 4: EXPORT
# ==========================================
echo ""
echo "[STAGE 4] Exporting final SHUTEN model..."

uv run llamafactory-cli export \
    --model_name_or_path /workspace/models/qwen3-235b-gptq \
    --adapter_name_or_path /workspace/outputs/shuten-dpo \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir /workspace/outputs/shuten-v0.1-merged \
    --export_size 5

echo ""
echo "============================================"
echo "  SHUTEN v0.1 Training Complete!"
echo "============================================"
echo "  SFT checkpoint: /workspace/outputs/shuten-sft/"
echo "  DPO checkpoint: /workspace/outputs/shuten-dpo/"
echo "  Merged model:   /workspace/outputs/shuten-v0.1-merged/"
echo "============================================"
echo ""
echo "To serve SHUTEN:"
echo "  vllm serve /workspace/outputs/shuten-v0.1-merged --tensor-parallel-size 3"
