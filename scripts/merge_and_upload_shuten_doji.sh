#!/bin/bash
# Merge SHUTEN LoRA into Qwen3.6-27B and upload to Hugging Face Hub.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/workspace/models/qwen3.6-27b}"
ADAPTER="${ADAPTER:-/workspace/outputs/shuten-sft-h200-v2}"
EXPORT_DIR="${EXPORT_DIR:-/workspace/outputs/NULLXES-SHUTEN-DOJI-merged}"
REPO_ID="${REPO_ID:-MagistrTheOne/SHUTEN-DOJI}"
CARD="${CARD:-/workspace/SHUTEN-D-JI/hub/NULLXES-SHUTEN-DOJI/README.md}"
PRIVATE="${PRIVATE:-false}"

source /root/shuten-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/SHUTEN-D-JI

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: hf not logged in. On pod run: hf auth login"
  echo "       (token needs write access — create at huggingface.co/settings/tokens)"
  exit 1
fi

echo "=== SHUTEN merge + HF upload ==="
echo "Base:    $BASE_MODEL"
echo "Adapter: $ADAPTER"
echo "Export:  $EXPORT_DIR"
echo "Hub:     $REPO_ID"
echo "User:    $(hf auth whoami 2>/dev/null | head -1 || true)"
echo "================================"

# Free GPU (vLLM holds VRAM)
pkill -9 -f 'vllm.entrypoints|VLLM::EngineCore' 2>/dev/null || true
sleep 5

test -f "$ADAPTER/adapter_model.safetensors" || { echo "Missing adapter"; exit 1; }
test -f "$BASE_MODEL/config.json" || { echo "Missing base model"; exit 1; }

AVAIL_GB=$(df -BG /workspace | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [ "$AVAIL_GB" -lt 120 ] 2>/dev/null; then
  echo "WARN: less than 120GB free on /workspace — merge needs ~60GB export"
fi

rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

echo "[1/4] Merging LoRA into base (llamafactory export)..."
llamafactory-cli export \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_name_or_path "$ADAPTER" \
  --template qwen3 \
  --finetuning_type lora \
  --export_dir "$EXPORT_DIR" \
  --export_size 5

echo "[2/4] Copy model card..."
cp "$CARD" "$EXPORT_DIR/README.md"

echo "[3/4] Verify export..."
test -f "$EXPORT_DIR/config.json" || { echo "Export failed — no config.json"; exit 1; }
du -sh "$EXPORT_DIR"
ls "$EXPORT_DIR" | head -20

echo "[4/4] Upload to Hugging Face (large folder, resumable)..."
PRIV_FLAG=""
[ "$PRIVATE" = "true" ] && PRIV_FLAG="--private"

hf repo create "$REPO_ID" --type model $PRIV_FLAG --exist-ok 2>/dev/null || true

hf upload-large-folder "$REPO_ID" "$EXPORT_DIR" \
  --repo-type model \
  $PRIV_FLAG

echo ""
echo "Done: https://huggingface.co/$REPO_ID"
echo "Serve: vllm serve $REPO_ID --trust-remote-code --language-model-only --gdn-prefill-backend triton"
