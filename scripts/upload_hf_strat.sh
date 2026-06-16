#!/bin/bash
# Upload SHUTEN STRAT LoRA to Hugging Face Hub
set -euo pipefail

ADAPTER_DIR="${1:-/workspace/outputs/shuten-sft-27b-v2.1}"
REPO_ID="${2:-NULLXES/SHUTEN-27B-STRAT-v0.1}"
CARD="${3:-/workspace/SHUTEN-D-JI/hub/SHUTEN-27B-STRAT-v0.1/README.md}"

source /root/shuten-venv/bin/activate

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: hf not logged in. Run: hf auth login"
  exit 1
fi

STAGING="/tmp/shuten-hf-upload"
rm -rf "$STAGING"
mkdir -p "$STAGING"

cp "$ADAPTER_DIR"/adapter_config.json "$STAGING/"
cp "$ADAPTER_DIR"/adapter_model.safetensors "$STAGING/"
cp "$CARD" "$STAGING/README.md"

echo "Uploading $STAGING -> $REPO_ID"
hf upload "$REPO_ID" "$STAGING" . \
  --repo-type model \
  --commit-message "feat: SHUTEN STRAT v0.1 LoRA adapter"

echo "Done: https://huggingface.co/$REPO_ID"
