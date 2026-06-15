#!/bin/bash
# Paste this entire block on a fresh RunPod 2x A100 pod (or run this file).
#
# Required env:
#   HF_TOKEN — Hugging Face token (optional for public Qwen3.6-27B, but recommended)
#
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running}"
export REPO="${REPO:-/workspace/SHUTEN-D-JI}"
export REPO_URL="${REPO_URL:-https://github.com/MagistrTheOne/SHUTEN-D-JI.git}"

mkdir -p /workspace
if [[ ! -d "$REPO/.git" ]]; then
  git clone "$REPO_URL" "$REPO"
else
  git -C "$REPO" fetch origin && git -C "$REPO" pull --ff-only || true
fi

exec bash "$REPO/scripts/setup_a100_27b.sh"
