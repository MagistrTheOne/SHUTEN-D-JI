#!/bin/bash
# venv on container local disk — avoids /workspace network volume I/O errors
set -euo pipefail

REPO=/workspace/SHUTEN-D-JI
VENV=/root/shuten-venv

cd "$REPO"
pkill -9 -f 'pip install' 2>/dev/null || true
rm -rf .venv "$VENV"
python3 -m venv "$VENV"
ln -sfn "$VENV" .venv

source "$VENV/bin/activate"
pip install -U pip setuptools wheel

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install vllm==0.19.1 llamafactory bitsandbytes accelerate datasets peft trl \
  transformers "huggingface_hub[cli]" openai aiohttp pydantic pyyaml numpy jsonschema \
  uuid7 rich typer

pip install -e "$REPO"

python -c "import torch, vllm; print('OK', torch.__version__, vllm.__version__, torch.cuda.device_count())"
