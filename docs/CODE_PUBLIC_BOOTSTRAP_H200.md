# SHUTEN-CODE Public Bootstrap — H200 Runbook

Goal: train `SHUTEN-CODE-PUBLIC-LORA-v1` on top of the already merged
`MagistrTheOne/SHUTEN-DOJI` using public code-instruction datasets plus SHUTEN
retention and verified CODE FORGE anchors.

This path replaces slow Layer B synthetic generation as the main data source.
Execution-verified CODE FORGE remains a quality anchor and eval gate.

## 1. Pod Setup

```bash
cd /workspace/SHUTEN-D-JI
git pull
python -m venv /root/forge-venv
source /root/forge-venv/bin/activate
pip install -U pip
pip install -e ".[dev,train]"
pip install -U llamafactory
```

Model path expected by configs:

```bash
/workspace/models/shuten-doji-27b
```

## 2. Build Local Anchors

```bash
source /root/forge-venv/bin/activate
cd /workspace/SHUTEN-D-JI

python -m scripts.forge_smoke
python -m scripts.forge_build_gold
python -m scripts.forge_build_failure
python -m src.forge.retention
python -m scripts.forge_build_eval
```

## 3. Import Public Code Data

Conservative starter mix (~15k attempted rows):

```bash
python -m scripts.forge_import_public \
  --alpaca iamtarun/code_instructions_120k_alpaca:8000 \
  --alpaca iamtarun/python_code_instructions_18k_alpaca:5000 \
  --messages cfahlgren1/react-code-instructions:2000 \
  --out data/code_forge/public/public_code.json

cat data/code_forge/public/public_stats.json
```

If bandwidth/time is tight, use a small smoke import first:

```bash
python -m scripts.forge_import_public \
  --alpaca iamtarun/python_code_instructions_18k_alpaca:200 \
  --messages cfahlgren1/react-code-instructions:100
```

## 4. Pack for LlamaFactory

```bash
python -m scripts.forge_pack
cat data/code_forge/pack/pack_stats.json
```

Expected: `public_code_rows` dominates, with replay/general retained.

## 5. Smoke SFT

```bash
llamafactory-cli train configs/training/shuten_code_public_sft_h200.yaml \
  max_samples=64 num_train_epochs=1 save_steps=20 eval_steps=20 \
  output_dir=/workspace/outputs/shuten-code-public-smoke
```

Pass criteria: no OOM, loss moves down, checkpoint saves.

## 6. Main SFT

```bash
llamafactory-cli train configs/training/shuten_code_public_sft_h200.yaml
```

Output:

```bash
/workspace/outputs/shuten-code-public-lora-v1
```

## 7. Quick Eval

Serve base + LoRA:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/shuten-doji-27b \
  --enable-lora \
  --lora-modules code=/workspace/outputs/shuten-code-public-lora-v1 \
  --served-model-name shuten-code-public \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --port 8000
```

In another shell:

```bash
python -m scripts.eval_code --provider vllm \
  --model /workspace/models/shuten-doji-27b \
  --label base --k 1 --out data/code_forge/eval/base.json

python -m scripts.eval_code --provider vllm \
  --model code \
  --label shuten_code_public_lora --k 1 \
  --baseline data/code_forge/eval/base.json \
  --out data/code_forge/eval/shuten_code_public_lora.json
```

Do not merge until `scripts.forge_ship_gate` passes and strategic regression is
checked separately.
