# CODE FORGE v1 — Pod Runbook ($30 budget)

End-to-end sequence to produce `SHUTEN-CODE-LORA-v1`. Nothing here downloads
models locally; all GPU work happens on the pod. Merge happens off-GPU only after
the eval gate is met (spec section 8).

GPU: 1x A100 SXM 80GB (~$1.49/h -> ~20h). H200 only if time-critical.

Budget split: setup+smoke $2-3 / main SFT $12-16 / eval+bench $4-6 /
one re-run $4-5 / reserve $2-4.

---

## 0. Local (no GPU) — build the dataset that does not need the model

These run on a laptop in the repo venv and are already validated:

```bash
python -m scripts.forge_build_gold       # Layer A  -> data/code_forge/gold/
python -m scripts.forge_build_failure    # Layer C  -> data/code_forge/failure/
python -m src.forge.retention            # replay+general -> data/code_forge/replay/
python -m scripts.forge_build_eval       # eval split (prompts + hidden separate)
python -m scripts.forge_gen --selfcheck  # validate Layer B selection logic
python -m scripts.eval_code --selfcheck  # validate the metric harness
```

## 1. Pod setup ($1-2, ~30 min)

```bash
# venv already exists from v2; reuse it
source /root/shuten-venv/bin/activate
cd /workspace/SHUTEN-D-JI && git pull
pip install -e ".[dev]"            # ruff/mypy/pytest for the sandbox
# Docker for non-Python sandboxes (TS/SQL/C++/Bash) — optional for v1 Python core
docker pull node:20.18-bookworm-slim postgres:16.4-bookworm gcc:14.2-bookworm \
            koalaman/shellcheck-alpine:v0.10.0 python:3.11-slim-bookworm || true
```

## 2. Serve the base for generation ($ — overlaps with gen)

```bash
# Serve the merged SHUTEN-DOJI for Layer B drafting + eval generation
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/shuten-doji-27b \
  --served-model-name MagistrTheOne/SHUTEN-DOJI \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92 &
```

## 3. Layer B — Executable Synthetic (~$2-4)

```bash
python -m scripts.forge_gen --provider vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model MagistrTheOne/SHUTEN-DOJI \
  --target 220 --candidates 4
# -> data/code_forge/synthetic/synthetic.json (only gated, minimal candidates kept)
```

Stop the vLLM server before training to free VRAM.

## 4. Pack ($0)

```bash
python -m scripts.forge_pack
# Verify code_fraction ~0.75 and format_mix_actual near the 35/30/20/10/5 target
cat data/code_forge/pack/pack_stats.json
```

## 5. Smoke SFT ($1-2, sanity before spend)

```bash
llamafactory-cli train configs/training/shuten_code_sft.yaml \
  max_samples=30 num_train_epochs=1 save_steps=15 eval_steps=15 \
  output_dir=/workspace/outputs/shuten-code-smoke
# PASS if: loss decreases, a checkpoint saves, no OOM. If OOM -> path (2) 8-bit base.
```

## 6. Main SFT ($12-16, ~8-11h)

```bash
llamafactory-cli train configs/training/shuten_code_sft.yaml
# -> adapter at /workspace/outputs/shuten-code-lora-v1
```

## 7. Eval ($4-6) — identical sampling/sandbox for every model

```bash
# Re-serve base with the LoRA adapter for the candidate
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/shuten-doji-27b \
  --enable-lora --lora-modules code=/workspace/outputs/shuten-code-lora-v1 \
  --served-model-name shuten-code-lora --dtype bfloat16 --max-model-len 8192 &

# Baseline (no adapter) first, then candidate, then compare
python -m scripts.eval_code --provider vllm --model MagistrTheOne/SHUTEN-DOJI \
  --label base --k 3 --out data/code_forge/eval/base.json
python -m scripts.eval_code --provider vllm --model code \
  --label shuten_code_lora --k 3 --baseline data/code_forge/eval/base.json \
  --out data/code_forge/eval/shuten_code_lora.json

# Strategic + general regression: re-run the existing v2 side-by-side eval and
# confirm Constitution structure compliance did not drop.
```

## 8. Gate + merge (off-GPU, only on a win)

Apply `docs/CODE_FORGE_V1_SHIP_GATE.md`. If all gates pass:

```bash
# merge LoRA into base off-GPU, then upload (reuse the v2 merge script pattern)
bash scripts/merge_and_upload_shuten_doji.sh  # adapted for shuten-code-lora-v1
```

If a gate fails: keep the adapter as an artifact, log the failing metric, and use
the rejected-candidate archive to seed a v2 DPO/GRPO pass — do NOT merge.

---

## Cost guardrails
- Kill vLLM before training and training before eval; never hold two large jobs.
- Checkpoint every 50 steps; a crash should cost minutes, not the run.
- One re-run is budgeted. A second failure means stop and re-plan, not re-spend.
