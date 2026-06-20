---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
tags:
  - strategic-planning
  - decision-intelligence
  - operational-intelligence
  - nullxes
  - shuten
  - merged-lora
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# NULLXES SHUTEN-DŌJI

**SHUTEN** — strategic intelligence system by **NULLXES DAI**.

This is **not a chatbot**. It produces structured operational intelligence:

`State → Causes → Options → Impact → Future State → Confidence`

Merged weights: **Qwen3.6-27B + SHUTEN Constitution SFT v2** (H200 MVP).

## Roadmap status

| Stage | Item | Status |
|-------|------|--------|
| Infra | H200 train → vLLM → LoRA serve | ✅ Done |
| Data v1 | bootstrap trajectories | ✅ Done (legacy — do not use for SFT) |
| SFT v1 | smoke / warm-start | ✅ Done — planner **failed** (Step N poison) |
| SFT v2 | 50 Constitution gold examples | ✅ Done |
| Eval A–H | side-by-side vs base Qwen | ✅ Done — v2 **0 poison**, **7/8** struct wins |
| SFT v2.1 | 110–130 gold + reviewed LLM + failure cases | ⬜ Next |
| Eval v2 | 20–30 held-out planner cases | ⬜ Next |
| DPO | preference pairs (chosen vs rejected plans) | ⬜ After eval v2 pass |
| IMPACT cluster | consequence prediction fine-tune | ⬜ Phase 2 |
| NATIVE MoE | custom NULLXES architecture | ⬜ Phase 3 |

## Base model

| | |
|---|---|
| Foundation | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Method | QLoRA SFT (rank 64, alpha 128) → **merged** into full weights |
| Train data | 50 Constitution ShareGPT examples (no bootstrap poison) |
| Checkpoint | `shuten-sft-h200-v2` on RunPod H200 |

## Training metrics (SFT v2)

| Metric | Value |
|--------|-------|
| Train examples | 50 |
| Eval examples | 10 |
| Epochs | 3 |
| Train loss | 1.03 |
| Eval loss | 0.49 |
| Trainable params | 41.9M LoRA (merged at export) |

## Eval metrics (A–H, side-by-side)

| Metric | qwen_base | SHUTEN v2 |
|--------|-----------|-----------|
| Poison (`Step N:`, `tool_use`, …) | 0/8 | **0/8** |
| Avg structure markers | 1.5 | **4.12** |
| Wins vs base (structure) | — | **7/8** |
| Avg output length | 5139 | 4684 |

**Verdict:** Constitution SFT removes bootstrap action-trace failure. Content quality still MVP — v2.1 dataset iteration required before DPO.

## Usage (vLLM — no LoRA adapter needed)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model NULLXES/SHUTEN-DOJI \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --trust-remote-code \
  --language-model-only \
  --gdn-prefill-backend triton
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "NULLXES/SHUTEN-DOJI",
    "messages": [
      {"role": "system", "content": "You are SHUTEN, strategic intelligence by NULLXES DAI. Reason: State → Causes → Options → Impact → Future State → Confidence."},
      {"role": "user", "content": "[SHUTEN business]\n\nWorld State:\nRevenue down 18%. Backlog up 42%.\n\nObjective:\nRestore EBITDA margin >12% in 90d.\n\nRequired Output:\nState → Causes → Options → Impact → Future State → Confidence"}
    ],
    "max_tokens": 1200,
    "temperature": 0.3
  }'
```

## Limitations

- MVP release — 50 training examples only
- May still prefix with Qwen-style reasoning traces
- Not trained for DPO / impact cluster yet
- Requires ~54GB VRAM at bf16 (single H200 / A100 80GB)

## Links

- Code: [github.com/MagistrTheOne/SHUTEN-D-JI](https://github.com/MagistrTheOne/SHUTEN-D-JI)
- Base: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)

## Citation

```bibtex
@misc{nullxes-shuten-doji-v2,
  title={NULLXES SHUTEN-DŌJI: Strategic Intelligence (Constitution SFT v2)},
  author={NULLXES DAI},
  year={2026},
  note={Merged Qwen3.6-27B + Constitution LoRA. MVP eval 7/8 struct wins, 0 poison.}
}
```
