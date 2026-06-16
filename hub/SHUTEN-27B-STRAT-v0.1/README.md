---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
tags:
  - lora
  - strategic-planning
  - decision-intelligence
  - nullxes
  - shuten
language:
  - en
pipeline_tag: text-generation
library_name: peft
---

# SHUTEN-27B-STRAT-v0.1

**SHUTEN** (Strategic Human-like Unified Tactical Evaluation Network) — strategic intelligence by **NULLXES DAI**.

This is **not a chatbot**. It is a **Strategic Planning Model** (Phase 1 cluster: **STRAT**).

## Model clusters (roadmap)

| Cluster | Codename | Capability | Status |
|---------|----------|------------|--------|
| **STRAT** | Planning | World State → Analysis → Action Plan → Expected Outcome | **v0.1 (this release)** |
| **IMPACT** | Consequence | State + Action → Impact(A/B/C), confidence, 2nd-order effects | Phase 2 |
| **NATIVE** | SHUTEN MoE | Custom NULLXES architecture, Qwen as teacher only | Phase 3 |

Defense, business, logistics, and markets are **environments** — not separate products.

## Base model

- **Foundation:** [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) pretrained weights
- **Method:** QLoRA SFT (rank 32)
- **Qwen role:** teacher / parent benchmark — **not** the final SHUTEN architecture

## Training data

- **Synthetic Strategic Experience Generator** (NULLXES factory)
  - StateGenerator → ScenarioGenerator → Environment → OutcomeEvaluator
- **285 LLM-enriched trajectories** (business, logistics, markets)
- Format: World State + Objective → structured strategic analysis + phased actions

## Metrics (v2.1)

| Metric | Value |
|--------|-------|
| Train examples | 270 |
| Eval examples | 15 |
| Eval loss | ~0.25 |

Side-by-side vs base Qwen: improved phased operational plans, explicit metric interconnections on World State prompts.

## Usage (vLLM)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B \
  --enable-lora \
  --lora-modules shuten=./ \
  --max-lora-rank 32 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --language-model-only
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "shuten",
    "messages": [{"role": "user", "content": "[SHUTEN business]\n\nWorld State:\nRevenue down 18%.\n...\n\nObjective:\nRestore profitability within 60 days.\n\nAnalyze and recommend actions."}],
    "max_tokens": 1200,
    "temperature": 0.2
  }'
```

## Limitations

- Phase 1 STRAT only — **impact prediction** (IMPACT cluster) not yet trained
- Small synthetic corpus (~285 episodes) — MVP release
- Requires Qwen3.6-27B base weights separately

## Citation

```
@misc{shuten-strat-v0.1,
  title={SHUTEN-27B-STRAT-v0.1: Strategic Planning LoRA by NULLXES DAI},
  author={NULLXES DAI},
  year={2026},
  note={Base: Qwen3.6-27B. Cluster: STRAT Phase 1.}
}
```
