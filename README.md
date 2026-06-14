# SHUTEN-DŌJI

**Synthetic Strategic Intelligence Factory**

By **NULLXES DAI** (Digital AI)

---

## What This Is

SHUTEN-DŌJI is NOT a chatbot. It is NOT another LLM wrapper.

It is an **intelligence-production system** that generates:
- Strategic decision trajectories
- Trained model checkpoints
- Evaluation environments
- Reward functions
- Self-improving training loops

The model is a byproduct. The factory is the asset.

## Architecture

```
Input: Domain specs + World states + Objectives
  ↓
State Generator → Scenario Generator → Agent Simulator → Environment
  ↓
Outcome Evaluator → Trajectory Store → Training Signal
  ↓
Output: Trained model specialized for strategic intelligence
```

## Core Components

| Component | Purpose | Location |
|---|---|---|
| Custom MoE Model | 132B/14B cognitive-specialized architecture | `src/model/` |
| State Generator | Produces machine-readable world states | `src/factory/state_generator.py` |
| Scenario Generator | Generates future branches with causal chains | `src/factory/scenario_generator.py` |
| Agent Simulator | Creates cognitive agents (analyst, planner, etc.) | `src/factory/agent_simulator.py` |
| Environment | Gym-like strategic decision environments | `src/factory/environment.py` |
| Outcome Evaluator | Verifiable scoring for RL signal | `src/factory/outcome_evaluator.py` |
| Trajectory Store | Storage + export for LLaMA Factory | `src/factory/trajectory_store.py` |
| LLaMA Factory Adapter | Training pipeline integration | `src/training/` |

## Model Architecture

Custom MoE transformer — not based on any pretrained architecture.

- **Total params:** ~132B
- **Active params:** ~14B
- **Experts:** 128 routed + 2 shared
- **Active per token:** 6
- **Context:** 128K
- **Attention:** GQA (48 heads, 8 KV heads)
- **Key innovation:** Cognitive expert groups + typed token routing + persistent memory

Training via **LLaMA Factory** (LoRA → DPO → PPO pipeline).

## Installation

### via uv (recommended)

[uv](https://docs.astral.sh/uv/) — fast Python package manager from Astral.

```bash
# Install uv (if not installed)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install project
uv sync                     # install all dependencies
uv sync --dev               # + dev tools (pytest, ruff, mypy)

# Run scripts via uv
uv run python scripts/generate_data.py --num 10000 --export
uv run pytest tests/

# Train via LLaMA Factory
uv run llamafactory-cli train configs/training/phase1_sft.yaml
```

### via pip (alternative)

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Generate training data
uv run python scripts/generate_data.py --num 10000 --export

# 2. Train Phase 1 (SFT via LLaMA Factory)
uv run llamafactory-cli train configs/training/phase1_sft.yaml

# 3. Train Phase 3 (DPO)
uv run llamafactory-cli train configs/training/phase3_dpo.yaml
```

## Training Phases

| Phase | Method | Config |
|---|---|---|
| 1. Trajectory SFT | Supervised fine-tuning on strategic trajectories | `configs/training/phase1_sft.yaml` |
| 2. Tool-Use | SFT on tool interaction demonstrations | `configs/training/phase2_tools.yaml` |
| 3. DPO | Preference optimization (good vs bad trajectories) | `configs/training/phase3_dpo.yaml` |
| 4. PPO | Reinforcement learning with verifiable rewards | `configs/training/phase4_ppo.yaml` |

## Project Structure

```
NULLXES SHUTEN-DŌJI/
├── ARCHITECTURE.md          # Full architecture document
├── README.md                # This file
├── pyproject.toml           # Dependencies + build config
├── configs/
│   ├── model/               # Model architecture configs
│   ├── training/            # LLaMA Factory training YAML
│   └── factory/             # Data generation pipeline config
├── src/
│   ├── model/               # Custom MoE architecture
│   │   ├── architecture.py  # Main model + config
│   │   ├── routing.py       # Cognitive router (typed token routing)
│   │   ├── experts.py       # Expert layers (SwiGLU)
│   │   └── memory.py        # Persistent working memory
│   ├── factory/             # Data generation subsystems
│   │   ├── state_generator.py
│   │   ├── scenario_generator.py
│   │   ├── agent_simulator.py
│   │   ├── environment.py
│   │   ├── outcome_evaluator.py
│   │   └── trajectory_store.py
│   ├── training/            # Training pipeline
│   │   ├── llamafactory_adapter.py
│   │   └── data_pipeline.py
│   └── evaluation/          # Metrics and benchmarks
│       └── metrics.py
├── data/
│   ├── schemas/             # JSON schemas for data validation
│   ├── seeds/               # Seed data from real sources
│   └── trajectories/        # Generated training data
├── scripts/
│   ├── generate_data.py     # Run data factory
│   └── train.sh             # Training orchestration
└── tests/
    ├── test_state_generator.py
    └── test_model.py
```

## Design Principles

1. **Factory first, model second.** The data generation system is the core asset.
2. **Verification closes the loop.** Every trajectory has measurable outcomes.
3. **Own architecture.** Custom MoE designed for strategic cognition, not borrowed.
4. **LLaMA Factory for training.** Industry-standard tooling, no custom training infra.
5. **Synthetic data at scale.** No dependency on expensive human annotation.
6. **Progressive complexity.** Start simple, scale difficulty as capability grows.

## Requirements

- **uv** (recommended) or pip
- Python 3.10+
- PyTorch 2.4+
- LLaMA Factory 0.9+
- CUDA 12.x (for training)
- 80GB+ VRAM per GPU (for full model), 24GB+ (for LoRA)

## License

Apache 2.0
