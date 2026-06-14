"""
LLaMA Factory Adapter — integration layer for training SHUTEN-DŌJI via LLaMA Factory.

LLaMA Factory handles:
  - Distributed training orchestration
  - LoRA / QLoRA / full fine-tuning
  - DPO / PPO / RLHF pipelines
  - Data loading and formatting
  - Evaluation during training

This adapter:
  - Registers SHUTEN-DŌJI as a custom model in LLaMA Factory
  - Converts trajectory data to LLaMA Factory dataset format
  - Generates training configs for each phase
  - Provides hooks for custom reward computation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LlamaFactoryDatasetEntry:
    """Single entry in LLaMA Factory dataset_info.json."""
    name: str
    file_name: str
    formatting: str = "sharegpt"  # sharegpt | alpaca
    columns: Optional[dict] = None
    tags: Optional[dict] = None


def generate_dataset_info(
    datasets: list[LlamaFactoryDatasetEntry],
    output_path: Path,
) -> Path:
    """
    Generate dataset_info.json for LLaMA Factory.
    This file tells LLaMA Factory where to find and how to parse training data.
    """
    info = {}
    for ds in datasets:
        entry = {"file_name": ds.file_name, "formatting": ds.formatting}
        if ds.columns:
            entry["columns"] = ds.columns
        if ds.tags:
            entry["tags"] = ds.tags
        info[ds.name] = entry

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return output_path


def generate_training_config(
    phase: str,
    model_name_or_path: str,
    dataset_names: list[str],
    output_dir: str,
    **overrides,
) -> dict:
    """
    Generate a LLaMA Factory training YAML config.

    Phases:
      - phase1_sft: Supervised trajectory learning
      - phase2_tools: Tool-use learning
      - phase3_simulation: Scenario simulation
      - phase4_dpo: Direct preference optimization
      - phase5_ppo: Reinforcement learning
    """
    base_config = {
        # Model
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": True,

        # Method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",

        # LoRA config
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target": "all",

        # Dataset
        "dataset": ",".join(dataset_names),
        "template": "default",
        "cutoff_len": 8192,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,

        # Training
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,

        # Eval
        "val_size": 0.02,
        "per_device_eval_batch_size": 4,
        "eval_strategy": "steps",
        "eval_steps": 500,
    }

    # Phase-specific overrides
    phase_configs = {
        "phase1_sft": {
            "stage": "sft",
            "num_train_epochs": 3.0,
            "learning_rate": 5e-5,
            "cutoff_len": 8192,
        },
        "phase2_tools": {
            "stage": "sft",
            "num_train_epochs": 2.0,
            "learning_rate": 3e-5,
            "cutoff_len": 16384,
        },
        "phase3_simulation": {
            "stage": "sft",
            "num_train_epochs": 2.0,
            "learning_rate": 2e-5,
            "cutoff_len": 16384,
        },
        "phase4_dpo": {
            "stage": "dpo",
            "finetuning_type": "lora",
            "dpo_beta": 0.1,
            "num_train_epochs": 1.0,
            "learning_rate": 1e-5,
            "cutoff_len": 8192,
        },
        "phase5_ppo": {
            "stage": "ppo",
            "reward_model": None,  # path to reward model
            "num_train_epochs": 1.0,
            "learning_rate": 5e-6,
            "cutoff_len": 8192,
        },
    }

    if phase in phase_configs:
        base_config.update(phase_configs[phase])

    base_config.update(overrides)
    return base_config


def write_training_config(config: dict, output_path: Path) -> Path:
    """Write training config as YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return output_path


def register_custom_model() -> dict:
    """
    Registration info for SHUTEN-DŌJI in LLaMA Factory.

    To use with LLaMA Factory, add to model registry or use
    --model_name_or_path pointing to the custom model directory.
    """
    return {
        "model_type": "shuten_doji",
        "architecture": "ShutenDojiModel",
        "config_class": "ShutenDojiConfig",
        "module_path": "src.model.architecture",
        "tokenizer": "meta-llama/Llama-3.1-8B",  # use standard tokenizer, extend later
        "template": "llama3",
        "special_tokens": {
            "state_token": "<|state|>",
            "analysis_token": "<|analysis|>",
            "simulation_token": "<|simulation|>",
            "planning_token": "<|planning|>",
            "prediction_token": "<|prediction|>",
            "memory_token": "<|memory|>",
            "tool_token": "<|tool|>",
            "role_token": "<|role|>",
            "critique_token": "<|critique|>",
        },
    }


# --- Convenience functions ---


def setup_phase1(
    model_path: str,
    data_dir: Path,
    output_dir: str,
) -> tuple[Path, Path]:
    """Set up Phase 1 (SFT) training with LLaMA Factory."""
    # Dataset info
    datasets = [
        LlamaFactoryDatasetEntry(
            name="shuten_sft_business",
            file_name=str(data_dir / "business" / "sft_sharegpt.json"),
            formatting="sharegpt",
        ),
        LlamaFactoryDatasetEntry(
            name="shuten_sft_logistics",
            file_name=str(data_dir / "logistics" / "sft_sharegpt.json"),
            formatting="sharegpt",
        ),
        LlamaFactoryDatasetEntry(
            name="shuten_sft_markets",
            file_name=str(data_dir / "markets" / "sft_sharegpt.json"),
            formatting="sharegpt",
        ),
    ]

    dataset_info_path = generate_dataset_info(datasets, data_dir / "dataset_info.json")

    # Training config
    config = generate_training_config(
        phase="phase1_sft",
        model_name_or_path=model_path,
        dataset_names=[d.name for d in datasets],
        output_dir=output_dir,
    )
    config_path = write_training_config(config, Path(output_dir) / "train_config.yaml")

    return dataset_info_path, config_path
