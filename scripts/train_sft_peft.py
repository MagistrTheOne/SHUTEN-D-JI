#!/usr/bin/env python3
"""
Native PEFT SFT fallback — no LLaMA Factory.
Use if LF keeps failing: python scripts/train_sft_peft.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Attention-only LoRA: skips 128 MoE experts per layer (LF lora_target=all hangs ~1h+).
MOE_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_sharegpt(path: Path) -> Dataset:
    rows = json.loads(path.read_text(encoding="utf-8"))

    def fmt(example: dict) -> dict:
        text = ""
        for turn in example.get("conversations", []):
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                text += f"<|im_start|>user\n{content}\n"
            else:
                text += f"<|im_start|>assistant\n{content}\n"
        return {"text": text}

    return Dataset.from_list([fmt(r) for r in rows])


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs/training/shuten_sft.yaml").read_text(encoding="utf-8"))

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
    model_path = cfg["model_name_or_path"]
    data_path = Path(cfg["dataset_dir"]) / "sft_train.json"
    output_dir = cfg["output_dir"]

    print(f"[train_sft_peft] model={model_path} data={data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from accelerate.utils import get_max_memory

    max_memory = {k: "72GiB" for k in get_max_memory() if k != "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg.get("lora_rank", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=MOE_LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    print(f"[train_sft_peft] LoRA targets={MOE_LORA_TARGETS} rank={lora.r}")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_sharegpt(data_path)

    train_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 2e-4),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        fp16=True,
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        max_length=cfg.get("cutoff_len", 8192),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train_sft_peft] saved to {output_dir}")


if __name__ == "__main__":
    main()
