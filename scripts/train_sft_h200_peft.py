#!/usr/bin/env python3
"""PEFT QLoRA SFT for Qwen3.5-122B on 1x H200 — explicit 4-bit load."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import yaml
from accelerate.utils import get_max_memory
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

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
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "configs/training/shuten_sft_h200.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_path = cfg["model_name_or_path"]
    data_path = Path(cfg["dataset_dir"]) / "sft_train.json"
    output_dir = cfg["output_dir"]

    print(f"[h200_peft] model={model_path} config={cfg_path} data={data_path}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Stage weights on CPU; quantize per-shard when dispatching to GPU (~61GB vs ~244GB bf16).
    max_memory = get_max_memory()
    if 0 in max_memory:
        max_memory[0] = "75GiB"
    max_memory["cpu"] = max_memory.get("cpu", "800GiB")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    offload_dir = Path("/tmp/shuten_offload")
    offload_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=cfg.get("lora_rank", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=MOE_LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    print(f"[h200_peft] LoRA targets={MOE_LORA_TARGETS} rank={lora.r}")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_sharegpt(data_path)
    max_len = cfg.get("cutoff_len", 2048)

    train_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 2e-4),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        bf16=True,
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        max_length=max_len,
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
    print(f"[h200_peft] saved to {output_dir}")


if __name__ == "__main__":
    main()
