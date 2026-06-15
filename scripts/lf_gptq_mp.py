"""
LLaMA Factory helpers for PTQ GPTQ models too large for one GPU.

235B GPTQ (~117GB) must shard via device_map=\"auto\".
LF 0.9.5 requires torchrun (parallel_mode=DISTRIBUTED) but assigns one GPU
per rank by default — we use NPROC_PER_NODE=1 and override device_map.
"""

from __future__ import annotations

from typing import Any


def apply_gptq_model_parallel_patch() -> None:
    """Patch get_train_args before llamafactory.train.tuner is imported."""
    if getattr(apply_gptq_model_parallel_patch, "_applied", False):
        return

    import llamafactory.hparams as hparams
    import llamafactory.hparams.parser as parser

    orig_get_train_args = parser.get_train_args

    def patched_get_train_args(args: Any):
        model_args, data_args, training_args, finetuning_args, generating_args = orig_get_train_args(args)
        model_args.device_map = "auto"
        return model_args, data_args, training_args, finetuning_args, generating_args

    parser.get_train_args = patched_get_train_args
    hparams.get_train_args = patched_get_train_args

    apply_gptq_model_parallel_patch._applied = True
