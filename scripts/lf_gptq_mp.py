"""
LLaMA Factory helpers for PTQ GPTQ models too large for one GPU.

235B GPTQ (~117GB) must shard via device_map="auto" in a single process.
LF 0.9.5 otherwise forces torchrun (DDP) and assigns one GPU per rank.
"""

from __future__ import annotations

from typing import Any


def apply_gptq_model_parallel_patch() -> None:
    """Patch get_train_args so single-process multi-GPU device_map works for PTQ GPTQ."""
    import llamafactory.extras.misc as misc
    import llamafactory.hparams.parser as parser

    real_get_device_count = misc.get_device_count
    orig_get_train_args = parser.get_train_args

    def fake_get_device_count() -> int:
        return 1

    def patched_get_train_args(args: Any):
        misc.get_device_count = fake_get_device_count
        try:
            out = orig_get_train_args(args)
        finally:
            misc.get_device_count = real_get_device_count

        model_args, data_args, training_args, finetuning_args, generating_args = out
        model_args.device_map = "auto"
        return model_args, data_args, training_args, finetuning_args, generating_args

    parser.get_train_args = patched_get_train_args
