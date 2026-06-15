"""
LLaMA Factory helpers for PTQ GPTQ models too large for one GPU.

235B GPTQ (~117GB) must shard via device_map=\"auto\".
LF 0.9.5 requires torchrun (parallel_mode=DISTRIBUTED) but assigns one GPU
per rank by default — we use NPROC_PER_NODE=1 and override device_map.
"""

from __future__ import annotations

from typing import Any


def _gpu_only_max_memory() -> dict[int | str, str]:
    """Reserve GPU-only memory map so accelerate never offloads to CPU/meta."""
    from accelerate.utils import get_max_memory

    max_memory = get_max_memory()
    max_memory = {k: v for k, v in max_memory.items() if k != "cpu"}
    # Headroom for LoRA, activations, and Marlin workspace during GPTQ post_init.
    capped: dict[int | str, str] = {}
    for device, limit in max_memory.items():
        if isinstance(device, int):
            capped[device] = "72GiB"
        else:
            capped[device] = limit
    return capped


def apply_gptq_model_parallel_patch() -> None:
    """Patch get_train_args before llamafactory.train.tuner is imported."""
    if getattr(apply_gptq_model_parallel_patch, "_applied", False):
        return

    import llamafactory.hparams as hparams
    import llamafactory.hparams.parser as parser
    import llamafactory.model.patcher as patcher

    orig_get_train_args = parser.get_train_args
    orig_patch_config = patcher.patch_config

    def patched_get_train_args(args: Any):
        model_args, data_args, training_args, finetuning_args, generating_args = orig_get_train_args(args)
        model_args.device_map = "auto"
        model_args.offload_folder = None
        return model_args, data_args, training_args, finetuning_args, generating_args

    def patched_patch_config(config, tokenizer, model_args, init_kwargs, is_trainable):
        orig_patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
        if init_kwargs.get("device_map") == "auto":
            init_kwargs["max_memory"] = _gpu_only_max_memory()
            init_kwargs.pop("offload_folder", None)

    parser.get_train_args = patched_get_train_args
    hparams.get_train_args = patched_get_train_args
    patcher.patch_config = patched_patch_config

    apply_gptq_model_parallel_patch._applied = True
