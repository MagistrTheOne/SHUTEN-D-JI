#!/usr/bin/env python3
"""Torchrun entrypoint: apply GPTQ MP patch, then run LLaMA Factory training."""

from __future__ import annotations

import os
import sys


def _setup() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    sys.path.insert(0, os.path.join(repo_root, "scripts"))
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/training/shuten_sft.yaml"
    sys.argv = [sys.argv[0], config]
    return config


def main() -> None:
    config = _setup()

    from lf_gptq_mp import apply_gptq_model_parallel_patch

    apply_gptq_model_parallel_patch()

    from llamafactory.train.tuner import run_exp

    print(f"[lf_train_entry] config={config} device_map=auto (GPTQ MP)")
    run_exp()


if __name__ == "__main__":
    main()
