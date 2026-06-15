#!/usr/bin/env python3
"""Single-process multi-GPU DPO for PTQ GPTQ models (Qwen3-235B on 3xA100)."""

from __future__ import annotations

import os
import sys


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    sys.path.insert(0, os.path.join(repo_root, "scripts"))

    config = sys.argv[1] if len(sys.argv) > 1 else "configs/training/shuten_dpo.yaml"
    sys.argv = [sys.argv[0], config]

    from lf_gptq_mp import apply_gptq_model_parallel_patch
    from llamafactory.train.tuner import run_exp

    apply_gptq_model_parallel_patch()
    run_exp()


if __name__ == "__main__":
    main()
