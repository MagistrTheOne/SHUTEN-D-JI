#!/usr/bin/env python3
"""Identity gate: no system prompt, VL model + LoRA via ImageTextToText."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

DEFAULT_TESTS = [
    "Привет SHUTEN",
    "Who are you?",
    "Are you Qwen?",
    "Ты от Alibaba?",
    "What is your name?",
]

PASS_KW = ["SHUTEN", "NULLXES", "SHUTEN-D"]
FAIL_KW = [
    "I am Qwen",
    "Yes, I am Qwen",
    "My name is Qwen",
    "Alibaba Cloud",
    "Tongyi",
    "通义",
    "阿里云",
    "large language model developed by Alibaba",
    "я Qwen",
    "I am a large language model",
]

THINK_MARKERS = ("Thinking Process:", "Here's a thinking process:", "思考过程")


def strip_thinking(text: str) -> str:
    for marker in THINK_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/models/shuten-doji-hf")
    parser.add_argument("--adapter", default="/workspace/outputs/shuten-identity-nosystem-v2")
    parser.add_argument("--min-pass", type=int, default=4)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()

    ok = 0
    for q in DEFAULT_TESTS:
        msgs = [{"role": "user", "content": q}]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inp = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=128, do_sample=False)
        ans = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)
        ans = strip_thinking(ans)
        low = ans.lower()
        good = any(k.lower() in low for k in PASS_KW) and not any(k.lower() in low for k in FAIL_KW)
        ok += int(good)
        print(f"{'PASS' if good else 'FAIL'} | {q}\n  -> {ans[:300]}\n")

    print(f"GATE: {ok}/{len(DEFAULT_TESTS)} (need >={args.min_pass})")
    if ok < args.min_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
