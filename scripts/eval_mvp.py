#!/usr/bin/env python3
"""MVP eval: base Qwen vs SHUTEN LoRA on fixed strategic prompts."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000/v1/chat/completions"
BASE_MODEL = "/workspace/models/qwen3.6-27b"
LORA_MODEL = "shuten"

PROMPTS = [
    "[SHUTEN business] Analyze and recommend actions.\n"
    "Objective: restore margin above 12% within 90 days under supplier delay risk.",
    "[SHUTEN logistics] Analyze and recommend actions.\n"
    "Objective: reroute 40% of volume from port A to port B with minimal SLA breach.",
    "[SHUTEN markets] Analyze and recommend actions.\n"
    "Objective: hedge FX exposure before rate decision while keeping liquidity buffer.",
]


def chat(model: str, user: str, max_tokens: int = 256) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        BASE,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def main() -> None:
    out = Path("/workspace/outputs/mvp_eval.json")
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n=== Prompt {i} ===")
        row = {"prompt": prompt, "base": "", "shuten": ""}
        for label, model in [("base", BASE_MODEL), ("shuten", LORA_MODEL)]:
            print(f"--- {label} ---")
            try:
                text = chat(model, prompt)
                row[label] = text
                print(text[:500])
            except Exception as e:
                row[label] = f"ERROR: {e}"
                print(row[label])
        results.append(row)

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
