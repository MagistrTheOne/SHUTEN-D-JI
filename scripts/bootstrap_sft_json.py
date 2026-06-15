#!/usr/bin/env python3
"""Build sft_train.json / sft_eval.json from template RL jsonl for bootstrap SFT."""

from __future__ import annotations

import json
import random
from pathlib import Path


def main() -> None:
    base = Path("/workspace/SHUTEN-D-JI/data/trajectories")
    rows: list[dict] = []

    for jsonl in sorted(base.glob("*/*/batch_*.jsonl")):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            traj = rec.get("trajectory", {})
            steps = traj.get("steps", [])
            if not steps:
                continue
            conv = [
                {
                    "from": "human",
                    "value": f"[SHUTEN {rec.get('domain', 'unknown')}] Analyze and recommend actions.",
                },
                {
                    "from": "gpt",
                    "value": "\n".join(
                        f"Step {s.get('step', i)}: {s.get('action', {}).get('type', 'action')}"
                        for i, s in enumerate(steps[:8])
                    ),
                },
            ]
            rows.append({"conversations": conv})

    random.seed(42)
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.95))
    train, eval_ = rows[:split], rows[split:] or rows[:1]

    (base / "sft_train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "sft_eval.json").write_text(json.dumps(eval_, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bootstrap: {len(train)} train, {len(eval_)} eval from {len(rows)} trajectories")


if __name__ == "__main__":
    main()
