"""
Persist the NULLXES-CODE-EVAL split as artifacts, keeping hidden tests and
reference solutions SEPARATE from the prompt-facing file (spec: hidden tests
stored separately; never published with prompts).

Outputs:
- data/code_forge/eval/eval_prompts.json   (prompt-facing: task + constraints + canary)
- data/code_forge/eval/eval_hidden.json    (hidden: tests + reference solution)
- data/code_forge/eval/eval_plan.json       (the full 100-task category plan)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.forge.bench import CATEGORY_PLAN, bench_seed

OUT = Path("data/code_forge/eval")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = bench_seed()

    prompts = [{
        "id": t.id, "category": t.category, "language": t.language.value,
        "task_type": t.task_type.value, "difficulty": t.difficulty.value,
        "edit_scope": t.edit_scope.value, "task": t.task,
        "repo_context": t.repo_context, "constraints": t.constraints,
        "canary": t.canary,
    } for t in tasks]

    hidden = [{
        "id": t.id, "tests": t.tests, "reference_exec_code": t.reference_exec_code,
        "canary": t.canary,
    } for t in tasks]

    (OUT / "eval_prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "eval_hidden.json").write_text(
        json.dumps(hidden, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "eval_plan.json").write_text(
        json.dumps({"category_plan": CATEGORY_PLAN,
                    "total_target": sum(CATEGORY_PLAN.values()),
                    "seed_authored": len(tasks)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"eval seed: {len(tasks)} tasks | full plan target: {sum(CATEGORY_PLAN.values())}")
    print(f"Wrote prompt-facing + hidden (separate) artifacts to {OUT}")


if __name__ == "__main__":
    main()
