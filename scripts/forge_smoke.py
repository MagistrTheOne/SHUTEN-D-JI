"""Smoke test: drive a real Python episode through sandbox -> gates -> judge."""

from __future__ import annotations

import json

from src.forge.judge import judge_episode
from src.forge.schema import (
    Difficulty,
    EditScope,
    Episode,
    Language,
    Layer,
    Solution,
    TaskPrompt,
    TaskType,
)

GOOD = Episode(
    id="cf-smoke-0001",
    layer=Layer.GOLD,
    task_type=TaskType.FUNCTION_IMPL,
    language=Language.PYTHON,
    difficulty=Difficulty.EASY,
    edit_scope=EditScope.SINGLE_FUNC,
    prompt=TaskPrompt(
        task="Implement `clamp(x, lo, hi)` returning x bounded to [lo, hi].",
        constraints=["no new third-party dependency"],
    ),
    solution=Solution(
        format="answer",
        content=(
            "## ASSUMPTIONS\n"
            "- lo <= hi; numeric inputs.\n\n"
            "## IMPLEMENTATION\n"
            "```python\n"
            "def clamp(x: float, lo: float, hi: float) -> float:\n"
            "    return max(lo, min(x, hi))\n"
            "```\n\n"
            "## TESTS\n"
            "Covered: below, inside, above range.\n\n"
            "## COMPLEXITY\n"
            "O(1) time and space.\n"
        ),
    ),
    exec_code=(
        "def clamp(x: float, lo: float, hi: float) -> float:\n"
        "    return max(lo, min(x, hi))\n"
    ),
    tests=(
        "from solution import clamp\n\n"
        "def test_inside():\n    assert clamp(5, 0, 10) == 5\n\n"
        "def test_below():\n    assert clamp(-1, 0, 10) == 0\n\n"
        "def test_above():\n    assert clamp(11, 0, 10) == 10\n"
    ),
)

def main() -> None:
    # Sandbox runs the extracted exec_code; gates check the full answer text.
    v = judge_episode(GOOD.model_copy(deep=True), canary="NULLXES-CANARY-XYZ")
    print("== execution-verified judgment ==")
    print(json.dumps(v.report, indent=2))
    print("accepted:", v.accepted, "failed:", v.failed_gates)
    assert v.accepted, f"expected acceptance, failed: {v.failed_gates}"
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
