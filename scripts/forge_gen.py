"""
Layer B driver.

Pod usage (model served by vLLM):
    python -m scripts.forge_gen --provider vllm \
        --base-url http://127.0.0.1:8000/v1 \
        --model MagistrTheOne/SHUTEN-DOJI \
        --target 220

Local self-check (no GPU, validates multi-candidate selection end-to-end):
    python -m scripts.forge_gen --selfcheck
"""

from __future__ import annotations

import argparse
import shutil

from src.forge.schema import (
    Difficulty,
    EditScope,
    Language,
    TaskType,
)
from src.forge.synth_gen import (
    Draft,
    EchoCandidateProvider,
    VLLMCandidateProvider,
    generate_for_spec,
    generate_layer_b,
)
from src.forge.task_factory import TaskSpec, plan_tasks


def selfcheck() -> None:
    """Three candidates for one task: a failing one, a bloated pass, a minimal pass.
    The judge must reject the failing one and pick the minimal accepted candidate."""
    spec = TaskSpec(
        slot_id="cf-selfcheck-0001",
        task_type=TaskType.FUNCTION_IMPL,
        language=Language.PYTHON,
        bug_class=None,
        difficulty=Difficulty.EASY,
        edit_scope=EditScope.SINGLE_FUNC,
        layer_hint="synthetic",
        constraints=["no new third-party dependency"],
    )
    tests = (
        "from solution import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_neg():\n    assert add(-1, 1) == 0\n"
    )
    answer_min = (
        "## ASSUMPTIONS\n- Numeric inputs.\n\n"
        "## IMPLEMENTATION\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```\n\n"
        "## TESTS\nCovers a positive and a sign-cancelling case.\n\n"
        "## COMPLEXITY\nO(1).\n"
    )
    answer_bloat = (
        "## ASSUMPTIONS\n- Numeric inputs.\n\n"
        "## IMPLEMENTATION\n```python\n"
        "def add(a: int, b: int) -> int:\n"
        "    total = 0\n    total += a\n    total += b\n    return total\n```\n\n"
        "## TESTS\nCovers a positive and a sign-cancelling case.\n\n"
        "## COMPLEXITY\nO(1).\n"
    )
    answer_fail = (
        "## ASSUMPTIONS\n- Numeric inputs.\n\n"
        "## IMPLEMENTATION\n```python\ndef add(a: int, b: int) -> int:\n    return a - b\n```\n\n"
        "## TESTS\nWrong operator on purpose.\n\n"
        "## COMPLEXITY\nO(1).\n"
    )
    code_bloat = (
        "def add(a: int, b: int) -> int:\n"
        "    total = 0\n    total += a\n    total += b\n    return total\n"
    )
    seeded = {spec.slot_id: [
        Draft(answer_fail, "def add(a: int, b: int) -> int:\n    return a - b\n", tests),
        Draft(answer_bloat, code_bloat, tests),
        Draft(answer_min, "def add(a: int, b: int) -> int:\n    return a + b\n", tests),
    ]}
    provider = EchoCandidateProvider(seeded)
    ep = generate_for_spec(spec, provider, n_candidates=3, canary="NULLXES-SELFCHECK")
    assert ep is not None, "expected a minimal accepted candidate"
    added = ep.verification.diff_stats.added
    print(f"selected candidate {ep.id} | code_lines={added} | verified={ep.verification.all_pass}")
    # The minimal correct candidate is c2 (single-expression body); the failing
    # c0 (wrong operator) and the bloated c1 must not be selected.
    assert ep.id.endswith("-c2"), f"expected minimal candidate c2, got {ep.id}"
    assert ep.verification.all_pass, "selected candidate must be fully verified"
    print("SELFCHECK OK — failing candidate rejected, minimal accepted candidate chosen.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--provider", choices=["vllm"], default=None)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="MagistrTheOne/SHUTEN-DOJI")
    p.add_argument("--target", type=int, default=220)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--languages",
        default=None,
        help="Comma-separated language filter, e.g. python or python,typescript.",
    )
    p.add_argument(
        "--executable-only",
        action="store_true",
        help="Skip languages that cannot be executed in the current environment.",
    )
    p.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Stop after considering this many synthetic task slots.",
    )
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write synthetic.partial.json after every N accepted episodes; 0 disables.",
    )
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    if args.provider == "vllm":
        provider = VLLMCandidateProvider(
            args.base_url,
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        # Synthetic layer skips the failure cluster (that is Layer C).
        specs = [s for s in plan_tasks(seed=args.seed)
                 if s.layer_hint == "synthetic"]
        if args.languages:
            allowed = {Language(v.strip()) for v in args.languages.split(",") if v.strip()}
            specs = [s for s in specs if s.language in allowed]
        if args.executable_only and shutil.which("docker") is None:
            specs = [s for s in specs if s.language == Language.PYTHON]
        if args.max_slots is not None:
            specs = specs[:args.max_slots]
        print(f"forge_gen plan: slots={len(specs)} target={args.target} candidates={args.candidates}")
        stats = generate_layer_b(
            specs,
            provider,
            target=args.target,
            n_candidates=args.candidates,
            checkpoint_every=args.checkpoint_every,
        )
        print(stats)
    else:
        raise SystemExit("Provide --selfcheck or --provider vllm")


if __name__ == "__main__":
    main()
