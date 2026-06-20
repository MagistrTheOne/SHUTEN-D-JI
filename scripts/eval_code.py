"""
NULLXES-CODE-EVAL-100 runner.

Pod usage (compare models under identical sampling/sandbox):
    python -m scripts.eval_code --provider vllm \
        --base-url http://127.0.0.1:8000/v1 \
        --model MagistrTheOne/SHUTEN-DOJI --label shuten_code_lora \
        --k 3 --out data/code_forge/eval/shuten_code_lora.json

Offline self-check (no GPU; oracle reference solutions must score ~1.0 pass@1):
    python -m scripts.eval_code --selfcheck

The harness parses each model completion into (answer, code), runs the hidden
tests in the sandbox, applies the constitution/format check, and aggregates the
spec metrics. Execution is the judge.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.forge import sandbox
from src.forge.bench import (
    BenchTask,
    SampleResult,
    aggregate_metrics,
    bench_seed,
    regression_rate,
    required_blocks_present,
)
from src.forge.schema import SYSTEM_PROMPT

_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def parse_completion(text: str) -> tuple[str, str]:
    """Return (answer_text, exec_code). exec_code = labeled ```code``` block, else
    the first/only fenced block; answer_text is the full completion."""
    blocks = _FENCE.findall(text)
    code = ""
    for tag, body in blocks:
        if tag.lower() in ("code", "python"):
            code = body.rstrip()
            break
    if not code and blocks:
        code = blocks[0][1].rstrip()
    return text, code


def score_sample(task: BenchTask, completion: str, idx: int) -> SampleResult:
    answer, code = parse_completion(completion)
    v = sandbox.verify(task.language, code, task.tests, solution_format="answer")
    fmt_ok = required_blocks_present(task.task_type, answer)
    # Hallucinated API proxy: code fails to compile/import in the sandbox.
    hallucinated = not v.compile_ok and ("import" in code)
    changed = v.diff_stats.added + v.diff_stats.removed
    accepted = v.all_pass and fmt_ok
    return SampleResult(
        task_id=task.id, sample_idx=idx,
        compiled=v.compile_ok, tests_passed=v.tests_ok,
        lint_ok=v.lint_ok, typecheck_ok=v.typecheck_ok, security_ok=v.security_ok,
        format_ok=fmt_ok, hallucinated_api=hallucinated,
        changed_lines=changed, accepted=accepted,
    )


def run_eval(tasks: list[BenchTask], completions: dict[str, list[str]], k: int) -> dict:
    results: list[SampleResult] = []
    for task in tasks:
        for idx, comp in enumerate(completions.get(task.id, [])):
            results.append(score_sample(task, comp, idx))
    return aggregate_metrics(results, k)


def reference_completion(task: BenchTask) -> str:
    """Oracle completion built from the reference solution (self-check only)."""
    from src.forge.schema import CONSTITUTION_BLOCKS
    blocks = "\n\n".join(f"## {b}\nReference satisfies this section."
                         for b in CONSTITUTION_BLOCKS[task.task_type])
    return f"{blocks}\n\n```code\n{task.reference_exec_code}```\n"


def selfcheck() -> None:
    tasks = bench_seed()
    completions = {t.id: [reference_completion(t)] for t in tasks}
    metrics = run_eval(tasks, completions, k=1)
    print(json.dumps(metrics, indent=2))
    assert metrics["pass@1"] == 1.0, f"oracle must pass all, got {metrics['pass@1']}"
    assert metrics["security_violations"] == 0.0
    print("\nSELFCHECK OK — oracle reference solves the eval seed; harness metrics valid.")


def _vllm_completions(tasks: list[BenchTask], base_url: str, model: str, k: int) -> dict:
    import asyncio

    from src.factory.llm_client import LLMClient, LLMConfig

    client = LLMClient(LLMConfig(base_url=base_url, model=model, temperature=0.6, max_tokens=2048))

    def prompt_for(t: BenchTask) -> str:
        lines = [t.task]
        if t.repo_context:
            lines.append("\n## REPO CONTEXT\n" + t.repo_context)
        if t.constraints:
            lines.append("\n## CONSTRAINTS\n" + "\n".join(f"- {c}" for c in t.constraints))
        lines.append("\nReturn the sectioned answer and a ```code``` block.")
        return "\n".join(lines)

    async def _run() -> dict:
        out: dict[str, list[str]] = {}
        for t in tasks:
            samples = await client.generate_batch([prompt_for(t)] * k, system_prompt=SYSTEM_PROMPT)
            out[t.id] = samples
        return out

    return asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--provider", choices=["vllm"], default=None)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="MagistrTheOne/SHUTEN-DOJI")
    p.add_argument("--label", default="model")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--baseline", default=None,
                   help="baseline metrics json for regression_rate")
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    if args.provider != "vllm":
        raise SystemExit("Provide --selfcheck or --provider vllm")

    tasks = bench_seed()
    completions = _vllm_completions(tasks, args.base_url, args.model, args.k)
    metrics = run_eval(tasks, completions, args.k)
    metrics["label"] = args.label

    if args.baseline and Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        metrics["regression_rate_vs_baseline"] = regression_rate(base, metrics)

    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
