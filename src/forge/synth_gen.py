"""
Layer B — Executable Synthetic generator.

For each task slot from the task factory:
  1. ask a CandidateProvider for N drafts (model-backed on the pod),
  2. parse each draft into (answer, exec_code, tests),
  3. judge every candidate (real execution),
  4. keep the MINIMAL accepted candidate; otherwise archive the task as rejected.

The provider is abstracted so the pipeline is testable without a GPU:
- `VLLMCandidateProvider` talks to vLLM via the existing LLMClient (used on pod).
- `EchoCandidateProvider` returns pre-seeded drafts (used for local pipeline tests).

No fabricated verification: an episode is written only after it passes the judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.forge.judge import judge_episode, select_minimal
from src.forge.schema import (
    SYSTEM_PROMPT,
    Episode,
    Language,
    Layer,
    Solution,
    TaskPrompt,
    TaskType,
)
from src.forge.task_factory import TaskSpec

_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


@dataclass
class Draft:
    """One model draft, already separated into trainable answer + runnable parts."""
    answer: str       # full Constitution-formatted text (training target)
    exec_code: str    # runnable solution code
    tests: str        # hidden tests to run


class CandidateProvider(Protocol):
    def draft(self, spec: TaskSpec, n: int) -> list[Draft]: ...


def build_prompt(spec: TaskSpec) -> str:
    """Author the generation prompt for a task slot (Constitution-aware)."""
    from src.forge.schema import CONSTITUTION_BLOCKS

    blocks = " / ".join(CONSTITUTION_BLOCKS[spec.task_type])
    lines = [
        f"Produce a {spec.difficulty.value} {spec.task_type.value} task in {spec.language.value} "
        f"and solve it.",
        f"Answer using exactly these sections: {blocks}.",
        "Return three labeled fenced blocks in this order:",
        "1) ```answer``` — the full sectioned answer (no chain-of-thought, no Step labels).",
        "2) ```code``` — the runnable solution module only.",
        "3) ```tests``` — pytest tests importing from `solution`.",
    ]
    if spec.bug_class:
        lines.append(f"The task must center on a '{spec.bug_class.value}' bug.")
    if spec.constraints:
        lines.append("Constraints: " + "; ".join(spec.constraints))
    return "\n".join(lines)


def parse_draft(text: str) -> Draft | None:
    """Extract the answer/code/tests fenced blocks from a model completion."""
    blocks = {tag.lower(): body.rstrip() for tag, body in _FENCE.findall(text)}
    answer = blocks.get("answer")
    code = blocks.get("code") or blocks.get("python")
    tests = blocks.get("tests")
    if not (answer and code and tests):
        return None
    return Draft(answer=answer, exec_code=code, tests=tests)


def _episode_from_draft(spec: TaskSpec, draft: Draft, idx: int, canary: str) -> Episode:
    is_diff = "```diff" in draft.answer or draft.answer.lstrip().startswith("--- ")
    fmt = "diff" if is_diff else "answer"
    return Episode(
        id=f"{spec.slot_id}-c{idx}",
        layer=Layer.SYNTHETIC,
        task_type=spec.task_type,
        language=spec.language,
        bug_class=spec.bug_class,
        difficulty=spec.difficulty,
        edit_scope=spec.edit_scope,
        prompt=TaskPrompt(task=build_prompt(spec), constraints=spec.constraints),
        solution=Solution(format=fmt, content=draft.answer),
        exec_code=draft.exec_code,
        tests=draft.tests,
        metadata={"source": "synthetic", "canary": canary, "slot": spec.slot_id},
    )


def generate_for_spec(
    spec: TaskSpec,
    provider: CandidateProvider,
    *,
    n_candidates: int = 4,
    canary: str = "NULLXES-FORGE-SYNTH",
) -> Episode | None:
    """Draft N, judge all, return the minimal accepted episode (or None)."""
    drafts = provider.draft(spec, n_candidates)
    verdicts = []
    for i, d in enumerate(drafts):
        if d is None:
            continue
        ep = _episode_from_draft(spec, d, i, canary)
        verdicts.append(judge_episode(ep, canary=canary))
    best = select_minimal(verdicts)
    return best.episode if best else None


def generate_layer_b(
    specs: list[TaskSpec],
    provider: CandidateProvider,
    *,
    target: int,
    n_candidates: int = 4,
    out_dir: str | Path = "data/code_forge/synthetic",
) -> dict:
    """Drive synthetic generation until `target` accepted episodes or specs run out."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted: list[dict] = []
    rejected = 0

    for spec in specs:
        if len(accepted) >= target:
            break
        ep = generate_for_spec(spec, provider, n_candidates=n_candidates)
        if ep is not None:
            accepted.append(ep.model_dump(mode="json"))
        else:
            rejected += 1

    (out / "synthetic.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = {
        "accepted": len(accepted),
        "rejected_tasks": rejected,
        "target": target,
        "acceptance_rate": round(len(accepted) / max(1, len(accepted) + rejected), 3),
    }
    (out / "synthetic_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


# --- providers ---------------------------------------------------------------


class EchoCandidateProvider:
    """Local provider: returns pre-seeded drafts keyed by slot_id. For pipeline tests."""

    def __init__(self, seeded: dict[str, list[Draft]]):
        self.seeded = seeded

    def draft(self, spec: TaskSpec, n: int) -> list[Draft]:
        return self.seeded.get(spec.slot_id, [])[:n]


class VLLMCandidateProvider:
    """Pod provider: draft N candidates per task via vLLM (OpenAI-compatible).

    Used by scripts/forge_gen.py on the pod where the model is served. Import of
    the async LLM client is deferred so this module loads without a server.
    """

    def __init__(self, base_url: str, model: str, temperature: float = 0.8):
        from src.factory.llm_client import LLMClient, LLMConfig

        self._client = LLMClient(LLMConfig(
            base_url=base_url, model=model, temperature=temperature, max_tokens=2048,
        ))

    def draft(self, spec: TaskSpec, n: int) -> list[Draft]:
        import asyncio

        prompt = build_prompt(spec)

        async def _run() -> list[str]:
            return await self._client.generate_batch([prompt] * n, system_prompt=SYSTEM_PROMPT)

        completions = asyncio.run(_run())
        return [d for d in (parse_draft(c) for c in completions) if d is not None]


# Languages the local sandbox can execute without Docker (for dry runs).
LOCAL_EXECUTABLE = {Language.PYTHON}
TASK_TYPES_WITH_DIFF = {TaskType.REPO_BUGFIX, TaskType.MULTI_FILE, TaskType.REFACTOR}
