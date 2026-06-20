"""
CODE FORGE Judge — execution-verified acceptance of candidates.

Flow per candidate:
    episode (with solution + hidden tests)
      -> sandbox.verify (real compile/tests/lint/typecheck/security facts)
      -> gates.run_gates (combine facts with static + contamination checks)
      -> verdict (accepted / rejected with failed gates)

Multi-candidate selection (Layer B): given N drafts for one task, pick the
minimal accepted candidate (fewest changed lines), else reject the task.

The LLM is never the judge. Execution is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.forge import gates, sandbox
from src.forge.schema import Episode

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def extract_code(episode: Episode) -> str:
    """Return runnable code: explicit `exec_code`, else concatenated fenced blocks."""
    if episode.exec_code.strip():
        return episode.exec_code
    blocks = _FENCE.findall(episode.solution.content)
    return "\n\n".join(b.rstrip() for b in blocks)


@dataclass
class Verdict:
    episode: Episode
    accepted: bool
    failed_gates: list[str]
    report: dict


def judge_episode(
    episode: Episode,
    *,
    canary: str | None = None,
    known_solution_ngrams: set[str] | None = None,
    declared_files: list[str] | None = None,
    trust_recorded_verification: bool = False,
) -> Verdict:
    """Verify one episode and return a verdict.

    `trust_recorded_verification=True` skips re-execution and uses the
    verification already stored on the episode (used when re-gating an
    already-verified gold corpus without a sandbox handy).
    """
    if trust_recorded_verification:
        verification = episode.verification
    else:
        verification = sandbox.verify(
            episode.language,
            extract_code(episode),
            episode.tests,
            solution_format=episode.solution.format,
        )
        episode.verification = verification

    report = gates.run_gates(
        episode,
        verification,
        canary=canary,
        known_solution_ngrams=known_solution_ngrams,
        declared_files=declared_files,
    )
    return Verdict(
        episode=episode,
        accepted=report.accepted,
        failed_gates=report.failed_gates,
        report=report.to_dict(),
    )


def select_minimal(verdicts: list[Verdict]) -> Verdict | None:
    """From multiple candidate verdicts for one task, keep the minimal accepted."""
    accepted = [v for v in verdicts if v.accepted]
    if not accepted:
        return None
    return min(
        accepted,
        key=lambda v: (
            v.episode.verification.diff_stats.added
            + v.episode.verification.diff_stats.removed
        ),
    )
