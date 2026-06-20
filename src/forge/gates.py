"""
CODE FORGE Gates — accept/reject rules over a candidate + its sandbox result.

The sandbox supplies the *execution* facts (compile/tests/lint/typecheck). These
gates combine those facts with *static integrity* and *contamination* checks to
produce a single verdict. The LLM never overrides a gate.

See docs/CODE_FORGE_V1_SPEC.md section 3.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from src.forge.schema import (
    FORBIDDEN_PATTERNS,
    CodeConstitution,
    Episode,
    TaskType,
    Verification,
)

# Edit-scope budgets: max changed lines (added+removed) before "patch too broad".
SCOPE_BUDGET = {
    "single_line": 6,
    "single_func": 60,
    "single_file": 200,
    "multi_file": 600,
}

# Secret detectors (coarse; the harness archives any hit for manual review).
SECRET_PATTERNS = (
    r"AKIA[0-9A-Z]{16}",                       # AWS access key id
    r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
    r"sk-[A-Za-z0-9]{20,}",                    # OpenAI-style key
    r"ghp_[A-Za-z0-9]{36}",                    # GitHub PAT
    r"(?i)(password|passwd|secret|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
)

# Tokens that mark output as a fabricated run rather than a real verification.
FABRICATED_RUN_HINTS = (
    r"\$ pytest.*\n.*passed",   # pretends a shell session happened inline
    r"Ran \d+ tests in .*OK",
)


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GateReport:
    candidate_id: str
    results: list[GateResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(GateResult(name, passed, detail))

    @property
    def accepted(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_gates(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "accepted": self.accepted,
            "failed_gates": self.failed_gates,
            "results": [
                {"name": r.name, "passed": r.passed, "detail": r.detail}
                for r in self.results
            ],
        }


def _changed_lines(diff: str) -> tuple[int, int, int]:
    added = removed = 0
    files: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].strip()
            if path and path not in ("/dev/null",):
                files.add(path.lstrip("ab/"))
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed, len(files)


def _touches_tests(diff: str) -> bool:
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "diff ")):
            low = line.lower()
            if "test" in low or "spec." in low or "__tests__" in low:
                return True
    return False


def run_gates(
    episode: Episode,
    verification: Verification,
    *,
    canary: str | None = None,
    known_solution_ngrams: set[str] | None = None,
    declared_files: list[str] | None = None,
) -> GateReport:
    """Apply all gates. `verification` comes from the sandbox; the rest are static."""
    rep = GateReport(candidate_id=episode.id)
    answer = episode.solution.content
    is_diff = episode.solution.format == "diff"

    # --- 3.1 execution gates (facts from sandbox) ---
    rep.add("compile_ok", verification.compile_ok)
    rep.add("tests_ok", verification.tests_ok)
    rep.add("lint_ok", verification.lint_ok)
    rep.add("typecheck_ok", verification.typecheck_ok)

    # --- 3.2 static / integrity gates ---
    # constitution block compliance
    missing = CodeConstitution.missing_blocks(episode.task_type, answer)
    rep.add("constitution_blocks", not missing,
            f"missing: {missing}" if missing else "")

    # forbidden patterns (CoT, Step N:, action-trace tokens, TODO/FIXME)
    forb = [p for p in FORBIDDEN_PATTERNS if re.search(p, answer)]
    rep.add("no_forbidden_patterns", not forb, f"hit: {forb}" if forb else "")

    # no fabricated run output
    fab = [p for p in FABRICATED_RUN_HINTS if re.search(p, answer)]
    rep.add("no_fabricated_run", not fab, f"hit: {fab}" if fab else "")

    # test-edit gate
    allows_test_edit = episode.task_type in (TaskType.TEST_GEN,)
    if is_diff and not allows_test_edit:
        rep.add("no_test_edit", not _touches_tests(answer))
    else:
        rep.add("no_test_edit", True)

    # patch boundedness
    if is_diff:
        added, removed, nfiles = _changed_lines(answer)
        budget = SCOPE_BUDGET[episode.edit_scope.value]
        rep.add("patch_bounded", (added + removed) <= budget,
                f"changed={added + removed} budget={budget}")
    else:
        rep.add("patch_bounded", True)

    # explanation matches diff: declared files must appear in the diff
    if is_diff and declared_files:
        diff_low = answer.lower()
        missing_files = [f for f in declared_files if f.lower() not in diff_low]
        rep.add("explanation_matches_diff", not missing_files,
                f"missing: {missing_files}" if missing_files else "")
    else:
        rep.add("explanation_matches_diff", True)

    # secrets
    sec_hits = [p for p in SECRET_PATTERNS if re.search(p, answer)]
    rep.add("no_secrets", not sec_hits, "secret-like string present" if sec_hits else "")

    # security gate (from sandbox: e.g. bandit/semgrep clean, parametrized SQL)
    rep.add("security_ok", verification.security_ok)

    # --- 3.3 contamination ---
    if canary:
        rep.add("canary_clean", canary not in answer)
    else:
        rep.add("canary_clean", True)

    if known_solution_ngrams:
        overlap = _ngram_overlap(answer, known_solution_ngrams)
        rep.add("ngram_clean", overlap < 0.5, f"overlap={overlap:.2f}")
    else:
        rep.add("ngram_clean", True)

    return rep


def _ngrams(text: str, n: int = 8) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def _ngram_overlap(text: str, known: set[str], n: int = 8) -> float:
    grams = _ngrams(text, n)
    if not grams:
        return 0.0
    return len(grams & known) / len(grams)


def content_hash(episode: Episode) -> str:
    """Stable hash of prompt+solution for dedup (spec packer)."""
    h = hashlib.sha256()
    h.update(episode.prompt.task.strip().lower().encode())
    h.update(b"\x00")
    h.update(episode.solution.content.strip().encode())
    return h.hexdigest()[:16]
