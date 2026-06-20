"""
NULLXES-CODE-EVAL-100 — private benchmark + metric harness.

Principles (spec section 6):
- Authored AFTER dataset freeze, never published, each task carries a unique
  canary string and hidden tests kept separate from the prompt.
- The judge is execution. Metrics are computed from real sandbox runs.

This module ships:
- The 100-task category plan (`CATEGORY_PLAN`) for the full benchmark.
- An executable Python seed (`bench_seed`) of post-freeze tasks DISTINCT from the
  training corpus, each with hidden tests + canary + a reference solution used
  only by the offline harness self-check (never trained on).
- Metric functions: pass@1, pass@3, compile_rate, test_pass_rate,
  patch_efficiency, regression_rate, hallucinated_api_rate, security_violations,
  format_compliance.

The pod runbook authors the remaining non-Python categories as Docker tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.forge.schema import (
    CodeConstitution,
    Difficulty,
    EditScope,
    Language,
    TaskType,
)

# Full 100-task split (spec 6). Counts only; content authored on the pod.
CATEGORY_PLAN: dict[str, int] = {
    "ts_next": 20,
    "py_fastapi": 20,
    "postgres": 10,
    "cpp_ue": 15,
    "docker_ci": 10,
    "multifile_bugfix": 15,
    "security_concurrency": 10,
}

CANARY_PREFIX = "NULLXES-EVAL-100"


@dataclass
class BenchTask:
    id: str
    category: str
    language: Language
    task_type: TaskType
    difficulty: Difficulty
    edit_scope: EditScope
    task: str
    tests: str                       # hidden, stored separately from the prompt
    canary: str
    constraints: list[str] = field(default_factory=list)
    repo_context: str = ""
    reference_exec_code: str = ""     # oracle for harness self-check only


def _bt(idx: int, **kw) -> BenchTask:
    kw.setdefault("canary", f"{CANARY_PREFIX}-{idx:03d}")
    return BenchTask(**kw)


def bench_seed() -> list[BenchTask]:
    """Executable Python seed, distinct from the training corpus."""
    tasks: list[BenchTask] = []

    tasks.append(_bt(1,
        id="eval-py-001", category="py_fastapi", language=Language.PYTHON,
        task_type=TaskType.FUNCTION_IMPL, difficulty=Difficulty.EASY,
        edit_scope=EditScope.SINGLE_FUNC,
        task="Implement `slugify(title)` -> lowercase, spaces to hyphens, drop non-alphanumeric (except hyphen), collapse repeats, strip edge hyphens.",
        constraints=["no new third-party dependency"],
        reference_exec_code=(
            "import re\n\n"
            "def slugify(title: str) -> str:\n"
            "    s = re.sub(r'[^a-z0-9]+', '-', title.lower())\n"
            "    return s.strip('-')\n"
        ),
        tests=(
            "from solution import slugify\n\n"
            "def test_basic():\n    assert slugify('Hello World') == 'hello-world'\n\n"
            "def test_punct():\n    assert slugify('  A, B & C!! ') == 'a-b-c'\n\n"
            "def test_collapse():\n    assert slugify('x---y') == 'x-y'\n"
        ),
    ))

    tasks.append(_bt(2,
        id="eval-py-002", category="multifile_bugfix", language=Language.PYTHON,
        task_type=TaskType.REPO_BUGFIX, difficulty=Difficulty.MEDIUM,
        edit_scope=EditScope.SINGLE_FUNC,
        task="`paginate(items, page, size)` returns the wrong slice (1-indexed pages). Fix so page 1 is the first `size` items; out-of-range pages return [].",
        repo_context=(
            "```python\ndef paginate(items, page, size):\n"
            "    start = page * size\n    return items[start:start + size]\n```"
        ),
        constraints=["minimal patch", "pages are 1-indexed"],
        reference_exec_code=(
            "def paginate(items, page, size):\n"
            "    start = (page - 1) * size\n"
            "    return items[start:start + size]\n"
        ),
        tests=(
            "from solution import paginate\n\n"
            "def test_first():\n    assert paginate([1, 2, 3, 4], 1, 2) == [1, 2]\n\n"
            "def test_second():\n    assert paginate([1, 2, 3, 4], 2, 2) == [3, 4]\n\n"
            "def test_oob():\n    assert paginate([1, 2], 5, 2) == []\n"
        ),
    ))

    tasks.append(_bt(3,
        id="eval-py-003", category="security_concurrency", language=Language.PYTHON,
        task_type=TaskType.SECURITY, difficulty=Difficulty.HARD,
        edit_scope=EditScope.SINGLE_FILE,
        task="Implement `verify_token(stored_hash, token)` using a constant-time comparison to avoid timing attacks. Use only the standard library.",
        constraints=["no new third-party dependency", "constant-time comparison"],
        reference_exec_code=(
            "import hmac\n\n"
            "def verify_token(stored_hash: str, token: str) -> bool:\n"
            "    return hmac.compare_digest(stored_hash, token)\n"
        ),
        tests=(
            "from solution import verify_token\n\n"
            "def test_match():\n    assert verify_token('abc123', 'abc123')\n\n"
            "def test_mismatch():\n    assert not verify_token('abc123', 'abc124')\n\n"
            "def test_len_mismatch():\n    assert not verify_token('abc', 'abcd')\n"
        ),
    ))

    tasks.append(_bt(4,
        id="eval-py-004", category="py_fastapi", language=Language.PYTHON,
        task_type=TaskType.FUNCTION_IMPL, difficulty=Difficulty.MEDIUM,
        edit_scope=EditScope.SINGLE_FUNC,
        task="Implement `parse_range(header)` for an HTTP 'bytes=start-end' range header, returning (start, end) ints; return None for malformed input.",
        constraints=["no new third-party dependency"],
        reference_exec_code=(
            "def parse_range(header: str):\n"
            "    if not header.startswith('bytes='):\n"
            "        return None\n"
            "    spec = header[len('bytes='):]\n"
            "    if '-' not in spec:\n"
            "        return None\n"
            "    a, _, b = spec.partition('-')\n"
            "    if not a.isdigit() or not b.isdigit():\n"
            "        return None\n"
            "    return int(a), int(b)\n"
        ),
        tests=(
            "from solution import parse_range\n\n"
            "def test_ok():\n    assert parse_range('bytes=0-499') == (0, 499)\n\n"
            "def test_bad_prefix():\n    assert parse_range('items=0-5') is None\n\n"
            "def test_bad_spec():\n    assert parse_range('bytes=abc') is None\n"
        ),
    ))

    tasks.append(_bt(5,
        id="eval-py-005", category="multifile_bugfix", language=Language.PYTHON,
        task_type=TaskType.REPAIR_TRAJECTORY, difficulty=Difficulty.MEDIUM,
        edit_scope=EditScope.SINGLE_FUNC,
        task="A first attempt at `dedupe_keep_order(xs)` loses order. Diagnose and provide the corrected version preserving first-seen order.",
        repo_context=(
            "Attempt:\n```python\ndef dedupe_keep_order(xs):\n    return list(set(xs))\n```\n"
            "Failing: dedupe_keep_order([3, 1, 3, 2]) order is not [3, 1, 2]."
        ),
        constraints=["preserve first-seen order"],
        reference_exec_code=(
            "def dedupe_keep_order(xs):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for x in xs:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        tests=(
            "from solution import dedupe_keep_order\n\n"
            "def test_order():\n    assert dedupe_keep_order([3, 1, 3, 2]) == [3, 1, 2]\n\n"
            "def test_empty():\n    assert dedupe_keep_order([]) == []\n"
        ),
    ))

    return tasks


# --- metric harness ----------------------------------------------------------


@dataclass
class SampleResult:
    task_id: str
    sample_idx: int
    compiled: bool
    tests_passed: bool
    lint_ok: bool
    typecheck_ok: bool
    security_ok: bool
    format_ok: bool
    hallucinated_api: bool
    changed_lines: int
    accepted: bool


def required_blocks_present(task_type: TaskType, answer: str) -> bool:
    return not CodeConstitution.missing_blocks(task_type, answer)


def aggregate_metrics(results: list[SampleResult], k: int) -> dict:
    """Compute benchmark metrics from per-sample execution results.

    pass@1 uses the first sample of each task; pass@k uses any-of-k.
    """
    by_task: dict[str, list[SampleResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    n_tasks = len(by_task)
    if n_tasks == 0:
        return {"n_tasks": 0}

    pass1 = sum(1 for rs in by_task.values()
                if _first(rs).tests_passed and _first(rs).accepted)
    passk = sum(1 for rs in by_task.values()
                if any(r.tests_passed and r.accepted for r in rs[:k]))

    all_samples = [r for rs in by_task.values() for r in rs]
    n = len(all_samples)

    def rate(pred) -> float:
        return round(sum(1 for r in all_samples if pred(r)) / n, 4)

    accepted_changed = [r.changed_lines for r in all_samples if r.accepted]
    patch_eff = round(sum(accepted_changed) / len(accepted_changed), 2) if accepted_changed else 0.0

    return {
        "n_tasks": n_tasks,
        "n_samples": n,
        "pass@1": round(pass1 / n_tasks, 4),
        f"pass@{k}": round(passk / n_tasks, 4),
        "compile_rate": rate(lambda r: r.compiled),
        "test_pass_rate": rate(lambda r: r.tests_passed),
        "lint_rate": rate(lambda r: r.lint_ok),
        "typecheck_rate": rate(lambda r: r.typecheck_ok),
        "format_compliance": rate(lambda r: r.format_ok),
        "hallucinated_api_rate": rate(lambda r: r.hallucinated_api),
        "security_violations": rate(lambda r: not r.security_ok),
        "patch_efficiency_avg_changed_lines": patch_eff,
    }


def _first(rs: list[SampleResult]) -> SampleResult:
    return sorted(rs, key=lambda r: r.sample_idx)[0]


def regression_rate(metrics_a: dict, metrics_b: dict, key: str = "test_pass_rate") -> float:
    """Drop in `key` from baseline a to candidate b, as a fraction (>=0 means worse)."""
    a, b = metrics_a.get(key, 0.0), metrics_b.get(key, 0.0)
    if a == 0:
        return 0.0
    return round(max(0.0, (a - b) / a), 4)
