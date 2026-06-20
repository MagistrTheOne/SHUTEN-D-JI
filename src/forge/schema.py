"""
CODE FORGE schema — canonical episode record + Code Constitution block sets.

Single source of truth shared by the task factory, gates, judge, packer, and
benchmark harness. See docs/CODE_FORGE_V1_SPEC.md sections 1, 2, 4.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Layer(str, Enum):
    GOLD = "gold"
    SYNTHETIC = "synthetic"
    FAILURE = "failure"
    REPLAY = "replay"
    GENERAL = "general"


class TaskType(str, Enum):
    FUNCTION_IMPL = "function_impl"
    REPO_BUGFIX = "repo_bugfix"
    MULTI_FILE = "multi_file"
    TEST_GEN = "test_gen"
    REFACTOR = "refactor"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    CODE_REVIEW = "code_review"
    REPAIR_TRAJECTORY = "repair_trajectory"


class Language(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    SQL = "sql"
    CPP = "cpp"
    BASH = "bash"
    YAML = "yaml"


class BugClass(str, Enum):
    OFF_BY_ONE = "off_by_one"
    NULL_HANDLING = "null_handling"
    RACE_CONDITION = "race_condition"
    SQL_INJECTION = "sql_injection"
    INCOMPATIBLE_API = "incompatible_api"
    BROKEN_TYPING = "broken_typing"
    BAD_MIGRATION = "bad_migration"
    FLAKY_PASS = "flaky_pass"
    PUBLIC_CONTRACT_BREAK = "public_contract_break"
    OVER_REFACTOR = "over_refactor"
    HALLUCINATED_PACKAGE = "hallucinated_package"
    SYMPTOM_NOT_CAUSE = "symptom_not_cause"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EditScope(str, Enum):
    SINGLE_LINE = "single_line"
    SINGLE_FUNC = "single_func"
    SINGLE_FILE = "single_file"
    MULTI_FILE = "multi_file"


# --- Code Constitution: conditional block sets per task type (spec section 1.1) ---

CONSTITUTION_BLOCKS: dict[TaskType, tuple[str, ...]] = {
    TaskType.FUNCTION_IMPL: ("ASSUMPTIONS", "IMPLEMENTATION", "TESTS", "COMPLEXITY"),
    TaskType.REPO_BUGFIX: ("DIAGNOSIS", "ROOT CAUSE", "PATCH", "TESTS", "VERIFICATION"),
    TaskType.MULTI_FILE: ("DIAGNOSIS", "PLAN", "PATCH", "TESTS", "VERIFICATION"),
    TaskType.TEST_GEN: ("COVERAGE TARGETS", "TESTS", "VERIFICATION"),
    TaskType.REFACTOR: ("INVARIANTS", "PLAN", "PATCH", "TESTS", "VERIFICATION"),
    TaskType.SECURITY: ("THREAT", "DIAGNOSIS", "PATCH", "TESTS", "VERIFICATION"),
    TaskType.PERFORMANCE: ("BOTTLENECK", "OPTIONS", "PATCH", "BENCHMARK", "VERIFICATION"),
    TaskType.ARCHITECTURE: (
        "STATE", "CONSTRAINTS", "OPTIONS", "DECISION",
        "TRADE-OFFS", "IMPLEMENTATION PLAN", "RISKS",
    ),
    TaskType.CODE_REVIEW: ("BLOCKING ISSUES", "PROPOSED CORRECTIONS"),
    TaskType.REPAIR_TRAJECTORY: ("ROOT CAUSE", "CORRECTED PATCH", "TESTS"),
}

# Tokens/phrases that must never appear in the assistant target (spec section 1.2).
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"Here's a thinking process",
    r"\bStep\s+\d+\s*:",
    r"\btool_use\b",
    r"\bcommunicate\b",
    r"\bobserve\b",
    r"\bcritique\b",
    r"<think>",
    r"\bTODO\b",
    r"\bFIXME\b",
)

SYSTEM_PROMPT = (
    "You are SHUTEN-CODE, the engineering module of SHUTEN by NULLXES DAI.\n"
    "You produce verified, minimal, production-grade code.\n"
    "Cycle: UNDERSTAND -> PLAN -> PATCH -> TEST -> DIAGNOSE -> FIX -> VERIFY.\n"
    "Rules: do not rewrite a whole project for one bug; do not invent files, "
    "APIs, or packages; keep public contracts; write tests; return a minimal "
    "complete result; state explicitly what was verified and what was not. "
    "No chain-of-thought, no Step labels, no fabricated run output."
)


class TaskPrompt(BaseModel):
    task: str
    repo_context: str = ""
    constraints: list[str] = Field(default_factory=list)


class Solution(BaseModel):
    format: str = "answer"  # "answer" | "diff"
    content: str


class DiffStats(BaseModel):
    added: int = 0
    removed: int = 0
    files: int = 0


class Verification(BaseModel):
    compile_ok: bool = False
    tests_ok: bool = False
    lint_ok: bool = False
    typecheck_ok: bool = False
    security_ok: bool = False
    diff_stats: DiffStats = Field(default_factory=DiffStats)

    @property
    def all_pass(self) -> bool:
        return (
            self.compile_ok
            and self.tests_ok
            and self.lint_ok
            and self.typecheck_ok
            and self.security_ok
        )


class Episode(BaseModel):
    id: str
    layer: Layer
    task_type: TaskType
    language: Language
    bug_class: BugClass | None = None
    difficulty: Difficulty = Difficulty.MEDIUM
    edit_scope: EditScope = EditScope.SINGLE_FUNC
    prompt: TaskPrompt
    solution: Solution
    # Runnable code the sandbox compiles/tests. If empty, the judge extracts it
    # from the fenced code blocks in `solution.content`.
    exec_code: str = ""
    tests: str = ""
    verification: Verification = Field(default_factory=Verification)
    metadata: dict = Field(default_factory=dict)


class CodeConstitution:
    """Helpers to validate that an assistant answer follows the block contract."""

    @staticmethod
    def required_blocks(task_type: TaskType) -> tuple[str, ...]:
        return CONSTITUTION_BLOCKS[task_type]

    @staticmethod
    def missing_blocks(task_type: TaskType, answer: str) -> list[str]:
        upper = answer.upper()
        return [b for b in CONSTITUTION_BLOCKS[task_type] if b not in upper]
