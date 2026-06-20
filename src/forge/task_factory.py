"""
CODE FORGE Task Factory — parameterized task-matrix planner.

Turns the spec distributions (language mix, cluster mix, bug classes, difficulty,
edit scope) into a concrete list of TaskSpec slots to be filled by gold authoring
or candidate generation. This is the planner, not the content generator.

See docs/CODE_FORGE_V1_SPEC.md section 2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.forge.schema import (
    BugClass,
    Difficulty,
    EditScope,
    Language,
    TaskType,
)

# Cluster -> count, scaled to ~500 (spec 2.3).
CLUSTER_COUNTS: dict[TaskType, int] = {
    TaskType.FUNCTION_IMPL: 75,
    TaskType.REPO_BUGFIX: 113,
    TaskType.MULTI_FILE: 75,
    TaskType.TEST_GEN: 50,
    TaskType.REFACTOR: 38,
    TaskType.SECURITY: 38,
    TaskType.PERFORMANCE: 25,
    TaskType.ARCHITECTURE: 38,
    TaskType.REPAIR_TRAJECTORY: 50,
}

# Language mix as weights (spec 2.2). "mixed-repo" is modeled as MULTI_FILE tasks.
LANGUAGE_WEIGHTS: dict[Language, float] = {
    Language.TYPESCRIPT: 0.25,
    Language.PYTHON: 0.25,
    Language.SQL: 0.12,
    Language.CPP: 0.15,
    Language.BASH: 0.08,
    Language.YAML: 0.05,
}
# Remaining 0.10 (mixed-repo) is absorbed by multi-file tasks across languages.

# Which bug classes are plausible per language (gate realism).
LANG_BUG_CLASSES: dict[Language, tuple[BugClass, ...]] = {
    Language.PYTHON: (
        BugClass.OFF_BY_ONE, BugClass.NULL_HANDLING, BugClass.BROKEN_TYPING,
        BugClass.SYMPTOM_NOT_CAUSE, BugClass.HALLUCINATED_PACKAGE,
        BugClass.PUBLIC_CONTRACT_BREAK, BugClass.OVER_REFACTOR, BugClass.FLAKY_PASS,
    ),
    Language.TYPESCRIPT: (
        BugClass.NULL_HANDLING, BugClass.BROKEN_TYPING, BugClass.OFF_BY_ONE,
        BugClass.INCOMPATIBLE_API, BugClass.PUBLIC_CONTRACT_BREAK,
        BugClass.HALLUCINATED_PACKAGE, BugClass.OVER_REFACTOR,
    ),
    Language.SQL: (
        BugClass.SQL_INJECTION, BugClass.BAD_MIGRATION, BugClass.OFF_BY_ONE,
        BugClass.PUBLIC_CONTRACT_BREAK,
    ),
    Language.CPP: (
        BugClass.OFF_BY_ONE, BugClass.RACE_CONDITION, BugClass.NULL_HANDLING,
        BugClass.INCOMPATIBLE_API, BugClass.SYMPTOM_NOT_CAUSE,
    ),
    Language.BASH: (
        BugClass.NULL_HANDLING, BugClass.SYMPTOM_NOT_CAUSE, BugClass.OFF_BY_ONE,
    ),
    Language.YAML: (
        BugClass.INCOMPATIBLE_API, BugClass.PUBLIC_CONTRACT_BREAK,
    ),
}

DIFFICULTY_WEIGHTS: dict[Difficulty, float] = {
    Difficulty.EASY: 0.30,
    Difficulty.MEDIUM: 0.45,
    Difficulty.HARD: 0.25,
}

# Default edit scope per task type.
TASK_EDIT_SCOPE: dict[TaskType, EditScope] = {
    TaskType.FUNCTION_IMPL: EditScope.SINGLE_FUNC,
    TaskType.REPO_BUGFIX: EditScope.SINGLE_FILE,
    TaskType.MULTI_FILE: EditScope.MULTI_FILE,
    TaskType.TEST_GEN: EditScope.SINGLE_FILE,
    TaskType.REFACTOR: EditScope.SINGLE_FILE,
    TaskType.SECURITY: EditScope.SINGLE_FILE,
    TaskType.PERFORMANCE: EditScope.SINGLE_FUNC,
    TaskType.ARCHITECTURE: EditScope.MULTI_FILE,
    TaskType.REPAIR_TRAJECTORY: EditScope.SINGLE_FUNC,
}

# Tasks that carry a bug class.
BUGGY_TASKS = {
    TaskType.REPO_BUGFIX,
    TaskType.SECURITY,
    TaskType.REPAIR_TRAJECTORY,
}


@dataclass
class TaskSpec:
    slot_id: str
    task_type: TaskType
    language: Language
    bug_class: BugClass | None
    difficulty: Difficulty
    edit_scope: EditScope
    layer_hint: str  # gold | synthetic | failure
    constraints: list[str] = field(default_factory=list)


def _weighted_choice(rng: random.Random, weights: dict):
    keys = list(weights.keys())
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _layer_for(task_type: TaskType) -> str:
    if task_type == TaskType.REPAIR_TRAJECTORY:
        return "failure"
    # Architecture and security skew to hand gold; rest mostly synthetic.
    if task_type in (TaskType.ARCHITECTURE, TaskType.SECURITY):
        return "gold"
    return "synthetic"


def _constraints_for(task_type: TaskType, language: Language) -> list[str]:
    base = ["no new third-party dependency", "keep the public API stable"]
    if task_type == TaskType.PERFORMANCE:
        base.append("preserve observable behavior; only improve performance")
    if task_type == TaskType.REFACTOR:
        base.append("no behavior change; tests must still pass unchanged")
    if task_type == TaskType.MULTI_FILE:
        base.append("minimal cross-file edits; do not touch unrelated modules")
    if language == Language.SQL:
        base.append("migration must be reversible")
    return base


def plan_tasks(seed: int = 42) -> list[TaskSpec]:
    """Produce the full ~500-slot task plan deterministically."""
    rng = random.Random(seed)
    specs: list[TaskSpec] = []
    counter = 0

    for task_type, count in CLUSTER_COUNTS.items():
        for _ in range(count):
            # Multi-file/architecture tasks pull from full language set incl. mixed.
            language = _weighted_choice(rng, LANGUAGE_WEIGHTS)
            difficulty = _weighted_choice(rng, DIFFICULTY_WEIGHTS)
            bug_class = None
            if task_type in BUGGY_TASKS:
                if task_type == TaskType.SECURITY and language == Language.SQL:
                    bug_class = BugClass.SQL_INJECTION
                else:
                    bug_class = rng.choice(LANG_BUG_CLASSES[language])
            counter += 1
            specs.append(
                TaskSpec(
                    slot_id=f"cf-{task_type.value}-{counter:04d}",
                    task_type=task_type,
                    language=language,
                    bug_class=bug_class,
                    difficulty=difficulty,
                    edit_scope=TASK_EDIT_SCOPE[task_type],
                    layer_hint=_layer_for(task_type),
                    constraints=_constraints_for(task_type, language),
                )
            )

    rng.shuffle(specs)
    return specs


def plan_summary(specs: list[TaskSpec]) -> dict:
    """Aggregate counts for verification against the spec distributions."""
    by_type: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for s in specs:
        by_type[s.task_type.value] = by_type.get(s.task_type.value, 0) + 1
        by_lang[s.language.value] = by_lang.get(s.language.value, 0) + 1
        by_layer[s.layer_hint] = by_layer.get(s.layer_hint, 0) + 1
        by_difficulty[s.difficulty.value] = by_difficulty.get(s.difficulty.value, 0) + 1
    return {
        "total": len(specs),
        "by_task_type": by_type,
        "by_language": by_lang,
        "by_layer": by_layer,
        "by_difficulty": by_difficulty,
    }


if __name__ == "__main__":
    import json

    plan = plan_tasks()
    print(json.dumps(plan_summary(plan), indent=2))
