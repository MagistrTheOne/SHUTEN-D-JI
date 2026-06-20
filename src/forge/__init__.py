"""
NULLXES CODE FORGE — execution-verified code dataset factory.

Pipeline:
    TaskFactory -> CandidateGen -> Sandbox -> Gates -> Judge -> Packer

The judge is execution. The LLM is never the judge of its own code.
"""

from __future__ import annotations

from src.forge.schema import (
    BugClass,
    CodeConstitution,
    Difficulty,
    EditScope,
    Episode,
    Language,
    Layer,
    TaskType,
    Verification,
)

__all__ = [
    "BugClass",
    "CodeConstitution",
    "Difficulty",
    "EditScope",
    "Episode",
    "Language",
    "Layer",
    "TaskType",
    "Verification",
]
