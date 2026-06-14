"""
Evaluation Metrics for SHUTEN-DŌJI.

Tiered evaluation:
  Tier 1: Automated verifiable (continuous)
  Tier 2: Automated non-verifiable (weekly)
  Tier 3: Human evaluation (monthly)
  Tier 4: Real-world anchoring (quarterly)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


@dataclass
class EvalResult:
    """Single evaluation result."""
    metric_name: str
    value: float
    confidence_interval: Optional[tuple[float, float]] = None
    num_samples: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# --- Tier 1: Automated Verifiable Metrics ---


def planning_success_rate(
    predicted_actions: list[list[str]],
    ground_truth_outcomes: list[bool],
) -> EvalResult:
    """Fraction of plans that achieved their objectives in simulation."""
    if not ground_truth_outcomes:
        return EvalResult("planning_success_rate", 0.0)
    rate = sum(ground_truth_outcomes) / len(ground_truth_outcomes)
    return EvalResult(
        metric_name="planning_success_rate",
        value=rate,
        num_samples=len(ground_truth_outcomes),
    )


def prediction_calibration(
    predicted_probs: list[float],
    actual_outcomes: list[bool],
    num_bins: int = 10,
) -> EvalResult:
    """
    Expected Calibration Error (ECE).
    Lower is better — measures how well predicted probabilities match actual frequencies.
    """
    if not predicted_probs:
        return EvalResult("calibration_ece", 1.0)

    probs = np.array(predicted_probs)
    outcomes = np.array(actual_outcomes, dtype=float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(num_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = outcomes[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

    return EvalResult(
        metric_name="calibration_ece",
        value=float(ece),
        num_samples=total,
    )


def brier_score(
    predicted_probs: list[float],
    actual_outcomes: list[bool],
) -> EvalResult:
    """Brier score — measures accuracy of probabilistic predictions."""
    if not predicted_probs:
        return EvalResult("brier_score", 1.0)
    probs = np.array(predicted_probs)
    outcomes = np.array(actual_outcomes, dtype=float)
    score = float(np.mean((probs - outcomes) ** 2))
    return EvalResult("brier_score", score, num_samples=len(probs))


def analysis_completeness(
    analysis_output: str,
    required_elements: list[str],
) -> EvalResult:
    """
    Fraction of required analytical elements present in output.
    Rubric-based: checks for mentions of key factors.
    """
    if not required_elements:
        return EvalResult("analysis_completeness", 1.0)

    found = sum(1 for elem in required_elements if elem.lower() in analysis_output.lower())
    score = found / len(required_elements)
    return EvalResult(
        metric_name="analysis_completeness",
        value=score,
        num_samples=1,
        metadata={"found": found, "total": len(required_elements)},
    )


def tool_use_accuracy(
    predicted_tool_calls: list[dict],
    expected_tool_calls: list[dict],
) -> EvalResult:
    """Accuracy of tool invocations (correct tool + correct args)."""
    if not expected_tool_calls:
        return EvalResult("tool_use_accuracy", 1.0)

    correct = 0
    for pred, exp in zip(predicted_tool_calls, expected_tool_calls):
        if pred.get("tool_name") == exp.get("tool_name"):
            if pred.get("args") == exp.get("args"):
                correct += 1
            else:
                correct += 0.5  # partial credit for right tool wrong args

    score = correct / len(expected_tool_calls)
    return EvalResult("tool_use_accuracy", score, num_samples=len(expected_tool_calls))


def format_compliance(
    outputs: list[str],
    required_tags: list[str],
) -> EvalResult:
    """Check that outputs use required structural tags."""
    if not outputs:
        return EvalResult("format_compliance", 0.0)

    compliant = 0
    for output in outputs:
        has_all = all(tag in output for tag in required_tags)
        if has_all:
            compliant += 1

    return EvalResult("format_compliance", compliant / len(outputs), num_samples=len(outputs))


# --- Tier 2: Automated Non-Verifiable ---


def scenario_diversity(
    scenarios: list[str],
    embedding_fn=None,
) -> EvalResult:
    """
    Measure diversity of generated scenarios via pairwise distance.
    Requires an embedding function (model inference).
    Returns placeholder if no embedding_fn provided.
    """
    if embedding_fn is None or len(scenarios) < 2:
        return EvalResult("scenario_diversity", 0.0, metadata={"note": "requires embedding_fn"})

    embeddings = [embedding_fn(s) for s in scenarios]
    embeddings = torch.stack(embeddings)

    # Pairwise cosine distance
    normed = embeddings / embeddings.norm(dim=-1, keepdim=True)
    similarity_matrix = torch.mm(normed, normed.t())

    # Average off-diagonal similarity (lower = more diverse)
    mask = ~torch.eye(len(scenarios), dtype=torch.bool)
    avg_similarity = similarity_matrix[mask].mean().item()
    diversity = 1.0 - avg_similarity

    return EvalResult("scenario_diversity", diversity, num_samples=len(scenarios))


def reasoning_consistency(
    reasoning_chain: list[str],
) -> EvalResult:
    """
    Check internal consistency of reasoning steps.
    Placeholder — requires NLI model for contradiction detection.
    """
    return EvalResult(
        "reasoning_consistency",
        0.0,
        metadata={"note": "requires NLI model for implementation"},
    )


# --- Composite Metrics ---


@dataclass
class EvalSuite:
    """Collection of evaluation results."""
    results: list[EvalResult]
    overall_score: float = 0.0

    def compute_overall(self, weights: Optional[dict[str, float]] = None) -> float:
        """Compute weighted overall score."""
        if not self.results:
            return 0.0

        if weights is None:
            self.overall_score = sum(r.value for r in self.results) / len(self.results)
        else:
            total_weight = 0.0
            weighted_sum = 0.0
            for r in self.results:
                w = weights.get(r.metric_name, 1.0)
                weighted_sum += r.value * w
                total_weight += w
            self.overall_score = weighted_sum / max(total_weight, 1e-8)

        return self.overall_score

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "metrics": {r.metric_name: r.value for r in self.results},
            "details": [
                {"name": r.metric_name, "value": r.value, "samples": r.num_samples}
                for r in self.results
            ],
        }
