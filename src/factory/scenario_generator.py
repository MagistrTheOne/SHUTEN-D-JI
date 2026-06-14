"""
Scenario Generator — produces future branches from world states.

Given a WorldState, generates multiple plausible futures with:
  - Causal chains
  - Branch points (decisions + uncertainties)
  - Terminal states
  - Probability estimates
  - Uncertainty decomposition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.factory.state_generator import WorldState, Event, Domain


class UncertaintyType(str, Enum):
    ALEATORIC = "aleatoric"           # irreducible randomness
    EPISTEMIC = "epistemic"           # reducible with more info
    DECISION_DEPENDENT = "decision"   # depends on choices made


class Resolvability(str, Enum):
    OBSERVABLE = "observable"
    PARTIALLY_OBSERVABLE = "partially_observable"
    UNOBSERVABLE = "unobservable"


class CausalStep(BaseModel):
    """Single step in a causal chain."""
    step: int
    event_description: str
    cause: str
    mechanism: str
    preconditions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class BranchPoint(BaseModel):
    """Point where futures diverge based on decisions or randomness."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    decision_required: bool
    options: list[str] = Field(default_factory=list)
    information_available: list[str] = Field(default_factory=list)
    timestamp_offset: str = ""


class OutcomeMetric(BaseModel):
    metric: str
    value: float
    relative_to_baseline: float = 0.0


class UncertaintyDecomposition(BaseModel):
    aleatoric: float = Field(ge=0.0, le=1.0)
    epistemic: float = Field(ge=0.0, le=1.0)
    decision_dependent: float = Field(ge=0.0, le=1.0)


class Scenario(BaseModel):
    """Single future scenario."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    causal_chain: list[CausalStep] = Field(default_factory=list)
    branch_points: list[BranchPoint] = Field(default_factory=list)
    terminal_state_summary: str = ""
    outcome_metrics: list[OutcomeMetric] = Field(default_factory=list)
    uncertainty: UncertaintyDecomposition = Field(
        default_factory=lambda: UncertaintyDecomposition(
            aleatoric=0.33, epistemic=0.33, decision_dependent=0.34
        )
    )


class CriticalUncertainty(BaseModel):
    description: str
    scenarios_affected: list[str] = Field(default_factory=list)
    resolvability: Resolvability = Resolvability.PARTIALLY_OBSERVABLE


class ScenarioBundle(BaseModel):
    """Complete scenario analysis for a world state."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    initial_state_id: str
    domain: Domain
    time_horizon: str  # e.g., "6_months", "1_year", "5_years"
    scenarios: list[Scenario] = Field(default_factory=list)
    critical_uncertainties: list[CriticalUncertainty] = Field(default_factory=list)
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


# --- Generator ---


@dataclass
class ScenarioGeneratorConfig:
    min_scenarios: int = 3
    max_scenarios: int = 5
    min_causal_steps: int = 3
    max_causal_steps: int = 10
    time_horizons: list[str] = field(
        default_factory=lambda: ["3_months", "6_months", "1_year", "3_years"]
    )
    require_divergent_outcomes: bool = True
    seed: Optional[int] = None


class ScenarioGenerator:
    """
    Generates scenario bundles from world states.

    Two modes:
      1. Rule-based: uses heuristics to branch on key uncertainties
      2. LLM-assisted: uses a model to generate rich causal narratives

    This implementation provides the structural framework.
    Rich content generation requires an LLM inference endpoint.
    """

    def __init__(self, config: ScenarioGeneratorConfig):
        self.config = config
        import random
        self.rng = random.Random(config.seed)

    def generate(self, world_state: WorldState) -> ScenarioBundle:
        """Generate a scenario bundle from a world state."""
        time_horizon = self.rng.choice(self.config.time_horizons)
        num_scenarios = self.rng.randint(self.config.min_scenarios, self.config.max_scenarios)

        # Identify key uncertainties from the world state
        uncertainties = self._identify_uncertainties(world_state)

        # Generate scenarios branching on those uncertainties
        scenarios = self._generate_scenarios(world_state, uncertainties, num_scenarios)

        # Normalize probabilities
        total_prob = sum(s.probability for s in scenarios)
        if total_prob > 0:
            for s in scenarios:
                s.probability = round(s.probability / max(total_prob, 1.0), 3)

        return ScenarioBundle(
            initial_state_id=world_state.id,
            domain=world_state.domain,
            time_horizon=time_horizon,
            scenarios=scenarios,
            critical_uncertainties=uncertainties,
            metadata={
                "generator": "rule_based",
                "complexity": world_state.complexity_level,
                "num_entities": len(world_state.entities),
            },
        )

    def _identify_uncertainties(self, state: WorldState) -> list[CriticalUncertainty]:
        """Extract critical uncertainties from world state events and risks."""
        uncertainties = []

        # High-impact events with moderate probability are key branch points
        for event in state.events:
            if abs(event.impact_magnitude) > 0.4 and 0.2 < event.probability < 0.8:
                uncertainties.append(CriticalUncertainty(
                    description=f"Will {event.type.value} occur? ({event.description})",
                    scenarios_affected=[],
                    resolvability=self.rng.choice(list(Resolvability)),
                ))

        # Risks with high impact
        for risk in state.risks:
            if risk.impact > 0.5:
                uncertainties.append(CriticalUncertainty(
                    description=f"Risk materialization: {risk.source}",
                    scenarios_affected=[],
                    resolvability=Resolvability.PARTIALLY_OBSERVABLE,
                ))

        return uncertainties[:5]  # Cap at 5 critical uncertainties

    def _generate_scenarios(
        self,
        state: WorldState,
        uncertainties: list[CriticalUncertainty],
        num_scenarios: int,
    ) -> list[Scenario]:
        """Generate divergent scenarios."""
        labels = ["Optimistic", "Baseline", "Pessimistic", "Disruption", "Wild Card"]
        scenarios = []

        for i in range(num_scenarios):
            label = labels[i] if i < len(labels) else f"Scenario_{i+1}"
            num_steps = self.rng.randint(
                self.config.min_causal_steps, self.config.max_causal_steps
            )

            causal_chain = self._generate_causal_chain(state, num_steps, label)
            branch_points = self._generate_branch_points(state, uncertainties)

            # Probability distribution: baseline highest, extremes lower
            if label == "Baseline":
                prob = self.rng.uniform(0.3, 0.5)
            elif label in ("Optimistic", "Pessimistic"):
                prob = self.rng.uniform(0.15, 0.3)
            else:
                prob = self.rng.uniform(0.05, 0.15)

            scenario = Scenario(
                label=label,
                description=f"[PLACEHOLDER: {label} scenario for {state.domain.value} state]",
                probability=round(prob, 3),
                causal_chain=causal_chain,
                branch_points=branch_points,
                terminal_state_summary=f"[Terminal state after {num_steps} causal steps in {label} path]",
                outcome_metrics=[
                    OutcomeMetric(metric="primary_objective", value=self._outcome_value(label)),
                    OutcomeMetric(metric="risk_exposure", value=round(self.rng.uniform(0.1, 0.9), 2)),
                ],
            )

            # Link to affected uncertainties
            for u in uncertainties:
                u.scenarios_affected.append(scenario.id)

            scenarios.append(scenario)

        return scenarios

    def _generate_causal_chain(
        self, state: WorldState, num_steps: int, scenario_label: str
    ) -> list[CausalStep]:
        """Generate a causal chain of events."""
        chain = []
        for step in range(num_steps):
            chain.append(CausalStep(
                step=step + 1,
                event_description=f"[Step {step+1} in {scenario_label} chain]",
                cause=f"[Caused by step {step}]" if step > 0 else "[Initial condition]",
                mechanism=f"[Mechanism placeholder for {state.domain.value}]",
                preconditions=[f"precondition_{step}"],
                confidence=round(self.rng.uniform(0.5, 0.95), 2),
            ))
        return chain

    def _generate_branch_points(
        self, state: WorldState, uncertainties: list[CriticalUncertainty]
    ) -> list[BranchPoint]:
        """Generate decision/branch points."""
        branch_points = []
        for u in uncertainties[:3]:
            branch_points.append(BranchPoint(
                description=u.description,
                decision_required=self.rng.random() > 0.5,
                options=["Option A", "Option B", "Option C"],
                information_available=["[Available info placeholder]"],
            ))
        return branch_points

    def _outcome_value(self, label: str) -> float:
        """Generate outcome metric value based on scenario type."""
        ranges = {
            "Optimistic": (0.6, 0.95),
            "Baseline": (0.35, 0.65),
            "Pessimistic": (0.1, 0.4),
            "Disruption": (0.0, 0.3),
            "Wild Card": (0.0, 1.0),
        }
        lo, hi = ranges.get(label, (0.2, 0.8))
        return round(self.rng.uniform(lo, hi), 2)
