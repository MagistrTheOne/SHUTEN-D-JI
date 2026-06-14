"""
State Generator — produces realistic machine-readable world states.

World states serve as initial conditions for strategic scenarios.
Each state contains entities, relationships, events, constraints, risks, objectives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    ORGANIZATION = "organization"
    PERSON = "person"
    SYSTEM = "system"
    RESOURCE = "resource"
    LOCATION = "location"
    ASSET = "asset"


class RelationshipType(str, Enum):
    CONTROLS = "controls"
    DEPENDS_ON = "depends_on"
    COMPETES_WITH = "competes_with"
    ALLIES_WITH = "allies_with"
    SUPPLIES = "supplies"
    REPORTS_TO = "reports_to"
    THREATENS = "threatens"
    FUNDS = "funds"


class EventType(str, Enum):
    DISRUPTION = "disruption"
    OPPORTUNITY = "opportunity"
    THREAT = "threat"
    DEADLINE = "deadline"
    TRANSITION = "transition"
    DISCOVERY = "discovery"


class ConstraintType(str, Enum):
    RESOURCE = "resource"
    TEMPORAL = "temporal"
    REGULATORY = "regulatory"
    PHYSICAL = "physical"
    INFORMATIONAL = "informational"
    POLITICAL = "political"


class Domain(str, Enum):
    BUSINESS = "business"
    GEOPOLITICS = "geopolitics"
    LOGISTICS = "logistics"
    MARKETS = "markets"
    TECHNOLOGY = "technology"
    ORGANIZATIONAL = "organizational"
    OPERATIONS = "operations"


# --- Pydantic models for validated world state ---


class Objective(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    owner_id: str
    description: str
    priority: float = Field(ge=0.0, le=1.0)
    deadline: Optional[str] = None
    measurable_criteria: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: EntityType
    attributes: dict[str, str | float | bool] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=list)


class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: RelationshipType
    strength: float = Field(ge=0.0, le=1.0)
    volatility: float = Field(ge=0.0, le=1.0, default=0.1)


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    description: str
    affected_entity_ids: list[str] = Field(default_factory=list)
    probability: float = Field(ge=0.0, le=1.0)
    impact_magnitude: float = Field(ge=-1.0, le=1.0)
    timestamp_offset: Optional[str] = None


class Constraint(BaseModel):
    type: ConstraintType
    scope_entity_ids: list[str] = Field(default_factory=list)
    description: str
    severity: float = Field(ge=0.0, le=1.0)


class Risk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    probability: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    affected_entity_ids: list[str] = Field(default_factory=list)
    mitigatable: bool = True


class InformationState(BaseModel):
    known_to_all: list[str] = Field(default_factory=list)
    known_to_subset: dict[str, list[str]] = Field(default_factory=dict)
    unknown_but_discoverable: list[str] = Field(default_factory=list)


class WorldState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    domain: Domain
    complexity_level: int = Field(ge=1, le=5)
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    information_state: InformationState = Field(default_factory=InformationState)
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


# --- Generator ---


@dataclass
class StateGeneratorConfig:
    """Configuration for state generation."""
    domains: list[Domain] = field(default_factory=lambda: list(Domain))
    min_entities: int = 3
    max_entities: int = 50
    min_relationships: int = 2
    max_relationships: int = 100
    complexity_distribution: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.35, 0.2, 0.1]  # levels 1-5
    )
    seed: Optional[int] = None


class StateGenerator:
    """
    Generates realistic world states for strategic scenarios.

    Two modes:
      1. Template-based: parameterized templates seeded from real-world patterns
      2. LLM-assisted: uses a language model to generate rich, diverse states

    This implementation provides the template-based mode.
    LLM-assisted mode requires an inference endpoint (see generate_with_llm).
    """

    def __init__(self, config: StateGeneratorConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def generate(self, domain: Optional[Domain] = None, complexity: Optional[int] = None) -> WorldState:
        """Generate a single world state."""
        if domain is None:
            domain = self.rng.choice(self.config.domains)

        if complexity is None:
            complexity = self.rng.choices(
                range(1, 6), weights=self.config.complexity_distribution
            )[0]

        num_entities = self._scale_by_complexity(
            complexity, self.config.min_entities, self.config.max_entities
        )
        num_relationships = self._scale_by_complexity(
            complexity, self.config.min_relationships, self.config.max_relationships
        )

        entities = self._generate_entities(domain, num_entities)
        relationships = self._generate_relationships(entities, num_relationships)
        events = self._generate_events(domain, entities, complexity)
        constraints = self._generate_constraints(domain, entities, complexity)
        risks = self._generate_risks(entities, events, complexity)
        objectives = self._assign_objectives(entities, domain, complexity)

        return WorldState(
            domain=domain,
            complexity_level=complexity,
            entities=entities,
            relationships=relationships,
            events=events,
            constraints=constraints,
            risks=risks,
            metadata={
                "generator": "template",
                "num_entities": len(entities),
                "num_relationships": len(relationships),
            },
        )

    def generate_batch(self, count: int, **kwargs) -> list[WorldState]:
        """Generate multiple world states."""
        return [self.generate(**kwargs) for _ in range(count)]

    def _scale_by_complexity(self, complexity: int, min_val: int, max_val: int) -> int:
        ratio = (complexity - 1) / 4.0
        return int(min_val + ratio * (max_val - min_val))

    def _generate_entities(self, domain: Domain, count: int) -> list[Entity]:
        """Generate entities appropriate for the domain."""
        templates = DOMAIN_ENTITY_TEMPLATES.get(domain, DOMAIN_ENTITY_TEMPLATES[Domain.BUSINESS])
        entities = []
        for i in range(count):
            template = self.rng.choice(templates)
            entity = Entity(
                name=f"{template['prefix']}_{i:03d}",
                type=template["type"],
                attributes=self._randomize_attributes(template.get("attributes", {})),
                capabilities=self.rng.sample(
                    template.get("capabilities", ["operate", "decide", "adapt"]),
                    k=min(3, len(template.get("capabilities", ["operate"]))),
                ),
            )
            entities.append(entity)
        return entities

    def _generate_relationships(self, entities: list[Entity], count: int) -> list[Relationship]:
        """Generate relationships between entities."""
        relationships = []
        entity_ids = [e.id for e in entities]
        rel_types = list(RelationshipType)

        for _ in range(min(count, len(entity_ids) * (len(entity_ids) - 1))):
            src, tgt = self.rng.sample(entity_ids, 2)
            relationships.append(Relationship(
                source_id=src,
                target_id=tgt,
                type=self.rng.choice(rel_types),
                strength=round(self.rng.uniform(0.2, 1.0), 2),
                volatility=round(self.rng.uniform(0.0, 0.5), 2),
            ))
        return relationships

    def _generate_events(self, domain: Domain, entities: list[Entity], complexity: int) -> list[Event]:
        """Generate pending/potential events."""
        num_events = max(1, complexity)
        events = []
        event_types = list(EventType)

        for _ in range(num_events):
            affected = self.rng.sample(
                [e.id for e in entities], k=min(3, len(entities))
            )
            events.append(Event(
                type=self.rng.choice(event_types),
                description=f"[PLACEHOLDER: {domain.value} event requiring LLM generation]",
                affected_entity_ids=affected,
                probability=round(self.rng.uniform(0.1, 0.9), 2),
                impact_magnitude=round(self.rng.uniform(-0.8, 0.8), 2),
            ))
        return events

    def _generate_constraints(
        self, domain: Domain, entities: list[Entity], complexity: int
    ) -> list[Constraint]:
        """Generate constraints on the system."""
        num_constraints = max(1, complexity - 1)
        constraints = []
        constraint_types = list(ConstraintType)

        for _ in range(num_constraints):
            scope = self.rng.sample(
                [e.id for e in entities], k=min(2, len(entities))
            )
            constraints.append(Constraint(
                type=self.rng.choice(constraint_types),
                scope_entity_ids=scope,
                description=f"[PLACEHOLDER: {domain.value} constraint]",
                severity=round(self.rng.uniform(0.3, 0.9), 2),
            ))
        return constraints

    def _generate_risks(
        self, entities: list[Entity], events: list[Event], complexity: int
    ) -> list[Risk]:
        """Generate risks based on entities and events."""
        num_risks = max(1, complexity)
        risks = []

        for _ in range(num_risks):
            affected = self.rng.sample(
                [e.id for e in entities], k=min(2, len(entities))
            )
            risks.append(Risk(
                source=f"[Derived from event/entity interaction]",
                probability=round(self.rng.uniform(0.1, 0.7), 2),
                impact=round(self.rng.uniform(0.2, 0.9), 2),
                affected_entity_ids=affected,
                mitigatable=self.rng.random() > 0.3,
            ))
        return risks

    def _assign_objectives(
        self, entities: list[Entity], domain: Domain, complexity: int
    ) -> list[Entity]:
        """Assign objectives to key entities."""
        num_with_objectives = max(1, len(entities) // 3)
        targets = self.rng.sample(entities, k=min(num_with_objectives, len(entities)))

        for entity in targets:
            num_objectives = self.rng.randint(1, min(3, complexity))
            for _ in range(num_objectives):
                entity.objectives.append(Objective(
                    owner_id=entity.id,
                    description=f"[PLACEHOLDER: {domain.value} objective for {entity.name}]",
                    priority=round(self.rng.uniform(0.3, 1.0), 2),
                    measurable_criteria=["criterion_1", "criterion_2"],
                ))
        return entities

    def _randomize_attributes(self, base: dict) -> dict:
        """Randomize numerical attributes within reasonable ranges."""
        result = {}
        for key, val in base.items():
            if isinstance(val, (int, float)):
                result[key] = round(val * self.rng.uniform(0.5, 1.5), 2)
            else:
                result[key] = val
        return result


# --- Domain-specific entity templates ---

DOMAIN_ENTITY_TEMPLATES = {
    Domain.BUSINESS: [
        {"prefix": "corp", "type": EntityType.ORGANIZATION, "attributes": {"revenue": 100.0, "employees": 500}, "capabilities": ["produce", "market", "hire", "invest"]},
        {"prefix": "competitor", "type": EntityType.ORGANIZATION, "attributes": {"market_share": 0.2}, "capabilities": ["compete", "innovate", "acquire"]},
        {"prefix": "market", "type": EntityType.SYSTEM, "attributes": {"size": 1000.0, "growth_rate": 0.05}, "capabilities": ["expand", "contract", "shift"]},
        {"prefix": "exec", "type": EntityType.PERSON, "attributes": {"influence": 0.8}, "capabilities": ["decide", "negotiate", "lead"]},
        {"prefix": "product", "type": EntityType.ASSET, "attributes": {"quality": 0.7, "margin": 0.3}, "capabilities": ["differentiate", "scale"]},
    ],
    Domain.LOGISTICS: [
        {"prefix": "warehouse", "type": EntityType.LOCATION, "attributes": {"capacity": 10000, "utilization": 0.7}, "capabilities": ["store", "distribute"]},
        {"prefix": "fleet", "type": EntityType.RESOURCE, "attributes": {"vehicles": 50, "availability": 0.9}, "capabilities": ["transport", "deliver"]},
        {"prefix": "supplier", "type": EntityType.ORGANIZATION, "attributes": {"reliability": 0.85, "lead_time": 14}, "capabilities": ["supply", "negotiate"]},
        {"prefix": "route", "type": EntityType.SYSTEM, "attributes": {"distance_km": 500, "risk": 0.1}, "capabilities": ["connect", "bottleneck"]},
        {"prefix": "demand_center", "type": EntityType.LOCATION, "attributes": {"demand_units": 1000}, "capabilities": ["consume", "fluctuate"]},
    ],
    Domain.MARKETS: [
        {"prefix": "asset", "type": EntityType.ASSET, "attributes": {"price": 100.0, "volatility": 0.2}, "capabilities": ["appreciate", "depreciate"]},
        {"prefix": "fund", "type": EntityType.ORGANIZATION, "attributes": {"aum": 500.0, "leverage": 1.5}, "capabilities": ["invest", "hedge", "liquidate"]},
        {"prefix": "exchange", "type": EntityType.SYSTEM, "attributes": {"volume": 1e6, "liquidity": 0.9}, "capabilities": ["facilitate", "halt"]},
        {"prefix": "regulator", "type": EntityType.ORGANIZATION, "attributes": {"authority": 0.95}, "capabilities": ["regulate", "intervene", "fine"]},
        {"prefix": "indicator", "type": EntityType.SYSTEM, "attributes": {"value": 50.0, "trend": 0.02}, "capabilities": ["signal", "mislead"]},
    ],
    Domain.TECHNOLOGY: [
        {"prefix": "platform", "type": EntityType.SYSTEM, "attributes": {"users": 1e6, "uptime": 0.999}, "capabilities": ["scale", "integrate", "deprecate"]},
        {"prefix": "team", "type": EntityType.ORGANIZATION, "attributes": {"headcount": 20, "velocity": 0.7}, "capabilities": ["develop", "ship", "debug"]},
        {"prefix": "infra", "type": EntityType.RESOURCE, "attributes": {"capacity": 1000, "cost_per_unit": 0.01}, "capabilities": ["serve", "fail", "scale"]},
        {"prefix": "dependency", "type": EntityType.SYSTEM, "attributes": {"version": "3.2", "health": 0.8}, "capabilities": ["break", "update", "deprecate"]},
        {"prefix": "customer", "type": EntityType.ORGANIZATION, "attributes": {"satisfaction": 0.75, "churn_risk": 0.1}, "capabilities": ["pay", "leave", "escalate"]},
    ],
    Domain.ORGANIZATIONAL: [
        {"prefix": "division", "type": EntityType.ORGANIZATION, "attributes": {"budget": 10.0, "headcount": 100}, "capabilities": ["execute", "request", "resist"]},
        {"prefix": "leader", "type": EntityType.PERSON, "attributes": {"authority": 0.8, "trust": 0.6}, "capabilities": ["direct", "motivate", "block"]},
        {"prefix": "process", "type": EntityType.SYSTEM, "attributes": {"efficiency": 0.6, "age_years": 5}, "capabilities": ["standardize", "bottleneck", "adapt"]},
        {"prefix": "culture", "type": EntityType.SYSTEM, "attributes": {"innovation_score": 0.5, "risk_tolerance": 0.4}, "capabilities": ["enable", "resist", "evolve"]},
        {"prefix": "talent", "type": EntityType.RESOURCE, "attributes": {"skill_level": 0.7, "retention_risk": 0.3}, "capabilities": ["perform", "leave", "grow"]},
    ],
}
