"""
Agent Simulator — creates synthetic cognitive agents and generates interaction trajectories.

Agent types: analyst, planner, researcher, critic, executor, forecaster, negotiator.
Each agent has a cognitive profile, knowledge state, and behavioral constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.factory.state_generator import WorldState


class AgentRole(str, Enum):
    ANALYST = "analyst"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    EXECUTOR = "executor"
    FORECASTER = "forecaster"
    NEGOTIATOR = "negotiator"


class ActionType(str, Enum):
    ANALYZE = "analyze"
    PLAN = "plan"
    RESEARCH = "research"
    CRITIQUE = "critique"
    EXECUTE = "execute"
    FORECAST = "forecast"
    NEGOTIATE = "negotiate"
    COMMUNICATE = "communicate"
    OBSERVE = "observe"
    DECIDE = "decide"
    TOOL_USE = "tool_use"


class CognitiveProfile(BaseModel):
    risk_tolerance: float = Field(ge=0.0, le=1.0, default=0.5)
    time_preference: float = Field(ge=0.0, le=1.0, default=0.5)
    information_seeking: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence_calibration: float = Field(ge=0.0, le=1.0, default=0.7)
    creativity: float = Field(ge=0.0, le=1.0, default=0.5)
    thoroughness: float = Field(ge=0.0, le=1.0, default=0.7)


class AgentConfig(BaseModel):
    """Complete agent specification."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: AgentRole
    name: str = ""
    cognitive_profile: CognitiveProfile = Field(default_factory=CognitiveProfile)
    domain_expertise: dict[str, float] = Field(default_factory=dict)
    available_actions: list[ActionType] = Field(default_factory=list)
    system_prompt: str = ""
    max_actions_per_turn: int = 5


class Action(BaseModel):
    """Single agent action in a trajectory."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: ActionType
    description: str
    target: Optional[str] = None
    parameters: dict[str, str | float | bool] = Field(default_factory=dict)
    reasoning: str = ""  # chain-of-thought


class Observation(BaseModel):
    """What the agent observes from the environment."""
    visible_state: dict[str, str | float | list] = Field(default_factory=dict)
    new_information: list[str] = Field(default_factory=list)
    feedback: Optional[str] = None
    reward_signal: Optional[float] = None


class TrajectoryStep(BaseModel):
    """Single turn in a trajectory."""
    turn: int
    observation: Observation
    reasoning: str  # agent's internal chain-of-thought
    action: Action
    outcome: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class AgentTrajectory(BaseModel):
    """Complete agent interaction trajectory."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_config: AgentConfig
    initial_state_id: str
    objective: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_outcome: str = ""
    success_score: float = Field(ge=0.0, le=1.0, default=0.0)
    total_tokens: int = 0
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


# --- Role-specific system prompts ---

ROLE_SYSTEM_PROMPTS = {
    AgentRole.ANALYST: (
        "You are a strategic analyst. Your task is to decompose complex situations "
        "into constituent factors, identify patterns, assess risks, and produce "
        "structured analytical outputs. Be thorough, evidence-based, and explicit "
        "about uncertainty."
    ),
    AgentRole.PLANNER: (
        "You are a strategic planner. Your task is to generate feasible action "
        "sequences toward given objectives. Consider resource constraints, timing, "
        "dependencies, and failure modes. Produce plans with clear milestones and "
        "contingencies."
    ),
    AgentRole.RESEARCHER: (
        "You are a research agent. Your task is to gather information, reduce "
        "uncertainty, and validate assumptions. Be systematic, prioritize high-value "
        "information, and report findings with confidence levels."
    ),
    AgentRole.CRITIC: (
        "You are a strategic critic. Your task is to identify flaws, risks, blind "
        "spots, and failure modes in plans and analyses. Be constructive but "
        "uncompromising. Challenge assumptions. Propose alternatives."
    ),
    AgentRole.EXECUTOR: (
        "You are an execution agent. Your task is to implement plans step-by-step, "
        "handle contingencies, adapt to changing conditions, and report status. "
        "Focus on completion, resource efficiency, and error recovery."
    ),
    AgentRole.FORECASTER: (
        "You are a forecasting agent. Your task is to predict outcomes, estimate "
        "probabilities, and quantify uncertainty. Be calibrated — when you say 70%, "
        "events should occur 70% of the time. Decompose predictions into factors."
    ),
    AgentRole.NEGOTIATOR: (
        "You are a negotiation agent. Your task is to manage multi-party interactions "
        "toward favorable outcomes. Understand interests, identify ZOPA, create value, "
        "and manage relationships while achieving objectives."
    ),
}

ROLE_ACTIONS = {
    AgentRole.ANALYST: [ActionType.ANALYZE, ActionType.OBSERVE, ActionType.RESEARCH, ActionType.COMMUNICATE],
    AgentRole.PLANNER: [ActionType.PLAN, ActionType.ANALYZE, ActionType.DECIDE, ActionType.COMMUNICATE],
    AgentRole.RESEARCHER: [ActionType.RESEARCH, ActionType.OBSERVE, ActionType.TOOL_USE, ActionType.COMMUNICATE],
    AgentRole.CRITIC: [ActionType.CRITIQUE, ActionType.ANALYZE, ActionType.OBSERVE, ActionType.COMMUNICATE],
    AgentRole.EXECUTOR: [ActionType.EXECUTE, ActionType.DECIDE, ActionType.TOOL_USE, ActionType.COMMUNICATE],
    AgentRole.FORECASTER: [ActionType.FORECAST, ActionType.ANALYZE, ActionType.RESEARCH, ActionType.COMMUNICATE],
    AgentRole.NEGOTIATOR: [ActionType.NEGOTIATE, ActionType.COMMUNICATE, ActionType.ANALYZE, ActionType.DECIDE],
}


# --- Simulator ---


@dataclass
class AgentSimulatorConfig:
    roles: list[AgentRole] = field(default_factory=lambda: list(AgentRole))
    max_trajectory_steps: int = 20
    min_trajectory_steps: int = 3
    cognitive_profile_variance: float = 0.3
    seed: Optional[int] = None


class AgentSimulator:
    """
    Creates synthetic cognitive agents and generates interaction trajectories.

    In template mode: generates structural trajectories with placeholders.
    In LLM mode: uses a language model to fill in reasoning and actions.
    """

    def __init__(self, config: AgentSimulatorConfig):
        self.config = config
        import random
        self.rng = random.Random(config.seed)

    def create_agent(self, role: Optional[AgentRole] = None, domain: str = "general") -> AgentConfig:
        """Create a configured agent with randomized cognitive profile."""
        if role is None:
            role = self.rng.choice(self.config.roles)

        profile = CognitiveProfile(
            risk_tolerance=self._random_trait(),
            time_preference=self._random_trait(),
            information_seeking=self._random_trait(),
            confidence_calibration=self._random_trait(),
            creativity=self._random_trait(),
            thoroughness=self._random_trait(),
        )

        return AgentConfig(
            role=role,
            name=f"{role.value}_{self.rng.randint(1000, 9999)}",
            cognitive_profile=profile,
            domain_expertise={domain: round(self.rng.uniform(0.4, 0.95), 2)},
            available_actions=ROLE_ACTIONS[role],
            system_prompt=ROLE_SYSTEM_PROMPTS[role],
        )

    def generate_trajectory(
        self,
        agent: AgentConfig,
        world_state: WorldState,
        objective: str,
    ) -> AgentTrajectory:
        """
        Generate a complete agent trajectory.

        In template mode, produces structural placeholders.
        Full content requires LLM inference (see generate_trajectory_with_llm).
        """
        num_steps = self.rng.randint(
            self.config.min_trajectory_steps, self.config.max_trajectory_steps
        )

        steps = []
        for turn in range(num_steps):
            observation = self._generate_observation(world_state, turn)
            action_type = self.rng.choice(agent.available_actions)

            step = TrajectoryStep(
                turn=turn,
                observation=observation,
                reasoning=f"[REASONING PLACEHOLDER: {agent.role.value} considering step {turn+1}]",
                action=Action(
                    type=action_type,
                    description=f"[ACTION: {action_type.value} at step {turn+1}]",
                    reasoning=f"[Because objective requires {action_type.value}]",
                ),
                outcome=f"[OUTCOME of {action_type.value} at step {turn+1}]",
            )
            steps.append(step)

        success = self.rng.uniform(0.2, 0.9)

        return AgentTrajectory(
            agent_config=agent,
            initial_state_id=world_state.id,
            objective=objective,
            steps=steps,
            final_outcome=f"[FINAL OUTCOME: success_score={success:.2f}]",
            success_score=round(success, 2),
            metadata={
                "generator": "template",
                "domain": world_state.domain.value,
                "num_steps": num_steps,
            },
        )

    def generate_multi_agent_trajectory(
        self,
        agents: list[AgentConfig],
        world_state: WorldState,
        shared_objective: str,
    ) -> list[AgentTrajectory]:
        """Generate coordinated multi-agent trajectories."""
        trajectories = []
        for agent in agents:
            role_objective = f"[{agent.role.value} contribution to: {shared_objective}]"
            traj = self.generate_trajectory(agent, world_state, role_objective)
            trajectories.append(traj)
        return trajectories

    def _generate_observation(self, state: WorldState, turn: int) -> Observation:
        """Generate what the agent observes at this turn."""
        visible_entities = self.rng.sample(
            [e.name for e in state.entities],
            k=min(5, len(state.entities)),
        )
        return Observation(
            visible_state={"entities": visible_entities, "turn": turn},
            new_information=[f"[New info revealed at turn {turn}]"] if turn > 0 else [],
            feedback=f"[Environment feedback for turn {turn}]" if turn > 0 else None,
        )

    def _random_trait(self) -> float:
        """Generate a cognitive trait value with configured variance."""
        base = 0.5
        offset = self.rng.gauss(0, self.config.cognitive_profile_variance)
        return round(max(0.0, min(1.0, base + offset)), 2)
