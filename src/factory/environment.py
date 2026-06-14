"""
Strategic Environment Simulator.

Gym-like interface for strategic decision environments.
Agents interact with the environment, receive observations and rewards.
Environments maintain hidden ground-truth state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.factory.state_generator import WorldState, Domain
from src.factory.agent_simulator import Action, Observation


class EnvironmentStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    TERMINATED = "terminated"


class StepResult(BaseModel):
    """Result of a single environment step."""
    observation: Observation
    reward: float
    done: bool
    status: EnvironmentStatus = EnvironmentStatus.RUNNING
    info: dict[str, str | float | bool] = Field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    domain: Domain = Domain.BUSINESS
    max_steps: int = 50
    stochasticity: float = 0.2  # probability of random events
    difficulty: int = 3         # 1-5
    delayed_feedback: bool = True
    hidden_information: bool = True
    multi_agent: bool = False
    seed: Optional[int] = None


class StrategicEnvironment(ABC):
    """
    Abstract base for strategic environments.
    Concrete implementations simulate specific domains.
    """

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.current_step = 0
        self.done = False
        self.status = EnvironmentStatus.RUNNING
        self._ground_truth: Optional[WorldState] = None
        self._visible_state: dict = {}
        self._history: list[dict] = []
        import random
        self.rng = random.Random(config.seed)

    @abstractmethod
    def reset(self, world_state: WorldState) -> Observation:
        """Reset environment with initial world state. Returns first observation."""
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Execute action, return observation + reward + done."""
        ...

    @abstractmethod
    def get_verifiable_metrics(self) -> dict[str, float]:
        """Get ground-truth metrics for reward computation."""
        ...

    def get_history(self) -> list[dict]:
        """Get full interaction history."""
        return self._history.copy()

    def is_done(self) -> bool:
        return self.done


class BusinessEnvironment(StrategicEnvironment):
    """
    Business strategy environment.

    Simulates: market dynamics, competition, resource allocation,
    product decisions, hiring, investment.
    """

    def __init__(self, config: EnvironmentConfig):
        config.domain = Domain.BUSINESS
        super().__init__(config)
        self._revenue = 0.0
        self._market_share = 0.0
        self._cash = 0.0
        self._morale = 0.5
        self._risk_exposure = 0.0

    def reset(self, world_state: WorldState) -> Observation:
        self.current_step = 0
        self.done = False
        self.status = EnvironmentStatus.RUNNING
        self._ground_truth = world_state
        self._history = []

        # Initialize metrics from state
        self._revenue = 100.0
        self._market_share = 0.2
        self._cash = 50.0
        self._morale = 0.6
        self._risk_exposure = 0.2

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._make_observation(),
                reward=0.0,
                done=True,
                status=self.status,
            )

        self.current_step += 1

        # Apply action effects (simplified)
        reward = self._apply_action(action)

        # Random events
        if self.rng.random() < self.config.stochasticity:
            self._apply_random_event()

        # Check termination
        if self.current_step >= self.config.max_steps:
            self.done = True
            self.status = EnvironmentStatus.TIMEOUT
        elif self._cash <= 0:
            self.done = True
            self.status = EnvironmentStatus.FAILURE
        elif self._market_share >= 0.5:
            self.done = True
            self.status = EnvironmentStatus.SUCCESS

        observation = self._make_observation()
        self._history.append({
            "step": self.current_step,
            "action": action.model_dump(),
            "reward": reward,
            "metrics": self.get_verifiable_metrics(),
        })

        return StepResult(
            observation=observation,
            reward=reward,
            done=self.done,
            status=self.status,
            info={"step": self.current_step},
        )

    def get_verifiable_metrics(self) -> dict[str, float]:
        return {
            "revenue": self._revenue,
            "market_share": self._market_share,
            "cash": self._cash,
            "morale": self._morale,
            "risk_exposure": self._risk_exposure,
            "steps_taken": float(self.current_step),
        }

    def _apply_action(self, action: Action) -> float:
        """Apply action effects and return immediate reward."""
        reward = 0.0
        noise = self.rng.gauss(0, 0.05)

        if action.type.value == "execute":
            self._revenue *= 1.02 + noise
            self._cash -= 5
            reward = 0.1
        elif action.type.value == "plan":
            self._morale += 0.02
            reward = 0.05
        elif action.type.value == "analyze":
            self._risk_exposure -= 0.03
            reward = 0.03
        elif action.type.value == "decide":
            self._market_share += 0.01 + noise
            self._cash -= 3
            reward = 0.08
        elif action.type.value == "research":
            self._risk_exposure -= 0.02
            reward = 0.02
        else:
            reward = 0.01

        # Clamp values
        self._market_share = max(0.0, min(1.0, self._market_share))
        self._morale = max(0.0, min(1.0, self._morale))
        self._risk_exposure = max(0.0, min(1.0, self._risk_exposure))

        return reward

    def _apply_random_event(self) -> None:
        """Inject a stochastic event."""
        event_type = self.rng.choice(["boost", "disruption", "opportunity"])
        if event_type == "boost":
            self._revenue *= 1.05
        elif event_type == "disruption":
            self._cash -= 10
            self._risk_exposure += 0.1
        elif event_type == "opportunity":
            self._market_share += 0.02

    def _make_observation(self) -> Observation:
        """Generate observation (partial information if configured)."""
        if self.config.hidden_information:
            visible = {
                "revenue_estimate": round(self._revenue * self.rng.uniform(0.9, 1.1), 1),
                "market_position": "growing" if self._market_share > 0.2 else "stable",
                "cash_status": "healthy" if self._cash > 20 else "tight",
                "step": self.current_step,
            }
        else:
            visible = {
                "revenue": round(self._revenue, 1),
                "market_share": round(self._market_share, 3),
                "cash": round(self._cash, 1),
                "morale": round(self._morale, 2),
            }

        return Observation(
            visible_state=visible,
            new_information=[f"Turn {self.current_step} update"],
        )


class LogisticsEnvironment(StrategicEnvironment):
    """Logistics and supply chain environment."""

    def __init__(self, config: EnvironmentConfig):
        config.domain = Domain.LOGISTICS
        super().__init__(config)
        self._throughput = 0.0
        self._cost = 0.0
        self._on_time_delivery = 0.0
        self._inventory_health = 0.0

    def reset(self, world_state: WorldState) -> Observation:
        self.current_step = 0
        self.done = False
        self.status = EnvironmentStatus.RUNNING
        self._ground_truth = world_state
        self._history = []
        self._throughput = 100.0
        self._cost = 50.0
        self._on_time_delivery = 0.85
        self._inventory_health = 0.7
        return Observation(visible_state={"throughput": self._throughput, "step": 0})

    def step(self, action: Action) -> StepResult:
        self.current_step += 1
        reward = self.rng.uniform(-0.1, 0.2)
        self._throughput += self.rng.gauss(0, 5)
        self._cost += self.rng.gauss(1, 2)

        if self.current_step >= self.config.max_steps:
            self.done = True
            self.status = EnvironmentStatus.TIMEOUT

        return StepResult(
            observation=Observation(
                visible_state={"throughput": round(self._throughput, 1), "step": self.current_step}
            ),
            reward=reward,
            done=self.done,
            status=self.status,
        )

    def get_verifiable_metrics(self) -> dict[str, float]:
        return {
            "throughput": self._throughput,
            "total_cost": self._cost,
            "on_time_delivery": self._on_time_delivery,
            "inventory_health": self._inventory_health,
        }


class MarketEnvironment(StrategicEnvironment):
    """Financial market environment."""

    def __init__(self, config: EnvironmentConfig):
        config.domain = Domain.MARKETS
        super().__init__(config)
        self._portfolio_value = 0.0
        self._pnl = 0.0
        self._max_drawdown = 0.0
        self._sharpe = 0.0

    def reset(self, world_state: WorldState) -> Observation:
        self.current_step = 0
        self.done = False
        self.status = EnvironmentStatus.RUNNING
        self._ground_truth = world_state
        self._history = []
        self._portfolio_value = 1000.0
        self._pnl = 0.0
        self._max_drawdown = 0.0
        return Observation(visible_state={"portfolio_value": self._portfolio_value, "step": 0})

    def step(self, action: Action) -> StepResult:
        self.current_step += 1
        change = self.rng.gauss(0.001, 0.02)
        self._portfolio_value *= (1 + change)
        self._pnl = self._portfolio_value - 1000.0

        drawdown = (1000.0 - self._portfolio_value) / 1000.0
        self._max_drawdown = max(self._max_drawdown, max(0, drawdown))

        if self.current_step >= self.config.max_steps:
            self.done = True
            self.status = EnvironmentStatus.TIMEOUT

        return StepResult(
            observation=Observation(
                visible_state={"portfolio_value": round(self._portfolio_value, 2), "step": self.current_step}
            ),
            reward=change,
            done=self.done,
            status=self.status,
        )

    def get_verifiable_metrics(self) -> dict[str, float]:
        return {
            "portfolio_value": self._portfolio_value,
            "pnl": self._pnl,
            "max_drawdown": self._max_drawdown,
            "return_pct": (self._portfolio_value - 1000.0) / 1000.0,
        }


# --- Environment Factory ---

ENVIRONMENT_REGISTRY: dict[Domain, type[StrategicEnvironment]] = {
    Domain.BUSINESS: BusinessEnvironment,
    Domain.LOGISTICS: LogisticsEnvironment,
    Domain.MARKETS: MarketEnvironment,
}


def create_environment(config: EnvironmentConfig) -> StrategicEnvironment:
    """Factory function to create domain-specific environments."""
    env_class = ENVIRONMENT_REGISTRY.get(config.domain)
    if env_class is None:
        raise ValueError(
            f"No environment for domain '{config.domain}'. "
            f"Available: {list(ENVIRONMENT_REGISTRY.keys())}"
        )
    return env_class(config)
