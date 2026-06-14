"""
Data Pipeline — orchestrates the full trajectory generation pipeline.

Flow:
  StateGenerator → ScenarioGenerator → AgentSimulator → Environment → Evaluator → Store

This module ties all factory components together into a single generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.factory.state_generator import StateGenerator, StateGeneratorConfig, WorldState, Domain
from src.factory.scenario_generator import ScenarioGenerator, ScenarioGeneratorConfig
from src.factory.agent_simulator import AgentSimulator, AgentSimulatorConfig, AgentRole
from src.factory.environment import (
    StrategicEnvironment,
    EnvironmentConfig,
    create_environment,
)
from src.factory.outcome_evaluator import OutcomeEvaluator, EvaluatorConfig
from src.factory.trajectory_store import TrajectoryStore, StoreConfig


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Generation targets
    num_trajectories: int = 1000
    domains: list[Domain] = field(default_factory=lambda: [Domain.BUSINESS, Domain.LOGISTICS, Domain.MARKETS])
    complexity_range: tuple[int, int] = (1, 5)

    # Agent config
    agent_roles: list[AgentRole] = field(default_factory=lambda: list(AgentRole))
    multi_agent: bool = False
    agents_per_scenario: int = 3

    # Environment config
    max_env_steps: int = 30
    stochasticity: float = 0.2

    # Storage
    output_dir: Path = Path("data/trajectories")

    # Parallelism
    num_workers: int = 4
    batch_size: int = 10

    seed: Optional[int] = 42


class DataPipeline:
    """
    Orchestrates end-to-end trajectory generation.

    Usage:
        pipeline = DataPipeline(PipelineConfig(num_trajectories=10000))
        stats = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        self.state_gen = StateGenerator(StateGeneratorConfig(
            domains=config.domains,
            seed=config.seed,
        ))
        self.scenario_gen = ScenarioGenerator(ScenarioGeneratorConfig(
            seed=config.seed,
        ))
        self.agent_sim = AgentSimulator(AgentSimulatorConfig(
            roles=config.agent_roles,
            max_trajectory_steps=config.max_env_steps,
            seed=config.seed,
        ))
        self.evaluator = OutcomeEvaluator(EvaluatorConfig())
        self.store = TrajectoryStore(StoreConfig(base_path=config.output_dir))

    def run(self) -> dict[str, int]:
        """
        Execute the full generation pipeline.
        Returns statistics about generated data.
        """
        stats = {"generated": 0, "accepted_sft": 0, "accepted_rl": 0, "rejected": 0}

        for i in range(self.config.num_trajectories):
            try:
                result = self._generate_one()
                stats["generated"] += 1

                if result == "sft":
                    stats["accepted_sft"] += 1
                elif result == "rl":
                    stats["accepted_rl"] += 1
                else:
                    stats["rejected"] += 1

            except Exception as e:
                stats["rejected"] += 1
                if i < 5:  # only log first few errors
                    print(f"[Pipeline] Error at iteration {i}: {e}")

            if (i + 1) % 100 == 0:
                print(f"[Pipeline] Progress: {i+1}/{self.config.num_trajectories} | Stats: {stats}")

        return stats

    def _generate_one(self) -> str:
        """Generate a single trajectory through the full pipeline."""
        import random
        rng = random.Random()

        # 1. Generate world state
        domain = rng.choice(self.config.domains)
        state = self.state_gen.generate(domain=domain)

        # 2. Generate scenario bundle
        scenario_bundle = self.scenario_gen.generate(state)

        # 3. Create agent
        role = rng.choice(self.config.agent_roles)
        agent = self.agent_sim.create_agent(role=role, domain=domain.value)

        # 4. Set up environment
        env_config = EnvironmentConfig(
            domain=domain,
            max_steps=self.config.max_env_steps,
            stochasticity=self.config.stochasticity,
            difficulty=state.complexity_level,
        )
        env = create_environment(env_config)

        # 5. Run agent in environment
        objective = f"Achieve optimal outcome in {domain.value} scenario (complexity {state.complexity_level})"
        trajectory = self.agent_sim.generate_trajectory(agent, state, objective)

        # 6. Evaluate (reset env for fresh metrics)
        env.reset(state)
        for step in trajectory.steps:
            if env.is_done():
                break
            env.step(step.action)

        evaluation = self.evaluator.evaluate_from_environment(trajectory, env)

        # 7. Store
        stored = self.store.store(trajectory, evaluation, domain, state.complexity_level)

        return stored.training_split or "rejected"

    def export_for_training(self, output_dir: Optional[Path] = None) -> dict[str, Path]:
        """Export stored trajectories in LLaMA Factory format."""
        out = output_dir or self.config.output_dir / "llamafactory_export"
        out.mkdir(parents=True, exist_ok=True)

        exports = {}
        for domain in self.config.domains:
            path = self.store.export_for_llamafactory(
                output_path=out / f"{domain.value}_sft.json",
                domain=domain,
                split="sft",
                format="sharegpt",
            )
            exports[f"{domain.value}_sft"] = path

        return exports
