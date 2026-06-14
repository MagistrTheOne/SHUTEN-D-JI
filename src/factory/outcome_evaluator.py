"""
Outcome Evaluator — produces verifiable scores for trajectories.

Evaluates:
  - Primary outcome (objective achievement)
  - Process quality (reasoning soundness)
  - Efficiency (resource usage)
  - Side effects (unintended consequences)
  - Risk management

Must produce scores that enable RL training signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.factory.agent_simulator import AgentTrajectory, AgentRole
from src.factory.environment import StrategicEnvironment, EnvironmentStatus


class SideEffect(BaseModel):
    description: str
    valence: float = Field(ge=-1.0, le=1.0)
    magnitude: float = Field(ge=0.0, le=1.0)
    was_anticipated: bool = False


class RiskAssessment(BaseModel):
    risk_id: str
    was_mitigated: bool
    damage_if_realized: float = Field(ge=0.0, le=1.0)


class ProcessQualityScores(BaseModel):
    information_utilization: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning_soundness: float = Field(ge=0.0, le=1.0, default=0.5)
    calibration: float = Field(ge=0.0, le=1.0, default=0.5)
    adaptability: float = Field(ge=0.0, le=1.0, default=0.5)


class TrajectoryEvaluation(BaseModel):
    """Complete evaluation of a trajectory."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    trajectory_id: str

    # Primary outcome
    objective_achieved: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    time_to_outcome: float = 0.0

    # Secondary outcomes
    side_effects: list[SideEffect] = Field(default_factory=list)
    risks_assessed: list[RiskAssessment] = Field(default_factory=list)

    # Process quality
    process_quality: ProcessQualityScores = Field(default_factory=ProcessQualityScores)

    # Counterfactual
    best_possible_outcome: float = 1.0
    worst_possible_outcome: float = 0.0
    agent_outcome_percentile: float = 0.5

    # Composite score (primary RL signal)
    composite_score: float = Field(ge=0.0, le=1.0, default=0.5)

    # Training signal categorization
    sft_quality: float = Field(ge=0.0, le=1.0, default=0.5)
    rl_quality: float = Field(ge=0.0, le=1.0, default=0.5)
    critique_quality: float = Field(ge=0.0, le=1.0, default=0.5)


@dataclass
class EvaluatorConfig:
    """Weights for composite score computation."""
    w_outcome: float = 0.4
    w_process: float = 0.25
    w_efficiency: float = 0.2
    w_risk_penalty: float = 0.15
    sft_threshold: float = 0.7
    rl_min_threshold: float = 0.3
    rl_max_threshold: float = 0.9


class OutcomeEvaluator:
    """
    Evaluates trajectory quality for training signal production.

    Three evaluation modes:
      1. Environment-based: uses verifiable metrics from simulation
      2. Rule-based: applies domain-specific rubrics
      3. Model-based: uses a critic model (requires trained critic)
    """

    def __init__(self, config: EvaluatorConfig = EvaluatorConfig()):
        self.config = config

    def evaluate_from_environment(
        self,
        trajectory: AgentTrajectory,
        environment: StrategicEnvironment,
    ) -> TrajectoryEvaluation:
        """
        Evaluate trajectory using ground-truth environment metrics.
        This is the primary verifiable reward source.
        """
        metrics = environment.get_verifiable_metrics()
        status = environment.status

        # Objective achievement from environment status
        if status == EnvironmentStatus.SUCCESS:
            objective_achieved = 0.9 + 0.1 * (1 - trajectory.success_score)
        elif status == EnvironmentStatus.TIMEOUT:
            objective_achieved = trajectory.success_score * 0.6
        else:
            objective_achieved = max(0.0, trajectory.success_score * 0.3)

        # Efficiency: fewer steps = more efficient
        max_steps = len(trajectory.steps)
        efficiency = 1.0 - (max_steps / 50.0) if max_steps < 50 else 0.1

        # Process quality (heuristic: based on action diversity and step count)
        action_types = set(s.action.type for s in trajectory.steps)
        action_diversity = len(action_types) / max(1, len(trajectory.agent_config.available_actions))

        process = ProcessQualityScores(
            information_utilization=min(1.0, action_diversity + 0.2),
            reasoning_soundness=trajectory.success_score,
            calibration=0.5,  # requires separate calibration evaluation
            adaptability=action_diversity,
        )

        # Composite score
        process_avg = (
            process.information_utilization
            + process.reasoning_soundness
            + process.calibration
            + process.adaptability
        ) / 4.0

        composite = (
            self.config.w_outcome * objective_achieved
            + self.config.w_process * process_avg
            + self.config.w_efficiency * efficiency
        )
        composite = max(0.0, min(1.0, composite))

        # Training signal categorization
        sft_quality = 1.0 if composite >= self.config.sft_threshold else composite / self.config.sft_threshold
        rl_quality = 1.0 if self.config.rl_min_threshold <= composite <= self.config.rl_max_threshold else 0.5

        return TrajectoryEvaluation(
            trajectory_id=trajectory.id,
            objective_achieved=round(objective_achieved, 3),
            efficiency=round(efficiency, 3),
            time_to_outcome=float(max_steps),
            process_quality=process,
            composite_score=round(composite, 3),
            sft_quality=round(sft_quality, 3),
            rl_quality=round(rl_quality, 3),
            critique_quality=round(1.0 - composite, 3),  # bad trajectories are good for critique
        )

    def evaluate_batch(
        self,
        trajectories: list[AgentTrajectory],
        environments: list[StrategicEnvironment],
    ) -> list[TrajectoryEvaluation]:
        """Evaluate multiple trajectories."""
        return [
            self.evaluate_from_environment(traj, env)
            for traj, env in zip(trajectories, environments)
        ]

    def filter_for_sft(
        self, evaluations: list[TrajectoryEvaluation], threshold: Optional[float] = None
    ) -> list[str]:
        """Return trajectory IDs suitable for SFT training."""
        t = threshold or self.config.sft_threshold
        return [e.trajectory_id for e in evaluations if e.composite_score >= t]

    def filter_for_rl(self, evaluations: list[TrajectoryEvaluation]) -> list[str]:
        """Return trajectory IDs suitable for RL training (need variance)."""
        return [
            e.trajectory_id
            for e in evaluations
            if self.config.rl_min_threshold <= e.composite_score <= self.config.rl_max_threshold
        ]

    def filter_for_critique(
        self, evaluations: list[TrajectoryEvaluation], threshold: float = 0.5
    ) -> list[str]:
        """Return trajectory IDs with informative failures for critique training."""
        return [
            e.trajectory_id
            for e in evaluations
            if e.composite_score < threshold and e.critique_quality > 0.5
        ]
