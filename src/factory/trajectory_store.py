"""
Trajectory Store — persistent storage and curation of training data.

Stores complete trajectories with evaluations.
Provides filtering, deduplication, and balanced sampling for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.factory.agent_simulator import AgentTrajectory
from src.factory.outcome_evaluator import TrajectoryEvaluation
from src.factory.state_generator import Domain


class StoredTrajectory(BaseModel):
    """Trajectory with all metadata for storage."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    trajectory: AgentTrajectory
    evaluation: TrajectoryEvaluation
    domain: Domain
    complexity_level: int
    filter_status: str = "pending"  # pending, accepted, rejected
    rejection_reason: Optional[str] = None
    training_split: Optional[str] = None  # sft, rl, critique, None


@dataclass
class StoreConfig:
    base_path: Path = Path("data/trajectories")
    max_per_domain: int = 100000
    dedup_threshold: float = 0.95
    sft_ratio: float = 0.3
    rl_ratio: float = 0.5
    critique_ratio: float = 0.2


class TrajectoryStore:
    """
    Manages trajectory storage, filtering, and retrieval.

    Storage format: JSONL files organized by domain and split.
    """

    def __init__(self, config: StoreConfig = StoreConfig()):
        self.config = config
        self._ensure_dirs()
        self._counts: dict[str, int] = {}

    def _ensure_dirs(self) -> None:
        """Create storage directories."""
        for domain in Domain:
            for split in ["sft", "rl", "critique", "raw"]:
                path = self.config.base_path / domain.value / split
                path.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        trajectory: AgentTrajectory,
        evaluation: TrajectoryEvaluation,
        domain: Domain,
        complexity_level: int,
    ) -> StoredTrajectory:
        """Store a trajectory with its evaluation."""
        stored = StoredTrajectory(
            trajectory=trajectory,
            evaluation=evaluation,
            domain=domain,
            complexity_level=complexity_level,
        )

        # Determine training split based on evaluation
        stored = self._assign_split(stored)

        # Write to appropriate file
        split = stored.training_split or "raw"
        path = self.config.base_path / domain.value / split / f"batch_{self._get_batch_id(domain, split)}.jsonl"

        with open(path, "a", encoding="utf-8") as f:
            f.write(stored.model_dump_json() + "\n")

        self._increment_count(domain, split)
        return stored

    def store_batch(
        self,
        trajectories: list[AgentTrajectory],
        evaluations: list[TrajectoryEvaluation],
        domain: Domain,
        complexity_level: int,
    ) -> list[StoredTrajectory]:
        """Store multiple trajectories."""
        results = []
        for traj, eval_ in zip(trajectories, evaluations):
            stored = self.store(traj, eval_, domain, complexity_level)
            results.append(stored)
        return results

    def load_split(self, domain: Domain, split: str, limit: Optional[int] = None) -> list[StoredTrajectory]:
        """Load trajectories from a specific domain and split."""
        path = self.config.base_path / domain.value / split
        if not path.exists():
            return []

        trajectories = []
        for file in sorted(path.glob("*.jsonl")):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        stored = StoredTrajectory.model_validate_json(line)
                        trajectories.append(stored)
                        if limit and len(trajectories) >= limit:
                            return trajectories
        return trajectories

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get count statistics per domain and split."""
        stats = {}
        for domain in Domain:
            stats[domain.value] = {}
            for split in ["sft", "rl", "critique", "raw"]:
                path = self.config.base_path / domain.value / split
                count = 0
                if path.exists():
                    for file in path.glob("*.jsonl"):
                        with open(file, "r", encoding="utf-8") as f:
                            count += sum(1 for line in f if line.strip())
                stats[domain.value][split] = count
        return stats

    def export_for_llamafactory(
        self,
        output_path: Path,
        domain: Optional[Domain] = None,
        split: str = "sft",
        format: str = "sharegpt",
    ) -> Path:
        """
        Export trajectories in LLaMA Factory compatible format.

        Formats:
          - sharegpt: multi-turn conversation format
          - alpaca: instruction-response format
        """
        domains = [domain] if domain else list(Domain)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for d in domains:
            trajectories = self.load_split(d, split)
            for stored in trajectories:
                if format == "sharegpt":
                    record = self._to_sharegpt(stored)
                else:
                    record = self._to_alpaca(stored)
                records.append(record)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        return output_path

    def _assign_split(self, stored: StoredTrajectory) -> StoredTrajectory:
        """Assign trajectory to training split based on evaluation."""
        score = stored.evaluation.composite_score

        if score >= self.config.sft_ratio + 0.4:  # top quality → SFT
            stored.training_split = "sft"
            stored.filter_status = "accepted"
        elif 0.3 <= score <= 0.9:  # medium variance → RL
            stored.training_split = "rl"
            stored.filter_status = "accepted"
        elif score < 0.5 and stored.evaluation.critique_quality > 0.5:  # informative failures → critique
            stored.training_split = "critique"
            stored.filter_status = "accepted"
        else:
            stored.training_split = "raw"
            stored.filter_status = "pending"

        return stored

    def _to_sharegpt(self, stored: StoredTrajectory) -> dict:
        """Convert trajectory to ShareGPT multi-turn format for LLaMA Factory."""
        conversations = []

        # System message with agent role
        conversations.append({
            "from": "system",
            "value": stored.trajectory.agent_config.system_prompt,
        })

        # Convert each step to a turn
        for step in stored.trajectory.steps:
            # User turn: observation
            conversations.append({
                "from": "human",
                "value": (
                    f"[Observation] {json.dumps(step.observation.visible_state)}\n"
                    f"[Objective] {stored.trajectory.objective}\n"
                    f"[Turn] {step.turn}"
                ),
            })
            # Assistant turn: reasoning + action
            conversations.append({
                "from": "gpt",
                "value": (
                    f"<analysis>{step.reasoning}</analysis>\n"
                    f"<action type=\"{step.action.type.value}\">{step.action.description}</action>\n"
                    f"<outcome>{step.outcome}</outcome>"
                ),
            })

        return {"conversations": conversations}

    def _to_alpaca(self, stored: StoredTrajectory) -> dict:
        """Convert trajectory to Alpaca instruction format."""
        instruction = (
            f"You are a {stored.trajectory.agent_config.role.value}. "
            f"Domain: {stored.domain.value}. "
            f"Objective: {stored.trajectory.objective}"
        )

        # Flatten trajectory into single response
        output_parts = []
        for step in stored.trajectory.steps:
            output_parts.append(
                f"[Step {step.turn}] {step.reasoning}\n"
                f"Action: {step.action.type.value} — {step.action.description}\n"
                f"Result: {step.outcome}"
            )

        return {
            "instruction": instruction,
            "input": json.dumps(stored.trajectory.steps[0].observation.visible_state) if stored.trajectory.steps else "",
            "output": "\n\n".join(output_parts),
        }

    def _get_batch_id(self, domain: Domain, split: str) -> str:
        key = f"{domain.value}_{split}"
        count = self._counts.get(key, 0)
        return f"{count // 10000:04d}"

    def _increment_count(self, domain: Domain, split: str) -> None:
        key = f"{domain.value}_{split}"
        self._counts[key] = self._counts.get(key, 0) + 1
