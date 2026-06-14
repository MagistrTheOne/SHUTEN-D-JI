"""
Expert modules for SHUTEN-DŌJI MoE.

Each expert is a compact SwiGLU FFN.
Shared experts are always active (provide baseline capability).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """Single routed expert — SwiGLU architecture."""

    def __init__(self, hidden_size: int, expert_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedExpert(nn.Module):
    """
    Shared expert — always active for every token.
    Provides baseline capabilities regardless of routing decisions.
    Uses slightly larger hidden dim for stable gradient flow.
    """

    def __init__(self, hidden_size: int, expert_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertParallelWrapper(nn.Module):
    """
    Wrapper for efficient batched expert computation.
    Groups tokens by assigned expert and processes in parallel.
    Used during training for better GPU utilization.
    """

    def __init__(self, experts: nn.ModuleList, num_experts_per_token: int):
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.top_k = num_experts_per_token

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched expert computation — routes each token to its assigned experts.

        Args:
            hidden_states: [num_tokens, hidden_size]
            expert_indices: [num_tokens, top_k]
            expert_weights: [num_tokens, top_k]

        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens, hidden_dim = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        # Group tokens by expert for efficient batched computation
        flat_indices = expert_indices.view(-1)
        flat_weights = expert_weights.view(-1, 1)

        # Repeat hidden states for each top-k assignment
        repeated_hidden = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, hidden_dim)

        for expert_id in range(self.num_experts):
            mask = flat_indices == expert_id
            if not mask.any():
                continue

            expert_input = repeated_hidden[mask]
            expert_output = self.experts[expert_id](expert_input)

            # Scatter weighted results back
            token_indices = torch.arange(num_tokens, device=hidden_states.device)
            token_indices = token_indices.unsqueeze(1).expand(-1, self.top_k).reshape(-1)
            active_token_indices = token_indices[mask]
            active_weights = flat_weights[mask]

            output.index_add_(0, active_token_indices, expert_output * active_weights)

        return output
