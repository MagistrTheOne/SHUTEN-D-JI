"""
Cognitive Router — typed token routing for MoE.

Key difference from standard routers:
  Token type (state/analysis/simulation/planning/etc.) biases the routing
  toward expert groups specialized for that cognitive mode.
  Groups are soft — a planning token CAN route to an analysis expert
  if the router learns that's useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenType(IntEnum):
    """Cognitive token types — signal reasoning mode."""
    GENERAL = 0
    STATE = 1
    ANALYSIS = 2
    SIMULATION = 3
    PLANNING = 4
    PREDICTION = 5
    MEMORY = 6
    TOOL = 7
    ROLE = 8
    CRITIQUE = 9


# Soft mapping: which expert indices are "preferred" for each token type.
# Router learns to deviate when useful.
TOKEN_TYPE_EXPERT_AFFINITY = {
    TokenType.GENERAL: list(range(115, 128)),
    TokenType.STATE: list(range(0, 20)),
    TokenType.ANALYSIS: list(range(20, 45)),
    TokenType.SIMULATION: list(range(45, 70)),
    TokenType.PLANNING: list(range(70, 95)),
    TokenType.PREDICTION: list(range(45, 70)) + list(range(95, 115)),
    TokenType.MEMORY: list(range(0, 20)),
    TokenType.TOOL: list(range(115, 128)),
    TokenType.ROLE: list(range(115, 128)),
    TokenType.CRITIQUE: list(range(95, 115)),
}


@dataclass
class RouterOutput:
    """Router decision output."""
    expert_indices: torch.Tensor  # [num_tokens, top_k]
    expert_weights: torch.Tensor  # [num_tokens, top_k]
    aux_loss: torch.Tensor  # scalar balancing loss


class CognitiveRouter(nn.Module):
    """
    Routes tokens to experts with cognitive type bias.

    Architecture:
      1. Linear projection: hidden_state → expert logits
      2. Token type bias: add learned bias based on token type
      3. Top-k selection with load balancing loss
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.aux_loss_coef = config.router_aux_loss_coef
        self.z_loss_coef = config.router_z_loss_coef

        # Main routing projection
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Token type bias — learned affinity between token types and experts
        self.type_bias = nn.Parameter(
            torch.zeros(config.num_token_types, config.num_experts)
        )
        self._init_type_bias(config)

    def _init_type_bias(self, config):
        """Initialize type bias to gently steer toward expert groups."""
        with torch.no_grad():
            for token_type, expert_ids in TOKEN_TYPE_EXPERT_AFFINITY.items():
                if token_type.value < config.num_token_types:
                    self.type_bias[token_type.value, expert_ids] = 0.1

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> RouterOutput:
        """
        Args:
            hidden_states: [num_tokens, hidden_size]
            token_type_ids: [num_tokens] or None
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]

        # Apply token type bias if available
        if token_type_ids is not None:
            flat_types = token_type_ids.view(-1)
            type_bias = self.type_bias[flat_types]  # [num_tokens, num_experts]
            router_logits = router_logits + type_bias

        # Z-loss for numerical stability (prevents logits from growing too large)
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() * self.z_loss_coef

        # Top-k selection
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize selected weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.type_as(hidden_states)

        # Load balancing auxiliary loss
        aux_loss = self._load_balance_loss(router_logits, topk_indices) + z_loss

        return RouterOutput(
            expert_indices=topk_indices,
            expert_weights=topk_weights,
            aux_loss=aux_loss,
        )

    def _load_balance_loss(
        self, router_logits: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Encourages uniform expert utilization."""
        num_tokens = router_logits.shape[0]

        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(selected_experts, self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=0).sum(dim=0) / num_tokens

        # Average routing probability per expert
        routing_probs = F.softmax(router_logits, dim=-1).mean(dim=0)

        # Dot product encourages uniform distribution
        return (tokens_per_expert * routing_probs).sum() * self.num_experts * self.aux_loss_coef
